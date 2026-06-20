"""Collect aligned online-search traces for adaptive policy distillation.

The evaluator showed that short counterfactual primitive search is a strong
wrapper for both fixed-v5 max500 and 8/12/16 Expander. This collector saves the
same deployment-shaped search decisions as NPZ shards so later training can
distill the action choice, score margin, and search value into the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from adaptive_common import (
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_action_to_index,
    adaptive_index_to_action,
    adaptive_input_channel_count,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    empty_adaptive_fog_memory,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
    reset_adaptive_fog_memory,
    reset_adaptive_scoreboard_history,
    update_adaptive_fog_memory,
)
from adaptive_network import load_or_create_adaptive_network
from adaptive_search_distill import outcome_class_from_winner
from adaptive_strategy_dataset import full_state_strategy_labels
from adaptive_teacher_imitation import policy_action_from_logits
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, POLICY_MODE_NAMES, opponent_action, policy_network_action
from evaluate_adaptive_policy import (
    POLICY_ADAPTER_MODE_TO_ID,
    policy_adapter_blend_logits,
    policy_adapter_delta_logits,
    scalar_reset_fog_memory,
    search_score_observation,
)
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from generals.core import game
from train import random_action, stack_learner_actions
from train_adaptive import crop_observation, split_mixed_env_counts

POLICY_MODE_TO_ID = {name: index for index, name in enumerate(POLICY_MODE_NAMES)}


def empty_scoreboard_history(num_envs: int) -> jnp.ndarray:
    """Return empty previous-scoreboard features for vectorized collection."""
    return jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)


def dynamic_adapter_logits(
    network,
    adapter_network,
    obs_arr: jnp.ndarray,
    mask: jnp.ndarray,
    active: jnp.ndarray,
    effective_size: jnp.ndarray,
    adapter_scale: float,
    adapter_mode: int,
    adapter_min_grid_size: int,
    adapter_max_grid_size: int,
) -> jnp.ndarray:
    """Return deployment logits with a size-gated policy adapter."""
    logits, _ = network.logits_value(obs_arr, mask, active)
    if adapter_network is None or adapter_scale <= 0.0:
        return logits
    size_allowed = (
        ((adapter_min_grid_size <= 0) | (effective_size >= adapter_min_grid_size))
        & ((adapter_max_grid_size <= 0) | (effective_size <= adapter_max_grid_size))
    )
    adapter_logits, _ = adapter_network.logits_value(obs_arr, mask, active)
    adapted = jax.lax.switch(
        adapter_mode,
        (
            lambda _: policy_adapter_delta_logits(logits, adapter_logits, adapter_scale),
            lambda _: policy_adapter_blend_logits(logits, adapter_logits, adapter_scale),
            lambda _: adapter_logits,
        ),
        None,
    )
    return jnp.where(size_allowed, adapted, logits)


def deployment_action_with_memory(
    network,
    adapter_network,
    obs,
    effective_size: int,
    key,
    policy_mode: int,
    pad_size: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    adapter_scale: float,
    adapter_mode: int,
    adapter_min_grid_size: int,
    adapter_max_grid_size: int,
):
    """Sample one deployment action while carrying learner context."""
    current_memory = update_adaptive_fog_memory(fog_memory, obs) if fog_memory_enabled else fog_memory
    current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
    history_context = (
        adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
        if scoreboard_history_enabled
        else None
    )
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
        fog_memory=current_memory if fog_memory_enabled else None,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits = dynamic_adapter_logits(
        network,
        adapter_network,
        obs_arr,
        mask,
        active,
        jnp.asarray(effective_size, dtype=jnp.int32),
        adapter_scale,
        adapter_mode,
        adapter_min_grid_size,
        adapter_max_grid_size,
    )
    action = policy_action_from_logits(logits, key, policy_mode, pad_size)
    return action, current_scoreboard, current_memory


def online_search_trace_policy_opponent(
    network,
    adapter_network,
    opponent_network,
    state,
    effective_size: int,
    key,
    opponent_first_action: jnp.ndarray,
    policy_player: int,
    policy_mode: int,
    opponent_policy_mode: int,
    opponent_policy_grid_size: int,
    pad_size: int,
    max_steps: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    adapter_scale: float,
    adapter_mode: int,
    adapter_min_grid_size: int,
    adapter_max_grid_size: int,
    top_k: int,
    rollout_steps: int,
    rollouts_per_action: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
):
    """Return candidate actions and scores against a fixed checkpoint opponent."""
    obs = game.get_observation(state, policy_player)
    current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
    history_context = (
        adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
        if scoreboard_history_enabled
        else None
    )
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
        fog_memory=fog_memory if fog_memory_enabled else None,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits = dynamic_adapter_logits(
        network,
        adapter_network,
        obs_arr,
        mask,
        active,
        jnp.asarray(effective_size, dtype=jnp.int32),
        adapter_scale,
        adapter_mode,
        adapter_min_grid_size,
        adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    opponent_player = 1 - policy_player

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, prev_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, policy_player)
            learner_action, next_scoreboard, next_memory = deployment_action_with_memory(
                network,
                adapter_network,
                learner_obs,
                effective_size,
                learner_key,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                prev_scoreboard,
                fog_memory_enabled,
                memory,
                adapter_scale,
                adapter_mode,
                adapter_min_grid_size,
                adapter_max_grid_size,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = policy_network_action(
                opponent_network,
                opponent_key,
                crop_observation(opponent_obs, opponent_policy_grid_size),
                opponent_policy_mode,
            )
            actions = jax.lax.cond(
                policy_player == 0,
                lambda _: jnp.stack([learner_action, opponent_action_value]),
                lambda _: jnp.stack([opponent_action_value, learner_action]),
                None,
            )
            next_state, _ = game.step(rollout_state, actions)
            current_info = game.get_info(rollout_state)
            already_done = current_info.is_done | (rollout_state.time >= max_steps)
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            final_info = game.get_info(final_state)
            final_scoreboard = reset_adaptive_scoreboard_history(next_scoreboard, final_info.is_done)
            final_memory = scalar_reset_fog_memory(next_memory, final_info.is_done)
            return (final_state, final_scoreboard, final_memory, step_key), None

        (final_state, _, _, _), _ = jax.lax.scan(
            body,
            (initial_state, current_scoreboard, fog_memory, rollout_key),
            None,
            length=rollout_steps,
        )
        final_info = game.get_info(final_state)
        truncated = (final_state.time >= max_steps) & ~final_info.is_done
        scored_info = final_info._replace(winner=jnp.where(truncated, -1, final_info.winner))
        final_obs = game.get_observation(final_state, policy_player)
        score = search_score_observation(scored_info, final_obs, policy_player, army_weight, land_weight, terminal_score)
        outcome = outcome_class_from_winner(jnp.where(final_info.is_done, final_info.winner, -1), policy_player)
        return score, outcome

    def score_candidate(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        scores, outcomes = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        best_rollout = jnp.argmax(scores)
        rollout_outcome = outcomes[best_rollout]
        first_outcome = outcome_class_from_winner(first_info.winner, policy_player)
        first_terminal = jnp.where(
            first_info.winner == policy_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        candidate_score = first_terminal + jnp.mean(scores) + prior_weight * jnp.clip(prior_score, -1.0e4, 1.0e4)
        candidate_outcome = jnp.where(first_info.is_done, first_outcome, rollout_outcome)
        return candidate_score, candidate_outcome

    candidate_keys = jrandom.split(key, top_k)
    scores, outcomes = jax.vmap(score_candidate)(candidate_actions, prior_scores, candidate_keys)
    return candidate_actions, candidate_indices, prior_scores, scores, outcomes


def online_search_trace_heuristic_opponent(
    network,
    adapter_network,
    opponent_id: int,
    state,
    effective_size: int,
    key,
    opponent_first_action: jnp.ndarray,
    policy_player: int,
    policy_mode: int,
    pad_size: int,
    max_steps: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    adapter_scale: float,
    adapter_mode: int,
    adapter_min_grid_size: int,
    adapter_max_grid_size: int,
    top_k: int,
    rollout_steps: int,
    rollouts_per_action: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
):
    """Return candidate actions and scores against a built-in heuristic opponent."""
    obs = game.get_observation(state, policy_player)
    current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
    history_context = (
        adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
        if scoreboard_history_enabled
        else None
    )
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
        fog_memory=fog_memory if fog_memory_enabled else None,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits = dynamic_adapter_logits(
        network,
        adapter_network,
        obs_arr,
        mask,
        active,
        jnp.asarray(effective_size, dtype=jnp.int32),
        adapter_scale,
        adapter_mode,
        adapter_min_grid_size,
        adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    opponent_player = 1 - policy_player

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, prev_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, policy_player)
            learner_action, next_scoreboard, next_memory = deployment_action_with_memory(
                network,
                adapter_network,
                learner_obs,
                effective_size,
                learner_key,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                prev_scoreboard,
                fog_memory_enabled,
                memory,
                adapter_scale,
                adapter_mode,
                adapter_min_grid_size,
                adapter_max_grid_size,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = opponent_action(opponent_id, opponent_key, opponent_obs, random_action)
            actions = jax.lax.cond(
                policy_player == 0,
                lambda _: jnp.stack([learner_action, opponent_action_value]),
                lambda _: jnp.stack([opponent_action_value, learner_action]),
                None,
            )
            next_state, _ = game.step(rollout_state, actions)
            current_info = game.get_info(rollout_state)
            already_done = current_info.is_done | (rollout_state.time >= max_steps)
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            final_info = game.get_info(final_state)
            final_scoreboard = reset_adaptive_scoreboard_history(next_scoreboard, final_info.is_done)
            final_memory = scalar_reset_fog_memory(next_memory, final_info.is_done)
            return (final_state, final_scoreboard, final_memory, step_key), None

        (final_state, _, _, _), _ = jax.lax.scan(
            body,
            (initial_state, current_scoreboard, fog_memory, rollout_key),
            None,
            length=rollout_steps,
        )
        final_info = game.get_info(final_state)
        truncated = (final_state.time >= max_steps) & ~final_info.is_done
        scored_info = final_info._replace(winner=jnp.where(truncated, -1, final_info.winner))
        final_obs = game.get_observation(final_state, policy_player)
        score = search_score_observation(scored_info, final_obs, policy_player, army_weight, land_weight, terminal_score)
        outcome = outcome_class_from_winner(jnp.where(final_info.is_done, final_info.winner, -1), policy_player)
        return score, outcome

    def score_candidate(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        scores, outcomes = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        best_rollout = jnp.argmax(scores)
        rollout_outcome = outcomes[best_rollout]
        first_outcome = outcome_class_from_winner(first_info.winner, policy_player)
        first_terminal = jnp.where(
            first_info.winner == policy_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        candidate_score = first_terminal + jnp.mean(scores) + prior_weight * jnp.clip(prior_score, -1.0e4, 1.0e4)
        candidate_outcome = jnp.where(first_info.is_done, first_outcome, rollout_outcome)
        return candidate_score, candidate_outcome

    candidate_keys = jrandom.split(key, top_k)
    scores, outcomes = jax.vmap(score_candidate)(candidate_actions, prior_scores, candidate_keys)
    return candidate_actions, candidate_indices, prior_scores, scores, outcomes


def default_trace_from_logits(logits: jnp.ndarray, top_k: int, pad_size: int):
    """Return a no-search trace with only prior top-k candidates."""
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    scores = jnp.zeros((top_k,), dtype=jnp.float32)
    outcomes = jnp.full((top_k,), -1, dtype=jnp.int32)
    return candidate_actions, candidate_indices, prior_scores, scores, outcomes


@eqx.filter_jit
def collect_online_search_step(
    states,
    effective_sizes,
    pool,
    network,
    adapter_network,
    opponent_network,
    key,
    truncation: int,
    opponent_id: int,
    learner_player: int,
    policy_mode: int,
    opponent_policy_mode: int,
    opponent_policy_grid_size: int,
    pad_size: int,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    adapter_scale: float = 0.0,
    adapter_mode: int = 0,
    adapter_min_grid_size: int = 0,
    adapter_max_grid_size: int = 0,
    search_top_k: int = 4,
    search_rollout_steps: int = 16,
    search_rollouts_per_action: int = 1,
    search_min_turn: int = 0,
    search_require_contact: bool = False,
    search_min_grid_size: int = 0,
    search_max_grid_size: int = 0,
    search_army_weight: float = 1.0,
    search_land_weight: float = 10.0,
    search_prior_weight: float = 0.001,
    search_terminal_score: float = 100.0,
):
    """Collect one deployment step plus aligned online-search trace labels."""
    num_envs = states.armies.shape[0]
    obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
    obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
    learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
    opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
    if scoreboard_history is None:
        scoreboard_history = empty_scoreboard_history(num_envs)
    if fog_memory is None:
        fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)
    current_memory = jax.vmap(update_adaptive_fog_memory)(fog_memory, learner_obs) if fog_memory_enabled else fog_memory

    if scoreboard_history_enabled:
        current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
            learner_obs,
            effective_sizes,
        )
        history_context = adaptive_scoreboard_history_context(scoreboard_history, current_scoreboard)
        if fog_memory_enabled:
            obs_arr, active = jax.vmap(
                lambda obs, size, history, memory: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                    fog_memory=memory,
                )
            )(learner_obs, effective_sizes, history_context, current_memory)
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size, history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                )
            )(learner_obs, effective_sizes, history_context)
    else:
        current_scoreboard = scoreboard_history
        if fog_memory_enabled:
            obs_arr, active = jax.vmap(
                lambda obs, size, memory: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=global_context,
                    fog_memory=memory,
                )
            )(learner_obs, effective_sizes, current_memory)
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
            )(learner_obs, effective_sizes)
    masks = jax.vmap(
        lambda obs, size: compute_adaptive_valid_move_mask(
            obs.armies,
            obs.owned_cells,
            obs.mountains,
            size,
            pad_size,
        )
    )(learner_obs, effective_sizes)
    logits = jax.vmap(
        lambda obs, mask, active_mask, size: dynamic_adapter_logits(
            network,
            adapter_network,
            obs,
            mask,
            active_mask,
            size,
            adapter_scale,
            adapter_mode,
            adapter_min_grid_size,
            adapter_max_grid_size,
        )
    )(obs_arr, masks, active, effective_sizes)

    key, policy_key, opponent_key, search_key = jrandom.split(key, 4)
    policy_keys = jrandom.split(policy_key, num_envs)
    opponent_keys = jrandom.split(opponent_key, num_envs)
    search_keys = jrandom.split(search_key, num_envs)
    base_actions = jax.vmap(lambda row_logits, row_key: policy_action_from_logits(row_logits, row_key, policy_mode, pad_size))(
        logits,
        policy_keys,
    )
    base_action_indices = jax.vmap(lambda action: adaptive_action_to_index(action, pad_size))(base_actions)

    if opponent_network is not None:
        opponent_actions = jax.vmap(
            lambda row_key, obs, size: policy_network_action(
                opponent_network,
                row_key,
                crop_observation(obs, opponent_policy_grid_size),
                opponent_policy_mode,
            )
        )(opponent_keys, opponent_obs, effective_sizes)
    else:
        opponent_actions = jax.vmap(lambda row_key, obs: opponent_action(opponent_id, row_key, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )

    visible_contact = jnp.sum(learner_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
    size_allowed = (
        ((search_min_grid_size <= 0) | (effective_sizes >= search_min_grid_size))
        & ((search_max_grid_size <= 0) | (effective_sizes <= search_max_grid_size))
    )
    turn_allowed = states.time >= search_min_turn
    contact_allowed = visible_contact | (not search_require_contact)
    pre_infos = jax.vmap(game.get_info)(states)
    use_search = (~pre_infos.is_done) & size_allowed & turn_allowed & contact_allowed

    if opponent_network is not None:
        trace = jax.vmap(
            lambda state, size, row_key, opponent_first, row_history, row_memory, row_logits, row_use_search: jax.lax.cond(
                row_use_search,
                lambda _: online_search_trace_policy_opponent(
                    network,
                    adapter_network,
                    opponent_network,
                    state,
                    size,
                    row_key,
                    opponent_first,
                    learner_player,
                    policy_mode,
                    opponent_policy_mode,
                    opponent_policy_grid_size,
                    pad_size,
                    truncation,
                    global_context,
                    scoreboard_history_enabled,
                    row_history,
                    fog_memory_enabled,
                    row_memory,
                    adapter_scale,
                    adapter_mode,
                    adapter_min_grid_size,
                    adapter_max_grid_size,
                    search_top_k,
                    search_rollout_steps,
                    search_rollouts_per_action,
                    search_army_weight,
                    search_land_weight,
                    search_prior_weight,
                    search_terminal_score,
                ),
                lambda _: default_trace_from_logits(row_logits, search_top_k, pad_size),
                None,
            )
        )(states, effective_sizes, search_keys, opponent_actions, scoreboard_history, current_memory, logits, use_search)
    else:
        trace = jax.vmap(
            lambda state, size, row_key, opponent_first, row_history, row_memory, row_logits, row_use_search: jax.lax.cond(
                row_use_search,
                lambda _: online_search_trace_heuristic_opponent(
                    network,
                    adapter_network,
                    opponent_id,
                    state,
                    size,
                    row_key,
                    opponent_first,
                    learner_player,
                    policy_mode,
                    pad_size,
                    truncation,
                    global_context,
                    scoreboard_history_enabled,
                    row_history,
                    fog_memory_enabled,
                    row_memory,
                    adapter_scale,
                    adapter_mode,
                    adapter_min_grid_size,
                    adapter_max_grid_size,
                    search_top_k,
                    search_rollout_steps,
                    search_rollouts_per_action,
                    search_army_weight,
                    search_land_weight,
                    search_prior_weight,
                    search_terminal_score,
                ),
                lambda _: default_trace_from_logits(row_logits, search_top_k, pad_size),
                None,
            )
        )(states, effective_sizes, search_keys, opponent_actions, scoreboard_history, current_memory, logits, use_search)

    candidate_actions, candidate_indices, prior_scores, search_scores, search_outcomes = trace
    best_positions = jnp.argmax(search_scores, axis=-1)
    search_actions = jnp.take_along_axis(candidate_actions, best_positions[:, None, None], axis=1)[:, 0]
    search_action_indices = jnp.take_along_axis(candidate_indices, best_positions[:, None], axis=1)[:, 0]
    executed_actions = jax.vmap(
        lambda use, search_action, base_action: jnp.where(use, search_action, base_action)
    )(use_search, search_actions, base_actions)
    executed_action_indices = jnp.where(use_search, search_action_indices, base_action_indices)
    best_scores = jnp.take_along_axis(search_scores, best_positions[:, None], axis=1)[:, 0]
    candidate_positions = jnp.arange(search_top_k)[None, :]
    finite_scores = jnp.isfinite(search_scores)
    finite_count = jnp.sum(finite_scores.astype(jnp.int32), axis=-1)
    second_scores = jnp.max(
        jnp.where(finite_scores & (candidate_positions != best_positions[:, None]), search_scores, -jnp.inf),
        axis=-1,
    )
    score_gaps = jnp.where((search_top_k > 1) & (finite_count > 1), best_scores - second_scores, 0.0)
    best_outcomes = jnp.take_along_axis(search_outcomes, best_positions[:, None], axis=1)[:, 0]
    action_changed = executed_action_indices != base_action_indices

    labels = jax.vmap(lambda state, obs, size: full_state_strategy_labels(state, obs, learner_player, size, pad_size))(
        states,
        learner_obs,
        effective_sizes,
    )

    actions = stack_learner_actions(executed_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)
    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated
    pool_size = pool.states.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda value: value[reset_indices], pool.states)
    reset_sizes = pool.effective_sizes[reset_indices]
    next_pool_idx = jnp.where(dones, new_states.pool_idx + num_envs, new_states.pool_idx)
    reset_states = reset_states._replace(pool_idx=next_pool_idx)
    current_states = new_states._replace(pool_idx=next_pool_idx)
    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        current_states,
    )
    final_sizes = jnp.where(dones, reset_sizes, effective_sizes)
    final_history = reset_adaptive_scoreboard_history(current_scoreboard, dones)
    final_memory = reset_adaptive_fog_memory(current_memory, dones)
    saved_winner = jnp.where(truncated, -1, infos.winner)
    return (
        final_states,
        final_sizes,
        final_history,
        final_memory,
        (
            obs_arr,
            masks,
            active,
            logits,
            base_action_indices,
            executed_action_indices,
            effective_sizes,
            jnp.full((num_envs,), learner_player, dtype=jnp.int32),
            states.time,
            use_search,
            action_changed,
            candidate_indices,
            prior_scores,
            search_scores,
            search_outcomes,
            best_positions,
            best_scores,
            score_gaps,
            best_outcomes,
            labels,
            dones,
            saved_winner,
            visible_contact,
        ),
        key,
    )


def collect_rollout(
    states,
    effective_sizes,
    pool,
    network,
    adapter_network,
    opponent_network,
    key,
    num_steps: int,
    truncation: int,
    opponent_id: int,
    learner_player: int,
    policy_mode: int,
    opponent_policy_mode: int,
    opponent_policy_grid_size: int,
    pad_size: int,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    adapter_scale: float = 0.0,
    adapter_mode: int = 0,
    adapter_min_grid_size: int = 0,
    adapter_max_grid_size: int = 0,
    search_top_k: int = 4,
    search_rollout_steps: int = 16,
    search_rollouts_per_action: int = 1,
    search_min_turn: int = 0,
    search_require_contact: bool = False,
    search_min_grid_size: int = 0,
    search_max_grid_size: int = 0,
    search_army_weight: float = 1.0,
    search_land_weight: float = 10.0,
    search_prior_weight: float = 0.001,
    search_terminal_score: float = 100.0,
):
    """Collect one learner-seat rollout."""
    step_data = []
    for _ in range(num_steps):
        states, effective_sizes, scoreboard_history, fog_memory, data, key = collect_online_search_step(
            states,
            effective_sizes,
            pool,
            network,
            adapter_network,
            opponent_network,
            key,
            truncation,
            opponent_id,
            learner_player,
            policy_mode,
            opponent_policy_mode,
            opponent_policy_grid_size,
            pad_size,
            global_context,
            scoreboard_history,
            scoreboard_history_enabled,
            fog_memory,
            fog_memory_enabled,
            adapter_scale,
            adapter_mode,
            adapter_min_grid_size,
            adapter_max_grid_size,
            search_top_k,
            search_rollout_steps,
            search_rollouts_per_action,
            search_min_turn,
            search_require_contact,
            search_min_grid_size,
            search_max_grid_size,
            search_army_weight,
            search_land_weight,
            search_prior_weight,
            search_terminal_score,
        )
        step_data.append(data)
    return states, effective_sizes, scoreboard_history, fog_memory, jax.tree.map(lambda *xs: jnp.stack(xs), *step_data), key


def concatenate_rollouts(*rollouts):
    """Concatenate p0/p1 rollout data along the environment axis."""
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *rollouts)


def prepare_arrays(rollout, logit_dtype: str) -> dict[str, np.ndarray]:
    """Flatten time/env axes and cast arrays for shard storage."""
    (
        obs,
        masks,
        active,
        logits,
        base_action_indices,
        executed_action_indices,
        effective_sizes,
        seats,
        times,
        search_used,
        action_changed,
        candidate_indices,
        prior_scores,
        search_scores,
        search_outcomes,
        best_positions,
        best_scores,
        score_gaps,
        best_outcomes,
        labels,
        dones,
        winners,
        contact,
    ) = rollout
    (
        enemy_general,
        _enemy_owned,
        _hidden_enemy_owned,
        _hidden_enemy_army,
        _city_map,
        source_heatmap,
        target_heatmap,
        intent,
        visible_enemy_count,
        visible_enemy_density,
        _contact_from_labels,
    ) = labels

    def flat(array):
        return np.asarray(array.reshape(array.shape[0] * array.shape[1], *array.shape[2:]))

    flat_logits = flat(logits)
    if logit_dtype == "float16":
        flat_logits = np.clip(flat_logits, -1.0e4, 1.0e4).astype(np.float16)
    else:
        flat_logits = flat_logits.astype(np.float32)

    def clipped_float16(array):
        return np.clip(flat(array).astype(np.float32), -1.0e4, 1.0e4).astype(np.float16)

    arrays = {
        "obs": flat(obs).astype(np.float16),
        "legal_mask": flat(masks).astype(np.bool_),
        "active": flat(active).astype(np.bool_),
        "teacher_logits": flat_logits,
        "teacher_action_index": flat(executed_action_indices).astype(np.int32),
        "base_action_index": flat(base_action_indices).astype(np.int32),
        "search_action_index": flat(executed_action_indices).astype(np.int32),
        "grid_size": flat(effective_sizes).astype(np.int16),
        "seat": flat(seats).astype(np.int8),
        "time": flat(times).astype(np.int16),
        "search_used": flat(search_used).astype(np.bool_),
        "search_action_changed": flat(action_changed).astype(np.bool_),
        "search_candidate_indices": flat(candidate_indices).astype(np.int32),
        "search_prior_scores": clipped_float16(prior_scores),
        "search_scores": clipped_float16(search_scores),
        "search_outcomes": flat(search_outcomes).astype(np.int8),
        "search_best_position": flat(best_positions).astype(np.int8),
        "search_best_score": clipped_float16(best_scores),
        "search_score_gap": clipped_float16(score_gaps),
        "search_best_outcome": flat(best_outcomes).astype(np.int8),
        "done": flat(dones).astype(np.bool_),
        "winner": flat(winners).astype(np.int8),
        "contact": flat(contact).astype(np.float16),
        "visible_enemy_count": flat(visible_enemy_count).astype(np.int16),
        "visible_enemy_density": flat(visible_enemy_density).astype(np.float16),
        "intent": flat(intent).astype(np.int32),
        "enemy_general_heatmap": flat(enemy_general).astype(np.float16),
        "source_heatmap": flat(source_heatmap).astype(np.float16),
        "target_heatmap": flat(target_heatmap).astype(np.float16),
        "outcome": np.full(flat(times).shape, 1, dtype=np.int32),
        "outcome_known": np.zeros(flat(times).shape, dtype=np.float16),
        "finish_within_50": np.zeros(flat(times).shape, dtype=np.float16),
        "finish_within_100": np.zeros(flat(times).shape, dtype=np.float16),
        "finish_within_250": np.zeros(flat(times).shape, dtype=np.float16),
    }
    return arrays


def filter_arrays(
    arrays: dict[str, np.ndarray],
    min_save_turn: int,
    require_search_used: bool,
    require_action_changed: bool,
    min_search_score_gap: float,
) -> tuple[dict[str, np.ndarray], int, int]:
    """Filter rows before writing while preserving aligned axes."""
    original = int(arrays["obs"].shape[0])
    keep = np.ones((original,), dtype=np.bool_)
    if min_save_turn > 0:
        keep &= arrays["time"].astype(np.int32) >= min_save_turn
    if require_search_used:
        keep &= arrays["search_used"].astype(np.bool_)
    if require_action_changed:
        keep &= arrays["search_action_changed"].astype(np.bool_)
    if min_search_score_gap > 0.0:
        keep &= arrays["search_score_gap"].astype(np.float32) >= min_search_score_gap
    return {name: value[keep] for name, value in arrays.items()}, original, int(np.sum(keep))


def save_shard(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write one compressed NPZ shard and JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect adaptive online-search trace shards.")
    parser.add_argument("num_envs", nargs="?", type=int, default=16)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=1024)
    parser.add_argument("--truncation", type=int, default=750)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="unet")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--fog-memory", action="store_true")
    parser.add_argument("--value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--value-head-sizes", default=None)
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--value-sigma", type=float, default=0.04)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--strategy-finish-outputs", type=int, default=2)
    parser.add_argument("--drop-mismatched-init-leaves", action="store_true")
    parser.add_argument("--policy-adapter-path", default=None)
    parser.add_argument("--policy-adapter-scale", type=float, default=0.0)
    parser.add_argument("--policy-adapter-mode", choices=tuple(POLICY_ADAPTER_MODE_TO_ID), default="delta")
    parser.add_argument("--policy-adapter-min-grid-size", type=int, default=0)
    parser.add_argument("--policy-adapter-max-grid-size", type=int, default=0)
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--learner-seat", choices=("mixed", "p0", "p1"), default="mixed")
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--search-top-k", type=int, default=4)
    parser.add_argument("--search-rollout-steps", type=int, default=16)
    parser.add_argument("--search-rollouts-per-action", type=int, default=1)
    parser.add_argument("--search-min-turn", type=int, default=80)
    parser.add_argument("--search-require-contact", action="store_true")
    parser.add_argument("--search-min-grid-size", type=int, default=0)
    parser.add_argument("--search-max-grid-size", type=int, default=0)
    parser.add_argument("--search-army-weight", type=float, default=1.0)
    parser.add_argument("--search-land-weight", type=float, default=10.0)
    parser.add_argument("--search-prior-weight", type=float, default=0.001)
    parser.add_argument("--search-terminal-score", type=float, default=100.0)
    parser.add_argument("--min-save-turn", type=int, default=0)
    parser.add_argument("--require-search-used", action="store_true")
    parser.add_argument("--require-action-changed", action="store_true")
    parser.add_argument("--min-search-score-gap", type=float, default=0.0)
    parser.add_argument("--output-dir", default="runs/adaptive-online-search-traces")
    parser.add_argument("--shard-prefix", default="online-search")
    parser.add_argument("--logit-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
        args.grid_size_weights = parse_grid_size_weights(args.grid_size_weights, args.grid_sizes)
        args.channels = parse_policy_channels(args.channels)
        args.opponent_channels = parse_policy_channels(args.opponent_channels)
        args.value_head_sizes = (
            parse_grid_sizes(args.value_head_sizes) if args.value_head_sizes is not None else args.grid_sizes
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_envs <= 0 or args.num_steps <= 0 or args.num_shards <= 0:
        parser.error("num_envs, --num-steps, and --num-shards must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.learner_seat == "mixed" and args.num_envs < 2:
        parser.error("mixed learner-seat requires num_envs >= 2")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.value_loss == "hl-gauss" and args.value_bins <= 1:
        parser.error("--value-bins must be greater than 1")
    if args.policy_adapter_scale < 0.0:
        parser.error("--policy-adapter-scale must be non-negative")
    if args.policy_adapter_path is None and args.policy_adapter_scale > 0.0:
        parser.error("--policy-adapter-scale requires --policy-adapter-path")
    if args.policy_adapter_min_grid_size < 0 or args.policy_adapter_max_grid_size < 0:
        parser.error("--policy-adapter-min/max-grid-size must be non-negative")
    if (
        args.policy_adapter_min_grid_size > 0
        and args.policy_adapter_max_grid_size > 0
        and args.policy_adapter_min_grid_size > args.policy_adapter_max_grid_size
    ):
        parser.error("--policy-adapter-min-grid-size must be <= --policy-adapter-max-grid-size")
    if args.opponent_policy_path is not None and len(args.grid_sizes) != 1:
        parser.error("--opponent-policy-path requires exactly one --grid-sizes value")
    if args.search_top_k <= 0 or args.search_rollout_steps <= 0 or args.search_rollouts_per_action <= 0:
        parser.error("search counts must be positive")
    if args.search_min_turn < 0 or args.min_save_turn < 0:
        parser.error("turn filters must be non-negative")
    if args.search_min_grid_size < 0 or args.search_max_grid_size < 0:
        parser.error("--search-min/max-grid-size must be non-negative")
    if (
        args.search_min_grid_size > 0
        and args.search_max_grid_size > 0
        and args.search_min_grid_size > args.search_max_grid_size
    ):
        parser.error("--search-min-grid-size must be <= --search-max-grid-size")
    if args.min_search_score_gap < 0.0:
        parser.error("--min-search-score-gap must be non-negative")
    return args


def load_adaptive_model(args, key):
    """Load the adaptive base model using the target template."""
    network_global_context = args.global_context or args.scoreboard_history
    input_channels = (
        args.input_channels
        if args.input_channels is not None
        else adaptive_input_channel_count(network_global_context, args.scoreboard_history, args.fog_memory)
    )
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    return load_or_create_adaptive_network(
        key,
        pad_size=args.pad_to,
        init_model_path=args.model_path,
        channels=args.channels,
        input_channels=input_channels,
        init_input_channels=input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        init_value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        value_bins=value_bins,
        init_value_bins=value_bins,
        value_min=args.value_min,
        value_max=args.value_max,
        value_sigma=args.value_sigma,
        outcome_head=args.outcome_head,
        init_outcome_head=args.outcome_head,
        strategy_aux=args.strategy_aux,
        init_strategy_aux=args.strategy_aux,
        strategy_spatial_aux=args.strategy_spatial_aux,
        init_strategy_spatial_aux=args.strategy_spatial_aux,
        strategy_finish_outputs=args.strategy_finish_outputs,
        init_strategy_finish_outputs=args.strategy_finish_outputs,
        global_context=network_global_context,
        init_global_context=network_global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
        drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
    )


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)
    key, pool_key, model_key, adapter_key, opponent_key = jrandom.split(key, 5)
    policy_mode = POLICY_MODE_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_TO_ID[args.opponent_policy_mode]
    adapter_mode = POLICY_ADAPTER_MODE_TO_ID[args.policy_adapter_mode]
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    network_global_context = args.global_context or args.scoreboard_history

    print("Adaptive online-search trace collection")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Environments:  {args.num_envs} learner_seat={args.learner_seat}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Model:         {args.model_path}")
    print(f"Opponent:      {args.opponent_policy_path or args.opponent}")
    print(
        "Search:        "
        f"top_k={args.search_top_k}, rollout={args.search_rollout_steps}, "
        f"rollouts/action={args.search_rollouts_per_action}, min_turn={args.search_min_turn}, "
        f"contact={args.search_require_contact}"
    )
    if args.warmup_steps > 0:
        print(f"Warmup:        {args.warmup_steps} deployment steps before saving")
    print(f"Output:        {args.output_dir}")

    network = load_adaptive_model(args, model_key)
    adapter_network = None
    if args.policy_adapter_path is not None:
        adapter_network = load_adaptive_model(
            argparse.Namespace(**{**vars(args), "model_path": args.policy_adapter_path}),
            adapter_key,
        )
    opponent_network = None
    if args.opponent_policy_path is not None:
        opponent_channels = args.opponent_channels or (32, 32, 32, 16)
        opponent_network = PolicyValueNetwork(
            opponent_key,
            grid_size=args.grid_sizes[0],
            channels=opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_network)

    pool = make_adaptive_state_pool(
        pool_key,
        args.pool_size,
        args.grid_sizes,
        args.pad_to,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
        args.grid_size_weights,
    )
    initial_states, initial_sizes = make_adaptive_initial_states(pool, args.num_envs)
    if args.learner_seat == "mixed":
        p0_envs, p1_envs = split_mixed_env_counts(args.num_envs)
        states_p0 = jax.tree.map(lambda value: value[:p0_envs], initial_states)
        sizes_p0 = initial_sizes[:p0_envs]
        states_p1 = jax.tree.map(lambda value: value[p0_envs : p0_envs + p1_envs], initial_states)
        sizes_p1 = initial_sizes[p0_envs : p0_envs + p1_envs]
        history_p0 = empty_scoreboard_history(p0_envs)
        history_p1 = empty_scoreboard_history(p1_envs)
        memory_p0 = empty_adaptive_fog_memory(p0_envs, args.pad_to)
        memory_p1 = empty_adaptive_fog_memory(p1_envs, args.pad_to)
    else:
        states = initial_states
        sizes = initial_sizes
        history = empty_scoreboard_history(args.num_envs)
        memory = empty_adaptive_fog_memory(args.num_envs, args.pad_to)
        learner_player = 0 if args.learner_seat == "p0" else 1

    if args.warmup_steps > 0:
        warmup_t0 = time.time()
        if args.learner_seat == "mixed":
            key, p0_key, p1_key = jrandom.split(key, 3)
            states_p0, sizes_p0, history_p0, memory_p0, _, _ = collect_rollout(
                states_p0,
                sizes_p0,
                pool,
                network,
                adapter_network,
                opponent_network,
                p0_key,
                args.warmup_steps,
                args.truncation,
                opponent_id,
                0,
                policy_mode,
                opponent_policy_mode,
                args.grid_sizes[0],
                args.pad_to,
                network_global_context,
                history_p0,
                args.scoreboard_history,
                memory_p0,
                args.fog_memory,
                args.policy_adapter_scale,
                adapter_mode,
                args.policy_adapter_min_grid_size,
                args.policy_adapter_max_grid_size,
                args.search_top_k,
                args.search_rollout_steps,
                args.search_rollouts_per_action,
                args.search_min_turn,
                args.search_require_contact,
                args.search_min_grid_size,
                args.search_max_grid_size,
                args.search_army_weight,
                args.search_land_weight,
                args.search_prior_weight,
                args.search_terminal_score,
            )
            states_p1, sizes_p1, history_p1, memory_p1, _, _ = collect_rollout(
                states_p1,
                sizes_p1,
                pool,
                network,
                adapter_network,
                opponent_network,
                p1_key,
                args.warmup_steps,
                args.truncation,
                opponent_id,
                1,
                policy_mode,
                opponent_policy_mode,
                args.grid_sizes[0],
                args.pad_to,
                network_global_context,
                history_p1,
                args.scoreboard_history,
                memory_p1,
                args.fog_memory,
                args.policy_adapter_scale,
                adapter_mode,
                args.policy_adapter_min_grid_size,
                args.policy_adapter_max_grid_size,
                args.search_top_k,
                args.search_rollout_steps,
                args.search_rollouts_per_action,
                args.search_min_turn,
                args.search_require_contact,
                args.search_min_grid_size,
                args.search_max_grid_size,
                args.search_army_weight,
                args.search_land_weight,
                args.search_prior_weight,
                args.search_terminal_score,
            )
        else:
            key, warmup_key = jrandom.split(key)
            states, sizes, history, memory, _, _ = collect_rollout(
                states,
                sizes,
                pool,
                network,
                adapter_network,
                opponent_network,
                warmup_key,
                args.warmup_steps,
                args.truncation,
                opponent_id,
                learner_player,
                policy_mode,
                opponent_policy_mode,
                args.grid_sizes[0],
                args.pad_to,
                network_global_context,
                history,
                args.scoreboard_history,
                memory,
                args.fog_memory,
                args.policy_adapter_scale,
                adapter_mode,
                args.policy_adapter_min_grid_size,
                args.policy_adapter_max_grid_size,
                args.search_top_k,
                args.search_rollout_steps,
                args.search_rollouts_per_action,
                args.search_min_turn,
                args.search_require_contact,
                args.search_min_grid_size,
                args.search_max_grid_size,
                args.search_army_weight,
                args.search_land_weight,
                args.search_prior_weight,
                args.search_terminal_score,
            )
        print(f"Warmup complete | time={time.time() - warmup_t0:.2f}s")

    output_dir = Path(args.output_dir)
    total_saved = 0
    for shard_index in range(args.num_shards):
        t0 = time.time()
        if args.learner_seat == "mixed":
            key, p0_key, p1_key = jrandom.split(key, 3)
            states_p0, sizes_p0, history_p0, memory_p0, rollout_p0, _ = collect_rollout(
                states_p0,
                sizes_p0,
                pool,
                network,
                adapter_network,
                opponent_network,
                p0_key,
                args.num_steps,
                args.truncation,
                opponent_id,
                0,
                policy_mode,
                opponent_policy_mode,
                args.grid_sizes[0],
                args.pad_to,
                network_global_context,
                history_p0,
                args.scoreboard_history,
                memory_p0,
                args.fog_memory,
                args.policy_adapter_scale,
                adapter_mode,
                args.policy_adapter_min_grid_size,
                args.policy_adapter_max_grid_size,
                args.search_top_k,
                args.search_rollout_steps,
                args.search_rollouts_per_action,
                args.search_min_turn,
                args.search_require_contact,
                args.search_min_grid_size,
                args.search_max_grid_size,
                args.search_army_weight,
                args.search_land_weight,
                args.search_prior_weight,
                args.search_terminal_score,
            )
            states_p1, sizes_p1, history_p1, memory_p1, rollout_p1, _ = collect_rollout(
                states_p1,
                sizes_p1,
                pool,
                network,
                adapter_network,
                opponent_network,
                p1_key,
                args.num_steps,
                args.truncation,
                opponent_id,
                1,
                policy_mode,
                opponent_policy_mode,
                args.grid_sizes[0],
                args.pad_to,
                network_global_context,
                history_p1,
                args.scoreboard_history,
                memory_p1,
                args.fog_memory,
                args.policy_adapter_scale,
                adapter_mode,
                args.policy_adapter_min_grid_size,
                args.policy_adapter_max_grid_size,
                args.search_top_k,
                args.search_rollout_steps,
                args.search_rollouts_per_action,
                args.search_min_turn,
                args.search_require_contact,
                args.search_min_grid_size,
                args.search_max_grid_size,
                args.search_army_weight,
                args.search_land_weight,
                args.search_prior_weight,
                args.search_terminal_score,
            )
            rollout = concatenate_rollouts(rollout_p0, rollout_p1)
        else:
            key, rollout_key = jrandom.split(key)
            states, sizes, history, memory, rollout, _ = collect_rollout(
                states,
                sizes,
                pool,
                network,
                adapter_network,
                opponent_network,
                rollout_key,
                args.num_steps,
                args.truncation,
                opponent_id,
                learner_player,
                policy_mode,
                opponent_policy_mode,
                args.grid_sizes[0],
                args.pad_to,
                network_global_context,
                history,
                args.scoreboard_history,
                memory,
                args.fog_memory,
                args.policy_adapter_scale,
                adapter_mode,
                args.policy_adapter_min_grid_size,
                args.policy_adapter_max_grid_size,
                args.search_top_k,
                args.search_rollout_steps,
                args.search_rollouts_per_action,
                args.search_min_turn,
                args.search_require_contact,
                args.search_min_grid_size,
                args.search_max_grid_size,
                args.search_army_weight,
                args.search_land_weight,
                args.search_prior_weight,
                args.search_terminal_score,
            )
        arrays = prepare_arrays(rollout, args.logit_dtype)
        arrays, original_count, saved_count = filter_arrays(
            arrays,
            args.min_save_turn,
            args.require_search_used,
            args.require_action_changed,
            args.min_search_score_gap,
        )
        if saved_count == 0:
            print(f"Shard {shard_index:04d} skipped | samples=0/{original_count}")
            continue
        shard_path = output_dir / f"{args.shard_prefix}-{shard_index:05d}.npz"
        metadata = {
            "shard_index": shard_index,
            "source": "adaptive_online_search_trace_dataset.py",
            "model_path": args.model_path,
            "policy_adapter_path": args.policy_adapter_path,
            "opponent": args.opponent,
            "opponent_policy_path": args.opponent_policy_path,
            "grid_sizes": list(args.grid_sizes),
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "num_envs": args.num_envs,
            "search_top_k": args.search_top_k,
            "search_rollout_steps": args.search_rollout_steps,
            "search_rollouts_per_action": args.search_rollouts_per_action,
            "search_min_turn": args.search_min_turn,
            "search_require_contact": args.search_require_contact,
            "search_min_grid_size": args.search_min_grid_size,
            "search_max_grid_size": args.search_max_grid_size,
            "original_count": original_count,
            "saved_count": saved_count,
        }
        save_shard(shard_path, arrays, metadata)
        total_saved += saved_count
        changed = float(np.mean(arrays["search_action_changed"])) if saved_count else 0.0
        used = float(np.mean(arrays["search_used"])) if saved_count else 0.0
        mean_gap = float(np.mean(arrays["search_score_gap"].astype(np.float32))) if saved_count else 0.0
        print(
            f"Shard {shard_index:04d} | samples={saved_count}/{original_count} | "
            f"search_used={used:.3f} changed={changed:.3f} mean_gap={mean_gap:.3f} | "
            f"path={shard_path} | time={time.time() - t0:.2f}s"
        )
    print(f"Done. saved_samples={total_saved}")


if __name__ == "__main__":
    main()
