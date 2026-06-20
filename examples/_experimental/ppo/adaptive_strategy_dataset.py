"""Generate offline strategy-supervision shards for adaptive Generals.io policies."""

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
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_action_to_index,
    adaptive_expander_target_probs,
    adaptive_index_to_action,
    adaptive_input_channel_count,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    active_cells_for_size,
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
from adaptive_network import adaptive_network_input_channels, load_or_create_adaptive_network
from adaptive_search_distill import adaptive_rollout_search_candidates, adaptive_score_observation, outcome_class_from_winner
from adaptive_strategy_aux import weak_intent_label
from adaptive_teacher_imitation import (
    fixed_policy_teacher_logits,
    policy_action_from_logits,
    rollout_steps_to_next_done,
)
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, POLICY_MODE_NAMES, opponent_action, policy_network_action
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from generals.core import game
from train import random_action, stack_learner_actions
from train_adaptive import (
    OUTCOME_DRAW,
    OUTCOME_WIN,
    crop_observation,
    rollout_outcome_targets,
    split_mixed_env_counts,
    teacher_obs_from_student_obs,
)

TEACHER_KINDS = ("adaptive", "fixed", "expander", "search", "fixed-search")
TEACHER_KIND_TO_ID = {name: index for index, name in enumerate(TEACHER_KINDS)}
POLICY_MODE_NAME_TO_ID = {name: index for index, name in enumerate(POLICY_MODE_NAMES)}


def empty_scoreboard_history(num_envs: int) -> jnp.ndarray:
    """Return empty previous-scoreboard features for vectorized dataset rollouts."""
    return jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)


def one_hot_cell(row: jnp.ndarray, col: jnp.ndarray, pad_size: int, active: jnp.ndarray) -> jnp.ndarray:
    """Return a one-hot spatial map clipped to active board cells."""
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    return ((rows == row) & (cols == col) & active).astype(jnp.float32)


def main_stack_heatmap(state, learner_player: int, active: jnp.ndarray) -> jnp.ndarray:
    """Return one-hot map for the learner-owned cell with the largest army."""
    own = state.ownership[learner_player] & active
    scores = jnp.where(own, state.armies, -1)
    flat_index = jnp.argmax(scores.reshape(-1))
    row = flat_index // state.armies.shape[1]
    col = flat_index % state.armies.shape[1]
    return one_hot_cell(row, col, state.armies.shape[0], active)


def full_state_strategy_labels(state, obs, learner_player: int, effective_size: int, pad_size: int):
    """Build belief, target, source, and weak-intent labels from the privileged state."""
    opponent = 1 - learner_player
    active = active_cells_for_size(effective_size, pad_size)
    enemy_general_row, enemy_general_col = state.general_positions[opponent]
    enemy_general = one_hot_cell(enemy_general_row, enemy_general_col, pad_size, active)
    enemy_owned = (state.ownership[opponent] & active).astype(jnp.float32)
    hidden_enemy_owned = (state.ownership[opponent] & obs.fog_cells & active).astype(jnp.float32)
    hidden_enemy_army = jnp.log1p(jnp.maximum(state.armies.astype(jnp.float32), 0.0)) * hidden_enemy_owned
    city_map = (state.cities & active).astype(jnp.float32)
    source = main_stack_heatmap(state, learner_player, active)
    no_search_outcomes = jnp.full((1,), -1, dtype=jnp.int32)
    intent = weak_intent_label(obs, state, learner_player, no_search_outcomes)
    visible_enemy_count = jnp.sum(obs.opponent_cells.astype(jnp.float32) * active.astype(jnp.float32))
    visible_enemy_density = visible_enemy_count / jnp.maximum(
        jnp.sum(active.astype(jnp.float32)),
        1.0,
    )
    contact = jnp.any(obs.opponent_cells & active).astype(jnp.float32)
    return (
        enemy_general,
        enemy_owned,
        hidden_enemy_owned,
        hidden_enemy_army,
        city_map,
        source,
        enemy_general,
        intent,
        visible_enemy_count,
        visible_enemy_density,
        contact,
    )


def teacher_logits_for_batch(
    teacher_kind_id: int,
    teacher_network,
    fixed_teacher_network,
    obs_arr,
    masks,
    active,
    learner_obs,
    effective_sizes,
    teacher_input_channels: int,
    fixed_teacher_grid_size: int,
    pad_size: int,
):
    """Return adaptive-space logits for one teacher kind."""
    if teacher_kind_id in (TEACHER_KIND_TO_ID["adaptive"], TEACHER_KIND_TO_ID["search"]):
        teacher_obs_arr = jax.vmap(lambda obs: teacher_obs_from_student_obs(obs, teacher_input_channels))(obs_arr)
        return jax.vmap(lambda obs, mask, active_mask: teacher_network.logits_value(obs, mask, active_mask)[0])(
            teacher_obs_arr,
            masks,
            active,
        )
    if teacher_kind_id in (TEACHER_KIND_TO_ID["fixed"], TEACHER_KIND_TO_ID["fixed-search"]):
        del effective_sizes
        return jax.vmap(
            lambda obs: fixed_policy_teacher_logits(fixed_teacher_network, obs, fixed_teacher_grid_size, pad_size)
        )(learner_obs)
    if teacher_kind_id == TEACHER_KIND_TO_ID["expander"]:
        probs = jax.vmap(lambda obs, size: adaptive_expander_target_probs(obs, size, pad_size))(
            learner_obs,
            effective_sizes,
        )
        return jnp.log(jnp.maximum(probs, 1.0e-8))
    raise ValueError(f"unknown teacher kind id: {teacher_kind_id}")


@eqx.filter_jit
def fixed_rollout_search_candidates(
    network,
    state,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    fixed_grid_size: int,
    pad_size: int,
):
    """Return top-k rollout-search candidates for an 8x8 fixed-policy teacher.

    The surrounding adaptive dataset uses padded states and adaptive action
    indices. This adapter keeps the fixed teacher on cropped observations while
    executing the selected actions in the padded game state.
    """
    obs = game.get_observation(state, player)
    logits = fixed_policy_teacher_logits(network, obs, fixed_grid_size, pad_size)
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda idx: adaptive_index_to_action(idx, pad_size))(candidate_indices)

    opponent_player = 1 - player
    opponent_obs = game.get_observation(state, opponent_player)
    key, opponent_key = jrandom.split(key)
    opponent_first_action = policy_network_action(
        network,
        opponent_key,
        crop_observation(opponent_obs, fixed_grid_size),
        policy_mode,
    )

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, step_key = carry
            step_key, k0, k1 = jrandom.split(step_key, 3)
            obs_p0 = game.get_observation(rollout_state, 0)
            obs_p1 = game.get_observation(rollout_state, 1)
            action_p0 = policy_network_action(
                network,
                k0,
                crop_observation(obs_p0, fixed_grid_size),
                policy_mode,
            )
            action_p1 = policy_network_action(
                network,
                k1,
                crop_observation(obs_p1, fixed_grid_size),
                policy_mode,
            )
            next_state, _ = game.step(rollout_state, jnp.stack([action_p0, action_p1]))
            already_done = game.get_info(rollout_state).is_done
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            return (final_state, step_key), None

        (final_state, _), _ = jax.lax.scan((body), (initial_state, rollout_key), None, length=rollout_steps)
        final_info = game.get_info(final_state)
        final_obs = game.get_observation(final_state, player)
        score = adaptive_score_observation(final_info, final_obs, player, army_weight, land_weight, terminal_score)
        outcome = outcome_class_from_winner(jnp.where(final_info.is_done, final_info.winner, -1), player)
        return score, outcome

    def candidate_score(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        scores, outcomes = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        best_rollout = jnp.argmax(scores)
        rollout_outcome = outcomes[best_rollout]
        first_outcome = outcome_class_from_winner(first_info.winner, player)
        first_terminal = jnp.where(
            first_info.winner == player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        candidate_score = first_terminal + jnp.mean(scores) + prior_weight * prior_score
        candidate_outcome = jnp.where(first_info.is_done, first_outcome, rollout_outcome)
        return candidate_score, candidate_outcome

    candidate_keys = jrandom.split(key, top_k)
    scores, outcomes = jax.vmap(candidate_score)(candidate_actions, prior_scores, candidate_keys)
    return candidate_actions, candidate_indices, prior_scores, scores, outcomes


@eqx.filter_jit
def collect_strategy_step(
    states,
    effective_sizes,
    pool,
    teacher_network,
    fixed_teacher_network,
    key,
    truncation,
    opponent_id,
    learner_player,
    teacher_kind_id: int,
    teacher_policy_mode_id: int,
    pad_size: int,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    fixed_teacher_grid_size: int = 0,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
    search_top_k: int = 4,
    search_rollout_steps: int = 16,
    search_rollouts_per_action: int = 2,
    search_army_weight: float = 12.0,
    search_land_weight: float = 8.0,
    search_prior_weight: float = 0.01,
    search_terminal_score: float = 1000.0,
):
    """Collect one vectorized strategy-dataset step with privileged labels."""
    num_envs = states.armies.shape[0]
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p0_prior, lambda _: obs_p1_prior, None)
    opponent_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p1_prior, lambda _: obs_p0_prior, None)

    if scoreboard_history is None:
        scoreboard_history = empty_scoreboard_history(num_envs)
    if fog_memory is None:
        fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)
    current_fog_memory = (
        jax.vmap(update_adaptive_fog_memory)(fog_memory, learner_obs_prior) if fog_memory_enabled else fog_memory
    )

    if scoreboard_history_enabled:
        current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
            learner_obs_prior,
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
            )(learner_obs_prior, effective_sizes, history_context, current_fog_memory)
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size, history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                )
            )(learner_obs_prior, effective_sizes, history_context)
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
            )(learner_obs_prior, effective_sizes, current_fog_memory)
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
            )(learner_obs_prior, effective_sizes)

    masks = jax.vmap(
        lambda obs, size: compute_adaptive_valid_move_mask(
            obs.armies,
            obs.owned_cells,
            obs.mountains,
            size,
            pad_size,
        )
    )(learner_obs_prior, effective_sizes)

    teacher_logits = teacher_logits_for_batch(
        teacher_kind_id,
        teacher_network,
        fixed_teacher_network,
        obs_arr,
        masks,
        active,
        learner_obs_prior,
        effective_sizes,
        teacher_input_channels,
        fixed_teacher_grid_size,
        pad_size,
    )
    key, teacher_key, search_key, opponent_key = jrandom.split(key, 4)
    teacher_keys = jrandom.split(teacher_key, num_envs)
    prior_actions = jax.vmap(
        lambda logits, sample_key: policy_action_from_logits(logits, sample_key, teacher_policy_mode_id, pad_size)
    )(
        teacher_logits,
        teacher_keys,
    )
    if teacher_kind_id == TEACHER_KIND_TO_ID["search"]:
        search_keys = jrandom.split(search_key, num_envs)
        empty_previous = empty_scoreboard_history(num_envs)
        search_previous_p0 = jax.lax.cond(
            learner_player == 0,
            lambda _: scoreboard_history,
            lambda _: empty_previous,
            None,
        )
        search_previous_p1 = jax.lax.cond(
            learner_player == 1,
            lambda _: scoreboard_history,
            lambda _: empty_previous,
            None,
        )
        empty_fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)
        search_previous_fog_p0 = jax.lax.cond(
            learner_player == 0,
            lambda _: fog_memory,
            lambda _: empty_fog_memory,
            None,
        )
        search_previous_fog_p1 = jax.lax.cond(
            learner_player == 1,
            lambda _: fog_memory,
            lambda _: empty_fog_memory,
            None,
        )
        search_actions, search_candidate_indices, search_prior_scores, search_scores, search_outcomes = jax.vmap(
            lambda state, size, action_key, previous_p0, previous_p1, previous_fog_p0, previous_fog_p1: adaptive_rollout_search_candidates(
                teacher_network,
                state,
                size,
                action_key,
                learner_player,
                search_top_k,
                search_rollout_steps,
                search_rollouts_per_action,
                teacher_policy_mode_id,
                search_army_weight,
                search_land_weight,
                search_prior_weight,
                search_terminal_score,
                pad_size,
                global_context=global_context,
                scoreboard_history_enabled=scoreboard_history_enabled,
                previous_scoreboard_p0=previous_p0,
                previous_scoreboard_p1=previous_p1,
                fog_memory_enabled=fog_memory_enabled,
                previous_fog_memory_p0=previous_fog_p0,
                previous_fog_memory_p1=previous_fog_p1,
            )
        )(
            states,
            effective_sizes,
            search_keys,
            search_previous_p0,
            search_previous_p1,
            search_previous_fog_p0,
            search_previous_fog_p1,
        )
        best_search_positions = jnp.argmax(search_scores, axis=-1)
        learner_actions = jnp.take_along_axis(search_actions, best_search_positions[:, None, None], axis=1)[:, 0]
        search_best_scores = jnp.take_along_axis(search_scores, best_search_positions[:, None], axis=1)[:, 0]
        valid_search_candidates = search_prior_scores > -1.0e8
        finite_count = jnp.sum(valid_search_candidates.astype(jnp.float32), axis=-1)
        finite_score_sum = jnp.sum(jnp.where(valid_search_candidates, search_scores, 0.0), axis=-1)
        search_mean_scores = finite_score_sum / jnp.maximum(finite_count, 1.0)
        candidate_positions = jnp.arange(search_top_k)[None, :]
        second_scores = jnp.max(
            jnp.where(
                valid_search_candidates & (candidate_positions != best_search_positions[:, None]),
                search_scores,
                -jnp.inf,
            ),
            axis=-1,
        )
        search_score_gaps = jnp.where(finite_count > 1.0, search_best_scores - second_scores, 0.0)
        search_best_outcomes = jnp.take_along_axis(search_outcomes, best_search_positions[:, None], axis=1)[:, 0]
    elif teacher_kind_id == TEACHER_KIND_TO_ID["fixed-search"]:
        search_keys = jrandom.split(search_key, num_envs)
        search_actions, search_candidate_indices, search_prior_scores, search_scores, search_outcomes = jax.vmap(
            lambda state, action_key: fixed_rollout_search_candidates(
                fixed_teacher_network,
                state,
                action_key,
                learner_player,
                search_top_k,
                search_rollout_steps,
                search_rollouts_per_action,
                teacher_policy_mode_id,
                search_army_weight,
                search_land_weight,
                search_prior_weight,
                search_terminal_score,
                fixed_teacher_grid_size,
                pad_size,
            )
        )(states, search_keys)
        best_search_positions = jnp.argmax(search_scores, axis=-1)
        learner_actions = jnp.take_along_axis(search_actions, best_search_positions[:, None, None], axis=1)[:, 0]
        search_best_scores = jnp.take_along_axis(search_scores, best_search_positions[:, None], axis=1)[:, 0]
        valid_search_candidates = search_prior_scores > -1.0e8
        finite_count = jnp.sum(valid_search_candidates.astype(jnp.float32), axis=-1)
        finite_score_sum = jnp.sum(jnp.where(valid_search_candidates, search_scores, 0.0), axis=-1)
        search_mean_scores = finite_score_sum / jnp.maximum(finite_count, 1.0)
        candidate_positions = jnp.arange(search_top_k)[None, :]
        second_scores = jnp.max(
            jnp.where(
                valid_search_candidates & (candidate_positions != best_search_positions[:, None]),
                search_scores,
                -jnp.inf,
            ),
            axis=-1,
        )
        search_score_gaps = jnp.where(finite_count > 1.0, search_best_scores - second_scores, 0.0)
        search_best_outcomes = jnp.take_along_axis(search_outcomes, best_search_positions[:, None], axis=1)[:, 0]
    else:
        learner_actions = prior_actions
        search_candidate_indices = jnp.zeros((num_envs, search_top_k), dtype=jnp.int32)
        search_prior_scores = jnp.zeros((num_envs, search_top_k), dtype=jnp.float32)
        search_scores = jnp.zeros((num_envs, search_top_k), dtype=jnp.float32)
        search_outcomes = jnp.full((num_envs, search_top_k), -1, dtype=jnp.int32)
        best_search_positions = jnp.zeros((num_envs,), dtype=jnp.int32)
        search_best_scores = jnp.zeros((num_envs,), dtype=jnp.float32)
        search_mean_scores = jnp.zeros((num_envs,), dtype=jnp.float32)
        search_score_gaps = jnp.zeros((num_envs,), dtype=jnp.float32)
        search_best_outcomes = jnp.full((num_envs,), -1, dtype=jnp.int32)
    teacher_indices = jax.vmap(lambda action: adaptive_action_to_index(action, pad_size))(learner_actions)
    prior_greedy_indices = jnp.argmax(teacher_logits, axis=-1).astype(jnp.int32)
    if teacher_kind_id in (TEACHER_KIND_TO_ID["search"], TEACHER_KIND_TO_ID["fixed-search"]):
        teacher_greedy_indices = teacher_indices
    else:
        teacher_greedy_indices = prior_greedy_indices

    labels = jax.vmap(lambda state, obs, size: full_state_strategy_labels(state, obs, learner_player, size, pad_size))(
        states,
        learner_obs_prior,
        effective_sizes,
    )

    opponent_keys = jrandom.split(opponent_key, num_envs)
    if opponent_policy_network is not None:
        opponent_actions = jax.vmap(
            lambda k, obs: policy_network_action(
                opponent_policy_network,
                k,
                crop_observation(obs, opponent_policy_grid_size),
                opponent_policy_mode,
            )
        )(opponent_keys, opponent_obs_prior)
    else:
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent_id, k, obs, random_action))(
            opponent_keys,
            opponent_obs_prior,
        )
    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)

    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated

    pool_size = pool.states.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda x: x[reset_indices], pool.states)
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
    final_scoreboard_history = reset_adaptive_scoreboard_history(current_scoreboard, dones)
    final_fog_memory = reset_adaptive_fog_memory(current_fog_memory, dones)
    return (
        final_states,
        final_sizes,
        final_scoreboard_history,
        final_fog_memory,
        (
            obs_arr,
            masks,
            active,
            learner_actions,
            teacher_indices,
            teacher_greedy_indices,
            teacher_logits,
            effective_sizes,
            labels,
            (
                search_candidate_indices,
                search_prior_scores,
                search_scores,
                search_outcomes,
                best_search_positions,
                search_best_scores,
                search_mean_scores,
                search_score_gaps,
                search_best_outcomes,
            ),
            dones,
            infos,
        ),
        key,
    )


def collect_strategy_rollout(
    states,
    effective_sizes,
    pool,
    teacher_network,
    fixed_teacher_network,
    key,
    num_steps,
    truncation,
    opponent_id,
    learner_player,
    teacher_kind_id,
    teacher_policy_mode_id,
    pad_size,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    fixed_teacher_grid_size: int = 0,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
    search_top_k: int = 4,
    search_rollout_steps: int = 16,
    search_rollouts_per_action: int = 2,
    search_army_weight: float = 12.0,
    search_land_weight: float = 8.0,
    search_prior_weight: float = 0.01,
    search_terminal_score: float = 1000.0,
):
    """Collect a rollout for one learner seat."""
    step_data = []
    for _ in range(num_steps):
        states, effective_sizes, scoreboard_history, fog_memory, data, key = collect_strategy_step(
            states,
            effective_sizes,
            pool,
            teacher_network,
            fixed_teacher_network,
            key,
            truncation,
            opponent_id,
            learner_player,
            teacher_kind_id,
            teacher_policy_mode_id,
            pad_size,
            global_context,
            scoreboard_history,
            scoreboard_history_enabled,
            fog_memory,
            fog_memory_enabled,
            teacher_input_channels,
            fixed_teacher_grid_size,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
            search_top_k,
            search_rollout_steps,
            search_rollouts_per_action,
            search_army_weight,
            search_land_weight,
            search_prior_weight,
            search_terminal_score,
        )
        step_data.append(data)
    return states, effective_sizes, scoreboard_history, fog_memory, jax.tree.map(lambda *xs: jnp.stack(xs), *step_data), key


def collect_mixed_strategy_rollout(
    states_p0,
    effective_sizes_p0,
    states_p1,
    effective_sizes_p1,
    pool,
    teacher_network,
    fixed_teacher_network,
    key,
    num_steps,
    truncation,
    opponent_id,
    teacher_kind_id,
    teacher_policy_mode_id,
    pad_size,
    global_context=False,
    scoreboard_history_p0=None,
    scoreboard_history_p1=None,
    scoreboard_history_enabled=False,
    fog_memory_p0=None,
    fog_memory_p1=None,
    fog_memory_enabled=False,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    fixed_teacher_grid_size: int = 0,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
    search_top_k: int = 4,
    search_rollout_steps: int = 16,
    search_rollouts_per_action: int = 2,
    search_army_weight: float = 12.0,
    search_land_weight: float = 8.0,
    search_prior_weight: float = 0.01,
    search_terminal_score: float = 1000.0,
):
    """Collect and concatenate p0 and p1 strategy data."""
    key, p0_key, p1_key = jrandom.split(key, 3)
    states_p0, effective_sizes_p0, scoreboard_history_p0, fog_memory_p0, rollout_p0, _ = collect_strategy_rollout(
        states_p0,
        effective_sizes_p0,
        pool,
        teacher_network,
        fixed_teacher_network,
        p0_key,
        num_steps,
        truncation,
        opponent_id,
        0,
        teacher_kind_id,
        teacher_policy_mode_id,
        pad_size,
        global_context,
        scoreboard_history_p0,
        scoreboard_history_enabled,
        fog_memory_p0,
        fog_memory_enabled,
        teacher_input_channels,
        fixed_teacher_grid_size,
        opponent_policy_network,
        opponent_policy_mode,
        opponent_policy_grid_size,
        search_top_k,
        search_rollout_steps,
        search_rollouts_per_action,
        search_army_weight,
        search_land_weight,
        search_prior_weight,
        search_terminal_score,
    )
    states_p1, effective_sizes_p1, scoreboard_history_p1, fog_memory_p1, rollout_p1, _ = collect_strategy_rollout(
        states_p1,
        effective_sizes_p1,
        pool,
        teacher_network,
        fixed_teacher_network,
        p1_key,
        num_steps,
        truncation,
        opponent_id,
        1,
        teacher_kind_id,
        teacher_policy_mode_id,
        pad_size,
        global_context,
        scoreboard_history_p1,
        scoreboard_history_enabled,
        fog_memory_p1,
        fog_memory_enabled,
        teacher_input_channels,
        fixed_teacher_grid_size,
        opponent_policy_network,
        opponent_policy_mode,
        opponent_policy_grid_size,
        search_top_k,
        search_rollout_steps,
        search_rollouts_per_action,
        search_army_weight,
        search_land_weight,
        search_prior_weight,
        search_terminal_score,
    )
    rollout_data = jax.tree.map(lambda left, right: jnp.concatenate([left, right], axis=1), rollout_p0, rollout_p1)
    return (
        states_p0,
        effective_sizes_p0,
        scoreboard_history_p0,
        fog_memory_p0,
        states_p1,
        effective_sizes_p1,
        scoreboard_history_p1,
        fog_memory_p1,
        rollout_data,
        key,
    )


def terminal_time_targets(times: jnp.ndarray, dones: jnp.ndarray) -> jnp.ndarray:
    """Return the next known terminal/truncation game time for every rollout sample."""
    sentinel = jnp.full(times.shape[1:], -1, dtype=jnp.int32)

    def scan_step(next_time, inputs):
        time, done = inputs
        current = jnp.where(done, time, next_time)
        return current, current

    _, targets_rev = jax.lax.scan(scan_step, sentinel, (times[::-1], dones[::-1]))
    return targets_rev[::-1]


def flatten_rollout_data(rollout_data, learner_players: jnp.ndarray, logit_dtype: str) -> dict[str, np.ndarray]:
    """Flatten time/env axes and build derived outcome/finish targets."""
    (
        obs,
        masks,
        active,
        actions,
        teacher_indices,
        teacher_greedy_indices,
        teacher_logits,
        effective_sizes,
        labels,
        search_labels,
        dones,
        infos,
    ) = rollout_data
    (
        enemy_general,
        enemy_owned,
        hidden_enemy_owned,
        hidden_enemy_army,
        city_map,
        source_heatmap,
        target_heatmap,
        intent,
        visible_enemy_count,
        visible_enemy_density,
        contact,
    ) = labels
    (
        search_candidate_indices,
        search_prior_scores,
        search_scores,
        search_outcomes,
        search_best_positions,
        search_best_scores,
        search_mean_scores,
        search_score_gaps,
        search_best_outcomes,
    ) = search_labels
    outcome_targets, outcome_known = rollout_outcome_targets(infos.winner, dones, learner_players)
    steps_to_terminal = rollout_steps_to_next_done(dones)
    terminal_times = terminal_time_targets(infos.time, dones)
    finish_within_50 = ((outcome_targets == OUTCOME_WIN) & (outcome_known > 0.0) & (steps_to_terminal <= 50)).astype(
        jnp.float32
    )
    finish_within_100 = ((outcome_targets == OUTCOME_WIN) & (outcome_known > 0.0) & (steps_to_terminal <= 100)).astype(
        jnp.float32
    )
    finish_within_250 = ((outcome_targets == OUTCOME_WIN) & (outcome_known > 0.0) & (steps_to_terminal <= 250)).astype(
        jnp.float32
    )
    draw_risk = ((outcome_targets == OUTCOME_DRAW) & (outcome_known > 0.0)).astype(jnp.float32)
    learner_player_grid = jnp.broadcast_to(learner_players[None, :], dones.shape)

    def flat(array):
        return np.asarray(array.reshape(array.shape[0] * array.shape[1], *array.shape[2:]))

    logits = flat(teacher_logits)
    if logit_dtype == "float16":
        logits = np.clip(logits, -1.0e4, 1.0e4).astype(np.float16)

    def score16(array):
        return np.clip(flat(array), -1.0e4, 1.0e4).astype(np.float16)

    return {
        "obs": flat(obs).astype(np.float16),
        "legal_mask": flat(masks).astype(np.bool_),
        "active": flat(active).astype(np.bool_),
        "action": flat(actions).astype(np.int16),
        "teacher_action_index": flat(teacher_indices).astype(np.int32),
        "teacher_greedy_index": flat(teacher_greedy_indices).astype(np.int32),
        "teacher_logits": logits,
        "grid_size": flat(effective_sizes).astype(np.int16),
        "seat": flat(learner_player_grid).astype(np.int8),
        "time": flat(infos.time).astype(np.int16),
        "done": flat(dones).astype(np.bool_),
        "winner": flat(infos.winner).astype(np.int8),
        "outcome": flat(outcome_targets).astype(np.int8),
        "outcome_known": flat(outcome_known).astype(np.float16),
        "steps_to_terminal": flat(steps_to_terminal).astype(np.int16),
        "terminal_time": flat(terminal_times).astype(np.int16),
        "finish_within_50": flat(finish_within_50).astype(np.float16),
        "finish_within_100": flat(finish_within_100).astype(np.float16),
        "finish_within_250": flat(finish_within_250).astype(np.float16),
        "draw_risk": flat(draw_risk).astype(np.float16),
        "enemy_general_heatmap": flat(enemy_general).astype(np.float16),
        "enemy_owned_map": flat(enemy_owned).astype(np.float16),
        "hidden_enemy_owned_map": flat(hidden_enemy_owned).astype(np.float16),
        "hidden_enemy_army_map": flat(hidden_enemy_army).astype(np.float16),
        "city_map": flat(city_map).astype(np.float16),
        "source_heatmap": flat(source_heatmap).astype(np.float16),
        "target_heatmap": flat(target_heatmap).astype(np.float16),
        "intent": flat(intent).astype(np.int8),
        "visible_enemy_count": flat(visible_enemy_count).astype(np.int16),
        "visible_enemy_density": flat(visible_enemy_density).astype(np.float16),
        "contact": flat(contact).astype(np.float16),
        "search_candidate_indices": flat(search_candidate_indices).astype(np.int32),
        "search_prior_scores": score16(search_prior_scores),
        "search_scores": score16(search_scores),
        "search_outcomes": flat(search_outcomes).astype(np.int8),
        "search_best_position": flat(search_best_positions).astype(np.int8),
        "search_best_score": score16(search_best_scores),
        "search_mean_score": score16(search_mean_scores),
        "search_score_gap": score16(search_score_gaps),
        "search_best_outcome": flat(search_best_outcomes).astype(np.int8),
    }


def save_filter_config(args) -> dict[str, int | float | bool | None]:
    """Return the active sample filter settings for shard metadata."""
    return {
        "min_save_turn": args.min_save_turn,
        "max_save_turn": args.max_save_turn,
        "require_contact": args.require_contact,
        "min_visible_enemy_cells": args.min_visible_enemy_cells,
        "min_visible_enemy_density": args.min_visible_enemy_density,
        "require_outcome_known": args.require_outcome_known,
        "require_win": args.require_win,
        "require_finish_within_250": args.require_finish_within_250,
        "require_win_or_finish_within_250": args.require_win_or_finish_within_250,
        "draw_only": args.draw_only,
        "terminal_window": args.terminal_window,
        "min_search_score_gap": args.min_search_score_gap,
        "require_search_best_win": args.require_search_best_win,
    }


def active_save_filters(args) -> list[str]:
    """Return compact labels for non-default save filters."""
    labels: list[str] = []
    if args.min_save_turn > 0:
        labels.append(f"time>={args.min_save_turn}")
    if args.max_save_turn is not None:
        labels.append(f"time<={args.max_save_turn}")
    if args.require_contact:
        labels.append("contact")
    if args.min_visible_enemy_cells > 0:
        labels.append(f"visible_enemy_cells>={args.min_visible_enemy_cells}")
    if args.min_visible_enemy_density > 0.0:
        labels.append(f"visible_enemy_density>={args.min_visible_enemy_density:g}")
    if args.require_outcome_known:
        labels.append("outcome_known")
    if args.require_win:
        labels.append("win")
    if args.require_finish_within_250:
        labels.append("finish250")
    if args.require_win_or_finish_within_250:
        labels.append("win_or_finish250")
    if args.draw_only:
        labels.append("draw_only")
    if args.terminal_window > 0:
        labels.append(f"terminal_window<={args.terminal_window}")
    if args.min_search_score_gap > 0.0:
        labels.append(f"search_gap>={args.min_search_score_gap:g}")
    if args.require_search_best_win:
        labels.append("search_best_win")
    return labels


def apply_save_filters(arrays: dict[str, np.ndarray], args) -> tuple[dict[str, np.ndarray], dict]:
    """Filter flattened rollout samples before shard saving."""
    sample_count = int(arrays["obs"].shape[0])
    keep = np.ones((sample_count,), dtype=np.bool_)
    filter_stats: list[dict[str, int | str]] = []

    def add_filter(name: str, mask: np.ndarray) -> None:
        nonlocal keep
        bool_mask = np.asarray(mask, dtype=np.bool_)
        filter_stats.append({"name": name, "matches": int(np.sum(bool_mask))})
        keep &= bool_mask

    if args.min_save_turn > 0:
        add_filter(f"time>={args.min_save_turn}", arrays["time"] >= args.min_save_turn)
    if args.max_save_turn is not None:
        add_filter(f"time<={args.max_save_turn}", arrays["time"] <= args.max_save_turn)
    if args.require_contact:
        add_filter("contact", arrays["contact"] > 0.5)
    if args.min_visible_enemy_cells > 0:
        add_filter(
            f"visible_enemy_cells>={args.min_visible_enemy_cells}",
            arrays["visible_enemy_count"] >= args.min_visible_enemy_cells,
        )
    if args.min_visible_enemy_density > 0.0:
        add_filter(
            f"visible_enemy_density>={args.min_visible_enemy_density:g}",
            arrays["visible_enemy_density"] >= args.min_visible_enemy_density,
        )

    known = arrays["outcome_known"] > 0.0
    wins = (arrays["outcome"] == OUTCOME_WIN) & known
    if args.require_outcome_known:
        add_filter("outcome_known", known)
    if args.require_win:
        add_filter("win", wins)
    if args.require_finish_within_250:
        add_filter("finish250", arrays["finish_within_250"] > 0.5)
    if args.require_win_or_finish_within_250:
        add_filter("win_or_finish250", wins | (arrays["finish_within_250"] > 0.5))
    if args.draw_only:
        add_filter("draw_only", arrays["draw_risk"] > 0.5)
    if args.terminal_window > 0:
        add_filter(
            f"terminal_window<={args.terminal_window}",
            known & (arrays["steps_to_terminal"] <= args.terminal_window),
        )
    if args.min_search_score_gap > 0.0:
        add_filter(
            f"search_gap>={args.min_search_score_gap:g}",
            arrays["search_score_gap"].astype(np.float32) >= args.min_search_score_gap,
        )
    if args.require_search_best_win:
        add_filter("search_best_win", arrays["search_best_outcome"] == OUTCOME_WIN)

    filtered = {name: value[keep] for name, value in arrays.items()}
    post_count = int(np.sum(keep))
    stats = {
        "pre_filter_samples": sample_count,
        "post_filter_samples": post_count,
        "dropped_samples": int(sample_count - post_count),
        "filters": filter_stats,
    }
    return filtered, stats


def save_shard(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write one compressed NPZ shard and a small sidecar metadata JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect adaptive strategy-supervision dataset shards.")
    parser.add_argument("num_envs", nargs="?", type=int, default=64)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--truncation", type=int, default=750)
    parser.add_argument("--teacher-kind", choices=TEACHER_KINDS, default="adaptive")
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--teacher-network-arch", choices=("cnn", "unet"), default="unet")
    parser.add_argument("--teacher-channels", default=None)
    parser.add_argument("--teacher-input-channels", type=int, default=None)
    parser.add_argument("--teacher-global-context", action="store_true")
    parser.add_argument("--teacher-scoreboard-history", action="store_true")
    parser.add_argument("--teacher-value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--teacher-value-head-sizes", default=None)
    parser.add_argument("--teacher-value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--teacher-value-bins", type=int, default=128)
    parser.add_argument("--teacher-outcome-head", action="store_true")
    parser.add_argument("--teacher-strategy-aux", action="store_true")
    parser.add_argument("--teacher-strategy-spatial-aux", action="store_true")
    parser.add_argument("--teacher-strategy-finish-outputs", type=int, default=2)
    parser.add_argument("--fixed-teacher-model-path", default=None)
    parser.add_argument("--fixed-teacher-channels", default=None)
    parser.add_argument("--fixed-teacher-input-channels", type=int, default=9)
    parser.add_argument("--teacher-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--search-top-k", type=int, default=4)
    parser.add_argument("--search-rollout-steps", type=int, default=16)
    parser.add_argument("--search-rollouts-per-action", type=int, default=2)
    parser.add_argument("--search-army-weight", type=float, default=12.0)
    parser.add_argument("--search-land-weight", type=float, default=8.0)
    parser.add_argument("--search-prior-weight", type=float, default=0.01)
    parser.add_argument("--search-terminal-score", type=float, default=1000.0)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--fog-memory", action="store_true")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--output-dir", default="runs/adaptive-strategy-dataset")
    parser.add_argument("--shard-prefix", default="strategy")
    parser.add_argument("--logit-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--min-save-turn", type=int, default=0)
    parser.add_argument("--max-save-turn", type=int, default=None)
    parser.add_argument("--require-contact", action="store_true")
    parser.add_argument("--min-visible-enemy-cells", type=int, default=0)
    parser.add_argument("--min-visible-enemy-density", type=float, default=0.0)
    parser.add_argument("--require-outcome-known", action="store_true")
    parser.add_argument("--require-win", action="store_true")
    parser.add_argument("--require-finish-within-250", action="store_true")
    parser.add_argument("--require-win-or-finish-within-250", action="store_true")
    parser.add_argument("--draw-only", action="store_true")
    parser.add_argument("--terminal-window", type=int, default=0)
    parser.add_argument("--min-search-score-gap", type=float, default=0.0)
    parser.add_argument("--require-search-best-win", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
        args.grid_size_weights = parse_grid_size_weights(args.grid_size_weights, args.grid_sizes)
        args.teacher_value_head_sizes = (
            parse_grid_sizes(args.teacher_value_head_sizes)
            if args.teacher_value_head_sizes is not None
            else args.grid_sizes
        )
        args.teacher_channels = parse_policy_channels(args.teacher_channels)
        args.fixed_teacher_channels = parse_policy_channels(args.fixed_teacher_channels)
        args.opponent_channels = parse_policy_channels(args.opponent_channels)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_envs < 2:
        parser.error("num_envs must be at least 2 for mixed-seat collection")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_steps <= 0 or args.num_shards <= 0:
        parser.error("--num-steps and --num-shards must be positive")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
    if args.teacher_kind in ("adaptive", "search") and args.teacher_model_path is None:
        parser.error("--teacher-kind adaptive/search requires --teacher-model-path")
    if args.teacher_kind in ("fixed", "fixed-search") and args.fixed_teacher_model_path is None:
        parser.error(f"--teacher-kind {args.teacher_kind} requires --fixed-teacher-model-path")
    if args.teacher_kind in ("fixed", "fixed-search") and len(args.grid_sizes) != 1:
        parser.error(f"--teacher-kind {args.teacher_kind} requires exactly one --grid-sizes value")
    if args.opponent_policy_path is not None and len(args.grid_sizes) != 1:
        parser.error("--opponent-policy-path requires exactly one --grid-sizes value")
    if args.teacher_input_channels is not None and args.teacher_input_channels <= 0:
        parser.error("--teacher-input-channels must be positive")
    if args.search_top_k <= 0:
        parser.error("--search-top-k must be positive")
    if args.search_rollout_steps <= 0 or args.search_rollouts_per_action <= 0:
        parser.error("--search-rollout-steps and --search-rollouts-per-action must be positive")
    if args.search_army_weight < 0.0 or args.search_land_weight < 0.0:
        parser.error("--search-army-weight and --search-land-weight must be non-negative")
    if args.search_terminal_score <= 0.0:
        parser.error("--search-terminal-score must be positive")
    if args.teacher_value_loss == "hl-gauss" and args.teacher_value_bins <= 1:
        parser.error("--teacher-value-bins must be greater than 1 for --teacher-value-loss hl-gauss")
    if args.teacher_strategy_finish_outputs <= 0:
        parser.error("--teacher-strategy-finish-outputs must be positive")
    if args.teacher_strategy_spatial_aux and not args.teacher_strategy_aux:
        parser.error("--teacher-strategy-spatial-aux requires --teacher-strategy-aux")
    if args.fixed_teacher_input_channels <= 0 or args.opponent_input_channels <= 0:
        parser.error("policy input channel counts must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.min_save_turn < 0:
        parser.error("--min-save-turn must be non-negative")
    if args.max_save_turn is not None and args.max_save_turn < args.min_save_turn:
        parser.error("--max-save-turn must be >= --min-save-turn")
    if args.min_visible_enemy_cells < 0:
        parser.error("--min-visible-enemy-cells must be non-negative")
    if not (0.0 <= args.min_visible_enemy_density <= 1.0):
        parser.error("--min-visible-enemy-density must be in [0, 1]")
    if args.terminal_window < 0:
        parser.error("--terminal-window must be non-negative")
    if args.min_search_score_gap < 0.0:
        parser.error("--min-search-score-gap must be non-negative")
    if (args.min_search_score_gap > 0.0 or args.require_search_best_win) and args.teacher_kind not in (
        "search",
        "fixed-search",
    ):
        parser.error("search score filters require --teacher-kind search or fixed-search")
    if args.draw_only and (args.require_win or args.require_finish_within_250 or args.require_win_or_finish_within_250):
        parser.error("--draw-only conflicts with win/finish save filters")
    return args


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)
    key, pool_key, teacher_key = jrandom.split(key, 3)
    network_global_context = args.global_context or args.scoreboard_history
    teacher_global_context = args.teacher_global_context or args.teacher_scoreboard_history
    teacher_kind_id = TEACHER_KIND_TO_ID[args.teacher_kind]
    teacher_policy_mode_id = POLICY_MODE_NAME_TO_ID[args.teacher_policy_mode]
    opponent_policy_mode = 0 if args.opponent_policy_mode == "greedy" else 1
    fixed_teacher_grid_size = args.grid_sizes[0]
    opponent_policy_grid_size = args.grid_sizes[0]

    print("Adaptive strategy dataset collection")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Environments:  {args.num_envs} mixed seats")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Teacher:       {args.teacher_kind}")
    print(f"Teacher mode:  {args.teacher_policy_mode}")
    if args.teacher_kind == "search":
        print(
            "Search:       "
            f"top_k={args.search_top_k}, rollout_steps={args.search_rollout_steps}, "
            f"rollouts/action={args.search_rollouts_per_action}"
        )
    print(f"Rollouts:      {args.num_shards} shards x {args.num_steps} steps")
    print(f"Output:        {args.output_dir}")
    if args.scoreboard_history:
        print("Score history: enabled")
    elif network_global_context:
        print("Global ctx:    enabled")
    if args.fog_memory:
        print("Fog memory:    enabled")
    filters = active_save_filters(args)
    if filters:
        print(f"Save filters:  {', '.join(filters)}")
    print()

    teacher_network = None
    fixed_teacher_network = None
    teacher_input_channels = ADAPTIVE_INPUT_CHANNELS
    if args.teacher_kind in ("adaptive", "search"):
        teacher_input_channels = (
            args.teacher_input_channels
            if args.teacher_input_channels is not None
            else adaptive_input_channel_count(teacher_global_context, args.teacher_scoreboard_history, False)
        )
        teacher_value_bins = args.teacher_value_bins if args.teacher_value_loss == "hl-gauss" else 0
        teacher_network = load_or_create_adaptive_network(
            teacher_key,
            pad_size=args.pad_to,
            init_model_path=args.teacher_model_path,
            channels=args.teacher_channels,
            input_channels=teacher_input_channels,
            init_input_channels=teacher_input_channels,
            value_head_sizes=args.teacher_value_head_sizes if args.teacher_value_heads == "per-size" else (),
            init_value_head_sizes=args.teacher_value_head_sizes if args.teacher_value_heads == "per-size" else (),
            value_bins=teacher_value_bins,
            init_value_bins=teacher_value_bins,
            outcome_head=args.teacher_outcome_head,
            init_outcome_head=args.teacher_outcome_head,
            strategy_aux=args.teacher_strategy_aux,
            init_strategy_aux=args.teacher_strategy_aux,
            strategy_spatial_aux=args.teacher_strategy_spatial_aux,
            init_strategy_spatial_aux=args.teacher_strategy_spatial_aux,
            strategy_finish_outputs=args.teacher_strategy_finish_outputs,
            init_strategy_finish_outputs=args.teacher_strategy_finish_outputs,
            global_context=teacher_global_context,
            init_global_context=teacher_global_context,
            network_arch=args.teacher_network_arch,
            init_network_arch=args.teacher_network_arch,
        )
        teacher_input_channels = adaptive_network_input_channels(teacher_network)
    elif args.teacher_kind in ("fixed", "fixed-search"):
        fixed_teacher_network = PolicyValueNetwork(
            teacher_key,
            grid_size=fixed_teacher_grid_size,
            channels=args.fixed_teacher_channels,
            input_channels=args.fixed_teacher_input_channels,
        )
        fixed_teacher_network = eqx.tree_deserialise_leaves(args.fixed_teacher_model_path, fixed_teacher_network)

    opponent_policy_network = None
    if args.opponent_policy_path is not None:
        opponent_policy_network = PolicyValueNetwork(
            teacher_key,
            grid_size=opponent_policy_grid_size,
            channels=args.opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_policy_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_policy_network)

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
    jax.block_until_ready(pool.states.armies)
    p0_envs, p1_envs = split_mixed_env_counts(args.num_envs)
    states, effective_sizes = make_adaptive_initial_states(pool, args.num_envs)
    states_p0 = jax.tree.map(lambda value: value[:p0_envs], states)
    effective_sizes_p0 = effective_sizes[:p0_envs]
    states_p1 = jax.tree.map(lambda value: value[p0_envs:], states)
    effective_sizes_p1 = effective_sizes[p0_envs:]
    scoreboard_history_p0 = empty_scoreboard_history(p0_envs)
    scoreboard_history_p1 = empty_scoreboard_history(p1_envs)
    fog_memory_p0 = empty_adaptive_fog_memory(p0_envs, args.pad_to)
    fog_memory_p1 = empty_adaptive_fog_memory(p1_envs, args.pad_to)
    learner_players = jnp.concatenate(
        [
            jnp.zeros((p0_envs,), dtype=jnp.int32),
            jnp.ones((p1_envs,), dtype=jnp.int32),
        ]
    )
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    output_dir = Path(args.output_dir)
    metadata_base = {
        "grid_sizes": list(args.grid_sizes),
        "pad_to": args.pad_to,
        "teacher_kind": args.teacher_kind,
        "teacher_model_path": args.teacher_model_path,
        "teacher_network_arch": args.teacher_network_arch,
        "teacher_channels": args.teacher_channels,
        "teacher_input_channels": teacher_input_channels,
        "teacher_value_head_sizes": list(args.teacher_value_head_sizes)
        if args.teacher_value_heads == "per-size"
        else [],
        "teacher_value_loss": args.teacher_value_loss,
        "teacher_outcome_head": args.teacher_outcome_head,
        "teacher_strategy_aux": args.teacher_strategy_aux,
        "teacher_strategy_spatial_aux": args.teacher_strategy_spatial_aux,
        "teacher_strategy_finish_outputs": args.teacher_strategy_finish_outputs,
        "fixed_teacher_model_path": args.fixed_teacher_model_path,
        "teacher_policy_mode": args.teacher_policy_mode,
        "opponent": args.opponent,
        "opponent_policy_path": args.opponent_policy_path,
        "opponent_policy_mode": args.opponent_policy_mode,
        "search_top_k": args.search_top_k,
        "search_rollout_steps": args.search_rollout_steps,
        "search_rollouts_per_action": args.search_rollouts_per_action,
        "search_army_weight": args.search_army_weight,
        "search_land_weight": args.search_land_weight,
        "search_prior_weight": args.search_prior_weight,
        "search_terminal_score": args.search_terminal_score,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "truncation": args.truncation,
        "scoreboard_history": args.scoreboard_history,
        "fog_memory": args.fog_memory,
        "logit_dtype": args.logit_dtype,
        "save_filters": save_filter_config(args),
        "seed": args.seed,
    }

    for shard_index in range(args.num_shards):
        t0 = time.time()
        key, rollout_key = jrandom.split(key)
        (
            states_p0,
            effective_sizes_p0,
            scoreboard_history_p0,
            fog_memory_p0,
            states_p1,
            effective_sizes_p1,
            scoreboard_history_p1,
            fog_memory_p1,
            rollout_data,
            key,
        ) = collect_mixed_strategy_rollout(
            states_p0,
            effective_sizes_p0,
            states_p1,
            effective_sizes_p1,
            pool,
            teacher_network,
            fixed_teacher_network,
            rollout_key,
            args.num_steps,
            args.truncation,
            opponent_id,
            teacher_kind_id,
            teacher_policy_mode_id,
            args.pad_to,
            network_global_context,
            scoreboard_history_p0,
            scoreboard_history_p1,
            args.scoreboard_history,
            fog_memory_p0,
            fog_memory_p1,
            args.fog_memory,
            teacher_input_channels,
            fixed_teacher_grid_size,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
            args.search_top_k,
            args.search_rollout_steps,
            args.search_rollouts_per_action,
            args.search_army_weight,
            args.search_land_weight,
            args.search_prior_weight,
            args.search_terminal_score,
        )
        jax.block_until_ready(states_p0)
        arrays = flatten_rollout_data(rollout_data, learner_players, args.logit_dtype)
        dones = rollout_data[-2]
        infos = rollout_data[-1]
        episodes = int(jnp.sum(dones))
        wins = int(jnp.sum(dones & (infos.winner == learner_players[None, :])))
        draws = int(jnp.sum(dones & (infos.winner < 0)))
        pre_filter_samples = int(arrays["obs"].shape[0])
        arrays, filter_stats = apply_save_filters(arrays, args)
        sample_count = int(arrays["obs"].shape[0])
        if sample_count <= 0:
            print(
                f"Shard {shard_index:04d} | samples=0/{pre_filter_samples} after filters | "
                f"episodes={episodes} wins={wins} draws={draws} | skipped | time={time.time() - t0:.2f}s"
            )
            continue

        shard_path = output_dir / f"{args.shard_prefix}-{shard_index:05d}.npz"
        metadata = dict(
            metadata_base,
            shard_index=shard_index,
            num_samples=sample_count,
            **filter_stats,
        )
        save_shard(shard_path, arrays, metadata)

        sample_known = arrays["outcome_known"] > 0.0
        sample_wins = int(np.sum((arrays["outcome"] == OUTCOME_WIN) & sample_known))
        sample_draws = int(np.sum(arrays["draw_risk"] > 0.5))
        sample_contact = float(np.mean(arrays["contact"] > 0.5))
        print(
            f"Shard {shard_index:04d} | samples={sample_count}/{pre_filter_samples} | "
            f"episodes={episodes} wins={wins} draws={draws} | "
            f"sample_wins={sample_wins} sample_draws={sample_draws} contact={sample_contact:.3f} | "
            f"path={shard_path} | time={time.time() - t0:.2f}s"
        )


if __name__ == "__main__":
    main()
