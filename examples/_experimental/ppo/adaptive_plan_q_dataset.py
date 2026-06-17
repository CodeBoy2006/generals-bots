"""Collect source-target plan-Q shards for adaptive strategy supervision.

This is the next step after source/target CE heads: score candidate plans by
short counterfactual rollouts, then save plan-level Q targets and source/target
marginals. The script intentionally writes only ignored `runs/` artifacts.
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
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    AdaptiveFogMemory,
    active_cells_for_size,
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
from adaptive_network import adaptive_network_input_channels, load_or_create_adaptive_network
from adaptive_search_distill import adaptive_score_observation, outcome_class_from_winner
from adaptive_teacher_imitation import policy_action_from_logits
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, POLICY_MODE_NAMES, opponent_action, policy_network_action
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from generals.core import game
from generals.core.action import DIRECTIONS
from train import random_action, stack_learner_actions
from train_adaptive import crop_observation, split_mixed_env_counts

POLICY_MODE_NAME_TO_ID = {name: index for index, name in enumerate(POLICY_MODE_NAMES)}


def empty_scoreboard_history(num_envs: int) -> jnp.ndarray:
    """Return empty previous-scoreboard features for vectorized collection."""
    return jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)


def reset_single_fog_memory(memory: AdaptiveFogMemory, done: jnp.ndarray) -> AdaptiveFogMemory:
    """Clear one unbatched fog-memory row after a terminal counterfactual rollout."""
    keep = (~done).astype(jnp.float32)
    return jax.tree.map(lambda value: value * keep, memory)


def topk_indices(score_map: jnp.ndarray, count: int) -> jnp.ndarray:
    """Return flat top-k cell indices from a spatial score map."""
    _, indices = jax.lax.top_k(score_map.reshape(-1), count)
    return indices.astype(jnp.int32)


def candidate_source_indices(state, learner_player: int, effective_size: int, pad_size: int, count: int) -> jnp.ndarray:
    """Pick source candidates from owned movable cells, ranked by army mass."""
    active = active_cells_for_size(effective_size, pad_size)
    own = state.ownership[learner_player] & active
    movable = own & (state.armies > 1)
    source_scores = jnp.where(movable, jnp.log1p(state.armies.astype(jnp.float32)), -1.0e9)
    return topk_indices(source_scores, count)


def candidate_target_indices(state, learner_player: int, effective_size: int, pad_size: int, count: int) -> jnp.ndarray:
    """Pick target candidates from privileged tactical and strategic targets."""
    active = active_cells_for_size(effective_size, pad_size)
    opponent = 1 - learner_player
    enemy = state.ownership[opponent] & active
    own = state.ownership[learner_player] & active
    neutral_or_enemy_city = state.cities & ~own & active
    enemy_general_row, enemy_general_col = state.general_positions[opponent]
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    enemy_general = (rows == enemy_general_row) & (cols == enemy_general_col) & active
    # General must always dominate, but other high-value visible/full-state targets
    # keep the candidate set from degenerating into one static label.
    target_scores = jnp.where(active & state.passable, 0.01, -1.0e9)
    target_scores = target_scores + enemy.astype(jnp.float32) * (20.0 + jnp.log1p(state.armies.astype(jnp.float32)))
    target_scores = target_scores + neutral_or_enemy_city.astype(jnp.float32) * 40.0
    target_scores = target_scores + enemy_general.astype(jnp.float32) * 1000.0
    return topk_indices(target_scores, count)


def plan_action_from_source_target(
    state,
    learner_player: int,
    source_index: jnp.ndarray,
    target_index: jnp.ndarray,
    pad_size: int,
) -> jnp.ndarray:
    """Return one primitive first step that moves source toward target when legal."""
    source_row = source_index // pad_size
    source_col = source_index % pad_size
    target_row = target_index // pad_size
    target_col = target_index % pad_size
    dest_rows = source_row + DIRECTIONS[:, 0]
    dest_cols = source_col + DIRECTIONS[:, 1]
    in_bounds = (dest_rows >= 0) & (dest_rows < pad_size) & (dest_cols >= 0) & (dest_cols < pad_size)
    safe_rows = jnp.clip(dest_rows, 0, pad_size - 1)
    safe_cols = jnp.clip(dest_cols, 0, pad_size - 1)
    owns_source = state.ownership[learner_player, source_row, source_col]
    movable = owns_source & (state.armies[source_row, source_col] > 1)
    passable = state.passable[safe_rows, safe_cols]
    current_distance = jnp.abs(source_row - target_row) + jnp.abs(source_col - target_col)
    next_distance = jnp.abs(dest_rows - target_row) + jnp.abs(dest_cols - target_col)
    progress = current_distance - next_distance
    direction_scores = jnp.where(in_bounds & passable, progress.astype(jnp.float32), -1.0e9)
    direction = jnp.argmax(direction_scores).astype(jnp.int32)
    has_direction = jnp.max(direction_scores) > -1.0e8
    should_pass = ~(movable & has_direction)
    return jnp.array(
        [
            should_pass.astype(jnp.int32),
            source_row.astype(jnp.int32),
            source_col.astype(jnp.int32),
            direction,
            jnp.int32(0),
        ],
        dtype=jnp.int32,
    )


def adaptive_policy_action_with_memory(
    network,
    obs,
    effective_size: int,
    key,
    policy_mode: int,
    pad_size: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory: AdaptiveFogMemory,
):
    """Dispatch an adaptive action and return updated one-player context."""
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
    logits, _ = network.logits_value(obs_arr, mask, active)
    action = policy_action_from_logits(logits, key, policy_mode, pad_size)
    return action, current_scoreboard, current_memory


def opponent_policy_action(
    fixed_opponent_network,
    opponent_id: int,
    key,
    obs,
    opponent_policy_mode: int,
    opponent_policy_grid_size: int,
):
    """Dispatch either a fixed policy opponent or a heuristic opponent."""
    if fixed_opponent_network is not None:
        return policy_network_action(
            fixed_opponent_network,
            key,
            crop_observation(obs, opponent_policy_grid_size),
            opponent_policy_mode,
        )
    return opponent_action(opponent_id, key, obs, random_action)


@eqx.filter_jit
def score_plan_candidates(
    network,
    fixed_opponent_network,
    state,
    effective_size,
    key,
    learner_player,
    source_count: int,
    target_count: int,
    rollout_steps: int,
    rollouts_per_plan: int,
    policy_mode: int,
    opponent_id: int,
    opponent_policy_mode: int,
    opponent_policy_grid_size: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
    score_scale: float,
    score_temperature: float,
    pad_size: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory: AdaptiveFogMemory,
):
    """Score source-target plans by forcing the first move and rolling out the base policy."""
    source_indices = candidate_source_indices(state, learner_player, effective_size, pad_size, source_count)
    target_indices = candidate_target_indices(state, learner_player, effective_size, pad_size, target_count)
    opponent_player = 1 - learner_player
    opponent_obs = game.get_observation(state, opponent_player)
    key, opponent_key = jrandom.split(key)
    opponent_first_action = opponent_policy_action(
        fixed_opponent_network,
        opponent_id,
        opponent_key,
        opponent_obs,
        opponent_policy_mode,
        opponent_policy_grid_size,
    )

    def rollout_result(initial_state, rollout_key, initial_scoreboard, initial_memory):
        def body(carry, _):
            rollout_state, previous_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, learner_player)
            learner_action, current_scoreboard, current_memory = adaptive_policy_action_with_memory(
                network,
                learner_obs,
                effective_size,
                learner_key,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                previous_scoreboard,
                fog_memory_enabled,
                memory,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = opponent_policy_action(
                fixed_opponent_network,
                opponent_id,
                opponent_key,
                opponent_obs,
                opponent_policy_mode,
                opponent_policy_grid_size,
            )
            actions = jax.lax.cond(
                learner_player == 0,
                lambda _: jnp.stack([learner_action, opponent_action_value]),
                lambda _: jnp.stack([opponent_action_value, learner_action]),
                None,
            )
            next_state, _ = game.step(rollout_state, actions)
            already_done = game.get_info(rollout_state).is_done
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            final_info = game.get_info(final_state)
            next_scoreboard = reset_adaptive_scoreboard_history(current_scoreboard, final_info.is_done)
            next_memory = reset_single_fog_memory(current_memory, final_info.is_done)
            return (final_state, next_scoreboard, next_memory, step_key), None

        (final_state, _, _, _), _ = jax.lax.scan(
            body,
            (initial_state, initial_scoreboard, initial_memory, rollout_key),
            None,
            length=rollout_steps,
        )
        final_info = game.get_info(final_state)
        final_obs = game.get_observation(final_state, learner_player)
        score = adaptive_score_observation(final_info, final_obs, learner_player, army_weight, land_weight, terminal_score)
        outcome = outcome_class_from_winner(jnp.where(final_info.is_done, final_info.winner, -1), learner_player)
        return score, outcome

    def plan_score(source_index, target_index, plan_key):
        plan_action = plan_action_from_source_target(state, learner_player, source_index, target_index, pad_size)
        first_actions = jax.lax.cond(
            learner_player == 0,
            lambda _: jnp.stack([plan_action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, plan_action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(plan_key, rollouts_per_plan)
        scores, outcomes = jax.vmap(
            lambda rollout_key: rollout_result(next_state, rollout_key, previous_scoreboard, fog_memory)
        )(rollout_keys)
        first_outcome = outcome_class_from_winner(first_info.winner, learner_player)
        first_terminal = jnp.where(
            first_info.winner == learner_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        best_rollout = jnp.argmax(scores)
        score = first_terminal + jnp.mean(scores)
        outcome = jnp.where(first_info.is_done, first_outcome, outcomes[best_rollout])
        action_index = plan_action_to_index(plan_action, pad_size)
        return score, outcome, action_index

    del prior_weight  # Reserved for later variants that mix policy prior into plan scores.
    plan_keys = jrandom.split(key, source_count * target_count).reshape(source_count, target_count, 2)
    scores, outcomes, action_indices = jax.vmap(
        lambda source_index, row_keys: jax.vmap(
            lambda target_index, plan_key: plan_score(source_index, target_index, plan_key)
        )(target_indices, row_keys)
    )(source_indices, plan_keys)
    plan_q = jnp.tanh(scores / score_scale)
    plan_probs = jax.nn.softmax((plan_q / score_temperature).reshape(-1), axis=-1).reshape(source_count, target_count)
    source_probs = jnp.sum(plan_probs, axis=1)
    target_probs = jnp.sum(plan_probs, axis=0)
    best_flat = jnp.argmax(plan_q.reshape(-1))
    best_source_pos = (best_flat // target_count).astype(jnp.int32)
    best_target_pos = (best_flat % target_count).astype(jnp.int32)
    return (
        source_indices,
        target_indices,
        scores,
        plan_q,
        outcomes,
        action_indices,
        source_probs,
        target_probs,
        best_source_pos,
        best_target_pos,
    )


def plan_action_to_index(action: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Encode a plan first action into adaptive policy-index layout."""
    is_pass, row, col, direction, is_half = action
    encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
    return (encoded_dir * pad_size * pad_size + row * pad_size + col).astype(jnp.int32)


@eqx.filter_jit
def collect_plan_q_step(
    states,
    effective_sizes,
    pool,
    network,
    fixed_opponent_network,
    key,
    truncation: int,
    opponent_id: int,
    learner_player: int,
    source_count: int,
    target_count: int,
    rollout_steps: int,
    rollouts_per_plan: int,
    policy_mode: int,
    opponent_policy_mode: int,
    opponent_policy_grid_size: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
    score_scale: float,
    score_temperature: float,
    pad_size: int,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
):
    """Collect one vectorized batch of plan-Q labels and advance behavior states."""
    num_envs = states.armies.shape[0]
    if scoreboard_history is None:
        scoreboard_history = empty_scoreboard_history(num_envs)
    if fog_memory is None:
        fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)

    obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
    obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
    learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
    opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
    current_fog_memory = (
        jax.vmap(update_adaptive_fog_memory)(fog_memory, learner_obs) if fog_memory_enabled else fog_memory
    )
    current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
        learner_obs,
        effective_sizes,
    )
    if scoreboard_history_enabled:
        history_context = jax.vmap(adaptive_scoreboard_history_context)(scoreboard_history, current_scoreboard)
        obs_arr, active = jax.vmap(
            lambda obs, size, history, memory: adaptive_obs_to_array(
                obs,
                size,
                pad_size,
                include_global_context=True,
                scoreboard_history=history,
                fog_memory=memory if fog_memory_enabled else None,
            )
        )(learner_obs, effective_sizes, history_context, current_fog_memory)
    else:
        obs_arr, active = jax.vmap(
            lambda obs, size, memory: adaptive_obs_to_array(
                obs,
                size,
                pad_size,
                include_global_context=global_context,
                fog_memory=memory if fog_memory_enabled else None,
            )
        )(learner_obs, effective_sizes, current_fog_memory)
    masks = jax.vmap(
        lambda obs, size: compute_adaptive_valid_move_mask(
            obs.armies,
            obs.owned_cells,
            obs.mountains,
            size,
            pad_size,
        )
    )(learner_obs, effective_sizes)
    logits = jax.vmap(lambda obs, mask, active_mask: network.logits_value(obs, mask, active_mask)[0])(
        obs_arr,
        masks,
        active,
    )

    key, plan_key, policy_key, opponent_key = jrandom.split(key, 4)
    plan_keys = jrandom.split(plan_key, num_envs)
    plan_outputs = jax.vmap(
        lambda state, size, sample_key, prev_scoreboard, memory: score_plan_candidates(
            network,
            fixed_opponent_network,
            state,
            size,
            sample_key,
            learner_player,
            source_count,
            target_count,
            rollout_steps,
            rollouts_per_plan,
            policy_mode,
            opponent_id,
            opponent_policy_mode,
            opponent_policy_grid_size,
            army_weight,
            land_weight,
            prior_weight,
            terminal_score,
            score_scale,
            score_temperature,
            pad_size,
            global_context,
            scoreboard_history_enabled,
            prev_scoreboard,
            fog_memory_enabled,
            memory,
        )
    )(states, effective_sizes, plan_keys, current_scoreboard, current_fog_memory)

    policy_keys = jrandom.split(policy_key, num_envs)
    learner_actions = jax.vmap(
        lambda sample_logits, sample_key: policy_action_from_logits(sample_logits, sample_key, policy_mode, pad_size)
    )(logits, policy_keys)
    learner_action_indices = jax.vmap(lambda action: plan_action_to_index(action, pad_size))(learner_actions)
    opponent_keys = jrandom.split(opponent_key, num_envs)
    opponent_actions = jax.vmap(
        lambda sample_key, obs: opponent_policy_action(
            fixed_opponent_network,
            opponent_id,
            sample_key,
            obs,
            opponent_policy_mode,
            opponent_policy_grid_size,
        )
    )(opponent_keys, opponent_obs)
    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
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
    final_scoreboard = reset_adaptive_scoreboard_history(current_scoreboard, dones)
    final_memory = reset_adaptive_fog_memory(current_fog_memory, dones)
    data = (
        obs_arr,
        masks,
        active,
        logits,
        learner_action_indices,
        effective_sizes,
        plan_outputs,
        dones,
        infos,
    )
    return final_states, final_sizes, final_scoreboard, final_memory, data, key


def collect_plan_q_rollout(
    states,
    effective_sizes,
    pool,
    network,
    fixed_opponent_network,
    key,
    num_steps,
    truncation,
    opponent_id,
    learner_player,
    source_count,
    target_count,
    rollout_steps,
    rollouts_per_plan,
    policy_mode,
    opponent_policy_mode,
    opponent_policy_grid_size,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    score_scale,
    score_temperature,
    pad_size,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
):
    """Collect multiple plan-Q steps for one learner seat."""
    step_data = []
    for _ in range(num_steps):
        states, effective_sizes, scoreboard_history, fog_memory, data, key = collect_plan_q_step(
            states,
            effective_sizes,
            pool,
            network,
            fixed_opponent_network,
            key,
            truncation,
            opponent_id,
            learner_player,
            source_count,
            target_count,
            rollout_steps,
            rollouts_per_plan,
            policy_mode,
            opponent_policy_mode,
            opponent_policy_grid_size,
            army_weight,
            land_weight,
            prior_weight,
            terminal_score,
            score_scale,
            score_temperature,
            pad_size,
            global_context,
            scoreboard_history,
            scoreboard_history_enabled,
            fog_memory,
            fog_memory_enabled,
        )
        step_data.append(data)
    return states, effective_sizes, scoreboard_history, fog_memory, jax.tree.map(lambda *xs: jnp.stack(xs), *step_data), key


def collect_mixed_plan_q_rollout(
    states_p0,
    effective_sizes_p0,
    states_p1,
    effective_sizes_p1,
    pool,
    network,
    fixed_opponent_network,
    key,
    num_steps,
    truncation,
    opponent_id,
    source_count,
    target_count,
    rollout_steps,
    rollouts_per_plan,
    policy_mode,
    opponent_policy_mode,
    opponent_policy_grid_size,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    score_scale,
    score_temperature,
    pad_size,
    global_context=False,
    scoreboard_history_p0=None,
    scoreboard_history_p1=None,
    scoreboard_history_enabled=False,
    fog_memory_p0=None,
    fog_memory_p1=None,
    fog_memory_enabled=False,
):
    """Collect plan-Q data for both seats and concatenate env dimension."""
    key, p0_key, p1_key = jrandom.split(key, 3)
    states_p0, effective_sizes_p0, scoreboard_history_p0, fog_memory_p0, rollout_p0, _ = collect_plan_q_rollout(
        states_p0,
        effective_sizes_p0,
        pool,
        network,
        fixed_opponent_network,
        p0_key,
        num_steps,
        truncation,
        opponent_id,
        0,
        source_count,
        target_count,
        rollout_steps,
        rollouts_per_plan,
        policy_mode,
        opponent_policy_mode,
        opponent_policy_grid_size,
        army_weight,
        land_weight,
        prior_weight,
        terminal_score,
        score_scale,
        score_temperature,
        pad_size,
        global_context,
        scoreboard_history_p0,
        scoreboard_history_enabled,
        fog_memory_p0,
        fog_memory_enabled,
    )
    states_p1, effective_sizes_p1, scoreboard_history_p1, fog_memory_p1, rollout_p1, _ = collect_plan_q_rollout(
        states_p1,
        effective_sizes_p1,
        pool,
        network,
        fixed_opponent_network,
        p1_key,
        num_steps,
        truncation,
        opponent_id,
        1,
        source_count,
        target_count,
        rollout_steps,
        rollouts_per_plan,
        policy_mode,
        opponent_policy_mode,
        opponent_policy_grid_size,
        army_weight,
        land_weight,
        prior_weight,
        terminal_score,
        score_scale,
        score_temperature,
        pad_size,
        global_context,
        scoreboard_history_p1,
        scoreboard_history_enabled,
        fog_memory_p1,
        fog_memory_enabled,
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


def flatten_plan_q_data(rollout_data, learner_players: jnp.ndarray, logit_dtype: str) -> dict[str, np.ndarray]:
    """Flatten time/env axes and prepare shard arrays."""
    (
        obs,
        masks,
        active,
        logits,
        learner_action_indices,
        effective_sizes,
        plan_outputs,
        dones,
        infos,
    ) = rollout_data
    (
        source_indices,
        target_indices,
        plan_scores,
        plan_q,
        plan_outcomes,
        plan_action_indices,
        source_probs,
        target_probs,
        best_source_pos,
        best_target_pos,
    ) = plan_outputs
    learner_player_grid = jnp.broadcast_to(learner_players[None, :], dones.shape)
    best_plan_q = jnp.take_along_axis(
        plan_q.reshape(*plan_q.shape[:2], -1),
        (best_source_pos * plan_q.shape[-1] + best_target_pos)[..., None],
        axis=-1,
    )[..., 0]
    mean_plan_q = jnp.mean(plan_q, axis=(-2, -1))

    def flat(array):
        return np.asarray(array.reshape(array.shape[0] * array.shape[1], *array.shape[2:]))

    flat_logits = flat(logits)
    if logit_dtype == "float16":
        flat_logits = np.clip(flat_logits, -1.0e4, 1.0e4).astype(np.float16)
    return {
        "obs": flat(obs).astype(np.float16),
        "legal_mask": flat(masks).astype(np.bool_),
        "active": flat(active).astype(np.bool_),
        "teacher_logits": flat_logits,
        "teacher_action_index": flat(learner_action_indices).astype(np.int32),
        "grid_size": flat(effective_sizes).astype(np.int16),
        "seat": flat(learner_player_grid).astype(np.int8),
        "done": flat(dones).astype(np.bool_),
        "winner": flat(infos.winner).astype(np.int8),
        "time": flat(infos.time).astype(np.int16),
        "source_indices": flat(source_indices).astype(np.int16),
        "target_indices": flat(target_indices).astype(np.int16),
        "plan_action_indices": flat(plan_action_indices).astype(np.int32),
        "plan_scores": flat(plan_scores).astype(np.float16),
        "plan_q": flat(plan_q).astype(np.float16),
        "plan_outcomes": flat(plan_outcomes).astype(np.int8),
        "source_score_probs": flat(source_probs).astype(np.float16),
        "target_score_probs": flat(target_probs).astype(np.float16),
        "best_source_pos": flat(best_source_pos).astype(np.int8),
        "best_target_pos": flat(best_target_pos).astype(np.int8),
        "best_plan_q": flat(best_plan_q).astype(np.float16),
        "mean_plan_q": flat(mean_plan_q).astype(np.float16),
        "plan_q_gap": flat(best_plan_q - mean_plan_q).astype(np.float16),
    }


def save_shard(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write one compressed NPZ shard and sidecar metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect adaptive source-target Plan-Q dataset shards.")
    parser.add_argument("num_envs", nargs="?", type=int, default=16)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=1024)
    parser.add_argument("--truncation", type=int, default=750)
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
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=4)
    parser.add_argument("--plan-rollout-steps", type=int, default=16)
    parser.add_argument("--rollouts-per-plan", type=int, default=2)
    parser.add_argument("--score-scale", type=float, default=10.0)
    parser.add_argument("--score-temperature", type=float, default=0.25)
    parser.add_argument("--army-weight", type=float, default=12.0)
    parser.add_argument("--land-weight", type=float, default=8.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--terminal-score", type=float, default=1000.0)
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
    parser.add_argument("--output-dir", default="runs/adaptive-plan-q-dataset")
    parser.add_argument("--shard-prefix", default="plan-q")
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
    if args.num_envs < 2:
        parser.error("num_envs must be at least 2 for mixed-seat collection")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_steps <= 0 or args.num_shards <= 0:
        parser.error("--num-steps and --num-shards must be positive")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.value_loss == "hl-gauss" and args.value_bins <= 1:
        parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
    if args.source_count <= 0 or args.target_count <= 0:
        parser.error("--source-count and --target-count must be positive")
    if args.plan_rollout_steps <= 0 or args.rollouts_per_plan <= 0:
        parser.error("--plan-rollout-steps and --rollouts-per-plan must be positive")
    if args.score_scale <= 0.0 or args.score_temperature <= 0.0:
        parser.error("--score-scale and --score-temperature must be positive")
    if args.opponent_policy_path is not None and len(args.grid_sizes) != 1:
        parser.error("--opponent-policy-path requires exactly one --grid-sizes value")
    if args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    return args


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)
    key, pool_key, net_key = jrandom.split(key, 3)
    network_global_context = args.global_context or args.scoreboard_history
    input_channels = (
        args.input_channels
        if args.input_channels is not None
        else adaptive_input_channel_count(network_global_context, args.scoreboard_history, args.fog_memory)
    )
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    opponent_policy_grid_size = args.grid_sizes[0]

    print("Adaptive Plan-Q dataset collection")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Environments:  {args.num_envs} mixed seats")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Model:         {args.model_path}")
    print(f"Plans/state:   {args.source_count}x{args.target_count}")
    print(f"Plan rollout:  {args.plan_rollout_steps} steps x {args.rollouts_per_plan}")
    print(f"Output:        {args.output_dir}")
    if args.scoreboard_history:
        print("Score history: enabled")
    elif network_global_context:
        print("Global ctx:    enabled")
    if args.fog_memory:
        print("Fog memory:    enabled")
    print()

    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.model_path,
        channels=args.channels,
        input_channels=input_channels,
        init_input_channels=input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        init_value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        value_bins=value_bins,
        init_value_bins=value_bins,
        outcome_head=args.outcome_head,
        init_outcome_head=args.outcome_head,
        strategy_aux=args.strategy_aux,
        init_strategy_aux=args.strategy_aux,
        strategy_spatial_aux=args.strategy_spatial_aux,
        init_strategy_spatial_aux=args.strategy_spatial_aux,
        global_context=network_global_context,
        init_global_context=network_global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
    )
    input_channels = adaptive_network_input_channels(network)

    fixed_opponent_network = None
    if args.opponent_policy_path is not None:
        fixed_opponent_network = PolicyValueNetwork(
            net_key,
            grid_size=opponent_policy_grid_size,
            channels=args.opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        fixed_opponent_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, fixed_opponent_network)
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
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
    states_p1 = jax.tree.map(lambda value: value[p0_envs:], states)
    effective_sizes_p0 = effective_sizes[:p0_envs]
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
    metadata_base = {
        "grid_sizes": list(args.grid_sizes),
        "pad_to": args.pad_to,
        "model_path": args.model_path,
        "network_arch": args.network_arch,
        "channels": args.channels,
        "input_channels": input_channels,
        "value_head_sizes": list(args.value_head_sizes) if args.value_heads == "per-size" else [],
        "value_loss": args.value_loss,
        "outcome_head": args.outcome_head,
        "strategy_aux": args.strategy_aux,
        "strategy_spatial_aux": args.strategy_spatial_aux,
        "policy_mode": args.policy_mode,
        "source_count": args.source_count,
        "target_count": args.target_count,
        "plan_rollout_steps": args.plan_rollout_steps,
        "rollouts_per_plan": args.rollouts_per_plan,
        "opponent": args.opponent,
        "opponent_policy_path": args.opponent_policy_path,
        "opponent_policy_mode": args.opponent_policy_mode,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "truncation": args.truncation,
        "scoreboard_history": args.scoreboard_history,
        "fog_memory": args.fog_memory,
        "score_scale": args.score_scale,
        "score_temperature": args.score_temperature,
        "seed": args.seed,
    }

    output_dir = Path(args.output_dir)
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
        ) = collect_mixed_plan_q_rollout(
            states_p0,
            effective_sizes_p0,
            states_p1,
            effective_sizes_p1,
            pool,
            network,
            fixed_opponent_network,
            rollout_key,
            args.num_steps,
            args.truncation,
            opponent_id,
            args.source_count,
            args.target_count,
            args.plan_rollout_steps,
            args.rollouts_per_plan,
            policy_mode,
            opponent_policy_mode,
            opponent_policy_grid_size,
            args.army_weight,
            args.land_weight,
            args.prior_weight,
            args.terminal_score,
            args.score_scale,
            args.score_temperature,
            args.pad_to,
            network_global_context,
            scoreboard_history_p0,
            scoreboard_history_p1,
            args.scoreboard_history,
            fog_memory_p0,
            fog_memory_p1,
            args.fog_memory,
        )
        jax.block_until_ready(states_p0.armies)
        arrays = flatten_plan_q_data(rollout_data, learner_players, args.logit_dtype)
        shard_path = output_dir / f"{args.shard_prefix}-{shard_index:05d}.npz"
        metadata = dict(metadata_base, shard_index=shard_index, num_samples=int(arrays["obs"].shape[0]))
        save_shard(shard_path, arrays, metadata)

        best_outcomes = arrays["plan_outcomes"].reshape(arrays["plan_outcomes"].shape[0], -1)
        best_flat = arrays["best_source_pos"].astype(np.int32) * args.target_count + arrays["best_target_pos"].astype(np.int32)
        best_plan_outcomes = best_outcomes[np.arange(best_outcomes.shape[0]), best_flat]
        print(
            f"Shard {shard_index:04d} | samples={arrays['obs'].shape[0]} | "
            f"mean_gap={float(np.mean(arrays['plan_q_gap'])):.4f} | "
            f"best_q={float(np.mean(arrays['best_plan_q'])):.4f} | "
            f"best_win={float(np.mean(best_plan_outcomes == 2)):.3f} | "
            f"best_draw={float(np.mean(best_plan_outcomes == 1)):.3f} | "
            f"path={shard_path} | time={time.time() - t0:.2f}s"
        )


if __name__ == "__main__":
    main()
