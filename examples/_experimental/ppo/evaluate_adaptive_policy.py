"""Evaluate adaptive multisize PPO checkpoints against heuristic opponents."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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

from adaptive_command_gate import CommandGateNetwork
from adaptive_common import (
    ADAPTIVE_GLOBAL_INPUT_CHANNELS,
    ADAPTIVE_HISTORY_INPUT_CHANNELS,
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_action_to_index,
    adaptive_input_channel_count,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    empty_adaptive_fog_memory,
    make_adaptive_state_pool,
    parse_grid_sizes,
    update_adaptive_fog_memory,
)
from adaptive_network import load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, POLICY_MODE_NAMES, opponent_action, policy_network_action
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from generals.core.action import DIRECTIONS
from generals.core import game
from train import random_action, stack_learner_actions


@dataclass(frozen=True)
class AdaptiveEvalRow:
    grid_size: int
    policy_player: int
    wins: int
    losses: int
    draws: int
    num_games: int
    mean_time: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.num_games

    @property
    def decisive_win_rate(self) -> float:
        return self.wins / max(self.wins + self.losses, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / self.num_games

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["win_rate"] = self.win_rate
        data["decisive_win_rate"] = self.decisive_win_rate
        data["draw_rate"] = self.draw_rate
        return data


@eqx.filter_jit
def _policy_action(
    network,
    plan_worker_network,
    command_gate_network,
    obs_arr,
    mask,
    active,
    key,
    policy_mode,
    policy_player: int,
    strategy_q_rerank_scale: float,
    strategy_q_replace_threshold: float,
    strategy_q_replace_policy_margin: float,
    strategy_q_replace_worker_candidate: bool,
    strategy_target_rerank_scale: float,
    strategy_target_finish_gate: bool,
    strategy_spatial_rerank_scale: float,
    strategy_worker_mix_prob: float,
    strategy_worker_finish_gate: bool,
    strategy_worker_policy_margin: float,
    strategy_plan_worker_rerank_scale: float,
    strategy_plan_worker_min_margin: float,
    strategy_command_gate_threshold: float,
):
    logits, _ = network.logits_value(obs_arr, mask, active)
    needs_aux = (
        strategy_q_rerank_scale > 0.0
        or strategy_q_replace_threshold >= 0.0
        or strategy_target_rerank_scale > 0.0
        or strategy_spatial_rerank_scale > 0.0
        or strategy_worker_mix_prob > 0.0
        or strategy_plan_worker_rerank_scale > 0.0
        or strategy_command_gate_threshold >= 0.0
    )
    if needs_aux:
        aux = network.strategy_auxiliary(obs_arr, mask, active)
    if strategy_q_rerank_scale > 0.0:
        logits = strategy_q_rerank_logits(logits[None, :], aux.action_q_values[None, :], strategy_q_rerank_scale)[0]
    if strategy_target_rerank_scale > 0.0:
        logits = strategy_target_rerank_logits(
            logits[None, :],
            aux.enemy_general_logits[None, :, :],
            aux.finish_logits[None, :],
            network.pad_size,
            strategy_target_rerank_scale,
            strategy_target_finish_gate,
        )[0]
    if strategy_spatial_rerank_scale > 0.0:
        logits = strategy_spatial_rerank_logits(
            logits[None, :],
            aux.source_logits[None, :, :],
            aux.target_logits[None, :, :],
            network.pad_size,
            strategy_spatial_rerank_scale,
        )[0]
    if strategy_plan_worker_rerank_scale > 0.0:
        worker_obs = strategy_plan_worker_obs(
            obs_arr,
            mask,
            active,
            aux.source_logits,
            aux.target_logits,
            network.pad_size,
        )
        worker_logits = plan_worker_network.logits_value(worker_obs, mask, active)[0]
        effective_scale = jnp.asarray(strategy_plan_worker_rerank_scale)
        if strategy_plan_worker_min_margin >= 0.0:
            legal_worker_logits = jnp.where(logits > -1.0e8, worker_logits, -1.0e9)
            top2 = jax.lax.top_k(legal_worker_logits, 2)[0]
            worker_margin = top2[0] - top2[1]
            effective_scale = jnp.where(worker_margin >= strategy_plan_worker_min_margin, effective_scale, 0.0)
        logits = strategy_q_rerank_logits(logits[None, :], worker_logits[None, :], effective_scale)[0]
    action_key, worker_key = jrandom.split(key)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(action_key, logits),
        None,
    )
    if strategy_command_gate_threshold >= 0.0:
        command_action, command_source, command_target = strategy_worker_command(
            obs_arr,
            mask,
            active,
            aux.source_logits,
            aux.target_logits,
            network.pad_size,
        )
        command_index = adaptive_action_to_index(command_action, network.pad_size)
        command_legal = logits[command_index] > -1.0e8
        gate_features = command_gate_features(
            obs_arr,
            logits,
            aux.action_q_values,
            aux.finish_logits,
            aux.source_logits,
            aux.target_logits,
            command_source,
            command_target,
            command_index,
            index,
            policy_player,
            network.pad_size,
        )
        gate_probability = jax.nn.sigmoid(command_gate_network(gate_features))
        use_command = (gate_probability >= strategy_command_gate_threshold) & command_legal & (command_index != index)
        index = jnp.where(use_command, command_index, index)
    if strategy_q_replace_threshold >= 0.0:
        if strategy_q_replace_worker_candidate:
            replacement_action = strategy_worker_action(
                obs_arr,
                mask,
                active,
                aux.source_logits,
                aux.target_logits,
                network.pad_size,
            )
            replacement_index = adaptive_action_to_index(replacement_action, network.pad_size)
            replacement_legal = logits[replacement_index] > -1.0e8
        else:
            legal = logits > -1.0e8
            replacement_index = jnp.argmax(jnp.where(legal, aux.action_q_values, -1.0e9))
            replacement_legal = jnp.asarray(True)
        q_advantage = aux.action_q_values[replacement_index] - aux.action_q_values[index]
        if strategy_q_replace_policy_margin >= 0.0:
            policy_supported = logits[replacement_index] >= jnp.max(logits) - strategy_q_replace_policy_margin
        else:
            policy_supported = jnp.asarray(True)
        use_replacement = (q_advantage >= strategy_q_replace_threshold) & policy_supported & replacement_legal
        index = jnp.where(use_replacement, replacement_index, index)
    if strategy_worker_mix_prob > 0.0:
        finish_probability = (
            jax.nn.softmax(aux.finish_logits, axis=-1)[1] if strategy_worker_finish_gate else jnp.asarray(1.0)
        )
        worker_probability = jnp.clip(strategy_worker_mix_prob * finish_probability, 0.0, 1.0)
        worker_action = strategy_worker_action(
            obs_arr,
            mask,
            active,
            aux.source_logits,
            aux.target_logits,
            network.pad_size,
        )
        worker_index = adaptive_action_to_index(worker_action, network.pad_size)
        if strategy_worker_policy_margin >= 0.0:
            worker_supported = logits[worker_index] >= jnp.max(logits) - strategy_worker_policy_margin
        else:
            worker_supported = jnp.asarray(True)
        use_worker = (jrandom.uniform(worker_key) < worker_probability) & worker_supported
        index = jnp.where(use_worker, worker_index, index)
    return adaptive_index_to_action(index, network.pad_size)


def strategy_q_rerank_logits(
    policy_logits: jnp.ndarray,
    action_q_values: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    """Use centered legal strategy-Q predictions as a bias on policy logits."""
    legal = policy_logits > -1.0e8
    legal_count = jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1)
    legal_mean = jnp.sum(jnp.where(legal, action_q_values, 0.0), axis=-1, keepdims=True) / legal_count
    q_bias = jnp.where(legal, action_q_values - legal_mean, 0.0)
    return policy_logits + scale * q_bias


def strategy_worker_action(
    obs_arr: jnp.ndarray,
    legal_mask: jnp.ndarray,
    active: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    pad_size: int,
) -> jnp.ndarray:
    """Choose a source-target plan and execute one legal target-conditioned worker step."""
    action, _, _ = strategy_worker_command(obs_arr, legal_mask, active, source_logits, target_logits, pad_size)
    return action


def strategy_worker_command(
    obs_arr: jnp.ndarray,
    legal_mask: jnp.ndarray,
    active: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    pad_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Choose a source-target plan and return its first worker action and command cells."""
    coords = jnp.arange(pad_size)
    rows = coords[:, None]
    cols = coords[None, :]
    target_scores = jnp.where(active, target_logits, -1.0e9)
    target_index = jnp.argmax(target_scores.reshape(-1))
    target_row = target_index // pad_size
    target_col = target_index % pad_size

    movable = jnp.any(legal_mask, axis=-1)
    army_score = 0.25 * jnp.log1p(jnp.maximum(obs_arr[0], 0.0))
    route_distance = jnp.abs(rows - target_row) + jnp.abs(cols - target_col)
    source_scores = source_logits + army_score - 0.05 * route_distance.astype(jnp.float32)
    source_index = jnp.argmax(jnp.where(movable, source_scores, -1.0e9).reshape(-1))
    source_row = source_index // pad_size
    source_col = source_index % pad_size

    direction_ids = jnp.arange(4)
    dest_rows = source_row + DIRECTIONS[:, 0]
    dest_cols = source_col + DIRECTIONS[:, 1]
    current_distance = jnp.abs(source_row - target_row) + jnp.abs(source_col - target_col)
    next_distance = jnp.abs(dest_rows - target_row) + jnp.abs(dest_cols - target_col)
    progress = current_distance - next_distance
    legal_dirs = legal_mask[source_row, source_col]
    direction_scores = jnp.where(legal_dirs, progress.astype(jnp.float32), -1.0e9)
    direction = jnp.argmax(direction_scores).astype(jnp.int32)
    has_move = jnp.max(direction_scores) > -1.0e8
    return jnp.array(
        [
            (~has_move).astype(jnp.int32),
            source_row.astype(jnp.int32),
            source_col.astype(jnp.int32),
            direction,
            jnp.int32(0),
        ],
        dtype=jnp.int32,
    ), source_index.astype(jnp.int32), target_index.astype(jnp.int32)


def command_gate_features(
    obs_arr: jnp.ndarray,
    policy_logits: jnp.ndarray,
    action_q_values: jnp.ndarray,
    finish_logits: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    source_index: jnp.ndarray,
    target_index: jnp.ndarray,
    candidate_index: jnp.ndarray,
    current_index: jnp.ndarray,
    policy_player: int,
    pad_size: int,
) -> jnp.ndarray:
    """Build the same command-gate feature vector used by offline training."""
    source_row = source_index // pad_size
    source_col = source_index % pad_size
    target_row = target_index // pad_size
    target_col = target_index % pad_size
    route_distance = (jnp.abs(source_row - target_row) + jnp.abs(source_col - target_col)).astype(jnp.float32)
    route_distance = route_distance / jnp.maximum(jnp.asarray(2 * (pad_size - 1), dtype=jnp.float32), 1.0)
    source_army = jnp.log1p(jnp.maximum(obs_arr[0, source_row, source_col], 0.0))
    candidate_policy = policy_logits[candidate_index]
    current_policy = policy_logits[current_index]
    candidate_q = action_q_values[candidate_index]
    current_q = action_q_values[current_index]
    finish_probability = jax.nn.softmax(finish_logits, axis=-1)[1]
    flat_source_logits = source_logits.reshape(-1)
    flat_target_logits = target_logits.reshape(-1)
    return jnp.stack(
        [
            candidate_policy - current_policy,
            candidate_q - current_q,
            flat_source_logits[source_index],
            flat_target_logits[target_index],
            finish_probability,
            source_army,
            route_distance,
            candidate_policy,
            current_policy,
            candidate_q,
            current_q,
            jnp.asarray(policy_player, dtype=jnp.float32),
        ]
    )


def strategy_plan_worker_obs(
    obs_arr: jnp.ndarray,
    legal_mask: jnp.ndarray,
    active: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    pad_size: int,
) -> jnp.ndarray:
    """Append source/target command planes for a learned Plan-Worker."""
    coords = jnp.arange(pad_size)
    rows = coords[:, None]
    cols = coords[None, :]
    target_scores = jnp.where(active, target_logits, -1.0e9)
    target_index = jnp.argmax(target_scores.reshape(-1))
    target_row = target_index // pad_size
    target_col = target_index % pad_size

    movable = jnp.any(legal_mask, axis=-1)
    army_score = 0.25 * obs_arr[0]
    route_distance = jnp.abs(rows - target_row) + jnp.abs(cols - target_col)
    source_scores = source_logits + army_score - 0.05 * route_distance.astype(jnp.float32)
    source_index = jnp.argmax(jnp.where(movable, source_scores, -1.0e9).reshape(-1))
    source_row = source_index // pad_size
    source_col = source_index % pad_size

    source_plane = jnp.zeros((pad_size * pad_size,), dtype=obs_arr.dtype).at[source_index].set(1.0)
    target_plane = jnp.zeros((pad_size * pad_size,), dtype=obs_arr.dtype).at[target_index].set(1.0)
    max_distance = jnp.maximum(jnp.asarray(2 * (pad_size - 1), dtype=jnp.float32), 1.0)
    route_potential = 1.0 - jnp.minimum(route_distance.astype(jnp.float32), max_distance) / max_distance
    command = jnp.stack(
        [
            source_plane.reshape(pad_size, pad_size),
            target_plane.reshape(pad_size, pad_size),
            route_potential * active.astype(jnp.float32),
        ],
        axis=0,
    )
    del source_row, source_col  # Kept by source_index; names make the command construction easier to audit.
    return jnp.concatenate([obs_arr, command], axis=0)


def strategy_target_rerank_logits(
    policy_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    finish_logits: jnp.ndarray,
    pad_size: int,
    scale: float,
    finish_gate: bool,
) -> jnp.ndarray:
    """Bias legal moves that reduce distance to the predicted enemy-general target."""
    target_probs = jax.nn.softmax(target_logits.reshape(target_logits.shape[0], -1), axis=-1)
    coords = jnp.arange(pad_size, dtype=jnp.float32)
    rows = jnp.repeat(coords, pad_size)
    cols = jnp.tile(coords, pad_size)
    target_row = jnp.sum(target_probs * rows[None, :], axis=-1)
    target_col = jnp.sum(target_probs * cols[None, :], axis=-1)

    source_rows = jnp.repeat(coords, pad_size)
    source_cols = jnp.tile(coords, pad_size)
    direction_ids = jnp.arange(8) % 4
    dest_rows = source_rows[None, :] + DIRECTIONS[direction_ids, 0][:, None]
    dest_cols = source_cols[None, :] + DIRECTIONS[direction_ids, 1][:, None]
    source_distance = jnp.abs(source_rows[None, None, :] - target_row[:, None, None])
    source_distance += jnp.abs(source_cols[None, None, :] - target_col[:, None, None])
    dest_distance = jnp.abs(dest_rows[None, :, :] - target_row[:, None, None])
    dest_distance += jnp.abs(dest_cols[None, :, :] - target_col[:, None, None])
    move_bias = (source_distance - dest_distance).reshape(target_logits.shape[0], 8 * pad_size * pad_size)
    action_bias = jnp.concatenate([move_bias, jnp.zeros((target_logits.shape[0], 1), dtype=move_bias.dtype)], axis=-1)

    if finish_gate:
        finish_probability = jax.nn.softmax(finish_logits, axis=-1)[:, 1]
        action_bias = action_bias * finish_probability[:, None]

    legal = policy_logits > -1.0e8
    legal_count = jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1)
    legal_mean = jnp.sum(jnp.where(legal, action_bias, 0.0), axis=-1, keepdims=True) / legal_count
    centered_bias = jnp.where(legal, action_bias - legal_mean, 0.0)
    return policy_logits + scale * centered_bias


def strategy_spatial_rerank_logits(
    policy_logits: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    pad_size: int,
    scale: float,
) -> jnp.ndarray:
    """Bias moves from predicted source cells toward the predicted target heatmap."""
    target_probs = jax.nn.softmax(target_logits.reshape(target_logits.shape[0], -1), axis=-1)
    coords = jnp.arange(pad_size, dtype=jnp.float32)
    rows = jnp.repeat(coords, pad_size)
    cols = jnp.tile(coords, pad_size)
    target_row = jnp.sum(target_probs * rows[None, :], axis=-1)
    target_col = jnp.sum(target_probs * cols[None, :], axis=-1)

    direction_ids = jnp.arange(8) % 4
    dest_rows = rows[None, :] + DIRECTIONS[direction_ids, 0][:, None]
    dest_cols = cols[None, :] + DIRECTIONS[direction_ids, 1][:, None]
    source_distance = jnp.abs(rows[None, None, :] - target_row[:, None, None])
    source_distance += jnp.abs(cols[None, None, :] - target_col[:, None, None])
    dest_distance = jnp.abs(dest_rows[None, :, :] - target_row[:, None, None])
    dest_distance += jnp.abs(dest_cols[None, :, :] - target_col[:, None, None])
    target_progress = (source_distance - dest_distance).reshape(target_logits.shape[0], 8 * pad_size * pad_size)

    centered_source = source_logits.reshape(source_logits.shape[0], -1)
    centered_source = centered_source - jnp.mean(centered_source, axis=-1, keepdims=True)
    source_bias = jnp.tile(centered_source[:, None, :], (1, 8, 1)).reshape(
        source_logits.shape[0],
        8 * pad_size * pad_size,
    )
    move_bias = 0.5 * source_bias + target_progress
    action_bias = jnp.concatenate([move_bias, jnp.zeros((source_logits.shape[0], 1), dtype=move_bias.dtype)], axis=-1)

    legal = policy_logits > -1.0e8
    legal_count = jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1)
    legal_mean = jnp.sum(jnp.where(legal, action_bias, 0.0), axis=-1, keepdims=True) / legal_count
    centered_bias = jnp.where(legal, action_bias - legal_mean, 0.0)
    return policy_logits + scale * centered_bias


def crop_observation(obs, size: int):
    """Crop padded adaptive observations before feeding a fixed-size policy."""
    return obs._replace(
        armies=obs.armies[:size, :size],
        generals=obs.generals[:size, :size],
        cities=obs.cities[:size, :size],
        mountains=obs.mountains[:size, :size],
        neutral_cells=obs.neutral_cells[:size, :size],
        owned_cells=obs.owned_cells[:size, :size],
        opponent_cells=obs.opponent_cells[:size, :size],
        fog_cells=obs.fog_cells[:size, :size],
        structures_in_fog=obs.structures_in_fog[:size, :size],
    )


def summarize_row(info, grid_size: int, policy_player: int, num_games: int) -> AdaptiveEvalRow:
    opponent_player = 1 - policy_player
    wins = jnp.sum(info.winner == policy_player)
    losses = jnp.sum(info.winner == opponent_player)
    draws = jnp.sum(info.winner < 0)
    return AdaptiveEvalRow(
        grid_size=grid_size,
        policy_player=policy_player,
        wins=wins,
        losses=losses,
        draws=draws,
        num_games=num_games,
        mean_time=jnp.mean(info.time),
    )


@eqx.filter_jit
def evaluate_batch(
    network,
    plan_worker_network,
    command_gate_network,
    states,
    effective_size,
    key,
    max_steps,
    opponent,
    policy_mode,
    policy_player,
    pad_size,
    global_context=False,
    scoreboard_history=False,
    fog_memory=False,
    strategy_q_rerank_scale=0.0,
    strategy_q_replace_threshold=-1.0,
    strategy_q_replace_policy_margin=-1.0,
    strategy_q_replace_worker_candidate=False,
    strategy_target_rerank_scale=0.0,
    strategy_target_finish_gate=False,
    strategy_spatial_rerank_scale=0.0,
    strategy_worker_mix_prob=0.0,
    strategy_worker_finish_gate=False,
    strategy_worker_policy_margin=-1.0,
    strategy_plan_worker_rerank_scale=0.0,
    strategy_plan_worker_min_margin=-1.0,
    strategy_command_gate_threshold=-1.0,
):
    """Evaluate one adaptive checkpoint on one grid size and player seat."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)
    initial_history = jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)
    initial_fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)

    def body(carry, _):
        states, key, history, memory = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        if fog_memory:
            current_memory = jax.vmap(update_adaptive_fog_memory)(memory, policy_obs)
        else:
            current_memory = memory

        if scoreboard_history:
            current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                policy_obs,
                effective_sizes,
            )
            history_context = adaptive_scoreboard_history_context(history, current_scoreboard)
            if fog_memory:
                obs_arr, active = jax.vmap(
                    lambda obs, size, row_history, row_memory: adaptive_obs_to_array(
                        obs,
                        size,
                        pad_size,
                        include_global_context=True,
                        scoreboard_history=row_history,
                        fog_memory=row_memory,
                    )
                )(
                    policy_obs,
                    effective_sizes,
                    history_context,
                    current_memory,
                )
            else:
                obs_arr, active = jax.vmap(
                    lambda obs, size, row_history: adaptive_obs_to_array(
                        obs,
                        size,
                        pad_size,
                        include_global_context=True,
                        scoreboard_history=row_history,
                    )
                )(
                    policy_obs,
                    effective_sizes,
                    history_context,
                )
        else:
            current_scoreboard = history
            if fog_memory:
                obs_arr, active = jax.vmap(
                    lambda obs, size, row_memory: adaptive_obs_to_array(
                        obs,
                        size,
                        pad_size,
                        include_global_context=global_context,
                        fog_memory=row_memory,
                    )
                )(
                    policy_obs,
                    effective_sizes,
                    current_memory,
                )
            else:
                obs_arr, active = jax.vmap(
                    lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
                )(
                    policy_obs,
                    effective_sizes,
                )
        masks = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(policy_obs, effective_sizes)

        key, policy_key, opponent_key = jrandom.split(key, 3)
        policy_keys = jrandom.split(policy_key, num_envs)
        policy_actions = jax.vmap(
            lambda o, m, a, k: _policy_action(
                network,
                plan_worker_network,
                command_gate_network,
                o,
                m,
                a,
                k,
                policy_mode,
                policy_player,
                strategy_q_rerank_scale,
                strategy_q_replace_threshold,
                strategy_q_replace_policy_margin,
                strategy_q_replace_worker_candidate,
                strategy_target_rerank_scale,
                strategy_target_finish_gate,
                strategy_spatial_rerank_scale,
                strategy_worker_mix_prob,
                strategy_worker_finish_gate,
                strategy_worker_policy_margin,
                strategy_plan_worker_rerank_scale,
                strategy_plan_worker_min_margin,
                strategy_command_gate_threshold,
            )
        )(
            obs_arr,
            masks,
            active,
            policy_keys,
        )
        opponent_keys = jrandom.split(opponent_key, num_envs)
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent, k, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_memory = current_memory
        return (final_states, key, current_scoreboard, final_memory), infos

    (states, key, _, _), _ = jax.lax.scan(body, (states, key, initial_history, initial_fog_memory), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


@eqx.filter_jit
def evaluate_policy_opponent_batch(
    network,
    plan_worker_network,
    command_gate_network,
    opponent_network,
    states,
    effective_size,
    key,
    max_steps,
    policy_mode,
    policy_player,
    pad_size,
    opponent_policy_mode,
    global_context=False,
    scoreboard_history=False,
    fog_memory=False,
    strategy_q_rerank_scale=0.0,
    strategy_q_replace_threshold=-1.0,
    strategy_q_replace_policy_margin=-1.0,
    strategy_q_replace_worker_candidate=False,
    strategy_target_rerank_scale=0.0,
    strategy_target_finish_gate=False,
    strategy_spatial_rerank_scale=0.0,
    strategy_worker_mix_prob=0.0,
    strategy_worker_finish_gate=False,
    strategy_worker_policy_margin=-1.0,
    strategy_plan_worker_rerank_scale=0.0,
    strategy_plan_worker_min_margin=-1.0,
    strategy_command_gate_threshold=-1.0,
):
    """Evaluate one adaptive checkpoint against one fixed-size PPO checkpoint."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)
    initial_history = jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)
    initial_fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)

    def body(carry, _):
        states, key, history, memory = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        if fog_memory:
            current_memory = jax.vmap(update_adaptive_fog_memory)(memory, policy_obs)
        else:
            current_memory = memory

        if scoreboard_history:
            current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                policy_obs,
                effective_sizes,
            )
            history_context = adaptive_scoreboard_history_context(history, current_scoreboard)
            if fog_memory:
                obs_arr, active = jax.vmap(
                    lambda obs, size, row_history, row_memory: adaptive_obs_to_array(
                        obs,
                        size,
                        pad_size,
                        include_global_context=True,
                        scoreboard_history=row_history,
                        fog_memory=row_memory,
                    )
                )(
                    policy_obs,
                    effective_sizes,
                    history_context,
                    current_memory,
                )
            else:
                obs_arr, active = jax.vmap(
                    lambda obs, size, row_history: adaptive_obs_to_array(
                        obs,
                        size,
                        pad_size,
                        include_global_context=True,
                        scoreboard_history=row_history,
                    )
                )(
                    policy_obs,
                    effective_sizes,
                    history_context,
                )
        else:
            current_scoreboard = history
            if fog_memory:
                obs_arr, active = jax.vmap(
                    lambda obs, size, row_memory: adaptive_obs_to_array(
                        obs,
                        size,
                        pad_size,
                        include_global_context=global_context,
                        fog_memory=row_memory,
                    )
                )(
                    policy_obs,
                    effective_sizes,
                    current_memory,
                )
            else:
                obs_arr, active = jax.vmap(
                    lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
                )(
                    policy_obs,
                    effective_sizes,
                )
        masks = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(policy_obs, effective_sizes)

        key, policy_key, opponent_key = jrandom.split(key, 3)
        policy_keys = jrandom.split(policy_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        policy_actions = jax.vmap(
            lambda o, m, a, k: _policy_action(
                network,
                plan_worker_network,
                command_gate_network,
                o,
                m,
                a,
                k,
                policy_mode,
                policy_player,
                strategy_q_rerank_scale,
                strategy_q_replace_threshold,
                strategy_q_replace_policy_margin,
                strategy_q_replace_worker_candidate,
                strategy_target_rerank_scale,
                strategy_target_finish_gate,
                strategy_spatial_rerank_scale,
                strategy_worker_mix_prob,
                strategy_worker_finish_gate,
                strategy_worker_policy_margin,
                strategy_plan_worker_rerank_scale,
                strategy_plan_worker_min_margin,
                strategy_command_gate_threshold,
            )
        )(
            obs_arr,
            masks,
            active,
            policy_keys,
        )
        opponent_actions = jax.vmap(
            lambda k, obs: policy_network_action(opponent_network, k, crop_observation(obs, effective_size), opponent_policy_mode)
        )(opponent_keys, opponent_obs)
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_memory = current_memory
        return (final_states, key, current_scoreboard, final_memory), infos

    (states, key, _, _), _ = jax.lax.scan(body, (states, key, initial_history, initial_fog_memory), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an adaptive multisize PPO checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--fog-memory", action="store_true")
    parser.add_argument("--context-residual", action="store_true")
    parser.add_argument("--pyramid-context", action="store_true")
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
    parser.add_argument("--strategy-q-rerank-scale", type=float, default=0.0)
    parser.add_argument("--strategy-q-replace-threshold", type=float, default=-1.0)
    parser.add_argument("--strategy-q-replace-policy-margin", type=float, default=-1.0)
    parser.add_argument("--strategy-q-replace-worker-candidate", action="store_true")
    parser.add_argument("--strategy-target-rerank-scale", type=float, default=0.0)
    parser.add_argument("--strategy-target-finish-gate", action="store_true")
    parser.add_argument("--strategy-spatial-rerank-scale", type=float, default=0.0)
    parser.add_argument("--strategy-worker-mix-prob", type=float, default=0.0)
    parser.add_argument("--strategy-worker-finish-gate", action="store_true")
    parser.add_argument("--strategy-worker-policy-margin", type=float, default=-1.0)
    parser.add_argument("--strategy-plan-worker-path", default=None)
    parser.add_argument("--strategy-plan-worker-channels", default=None)
    parser.add_argument("--strategy-plan-worker-network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--strategy-plan-worker-rerank-scale", type=float, default=0.0)
    parser.add_argument("--strategy-plan-worker-min-margin", type=float, default=-1.0)
    parser.add_argument("--strategy-command-gate-path", default=None)
    parser.add_argument("--strategy-command-gate-threshold", type=float, default=-1.0)
    parser.add_argument("--strategy-command-gate-hidden-dim", type=int, default=32)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--require-win-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.value_head_sizes = (
            parse_grid_sizes(args.value_head_sizes) if args.value_head_sizes is not None else args.grid_sizes
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    if args.opponent_policy_path is not None and len(args.grid_sizes) != 1:
        parser.error("--opponent-policy-path requires exactly one --grid-sizes value")
    try:
        args.opponent_channels = parse_policy_channels(args.opponent_channels)
    except ValueError as exc:
        parser.error(str(exc))
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.value_loss == "hl-gauss":
        if args.value_bins <= 1:
            parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
        if args.value_min >= args.value_max:
            parser.error("--value-min must be less than --value-max")
        if args.value_sigma <= 0.0:
            parser.error("--value-sigma must be positive")
    if args.require_win_rate is not None and not (0.0 <= args.require_win_rate <= 1.0):
        parser.error("--require-win-rate must be between 0 and 1")
    if args.strategy_q_rerank_scale < 0.0:
        parser.error("--strategy-q-rerank-scale must be non-negative")
    if args.strategy_q_rerank_scale > 0.0 and not args.strategy_aux:
        parser.error("--strategy-q-rerank-scale requires --strategy-aux")
    if args.strategy_q_replace_threshold >= 0.0 and not args.strategy_aux:
        parser.error("--strategy-q-replace-threshold requires --strategy-aux")
    if args.strategy_q_replace_policy_margin < 0.0 and args.strategy_q_replace_policy_margin != -1.0:
        parser.error("--strategy-q-replace-policy-margin must be non-negative, or -1 to disable")
    if args.strategy_q_replace_policy_margin >= 0.0 and args.strategy_q_replace_threshold < 0.0:
        parser.error("--strategy-q-replace-policy-margin requires --strategy-q-replace-threshold")
    if args.strategy_q_replace_worker_candidate and args.strategy_q_replace_threshold < 0.0:
        parser.error("--strategy-q-replace-worker-candidate requires --strategy-q-replace-threshold")
    if args.strategy_q_replace_worker_candidate and not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("--strategy-q-replace-worker-candidate requires --strategy-aux --strategy-spatial-aux")
    if args.strategy_target_rerank_scale < 0.0:
        parser.error("--strategy-target-rerank-scale must be non-negative")
    if args.strategy_target_rerank_scale > 0.0 and not args.strategy_aux:
        parser.error("--strategy-target-rerank-scale requires --strategy-aux")
    if args.strategy_target_finish_gate and args.strategy_target_rerank_scale <= 0.0:
        parser.error("--strategy-target-finish-gate requires --strategy-target-rerank-scale")
    if args.strategy_spatial_rerank_scale < 0.0:
        parser.error("--strategy-spatial-rerank-scale must be non-negative")
    if args.strategy_spatial_rerank_scale > 0.0 and not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("--strategy-spatial-rerank-scale requires --strategy-aux --strategy-spatial-aux")
    if not (0.0 <= args.strategy_worker_mix_prob <= 1.0):
        parser.error("--strategy-worker-mix-prob must be between 0 and 1")
    if args.strategy_worker_mix_prob > 0.0 and not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("--strategy-worker-mix-prob requires --strategy-aux --strategy-spatial-aux")
    if args.strategy_worker_finish_gate and args.strategy_worker_mix_prob <= 0.0:
        parser.error("--strategy-worker-finish-gate requires --strategy-worker-mix-prob")
    if args.strategy_worker_policy_margin < 0.0 and args.strategy_worker_policy_margin != -1.0:
        parser.error("--strategy-worker-policy-margin must be non-negative, or -1 to disable")
    if args.strategy_plan_worker_rerank_scale < 0.0:
        parser.error("--strategy-plan-worker-rerank-scale must be non-negative")
    if args.strategy_plan_worker_rerank_scale > 0.0 and args.strategy_plan_worker_path is None:
        parser.error("--strategy-plan-worker-rerank-scale requires --strategy-plan-worker-path")
    if args.strategy_plan_worker_rerank_scale > 0.0 and not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("--strategy-plan-worker-rerank-scale requires --strategy-aux --strategy-spatial-aux")
    if args.strategy_plan_worker_min_margin < 0.0 and args.strategy_plan_worker_min_margin != -1.0:
        parser.error("--strategy-plan-worker-min-margin must be non-negative, or -1 to disable")
    if args.strategy_plan_worker_min_margin >= 0.0 and args.strategy_plan_worker_rerank_scale <= 0.0:
        parser.error("--strategy-plan-worker-min-margin requires --strategy-plan-worker-rerank-scale")
    if args.strategy_command_gate_threshold < 0.0 and args.strategy_command_gate_threshold != -1.0:
        parser.error("--strategy-command-gate-threshold must be between 0 and 1, or -1 to disable")
    if args.strategy_command_gate_threshold > 1.0:
        parser.error("--strategy-command-gate-threshold must be between 0 and 1")
    if args.strategy_command_gate_threshold >= 0.0 and args.strategy_command_gate_path is None:
        parser.error("--strategy-command-gate-threshold requires --strategy-command-gate-path")
    if args.strategy_command_gate_threshold >= 0.0 and not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("--strategy-command-gate-threshold requires --strategy-aux --strategy-spatial-aux")
    if args.strategy_command_gate_hidden_dim <= 0:
        parser.error("--strategy-command-gate-hidden-dim must be positive")
    try:
        args.strategy_plan_worker_channels = parse_policy_channels(args.strategy_plan_worker_channels)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _row_to_printable(row: AdaptiveEvalRow) -> str:
    return (
        f"{row.grid_size}x{row.grid_size} player {row.policy_player}: "
        f"wins/losses/draws={row.wins}/{row.losses}/{row.draws}, "
        f"win_rate={row.win_rate * 100:.2f}%, "
        f"decisive={row.decisive_win_rate * 100:.2f}%, "
        f"draw={row.draw_rate * 100:.2f}%, "
        f"mean_time={row.mean_time:.1f}"
    )


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)
    key, net_key = jrandom.split(key)
    network_global_context = args.global_context or args.scoreboard_history
    input_channels = adaptive_input_channel_count(network_global_context, args.scoreboard_history, args.fog_memory)
    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.model_path,
        channels=args.channels,
        input_channels=input_channels,
        init_input_channels=input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        value_bins=args.value_bins if args.value_loss == "hl-gauss" else 0,
        value_min=args.value_min,
        value_max=args.value_max,
        value_sigma=args.value_sigma,
        outcome_head=args.outcome_head,
        strategy_aux=args.strategy_aux,
        strategy_spatial_aux=args.strategy_spatial_aux,
        global_context=network_global_context,
        init_global_context=network_global_context,
        context_residual=args.context_residual,
        init_context_residual=args.context_residual,
        pyramid_context=args.pyramid_context,
        init_pyramid_context=args.pyramid_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
    )
    plan_worker_network = None
    if args.strategy_plan_worker_path is not None:
        plan_worker_input_channels = input_channels + 3
        plan_worker_network = load_or_create_adaptive_network(
            net_key,
            pad_size=args.pad_to,
            init_model_path=args.strategy_plan_worker_path,
            channels=args.strategy_plan_worker_channels,
            input_channels=plan_worker_input_channels,
            init_input_channels=plan_worker_input_channels,
            network_arch=args.strategy_plan_worker_network_arch,
            init_network_arch=args.strategy_plan_worker_network_arch,
        )
    command_gate_network = None
    if args.strategy_command_gate_path is not None:
        command_gate_network = CommandGateNetwork(net_key, hidden_dim=args.strategy_command_gate_hidden_dim)
        command_gate_network = eqx.tree_deserialise_leaves(args.strategy_command_gate_path, command_gate_network)
    opponent_network = None
    if args.opponent_policy_path is not None:
        opponent_network = PolicyValueNetwork(
            net_key,
            grid_size=args.grid_sizes[0],
            channels=args.opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_network)
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    policy_mode = 0 if args.policy_mode == "greedy" else 1
    opponent_policy_mode = 0 if args.opponent_policy_mode == "greedy" else 1
    rows = []

    print("Adaptive policy evaluation")
    print(f"Model:       {args.model_path}")
    print(f"Device:      {jax.devices()[0]}")
    print(f"Grid sizes:  {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if opponent_network is None:
        print(f"Opponent:    {args.opponent}")
    else:
        print("Opponent:    policy checkpoint")
        print(f"Opp model:   {args.opponent_policy_path}")
        print(f"Opp mode:    {args.opponent_policy_mode}")
        print(f"Opp channels:{args.opponent_channels}")
        print(f"Opp inputs:  {args.opponent_input_channels}")
    print(f"Mode:        {args.policy_mode}")
    print(f"Arch:        {args.network_arch}")
    if args.value_heads != "shared":
        print(f"Value heads: {args.value_heads}")
    if args.value_loss == "hl-gauss":
        print(
            "Value loss:  "
            f"hl-gauss bins={args.value_bins} range=[{args.value_min:g},{args.value_max:g}] "
            f"sigma={args.value_sigma:g}"
        )
    if args.outcome_head:
        print("Outcome:    auxiliary head loaded")
    if args.strategy_aux:
        print("Strategy:   auxiliary heads loaded")
    if args.strategy_spatial_aux:
        print("Spatial:    source/target strategy heads loaded")
    if args.strategy_q_rerank_scale > 0.0:
        print(f"StratQ bias: scale={args.strategy_q_rerank_scale:g}")
    if args.strategy_q_replace_threshold >= 0.0:
        print(f"StratQ gate: threshold={args.strategy_q_replace_threshold:g}")
        if args.strategy_q_replace_policy_margin >= 0.0:
            print(f"StratQ gate: policy_margin={args.strategy_q_replace_policy_margin:g}")
        if args.strategy_q_replace_worker_candidate:
            print("StratQ gate: worker candidate only")
    if args.strategy_target_rerank_scale > 0.0:
        gate_label = " finish-gated" if args.strategy_target_finish_gate else ""
        print(f"Target bias: scale={args.strategy_target_rerank_scale:g}{gate_label}")
    if args.strategy_spatial_rerank_scale > 0.0:
        print(f"Spatial bias: scale={args.strategy_spatial_rerank_scale:g}")
    if args.strategy_worker_mix_prob > 0.0:
        gate_label = " finish-gated" if args.strategy_worker_finish_gate else ""
        margin_label = (
            f", policy-margin={args.strategy_worker_policy_margin:g}"
            if args.strategy_worker_policy_margin >= 0.0
            else ""
        )
        print(f"Worker mix:  p={args.strategy_worker_mix_prob:g}{gate_label}{margin_label}")
    if args.strategy_plan_worker_rerank_scale > 0.0:
        print(f"Plan worker: {args.strategy_plan_worker_path}")
        print(
            "Plan worker: "
            f"arch={args.strategy_plan_worker_network_arch}, scale={args.strategy_plan_worker_rerank_scale:g}"
        )
        if args.strategy_plan_worker_min_margin >= 0.0:
            print(f"Plan worker: min_margin={args.strategy_plan_worker_min_margin:g}")
    if args.strategy_command_gate_threshold >= 0.0:
        print(f"Command gate: {args.strategy_command_gate_path}")
        print(
            "Command gate: "
            f"threshold={args.strategy_command_gate_threshold:g}, hidden={args.strategy_command_gate_hidden_dim}"
        )
    if args.context_residual:
        print("Context res: 5x5 residual branch")
    if args.pyramid_context:
        print("Pyramid ctx: U-Net branch")
    if network_global_context:
        print(f"Global ctx: {input_channels} input channels")
    if args.scoreboard_history:
        print("Score hist: previous+delta channels")
    if args.fog_memory:
        print("Fog memory: explored/enemy/city/general planes")
    print()

    for grid_size in args.grid_sizes:
        for policy_player in (0, 1):
            key, pool_key, eval_key = jrandom.split(key, 3)
            pool = make_adaptive_state_pool(
                pool_key,
                args.num_games,
                (grid_size,),
                args.pad_to,
                args.map_generator,
                (args.mountain_density_min, args.mountain_density_max),
                (args.num_cities_min, args.num_cities_max),
                args.max_generals_distance,
                (args.city_army_min, args.city_army_max),
            )
            states = pool.states
            t0 = time.time()
            if opponent_network is None:
                info = evaluate_batch(
                    network,
                    plan_worker_network,
                    command_gate_network,
                    states,
                    grid_size,
                    eval_key,
                    args.max_steps,
                    opponent_id,
                    policy_mode,
                    policy_player,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    args.fog_memory,
                    args.strategy_q_rerank_scale,
                    args.strategy_q_replace_threshold,
                    args.strategy_q_replace_policy_margin,
                    args.strategy_q_replace_worker_candidate,
                    args.strategy_target_rerank_scale,
                    args.strategy_target_finish_gate,
                    args.strategy_spatial_rerank_scale,
                    args.strategy_worker_mix_prob,
                    args.strategy_worker_finish_gate,
                    args.strategy_worker_policy_margin,
                    args.strategy_plan_worker_rerank_scale,
                    args.strategy_plan_worker_min_margin,
                    args.strategy_command_gate_threshold,
                )
            else:
                info = evaluate_policy_opponent_batch(
                    network,
                    plan_worker_network,
                    command_gate_network,
                    opponent_network,
                    states,
                    grid_size,
                    eval_key,
                    args.max_steps,
                    policy_mode,
                    policy_player,
                    args.pad_to,
                    opponent_policy_mode,
                    network_global_context,
                    args.scoreboard_history,
                    args.fog_memory,
                    args.strategy_q_rerank_scale,
                    args.strategy_q_replace_threshold,
                    args.strategy_q_replace_policy_margin,
                    args.strategy_q_replace_worker_candidate,
                    args.strategy_target_rerank_scale,
                    args.strategy_target_finish_gate,
                    args.strategy_spatial_rerank_scale,
                    args.strategy_worker_mix_prob,
                    args.strategy_worker_finish_gate,
                    args.strategy_worker_policy_margin,
                    args.strategy_plan_worker_rerank_scale,
                    args.strategy_plan_worker_min_margin,
                    args.strategy_command_gate_threshold,
                )
            jax.block_until_ready(info.winner)
            row_jax = summarize_row(info, grid_size, policy_player, args.num_games)
            row = AdaptiveEvalRow(
                grid_size=grid_size,
                policy_player=policy_player,
                wins=int(row_jax.wins),
                losses=int(row_jax.losses),
                draws=int(row_jax.draws),
                num_games=args.num_games,
                mean_time=float(row_jax.mean_time),
            )
            rows.append(row)
            elapsed = time.time() - t0
            print(f"{_row_to_printable(row)} | elapsed={elapsed:.2f}s")

    min_win_rate = min(row.win_rate for row in rows)
    payload = {
        "model_path": args.model_path,
        "grid_sizes": list(args.grid_sizes),
        "pad_to": args.pad_to,
        "opponent": args.opponent,
        "opponent_policy_path": args.opponent_policy_path,
        "opponent_policy_mode": args.opponent_policy_mode,
        "opponent_channels": args.opponent_channels,
        "opponent_input_channels": args.opponent_input_channels,
        "value_head_sizes": list(args.value_head_sizes) if args.value_heads == "per-size" else [],
        "policy_mode": args.policy_mode,
        "num_games": args.num_games,
        "max_steps": args.max_steps,
        "global_context": network_global_context,
        "scoreboard_history": args.scoreboard_history,
        "fog_memory": args.fog_memory,
        "network_arch": args.network_arch,
        "context_residual": args.context_residual,
        "pyramid_context": args.pyramid_context,
        "strategy_aux": args.strategy_aux,
        "strategy_spatial_aux": args.strategy_spatial_aux,
        "strategy_q_rerank_scale": args.strategy_q_rerank_scale,
        "strategy_q_replace_threshold": args.strategy_q_replace_threshold,
        "strategy_q_replace_policy_margin": args.strategy_q_replace_policy_margin,
        "strategy_q_replace_worker_candidate": args.strategy_q_replace_worker_candidate,
        "strategy_target_rerank_scale": args.strategy_target_rerank_scale,
        "strategy_target_finish_gate": args.strategy_target_finish_gate,
        "strategy_spatial_rerank_scale": args.strategy_spatial_rerank_scale,
        "strategy_worker_mix_prob": args.strategy_worker_mix_prob,
        "strategy_worker_finish_gate": args.strategy_worker_finish_gate,
        "strategy_worker_policy_margin": args.strategy_worker_policy_margin,
        "strategy_plan_worker_path": args.strategy_plan_worker_path,
        "strategy_plan_worker_network_arch": args.strategy_plan_worker_network_arch,
        "strategy_plan_worker_rerank_scale": args.strategy_plan_worker_rerank_scale,
        "strategy_plan_worker_min_margin": args.strategy_plan_worker_min_margin,
        "strategy_command_gate_path": args.strategy_command_gate_path,
        "strategy_command_gate_threshold": args.strategy_command_gate_threshold,
        "strategy_command_gate_hidden_dim": args.strategy_command_gate_hidden_dim,
        "min_win_rate": min_win_rate,
        "rows": [row.to_dict() for row in rows],
    }
    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Minimum win rate: {min_win_rate * 100:.2f}%")
    if args.require_win_rate is not None and min_win_rate < args.require_win_rate:
        print(f"Required win rate {args.require_win_rate * 100:.2f}% not reached")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
