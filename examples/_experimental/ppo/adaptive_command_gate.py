"""Lightweight command-acceptance gate for adaptive source-target plans."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.adaptive_common import (
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS,
)

COMMAND_GATE_FEATURE_NAMES = (
    "policy_logit_delta",
    "action_q_delta",
    "source_logit",
    "target_logit",
    "finish_probability",
    "source_army_log1p",
    "route_distance_norm",
    "candidate_policy_logit",
    "current_policy_logit",
    "candidate_q",
    "current_q",
    "seat",
    "active_area_fraction",
)
COMMAND_GATE_FEATURE_DIM = len(COMMAND_GATE_FEATURE_NAMES)

POLICY_ADAPTER_GATE_FEATURE_NAMES = (
    "adapter_delta_at_adapter_top",
    "adapter_delta_at_policy_top",
    "policy_support_for_adapter_top",
    "adapter_top_margin",
    "policy_top_margin",
    "adapter_finish_probability",
    "adapter_draw_probability",
    "adapter_win_probability",
    "visible_enemy_density",
    "visible_enemy_army_log_density",
    "owned_army_log_density",
    "active_fraction",
    "adapter_changes_action",
    "seat",
    "scoreboard_time",
    "scoreboard_land_advantage",
    "scoreboard_army_advantage",
    "contact_binary",
)


class CommandGateNetwork(eqx.Module):
    """Small normalized MLP that predicts whether to accept a command action."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int = COMMAND_GATE_FEATURE_DIM,
        hidden_dim: int = 32,
        feature_mean: jnp.ndarray | None = None,
        feature_std: jnp.ndarray | None = None,
    ):
        key1, key2 = jrandom.split(key)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.linear1 = eqx.nn.Linear(self.input_dim, self.hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(self.hidden_dim, 1, key=key2)
        self.feature_mean = (
            jnp.zeros((self.input_dim,), dtype=jnp.float32)
            if feature_mean is None
            else jnp.asarray(feature_mean, dtype=jnp.float32)
        )
        self.feature_std = (
            jnp.ones((self.input_dim,), dtype=jnp.float32)
            if feature_std is None
            else jnp.asarray(feature_std, dtype=jnp.float32)
        )

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        x = (features - self.feature_mean) / jnp.maximum(self.feature_std, 1.0e-6)
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)[0]


def strategy_finish_probability(finish_logits: jnp.ndarray) -> jnp.ndarray:
    """Return the most terminal-horizon finish probability across finish head layouts."""
    if finish_logits.shape[0] == 1:
        return jax.nn.sigmoid(finish_logits[0])
    if finish_logits.shape[0] == 2:
        return jax.nn.softmax(finish_logits, axis=-1)[1]
    return jax.nn.sigmoid(finish_logits[-1])


def policy_adapter_gate_features(
    obs_arr: jnp.ndarray,
    policy_logits: jnp.ndarray,
    adapter_logits: jnp.ndarray,
    finish_logits: jnp.ndarray,
    outcome_logits: jnp.ndarray,
    active: jnp.ndarray,
    policy_player: int,
    pad_size: int,
    feature_dim: int = 12,
) -> jnp.ndarray:
    """Build the adapter-gate feature vector used by offline training."""
    legal = policy_logits > -1.0e8
    policy_values, policy_indices = jax.lax.top_k(jnp.where(legal, policy_logits, -1.0e9), 2)
    adapter_values, adapter_indices = jax.lax.top_k(jnp.where(legal, adapter_logits, -1.0e9), 2)
    policy_index = policy_indices[0]
    adapter_index = adapter_indices[0]
    raw_delta = adapter_logits - policy_logits
    active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    visible_enemy = obs_arr[6] * active.astype(jnp.float32)
    owned = obs_arr[5] * active.astype(jnp.float32)
    army_log = obs_arr[0]
    finish_probability = strategy_finish_probability(finish_logits)
    outcome_probabilities = jax.nn.softmax(outcome_logits, axis=-1)
    visible_enemy_density = jnp.sum(visible_enemy) / active_count
    channel_count = obs_arr.shape[0]
    scoreboard_time = jnp.asarray(0.0, dtype=obs_arr.dtype)
    scoreboard_land_advantage = jnp.asarray(0.0, dtype=obs_arr.dtype)
    scoreboard_army_advantage = jnp.asarray(0.0, dtype=obs_arr.dtype)
    has_global = channel_count >= ADAPTIVE_INPUT_CHANNELS + ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS
    if has_global:
        has_history = (
            channel_count
            >= ADAPTIVE_INPUT_CHANNELS + ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS + ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS
        )
        global_width = ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS + (
            ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS if has_history else 0
        )
        current_start = channel_count - global_width
        current_scoreboard = jnp.stack(
            [
                jnp.sum(obs_arr[current_start + index] * active.astype(jnp.float32)) / active_count
                for index in range(ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS)
            ]
        )
        scoreboard_time = current_scoreboard[4]
        scoreboard_land_advantage = current_scoreboard[0] - current_scoreboard[2]
        scoreboard_army_advantage = current_scoreboard[1] - current_scoreboard[3]
    features = jnp.stack(
        [
            raw_delta[adapter_index],
            raw_delta[policy_index],
            policy_logits[adapter_index] - policy_values[0],
            adapter_values[0] - adapter_values[1],
            policy_values[0] - policy_values[1],
            finish_probability,
            outcome_probabilities[1],
            outcome_probabilities[2],
            visible_enemy_density,
            jnp.sum(army_log * visible_enemy) / active_count,
            jnp.sum(army_log * owned) / active_count,
            active_count / jnp.asarray(pad_size * pad_size, dtype=jnp.float32),
            (adapter_index != policy_index).astype(jnp.float32),
            jnp.asarray(policy_player, dtype=jnp.float32),
            scoreboard_time,
            scoreboard_land_advantage,
            scoreboard_army_advantage,
            (visible_enemy_density > 0.0).astype(jnp.float32),
        ]
    )
    return features[:feature_dim]


def command_gate_features(
    obs_arr: jnp.ndarray,
    active: jnp.ndarray,
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
    feature_dim: int = COMMAND_GATE_FEATURE_DIM,
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
    finish_probability = strategy_finish_probability(finish_logits)
    active_area_fraction = jnp.sum(active.astype(jnp.float32)) / jnp.asarray(pad_size * pad_size, dtype=jnp.float32)
    flat_source_logits = source_logits.reshape(-1)
    flat_target_logits = target_logits.reshape(-1)
    features = jnp.stack(
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
            active_area_fraction,
        ]
    )
    return features[:feature_dim]
