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

from adaptive_command_gate import COMMAND_GATE_FEATURE_DIM, CommandGateNetwork
from adaptive_common import (
    ADAPTIVE_GLOBAL_INPUT_CHANNELS,
    ADAPTIVE_HISTORY_INPUT_CHANNELS,
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_MOVE_PLANES,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS,
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
    reset_adaptive_scoreboard_history,
    update_adaptive_fog_memory,
)
from adaptive_network import load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, POLICY_MODE_NAMES, opponent_action, policy_network_action
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from generals.core.action import DIRECTIONS
from generals.core import game
from train import random_action, stack_learner_actions

PLAN_WORKER_COMMAND_SOURCE_NAMES = ("spatial", "belief-main-stack", "main-stack-heuristic")
PLAN_WORKER_COMMAND_SOURCE_TO_ID = {name: index for index, name in enumerate(PLAN_WORKER_COMMAND_SOURCE_NAMES)}
POLICY_ADAPTER_MODE_NAMES = ("delta", "blend", "replace")
POLICY_ADAPTER_MODE_TO_ID = {name: index for index, name in enumerate(POLICY_ADAPTER_MODE_NAMES)}
ONLINE_SEARCH_GATE_FEATURE_DIM = 17
CANDIDATE_SCORER_BASE_FEATURE_NAMES = (
    "prior_score",
    "prior_rank",
    "candidate_minus_base_prior",
    "candidate_is_base_action",
    "is_pass",
    "is_half",
    "dir_up",
    "dir_down",
    "dir_left",
    "dir_right",
    "source_row_norm",
    "source_col_norm",
    "dest_row_norm",
    "dest_col_norm",
    "source_active",
    "dest_active",
    "source_legal_dir",
    "source_army_log",
    "dest_army_log",
    "source_owned",
    "dest_owned",
    "dest_enemy",
    "dest_neutral",
    "dest_city",
    "dest_fog",
    "dest_structure_fog",
    "source_general",
    "dest_general",
    "full_capture_margin_log",
    "half_capture_margin_log",
    "time_norm",
    "seat",
    "grid_norm",
    "active_fraction",
    "visible_enemy_density",
    "contact",
)


class OnlineSearchCandidateScorer(eqx.Module):
    """Small normalized MLP used to score top-k primitive candidates at inference."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int,
        hidden_dim: int = 128,
        feature_mean: jnp.ndarray | None = None,
        feature_std: jnp.ndarray | None = None,
    ):
        key1, key2, key3 = jrandom.split(key, 3)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.linear1 = eqx.nn.Linear(self.input_dim, self.hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=key2)
        self.linear3 = eqx.nn.Linear(self.hidden_dim, 1, key=key3)
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
        feature_mean = jax.lax.stop_gradient(self.feature_mean)
        feature_std = jnp.maximum(jax.lax.stop_gradient(self.feature_std), 1.0e-6)
        x = (features - feature_mean) / feature_std
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)[0]


def candidate_scorer_feature_names(local_channels: int) -> list[str]:
    """Return the currently supported online candidate-scorer feature layout."""
    names = list(CANDIDATE_SCORER_BASE_FEATURE_NAMES)
    names.extend(f"source_ch{idx}" for idx in range(local_channels))
    names.extend(f"dest_ch{idx}" for idx in range(local_channels))
    names.extend(f"source_minus_dest_ch{idx}" for idx in range(local_channels))
    return names


def candidate_scorer_sidecar_path(model_path: Path) -> Path:
    """Return the JSON metadata path for a candidate scorer checkpoint."""
    direct = model_path.with_suffix(".json")
    if direct.exists():
        return direct
    if model_path.stem.endswith(".best"):
        fallback = model_path.with_name(model_path.stem.removesuffix(".best") + ".json")
        if fallback.exists():
            return fallback
    return direct


def load_candidate_scorer(model_path: str, seed: int = 0) -> tuple[OnlineSearchCandidateScorer, int, list[str]]:
    """Load an offline candidate scorer and validate its online feature schema."""
    path = Path(model_path)
    sidecar_path = candidate_scorer_sidecar_path(path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Candidate-scorer sidecar not found: {sidecar_path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    feature_names = list(metadata.get("feature_names", ()))
    if not feature_names:
        raise ValueError("Candidate-scorer sidecar is missing feature_names")
    base_dim = len(CANDIDATE_SCORER_BASE_FEATURE_NAMES)
    extra_dim = len(feature_names) - base_dim
    if extra_dim < 0 or extra_dim % 3 != 0:
        raise ValueError("Candidate-scorer feature layout is not supported by online evaluation")
    local_channels = extra_dim // 3
    expected_names = candidate_scorer_feature_names(local_channels)
    if feature_names != expected_names:
        raise ValueError(
            "Candidate-scorer online evaluation currently supports only base+local-channel "
            "feature sidecars without heatmap or trunk features"
        )
    hidden_dim = int(metadata.get("hidden_dim", 128))
    template = OnlineSearchCandidateScorer(
        jrandom.PRNGKey(seed),
        input_dim=len(feature_names),
        hidden_dim=hidden_dim,
    )
    return eqx.tree_deserialise_leaves(path, template), local_channels, feature_names


@dataclass(frozen=True)
class AdaptiveEvalRow:
    grid_size: int
    policy_player: int
    wins: int
    losses: int
    draws: int
    num_games: int
    mean_time: float
    adapter_trigger_rate: float = 0.0
    adapter_used_rate: float = 0.0
    adapter_action_diff_rate: float = 0.0

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


def parse_policy_players(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of policy seats for focused evaluation."""
    players = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not players:
        raise ValueError("at least one policy player is required")
    if any(player not in (0, 1) for player in players):
        raise ValueError("--policy-players entries must be 0 or 1")
    if len(set(players)) != len(players):
        raise ValueError("--policy-players cannot repeat a seat")
    return players


@eqx.filter_jit
def _policy_action(
    network,
    policy_adapter_network,
    policy_adapter_feature_network,
    late_policy_adapter_network,
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
    strategy_plan_worker_command_source: int,
    strategy_plan_worker_gate_threshold: float,
    strategy_command_gate_threshold: float,
    strategy_command_gate_source_count: int,
    strategy_command_gate_target_count: int,
    command_gate_feature_dim: int,
    policy_adapter_scale: float,
    conversion_policy_scale: float,
    conversion_policy_mode: int,
    policy_adapter_finish_threshold: float,
    policy_adapter_gate_threshold: float,
    policy_adapter_mode: int,
    late_policy_adapter_scale: float,
    late_policy_adapter_mode: int,
    policy_adapter_commit_active,
    policy_adapter_context_allowed=True,
    late_policy_adapter_context_allowed=True,
):
    logits, _ = network.logits_value(obs_arr, mask, active)
    adapter_trigger = jnp.asarray(0.0, dtype=logits.dtype)
    adapter_used = jnp.asarray(0.0, dtype=logits.dtype)
    adapter_action_diff = jnp.asarray(0.0, dtype=logits.dtype)
    plan_worker_enabled = strategy_plan_worker_rerank_scale > 0.0 or strategy_plan_worker_gate_threshold >= 0.0
    plan_worker_uses_aux_command = (
        plan_worker_enabled
        and strategy_plan_worker_command_source != PLAN_WORKER_COMMAND_SOURCE_TO_ID["main-stack-heuristic"]
    )
    needs_aux = (
        strategy_q_rerank_scale > 0.0
        or strategy_q_replace_threshold >= 0.0
        or strategy_target_rerank_scale > 0.0
        or strategy_spatial_rerank_scale > 0.0
        or strategy_worker_mix_prob > 0.0
        or plan_worker_uses_aux_command
        or strategy_plan_worker_gate_threshold >= 0.0
        or strategy_command_gate_threshold >= 0.0
    )
    if needs_aux:
        aux = network.strategy_auxiliary(obs_arr, mask, active)
    if conversion_policy_scale > 0.0:
        base_logits_for_conversion = logits
        conversion_logits = network.conversion_policy_logits(obs_arr, mask, active)
        conversion_gate = jnp.where(policy_adapter_context_allowed, 1.0, 0.0)
        conversion_weight = conversion_policy_scale * conversion_gate
        converted_logits = jax.lax.switch(
            conversion_policy_mode,
            (
                lambda _: policy_adapter_delta_logits(logits, conversion_logits, conversion_weight),
                lambda _: policy_adapter_blend_logits(logits, conversion_logits, conversion_weight),
                lambda _: jnp.where(conversion_gate > 0.0, conversion_logits, logits),
            ),
            None,
        )
        legal = base_logits_for_conversion > -1.0e8
        base_top = jnp.argmax(jnp.where(legal, base_logits_for_conversion, -1.0e9))
        converted_top = jnp.argmax(jnp.where(legal, converted_logits, -1.0e9))
        conversion_used = (conversion_gate > 0.0).astype(logits.dtype)
        adapter_trigger = jnp.maximum(adapter_trigger, conversion_used)
        adapter_used = jnp.maximum(adapter_used, conversion_used)
        adapter_action_diff = jnp.maximum(
            adapter_action_diff,
            conversion_used * (converted_top != base_top).astype(logits.dtype),
        )
        logits = converted_logits
    if policy_adapter_scale > 0.0:
        base_logits_for_adapter = logits
        adapter_logits, _ = policy_adapter_network.logits_value(obs_arr, mask, active)
        adapter_gate = jnp.asarray(1.0, dtype=logits.dtype)
        if policy_adapter_gate_threshold >= 0.0:
            adapter_feature_network = (
                policy_adapter_feature_network if policy_adapter_feature_network is not None else policy_adapter_network
            )
            adapter_aux = adapter_feature_network.strategy_auxiliary(obs_arr, mask, active)
            if adapter_feature_network.outcome_head:
                _, _, _, adapter_outcome_logits = adapter_feature_network.logits_value_auxiliary(obs_arr, mask, active)
            else:
                adapter_outcome_logits = jnp.zeros((3,), dtype=logits.dtype)
            adapter_features = policy_adapter_gate_features(
                obs_arr,
                logits,
                adapter_logits,
                adapter_aux.finish_logits,
                adapter_outcome_logits,
                active,
                policy_player,
                network.pad_size,
                command_gate_feature_dim,
            )
            gate_probability = jax.nn.sigmoid(command_gate_network(adapter_features))
            adapter_trigger = jnp.where(gate_probability >= policy_adapter_gate_threshold, 1.0, 0.0)
            adapter_gate = adapter_trigger
        elif policy_adapter_finish_threshold >= 0.0:
            adapter_feature_network = (
                policy_adapter_feature_network if policy_adapter_feature_network is not None else policy_adapter_network
            )
            adapter_aux = adapter_feature_network.strategy_auxiliary(obs_arr, mask, active)
            finish_probability = strategy_finish_probability(adapter_aux.finish_logits)
            adapter_trigger = jnp.where(finish_probability >= policy_adapter_finish_threshold, 1.0, 0.0)
            adapter_gate = adapter_trigger
        adapter_gate = jnp.where(policy_adapter_commit_active > 0, 1.0, adapter_gate)
        adapter_gate = jnp.where(policy_adapter_context_allowed, adapter_gate, 0.0)
        adapter_weight = policy_adapter_scale * adapter_gate
        adapted_logits = jax.lax.switch(
            policy_adapter_mode,
            (
                lambda _: policy_adapter_delta_logits(logits, adapter_logits, adapter_weight),
                lambda _: policy_adapter_blend_logits(logits, adapter_logits, adapter_weight),
                lambda _: jnp.where(adapter_gate > 0.0, adapter_logits, logits),
            ),
            None,
        )
        legal = base_logits_for_adapter > -1.0e8
        base_top = jnp.argmax(jnp.where(legal, base_logits_for_adapter, -1.0e9))
        adapted_top = jnp.argmax(jnp.where(legal, adapted_logits, -1.0e9))
        primary_adapter_used = (adapter_gate > 0.0).astype(logits.dtype)
        adapter_used = jnp.maximum(adapter_used, primary_adapter_used)
        adapter_action_diff = jnp.maximum(
            adapter_action_diff,
            primary_adapter_used * (adapted_top != base_top).astype(logits.dtype),
        )
        logits = adapted_logits
    if late_policy_adapter_scale > 0.0:
        base_logits_for_late_adapter = logits
        late_adapter_logits, _ = late_policy_adapter_network.logits_value(obs_arr, mask, active)
        late_adapter_gate = jnp.where(late_policy_adapter_context_allowed, 1.0, 0.0)
        late_adapter_weight = late_policy_adapter_scale * late_adapter_gate
        late_adapted_logits = jax.lax.switch(
            late_policy_adapter_mode,
            (
                lambda _: policy_adapter_delta_logits(logits, late_adapter_logits, late_adapter_weight),
                lambda _: policy_adapter_blend_logits(logits, late_adapter_logits, late_adapter_weight),
                lambda _: jnp.where(late_adapter_gate > 0.0, late_adapter_logits, logits),
            ),
            None,
        )
        legal = base_logits_for_late_adapter > -1.0e8
        base_top = jnp.argmax(jnp.where(legal, base_logits_for_late_adapter, -1.0e9))
        late_top = jnp.argmax(jnp.where(legal, late_adapted_logits, -1.0e9))
        late_adapter_used = (late_adapter_gate > 0.0).astype(logits.dtype)
        adapter_used = jnp.maximum(adapter_used, late_adapter_used)
        adapter_action_diff = jnp.maximum(
            adapter_action_diff,
            late_adapter_used * (late_top != base_top).astype(logits.dtype),
        )
        logits = late_adapted_logits
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
    if plan_worker_enabled:
        plan_worker_policy_logits = logits
        if strategy_plan_worker_command_source == PLAN_WORKER_COMMAND_SOURCE_TO_ID["main-stack-heuristic"]:
            worker_source_logits, worker_target_logits = main_stack_heuristic_worker_command_logits(obs_arr, active)
        elif strategy_plan_worker_command_source == PLAN_WORKER_COMMAND_SOURCE_TO_ID["belief-main-stack"]:
            worker_source_logits = jnp.zeros_like(aux.enemy_general_logits)
            worker_target_logits = aux.enemy_general_logits
        else:
            worker_source_logits = aux.source_logits
            worker_target_logits = aux.target_logits
        worker_obs = strategy_plan_worker_obs(
            obs_arr,
            mask,
            active,
            worker_source_logits,
            worker_target_logits,
            network.pad_size,
        )
        worker_logits = plan_worker_network.logits_value(worker_obs, mask, active)[0]
    if strategy_plan_worker_rerank_scale > 0.0:
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
        command_index, gate_probability = strategy_command_gate_index(
            command_gate_network,
            obs_arr,
            logits,
            aux.action_q_values,
            aux.finish_logits,
            mask,
            active,
            aux.source_logits,
            aux.target_logits,
            index,
            policy_player,
            network.pad_size,
            command_gate_feature_dim,
            strategy_command_gate_source_count,
            strategy_command_gate_target_count,
        )
        command_legal = logits[command_index] > -1.0e8
        use_command = (gate_probability >= strategy_command_gate_threshold) & command_legal & (command_index != index)
        index = jnp.where(use_command, command_index, index)
    if strategy_plan_worker_gate_threshold >= 0.0:
        legal_worker_logits = jnp.where(plan_worker_policy_logits > -1.0e8, worker_logits, -1.0e9)
        worker_index = jnp.argmax(legal_worker_logits)
        pass_index = ADAPTIVE_MOVE_PLANES * network.pad_size * network.pad_size
        worker_source_index = jnp.minimum(worker_index, pass_index - 1) % (network.pad_size * network.pad_size)
        worker_target_scores = jnp.where(active, worker_target_logits, -1.0e9)
        worker_target_index = jnp.argmax(worker_target_scores.reshape(-1))
        worker_features = command_gate_features(
            obs_arr,
            active,
            plan_worker_policy_logits,
            aux.action_q_values,
            aux.finish_logits,
            worker_source_logits,
            worker_target_logits,
            worker_source_index,
            worker_target_index,
            worker_index,
            index,
            policy_player,
            network.pad_size,
            command_gate_feature_dim,
        )
        worker_probability = jax.nn.sigmoid(command_gate_network(worker_features))
        worker_legal = plan_worker_policy_logits[worker_index] > -1.0e8
        use_worker = (
            (worker_probability >= strategy_plan_worker_gate_threshold)
            & worker_legal
            & (worker_index != index)
            & (worker_index != pass_index)
        )
        index = jnp.where(use_worker, worker_index, index)
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
    return adaptive_index_to_action(index, network.pad_size), adapter_trigger, adapter_used, adapter_action_diff


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


def strategy_finish_probability(finish_logits: jnp.ndarray) -> jnp.ndarray:
    """Return the most terminal-horizon finish probability across finish head layouts."""
    if finish_logits.shape[0] == 1:
        return jax.nn.sigmoid(finish_logits[0])
    if finish_logits.shape[0] == 2:
        return jax.nn.softmax(finish_logits, axis=-1)[1]
    return jax.nn.sigmoid(finish_logits[-1])


def main_stack_heuristic_worker_command_logits(
    obs_arr: jnp.ndarray,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build commands matching main-stack/heuristic Plan-Q prefix data."""
    army_log = jnp.maximum(obs_arr[0], 0.0)
    generals = obs_arr[1]
    cities = obs_arr[2]
    mountains = obs_arr[3] > 0.5
    neutral = obs_arr[4]
    owned = obs_arr[5]
    enemy = obs_arr[6]
    structures_in_fog = obs_arr[8]
    passable = active & ~mountains
    not_owned = 1.0 - owned

    target_logits = jnp.where(passable, 0.01, -1.0e9)
    target_logits = target_logits + enemy * (20.0 + army_log)
    target_logits = target_logits + cities * not_owned * 40.0
    target_logits = target_logits + generals * not_owned * 1000.0
    target_logits = target_logits + structures_in_fog * not_owned * 12.0
    target_logits = target_logits + neutral * not_owned * 0.05
    source_logits = jnp.zeros_like(target_logits)
    return source_logits, target_logits


def policy_adapter_delta_logits(
    policy_logits: jnp.ndarray,
    adapter_logits: jnp.ndarray,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
    """Add a centered legal delta from a separately trained policy-head adapter."""
    legal = policy_logits > -1.0e8
    raw_delta = adapter_logits - policy_logits
    legal_count = jnp.maximum(jnp.sum(legal), 1)
    legal_mean = jnp.sum(jnp.where(legal, raw_delta, 0.0)) / legal_count
    centered_delta = jnp.where(legal, raw_delta - legal_mean, 0.0)
    return policy_logits + scale * centered_delta


def policy_adapter_blend_logits(
    policy_logits: jnp.ndarray,
    adapter_logits: jnp.ndarray,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
    """Interpolate legal logits between the base policy and adapter policy."""
    legal = policy_logits > -1.0e8
    weight = jnp.clip(jnp.asarray(scale, dtype=policy_logits.dtype), 0.0, 1.0)
    blended = (1.0 - weight) * policy_logits + weight * adapter_logits
    return jnp.where(legal, blended, policy_logits)


def adapter_composed_policy_logits(
    network,
    policy_adapter_network,
    obs_arr: jnp.ndarray,
    mask: jnp.ndarray,
    active: jnp.ndarray,
    effective_size: int,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
) -> jnp.ndarray:
    """Return base policy logits with the deployment policy adapter composed in."""
    logits, _ = network.logits_value(obs_arr, mask, active)
    if policy_adapter_network is None or policy_adapter_scale <= 0.0:
        return logits
    size_allowed = (policy_adapter_min_grid_size <= 0 or effective_size >= policy_adapter_min_grid_size) and (
        policy_adapter_max_grid_size <= 0 or effective_size <= policy_adapter_max_grid_size
    )
    adapter_logits, _ = policy_adapter_network.logits_value(obs_arr, mask, active)
    adapted_logits = jax.lax.switch(
        policy_adapter_mode,
        (
            lambda _: policy_adapter_delta_logits(logits, adapter_logits, policy_adapter_scale),
            lambda _: policy_adapter_blend_logits(logits, adapter_logits, policy_adapter_scale),
            lambda _: adapter_logits,
        ),
        None,
    )
    return jnp.where(size_allowed, adapted_logits, logits)


def scalar_reset_fog_memory(memory, done: jnp.ndarray):
    """Reset one scalar fog-memory state after a terminal search rollout."""
    keep = (~done).astype(jnp.float32)
    return jax.tree.map(lambda value: value * keep, memory)


def search_score_observation(info, obs, player: int, army_weight: float, land_weight: float, terminal_score: float):
    """Score a search rollout leaf from one player's perspective."""
    army_balance = (obs.owned_army_count.astype(jnp.float32) - obs.opponent_army_count.astype(jnp.float32)) / jnp.maximum(
        obs.owned_army_count + obs.opponent_army_count,
        1,
    )
    land_balance = (obs.owned_land_count.astype(jnp.float32) - obs.opponent_land_count.astype(jnp.float32)) / obs.armies.size
    terminal = jnp.where(
        info.winner == player,
        terminal_score,
        jnp.where(info.winner == 1 - player, -terminal_score, 0.0),
    )
    return terminal + army_weight * army_balance + land_weight * land_balance


def adapter_policy_action_with_memory(
    network,
    policy_adapter_network,
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
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
):
    """Dispatch the deployment policy while carrying one-player context."""
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
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, pad_size), current_scoreboard, current_memory


def candidate_scorer_features(
    obs_arr: jnp.ndarray,
    active: jnp.ndarray,
    mask: jnp.ndarray,
    policy_logits: jnp.ndarray,
    fallback_action: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    prior_scores: jnp.ndarray,
    effective_size: int,
    time: jnp.ndarray,
    policy_player: int,
    max_steps: int,
    pad_size: int,
    local_channels: int,
) -> jnp.ndarray:
    """Build online features matching adaptive_online_search_candidate_scorer.py."""
    top_k = candidate_indices.shape[0]
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    safe_indices = jnp.minimum(jnp.clip(candidate_indices, 0, pass_index), pass_index - 1)
    plane = safe_indices // (pad_size * pad_size)
    position = safe_indices % (pad_size * pad_size)
    source_row = position // pad_size
    source_col = position % pad_size
    direction = plane % 4
    is_pass = candidate_indices == pass_index
    is_half = plane >= 4
    direction_deltas = jnp.asarray(DIRECTIONS, dtype=jnp.int32)
    raw_dest_row = source_row + direction_deltas[direction, 0]
    raw_dest_col = source_col + direction_deltas[direction, 1]
    dest_in_bounds = (
        (raw_dest_row >= 0)
        & (raw_dest_row < pad_size)
        & (raw_dest_col >= 0)
        & (raw_dest_col < pad_size)
    )
    dest_row = jnp.clip(raw_dest_row, 0, pad_size - 1)
    dest_col = jnp.clip(raw_dest_col, 0, pad_size - 1)

    source_planes = jnp.moveaxis(obs_arr[:, source_row, source_col], 0, 1)
    dest_planes = jnp.moveaxis(obs_arr[:, dest_row, dest_col], 0, 1)
    source_active = active[source_row, source_col]
    dest_active = active[dest_row, dest_col] & dest_in_bounds
    source_legal_dir = mask[source_row, source_col, direction]
    fallback_index = adaptive_action_to_index(fallback_action, pad_size)
    fallback_prior = policy_logits[fallback_index]
    source_army = source_planes[:, 0]
    dest_army = dest_planes[:, 0]
    move_all = jnp.maximum(jnp.expm1(jnp.maximum(source_army, 0.0)) - 1.0, 0.0)
    move_half = jnp.floor(jnp.maximum(jnp.expm1(jnp.maximum(source_army, 0.0)), 0.0) / 2.0)
    dest_army_raw = jnp.maximum(jnp.expm1(jnp.maximum(dest_army, 0.0)), 0.0)
    active_float = active.astype(jnp.float32)
    active_fraction = jnp.mean(active_float)
    active_count = jnp.maximum(jnp.sum(active_float), 1.0)
    visible_enemy_density = jnp.sum(obs_arr[6] * active_float) / active_count
    contact = (visible_enemy_density > 0.0).astype(jnp.float32)
    rank_denom = jnp.maximum(jnp.asarray(top_k - 1, dtype=jnp.float32), 1.0)
    coords_denom = jnp.maximum(jnp.asarray(pad_size - 1, dtype=jnp.float32), 1.0)

    feature_parts = [
        prior_scores[:, None],
        (jnp.arange(top_k, dtype=jnp.float32) / rank_denom)[:, None],
        (prior_scores - fallback_prior)[:, None],
        (candidate_indices == fallback_index).astype(jnp.float32)[:, None],
        is_pass.astype(jnp.float32)[:, None],
        is_half.astype(jnp.float32)[:, None],
        jax.nn.one_hot(direction, 4, dtype=jnp.float32),
        (source_row.astype(jnp.float32) / coords_denom)[:, None],
        (source_col.astype(jnp.float32) / coords_denom)[:, None],
        (dest_row.astype(jnp.float32) / coords_denom)[:, None],
        (dest_col.astype(jnp.float32) / coords_denom)[:, None],
        source_active.astype(jnp.float32)[:, None],
        dest_active.astype(jnp.float32)[:, None],
        source_legal_dir.astype(jnp.float32)[:, None],
        source_army[:, None],
        dest_army[:, None],
        source_planes[:, 5:6],
        dest_planes[:, 5:6],
        dest_planes[:, 6:7],
        dest_planes[:, 4:5],
        dest_planes[:, 2:3],
        dest_planes[:, 7:8],
        dest_planes[:, 8:9],
        (source_planes[:, 1:2] * source_planes[:, 5:6]),
        (dest_planes[:, 1:2] * dest_planes[:, 6:7]),
        jnp.log1p(jnp.maximum(move_all - dest_army_raw, 0.0))[:, None],
        jnp.log1p(jnp.maximum(move_half - dest_army_raw, 0.0))[:, None],
        jnp.full((top_k, 1), time.astype(jnp.float32) / jnp.asarray(max(max_steps, 1), dtype=jnp.float32)),
        jnp.full((top_k, 1), jnp.asarray(policy_player, dtype=jnp.float32)),
        jnp.full((top_k, 1), jnp.asarray(effective_size / max(pad_size, 1), dtype=jnp.float32)),
        jnp.full((top_k, 1), active_fraction),
        jnp.full((top_k, 1), visible_enemy_density),
        jnp.full((top_k, 1), contact),
    ]
    kept_channels = min(local_channels, obs_arr.shape[0])
    if kept_channels > 0:
        feature_parts.extend(
            [
                source_planes[:, :kept_channels],
                dest_planes[:, :kept_channels],
                source_planes[:, :kept_channels] - dest_planes[:, :kept_channels],
            ]
        )
    return jnp.concatenate(feature_parts, axis=-1)


def candidate_scorer_action(
    network,
    policy_adapter_network,
    candidate_scorer_network,
    obs_arr: jnp.ndarray,
    mask: jnp.ndarray,
    active: jnp.ndarray,
    fallback_action: jnp.ndarray,
    effective_size: int,
    time: jnp.ndarray,
    policy_player: int,
    max_steps: int,
    pad_size: int,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
    top_k: int,
    min_score_gap: float,
    local_channels: int,
) -> jnp.ndarray:
    """Choose a top-k prior action with a learned offline search scorer."""
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    features = candidate_scorer_features(
        obs_arr,
        active,
        mask,
        logits,
        fallback_action,
        candidate_indices,
        prior_scores,
        effective_size,
        time,
        policy_player,
        max_steps,
        pad_size,
        local_channels,
    )
    candidate_scores = jax.vmap(candidate_scorer_network)(features)
    best_position = jnp.argmax(candidate_scores)
    best_index = candidate_indices[best_position]
    score_gap = jnp.asarray(jnp.inf, dtype=candidate_scores.dtype)
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(candidate_scores, 2)
        score_gap = top_scores[0] - top_scores[1]
    best_action = adaptive_index_to_action(best_index, pad_size)
    return jnp.where(score_gap >= min_score_gap, best_action, fallback_action)


def online_search_action_policy_opponent(
    network,
    policy_adapter_network,
    command_gate_network,
    opponent_network,
    state,
    effective_size: int,
    key,
    fallback_action: jnp.ndarray,
    opponent_first_action: jnp.ndarray,
    policy_player: int,
    policy_mode: int,
    opponent_policy_mode: int,
    pad_size: int,
    max_steps: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
    top_k: int,
    rollout_steps: int,
    rollouts_per_action: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
    min_score_gap: float,
    online_search_gate_threshold: float,
    online_search_gate_feature_dim: int,
) -> jnp.ndarray:
    """Choose a primitive action by online counterfactual rollout search against a fixed policy."""
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
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    opponent_player = 1 - policy_player

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, prev_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, policy_player)
            learner_action, next_scoreboard, next_memory = adapter_policy_action_with_memory(
                network,
                policy_adapter_network,
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
                policy_adapter_scale,
                policy_adapter_mode,
                policy_adapter_min_grid_size,
                policy_adapter_max_grid_size,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = policy_network_action(
                opponent_network,
                opponent_key,
                crop_observation(opponent_obs, effective_size),
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
        return search_score_observation(scored_info, final_obs, policy_player, army_weight, land_weight, terminal_score)

    def score_candidate(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        rollout_scores = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        first_terminal = jnp.where(
            first_info.winner == policy_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        return first_terminal + jnp.mean(rollout_scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(score_candidate)(candidate_actions, prior_scores, candidate_keys)
    best_position = jnp.argmax(scores)
    best_action = candidate_actions[best_position]
    score_gap = jnp.asarray(jnp.inf, dtype=scores.dtype)
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(scores, 2)
        score_gap = top_scores[0] - top_scores[1]
    accept_search = score_gap >= min_score_gap
    if online_search_gate_threshold >= 0.0:
        gate_features = online_search_gate_features(
            scores,
            prior_scores,
            best_position,
            adaptive_action_to_index(best_action, pad_size),
            logits,
            fallback_action,
            active,
            obs_arr,
            state.time,
            policy_player,
            max_steps,
            pad_size,
            online_search_gate_feature_dim,
        )
        gate_probability = jax.nn.sigmoid(command_gate_network(gate_features))
        accept_search = accept_search & (gate_probability >= online_search_gate_threshold)
    return jnp.where(accept_search, best_action, fallback_action)


def online_search_action_heuristic_opponent(
    network,
    policy_adapter_network,
    command_gate_network,
    opponent,
    state,
    effective_size: int,
    key,
    fallback_action: jnp.ndarray,
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
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
    top_k: int,
    rollout_steps: int,
    rollouts_per_action: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
    min_score_gap: float,
    online_search_gate_threshold: float,
    online_search_gate_feature_dim: int,
) -> jnp.ndarray:
    """Choose a primitive action by online counterfactual rollout search against a heuristic opponent."""
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
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    opponent_player = 1 - policy_player

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, prev_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, policy_player)
            learner_action, next_scoreboard, next_memory = adapter_policy_action_with_memory(
                network,
                policy_adapter_network,
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
                policy_adapter_scale,
                policy_adapter_mode,
                policy_adapter_min_grid_size,
                policy_adapter_max_grid_size,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = opponent_action(opponent, opponent_key, opponent_obs, random_action)
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
        return search_score_observation(scored_info, final_obs, policy_player, army_weight, land_weight, terminal_score)

    def score_candidate(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        rollout_scores = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        first_terminal = jnp.where(
            first_info.winner == policy_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        return first_terminal + jnp.mean(rollout_scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(score_candidate)(candidate_actions, prior_scores, candidate_keys)
    best_position = jnp.argmax(scores)
    best_action = candidate_actions[best_position]
    score_gap = jnp.asarray(jnp.inf, dtype=scores.dtype)
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(scores, 2)
        score_gap = top_scores[0] - top_scores[1]
    accept_search = score_gap >= min_score_gap
    if online_search_gate_threshold >= 0.0:
        gate_features = online_search_gate_features(
            scores,
            prior_scores,
            best_position,
            adaptive_action_to_index(best_action, pad_size),
            logits,
            fallback_action,
            active,
            obs_arr,
            state.time,
            policy_player,
            max_steps,
            pad_size,
            online_search_gate_feature_dim,
        )
        gate_probability = jax.nn.sigmoid(command_gate_network(gate_features))
        accept_search = accept_search & (gate_probability >= online_search_gate_threshold)
    return jnp.where(accept_search, best_action, fallback_action)


def online_search_gate_features(
    scores: jnp.ndarray,
    prior_scores: jnp.ndarray,
    best_position: jnp.ndarray,
    best_action_index: jnp.ndarray,
    policy_logits: jnp.ndarray,
    fallback_action: jnp.ndarray,
    active: jnp.ndarray,
    obs_arr: jnp.ndarray,
    time: jnp.ndarray,
    policy_player: int,
    max_steps: int,
    pad_size: int,
    feature_dim: int = ONLINE_SEARCH_GATE_FEATURE_DIM,
) -> jnp.ndarray:
    """Build online-search accept/reject features matching the offline trainer."""
    top_k = scores.shape[0]
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(scores, 2)
        second_score = top_scores[1]
        top_prior, _ = jax.lax.top_k(prior_scores, 2)
        second_prior = top_prior[1]
        prior_gap = top_prior[0] - top_prior[1]
    else:
        second_score = scores[best_position]
        second_prior = prior_scores[best_position]
        prior_gap = jnp.asarray(0.0, dtype=prior_scores.dtype)
    best_score = scores[best_position]
    mean_score = jnp.mean(scores)
    std_score = jnp.std(scores)
    best_prior = prior_scores[best_position]
    fallback_index = adaptive_action_to_index(fallback_action, pad_size)
    fallback_prior = policy_logits[fallback_index]
    search_action_changed = (best_action_index != fallback_index).astype(jnp.float32)
    active_float = active.astype(jnp.float32)
    active_fraction = jnp.mean(active_float)
    active_count = jnp.maximum(jnp.sum(active_float), 1.0)
    visible_enemy_density = jnp.sum(obs_arr[6] * active_float) / active_count
    contact = (visible_enemy_density > 0.0).astype(jnp.float32)
    denom = jnp.maximum(jnp.asarray(top_k - 1, dtype=jnp.float32), 1.0)
    features = jnp.stack(
        [
            best_score,
            second_score,
            best_score - second_score,
            mean_score,
            std_score,
            best_prior,
            second_prior,
            prior_gap,
            best_position.astype(jnp.float32) / denom,
            fallback_prior,
            best_prior - fallback_prior,
            search_action_changed,
            time.astype(jnp.float32) / jnp.asarray(max(max_steps, 1), dtype=jnp.float32),
            jnp.asarray(policy_player, dtype=jnp.float32),
            active_fraction,
            visible_enemy_density,
            contact,
        ]
    )
    return features[:feature_dim]


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


def strategy_command_action_from_indices(
    legal_mask: jnp.ndarray,
    source_index: jnp.ndarray,
    target_index: jnp.ndarray,
    pad_size: int,
) -> jnp.ndarray:
    """Return one legal move from a source cell toward a target cell."""
    source_row = source_index // pad_size
    source_col = source_index % pad_size
    target_row = target_index // pad_size
    target_col = target_index % pad_size
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
    )


def strategy_command_gate_index(
    command_gate_network,
    obs_arr: jnp.ndarray,
    policy_logits: jnp.ndarray,
    action_q_values: jnp.ndarray,
    finish_logits: jnp.ndarray,
    legal_mask: jnp.ndarray,
    active: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    current_index: jnp.ndarray,
    policy_player: int,
    pad_size: int,
    command_gate_feature_dim: int,
    source_count: int,
    target_count: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Score top-k source/target commands and return the highest gate-probability action."""
    movable = jnp.any(legal_mask, axis=-1)
    source_scores = jnp.where(movable, source_logits, -1.0e9)
    target_scores = jnp.where(active, target_logits, -1.0e9)
    _, source_indices = jax.lax.top_k(source_scores.reshape(-1), source_count)
    _, target_indices = jax.lax.top_k(target_scores.reshape(-1), target_count)
    pair_sources = jnp.repeat(source_indices.astype(jnp.int32), target_count)
    pair_targets = jnp.tile(target_indices.astype(jnp.int32), source_count)

    def score_pair(source_index, target_index):
        action = strategy_command_action_from_indices(legal_mask, source_index, target_index, pad_size)
        action_index = adaptive_action_to_index(action, pad_size)
        features = command_gate_features(
            obs_arr,
            active,
            policy_logits,
            action_q_values,
            finish_logits,
            source_logits,
            target_logits,
            source_index,
            target_index,
            action_index,
            current_index,
            policy_player,
            pad_size,
            command_gate_feature_dim,
        )
        probability = jax.nn.sigmoid(command_gate_network(features))
        legal = policy_logits[action_index] > -1.0e8
        usable = legal & (action_index != current_index)
        return action_index, jnp.where(usable, probability, -1.0)

    action_indices, probabilities = jax.vmap(score_pair)(pair_sources, pair_targets)
    best_pos = jnp.argmax(probabilities)
    return action_indices[best_pos], probabilities[best_pos]


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
    if finish_logits.shape[0] == 1:
        finish_probability = jax.nn.sigmoid(finish_logits[0])
    elif finish_logits.shape[0] == 2:
        finish_probability = jax.nn.softmax(finish_logits, axis=-1)[1]
    else:
        finish_probability = jax.nn.sigmoid(finish_logits[-1])
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


def summarize_row(info, grid_size: int, policy_player: int, num_games: int, adapter_stats=None) -> AdaptiveEvalRow:
    opponent_player = 1 - policy_player
    wins = jnp.sum(info.winner == policy_player)
    losses = jnp.sum(info.winner == opponent_player)
    draws = jnp.sum(info.winner < 0)
    if adapter_stats is None:
        adapter_trigger_rate = jnp.asarray(0.0)
        adapter_used_rate = jnp.asarray(0.0)
        adapter_action_diff_rate = jnp.asarray(0.0)
    else:
        adapter_trigger_sum, adapter_used_sum, adapter_action_diff_sum, active_decision_sum = adapter_stats
        denominator = jnp.maximum(active_decision_sum, 1.0)
        adapter_trigger_rate = adapter_trigger_sum / denominator
        adapter_used_rate = adapter_used_sum / denominator
        adapter_action_diff_rate = adapter_action_diff_sum / denominator
    return AdaptiveEvalRow(
        grid_size=grid_size,
        policy_player=policy_player,
        wins=wins,
        losses=losses,
        draws=draws,
        num_games=num_games,
        mean_time=jnp.mean(info.time),
        adapter_trigger_rate=adapter_trigger_rate,
        adapter_used_rate=adapter_used_rate,
        adapter_action_diff_rate=adapter_action_diff_rate,
    )


@eqx.filter_jit
def evaluate_batch(
    network,
    policy_adapter_network,
    policy_adapter_feature_network,
    late_policy_adapter_network,
    plan_worker_network,
    command_gate_network,
    candidate_scorer_network,
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
    strategy_plan_worker_command_source=0,
    strategy_plan_worker_gate_threshold=-1.0,
    strategy_plan_worker_min_grid_size=0,
    strategy_plan_worker_max_grid_size=0,
    strategy_command_gate_threshold=-1.0,
    strategy_command_gate_source_count=1,
    strategy_command_gate_target_count=1,
    command_gate_feature_dim=COMMAND_GATE_FEATURE_DIM,
    policy_adapter_scale=0.0,
    conversion_policy_scale=0.0,
    conversion_policy_mode=0,
    policy_adapter_finish_threshold=-1.0,
    policy_adapter_gate_threshold=-1.0,
    policy_adapter_mode=0,
    late_policy_adapter_scale=0.0,
    late_policy_adapter_mode=0,
    late_policy_adapter_min_grid_size=0,
    late_policy_adapter_max_grid_size=0,
    late_policy_adapter_min_turn=0,
    late_policy_adapter_require_contact=False,
    policy_adapter_min_grid_size=0,
    policy_adapter_max_grid_size=0,
    policy_adapter_min_turn=0,
    policy_adapter_require_contact=False,
    policy_adapter_commit_steps=0,
    online_search_top_k=0,
    online_search_rollout_steps=16,
    online_search_rollouts_per_action=1,
    online_search_min_turn=0,
    online_search_require_contact=False,
    online_search_min_grid_size=0,
    online_search_max_grid_size=0,
    online_search_army_weight=1.0,
    online_search_land_weight=10.0,
    online_search_prior_weight=0.001,
    online_search_terminal_score=100.0,
    online_search_min_score_gap=0.0,
    online_search_gate_threshold=-1.0,
    online_search_gate_feature_dim=ONLINE_SEARCH_GATE_FEATURE_DIM,
    candidate_scorer_top_k=0,
    candidate_scorer_min_turn=0,
    candidate_scorer_require_contact=False,
    candidate_scorer_min_grid_size=0,
    candidate_scorer_max_grid_size=0,
    candidate_scorer_min_score_gap=0.0,
    candidate_scorer_local_channels=0,
):
    """Evaluate one adaptive checkpoint on one grid size and player seat."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)
    initial_history = jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)
    initial_fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)
    plan_worker_size_allowed = (
        (strategy_plan_worker_min_grid_size <= 0 or effective_size >= strategy_plan_worker_min_grid_size)
        and (strategy_plan_worker_max_grid_size <= 0 or effective_size <= strategy_plan_worker_max_grid_size)
    )
    effective_plan_worker_rerank_scale = (
        strategy_plan_worker_rerank_scale if plan_worker_size_allowed else 0.0
    )
    effective_plan_worker_gate_threshold = (
        strategy_plan_worker_gate_threshold if plan_worker_size_allowed else -1.0
    )
    policy_adapter_size_allowed = (
        (policy_adapter_min_grid_size <= 0 or effective_size >= policy_adapter_min_grid_size)
        and (policy_adapter_max_grid_size <= 0 or effective_size <= policy_adapter_max_grid_size)
    )
    size_policy_adapter_scale = policy_adapter_scale if policy_adapter_size_allowed else 0.0
    late_policy_adapter_size_allowed = (
        (late_policy_adapter_min_grid_size <= 0 or effective_size >= late_policy_adapter_min_grid_size)
        and (late_policy_adapter_max_grid_size <= 0 or effective_size <= late_policy_adapter_max_grid_size)
    )
    size_late_policy_adapter_scale = late_policy_adapter_scale if late_policy_adapter_size_allowed else 0.0
    online_search_size_allowed = (
        (online_search_min_grid_size <= 0 or effective_size >= online_search_min_grid_size)
        and (online_search_max_grid_size <= 0 or effective_size <= online_search_max_grid_size)
    )
    online_search_enabled = online_search_top_k > 0 and online_search_size_allowed
    candidate_scorer_size_allowed = (
        (candidate_scorer_min_grid_size <= 0 or effective_size >= candidate_scorer_min_grid_size)
        and (candidate_scorer_max_grid_size <= 0 or effective_size <= candidate_scorer_max_grid_size)
    )
    candidate_scorer_enabled = candidate_scorer_top_k > 0 and candidate_scorer_size_allowed
    initial_adapter_commit = jnp.zeros((num_envs,), dtype=jnp.int32)

    def body(carry, _):
        states, key, history, memory, adapter_commit = carry
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
        pre_infos = jax.vmap(game.get_info)(states)
        active_decisions = (~pre_infos.is_done).astype(jnp.float32)
        adapter_visible_contact = jnp.sum(policy_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
        adapter_turn_allowed = states.time >= policy_adapter_min_turn
        adapter_contact_allowed = adapter_visible_contact | (not policy_adapter_require_contact)
        row_policy_adapter_allowed = adapter_turn_allowed & adapter_contact_allowed
        late_adapter_turn_allowed = states.time >= late_policy_adapter_min_turn
        late_adapter_contact_allowed = adapter_visible_contact | (not late_policy_adapter_require_contact)
        row_late_policy_adapter_allowed = late_adapter_turn_allowed & late_adapter_contact_allowed
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent, k, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )
        policy_actions, adapter_triggers, adapter_used, adapter_action_diff = jax.vmap(
            lambda o, m, a, k, c, adapter_allowed, late_adapter_allowed: _policy_action(
                network,
                policy_adapter_network,
                policy_adapter_feature_network,
                late_policy_adapter_network,
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
                effective_plan_worker_rerank_scale,
                strategy_plan_worker_min_margin,
                strategy_plan_worker_command_source,
                effective_plan_worker_gate_threshold,
                strategy_command_gate_threshold,
                strategy_command_gate_source_count,
                strategy_command_gate_target_count,
                command_gate_feature_dim,
                size_policy_adapter_scale,
                conversion_policy_scale if policy_adapter_size_allowed else 0.0,
                conversion_policy_mode,
                policy_adapter_finish_threshold,
                policy_adapter_gate_threshold,
                policy_adapter_mode,
                size_late_policy_adapter_scale,
                late_policy_adapter_mode,
                c,
                adapter_allowed,
                late_adapter_allowed,
            )
        )(
            obs_arr,
            masks,
            active,
            policy_keys,
            adapter_commit,
            row_policy_adapter_allowed,
            row_late_policy_adapter_allowed,
        )
        if candidate_scorer_enabled:
            visible_contact = jnp.sum(policy_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
            scorer_turn_allowed = states.time >= candidate_scorer_min_turn
            scorer_contact_allowed = visible_contact | (not candidate_scorer_require_contact)
            use_candidate_scorer = (~pre_infos.is_done) & scorer_turn_allowed & scorer_contact_allowed
            policy_actions = jax.vmap(
                lambda o, m, a, base_action, turn, use_scorer: jax.lax.cond(
                    use_scorer,
                    lambda _: candidate_scorer_action(
                        network,
                        policy_adapter_network,
                        candidate_scorer_network,
                        o,
                        m,
                        a,
                        base_action,
                        effective_size,
                        turn,
                        policy_player,
                        max_steps,
                        pad_size,
                        size_policy_adapter_scale,
                        policy_adapter_mode,
                        policy_adapter_min_grid_size,
                        policy_adapter_max_grid_size,
                        candidate_scorer_top_k,
                        candidate_scorer_min_score_gap,
                        candidate_scorer_local_channels,
                    ),
                    lambda _: base_action,
                    None,
                )
            )(obs_arr, masks, active, policy_actions, states.time, use_candidate_scorer)
        if online_search_enabled:
            key, search_key = jrandom.split(key)
            search_keys = jrandom.split(search_key, num_envs)
            visible_contact = jnp.sum(policy_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
            search_turn_allowed = states.time >= online_search_min_turn
            search_contact_allowed = visible_contact | (not online_search_require_contact)
            use_online_search = (~pre_infos.is_done) & search_turn_allowed & search_contact_allowed
            policy_actions = jax.vmap(
                lambda state, sample_key, base_action, opponent_action_value, row_history, row_memory, use_search: jax.lax.cond(
                    use_search,
                    lambda _: online_search_action_heuristic_opponent(
                        network,
                        policy_adapter_network,
                        command_gate_network,
                        opponent,
                        state,
                        effective_size,
                        sample_key,
                        base_action,
                        opponent_action_value,
                        policy_player,
                        policy_mode,
                        pad_size,
                        max_steps,
                        global_context,
                        scoreboard_history,
                        row_history,
                        fog_memory,
                        row_memory,
                        size_policy_adapter_scale,
                        policy_adapter_mode,
                        policy_adapter_min_grid_size,
                        policy_adapter_max_grid_size,
                        online_search_top_k,
                        online_search_rollout_steps,
                        online_search_rollouts_per_action,
                        online_search_army_weight,
                        online_search_land_weight,
                        online_search_prior_weight,
                        online_search_terminal_score,
                        online_search_min_score_gap,
                        online_search_gate_threshold,
                        online_search_gate_feature_dim,
                    ),
                    lambda _: base_action,
                    None,
                )
            )(states, search_keys, policy_actions, opponent_actions, history, current_memory, use_online_search)
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = pre_infos.is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_memory = current_memory
        decayed_commit = jnp.maximum(adapter_commit - 1, 0)
        next_adapter_commit = jnp.where(
            adapter_triggers > 0.0,
            jnp.asarray(policy_adapter_commit_steps, dtype=jnp.int32),
            decayed_commit,
        )
        adapter_stats = (
            adapter_triggers * active_decisions,
            adapter_used * active_decisions,
            adapter_action_diff * active_decisions,
            active_decisions,
        )
        return (final_states, key, current_scoreboard, final_memory, next_adapter_commit), (infos, adapter_stats)

    (states, key, _, _, _), (_, adapter_stats_steps) = jax.lax.scan(
        body,
        (states, key, initial_history, initial_fog_memory, initial_adapter_commit),
        None,
        length=max_steps,
    )
    adapter_stats = jax.tree.map(lambda value: jnp.sum(value), adapter_stats_steps)
    return jax.vmap(game.get_info)(states), adapter_stats


@eqx.filter_jit
def evaluate_policy_opponent_batch(
    network,
    policy_adapter_network,
    policy_adapter_feature_network,
    late_policy_adapter_network,
    plan_worker_network,
    command_gate_network,
    candidate_scorer_network,
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
    strategy_plan_worker_command_source=0,
    strategy_plan_worker_gate_threshold=-1.0,
    strategy_plan_worker_min_grid_size=0,
    strategy_plan_worker_max_grid_size=0,
    strategy_command_gate_threshold=-1.0,
    strategy_command_gate_source_count=1,
    strategy_command_gate_target_count=1,
    command_gate_feature_dim=COMMAND_GATE_FEATURE_DIM,
    policy_adapter_scale=0.0,
    conversion_policy_scale=0.0,
    conversion_policy_mode=0,
    policy_adapter_finish_threshold=-1.0,
    policy_adapter_gate_threshold=-1.0,
    policy_adapter_mode=0,
    late_policy_adapter_scale=0.0,
    late_policy_adapter_mode=0,
    late_policy_adapter_min_grid_size=0,
    late_policy_adapter_max_grid_size=0,
    late_policy_adapter_min_turn=0,
    late_policy_adapter_require_contact=False,
    policy_adapter_min_grid_size=0,
    policy_adapter_max_grid_size=0,
    policy_adapter_min_turn=0,
    policy_adapter_require_contact=False,
    policy_adapter_commit_steps=0,
    online_search_top_k=0,
    online_search_rollout_steps=16,
    online_search_rollouts_per_action=1,
    online_search_min_turn=0,
    online_search_require_contact=False,
    online_search_min_grid_size=0,
    online_search_max_grid_size=0,
    online_search_army_weight=1.0,
    online_search_land_weight=10.0,
    online_search_prior_weight=0.001,
    online_search_terminal_score=100.0,
    online_search_min_score_gap=0.0,
    online_search_gate_threshold=-1.0,
    online_search_gate_feature_dim=ONLINE_SEARCH_GATE_FEATURE_DIM,
    candidate_scorer_top_k=0,
    candidate_scorer_min_turn=0,
    candidate_scorer_require_contact=False,
    candidate_scorer_min_grid_size=0,
    candidate_scorer_max_grid_size=0,
    candidate_scorer_min_score_gap=0.0,
    candidate_scorer_local_channels=0,
):
    """Evaluate one adaptive checkpoint against one fixed-size PPO checkpoint."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)
    initial_history = jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)
    initial_fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)
    plan_worker_size_allowed = (
        (strategy_plan_worker_min_grid_size <= 0 or effective_size >= strategy_plan_worker_min_grid_size)
        and (strategy_plan_worker_max_grid_size <= 0 or effective_size <= strategy_plan_worker_max_grid_size)
    )
    effective_plan_worker_rerank_scale = (
        strategy_plan_worker_rerank_scale if plan_worker_size_allowed else 0.0
    )
    effective_plan_worker_gate_threshold = (
        strategy_plan_worker_gate_threshold if plan_worker_size_allowed else -1.0
    )
    policy_adapter_size_allowed = (
        (policy_adapter_min_grid_size <= 0 or effective_size >= policy_adapter_min_grid_size)
        and (policy_adapter_max_grid_size <= 0 or effective_size <= policy_adapter_max_grid_size)
    )
    size_policy_adapter_scale = policy_adapter_scale if policy_adapter_size_allowed else 0.0
    late_policy_adapter_size_allowed = (
        (late_policy_adapter_min_grid_size <= 0 or effective_size >= late_policy_adapter_min_grid_size)
        and (late_policy_adapter_max_grid_size <= 0 or effective_size <= late_policy_adapter_max_grid_size)
    )
    size_late_policy_adapter_scale = late_policy_adapter_scale if late_policy_adapter_size_allowed else 0.0
    online_search_size_allowed = (
        (online_search_min_grid_size <= 0 or effective_size >= online_search_min_grid_size)
        and (online_search_max_grid_size <= 0 or effective_size <= online_search_max_grid_size)
    )
    online_search_enabled = online_search_top_k > 0 and online_search_size_allowed
    candidate_scorer_size_allowed = (
        (candidate_scorer_min_grid_size <= 0 or effective_size >= candidate_scorer_min_grid_size)
        and (candidate_scorer_max_grid_size <= 0 or effective_size <= candidate_scorer_max_grid_size)
    )
    candidate_scorer_enabled = candidate_scorer_top_k > 0 and candidate_scorer_size_allowed
    initial_adapter_commit = jnp.zeros((num_envs,), dtype=jnp.int32)

    def body(carry, _):
        states, key, history, memory, adapter_commit = carry
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
        pre_infos = jax.vmap(game.get_info)(states)
        active_decisions = (~pre_infos.is_done).astype(jnp.float32)
        adapter_visible_contact = jnp.sum(policy_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
        adapter_turn_allowed = states.time >= policy_adapter_min_turn
        adapter_contact_allowed = adapter_visible_contact | (not policy_adapter_require_contact)
        row_policy_adapter_allowed = adapter_turn_allowed & adapter_contact_allowed
        late_adapter_turn_allowed = states.time >= late_policy_adapter_min_turn
        late_adapter_contact_allowed = adapter_visible_contact | (not late_policy_adapter_require_contact)
        row_late_policy_adapter_allowed = late_adapter_turn_allowed & late_adapter_contact_allowed
        opponent_actions = jax.vmap(
            lambda k, obs: policy_network_action(
                opponent_network,
                k,
                crop_observation(obs, effective_size),
                opponent_policy_mode,
            )
        )(opponent_keys, opponent_obs)
        policy_actions, adapter_triggers, adapter_used, adapter_action_diff = jax.vmap(
            lambda o, m, a, k, c, adapter_allowed, late_adapter_allowed: _policy_action(
                network,
                policy_adapter_network,
                policy_adapter_feature_network,
                late_policy_adapter_network,
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
                effective_plan_worker_rerank_scale,
                strategy_plan_worker_min_margin,
                strategy_plan_worker_command_source,
                effective_plan_worker_gate_threshold,
                strategy_command_gate_threshold,
                strategy_command_gate_source_count,
                strategy_command_gate_target_count,
                command_gate_feature_dim,
                size_policy_adapter_scale,
                conversion_policy_scale if policy_adapter_size_allowed else 0.0,
                conversion_policy_mode,
                policy_adapter_finish_threshold,
                policy_adapter_gate_threshold,
                policy_adapter_mode,
                size_late_policy_adapter_scale,
                late_policy_adapter_mode,
                c,
                adapter_allowed,
                late_adapter_allowed,
            )
        )(
            obs_arr,
            masks,
            active,
            policy_keys,
            adapter_commit,
            row_policy_adapter_allowed,
            row_late_policy_adapter_allowed,
        )
        if candidate_scorer_enabled:
            visible_contact = jnp.sum(policy_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
            scorer_turn_allowed = states.time >= candidate_scorer_min_turn
            scorer_contact_allowed = visible_contact | (not candidate_scorer_require_contact)
            use_candidate_scorer = (~pre_infos.is_done) & scorer_turn_allowed & scorer_contact_allowed
            policy_actions = jax.vmap(
                lambda o, m, a, base_action, turn, use_scorer: jax.lax.cond(
                    use_scorer,
                    lambda _: candidate_scorer_action(
                        network,
                        policy_adapter_network,
                        candidate_scorer_network,
                        o,
                        m,
                        a,
                        base_action,
                        effective_size,
                        turn,
                        policy_player,
                        max_steps,
                        pad_size,
                        size_policy_adapter_scale,
                        policy_adapter_mode,
                        policy_adapter_min_grid_size,
                        policy_adapter_max_grid_size,
                        candidate_scorer_top_k,
                        candidate_scorer_min_score_gap,
                        candidate_scorer_local_channels,
                    ),
                    lambda _: base_action,
                    None,
                )
            )(obs_arr, masks, active, policy_actions, states.time, use_candidate_scorer)
        if online_search_enabled:
            key, search_key = jrandom.split(key)
            search_keys = jrandom.split(search_key, num_envs)
            visible_contact = jnp.sum(policy_obs.opponent_cells.reshape(num_envs, -1), axis=-1) > 0
            search_turn_allowed = states.time >= online_search_min_turn
            search_contact_allowed = visible_contact | (not online_search_require_contact)
            use_online_search = (~pre_infos.is_done) & search_turn_allowed & search_contact_allowed
            policy_actions = jax.vmap(
                lambda state, sample_key, base_action, opponent_action_value, row_history, row_memory, use_search: jax.lax.cond(
                    use_search,
                    lambda _: online_search_action_policy_opponent(
                        network,
                        policy_adapter_network,
                        command_gate_network,
                        opponent_network,
                        state,
                        effective_size,
                        sample_key,
                        base_action,
                        opponent_action_value,
                        policy_player,
                        policy_mode,
                        opponent_policy_mode,
                        pad_size,
                        max_steps,
                        global_context,
                        scoreboard_history,
                        row_history,
                        fog_memory,
                        row_memory,
                        size_policy_adapter_scale,
                        policy_adapter_mode,
                        policy_adapter_min_grid_size,
                        policy_adapter_max_grid_size,
                        online_search_top_k,
                        online_search_rollout_steps,
                        online_search_rollouts_per_action,
                        online_search_army_weight,
                        online_search_land_weight,
                        online_search_prior_weight,
                        online_search_terminal_score,
                        online_search_min_score_gap,
                        online_search_gate_threshold,
                        online_search_gate_feature_dim,
                    ),
                    lambda _: base_action,
                    None,
                )
            )(states, search_keys, policy_actions, opponent_actions, history, current_memory, use_online_search)
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = pre_infos.is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_memory = current_memory
        decayed_commit = jnp.maximum(adapter_commit - 1, 0)
        next_adapter_commit = jnp.where(
            adapter_triggers > 0.0,
            jnp.asarray(policy_adapter_commit_steps, dtype=jnp.int32),
            decayed_commit,
        )
        adapter_stats = (
            adapter_triggers * active_decisions,
            adapter_used * active_decisions,
            adapter_action_diff * active_decisions,
            active_decisions,
        )
        return (final_states, key, current_scoreboard, final_memory, next_adapter_commit), (infos, adapter_stats)

    (states, key, _, _, _), (_, adapter_stats_steps) = jax.lax.scan(
        body,
        (states, key, initial_history, initial_fog_memory, initial_adapter_commit),
        None,
        length=max_steps,
    )
    adapter_stats = jax.tree.map(lambda value: jnp.sum(value), adapter_stats_steps)
    return jax.vmap(game.get_info)(states), adapter_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an adaptive multisize PPO checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument(
        "--policy-players",
        default="0,1",
        help="Comma-separated policy seats to evaluate. Defaults to both seats: 0,1.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="If positive, split each size/seat row into this many games per JAX evaluation batch.",
    )
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
    parser.add_argument("--init-value-heads", choices=("shared", "per-size"), default=None)
    parser.add_argument("--value-head-sizes", default=None)
    parser.add_argument("--init-value-head-sizes", default=None)
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--init-value-loss", choices=("mse", "hl-gauss"), default=None)
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--init-value-bins", type=int, default=None)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--value-sigma", type=float, default=0.04)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--init-outcome-head", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--conversion-policy-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load a checkpoint with an auxiliary conversion/planner policy head.",
    )
    parser.add_argument("--init-conversion-policy-head", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument("--init-strategy-aux", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-strategy-spatial-aux", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strategy-finish-outputs", type=int, default=2)
    parser.add_argument("--init-strategy-finish-outputs", type=int, default=None)
    parser.add_argument(
        "--drop-mismatched-init-leaves",
        action="store_true",
        help="Load matching checkpoint leaves and reinitialize shape-mismatched legacy leaves.",
    )
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
    parser.add_argument(
        "--strategy-plan-worker-min-grid-size",
        type=int,
        default=0,
        help="If positive, only enable Plan-Worker inference on grid sizes at least this value.",
    )
    parser.add_argument(
        "--strategy-plan-worker-max-grid-size",
        type=int,
        default=0,
        help="If positive, only enable Plan-Worker inference on grid sizes up to this value.",
    )
    parser.add_argument("--strategy-plan-worker-gate-path", default=None)
    parser.add_argument("--strategy-plan-worker-gate-threshold", type=float, default=-1.0)
    parser.add_argument("--strategy-plan-worker-gate-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--strategy-plan-worker-command-source",
        choices=PLAN_WORKER_COMMAND_SOURCE_NAMES,
        default="spatial",
        help="Command source for learned Plan-Worker inference.",
    )
    parser.add_argument("--strategy-command-gate-path", default=None)
    parser.add_argument("--strategy-command-gate-threshold", type=float, default=-1.0)
    parser.add_argument("--strategy-command-gate-hidden-dim", type=int, default=32)
    parser.add_argument("--strategy-command-gate-source-count", type=int, default=1)
    parser.add_argument("--strategy-command-gate-target-count", type=int, default=1)
    parser.add_argument("--policy-adapter-path", default=None)
    parser.add_argument(
        "--policy-adapter-feature-model-path",
        default=None,
        help="Optional strategy-aux model used only for policy-adapter gate/finish features.",
    )
    parser.add_argument(
        "--policy-adapter-feature-outcome-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Outcome-head schema for --policy-adapter-feature-model-path. Defaults to --outcome-head.",
    )
    parser.add_argument(
        "--policy-adapter-feature-strategy-aux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Strategy-aux schema for --policy-adapter-feature-model-path. Defaults to --strategy-aux.",
    )
    parser.add_argument(
        "--policy-adapter-feature-strategy-spatial-aux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Spatial strategy-aux schema for --policy-adapter-feature-model-path. Defaults to --strategy-spatial-aux.",
    )
    parser.add_argument(
        "--policy-adapter-feature-strategy-finish-outputs",
        type=int,
        default=None,
        help="Finish-output count for --policy-adapter-feature-model-path. Defaults to --strategy-finish-outputs.",
    )
    parser.add_argument("--policy-adapter-scale", type=float, default=0.0)
    parser.add_argument("--policy-adapter-mode", choices=POLICY_ADAPTER_MODE_NAMES, default="delta")
    parser.add_argument(
        "--late-policy-adapter-path",
        default=None,
        help="Optional second policy adapter composed after the primary adapter.",
    )
    parser.add_argument("--late-policy-adapter-scale", type=float, default=0.0)
    parser.add_argument("--late-policy-adapter-mode", choices=POLICY_ADAPTER_MODE_NAMES, default="delta")
    parser.add_argument(
        "--conversion-policy-scale",
        type=float,
        default=0.0,
        help="Compose the main checkpoint's conversion policy head before any separate policy adapter.",
    )
    parser.add_argument("--conversion-policy-mode", choices=POLICY_ADAPTER_MODE_NAMES, default="delta")
    parser.add_argument("--policy-adapter-finish-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--policy-adapter-min-grid-size",
        type=int,
        default=0,
        help="If positive, only enable Policy Adapter inference on grid sizes at least this value.",
    )
    parser.add_argument(
        "--policy-adapter-max-grid-size",
        type=int,
        default=0,
        help="If positive, only enable Policy Adapter inference on grid sizes up to this value.",
    )
    parser.add_argument(
        "--policy-adapter-min-turn",
        type=int,
        default=0,
        help="Only enable Policy Adapter inference at or after this game turn.",
    )
    parser.add_argument(
        "--policy-adapter-require-contact",
        action="store_true",
        help="Only enable Policy Adapter inference when the learner currently sees an enemy cell.",
    )
    parser.add_argument("--policy-adapter-gate-path", default=None)
    parser.add_argument("--policy-adapter-gate-threshold", type=float, default=-1.0)
    parser.add_argument("--policy-adapter-gate-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--policy-adapter-commit-steps",
        type=int,
        default=0,
        help="After an adapter gate/finish trigger, force the adapter for this many following policy turns.",
    )
    parser.add_argument(
        "--late-policy-adapter-min-grid-size",
        type=int,
        default=0,
        help="If positive, only enable the late Policy Adapter on grid sizes at least this value.",
    )
    parser.add_argument(
        "--late-policy-adapter-max-grid-size",
        type=int,
        default=0,
        help="If positive, only enable the late Policy Adapter on grid sizes up to this value.",
    )
    parser.add_argument(
        "--late-policy-adapter-min-turn",
        type=int,
        default=0,
        help="Only enable the late Policy Adapter at or after this game turn.",
    )
    parser.add_argument(
        "--late-policy-adapter-require-contact",
        action="store_true",
        help="Only enable the late Policy Adapter when the learner currently sees an enemy cell.",
    )
    parser.add_argument(
        "--online-search-top-k",
        type=int,
        default=0,
        help="If positive, replace the policy action with online rollout search over the top-k prior actions.",
    )
    parser.add_argument("--online-search-rollout-steps", type=int, default=16)
    parser.add_argument("--online-search-rollouts-per-action", type=int, default=1)
    parser.add_argument("--online-search-min-turn", type=int, default=0)
    parser.add_argument("--online-search-require-contact", action="store_true")
    parser.add_argument("--online-search-min-grid-size", type=int, default=0)
    parser.add_argument("--online-search-max-grid-size", type=int, default=0)
    parser.add_argument("--online-search-army-weight", type=float, default=1.0)
    parser.add_argument("--online-search-land-weight", type=float, default=10.0)
    parser.add_argument("--online-search-prior-weight", type=float, default=0.001)
    parser.add_argument("--online-search-terminal-score", type=float, default=100.0)
    parser.add_argument(
        "--online-search-min-score-gap",
        type=float,
        default=0.0,
        help="If positive, execute the online-search action only when best-minus-second rollout score gap clears it.",
    )
    parser.add_argument("--online-search-gate-path", default=None)
    parser.add_argument("--online-search-gate-threshold", type=float, default=-1.0)
    parser.add_argument("--online-search-gate-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--candidate-scorer-path",
        default=None,
        help="Optional offline online-search candidate scorer used as a cheap top-k action selector.",
    )
    parser.add_argument(
        "--candidate-scorer-top-k",
        type=int,
        default=0,
        help="If positive, score this many prior candidates with --candidate-scorer-path.",
    )
    parser.add_argument("--candidate-scorer-min-turn", type=int, default=0)
    parser.add_argument("--candidate-scorer-require-contact", action="store_true")
    parser.add_argument("--candidate-scorer-min-grid-size", type=int, default=0)
    parser.add_argument("--candidate-scorer-max-grid-size", type=int, default=0)
    parser.add_argument(
        "--candidate-scorer-min-score-gap",
        type=float,
        default=0.0,
        help="If positive, keep the base action unless scorer best-minus-second clears this margin.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--require-win-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.policy_players = parse_policy_players(args.policy_players)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.value_head_sizes = (
            parse_grid_sizes(args.value_head_sizes) if args.value_head_sizes is not None else args.grid_sizes
        )
        args.init_value_head_sizes = (
            parse_grid_sizes(args.init_value_head_sizes) if args.init_value_head_sizes is not None else args.value_head_sizes
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.init_value_heads is None:
        args.init_value_heads = args.value_heads
    if args.init_value_loss is None:
        args.init_value_loss = args.value_loss
    if args.init_outcome_head is None:
        args.init_outcome_head = args.outcome_head
    if args.init_strategy_aux is None:
        args.init_strategy_aux = args.strategy_aux
    if args.init_strategy_spatial_aux is None:
        args.init_strategy_spatial_aux = args.strategy_spatial_aux
    if args.init_strategy_finish_outputs is None:
        args.init_strategy_finish_outputs = args.strategy_finish_outputs
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.eval_batch_size < 0:
        parser.error("--eval-batch-size must be non-negative")
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
    if args.init_value_loss == "hl-gauss":
        init_bins = args.value_bins if args.init_value_bins is None else args.init_value_bins
        if init_bins <= 1:
            parser.error("--init-value-bins must be greater than 1 for --init-value-loss hl-gauss")
    elif args.init_value_bins is not None:
        parser.error("--init-value-bins requires --init-value-loss hl-gauss")
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
    if args.strategy_finish_outputs <= 0:
        parser.error("--strategy-finish-outputs must be positive")
    if args.init_strategy_finish_outputs <= 0:
        parser.error("--init-strategy-finish-outputs must be positive")
    if args.strategy_finish_outputs != 2 and (args.strategy_target_finish_gate or args.strategy_worker_finish_gate):
        parser.error("finish-gated rerank currently expects a 2-logit binary finish head")
    if args.strategy_worker_policy_margin < 0.0 and args.strategy_worker_policy_margin != -1.0:
        parser.error("--strategy-worker-policy-margin must be non-negative, or -1 to disable")
    if args.strategy_plan_worker_rerank_scale < 0.0:
        parser.error("--strategy-plan-worker-rerank-scale must be non-negative")
    if args.strategy_plan_worker_gate_threshold < 0.0 and args.strategy_plan_worker_gate_threshold != -1.0:
        parser.error("--strategy-plan-worker-gate-threshold must be between 0 and 1, or -1 to disable")
    if args.strategy_plan_worker_gate_threshold > 1.0:
        parser.error("--strategy-plan-worker-gate-threshold must be between 0 and 1")
    plan_worker_active = (
        args.strategy_plan_worker_rerank_scale > 0.0 or args.strategy_plan_worker_gate_threshold >= 0.0
    )
    if plan_worker_active and args.strategy_plan_worker_path is None:
        parser.error("Plan-Worker inference requires --strategy-plan-worker-path")
    if (
        plan_worker_active
        and not args.strategy_aux
        and (
            args.strategy_plan_worker_command_source != "main-stack-heuristic"
            or args.strategy_plan_worker_gate_threshold >= 0.0
        )
    ):
        parser.error("Plan-Worker inference requires --strategy-aux for this command source or gate")
    if (
        plan_worker_active
        and args.strategy_plan_worker_command_source == "spatial"
        and not args.strategy_spatial_aux
    ):
        parser.error("--strategy-plan-worker-command-source spatial requires --strategy-spatial-aux")
    if args.strategy_plan_worker_min_margin < 0.0 and args.strategy_plan_worker_min_margin != -1.0:
        parser.error("--strategy-plan-worker-min-margin must be non-negative, or -1 to disable")
    if args.strategy_plan_worker_min_margin >= 0.0 and args.strategy_plan_worker_rerank_scale <= 0.0:
        parser.error("--strategy-plan-worker-min-margin requires --strategy-plan-worker-rerank-scale")
    if args.strategy_plan_worker_min_grid_size < 0:
        parser.error("--strategy-plan-worker-min-grid-size must be non-negative")
    if args.strategy_plan_worker_max_grid_size < 0:
        parser.error("--strategy-plan-worker-max-grid-size must be non-negative")
    if args.strategy_plan_worker_gate_threshold >= 0.0 and args.strategy_plan_worker_gate_path is None:
        parser.error("--strategy-plan-worker-gate-threshold requires --strategy-plan-worker-gate-path")
    if args.strategy_plan_worker_gate_hidden_dim <= 0:
        parser.error("--strategy-plan-worker-gate-hidden-dim must be positive")
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
    if args.strategy_command_gate_source_count <= 0 or args.strategy_command_gate_target_count <= 0:
        parser.error("--strategy-command-gate-source-count and --strategy-command-gate-target-count must be positive")
    if (
        (args.strategy_command_gate_source_count > 1 or args.strategy_command_gate_target_count > 1)
        and args.strategy_command_gate_threshold < 0.0
    ):
        parser.error("multi-command gate counts require --strategy-command-gate-threshold")
    if (
        args.strategy_command_gate_threshold >= 0.0
        and args.strategy_plan_worker_gate_threshold >= 0.0
    ):
        parser.error("Use either --strategy-command-gate-threshold or --strategy-plan-worker-gate-threshold, not both")
    if args.policy_adapter_scale < 0.0:
        parser.error("--policy-adapter-scale must be non-negative")
    if args.conversion_policy_scale < 0.0:
        parser.error("--conversion-policy-scale must be non-negative")
    if args.conversion_policy_scale > 0.0 and not args.conversion_policy_head:
        parser.error("--conversion-policy-scale requires --conversion-policy-head")
    if args.policy_adapter_scale > 0.0 and args.policy_adapter_path is None:
        parser.error("--policy-adapter-scale requires --policy-adapter-path")
    if args.policy_adapter_path is not None and args.policy_adapter_scale <= 0.0:
        parser.error("--policy-adapter-path requires --policy-adapter-scale > 0")
    if args.late_policy_adapter_scale < 0.0:
        parser.error("--late-policy-adapter-scale must be non-negative")
    if args.late_policy_adapter_scale > 0.0 and args.late_policy_adapter_path is None:
        parser.error("--late-policy-adapter-scale requires --late-policy-adapter-path")
    if args.late_policy_adapter_path is not None and args.late_policy_adapter_scale <= 0.0:
        parser.error("--late-policy-adapter-path requires --late-policy-adapter-scale > 0")
    if args.late_policy_adapter_min_grid_size < 0 or args.late_policy_adapter_max_grid_size < 0:
        parser.error("--late-policy-adapter-min/max-grid-size must be non-negative")
    if (
        args.late_policy_adapter_min_grid_size > 0
        and args.late_policy_adapter_max_grid_size > 0
        and args.late_policy_adapter_min_grid_size > args.late_policy_adapter_max_grid_size
    ):
        parser.error("--late-policy-adapter-min-grid-size must be <= --late-policy-adapter-max-grid-size")
    if args.late_policy_adapter_min_turn < 0:
        parser.error("--late-policy-adapter-min-turn must be non-negative")
    if args.policy_adapter_feature_model_path is not None and args.policy_adapter_path is None:
        parser.error("--policy-adapter-feature-model-path requires --policy-adapter-path")
    policy_adapter_feature_strategy_aux = (
        args.strategy_aux if args.policy_adapter_feature_strategy_aux is None else args.policy_adapter_feature_strategy_aux
    )
    policy_adapter_feature_strategy_spatial_aux = (
        args.strategy_spatial_aux
        if args.policy_adapter_feature_strategy_spatial_aux is None
        else args.policy_adapter_feature_strategy_spatial_aux
    )
    if (
        args.policy_adapter_feature_strategy_finish_outputs is not None
        and args.policy_adapter_feature_strategy_finish_outputs <= 0
    ):
        parser.error("--policy-adapter-feature-strategy-finish-outputs must be positive")
    if policy_adapter_feature_strategy_spatial_aux and not policy_adapter_feature_strategy_aux:
        parser.error("--policy-adapter-feature-strategy-spatial-aux requires feature strategy aux")
    if args.policy_adapter_finish_threshold < 0.0 and args.policy_adapter_finish_threshold != -1.0:
        parser.error("--policy-adapter-finish-threshold must be between 0 and 1, or -1 to disable")
    if args.policy_adapter_finish_threshold > 1.0:
        parser.error("--policy-adapter-finish-threshold must be between 0 and 1")
    if args.policy_adapter_finish_threshold >= 0.0 and not args.strategy_aux:
        parser.error("--policy-adapter-finish-threshold requires --strategy-aux")
    if args.policy_adapter_finish_threshold >= 0.0 and args.policy_adapter_scale <= 0.0:
        parser.error("--policy-adapter-finish-threshold requires --policy-adapter-scale > 0")
    if args.policy_adapter_gate_threshold < 0.0 and args.policy_adapter_gate_threshold != -1.0:
        parser.error("--policy-adapter-gate-threshold must be between 0 and 1, or -1 to disable")
    if args.policy_adapter_gate_threshold > 1.0:
        parser.error("--policy-adapter-gate-threshold must be between 0 and 1")
    if args.policy_adapter_gate_threshold >= 0.0 and args.policy_adapter_gate_path is None:
        parser.error("--policy-adapter-gate-threshold requires --policy-adapter-gate-path")
    if args.policy_adapter_gate_threshold >= 0.0 and args.policy_adapter_scale <= 0.0:
        parser.error("--policy-adapter-gate-threshold requires --policy-adapter-scale > 0")
    if (
        args.policy_adapter_gate_threshold >= 0.0
        and not args.strategy_aux
        and not (args.policy_adapter_feature_model_path is not None and policy_adapter_feature_strategy_aux)
    ):
        parser.error("--policy-adapter-gate-threshold requires --strategy-aux or a strategy-aux feature model")
    if args.policy_adapter_gate_threshold >= 0.0 and args.policy_adapter_finish_threshold >= 0.0:
        parser.error("Use either --policy-adapter-gate-threshold or --policy-adapter-finish-threshold")
    if args.policy_adapter_gate_threshold >= 0.0 and (
        args.strategy_command_gate_threshold >= 0.0 or args.strategy_plan_worker_gate_threshold >= 0.0
    ):
        parser.error("Use only one learned gate type per evaluation command")
    if args.policy_adapter_gate_path is not None and args.policy_adapter_gate_threshold < 0.0:
        parser.error("--policy-adapter-gate-path requires --policy-adapter-gate-threshold")
    if args.policy_adapter_gate_hidden_dim <= 0:
        parser.error("--policy-adapter-gate-hidden-dim must be positive")
    if args.policy_adapter_commit_steps < 0:
        parser.error("--policy-adapter-commit-steps must be non-negative")
    if args.policy_adapter_commit_steps > 0 and args.policy_adapter_gate_threshold < 0.0 and args.policy_adapter_finish_threshold < 0.0:
        parser.error("--policy-adapter-commit-steps requires a policy-adapter gate or finish threshold")
    if args.policy_adapter_min_grid_size < 0 or args.policy_adapter_max_grid_size < 0:
        parser.error("--policy-adapter-min-grid-size/max-grid-size must be non-negative")
    if (
        args.policy_adapter_min_grid_size > 0
        and args.policy_adapter_max_grid_size > 0
        and args.policy_adapter_min_grid_size > args.policy_adapter_max_grid_size
    ):
        parser.error("--policy-adapter-min-grid-size must be <= --policy-adapter-max-grid-size")
    if args.policy_adapter_min_turn < 0:
        parser.error("--policy-adapter-min-turn must be non-negative")
    if args.online_search_top_k < 0:
        parser.error("--online-search-top-k must be non-negative")
    if args.online_search_top_k > 0 and args.policy_adapter_gate_threshold >= 0.0:
        parser.error("--online-search-top-k currently supports ungated policy adapters only")
    if args.online_search_top_k > 0 and args.policy_adapter_finish_threshold >= 0.0:
        parser.error("--online-search-top-k currently supports ungated policy adapters only")
    if args.online_search_top_k > 0 and args.policy_adapter_commit_steps > 0:
        parser.error("--online-search-top-k currently does not support policy-adapter commit state")
    if args.online_search_rollout_steps <= 0:
        parser.error("--online-search-rollout-steps must be positive")
    if args.online_search_rollouts_per_action <= 0:
        parser.error("--online-search-rollouts-per-action must be positive")
    if args.online_search_min_turn < 0:
        parser.error("--online-search-min-turn must be non-negative")
    if args.online_search_min_score_gap < 0.0:
        parser.error("--online-search-min-score-gap must be non-negative")
    if args.online_search_gate_threshold < 0.0 and args.online_search_gate_threshold != -1.0:
        parser.error("--online-search-gate-threshold must be between 0 and 1, or -1 to disable")
    if args.online_search_gate_threshold > 1.0:
        parser.error("--online-search-gate-threshold must be between 0 and 1")
    if args.online_search_gate_threshold >= 0.0 and args.online_search_gate_path is None:
        parser.error("--online-search-gate-threshold requires --online-search-gate-path")
    if args.online_search_gate_path is not None and args.online_search_gate_threshold < 0.0:
        parser.error("--online-search-gate-path requires --online-search-gate-threshold")
    if args.online_search_gate_threshold >= 0.0 and args.online_search_top_k <= 0:
        parser.error("--online-search-gate-threshold requires --online-search-top-k > 0")
    if args.online_search_gate_hidden_dim <= 0:
        parser.error("--online-search-gate-hidden-dim must be positive")
    if args.candidate_scorer_top_k < 0:
        parser.error("--candidate-scorer-top-k must be non-negative")
    if args.candidate_scorer_path is not None and args.candidate_scorer_top_k <= 0:
        parser.error("--candidate-scorer-path requires --candidate-scorer-top-k > 0")
    if args.candidate_scorer_top_k > 0 and args.candidate_scorer_path is None:
        parser.error("--candidate-scorer-top-k requires --candidate-scorer-path")
    if args.candidate_scorer_top_k > 0 and args.online_search_top_k > 0:
        parser.error("Use either --candidate-scorer-top-k or --online-search-top-k, not both")
    if args.candidate_scorer_top_k > ADAPTIVE_MOVE_PLANES * args.pad_to * args.pad_to + 1:
        parser.error("--candidate-scorer-top-k exceeds the adaptive action space")
    if args.candidate_scorer_top_k > 0 and args.policy_adapter_gate_threshold >= 0.0:
        parser.error("--candidate-scorer-top-k currently supports ungated policy adapters only")
    if args.candidate_scorer_top_k > 0 and args.policy_adapter_finish_threshold >= 0.0:
        parser.error("--candidate-scorer-top-k currently supports ungated policy adapters only")
    if args.candidate_scorer_top_k > 0 and args.policy_adapter_commit_steps > 0:
        parser.error("--candidate-scorer-top-k currently does not support policy-adapter commit state")
    if args.candidate_scorer_min_turn < 0:
        parser.error("--candidate-scorer-min-turn must be non-negative")
    if args.candidate_scorer_min_score_gap < 0.0:
        parser.error("--candidate-scorer-min-score-gap must be non-negative")
    if args.candidate_scorer_min_grid_size < 0 or args.candidate_scorer_max_grid_size < 0:
        parser.error("--candidate-scorer-min/max-grid-size must be non-negative")
    if (
        args.candidate_scorer_min_grid_size > 0
        and args.candidate_scorer_max_grid_size > 0
        and args.candidate_scorer_min_grid_size > args.candidate_scorer_max_grid_size
    ):
        parser.error("--candidate-scorer-min-grid-size must be <= --candidate-scorer-max-grid-size")
    learned_gate_count = sum(
        threshold >= 0.0
        for threshold in (
            args.strategy_command_gate_threshold,
            args.strategy_plan_worker_gate_threshold,
            args.policy_adapter_gate_threshold,
            args.online_search_gate_threshold,
        )
    )
    if learned_gate_count > 1:
        parser.error("Use only one learned gate type per evaluation command")
    if args.online_search_min_grid_size < 0 or args.online_search_max_grid_size < 0:
        parser.error("--online-search-min-grid-size/max-grid-size must be non-negative")
    if (
        args.online_search_min_grid_size > 0
        and args.online_search_max_grid_size > 0
        and args.online_search_min_grid_size > args.online_search_max_grid_size
    ):
        parser.error("--online-search-min-grid-size must be <= --online-search-max-grid-size")
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
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )
    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.model_path,
        channels=args.channels,
        input_channels=input_channels,
        init_input_channels=input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        init_value_head_sizes=args.init_value_head_sizes if args.init_value_heads == "per-size" else (),
        value_bins=value_bins,
        init_value_bins=init_value_bins,
        value_min=args.value_min,
        value_max=args.value_max,
        value_sigma=args.value_sigma,
        outcome_head=args.outcome_head,
        init_outcome_head=args.init_outcome_head,
        conversion_policy_head=args.conversion_policy_head,
        init_conversion_policy_head=args.init_conversion_policy_head,
        strategy_aux=args.strategy_aux,
        init_strategy_aux=args.init_strategy_aux,
        strategy_spatial_aux=args.strategy_spatial_aux,
        init_strategy_spatial_aux=args.init_strategy_spatial_aux,
        strategy_finish_outputs=args.strategy_finish_outputs,
        init_strategy_finish_outputs=args.init_strategy_finish_outputs,
        global_context=network_global_context,
        init_global_context=network_global_context,
        context_residual=args.context_residual,
        init_context_residual=args.context_residual,
        pyramid_context=args.pyramid_context,
        init_pyramid_context=args.pyramid_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
        drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
    )
    policy_adapter_network = None
    policy_adapter_feature_network = None
    policy_adapter_feature_outcome_head = (
        args.outcome_head
        if args.policy_adapter_feature_outcome_head is None
        else args.policy_adapter_feature_outcome_head
    )
    policy_adapter_feature_strategy_aux = (
        args.strategy_aux
        if args.policy_adapter_feature_strategy_aux is None
        else args.policy_adapter_feature_strategy_aux
    )
    policy_adapter_feature_strategy_spatial_aux = (
        args.strategy_spatial_aux
        if args.policy_adapter_feature_strategy_spatial_aux is None
        else args.policy_adapter_feature_strategy_spatial_aux
    )
    policy_adapter_feature_strategy_finish_outputs = (
        args.strategy_finish_outputs
        if args.policy_adapter_feature_strategy_finish_outputs is None
        else args.policy_adapter_feature_strategy_finish_outputs
    )
    if args.policy_adapter_path is not None:
        policy_adapter_network = load_or_create_adaptive_network(
            net_key,
            pad_size=args.pad_to,
            init_model_path=args.policy_adapter_path,
            channels=args.channels,
            input_channels=input_channels,
            init_input_channels=input_channels,
            value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
            value_bins=value_bins,
            value_min=args.value_min,
            value_max=args.value_max,
            value_sigma=args.value_sigma,
            outcome_head=args.outcome_head,
            strategy_aux=args.strategy_aux,
            strategy_spatial_aux=args.strategy_spatial_aux,
            strategy_finish_outputs=args.strategy_finish_outputs,
            init_strategy_finish_outputs=args.strategy_finish_outputs,
            global_context=network_global_context,
            init_global_context=network_global_context,
            context_residual=args.context_residual,
            init_context_residual=args.context_residual,
            pyramid_context=args.pyramid_context,
            init_pyramid_context=args.pyramid_context,
            network_arch=args.network_arch,
            init_network_arch=args.network_arch,
            drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
        )
        if args.policy_adapter_feature_model_path is not None:
            policy_adapter_feature_network = load_or_create_adaptive_network(
                net_key,
                pad_size=args.pad_to,
                init_model_path=args.policy_adapter_feature_model_path,
                channels=args.channels,
                input_channels=input_channels,
                init_input_channels=input_channels,
                value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
                value_bins=value_bins,
                value_min=args.value_min,
                value_max=args.value_max,
                value_sigma=args.value_sigma,
                outcome_head=policy_adapter_feature_outcome_head,
                strategy_aux=policy_adapter_feature_strategy_aux,
                strategy_spatial_aux=policy_adapter_feature_strategy_spatial_aux,
                strategy_finish_outputs=policy_adapter_feature_strategy_finish_outputs,
                init_strategy_finish_outputs=policy_adapter_feature_strategy_finish_outputs,
                global_context=network_global_context,
                init_global_context=network_global_context,
                context_residual=args.context_residual,
                init_context_residual=args.context_residual,
                pyramid_context=args.pyramid_context,
                init_pyramid_context=args.pyramid_context,
                network_arch=args.network_arch,
                init_network_arch=args.network_arch,
                drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
            )
    late_policy_adapter_network = None
    if args.late_policy_adapter_path is not None:
        late_policy_adapter_network = load_or_create_adaptive_network(
            net_key,
            pad_size=args.pad_to,
            init_model_path=args.late_policy_adapter_path,
            channels=args.channels,
            input_channels=input_channels,
            init_input_channels=input_channels,
            value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
            value_bins=value_bins,
            value_min=args.value_min,
            value_max=args.value_max,
            value_sigma=args.value_sigma,
            outcome_head=args.outcome_head,
            strategy_aux=args.strategy_aux,
            strategy_spatial_aux=args.strategy_spatial_aux,
            strategy_finish_outputs=args.strategy_finish_outputs,
            init_strategy_finish_outputs=args.strategy_finish_outputs,
            global_context=network_global_context,
            init_global_context=network_global_context,
            context_residual=args.context_residual,
            init_context_residual=args.context_residual,
            pyramid_context=args.pyramid_context,
            init_pyramid_context=args.pyramid_context,
            network_arch=args.network_arch,
            init_network_arch=args.network_arch,
            drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
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
    command_gate_feature_dim = COMMAND_GATE_FEATURE_DIM
    gate_path = (
        args.strategy_command_gate_path
        or args.strategy_plan_worker_gate_path
        or args.policy_adapter_gate_path
        or args.online_search_gate_path
    )
    if gate_path is not None:
        if args.strategy_plan_worker_gate_path is not None:
            gate_hidden_dim = args.strategy_plan_worker_gate_hidden_dim
        elif args.policy_adapter_gate_path is not None:
            gate_hidden_dim = args.policy_adapter_gate_hidden_dim
        elif args.online_search_gate_path is not None:
            gate_hidden_dim = args.online_search_gate_hidden_dim
        else:
            gate_hidden_dim = args.strategy_command_gate_hidden_dim
        gate_sidecar = Path(gate_path).with_suffix(".json")
        if gate_sidecar.exists():
            gate_metadata = json.loads(gate_sidecar.read_text(encoding="utf-8"))
            feature_names = gate_metadata.get("feature_names")
            if isinstance(feature_names, list) and feature_names:
                command_gate_feature_dim = len(feature_names)
        command_gate_network = CommandGateNetwork(net_key, input_dim=command_gate_feature_dim, hidden_dim=gate_hidden_dim)
        command_gate_network = eqx.tree_deserialise_leaves(gate_path, command_gate_network)
    candidate_scorer_network = None
    candidate_scorer_local_channels = 0
    candidate_scorer_features_used: list[str] = []
    if args.candidate_scorer_path is not None:
        candidate_scorer_network, candidate_scorer_local_channels, candidate_scorer_features_used = (
            load_candidate_scorer(args.candidate_scorer_path, seed=args.seed + 707)
        )
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
    policy_adapter_mode = POLICY_ADAPTER_MODE_TO_ID[args.policy_adapter_mode]
    late_policy_adapter_mode = POLICY_ADAPTER_MODE_TO_ID[args.late_policy_adapter_mode]
    conversion_policy_mode = POLICY_ADAPTER_MODE_TO_ID[args.conversion_policy_mode]
    opponent_policy_mode = 0 if args.opponent_policy_mode == "greedy" else 1
    rows = []

    print("Adaptive policy evaluation")
    print(f"Model:       {args.model_path}")
    print(f"Device:      {jax.devices()[0]}")
    print(f"Grid sizes:  {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Policy seats:{','.join(str(player) for player in args.policy_players)}")
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
    if args.conversion_policy_head:
        print("Conversion: policy head loaded")
    if args.conversion_policy_scale > 0.0:
        print(
            "Conversion: "
            f"mode={args.conversion_policy_mode}, scale={args.conversion_policy_scale:g} "
            "(uses policy-adapter size/turn/contact gates)"
        )
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
            f"arch={args.strategy_plan_worker_network_arch}, scale={args.strategy_plan_worker_rerank_scale:g}, "
            f"command={args.strategy_plan_worker_command_source}"
        )
        if args.strategy_plan_worker_min_margin >= 0.0:
            print(f"Plan worker: min_margin={args.strategy_plan_worker_min_margin:g}")
        if args.strategy_plan_worker_min_grid_size > 0:
            print(f"Plan worker: min_grid_size={args.strategy_plan_worker_min_grid_size}")
        if args.strategy_plan_worker_max_grid_size > 0:
            print(f"Plan worker: max_grid_size={args.strategy_plan_worker_max_grid_size}")
    if args.strategy_plan_worker_gate_threshold >= 0.0:
        print(f"Plan worker gate: {args.strategy_plan_worker_gate_path}")
        print(
            "Plan worker gate: "
            f"threshold={args.strategy_plan_worker_gate_threshold:g}, "
            f"hidden={args.strategy_plan_worker_gate_hidden_dim}, "
            f"command={args.strategy_plan_worker_command_source}"
        )
        if args.strategy_plan_worker_min_grid_size > 0:
            print(f"Plan worker gate: min_grid_size={args.strategy_plan_worker_min_grid_size}")
        if args.strategy_plan_worker_max_grid_size > 0:
            print(f"Plan worker gate: max_grid_size={args.strategy_plan_worker_max_grid_size}")
        print(f"Plan worker gate: feature_dim={command_gate_feature_dim}")
    if args.strategy_command_gate_threshold >= 0.0:
        print(f"Command gate: {args.strategy_command_gate_path}")
        print(
            "Command gate: "
            f"threshold={args.strategy_command_gate_threshold:g}, hidden={args.strategy_command_gate_hidden_dim}"
        )
        print(
            "Command gate: "
            f"candidates={args.strategy_command_gate_source_count}x{args.strategy_command_gate_target_count}"
        )
        print(f"Command gate: feature_dim={command_gate_feature_dim}")
    if args.policy_adapter_path is not None:
        if args.policy_adapter_gate_threshold >= 0.0:
            gate_label = f", learned-gate={args.policy_adapter_gate_threshold:g}"
        elif args.policy_adapter_finish_threshold >= 0.0:
            gate_label = f", finish-threshold={args.policy_adapter_finish_threshold:g}"
        else:
            gate_label = ""
        if args.policy_adapter_min_grid_size > 0 or args.policy_adapter_max_grid_size > 0:
            min_label = args.policy_adapter_min_grid_size if args.policy_adapter_min_grid_size > 0 else "-inf"
            max_label = args.policy_adapter_max_grid_size if args.policy_adapter_max_grid_size > 0 else "inf"
            gate_label += f", size=[{min_label},{max_label}]"
        if args.policy_adapter_min_turn > 0:
            gate_label += f", turn>={args.policy_adapter_min_turn}"
        if args.policy_adapter_require_contact:
            gate_label += ", contact"
        print(f"Policy adapter: {args.policy_adapter_path}")
        print(f"Policy adapter: mode={args.policy_adapter_mode}, scale={args.policy_adapter_scale:g}{gate_label}")
        if args.policy_adapter_feature_model_path is not None:
            print(f"Policy adapter features: {args.policy_adapter_feature_model_path}")
            print(
                "Policy adapter features: "
                f"outcome={policy_adapter_feature_outcome_head}, "
                f"strategy_aux={policy_adapter_feature_strategy_aux}, "
                f"spatial={policy_adapter_feature_strategy_spatial_aux}, "
                f"finish_outputs={policy_adapter_feature_strategy_finish_outputs}"
            )
        if args.policy_adapter_gate_threshold >= 0.0:
            print(f"Policy adapter gate: {args.policy_adapter_gate_path}")
            print(f"Policy adapter gate: feature_dim={command_gate_feature_dim}")
        if args.policy_adapter_commit_steps > 0:
            print(f"Policy adapter commit: {args.policy_adapter_commit_steps} steps")
    if args.late_policy_adapter_path is not None:
        gate_label = ""
        if args.late_policy_adapter_min_grid_size > 0 or args.late_policy_adapter_max_grid_size > 0:
            min_label = args.late_policy_adapter_min_grid_size if args.late_policy_adapter_min_grid_size > 0 else "-inf"
            max_label = args.late_policy_adapter_max_grid_size if args.late_policy_adapter_max_grid_size > 0 else "inf"
            gate_label += f", size=[{min_label},{max_label}]"
        if args.late_policy_adapter_min_turn > 0:
            gate_label += f", turn>={args.late_policy_adapter_min_turn}"
        if args.late_policy_adapter_require_contact:
            gate_label += ", contact"
        print(f"Late adapter: {args.late_policy_adapter_path}")
        print(
            f"Late adapter: mode={args.late_policy_adapter_mode}, "
            f"scale={args.late_policy_adapter_scale:g}{gate_label}"
        )
    if args.online_search_top_k > 0:
        if args.online_search_min_grid_size > 0 or args.online_search_max_grid_size > 0:
            min_label = args.online_search_min_grid_size if args.online_search_min_grid_size > 0 else "-inf"
            max_label = args.online_search_max_grid_size if args.online_search_max_grid_size > 0 else "inf"
            size_label = f", size=[{min_label},{max_label}]"
        else:
            size_label = ""
        contact_label = ", contact-only" if args.online_search_require_contact else ""
        print(
            "Online search: "
            f"top_k={args.online_search_top_k}, rollout_steps={args.online_search_rollout_steps}, "
            f"rollouts/action={args.online_search_rollouts_per_action}, min_turn={args.online_search_min_turn}"
            f", min_score_gap={args.online_search_min_score_gap:g}{contact_label}{size_label}"
        )
        if args.online_search_gate_threshold >= 0.0:
            print(f"Online search gate: {args.online_search_gate_path}")
            print(
                "Online search gate: "
                f"threshold={args.online_search_gate_threshold:g}, feature_dim={command_gate_feature_dim}"
            )
    if args.candidate_scorer_top_k > 0:
        if args.candidate_scorer_min_grid_size > 0 or args.candidate_scorer_max_grid_size > 0:
            min_label = args.candidate_scorer_min_grid_size if args.candidate_scorer_min_grid_size > 0 else "-inf"
            max_label = args.candidate_scorer_max_grid_size if args.candidate_scorer_max_grid_size > 0 else "inf"
            size_label = f", size=[{min_label},{max_label}]"
        else:
            size_label = ""
        contact_label = ", contact-only" if args.candidate_scorer_require_contact else ""
        print(f"Candidate scorer: {args.candidate_scorer_path}")
        print(
            "Candidate scorer: "
            f"top_k={args.candidate_scorer_top_k}, min_turn={args.candidate_scorer_min_turn}, "
            f"min_score_gap={args.candidate_scorer_min_score_gap:g}, "
            f"features={len(candidate_scorer_features_used)}, local_channels={candidate_scorer_local_channels}"
            f"{contact_label}{size_label}"
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
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.num_games
    if eval_batch_size < args.num_games:
        print(f"Eval batch: {eval_batch_size} games per compiled batch")
    print()

    for grid_size in args.grid_sizes:
        for policy_player in args.policy_players:
            t0 = time.time()
            wins = 0
            losses = 0
            draws = 0
            weighted_time = 0.0
            total_games = 0
            adapter_trigger_sum = 0.0
            adapter_used_sum = 0.0
            adapter_action_diff_sum = 0.0
            adapter_active_decision_sum = 0.0
            remaining_games = args.num_games

            while remaining_games > 0:
                chunk_games = min(eval_batch_size, remaining_games)
                key, pool_key, eval_key = jrandom.split(key, 3)
                pool = make_adaptive_state_pool(
                    pool_key,
                    chunk_games,
                    (grid_size,),
                    args.pad_to,
                    args.map_generator,
                    (args.mountain_density_min, args.mountain_density_max),
                    (args.num_cities_min, args.num_cities_max),
                    args.max_generals_distance,
                    (args.city_army_min, args.city_army_max),
                )
                states = pool.states
                if opponent_network is None:
                    info, adapter_stats = evaluate_batch(
                        network,
                        policy_adapter_network,
                        policy_adapter_feature_network,
                        late_policy_adapter_network,
                        plan_worker_network,
                        command_gate_network,
                        candidate_scorer_network,
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
                        PLAN_WORKER_COMMAND_SOURCE_TO_ID[args.strategy_plan_worker_command_source],
                        args.strategy_plan_worker_gate_threshold,
                        args.strategy_plan_worker_min_grid_size,
                        args.strategy_plan_worker_max_grid_size,
                        args.strategy_command_gate_threshold,
                        args.strategy_command_gate_source_count,
                        args.strategy_command_gate_target_count,
                        command_gate_feature_dim,
                        args.policy_adapter_scale,
                        args.conversion_policy_scale,
                        conversion_policy_mode,
                        args.policy_adapter_finish_threshold,
                        args.policy_adapter_gate_threshold,
                        policy_adapter_mode,
                        args.late_policy_adapter_scale,
                        late_policy_adapter_mode,
                        args.late_policy_adapter_min_grid_size,
                        args.late_policy_adapter_max_grid_size,
                        args.late_policy_adapter_min_turn,
                        args.late_policy_adapter_require_contact,
                        args.policy_adapter_min_grid_size,
                        args.policy_adapter_max_grid_size,
                        args.policy_adapter_min_turn,
                        args.policy_adapter_require_contact,
                        args.policy_adapter_commit_steps,
                        args.online_search_top_k,
                        args.online_search_rollout_steps,
                        args.online_search_rollouts_per_action,
                        args.online_search_min_turn,
                        args.online_search_require_contact,
                        args.online_search_min_grid_size,
                        args.online_search_max_grid_size,
                        args.online_search_army_weight,
                        args.online_search_land_weight,
                        args.online_search_prior_weight,
                        args.online_search_terminal_score,
                        args.online_search_min_score_gap,
                        args.online_search_gate_threshold,
                        command_gate_feature_dim,
                        args.candidate_scorer_top_k,
                        args.candidate_scorer_min_turn,
                        args.candidate_scorer_require_contact,
                        args.candidate_scorer_min_grid_size,
                        args.candidate_scorer_max_grid_size,
                        args.candidate_scorer_min_score_gap,
                        candidate_scorer_local_channels,
                    )
                else:
                    info, adapter_stats = evaluate_policy_opponent_batch(
                        network,
                        policy_adapter_network,
                        policy_adapter_feature_network,
                        late_policy_adapter_network,
                        plan_worker_network,
                        command_gate_network,
                        candidate_scorer_network,
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
                        PLAN_WORKER_COMMAND_SOURCE_TO_ID[args.strategy_plan_worker_command_source],
                        args.strategy_plan_worker_gate_threshold,
                        args.strategy_plan_worker_min_grid_size,
                        args.strategy_plan_worker_max_grid_size,
                        args.strategy_command_gate_threshold,
                        args.strategy_command_gate_source_count,
                        args.strategy_command_gate_target_count,
                        command_gate_feature_dim,
                        args.policy_adapter_scale,
                        args.conversion_policy_scale,
                        conversion_policy_mode,
                        args.policy_adapter_finish_threshold,
                        args.policy_adapter_gate_threshold,
                        policy_adapter_mode,
                        args.late_policy_adapter_scale,
                        late_policy_adapter_mode,
                        args.late_policy_adapter_min_grid_size,
                        args.late_policy_adapter_max_grid_size,
                        args.late_policy_adapter_min_turn,
                        args.late_policy_adapter_require_contact,
                        args.policy_adapter_min_grid_size,
                        args.policy_adapter_max_grid_size,
                        args.policy_adapter_min_turn,
                        args.policy_adapter_require_contact,
                        args.policy_adapter_commit_steps,
                        args.online_search_top_k,
                        args.online_search_rollout_steps,
                        args.online_search_rollouts_per_action,
                        args.online_search_min_turn,
                        args.online_search_require_contact,
                        args.online_search_min_grid_size,
                        args.online_search_max_grid_size,
                        args.online_search_army_weight,
                        args.online_search_land_weight,
                        args.online_search_prior_weight,
                        args.online_search_terminal_score,
                        args.online_search_min_score_gap,
                        args.online_search_gate_threshold,
                        command_gate_feature_dim,
                        args.candidate_scorer_top_k,
                        args.candidate_scorer_min_turn,
                        args.candidate_scorer_require_contact,
                        args.candidate_scorer_min_grid_size,
                        args.candidate_scorer_max_grid_size,
                        args.candidate_scorer_min_score_gap,
                        candidate_scorer_local_channels,
                    )
                jax.block_until_ready(info.winner)
                row_jax = summarize_row(info, grid_size, policy_player, chunk_games, adapter_stats)
                wins += int(row_jax.wins)
                losses += int(row_jax.losses)
                draws += int(row_jax.draws)
                weighted_time += float(row_jax.mean_time) * chunk_games
                total_games += chunk_games
                if adapter_stats is not None:
                    adapter_trigger_sum += float(adapter_stats[0])
                    adapter_used_sum += float(adapter_stats[1])
                    adapter_action_diff_sum += float(adapter_stats[2])
                    adapter_active_decision_sum += float(adapter_stats[3])
                remaining_games -= chunk_games

            adapter_denominator = max(adapter_active_decision_sum, 1.0)
            row = AdaptiveEvalRow(
                grid_size=grid_size,
                policy_player=policy_player,
                wins=wins,
                losses=losses,
                draws=draws,
                num_games=total_games,
                mean_time=weighted_time / total_games,
                adapter_trigger_rate=adapter_trigger_sum / adapter_denominator,
                adapter_used_rate=adapter_used_sum / adapter_denominator,
                adapter_action_diff_rate=adapter_action_diff_sum / adapter_denominator,
            )
            rows.append(row)
            elapsed = time.time() - t0
            adapter_label = ""
            if (
                args.policy_adapter_path is not None
                or args.late_policy_adapter_path is not None
                or args.conversion_policy_scale > 0.0
            ):
                adapter_label = (
                    f", adapter_used={row.adapter_used_rate * 100:.2f}%, "
                    f"adapter_diff={row.adapter_action_diff_rate * 100:.2f}%, "
                    f"adapter_trigger={row.adapter_trigger_rate * 100:.2f}%"
                )
            print(f"{_row_to_printable(row)}{adapter_label} | elapsed={elapsed:.2f}s")

    min_win_rate = min(row.win_rate for row in rows)
    payload = {
        "model_path": args.model_path,
        "grid_sizes": list(args.grid_sizes),
        "policy_players": list(args.policy_players),
        "pad_to": args.pad_to,
        "opponent": args.opponent,
        "opponent_policy_path": args.opponent_policy_path,
        "opponent_policy_mode": args.opponent_policy_mode,
        "opponent_channels": args.opponent_channels,
        "opponent_input_channels": args.opponent_input_channels,
        "value_head_sizes": list(args.value_head_sizes) if args.value_heads == "per-size" else [],
        "policy_mode": args.policy_mode,
        "num_games": args.num_games,
        "eval_batch_size": eval_batch_size,
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
        "strategy_plan_worker_min_grid_size": args.strategy_plan_worker_min_grid_size,
        "strategy_plan_worker_max_grid_size": args.strategy_plan_worker_max_grid_size,
        "strategy_plan_worker_command_source": args.strategy_plan_worker_command_source,
        "strategy_plan_worker_gate_path": args.strategy_plan_worker_gate_path,
        "strategy_plan_worker_gate_threshold": args.strategy_plan_worker_gate_threshold,
        "strategy_plan_worker_gate_hidden_dim": args.strategy_plan_worker_gate_hidden_dim,
        "strategy_command_gate_path": args.strategy_command_gate_path,
        "strategy_command_gate_threshold": args.strategy_command_gate_threshold,
        "strategy_command_gate_hidden_dim": args.strategy_command_gate_hidden_dim,
        "strategy_command_gate_source_count": args.strategy_command_gate_source_count,
        "strategy_command_gate_target_count": args.strategy_command_gate_target_count,
        "policy_adapter_path": args.policy_adapter_path,
        "policy_adapter_feature_model_path": args.policy_adapter_feature_model_path,
        "policy_adapter_feature_outcome_head": policy_adapter_feature_outcome_head,
        "policy_adapter_feature_strategy_aux": policy_adapter_feature_strategy_aux,
        "policy_adapter_feature_strategy_spatial_aux": policy_adapter_feature_strategy_spatial_aux,
        "policy_adapter_feature_strategy_finish_outputs": policy_adapter_feature_strategy_finish_outputs,
        "policy_adapter_scale": args.policy_adapter_scale,
        "policy_adapter_mode": args.policy_adapter_mode,
        "late_policy_adapter_path": args.late_policy_adapter_path,
        "late_policy_adapter_scale": args.late_policy_adapter_scale,
        "late_policy_adapter_mode": args.late_policy_adapter_mode,
        "late_policy_adapter_min_grid_size": args.late_policy_adapter_min_grid_size,
        "late_policy_adapter_max_grid_size": args.late_policy_adapter_max_grid_size,
        "late_policy_adapter_min_turn": args.late_policy_adapter_min_turn,
        "late_policy_adapter_require_contact": args.late_policy_adapter_require_contact,
        "conversion_policy_head": args.conversion_policy_head,
        "conversion_policy_scale": args.conversion_policy_scale,
        "conversion_policy_mode": args.conversion_policy_mode,
        "policy_adapter_finish_threshold": args.policy_adapter_finish_threshold,
        "policy_adapter_min_grid_size": args.policy_adapter_min_grid_size,
        "policy_adapter_max_grid_size": args.policy_adapter_max_grid_size,
        "policy_adapter_min_turn": args.policy_adapter_min_turn,
        "policy_adapter_require_contact": args.policy_adapter_require_contact,
        "policy_adapter_gate_path": args.policy_adapter_gate_path,
        "policy_adapter_gate_threshold": args.policy_adapter_gate_threshold,
        "policy_adapter_gate_hidden_dim": args.policy_adapter_gate_hidden_dim,
        "policy_adapter_commit_steps": args.policy_adapter_commit_steps,
        "online_search_top_k": args.online_search_top_k,
        "online_search_rollout_steps": args.online_search_rollout_steps,
        "online_search_rollouts_per_action": args.online_search_rollouts_per_action,
        "online_search_min_turn": args.online_search_min_turn,
        "online_search_require_contact": args.online_search_require_contact,
        "online_search_min_grid_size": args.online_search_min_grid_size,
        "online_search_max_grid_size": args.online_search_max_grid_size,
        "online_search_army_weight": args.online_search_army_weight,
        "online_search_land_weight": args.online_search_land_weight,
        "online_search_prior_weight": args.online_search_prior_weight,
        "online_search_terminal_score": args.online_search_terminal_score,
        "online_search_min_score_gap": args.online_search_min_score_gap,
        "online_search_gate_path": args.online_search_gate_path,
        "online_search_gate_threshold": args.online_search_gate_threshold,
        "online_search_gate_hidden_dim": args.online_search_gate_hidden_dim,
        "candidate_scorer_path": args.candidate_scorer_path,
        "candidate_scorer_top_k": args.candidate_scorer_top_k,
        "candidate_scorer_min_turn": args.candidate_scorer_min_turn,
        "candidate_scorer_require_contact": args.candidate_scorer_require_contact,
        "candidate_scorer_min_grid_size": args.candidate_scorer_min_grid_size,
        "candidate_scorer_max_grid_size": args.candidate_scorer_max_grid_size,
        "candidate_scorer_min_score_gap": args.candidate_scorer_min_score_gap,
        "candidate_scorer_local_channels": candidate_scorer_local_channels,
        "candidate_scorer_feature_names": candidate_scorer_features_used,
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
