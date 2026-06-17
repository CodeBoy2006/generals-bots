"""Adaptive conservative rollout-search distillation for multisize checkpoints."""

from __future__ import annotations

import argparse
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
import optax

from generals.agents.ppo_policy_agent import parse_policy_channels
from generals.core import game
from generals.core.game import GameInfo
from generals.core.observation import Observation

from adaptive_common import (
    ADAPTIVE_GLOBAL_INPUT_CHANNELS,
    ADAPTIVE_HISTORY_INPUT_CHANNELS,
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
    reset_adaptive_scoreboard_history,
)
from adaptive_network import load_or_create_adaptive_network
from adaptive_strategy_aux import strategy_aux_targets
from common import POLICY_MODE_NAME_TO_ID, POLICY_MODE_NAMES
from conservative_search_distill import (
    search_score_target_probs,
    select_search_improvements,
    weighted_topk_cross_entropy,
)
from train import checkpoint_path_for_iteration, prune_old_checkpoints, stack_learner_actions

TARGET_MODE_NAMES = ("hard", "soft")
SOFT_WEIGHT_MODE_NAMES = ("active", "improvement")
SOFT_WEIGHT_MODE_NAME_TO_ID = {name: idx for idx, name in enumerate(SOFT_WEIGHT_MODE_NAMES)}
OUTCOME_LOSS = 0
OUTCOME_DRAW = 1
OUTCOME_WIN = 2


def empty_scoreboard_history(num_envs: int) -> jnp.ndarray:
    """Return empty previous-scoreboard features for vectorized distillation rollouts."""
    return jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)


def split_mixed_env_counts(num_envs: int) -> tuple[int, int]:
    """Split total vectorized environments across both learner seats."""
    if num_envs < 2:
        raise ValueError("mixed learner mode requires at least two environments")
    p0_envs = num_envs // 2
    return p0_envs, num_envs - p0_envs


def concatenate_flat_batches(*batches):
    """Concatenate already-flattened p0/p1 distillation batches."""
    return tuple(jnp.concatenate(items, axis=0) for items in zip(*batches, strict=True))


def search_value_targets(search_scores: jnp.ndarray, score_scale: float) -> jnp.ndarray:
    """Convert top-k rollout-search scores into bounded scalar value targets."""
    return jnp.tanh(jnp.max(search_scores, axis=-1) / score_scale).astype(jnp.float32)


def outcome_class_from_winner(winner: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """Map a winner id to loss/draw/win from one learner perspective."""
    return jnp.where(
        winner < 0,
        OUTCOME_DRAW,
        jnp.where(winner == player, OUTCOME_WIN, OUTCOME_LOSS),
    ).astype(jnp.int32)


def mask_legacy_distill_grads(grads, legacy_input_channels: int = ADAPTIVE_INPUT_CHANNELS):
    """Keep gradients only for newly added adaptive input paths."""
    masked = jax.tree.map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, grads)
    if grads.conv1.weight is not None:
        conv1_weight = jnp.zeros_like(grads.conv1.weight)
        conv1_weight = conv1_weight.at[:, legacy_input_channels:].set(grads.conv1.weight[:, legacy_input_channels:])
    else:
        conv1_weight = None
    conv1_bias = jnp.zeros_like(grads.conv1.bias) if grads.conv1.bias is not None else None
    conv1_grad = eqx.tree_at(lambda layer: (layer.weight, layer.bias), grads.conv1, (conv1_weight, conv1_bias))
    masked = eqx.tree_at(lambda net: net.conv1, masked, conv1_grad)
    if grads.global_linear1 is not None:
        masked = eqx.tree_at(lambda net: net.global_linear1, masked, grads.global_linear1)
    if grads.global_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.global_linear2, masked, grads.global_linear2)
    if grads.outcome_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.outcome_linear2, masked, grads.outcome_linear2)
    if grads.strategy_intent_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.strategy_intent_linear2, masked, grads.strategy_intent_linear2)
    if grads.strategy_finish_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.strategy_finish_linear2, masked, grads.strategy_finish_linear2)
    if grads.strategy_q_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_q_conv, masked, grads.strategy_q_conv)
    if grads.strategy_q_pass_linear is not None:
        masked = eqx.tree_at(lambda net: net.strategy_q_pass_linear, masked, grads.strategy_q_pass_linear)
    if grads.strategy_enemy_general_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_enemy_general_conv, masked, grads.strategy_enemy_general_conv)
    return masked


def mask_strategy_aux_grads(grads):
    """Keep gradients only for strategy auxiliary heads."""
    masked = jax.tree.map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, grads)
    if grads.strategy_intent_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.strategy_intent_linear2, masked, grads.strategy_intent_linear2)
    if grads.strategy_finish_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.strategy_finish_linear2, masked, grads.strategy_finish_linear2)
    if grads.strategy_q_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_q_conv, masked, grads.strategy_q_conv)
    if grads.strategy_q_pass_linear is not None:
        masked = eqx.tree_at(lambda net: net.strategy_q_pass_linear, masked, grads.strategy_q_pass_linear)
    if grads.strategy_enemy_general_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_enemy_general_conv, masked, grads.strategy_enemy_general_conv)
    return masked


def compute_adaptive_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    target_indices,
    improve_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return adaptive KL-to-base plus weighted hard search-target loss."""

    def logits_for_sample(network, sample_obs, sample_mask, sample_active):
        logits, _ = network.logits_value(sample_obs, sample_mask, sample_active)
        return logits

    student_logits = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: logits_for_sample(
            student_network,
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)
    base_logits = jax.lax.stop_gradient(
        jax.vmap(
            lambda sample_obs, sample_mask, sample_active: logits_for_sample(
                base_network,
                sample_obs,
                sample_mask,
                sample_active,
            )
        )(base_obs, base_masks, base_active)
    )

    student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    base_log_probs = jax.nn.log_softmax(base_logits / temperature, axis=-1)
    base_probs = jax.nn.softmax(base_logits / temperature, axis=-1)
    kl_per_sample = jnp.sum(base_probs * (base_log_probs - student_log_probs_for_kl), axis=-1)
    kl_normalizer = jnp.maximum(jnp.sum(kl_weights), 1.0)
    kl_loss = jnp.sum(kl_per_sample * kl_weights) / kl_normalizer

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    action_losses = -jnp.take_along_axis(student_log_probs, target_indices[:, None], axis=1)[:, 0]
    improve_normalizer = jnp.maximum(jnp.sum(improve_weights), 1.0)
    improve_loss = jnp.sum(action_losses * improve_weights) / improve_normalizer

    loss = kl_weight * kl_loss + improve_weight * improve_loss
    selected = improve_weights > 0.0
    selected_count = jnp.sum(selected.astype(jnp.float32))
    active_count = jnp.maximum(jnp.sum(kl_weights), 1.0)
    accuracy = jnp.sum((jnp.argmax(student_logits, axis=-1) == target_indices) * improve_weights) / improve_normalizer
    metrics = {
        "kl_loss": kl_loss,
        "improve_loss": jnp.where(selected_count > 0.0, improve_loss, 0.0),
        "selected_fraction": selected_count / active_count,
        "accuracy": jnp.where(selected_count > 0.0, accuracy, 0.0),
    }
    return loss, metrics


def compute_adaptive_soft_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    candidate_indices,
    target_probs,
    search_weights,
    improvement_extra_weights,
    search_value_targets,
    search_value_weights,
    search_outcome_targets,
    search_outcome_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    improvement_extra_weight: float,
    search_value_weight: float,
    search_outcome_weight: float,
    temperature: float,
):
    """Return adaptive KL-to-base plus weighted soft top-k search-target loss."""

    def logits_for_sample(network, sample_obs, sample_mask, sample_active):
        logits, _ = network.logits_value(sample_obs, sample_mask, sample_active)
        return logits

    student_logits = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: logits_for_sample(
            student_network,
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)
    base_logits = jax.lax.stop_gradient(
        jax.vmap(
            lambda sample_obs, sample_mask, sample_active: logits_for_sample(
                base_network,
                sample_obs,
                sample_mask,
                sample_active,
            )
        )(base_obs, base_masks, base_active)
    )

    student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    base_log_probs = jax.nn.log_softmax(base_logits / temperature, axis=-1)
    base_probs = jax.nn.softmax(base_logits / temperature, axis=-1)
    kl_per_sample = jnp.sum(base_probs * (base_log_probs - student_log_probs_for_kl), axis=-1)
    kl_normalizer = jnp.maximum(jnp.sum(kl_weights), 1.0)
    kl_loss = jnp.sum(kl_per_sample * kl_weights) / kl_normalizer

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    search_loss = weighted_topk_cross_entropy(student_log_probs, candidate_indices, target_probs, search_weights)
    improvement_extra_loss = weighted_topk_cross_entropy(
        student_log_probs,
        candidate_indices,
        target_probs,
        improvement_extra_weights,
    )
    student_values = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: student_network.logits_value(
            sample_obs,
            sample_mask,
            sample_active,
        )[1]
    )(obs, masks, active)
    value_normalizer = jnp.maximum(jnp.sum(search_value_weights), 1.0)
    search_value_errors = (student_values - search_value_targets) ** 2
    search_value_loss = jnp.sum(search_value_errors * search_value_weights) / value_normalizer
    if student_network.outcome_head:
        outcome_logits = jax.vmap(
            lambda sample_obs, sample_mask, sample_active: student_network.logits_value_auxiliary(
                sample_obs,
                sample_mask,
                sample_active,
            )[3]
        )(obs, masks, active)
        outcome_log_probs = jax.nn.log_softmax(outcome_logits, axis=-1)
        outcome_losses = -jnp.take_along_axis(outcome_log_probs, search_outcome_targets[:, None], axis=1)[:, 0]
        outcome_predictions = jnp.argmax(outcome_logits, axis=-1)
    else:
        outcome_losses = jnp.zeros_like(search_outcome_weights)
        outcome_predictions = jnp.zeros_like(search_outcome_targets)
    outcome_normalizer = jnp.maximum(jnp.sum(search_outcome_weights), 1.0)
    search_outcome_loss = jnp.sum(outcome_losses * search_outcome_weights) / outcome_normalizer
    search_outcome_accuracy = (
        jnp.sum((outcome_predictions == search_outcome_targets) * search_outcome_weights) / outcome_normalizer
    )
    loss = (
        kl_weight * kl_loss
        + improve_weight * search_loss
        + improvement_extra_weight * improvement_extra_loss
        + search_value_weight * search_value_loss
        + search_outcome_weight * search_outcome_loss
    )

    best_targets = jnp.take_along_axis(candidate_indices, jnp.argmax(target_probs, axis=-1)[:, None], axis=1)[:, 0]
    search_normalizer = jnp.maximum(jnp.sum(search_weights), 1.0)
    extra_normalizer = jnp.maximum(jnp.sum(improvement_extra_weights), 1.0)
    accuracy = jnp.sum((jnp.argmax(student_logits, axis=-1) == best_targets) * search_weights) / search_normalizer
    extra_accuracy = (
        jnp.sum((jnp.argmax(student_logits, axis=-1) == best_targets) * improvement_extra_weights) / extra_normalizer
    )
    target_entropy = -jnp.sum(target_probs * jnp.log(jnp.clip(target_probs, 1e-8, 1.0)), axis=-1)
    metrics = {
        "kl_loss": kl_loss,
        "improve_loss": search_loss,
        "improvement_extra_loss": jnp.where(improvement_extra_weight > 0.0, improvement_extra_loss, 0.0),
        "search_value_loss": jnp.where(search_value_weight > 0.0, search_value_loss, 0.0),
        "search_outcome_loss": jnp.where(search_outcome_weight > 0.0, search_outcome_loss, 0.0),
        "search_outcome_accuracy": jnp.where(search_outcome_weight > 0.0, search_outcome_accuracy, 0.0),
        "mean_search_value_target": jnp.sum(search_value_targets * search_value_weights) / value_normalizer,
        "selected_fraction": jnp.sum(search_weights) / kl_normalizer,
        "improvement_extra_fraction": jnp.sum(improvement_extra_weights) / kl_normalizer,
        "accuracy": accuracy,
        "improvement_extra_accuracy": jnp.where(jnp.sum(improvement_extra_weights) > 0.0, extra_accuracy, 0.0),
        "target_entropy": jnp.sum(target_entropy * search_weights) / search_normalizer,
    }
    return loss, metrics


def binary_cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable binary cross-entropy."""
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def compute_strategy_aux_loss(
    student_network,
    obs,
    masks,
    active,
    candidate_indices,
    candidate_q_targets,
    q_weights,
    intent_targets,
    intent_weights,
    finish_targets,
    finish_weights,
    enemy_general_targets,
    belief_weights,
    q_weight: float,
    intent_weight: float,
    finish_weight: float,
    belief_weight: float,
):
    """Return strategic Q/intent/finish/belief auxiliary loss and metrics."""
    outputs = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: student_network.strategy_auxiliary(
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)

    candidate_q = jnp.take_along_axis(outputs.action_q_values, candidate_indices, axis=1)
    q_errors = jnp.mean((candidate_q - candidate_q_targets) ** 2, axis=1)
    q_normalizer = jnp.maximum(jnp.sum(q_weights), 1.0)
    q_loss = jnp.sum(q_errors * q_weights) / q_normalizer

    intent_log_probs = jax.nn.log_softmax(outputs.intent_logits, axis=-1)
    intent_losses = -jnp.take_along_axis(intent_log_probs, intent_targets[:, None], axis=1)[:, 0]
    intent_normalizer = jnp.maximum(jnp.sum(intent_weights), 1.0)
    intent_loss = jnp.sum(intent_losses * intent_weights) / intent_normalizer
    intent_accuracy = jnp.sum((jnp.argmax(outputs.intent_logits, axis=-1) == intent_targets) * intent_weights)
    intent_accuracy = intent_accuracy / intent_normalizer

    finish_log_probs = jax.nn.log_softmax(outputs.finish_logits, axis=-1)
    finish_losses = -jnp.take_along_axis(finish_log_probs, finish_targets[:, None], axis=1)[:, 0]
    finish_normalizer = jnp.maximum(jnp.sum(finish_weights), 1.0)
    finish_loss = jnp.sum(finish_losses * finish_weights) / finish_normalizer
    finish_accuracy = jnp.sum((jnp.argmax(outputs.finish_logits, axis=-1) == finish_targets) * finish_weights)
    finish_accuracy = finish_accuracy / finish_normalizer

    active_f = active.astype(jnp.float32)
    per_cell_bce = binary_cross_entropy_with_logits(outputs.enemy_general_logits, enemy_general_targets)
    per_sample_belief = jnp.sum(per_cell_bce * active_f, axis=(1, 2)) / jnp.maximum(jnp.sum(active_f, axis=(1, 2)), 1.0)
    belief_normalizer = jnp.maximum(jnp.sum(belief_weights), 1.0)
    belief_loss = jnp.sum(per_sample_belief * belief_weights) / belief_normalizer

    loss = q_weight * q_loss + intent_weight * intent_loss + finish_weight * finish_loss + belief_weight * belief_loss
    metrics = {
        "strategy_q_loss": jnp.where(q_weight > 0.0, q_loss, 0.0),
        "strategy_intent_loss": jnp.where(intent_weight > 0.0, intent_loss, 0.0),
        "strategy_finish_loss": jnp.where(finish_weight > 0.0, finish_loss, 0.0),
        "strategy_belief_loss": jnp.where(belief_weight > 0.0, belief_loss, 0.0),
        "strategy_intent_accuracy": jnp.where(intent_weight > 0.0, intent_accuracy, 0.0),
        "strategy_finish_accuracy": jnp.where(finish_weight > 0.0, finish_accuracy, 0.0),
    }
    return loss, metrics


def soft_search_weights(
    candidate_indices,
    search_scores,
    active_weights,
    soft_weight_mode,
    min_margin: float,
    margin_scale: float,
    max_weight: float,
):
    """Return soft-target sample weights from active rows or confident search improvements."""
    _, improve_weights, _ = select_search_improvements(
        candidate_indices,
        search_scores,
        min_margin,
        margin_scale,
        max_weight,
    )
    return jax.lax.cond(
        soft_weight_mode == SOFT_WEIGHT_MODE_NAME_TO_ID["active"],
        lambda _: active_weights,
        lambda _: improve_weights * active_weights,
        None,
    ).astype(jnp.float32)


def adaptive_score_observation(
    info: GameInfo,
    obs: Observation,
    player: int,
    army_weight: float = 12.0,
    land_weight: float = 8.0,
    terminal_score: float = 1000.0,
):
    """Score a final adaptive rollout observation from one player's perspective."""
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


def adaptive_policy_action(
    network,
    obs,
    effective_size,
    key,
    policy_mode,
    pad_size: int,
    global_context: bool = False,
    scoreboard_history: jnp.ndarray | None = None,
):
    """Dispatch an adaptive checkpoint action using greedy or sampled execution."""
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=scoreboard_history,
    )
    mask = compute_adaptive_valid_move_mask(
        obs.armies,
        obs.owned_cells,
        obs.mountains,
        effective_size,
        pad_size,
    )
    logits, _ = network.logits_value(obs_arr, mask, active)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, pad_size)


def adaptive_obs_to_array_with_context(
    obs,
    effective_size,
    pad_size: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
):
    """Convert an observation with optional global or scoreboard-history context."""
    history_context = None
    if scoreboard_history_enabled:
        current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
        history_context = adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
    return adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
    )


def adaptive_policy_action_with_context(
    network,
    obs,
    effective_size,
    key,
    policy_mode,
    pad_size: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
):
    """Dispatch an adaptive action while constructing optional scoreboard-history context."""
    history_context = None
    if scoreboard_history_enabled:
        current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
        history_context = adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
    return adaptive_policy_action(
        network,
        obs,
        effective_size,
        key,
        policy_mode,
        pad_size,
        global_context=global_context,
        scoreboard_history=history_context,
    )


@eqx.filter_jit
def adaptive_rollout_search_candidates(
    network,
    state,
    effective_size,
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
    pad_size,
    global_context=False,
    scoreboard_history_enabled=False,
    previous_scoreboard_p0=None,
    previous_scoreboard_p1=None,
):
    """Return adaptive top-k prior candidates and short rollout-search scores."""
    if previous_scoreboard_p0 is None:
        previous_scoreboard_p0 = jnp.zeros((ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,), dtype=jnp.float32)
    if previous_scoreboard_p1 is None:
        previous_scoreboard_p1 = jnp.zeros((ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,), dtype=jnp.float32)
    obs = game.get_observation(state, player)
    player_previous_scoreboard = jax.lax.cond(
        player == 0,
        lambda _: previous_scoreboard_p0,
        lambda _: previous_scoreboard_p1,
        None,
    )
    player_history_context = None
    if scoreboard_history_enabled:
        player_current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
        player_history_context = adaptive_scoreboard_history_context(player_previous_scoreboard, player_current_scoreboard)
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=player_history_context,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits, _ = network.logits_value(obs_arr, mask, active)
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda idx: adaptive_index_to_action(idx, pad_size))(candidate_indices)

    opponent_player = 1 - player
    opponent_obs = game.get_observation(state, opponent_player)
    current_scoreboard_p0 = adaptive_scoreboard_features(game.get_observation(state, 0), effective_size)
    current_scoreboard_p1 = adaptive_scoreboard_features(game.get_observation(state, 1), effective_size)
    key, opponent_key = jrandom.split(key)
    opponent_first_action = adaptive_policy_action_with_context(
        network,
        opponent_obs,
        effective_size,
        opponent_key,
        policy_mode,
        pad_size,
        global_context,
        scoreboard_history_enabled,
        jax.lax.cond(
            opponent_player == 0,
            lambda _: previous_scoreboard_p0,
            lambda _: previous_scoreboard_p1,
            None,
        ),
    )

    def rollout_result(initial_state, rollout_key, initial_previous_p0, initial_previous_p1):
        def body(carry, _):
            rollout_state, previous_p0, previous_p1, step_key = carry
            step_key, k0, k1 = jrandom.split(step_key, 3)
            obs_p0 = game.get_observation(rollout_state, 0)
            obs_p1 = game.get_observation(rollout_state, 1)
            current_p0 = adaptive_scoreboard_features(obs_p0, effective_size)
            current_p1 = adaptive_scoreboard_features(obs_p1, effective_size)
            action_p0 = adaptive_policy_action_with_context(
                network,
                obs_p0,
                effective_size,
                k0,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                previous_p0,
            )
            action_p1 = adaptive_policy_action_with_context(
                network,
                obs_p1,
                effective_size,
                k1,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                previous_p1,
            )
            next_state, _ = game.step(rollout_state, jnp.stack([action_p0, action_p1]))
            already_done = game.get_info(rollout_state).is_done
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            final_info = game.get_info(final_state)
            next_p0 = reset_adaptive_scoreboard_history(current_p0, final_info.is_done)
            next_p1 = reset_adaptive_scoreboard_history(current_p1, final_info.is_done)
            return (final_state, next_p0, next_p1, step_key), None

        (final_state, _, _, _), _ = jax.lax.scan(
            body,
            (initial_state, initial_previous_p0, initial_previous_p1, rollout_key),
            None,
            length=rollout_steps,
        )
        final_info = game.get_info(final_state)
        final_obs = game.get_observation(final_state, player)
        score = adaptive_score_observation(final_info, final_obs, player, army_weight, land_weight, terminal_score)
        outcome = outcome_class_from_winner(
            jnp.where(final_info.is_done, final_info.winner, -1),
            player,
        )
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
        scores, outcomes = jax.vmap(
            lambda rollout_key: rollout_result(
                next_state,
                rollout_key,
                current_scoreboard_p0,
                current_scoreboard_p1,
            )
        )(rollout_keys)
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
def adaptive_rollout_search_action(
    network,
    state,
    effective_size,
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
    pad_size,
):
    """Choose one adaptive action by scoring top-k prior actions with short rollouts."""
    candidate_actions, _, _, scores, _ = adaptive_rollout_search_candidates(
        network,
        state,
        effective_size,
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
        pad_size,
    )
    return candidate_actions[jnp.argmax(scores)]


@eqx.filter_jit
def train_adaptive_conservative_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
    freeze_legacy_weights=False,
    legacy_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
):
    """Train one adaptive hard-target distillation minibatch."""
    obs, masks, active, base_obs, base_masks, base_active, target_indices, improve_weights, kl_weights = minibatch

    def loss_fn(net):
        return compute_adaptive_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            active,
            base_obs,
            base_masks,
            base_active,
            target_indices,
            improve_weights,
            kl_weights,
            kl_weight,
            improve_weight,
            temperature,
        )

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(student_network)
    if freeze_legacy_weights:
        grads = mask_legacy_distill_grads(grads, legacy_input_channels)
    params = eqx.filter(student_network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    student_network = eqx.apply_updates(student_network, updates)
    return student_network, opt_state, loss, metrics


@eqx.filter_jit
def train_adaptive_soft_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    improvement_extra_weight,
    search_value_weight,
    search_outcome_weight,
    strategy_q_weight,
    strategy_intent_weight,
    strategy_finish_weight,
    strategy_belief_weight,
    temperature,
    freeze_legacy_weights=False,
    freeze_strategy_aux_only=False,
    legacy_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
):
    """Train one adaptive soft-target distillation minibatch."""
    (
        obs,
        masks,
        active,
        base_obs,
        base_masks,
        base_active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        strategy_candidate_q_targets,
        strategy_q_weights,
        strategy_intent_targets,
        strategy_intent_weights,
        strategy_finish_targets,
        strategy_finish_weights,
        strategy_enemy_general_targets,
        strategy_belief_weights,
    ) = minibatch

    def loss_fn(net):
        distill_loss, metrics = compute_adaptive_soft_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            active,
            base_obs,
            base_masks,
            base_active,
            candidate_indices,
            target_probs,
            search_weights,
            improvement_extra_weights,
            search_value_targets,
            search_value_weights,
            search_outcome_targets,
            search_outcome_weights,
            kl_weights,
            kl_weight,
            improve_weight,
            improvement_extra_weight,
            search_value_weight,
            search_outcome_weight,
            temperature,
        )
        if (
            strategy_q_weight > 0.0
            or strategy_intent_weight > 0.0
            or strategy_finish_weight > 0.0
            or strategy_belief_weight > 0.0
        ):
            strategy_loss, strategy_metrics = compute_strategy_aux_loss(
                net,
                obs,
                masks,
                active,
                candidate_indices,
                strategy_candidate_q_targets,
                strategy_q_weights,
                strategy_intent_targets,
                strategy_intent_weights,
                strategy_finish_targets,
                strategy_finish_weights,
                strategy_enemy_general_targets,
                strategy_belief_weights,
                strategy_q_weight,
                strategy_intent_weight,
                strategy_finish_weight,
                strategy_belief_weight,
            )
        else:
            strategy_loss = jnp.asarray(0.0, dtype=jnp.float32)
            strategy_metrics = {
                "strategy_q_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "strategy_intent_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "strategy_finish_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "strategy_belief_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "strategy_intent_accuracy": jnp.asarray(0.0, dtype=jnp.float32),
                "strategy_finish_accuracy": jnp.asarray(0.0, dtype=jnp.float32),
            }
        metrics = dict(metrics)
        metrics.update(strategy_metrics)
        return distill_loss + strategy_loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(student_network)
    if freeze_strategy_aux_only:
        grads = mask_strategy_aux_grads(grads)
    elif freeze_legacy_weights:
        grads = mask_legacy_distill_grads(grads, legacy_input_channels)
    params = eqx.filter(student_network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    student_network = eqx.apply_updates(student_network, updates)
    return student_network, opt_state, loss, metrics


def flatten_adaptive_conservative_batch(
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    target_indices,
    improve_weights,
    kl_weights,
    margins,
):
    """Flatten time/environment axes for adaptive hard-target distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        base_active.reshape(batch_size, *base_active.shape[2:]),
        target_indices.reshape(batch_size),
        improve_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
        margins.reshape(batch_size),
    )


def flatten_adaptive_soft_batch(
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    candidate_indices,
    target_probs,
    search_weights,
    improvement_extra_weights,
    search_value_targets,
    search_value_weights,
    search_outcome_targets,
    search_outcome_weights,
    kl_weights,
    strategy_candidate_q_targets,
    strategy_q_weights,
    strategy_intent_targets,
    strategy_intent_weights,
    strategy_finish_targets,
    strategy_finish_weights,
    strategy_enemy_general_targets,
    strategy_belief_weights,
):
    """Flatten time/environment axes for adaptive soft-target distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        base_active.reshape(batch_size, *base_active.shape[2:]),
        candidate_indices.reshape(batch_size, *candidate_indices.shape[2:]),
        target_probs.reshape(batch_size, *target_probs.shape[2:]),
        search_weights.reshape(batch_size),
        improvement_extra_weights.reshape(batch_size),
        search_value_targets.reshape(batch_size),
        search_value_weights.reshape(batch_size),
        search_outcome_targets.reshape(batch_size),
        search_outcome_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
        strategy_candidate_q_targets.reshape(batch_size, *strategy_candidate_q_targets.shape[2:]),
        strategy_q_weights.reshape(batch_size),
        strategy_intent_targets.reshape(batch_size),
        strategy_intent_weights.reshape(batch_size),
        strategy_finish_targets.reshape(batch_size),
        strategy_finish_weights.reshape(batch_size),
        strategy_enemy_general_targets.reshape(batch_size, *strategy_enemy_general_targets.shape[2:]),
        strategy_belief_weights.reshape(batch_size),
    )


def train_adaptive_conservative_epoch(
    student_network,
    base_network,
    opt_state,
    flat_batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    improve_weight,
    temperature,
    freeze_legacy_weights=False,
    legacy_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
):
    """Run adaptive hard-target distillation over shuffled minibatches."""
    obs, masks, active, base_obs, base_masks, base_active, target_indices, improve_weights, kl_weights, margins = (
        flat_batch
    )
    batch_size = obs.shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = (
            obs[permutation],
            masks[permutation],
            active[permutation],
            base_obs[permutation],
            base_masks[permutation],
            base_active[permutation],
            target_indices[permutation],
            improve_weights[permutation],
            kl_weights[permutation],
        )
        epoch_loss = 0.0
        epoch_metrics = None

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            student_network, opt_state, loss, metrics = train_adaptive_conservative_minibatch(
                student_network,
                base_network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                improve_weight,
                temperature,
                freeze_legacy_weights,
                legacy_input_channels,
            )
            epoch_loss += loss
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, metrics)

        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda x: x / num_complete_batches, epoch_metrics)

    selected_margins = jnp.where(improve_weights > 0.0, margins, 0.0)
    selected_count = jnp.maximum(jnp.sum((improve_weights > 0.0).astype(jnp.float32)), 1.0)
    avg_metrics = dict(avg_metrics)
    avg_metrics["mean_selected_margin"] = jnp.sum(selected_margins) / selected_count
    avg_metrics["selected_samples"] = jnp.sum((improve_weights > 0.0).astype(jnp.float32))
    return student_network, opt_state, avg_loss, avg_metrics, key


def train_adaptive_soft_epoch(
    student_network,
    base_network,
    opt_state,
    flat_batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    improve_weight,
    improvement_extra_weight,
    search_value_weight,
    search_outcome_weight,
    strategy_q_weight,
    strategy_intent_weight,
    strategy_finish_weight,
    strategy_belief_weight,
    temperature,
    freeze_legacy_weights=False,
    freeze_strategy_aux_only=False,
    legacy_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
):
    """Run adaptive soft-target distillation over shuffled minibatches."""
    (
        obs,
        masks,
        active,
        base_obs,
        base_masks,
        base_active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        strategy_candidate_q_targets,
        strategy_q_weights,
        strategy_intent_targets,
        strategy_intent_weights,
        strategy_finish_targets,
        strategy_finish_weights,
        strategy_enemy_general_targets,
        strategy_belief_weights,
    ) = flat_batch
    batch_size = obs.shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = (
            obs[permutation],
            masks[permutation],
            active[permutation],
            base_obs[permutation],
            base_masks[permutation],
            base_active[permutation],
            candidate_indices[permutation],
            target_probs[permutation],
            search_weights[permutation],
            improvement_extra_weights[permutation],
            search_value_targets[permutation],
            search_value_weights[permutation],
            search_outcome_targets[permutation],
            search_outcome_weights[permutation],
            kl_weights[permutation],
            strategy_candidate_q_targets[permutation],
            strategy_q_weights[permutation],
            strategy_intent_targets[permutation],
            strategy_intent_weights[permutation],
            strategy_finish_targets[permutation],
            strategy_finish_weights[permutation],
            strategy_enemy_general_targets[permutation],
            strategy_belief_weights[permutation],
        )
        epoch_loss = 0.0
        epoch_metrics = None

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            student_network, opt_state, loss, metrics = train_adaptive_soft_minibatch(
                student_network,
                base_network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                improve_weight,
                improvement_extra_weight,
                search_value_weight,
                search_outcome_weight,
                strategy_q_weight,
                strategy_intent_weight,
                strategy_finish_weight,
                strategy_belief_weight,
                temperature,
                freeze_legacy_weights,
                freeze_strategy_aux_only,
                legacy_input_channels,
            )
            epoch_loss += loss
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, metrics)

        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda x: x / num_complete_batches, epoch_metrics)

    avg_metrics = dict(avg_metrics)
    avg_metrics["selected_samples"] = jnp.sum((search_weights > 0.0).astype(jnp.float32))
    avg_metrics["improvement_extra_samples"] = jnp.sum((improvement_extra_weights > 0.0).astype(jnp.float32))
    avg_metrics["mean_selected_margin"] = 0.0
    avg_metrics["search_value_samples"] = jnp.sum((search_value_weights > 0.0).astype(jnp.float32))
    avg_metrics["search_outcome_samples"] = jnp.sum((search_outcome_weights > 0.0).astype(jnp.float32))
    avg_metrics["strategy_samples"] = jnp.sum((strategy_q_weights > 0.0).astype(jnp.float32))
    return student_network, opt_state, avg_loss, avg_metrics, key


@eqx.filter_jit
def collect_adaptive_conservative_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    effective_sizes,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    min_margin,
    margin_scale,
    max_weight,
    pad_size,
    global_context=False,
    scoreboard_history_enabled=False,
    base_global_context=False,
    base_scoreboard_history_enabled=False,
):
    """Collect adaptive learner states labeled by hard search improvements."""
    num_envs = states.armies.shape[0]
    scoreboard_history = empty_scoreboard_history(num_envs)
    base_scoreboard_history_p0 = empty_scoreboard_history(num_envs)
    base_scoreboard_history_p1 = empty_scoreboard_history(num_envs)

    def body(carry, _):
        states, key, scoreboard_history, base_scoreboard_history_p0, base_scoreboard_history_p1 = carry
        prior_info = jax.vmap(game.get_info)(states)
        is_active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        base_learner_previous = jax.lax.cond(
            learner_player == 0,
            lambda _: base_scoreboard_history_p0,
            lambda _: base_scoreboard_history_p1,
            None,
        )
        base_opponent_previous = jax.lax.cond(
            learner_player == 0,
            lambda _: base_scoreboard_history_p1,
            lambda _: base_scoreboard_history_p0,
            None,
        )
        if scoreboard_history_enabled:
            current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                learner_obs,
                effective_sizes,
            )
            history_context = adaptive_scoreboard_history_context(scoreboard_history, current_scoreboard)
            learner_obs_arr, active = jax.vmap(
                lambda obs, size, history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                )
            )(
                learner_obs,
                effective_sizes,
                history_context,
            )
        else:
            current_scoreboard = scoreboard_history
            learner_obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=global_context,
                )
            )(
                learner_obs,
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
        )(learner_obs, effective_sizes)
        if base_global_context:
            base_obs_arr, base_active = jax.vmap(
                lambda obs, size, previous: adaptive_obs_to_array_with_context(
                    obs,
                    size,
                    pad_size,
                    base_global_context,
                    base_scoreboard_history_enabled,
                    previous,
                )
            )(
                learner_obs,
                effective_sizes,
                base_learner_previous,
            )
        else:
            base_obs_arr, base_active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
                learner_obs,
                effective_sizes,
            )
        base_masks = masks

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores, _ = jax.vmap(
            lambda state, size, sample_key, previous_p0, previous_p1: adaptive_rollout_search_candidates(
                base_network,
                state,
                size,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
                terminal_score,
                pad_size,
                base_global_context,
                base_scoreboard_history_enabled,
                previous_p0,
                previous_p1,
            )
        )(states, effective_sizes, search_keys, base_scoreboard_history_p0, base_scoreboard_history_p1)
        target_indices, improve_weights, margins = select_search_improvements(
            candidate_indices,
            search_scores,
            min_margin,
            margin_scale,
            max_weight,
        )
        active_weights = is_active.astype(jnp.float32)
        improve_weights = improve_weights * active_weights

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        if scoreboard_history_enabled:
            learner_actions = jax.vmap(
                lambda obs, size, sample_key, history: adaptive_policy_action(
                    student_network,
                    obs,
                    size,
                    sample_key,
                    policy_mode,
                    pad_size,
                    global_context=True,
                    scoreboard_history=history,
                )
            )(learner_obs, effective_sizes, learner_keys, history_context)
        else:
            learner_actions = jax.vmap(
                lambda obs, size, sample_key: adaptive_policy_action(
                    student_network,
                    obs,
                    size,
                    sample_key,
                    policy_mode,
                    pad_size,
                    global_context=global_context,
                )
            )(learner_obs, effective_sizes, learner_keys)
        if base_global_context:
            opponent_actions = jax.vmap(
                lambda obs, size, sample_key, previous: adaptive_policy_action_with_context(
                    opponent_network,
                    obs,
                    size,
                    sample_key,
                    opponent_policy_mode,
                    pad_size,
                    base_global_context,
                    base_scoreboard_history_enabled,
                    previous,
                )
            )(opponent_obs, effective_sizes, opponent_keys, base_opponent_previous)
        else:
            opponent_actions = jax.vmap(
                lambda obs, size, sample_key: adaptive_policy_action(
                    opponent_network,
                    obs,
                    size,
                    sample_key,
                    opponent_policy_mode,
                    pad_size,
                )
            )(opponent_obs, effective_sizes, opponent_keys)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(is_active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        final_info = jax.vmap(game.get_info)(final_states)
        next_scoreboard_history = reset_adaptive_scoreboard_history(current_scoreboard, final_info.is_done)
        if base_scoreboard_history_enabled:
            current_base_scoreboard_p0 = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                obs_p0,
                effective_sizes,
            )
            current_base_scoreboard_p1 = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                obs_p1,
                effective_sizes,
            )
            next_base_scoreboard_p0 = reset_adaptive_scoreboard_history(current_base_scoreboard_p0, final_info.is_done)
            next_base_scoreboard_p1 = reset_adaptive_scoreboard_history(current_base_scoreboard_p1, final_info.is_done)
        else:
            next_base_scoreboard_p0 = base_scoreboard_history_p0
            next_base_scoreboard_p1 = base_scoreboard_history_p1
        return (
            final_states,
            key,
            next_scoreboard_history,
            next_base_scoreboard_p0,
            next_base_scoreboard_p1,
        ), (
            learner_obs_arr,
            masks,
            active,
            base_obs_arr,
            base_masks,
            base_active,
            target_indices,
            improve_weights,
            active_weights,
            margins,
        )

    (states, key, _, _, _), batch = jax.lax.scan(
        body,
        (states, key, scoreboard_history, base_scoreboard_history_p0, base_scoreboard_history_p1),
        None,
        length=num_steps,
    )
    return states, batch, key


@eqx.filter_jit
def collect_adaptive_soft_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    effective_sizes,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    soft_weight_mode,
    min_margin,
    margin_scale,
    max_weight,
    score_temperature,
    search_value_scale,
    pad_size,
    global_context=False,
    scoreboard_history_enabled=False,
    base_global_context=False,
    base_scoreboard_history_enabled=False,
):
    """Collect adaptive learner states labeled by soft search-score targets."""
    num_envs = states.armies.shape[0]
    scoreboard_history = empty_scoreboard_history(num_envs)
    base_scoreboard_history_p0 = empty_scoreboard_history(num_envs)
    base_scoreboard_history_p1 = empty_scoreboard_history(num_envs)

    def body(carry, _):
        states, key, scoreboard_history, base_scoreboard_history_p0, base_scoreboard_history_p1 = carry
        prior_info = jax.vmap(game.get_info)(states)
        is_active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        base_learner_previous = jax.lax.cond(
            learner_player == 0,
            lambda _: base_scoreboard_history_p0,
            lambda _: base_scoreboard_history_p1,
            None,
        )
        base_opponent_previous = jax.lax.cond(
            learner_player == 0,
            lambda _: base_scoreboard_history_p1,
            lambda _: base_scoreboard_history_p0,
            None,
        )
        if scoreboard_history_enabled:
            current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                learner_obs,
                effective_sizes,
            )
            history_context = adaptive_scoreboard_history_context(scoreboard_history, current_scoreboard)
            learner_obs_arr, active = jax.vmap(
                lambda obs, size, history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                )
            )(
                learner_obs,
                effective_sizes,
                history_context,
            )
        else:
            current_scoreboard = scoreboard_history
            learner_obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=global_context,
                )
            )(
                learner_obs,
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
        )(learner_obs, effective_sizes)
        if base_global_context:
            base_obs_arr, base_active = jax.vmap(
                lambda obs, size, previous: adaptive_obs_to_array_with_context(
                    obs,
                    size,
                    pad_size,
                    base_global_context,
                    base_scoreboard_history_enabled,
                    previous,
                )
            )(
                learner_obs,
                effective_sizes,
                base_learner_previous,
            )
        else:
            base_obs_arr, base_active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
                learner_obs,
                effective_sizes,
            )
        base_masks = masks

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores, candidate_outcomes = jax.vmap(
            lambda state, size, sample_key, previous_p0, previous_p1: adaptive_rollout_search_candidates(
                base_network,
                state,
                size,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
                terminal_score,
                pad_size,
                base_global_context,
                base_scoreboard_history_enabled,
                previous_p0,
                previous_p1,
            )
        )(states, effective_sizes, search_keys, base_scoreboard_history_p0, base_scoreboard_history_p1)
        target_probs = search_score_target_probs(search_scores, score_temperature)
        value_targets = search_value_targets(search_scores, search_value_scale)
        best_candidate = jnp.argmax(search_scores, axis=-1)
        outcome_targets = jnp.take_along_axis(candidate_outcomes, best_candidate[:, None], axis=1)[:, 0]
        strategy_targets = jax.vmap(
            lambda state, obs, size, scores, outcomes: strategy_aux_targets(
                state,
                obs,
                learner_player,
                size,
                pad_size,
                scores,
                outcomes,
                search_value_scale,
            )
        )(states, learner_obs, effective_sizes, search_scores, candidate_outcomes)
        strategy_candidate_q_targets = jnp.tanh(search_scores / search_value_scale).astype(jnp.float32)
        active_weights = is_active.astype(jnp.float32)
        search_weights = soft_search_weights(
            candidate_indices,
            search_scores,
            active_weights,
            soft_weight_mode,
            min_margin,
            margin_scale,
            max_weight,
        )
        _, improvement_extra_weights, _ = select_search_improvements(
            candidate_indices,
            search_scores,
            min_margin,
            margin_scale,
            max_weight,
        )
        improvement_extra_weights = improvement_extra_weights * active_weights

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        if scoreboard_history_enabled:
            learner_actions = jax.vmap(
                lambda obs, size, sample_key, history: adaptive_policy_action(
                    student_network,
                    obs,
                    size,
                    sample_key,
                    policy_mode,
                    pad_size,
                    global_context=True,
                    scoreboard_history=history,
                )
            )(learner_obs, effective_sizes, learner_keys, history_context)
        else:
            learner_actions = jax.vmap(
                lambda obs, size, sample_key: adaptive_policy_action(
                    student_network,
                    obs,
                    size,
                    sample_key,
                    policy_mode,
                    pad_size,
                    global_context=global_context,
                )
            )(learner_obs, effective_sizes, learner_keys)
        if base_global_context:
            opponent_actions = jax.vmap(
                lambda obs, size, sample_key, previous: adaptive_policy_action_with_context(
                    opponent_network,
                    obs,
                    size,
                    sample_key,
                    opponent_policy_mode,
                    pad_size,
                    base_global_context,
                    base_scoreboard_history_enabled,
                    previous,
                )
            )(opponent_obs, effective_sizes, opponent_keys, base_opponent_previous)
        else:
            opponent_actions = jax.vmap(
                lambda obs, size, sample_key: adaptive_policy_action(
                    opponent_network,
                    obs,
                    size,
                    sample_key,
                    opponent_policy_mode,
                    pad_size,
                )
            )(opponent_obs, effective_sizes, opponent_keys)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(is_active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        final_info = jax.vmap(game.get_info)(final_states)
        next_scoreboard_history = reset_adaptive_scoreboard_history(current_scoreboard, final_info.is_done)
        if base_scoreboard_history_enabled:
            current_base_scoreboard_p0 = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                obs_p0,
                effective_sizes,
            )
            current_base_scoreboard_p1 = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                obs_p1,
                effective_sizes,
            )
            next_base_scoreboard_p0 = reset_adaptive_scoreboard_history(current_base_scoreboard_p0, final_info.is_done)
            next_base_scoreboard_p1 = reset_adaptive_scoreboard_history(current_base_scoreboard_p1, final_info.is_done)
        else:
            next_base_scoreboard_p0 = base_scoreboard_history_p0
            next_base_scoreboard_p1 = base_scoreboard_history_p1
        return (
            final_states,
            key,
            next_scoreboard_history,
            next_base_scoreboard_p0,
            next_base_scoreboard_p1,
        ), (
            learner_obs_arr,
            masks,
            active,
            base_obs_arr,
            base_masks,
            base_active,
            candidate_indices,
            target_probs,
            search_weights,
            improvement_extra_weights,
            value_targets,
            active_weights,
            outcome_targets,
            active_weights,
            active_weights,
            strategy_candidate_q_targets,
            active_weights,
            strategy_targets.intent,
            active_weights,
            strategy_targets.finish,
            active_weights,
            strategy_targets.enemy_general_heatmap,
            active_weights,
        )

    (states, key, _, _, _), batch = jax.lax.scan(
        body,
        (states, key, scoreboard_history, base_scoreboard_history_p0, base_scoreboard_history_p1),
        None,
        length=num_steps,
    )
    return states, batch, key


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptively distill rollout-search improvements into one multisize checkpoint."
    )
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/generals-adaptive-search-distill.eqx")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--base-channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--base-global-context", action="store_true")
    parser.add_argument("--base-scoreboard-history", action="store_true")
    parser.add_argument("--init-global-context", action="store_true")
    parser.add_argument("--init-input-channels", type=int, default=None)
    parser.add_argument("--init-outcome-head", action="store_true")
    parser.add_argument("--init-strategy-aux", action="store_true")
    parser.add_argument("--freeze-legacy-weights", action="store_true")
    parser.add_argument("--freeze-strategy-aux-only", action="store_true")
    parser.add_argument("--target-mode", choices=TARGET_MODE_NAMES, default="soft")
    parser.add_argument("--soft-weight-mode", choices=SOFT_WEIGHT_MODE_NAMES, default="active")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--learner-player", choices=("0", "1", "mixed"), default="0")
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--improve-weight", type=float, default=0.05)
    parser.add_argument("--soft-improvement-extra-weight", type=float, default=0.0)
    parser.add_argument("--search-value-weight", type=float, default=0.0)
    parser.add_argument("--search-value-scale", type=float, default=100.0)
    parser.add_argument("--search-outcome-weight", type=float, default=0.0)
    parser.add_argument("--strategy-q-weight", type=float, default=0.0)
    parser.add_argument("--strategy-intent-weight", type=float, default=0.0)
    parser.add_argument("--strategy-finish-weight", type=float, default=0.0)
    parser.add_argument("--strategy-belief-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=25.0)
    parser.add_argument("--margin-scale", type=float, default=100.0)
    parser.add_argument("--max-improve-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--army-weight", type=float, default=12.0)
    parser.add_argument("--land-weight", type=float, default=8.0)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument("--terminal-score", type=float, default=1000.0)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--keep-checkpoints", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.grid_size_weights = parse_grid_size_weights(args.grid_size_weights, args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_envs <= 0:
        parser.error("num_envs must be positive")
    if args.learner_player == "mixed" and args.num_envs < 2:
        parser.error("--learner-player mixed requires num_envs >= 2")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_steps <= 0 or args.num_iterations <= 0 or args.num_epochs <= 0:
        parser.error("--num-steps, --num-iterations, and --num-epochs must be positive")
    if args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if (
        args.kl_weight < 0.0
        or args.improve_weight < 0.0
        or args.soft_improvement_extra_weight < 0.0
        or args.search_value_weight < 0.0
        or args.search_outcome_weight < 0.0
        or args.strategy_q_weight < 0.0
        or args.strategy_intent_weight < 0.0
        or args.strategy_finish_weight < 0.0
        or args.strategy_belief_weight < 0.0
    ):
        parser.error(
            "--kl-weight, --improve-weight, --soft-improvement-extra-weight, "
            "--search-value-weight, --search-outcome-weight, and strategy weights must be non-negative"
        )
    if args.search_value_scale <= 0.0:
        parser.error("--search-value-scale must be positive")
    if args.temperature <= 0.0 or args.score_temperature <= 0.0:
        parser.error("--temperature and --score-temperature must be positive")
    if args.margin_scale <= 0.0 or args.max_improve_weight <= 0.0:
        parser.error("--margin-scale and --max-improve-weight must be positive")
    if args.top_k <= 0 or args.rollout_steps <= 0 or args.rollouts_per_action <= 0:
        parser.error("--top-k, --rollout-steps, and --rollouts-per-action must be positive")
    if args.terminal_score <= 0.0:
        parser.error("--terminal-score must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.checkpoint_every < 0 or args.keep_checkpoints < 0:
        parser.error("--checkpoint-every and --keep-checkpoints cannot be negative")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.freeze_legacy_weights and not (args.global_context or args.scoreboard_history):
        parser.error("--freeze-legacy-weights requires --global-context or --scoreboard-history")
    if args.freeze_strategy_aux_only and not (
        args.strategy_q_weight > 0.0
        or args.strategy_intent_weight > 0.0
        or args.strategy_finish_weight > 0.0
        or args.strategy_belief_weight > 0.0
    ):
        parser.error("--freeze-strategy-aux-only requires at least one strategy auxiliary loss")
    try:
        args.channels = parse_policy_channels(args.channels)
        args.base_channels = parse_policy_channels(args.base_channels or args.channels)
        args.init_channels = parse_policy_channels(args.init_channels) if args.init_channels is not None else None
    except ValueError as exc:
        parser.error(str(exc))
    args.soft_weight_mode_id = SOFT_WEIGHT_MODE_NAME_TO_ID[args.soft_weight_mode]
    return args


def main():
    args = parse_args()
    init_model_path = args.init_model_path or args.base_model_path
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    mixed_learner = args.learner_player == "mixed"
    fixed_learner_player = 0 if args.learner_player == "0" else 1
    network_global_context = args.global_context or args.scoreboard_history
    base_network_global_context = args.base_global_context or args.base_scoreboard_history
    strategy_aux_enabled = (
        args.strategy_q_weight > 0.0
        or args.strategy_intent_weight > 0.0
        or args.strategy_finish_weight > 0.0
        or args.strategy_belief_weight > 0.0
    )
    if args.scoreboard_history:
        input_channels = ADAPTIVE_HISTORY_INPUT_CHANNELS
    elif network_global_context:
        input_channels = ADAPTIVE_GLOBAL_INPUT_CHANNELS
    else:
        input_channels = ADAPTIVE_INPUT_CHANNELS
    if args.base_scoreboard_history:
        base_input_channels = ADAPTIVE_HISTORY_INPUT_CHANNELS
    elif base_network_global_context:
        base_input_channels = ADAPTIVE_GLOBAL_INPUT_CHANNELS
    else:
        base_input_channels = ADAPTIVE_INPUT_CHANNELS
    init_input_channels = args.init_input_channels
    if init_input_channels is None and network_global_context and not args.init_global_context:
        init_input_channels = ADAPTIVE_INPUT_CHANNELS

    print("Adaptive conservative rollout-search distillation")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Student:       {init_model_path} channels={args.channels}")
    print(f"Base/Search:   {args.base_model_path} channels={args.base_channels}")
    print(f"Student input: {input_channels} channels")
    print(f"Base input:    {base_input_channels} channels")
    if init_input_channels is not None:
        print(f"Warm inputs:   {init_input_channels} channels")
    if args.init_global_context:
        print("Warm global:   enabled")
    if args.scoreboard_history:
        print("Score history: enabled")
    elif network_global_context:
        print("Global ctx:    enabled")
    if args.base_scoreboard_history:
        print("Base history:  enabled")
    elif base_network_global_context:
        print("Base global:   enabled")
    if args.freeze_legacy_weights:
        print("Frozen legacy: conv1 old inputs + trunk/heads")
    if args.freeze_strategy_aux_only:
        print("Frozen policy: strategy auxiliary heads only")
    if mixed_learner:
        mixed_p0_envs, mixed_p1_envs = split_mixed_env_counts(args.num_envs)
        learner_label = f"mixed players 0/1 ({mixed_p0_envs}+{mixed_p1_envs} envs)"
    else:
        learner_label = f"player {fixed_learner_player}"
    print(f"Learner:       {learner_label}")
    print(f"Rollout:       {args.num_iterations} x {args.num_steps} steps, envs={args.num_envs}")
    print(
        "Search:        "
        f"top_k={args.top_k}, rollout_steps={args.rollout_steps}, rollouts/action={args.rollouts_per_action}"
    )
    print(
        "Objective:     "
        f"kl={args.kl_weight:g}, improve={args.improve_weight:g}, "
        f"mode={args.target_mode}, soft_weight={args.soft_weight_mode}, "
        f"soft_extra={args.soft_improvement_extra_weight:g}, value={args.search_value_weight:g}, "
        f"outcome={args.search_outcome_weight:g}, "
        f"score_temp={args.score_temperature:g}"
    )
    if args.search_value_weight > 0.0:
        print(f"Search value:  scale={args.search_value_scale:g}")
    if args.search_outcome_weight > 0.0:
        print("Search outcome: enabled")
    if strategy_aux_enabled:
        print(
            "Strategy aux:   "
            f"q={args.strategy_q_weight:g}, intent={args.strategy_intent_weight:g}, "
            f"finish={args.strategy_finish_weight:g}, belief={args.strategy_belief_weight:g}"
        )
    if args.checkpoint_dir is not None and args.checkpoint_every > 0:
        print(f"Checkpoints:   every {args.checkpoint_every} iterations in {args.checkpoint_dir}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, student_key, base_key = jrandom.split(key, 3)
    student_network = load_or_create_adaptive_network(
        student_key,
        pad_size=args.pad_to,
        init_model_path=init_model_path,
        channels=args.channels,
        init_channels=args.init_channels,
        input_channels=input_channels,
        init_input_channels=init_input_channels,
        outcome_head=args.search_outcome_weight > 0.0,
        init_outcome_head=args.init_outcome_head,
        strategy_aux=strategy_aux_enabled,
        init_strategy_aux=args.init_strategy_aux,
        global_context=network_global_context,
        init_global_context=args.init_global_context,
    )
    base_network = load_or_create_adaptive_network(
        base_key,
        pad_size=args.pad_to,
        init_model_path=args.base_model_path,
        channels=args.base_channels,
        input_channels=base_input_channels,
        global_context=base_network_global_context,
    )
    opponent_network = base_network
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(student_network, eqx.is_inexact_array))

    checkpoint_paths = []
    model_stem = Path(args.model_path).stem
    for iteration in range(1, args.num_iterations + 1):
        t0 = time.time()
        key, rollout_key, update_key, pool_key = jrandom.split(key, 4)
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
        states, effective_sizes = make_adaptive_initial_states(pool, args.num_envs)
        if mixed_learner:
            mixed_p0_envs, _ = split_mixed_env_counts(args.num_envs)
            states_p0 = jax.tree.map(lambda x: x[:mixed_p0_envs], states)
            effective_sizes_p0 = effective_sizes[:mixed_p0_envs]
            states_p1 = jax.tree.map(lambda x: x[mixed_p0_envs:], states)
            effective_sizes_p1 = effective_sizes[mixed_p0_envs:]

        if args.target_mode == "hard":
            if mixed_learner:
                _, batch_p0, rollout_key = collect_adaptive_conservative_batch(
                    student_network,
                    base_network,
                    opponent_network,
                    states_p0,
                    effective_sizes_p0,
                    rollout_key,
                    args.num_steps,
                    policy_mode,
                    opponent_policy_mode,
                    0,
                    args.top_k,
                    args.rollout_steps,
                    args.rollouts_per_action,
                    args.army_weight,
                    args.land_weight,
                    args.prior_weight,
                    args.terminal_score,
                    args.min_margin,
                    args.margin_scale,
                    args.max_improve_weight,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    base_network_global_context,
                    args.base_scoreboard_history,
                )
                _, batch_p1, rollout_key = collect_adaptive_conservative_batch(
                    student_network,
                    base_network,
                    opponent_network,
                    states_p1,
                    effective_sizes_p1,
                    rollout_key,
                    args.num_steps,
                    policy_mode,
                    opponent_policy_mode,
                    1,
                    args.top_k,
                    args.rollout_steps,
                    args.rollouts_per_action,
                    args.army_weight,
                    args.land_weight,
                    args.prior_weight,
                    args.terminal_score,
                    args.min_margin,
                    args.margin_scale,
                    args.max_improve_weight,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    base_network_global_context,
                    args.base_scoreboard_history,
                )
                flat_batch = concatenate_flat_batches(
                    flatten_adaptive_conservative_batch(*batch_p0),
                    flatten_adaptive_conservative_batch(*batch_p1),
                )
            else:
                _, batch, rollout_key = collect_adaptive_conservative_batch(
                    student_network,
                    base_network,
                    opponent_network,
                    states,
                    effective_sizes,
                    rollout_key,
                    args.num_steps,
                    policy_mode,
                    opponent_policy_mode,
                    fixed_learner_player,
                    args.top_k,
                    args.rollout_steps,
                    args.rollouts_per_action,
                    args.army_weight,
                    args.land_weight,
                    args.prior_weight,
                    args.terminal_score,
                    args.min_margin,
                    args.margin_scale,
                    args.max_improve_weight,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    base_network_global_context,
                    args.base_scoreboard_history,
                )
                flat_batch = flatten_adaptive_conservative_batch(*batch)
            student_network, opt_state, loss, metrics, update_key = train_adaptive_conservative_epoch(
                student_network,
                base_network,
                opt_state,
                flat_batch,
                optimizer,
                update_key,
                args.num_epochs,
                args.minibatch_size,
                args.kl_weight,
                args.improve_weight,
                args.temperature,
                args.freeze_legacy_weights,
                ADAPTIVE_INPUT_CHANNELS,
            )
        else:
            if mixed_learner:
                _, batch_p0, rollout_key = collect_adaptive_soft_batch(
                    student_network,
                    base_network,
                    opponent_network,
                    states_p0,
                    effective_sizes_p0,
                    rollout_key,
                    args.num_steps,
                    policy_mode,
                    opponent_policy_mode,
                    0,
                    args.top_k,
                    args.rollout_steps,
                    args.rollouts_per_action,
                    args.army_weight,
                    args.land_weight,
                    args.prior_weight,
                    args.terminal_score,
                    args.soft_weight_mode_id,
                    args.min_margin,
                    args.margin_scale,
                    args.max_improve_weight,
                    args.score_temperature,
                    args.search_value_scale,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    base_network_global_context,
                    args.base_scoreboard_history,
                )
                _, batch_p1, rollout_key = collect_adaptive_soft_batch(
                    student_network,
                    base_network,
                    opponent_network,
                    states_p1,
                    effective_sizes_p1,
                    rollout_key,
                    args.num_steps,
                    policy_mode,
                    opponent_policy_mode,
                    1,
                    args.top_k,
                    args.rollout_steps,
                    args.rollouts_per_action,
                    args.army_weight,
                    args.land_weight,
                    args.prior_weight,
                    args.terminal_score,
                    args.soft_weight_mode_id,
                    args.min_margin,
                    args.margin_scale,
                    args.max_improve_weight,
                    args.score_temperature,
                    args.search_value_scale,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    base_network_global_context,
                    args.base_scoreboard_history,
                )
                flat_batch = concatenate_flat_batches(
                    flatten_adaptive_soft_batch(*batch_p0),
                    flatten_adaptive_soft_batch(*batch_p1),
                )
            else:
                _, batch, rollout_key = collect_adaptive_soft_batch(
                    student_network,
                    base_network,
                    opponent_network,
                    states,
                    effective_sizes,
                    rollout_key,
                    args.num_steps,
                    policy_mode,
                    opponent_policy_mode,
                    fixed_learner_player,
                    args.top_k,
                    args.rollout_steps,
                    args.rollouts_per_action,
                    args.army_weight,
                    args.land_weight,
                    args.prior_weight,
                    args.terminal_score,
                    args.soft_weight_mode_id,
                    args.min_margin,
                    args.margin_scale,
                    args.max_improve_weight,
                    args.score_temperature,
                    args.search_value_scale,
                    args.pad_to,
                    network_global_context,
                    args.scoreboard_history,
                    base_network_global_context,
                    args.base_scoreboard_history,
                )
                flat_batch = flatten_adaptive_soft_batch(*batch)
            student_network, opt_state, loss, metrics, update_key = train_adaptive_soft_epoch(
                student_network,
                base_network,
                opt_state,
                flat_batch,
                optimizer,
                update_key,
                args.num_epochs,
                args.minibatch_size,
                args.kl_weight,
                args.improve_weight,
                args.soft_improvement_extra_weight,
                args.search_value_weight,
                args.search_outcome_weight,
                args.strategy_q_weight,
                args.strategy_intent_weight,
                args.strategy_finish_weight,
                args.strategy_belief_weight,
                args.temperature,
                args.freeze_legacy_weights,
                args.freeze_strategy_aux_only,
                ADAPTIVE_INPUT_CHANNELS,
            )
        jax.block_until_ready(student_network)

        if args.checkpoint_dir is not None and args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, model_stem, iteration)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(checkpoint_path, student_network)
            checkpoint_paths.append(checkpoint_path)
            prune_old_checkpoints(checkpoint_paths, args.keep_checkpoints)

        if iteration % 5 == 0 or iteration == 1 or iteration == args.num_iterations:
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.5f} | "
                f"KL: {float(metrics['kl_loss']):.5f} | "
                f"Improve: {float(metrics['improve_loss']):.4f} | "
                f"Value: {float(metrics.get('search_value_loss', 0.0)):.4f} | "
                f"Outcome: {float(metrics.get('search_outcome_loss', 0.0)):.4f} | "
                f"StratQ: {float(metrics.get('strategy_q_loss', 0.0)):.4f} | "
                f"Intent: {float(metrics.get('strategy_intent_loss', 0.0)):.4f} | "
                f"Belief: {float(metrics.get('strategy_belief_loss', 0.0)):.4f} | "
                f"Selected: {int(metrics['selected_samples']):5d}/{samples} "
                f"({float(metrics['selected_fraction']) * 100:4.1f}%) | "
                f"Margin: {float(metrics['mean_selected_margin']):6.1f} | "
                f"SPS: {samples / elapsed:7.0f} | Time: {elapsed:.2f}s"
            )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, student_network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
