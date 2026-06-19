"""Offline supervised training for adaptive strategy auxiliary heads."""

from __future__ import annotations

import argparse
import glob
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
import optax

from adaptive_common import parse_grid_sizes
from adaptive_network import load_or_create_adaptive_network
from adaptive_search_distill import binary_cross_entropy_with_logits
from generals.agents.ppo_policy_agent import parse_policy_channels

OUTCOME_LOSS = 0
OUTCOME_DRAW = 1
OUTCOME_WIN = 2
ACTION_CE_WEIGHT_MODES = ("all", "non-draw", "wins", "search-best-win")
BALANCE_STRATA_MODES = ("none", "size-seat", "size-seat-domain")
LABEL_SOURCE_MODES = ("trajectory", "search-best")


def expand_dataset_paths(patterns: list[str]) -> list[Path]:
    """Expand explicit paths or glob patterns into a stable shard list."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique = sorted(dict.fromkeys(paths))
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset shard not found: {missing[0]}")
    return unique


def dataset_domain_name(path: Path) -> str:
    """Return a coarse data-domain label for balancing mixed offline shards."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        opponent_policy_path = metadata.get("opponent_policy_path")
        if opponent_policy_path:
            return f"policy:{Path(opponent_policy_path).name}"
        opponent = metadata.get("opponent")
        if opponent:
            return f"opponent:{opponent}"
    name = path.parent.name
    if "fixed-v5" in name:
        return "policy:fixed-v5"
    if "expander" in name:
        return "opponent:expander"
    return f"path:{name}"


def load_strategy_dataset(
    paths: list[Path],
    max_samples: int | None = None,
    max_samples_per_shard: int | None = None,
    seed: int = 0,
    finish_head_mode: str = "binary",
    action_ce_weight_mode: str = "all",
    label_source: str = "trajectory",
) -> dict[str, jnp.ndarray]:
    """Load the subset of NPZ fields needed by the frozen-head trainer."""
    rng = np.random.default_rng(seed)
    search_candidate_count = 1
    for path in paths:
        with np.load(path) as shard:
            if "search_candidate_indices" in shard:
                search_candidate_count = max(search_candidate_count, int(shard["search_candidate_indices"].shape[1]))
    chunks: dict[str, list[np.ndarray]] = {
        "obs": [],
        "legal_mask": [],
        "active": [],
        "intent": [],
        "finish": [],
        "finish_weight": [],
        "outcome": [],
        "outcome_weight": [],
        "enemy_general": [],
        "source_heatmap": [],
        "target_heatmap": [],
        "teacher_logits": [],
        "teacher_action": [],
        "action_weight": [],
        "grid_size": [],
        "seat": [],
        "domain": [],
        "search_candidate_indices": [],
        "search_prior_scores": [],
        "search_scores": [],
        "search_outcomes": [],
        "search_score_gap": [],
    }
    domain_to_id: dict[str, int] = {}
    for path in paths:
        shard = np.load(path)
        domain_name = dataset_domain_name(path)
        domain_id = domain_to_id.setdefault(domain_name, len(domain_to_id))
        shard_samples = shard["obs"].shape[0]
        shard_indices = np.arange(shard_samples)
        if max_samples_per_shard is not None and shard_samples > max_samples_per_shard:
            shard_indices = np.sort(rng.choice(shard_samples, size=max_samples_per_shard, replace=False))
        chunks["obs"].append(shard["obs"][shard_indices].astype(np.float32))
        chunks["legal_mask"].append(shard["legal_mask"][shard_indices].astype(np.bool_))
        chunks["active"].append(shard["active"][shard_indices].astype(np.bool_))
        chunks["intent"].append(shard["intent"][shard_indices].astype(np.int32))
        trajectory_outcome = shard["outcome"][shard_indices].astype(np.int32)
        trajectory_outcome_weight = shard["outcome_known"][shard_indices].astype(np.float32)
        if "search_best_outcome" in shard:
            search_best_outcome = shard["search_best_outcome"][shard_indices].astype(np.int32)
        else:
            search_best_outcome = np.full((shard_indices.shape[0],), -1, dtype=np.int32)
        if label_source == "search-best":
            if np.any(search_best_outcome >= 0):
                label_known = search_best_outcome >= 0
                outcome_target = np.where(label_known, search_best_outcome, OUTCOME_DRAW).astype(np.int32)
                outcome_weight = label_known.astype(np.float32)
                finish_target = (search_best_outcome == OUTCOME_WIN).astype(np.float32)
            else:
                shard_count = shard_indices.shape[0]
                outcome_target = np.full((shard_count,), OUTCOME_DRAW, dtype=np.int32)
                outcome_weight = np.zeros((shard_count,), dtype=np.float32)
                finish_target = np.zeros((shard_count,), dtype=np.float32)
        else:
            outcome_target = trajectory_outcome
            outcome_weight = trajectory_outcome_weight
            finish_target = shard["finish_within_250"][shard_indices].astype(np.float32)
        if finish_head_mode == "multi-horizon":
            if label_source == "search-best":
                # Search-best labels are horizon-free; repeat the same target so
                # multi-output checkpoints can still learn the search win signal.
                finish_targets = np.repeat(finish_target[:, None], 3, axis=1)
            else:
                finish_targets = np.stack(
                    [
                        shard["finish_within_50"][shard_indices],
                        shard["finish_within_100"][shard_indices],
                        shard["finish_within_250"][shard_indices],
                    ],
                    axis=-1,
                )
            chunks["finish"].append(finish_targets.astype(np.float32))
        else:
            chunks["finish"].append((finish_target > 0.5).astype(np.int32))
        chunks["finish_weight"].append(outcome_weight.astype(np.float32))
        chunks["outcome"].append(outcome_target.astype(np.int32))
        chunks["outcome_weight"].append(outcome_weight.astype(np.float32))
        chunks["enemy_general"].append(shard["enemy_general_heatmap"][shard_indices].astype(np.float32))
        chunks["source_heatmap"].append(shard["source_heatmap"][shard_indices].astype(np.float32))
        chunks["target_heatmap"].append(shard["target_heatmap"][shard_indices].astype(np.float32))
        chunks["teacher_logits"].append(shard["teacher_logits"][shard_indices].astype(np.float32))
        chunks["teacher_action"].append(shard["teacher_action_index"][shard_indices].astype(np.int32))
        chunks["grid_size"].append(shard["grid_size"][shard_indices].astype(np.int32))
        chunks["seat"].append(shard["seat"][shard_indices].astype(np.int32))
        chunks["domain"].append(np.full((shard_indices.shape[0],), domain_id, dtype=np.int32))
        shard_count = shard_indices.shape[0]
        if "search_candidate_indices" in shard:
            candidate_indices = shard["search_candidate_indices"][shard_indices].astype(np.int32)
            prior_scores = shard["search_prior_scores"][shard_indices].astype(np.float32)
            search_scores = shard["search_scores"][shard_indices].astype(np.float32)
            if "search_outcomes" in shard:
                search_outcomes = shard["search_outcomes"][shard_indices].astype(np.int32)
            else:
                search_outcomes = np.full_like(candidate_indices, -1, dtype=np.int32)
            if candidate_indices.shape[1] < search_candidate_count:
                pad_width = search_candidate_count - candidate_indices.shape[1]
                candidate_indices = np.pad(candidate_indices, ((0, 0), (0, pad_width)), constant_values=0)
                prior_scores = np.pad(prior_scores, ((0, 0), (0, pad_width)), constant_values=-1.0e4)
                search_scores = np.pad(search_scores, ((0, 0), (0, pad_width)), constant_values=-1.0e4)
                search_outcomes = np.pad(search_outcomes, ((0, 0), (0, pad_width)), constant_values=-1)
            chunks["search_candidate_indices"].append(candidate_indices[:, :search_candidate_count])
            chunks["search_prior_scores"].append(prior_scores[:, :search_candidate_count])
            chunks["search_scores"].append(search_scores[:, :search_candidate_count])
            chunks["search_outcomes"].append(search_outcomes[:, :search_candidate_count])
            chunks["search_score_gap"].append(shard["search_score_gap"][shard_indices].astype(np.float32))
        else:
            chunks["search_candidate_indices"].append(np.zeros((shard_count, search_candidate_count), dtype=np.int32))
            chunks["search_prior_scores"].append(
                np.full((shard_count, search_candidate_count), -1.0e4, dtype=np.float32)
            )
            chunks["search_scores"].append(np.full((shard_count, search_candidate_count), -1.0e4, dtype=np.float32))
            chunks["search_outcomes"].append(np.full((shard_count, search_candidate_count), -1, dtype=np.int32))
            chunks["search_score_gap"].append(np.zeros((shard_count,), dtype=np.float32))
        outcome = trajectory_outcome
        outcome_known = trajectory_outcome_weight > 0.0
        if action_ce_weight_mode == "non-draw":
            action_weight = ~(outcome_known & (outcome == OUTCOME_DRAW))
        elif action_ce_weight_mode == "wins":
            action_weight = outcome_known & (outcome == OUTCOME_WIN)
        elif action_ce_weight_mode == "search-best-win":
            action_weight = search_best_outcome == OUTCOME_WIN
        else:
            action_weight = np.ones_like(outcome, dtype=np.bool_)
        chunks["action_weight"].append(action_weight.astype(np.float32))

    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_samples is not None:
        arrays = {name: value[:max_samples] for name, value in arrays.items()}
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def balance_strategy_dataset(dataset: dict[str, jnp.ndarray], mode: str, seed: int) -> dict[str, jnp.ndarray]:
    """Downsample strategy rows to equal task strata before JAX training."""
    if mode == "none":
        return dataset
    if mode not in ("size-seat", "size-seat-domain"):
        raise ValueError(f"unknown balance mode: {mode}")
    grid_size = np.asarray(dataset["grid_size"])
    seat = np.asarray(dataset["seat"])
    domain = np.asarray(dataset["domain"])
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    for size in sorted(np.unique(grid_size)):
        for player in sorted(np.unique(seat)):
            if mode == "size-seat-domain":
                for domain_id in sorted(np.unique(domain)):
                    indices = np.flatnonzero((grid_size == size) & (seat == player) & (domain == domain_id))
                    if indices.size > 0:
                        groups.append(indices)
            else:
                indices = np.flatnonzero((grid_size == size) & (seat == player))
                if indices.size > 0:
                    groups.append(indices)
    if not groups:
        return dataset
    target_count = min(group.size for group in groups)
    selected = np.concatenate(
        [np.sort(rng.choice(group, size=target_count, replace=False)) for group in groups],
        axis=0,
    )
    rng.shuffle(selected)
    return {name: value[selected] for name, value in dataset.items()}


def mask_strategy_supervised_grads(grads, keep_outcome: bool, keep_value_bottleneck: bool = False):
    """Keep gradients only for selected supervised heads and optional pooled bottleneck."""
    masked = jax.tree.map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, grads)
    if keep_value_bottleneck:
        masked = eqx.tree_at(lambda net: net.value_linear1, masked, grads.value_linear1)
    if keep_outcome and grads.outcome_linear2 is not None:
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
    if grads.strategy_source_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_source_conv, masked, grads.strategy_source_conv)
    if grads.strategy_target_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_target_conv, masked, grads.strategy_target_conv)
    return masked


def spatial_ce_metrics(logits: jnp.ndarray, targets: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cross-entropy and argmax accuracy for one-hot or soft spatial heatmap targets."""
    active_f = active.astype(jnp.float32)
    target_mass = jnp.sum(targets * active_f, axis=(1, 2))
    valid = target_mass > 1.0e-6
    target_probs = targets * active_f / jnp.maximum(target_mass[:, None, None], 1.0e-6)
    masked_logits = jnp.where(active, logits, -1.0e9).reshape(logits.shape[0], -1)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1).reshape(logits.shape)
    per_sample_loss = -jnp.sum(target_probs * log_probs, axis=(1, 2))
    normalizer = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    loss = jnp.sum(per_sample_loss * valid.astype(jnp.float32)) / normalizer

    predicted = jnp.argmax(masked_logits, axis=-1)
    target_index = jnp.argmax(target_probs.reshape(targets.shape[0], -1), axis=-1)
    accuracy = jnp.sum((predicted == target_index).astype(jnp.float32) * valid.astype(jnp.float32)) / normalizer
    return loss, accuracy


def search_q_rank_metrics(
    action_q_values: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    prior_scores: jnp.ndarray,
    search_scores: jnp.ndarray,
    score_gaps: jnp.ndarray,
    temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit action-Q values to search top-k score rankings without policy CE."""
    action_count = action_q_values.shape[1]
    valid = (candidate_indices >= 0) & (candidate_indices < action_count) & (prior_scores > -9999.0)
    valid_count = jnp.sum(valid.astype(jnp.float32), axis=1)
    sample_weight = ((valid_count > 1.0) & (score_gaps > 0.0)).astype(jnp.float32)
    safe_indices = jnp.clip(candidate_indices, 0, action_count - 1)
    candidate_q = jnp.take_along_axis(action_q_values, safe_indices, axis=1)
    target_logits = jnp.where(valid, search_scores / temperature, -1.0e9)
    target_log_probs = jax.nn.log_softmax(target_logits, axis=1)
    target_probs = jnp.exp(target_log_probs)
    q_log_probs = jax.nn.log_softmax(jnp.where(valid, candidate_q, -1.0e9), axis=1)
    per_sample_loss = -jnp.sum(target_probs * q_log_probs, axis=1)
    normalizer = jnp.maximum(jnp.sum(sample_weight), 1.0)
    loss = jnp.sum(per_sample_loss * sample_weight) / normalizer
    pred_best = jnp.argmax(jnp.where(valid, candidate_q, -1.0e9), axis=1)
    target_best = jnp.argmax(target_logits, axis=1)
    accuracy = jnp.sum((pred_best == target_best).astype(jnp.float32) * sample_weight) / normalizer
    entropy = -jnp.sum(target_probs * jnp.log(jnp.clip(target_probs, 1.0e-8, 1.0)), axis=1)
    entropy = jnp.sum(entropy * sample_weight) / normalizer
    weight_mean = jnp.mean(sample_weight)
    return loss, accuracy, entropy, weight_mean


def search_q_value_metrics(
    action_q_values: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    prior_scores: jnp.ndarray,
    search_scores: jnp.ndarray,
    search_outcomes: jnp.ndarray,
    score_gaps: jnp.ndarray,
    score_scale: float,
    outcome_score_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Regress candidate action-Q values to search outcome values."""
    action_count = action_q_values.shape[1]
    valid = (
        (candidate_indices >= 0)
        & (candidate_indices < action_count)
        & (prior_scores > -9999.0)
        & (search_outcomes >= OUTCOME_LOSS)
        & (search_outcomes <= OUTCOME_WIN)
    )
    valid_count = jnp.sum(valid.astype(jnp.float32), axis=1)
    sample_weight = ((valid_count > 0.0) & (score_gaps > 0.0)).astype(jnp.float32)
    safe_indices = jnp.clip(candidate_indices, 0, action_count - 1)
    candidate_q = jnp.take_along_axis(action_q_values, safe_indices, axis=1)
    outcome_targets = search_outcomes.astype(jnp.float32) - float(OUTCOME_DRAW)
    score_targets = jnp.tanh(search_scores / score_scale)
    targets = outcome_targets + outcome_score_weight * score_targets
    squared_error = jnp.square(candidate_q - targets)
    per_sample_loss = jnp.sum(jnp.where(valid, squared_error, 0.0), axis=1) / jnp.maximum(valid_count, 1.0)
    normalizer = jnp.maximum(jnp.sum(sample_weight), 1.0)
    loss = jnp.sum(per_sample_loss * sample_weight) / normalizer
    pred_best = jnp.argmax(jnp.where(valid, candidate_q, -1.0e9), axis=1)
    target_best = jnp.argmax(jnp.where(valid, targets, -1.0e9), axis=1)
    accuracy = jnp.sum((pred_best == target_best).astype(jnp.float32) * sample_weight) / normalizer
    return loss, accuracy, jnp.mean(sample_weight)


def binary_balance_weights(labels: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Return per-label weights that give positive and negative labels equal mass."""
    labels_f = labels.astype(jnp.float32)
    weights_f = weights.astype(jnp.float32)
    positives = jnp.sum(weights_f * labels_f)
    negatives = jnp.sum(weights_f * (1.0 - labels_f))
    total = positives + negatives
    positive_scale = total / jnp.maximum(2.0 * positives, 1.0)
    negative_scale = total / jnp.maximum(2.0 * negatives, 1.0)
    return jnp.where(labels_f > 0.5, positive_scale, negative_scale)


def class_balance_weights(targets: jnp.ndarray, weights: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Return inverse-frequency per-sample class weights for present classes."""
    weights_f = weights.astype(jnp.float32)
    class_ids = jnp.arange(num_classes)
    counts = jnp.sum((targets[:, None] == class_ids[None, :]).astype(jnp.float32) * weights_f[:, None], axis=0)
    present = counts > 0.0
    present_count = jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)
    total = jnp.sum(counts)
    scales = jnp.where(present, total / jnp.maximum(present_count * counts, 1.0), 0.0)
    return scales[targets]


@eqx.filter_jit
def train_step(
    network,
    opt_state,
    batch,
    optimizer,
    intent_weight: float,
    finish_weight: float,
    belief_weight: float,
    outcome_weight: float,
    policy_kl_weight: float,
    action_ce_weight: float,
    q_kl_weight: float,
    q_action_ce_weight: float,
    search_q_rank_weight: float,
    search_q_temperature: float,
    search_q_value_weight: float,
    search_q_score_scale: float,
    search_q_outcome_score_weight: float,
    source_weight: float,
    target_weight: float,
    balance_finish_labels: bool,
    balance_outcome_labels: bool,
    freeze_base: bool,
    train_value_bottleneck: bool,
    multi_horizon_finish: bool,
):
    """Train one minibatch of frozen-trunk strategy auxiliary losses."""
    (
        obs,
        masks,
        active,
        intent_targets,
        finish_targets,
        finish_weights,
        outcome_targets,
        outcome_weights,
        enemy_general,
        source_heatmap,
        target_heatmap,
        teacher_logits,
        teacher_actions,
        action_weights,
        search_candidate_indices,
        search_prior_scores,
        search_scores,
        search_outcomes,
        search_score_gaps,
    ) = batch

    def loss_fn(net):
        outputs = jax.vmap(lambda o, m, a: net.strategy_auxiliary(o, m, a))(obs, masks, active)
        teacher_legal = teacher_logits > -9999.0

        intent_log_probs = jax.nn.log_softmax(outputs.intent_logits, axis=-1)
        intent_losses = -intent_log_probs[jnp.arange(intent_log_probs.shape[0]), intent_targets]
        intent_loss = jnp.mean(intent_losses)
        intent_accuracy = jnp.mean((jnp.argmax(outputs.intent_logits, axis=-1) == intent_targets).astype(jnp.float32))

        finish_normalizer = jnp.maximum(jnp.sum(finish_weights), 1.0)
        if multi_horizon_finish:
            finish_losses = binary_cross_entropy_with_logits(outputs.finish_logits, finish_targets)
            finish_label_weights = jnp.where(
                balance_finish_labels,
                binary_balance_weights(finish_targets, finish_weights[:, None]),
                1.0,
            )
            weighted_finish = finish_losses * finish_weights[:, None] * finish_label_weights
            finish_loss = jnp.sum(weighted_finish)
            finish_loss = finish_loss / jnp.maximum(jnp.sum(finish_weights[:, None] * finish_label_weights), 1.0)
            finish_predictions = (jax.nn.sigmoid(outputs.finish_logits) >= 0.5).astype(jnp.float32)
            finish_accuracy = jnp.sum(
                (finish_predictions == finish_targets).astype(jnp.float32) * finish_weights[:, None]
            )
            finish_accuracy = finish_accuracy / jnp.maximum(finish_normalizer * finish_targets.shape[-1], 1.0)
        else:
            finish_log_probs = jax.nn.log_softmax(outputs.finish_logits, axis=-1)
            finish_losses = -finish_log_probs[jnp.arange(finish_log_probs.shape[0]), finish_targets]
            finish_label_weights = jnp.where(
                balance_finish_labels,
                binary_balance_weights(finish_targets, finish_weights),
                1.0,
            )
            finish_loss = jnp.sum(finish_losses * finish_weights * finish_label_weights)
            finish_loss = finish_loss / jnp.maximum(jnp.sum(finish_weights * finish_label_weights), 1.0)
            finish_accuracy = jnp.sum(
                (jnp.argmax(outputs.finish_logits, axis=-1) == finish_targets).astype(jnp.float32) * finish_weights
            )
            finish_accuracy = finish_accuracy / finish_normalizer

        active_f = active.astype(jnp.float32)
        belief_per_cell = binary_cross_entropy_with_logits(outputs.enemy_general_logits, enemy_general)
        belief_per_sample = jnp.sum(belief_per_cell * active_f, axis=(1, 2)) / jnp.maximum(
            jnp.sum(active_f, axis=(1, 2)),
            1.0,
        )
        belief_loss = jnp.mean(belief_per_sample)

        outcome_loss = jnp.asarray(0.0, dtype=jnp.float32)
        outcome_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if outcome_weight > 0.0:
            _, _, _, outcome_logits = jax.vmap(lambda o, m, a: net.logits_value_auxiliary(o, m, a))(obs, masks, active)
            outcome_log_probs = jax.nn.log_softmax(outcome_logits, axis=-1)
            outcome_losses = -outcome_log_probs[jnp.arange(outcome_log_probs.shape[0]), outcome_targets]
            outcome_normalizer = jnp.maximum(jnp.sum(outcome_weights), 1.0)
            outcome_label_weights = jnp.where(
                balance_outcome_labels,
                class_balance_weights(outcome_targets, outcome_weights, 3),
                1.0,
            )
            outcome_loss = jnp.sum(outcome_losses * outcome_weights * outcome_label_weights)
            outcome_loss = outcome_loss / jnp.maximum(jnp.sum(outcome_weights * outcome_label_weights), 1.0)
            outcome_accuracy = jnp.sum(
                (jnp.argmax(outcome_logits, axis=-1) == outcome_targets).astype(jnp.float32) * outcome_weights
            )
            outcome_accuracy = outcome_accuracy / outcome_normalizer

        policy_kl = jnp.asarray(0.0, dtype=jnp.float32)
        action_ce = jnp.asarray(0.0, dtype=jnp.float32)
        teacher_action_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if policy_kl_weight > 0.0 or action_ce_weight > 0.0:
            student_logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs, masks, active)
            masked_teacher_logits = jnp.where(teacher_legal, teacher_logits, -1.0e9)
            teacher_log_probs = jax.nn.log_softmax(masked_teacher_logits, axis=-1)
            teacher_probs = jnp.exp(teacher_log_probs)
            student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
            policy_kl_per_sample = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
            policy_kl = jnp.mean(policy_kl_per_sample)

            action_ce_losses = -student_log_probs[jnp.arange(student_log_probs.shape[0]), teacher_actions]
            action_normalizer = jnp.maximum(jnp.sum(action_weights), 1.0)
            action_ce = jnp.sum(action_ce_losses * action_weights) / action_normalizer
            teacher_action_accuracy = jnp.sum(
                (jnp.argmax(student_logits, axis=-1) == teacher_actions).astype(jnp.float32) * action_weights
            ) / action_normalizer

        q_policy_kl = jnp.asarray(0.0, dtype=jnp.float32)
        q_action_ce = jnp.asarray(0.0, dtype=jnp.float32)
        q_action_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if q_kl_weight > 0.0 or q_action_ce_weight > 0.0:
            masked_teacher_logits = jnp.where(teacher_legal, teacher_logits, -1.0e9)
            teacher_log_probs = jax.nn.log_softmax(masked_teacher_logits, axis=-1)
            teacher_probs = jnp.exp(teacher_log_probs)
            q_logits = jnp.where(teacher_legal, outputs.action_q_values, -1.0e9)
            q_log_probs = jax.nn.log_softmax(q_logits, axis=-1)
            q_policy_kl_per_sample = jnp.sum(teacher_probs * (teacher_log_probs - q_log_probs), axis=-1)
            q_policy_kl = jnp.mean(q_policy_kl_per_sample)

            q_action_ce_losses = -q_log_probs[jnp.arange(q_log_probs.shape[0]), teacher_actions]
            action_normalizer = jnp.maximum(jnp.sum(action_weights), 1.0)
            q_action_ce = jnp.sum(q_action_ce_losses * action_weights) / action_normalizer
            q_action_accuracy = jnp.sum(
                (jnp.argmax(q_logits, axis=-1) == teacher_actions).astype(jnp.float32) * action_weights
            ) / action_normalizer

        search_q_rank_loss = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_rank_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_target_entropy = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_weight_mean = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_value_loss = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_value_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_value_weight_mean = jnp.asarray(0.0, dtype=jnp.float32)
        if search_q_rank_weight > 0.0:
            (
                search_q_rank_loss,
                search_q_rank_accuracy,
                search_q_target_entropy,
                search_q_weight_mean,
            ) = search_q_rank_metrics(
                outputs.action_q_values,
                search_candidate_indices,
                search_prior_scores,
                search_scores,
                search_score_gaps,
                search_q_temperature,
            )
        if search_q_value_weight > 0.0:
            (
                search_q_value_loss,
                search_q_value_accuracy,
                search_q_value_weight_mean,
            ) = search_q_value_metrics(
                outputs.action_q_values,
                search_candidate_indices,
                search_prior_scores,
                search_scores,
                search_outcomes,
                search_score_gaps,
                search_q_score_scale,
                search_q_outcome_score_weight,
            )

        source_loss = jnp.asarray(0.0, dtype=jnp.float32)
        source_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        target_loss = jnp.asarray(0.0, dtype=jnp.float32)
        target_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if source_weight > 0.0 or target_weight > 0.0:
            if outputs.source_logits is None or outputs.target_logits is None:
                raise ValueError("source/target losses require strategy_spatial_aux")
            source_loss, source_accuracy = spatial_ce_metrics(outputs.source_logits, source_heatmap, active)
            target_loss, target_accuracy = spatial_ce_metrics(outputs.target_logits, target_heatmap, active)

        loss = (
            intent_weight * intent_loss
            + finish_weight * finish_loss
            + belief_weight * belief_loss
            + outcome_weight * outcome_loss
            + policy_kl_weight * policy_kl
            + action_ce_weight * action_ce
            + q_kl_weight * q_policy_kl
            + q_action_ce_weight * q_action_ce
            + search_q_rank_weight * search_q_rank_loss
            + search_q_value_weight * search_q_value_loss
            + source_weight * source_loss
            + target_weight * target_loss
        )
        metrics = {
            "intent_loss": intent_loss,
            "finish_loss": finish_loss,
            "belief_loss": belief_loss,
            "outcome_loss": outcome_loss,
            "policy_kl": policy_kl,
            "action_ce": action_ce,
            "q_policy_kl": q_policy_kl,
            "q_action_ce": q_action_ce,
            "search_q_rank_loss": search_q_rank_loss,
            "search_q_value_loss": search_q_value_loss,
            "source_loss": source_loss,
            "target_loss": target_loss,
            "intent_accuracy": intent_accuracy,
            "finish_accuracy": finish_accuracy,
            "outcome_accuracy": outcome_accuracy,
            "teacher_action_accuracy": teacher_action_accuracy,
            "q_action_accuracy": q_action_accuracy,
            "search_q_rank_accuracy": search_q_rank_accuracy,
            "search_q_value_accuracy": search_q_value_accuracy,
            "search_q_target_entropy": search_q_target_entropy,
            "source_accuracy": source_accuracy,
            "target_accuracy": target_accuracy,
            "finish_weight_mean": jnp.mean(finish_weights),
            "outcome_weight_mean": jnp.mean(outcome_weights),
            "action_weight_mean": jnp.mean(action_weights),
            "search_q_weight_mean": search_q_weight_mean,
            "search_q_value_weight_mean": search_q_value_weight_mean,
        }
        return loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    if freeze_base:
        grads = mask_strategy_supervised_grads(grads, outcome_weight > 0.0, train_value_bottleneck)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss, metrics


def train_epoch(
    network,
    opt_state,
    dataset,
    optimizer,
    key,
    minibatch_size: int,
    intent_weight: float,
    finish_weight: float,
    belief_weight: float,
    outcome_weight: float,
    policy_kl_weight: float,
    action_ce_weight: float,
    q_kl_weight: float,
    q_action_ce_weight: float,
    search_q_rank_weight: float,
    search_q_temperature: float,
    search_q_value_weight: float,
    search_q_score_scale: float,
    search_q_outcome_score_weight: float,
    source_weight: float,
    target_weight: float,
    balance_finish_labels: bool,
    balance_outcome_labels: bool,
    freeze_base: bool,
    train_value_bottleneck: bool,
    multi_horizon_finish: bool,
):
    """Shuffle one full pass over the loaded shards."""
    num_samples = dataset["obs"].shape[0]
    permutation = jrandom.permutation(key, num_samples)
    num_batches = max(num_samples // minibatch_size, 1)
    metrics_sum = None
    loss_sum = 0.0
    for batch_index in range(num_batches):
        start = batch_index * minibatch_size
        end = min(start + minibatch_size, num_samples)
        idx = permutation[start:end]
        batch = (
            dataset["obs"][idx],
            dataset["legal_mask"][idx],
            dataset["active"][idx],
            dataset["intent"][idx],
            dataset["finish"][idx],
            dataset["finish_weight"][idx],
            dataset["outcome"][idx],
            dataset["outcome_weight"][idx],
            dataset["enemy_general"][idx],
            dataset["source_heatmap"][idx],
            dataset["target_heatmap"][idx],
            dataset["teacher_logits"][idx],
            dataset["teacher_action"][idx],
            dataset["action_weight"][idx],
            dataset["search_candidate_indices"][idx],
            dataset["search_prior_scores"][idx],
            dataset["search_scores"][idx],
            dataset["search_outcomes"][idx],
            dataset["search_score_gap"][idx],
        )
        network, opt_state, loss, metrics = train_step(
            network,
            opt_state,
            batch,
            optimizer,
            intent_weight,
            finish_weight,
            belief_weight,
            outcome_weight,
            policy_kl_weight,
            action_ce_weight,
            q_kl_weight,
            q_action_ce_weight,
            search_q_rank_weight,
            search_q_temperature,
            search_q_value_weight,
            search_q_score_scale,
            search_q_outcome_score_weight,
            source_weight,
            target_weight,
            balance_finish_labels,
            balance_outcome_labels,
            freeze_base,
            train_value_bottleneck,
            multi_horizon_finish,
        )
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return network, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train adaptive strategy auxiliary heads from NPZ shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--balance-strata", choices=BALANCE_STRATA_MODES, default="none")
    parser.add_argument(
        "--label-source",
        choices=LABEL_SOURCE_MODES,
        default="trajectory",
        help="Use trajectory outcome labels or rollout-search best-action outcome labels for finish/outcome heads.",
    )
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="unet")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--input-channels", type=int, default=35)
    parser.add_argument("--init-input-channels", type=int, default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--init-value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--value-head-sizes", default="8,12,16")
    parser.add_argument("--init-value-head-sizes", default="8,12,16")
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--init-value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--init-value-bins", type=int, default=None)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--init-outcome-head", action="store_true")
    parser.add_argument("--init-strategy-aux", action="store_true")
    parser.add_argument("--finish-head-mode", choices=("binary", "multi-horizon"), default="binary")
    parser.add_argument("--init-finish-head-mode", choices=("binary", "multi-horizon"), default="binary")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--model-path", default="runs/adaptive-strategy-supervised/generals-adaptive-strategy-supervised.eqx")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--update-scope", choices=("strategy-heads", "strategy-value-heads", "all"), default="strategy-heads")
    parser.add_argument("--intent-weight", type=float, default=0.2)
    parser.add_argument("--finish-weight", type=float, default=0.4)
    parser.add_argument("--belief-weight", type=float, default=0.3)
    parser.add_argument("--outcome-weight", type=float, default=0.0)
    parser.add_argument("--balance-finish-labels", action="store_true")
    parser.add_argument("--balance-outcome-labels", action="store_true")
    parser.add_argument("--policy-kl-weight", type=float, default=0.0)
    parser.add_argument("--action-ce-weight", type=float, default=0.0)
    parser.add_argument("--action-ce-weight-mode", choices=ACTION_CE_WEIGHT_MODES, default="all")
    parser.add_argument("--q-kl-weight", type=float, default=0.0)
    parser.add_argument("--q-action-ce-weight", type=float, default=0.0)
    parser.add_argument("--search-q-rank-weight", type=float, default=0.0)
    parser.add_argument(
        "--search-q-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for rollout-search top-k score targets.",
    )
    parser.add_argument(
        "--search-q-value-weight",
        type=float,
        default=0.0,
        help="MSE weight for fitting strategy-Q values to rollout-search candidate outcomes.",
    )
    parser.add_argument(
        "--search-q-score-scale",
        type=float,
        default=1000.0,
        help="Scale for optional tanh(search_score / scale) tie-break in search-Q value targets.",
    )
    parser.add_argument(
        "--search-q-outcome-score-weight",
        type=float,
        default=0.0,
        help="Optional shaped-score tie-break weight added to loss/draw/win outcome targets.",
    )
    parser.add_argument("--source-weight", type=float, default=0.0)
    parser.add_argument("--target-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.channels = parse_policy_channels(args.channels)
        args.init_channels = parse_policy_channels(args.init_channels) if args.init_channels is not None else None
        args.value_head_sizes = parse_grid_sizes(args.value_head_sizes)
        args.init_value_head_sizes = parse_grid_sizes(args.init_value_head_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_samples_per_shard is not None and args.max_samples_per_shard <= 0:
        parser.error("--max-samples-per-shard must be positive")
    if args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.num_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("--num-epochs and --minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.weight_decay != 0.0 and args.update_scope != "all":
        parser.error("--weight-decay must stay 0 because this trainer freezes most parameters with a gradient mask")
    if args.value_loss == "hl-gauss" and args.value_bins <= 1:
        parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
    if args.init_value_loss == "hl-gauss":
        init_bins = args.value_bins if args.init_value_bins is None else args.init_value_bins
        if init_bins <= 1:
            parser.error("--init-value-bins must be greater than 1 for --init-value-loss hl-gauss")
    elif args.init_value_bins is not None:
        parser.error("--init-value-bins requires --init-value-loss hl-gauss")
    if any(
        weight < 0.0
        for weight in (
            args.intent_weight,
            args.finish_weight,
            args.belief_weight,
            args.outcome_weight,
            args.policy_kl_weight,
            args.action_ce_weight,
            args.q_kl_weight,
            args.q_action_ce_weight,
            args.search_q_rank_weight,
            args.search_q_value_weight,
            args.source_weight,
            args.target_weight,
        )
    ):
        parser.error("loss weights must be non-negative")
    if args.search_q_temperature <= 0.0:
        parser.error("--search-q-temperature must be positive")
    if args.search_q_score_scale <= 0.0:
        parser.error("--search-q-score-scale must be positive")
    if args.search_q_outcome_score_weight < 0.0:
        parser.error("--search-q-outcome-score-weight must be non-negative")
    if args.outcome_weight > 0.0 and not args.outcome_head:
        parser.error("--outcome-weight requires --outcome-head")
    if args.update_scope == "all" and args.policy_kl_weight <= 0.0:
        parser.error("--update-scope all requires a positive --policy-kl-weight to anchor policy drift")
    if (args.source_weight > 0.0 or args.target_weight > 0.0) and not args.strategy_spatial_aux:
        parser.error("--source-weight/--target-weight require --strategy-spatial-aux")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    dataset = load_strategy_dataset(
        paths,
        args.max_samples,
        args.max_samples_per_shard,
        args.seed,
        args.finish_head_mode,
        args.action_ce_weight_mode,
        args.label_source,
    )
    dataset = balance_strategy_dataset(dataset, args.balance_strata, args.seed)
    if args.label_source == "search-best" and float(jnp.sum(dataset["outcome_weight"])) <= 0.0:
        raise ValueError("--label-source search-best requires shards with search_best_outcome labels")
    key = jrandom.PRNGKey(args.seed)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )
    finish_outputs = 3 if args.finish_head_mode == "multi-horizon" else 2
    init_finish_outputs = 3 if args.init_finish_head_mode == "multi-horizon" else 2

    print("Adaptive strategy supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Samples:       {dataset['obs'].shape[0]}")
    if args.max_samples_per_shard is not None:
        print(f"Shard cap:     {args.max_samples_per_shard}")
    if args.balance_strata != "none":
        print(f"Balance:       {args.balance_strata}")
    print(f"Network arch:  {args.network_arch}")
    print(f"Warm start:    {args.init_model_path}")
    print(f"Finish head:   {args.finish_head_mode} ({finish_outputs} logits)")
    print(f"Label source:  {args.label_source}")
    print(
        "Loss weights:  "
        f"intent={args.intent_weight:g}, finish={args.finish_weight:g}, "
        f"belief={args.belief_weight:g}, outcome={args.outcome_weight:g}, "
        f"balance_finish={args.balance_finish_labels}, balance_outcome={args.balance_outcome_labels}, "
        f"policy_kl={args.policy_kl_weight:g}, action_ce={args.action_ce_weight:g}, "
        f"action_ce_mode={args.action_ce_weight_mode}, "
        f"q_kl={args.q_kl_weight:g}, q_action_ce={args.q_action_ce_weight:g}, "
        f"search_q_rank={args.search_q_rank_weight:g}, search_q_temp={args.search_q_temperature:g}, "
        f"search_q_value={args.search_q_value_weight:g}, search_q_score_scale={args.search_q_score_scale:g}, "
        f"search_q_outcome_score={args.search_q_outcome_score_weight:g}, "
        f"source={args.source_weight:g}, target={args.target_weight:g}"
    )
    if args.update_scope == "strategy-heads":
        scope_label = "strategy auxiliary heads" + (" + outcome head" if args.outcome_weight > 0.0 else "")
    elif args.update_scope == "strategy-value-heads":
        scope_label = "strategy auxiliary heads + pooled value bottleneck"
        if args.outcome_weight > 0.0:
            scope_label += " + outcome head"
    else:
        scope_label = "all trainable network weights with policy KL anchor"
    print(f"Update scope:  {scope_label}")
    print()

    network = load_or_create_adaptive_network(
        key,
        pad_size=args.pad_to,
        init_model_path=args.init_model_path,
        channels=args.channels,
        init_channels=args.init_channels,
        input_channels=args.input_channels,
        init_input_channels=args.init_input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        init_value_head_sizes=args.init_value_head_sizes if args.init_value_heads == "per-size" else (),
        value_bins=value_bins,
        init_value_bins=init_value_bins,
        outcome_head=args.outcome_head,
        init_outcome_head=args.init_outcome_head,
        strategy_aux=True,
        init_strategy_aux=args.init_strategy_aux,
        strategy_finish_outputs=finish_outputs,
        init_strategy_finish_outputs=init_finish_outputs,
        strategy_spatial_aux=args.strategy_spatial_aux,
        init_strategy_spatial_aux=args.init_strategy_spatial_aux,
        global_context=args.global_context,
        init_global_context=args.global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
    )
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        key, epoch_key = jrandom.split(key)
        network, opt_state, loss, metrics = train_epoch(
            network,
            opt_state,
            dataset,
            optimizer,
            epoch_key,
            args.minibatch_size,
            args.intent_weight,
            args.finish_weight,
            args.belief_weight,
            args.outcome_weight,
            args.policy_kl_weight,
            args.action_ce_weight,
            args.q_kl_weight,
            args.q_action_ce_weight,
            args.search_q_rank_weight,
            args.search_q_temperature,
            args.search_q_value_weight,
            args.search_q_score_scale,
            args.search_q_outcome_score_weight,
            args.source_weight,
            args.target_weight,
            args.balance_finish_labels,
            args.balance_outcome_labels,
            args.update_scope in ("strategy-heads", "strategy-value-heads"),
            args.update_scope == "strategy-value-heads",
            args.finish_head_mode == "multi-horizon",
        )
        jax.block_until_ready(network)
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Intent {float(metrics['intent_loss']):.4f}/{float(metrics['intent_accuracy']) * 100:5.1f}% | "
            f"Finish {float(metrics['finish_loss']):.4f}/{float(metrics['finish_accuracy']) * 100:5.1f}% | "
            f"Belief {float(metrics['belief_loss']):.4f} | "
            f"Outcome {float(metrics['outcome_loss']):.4f}/{float(metrics['outcome_accuracy']) * 100:5.1f}% | "
            f"KL {float(metrics['policy_kl']):.4f} | "
            f"ActCE {float(metrics['action_ce']):.4f}/{float(metrics['teacher_action_accuracy']) * 100:5.1f}% | "
            f"ActW {float(metrics['action_weight_mean']):.3f} | "
            f"QKL {float(metrics['q_policy_kl']):.4f} | "
            f"QCE {float(metrics['q_action_ce']):.4f}/{float(metrics['q_action_accuracy']) * 100:5.1f}% | "
            f"SQ {float(metrics['search_q_rank_loss']):.4f}/{float(metrics['search_q_rank_accuracy']) * 100:5.1f}% | "
            f"SQw {float(metrics['search_q_weight_mean']):.3f} | "
            f"SQV {float(metrics['search_q_value_loss']):.4f}/{float(metrics['search_q_value_accuracy']) * 100:5.1f}% | "
            f"SQVw {float(metrics['search_q_value_weight_mean']):.3f} | "
            f"Src {float(metrics['source_loss']):.4f}/{float(metrics['source_accuracy']) * 100:5.1f}% | "
            f"Tgt {float(metrics['target_loss']):.4f}/{float(metrics['target_accuracy']) * 100:5.1f}% | "
            f"Time {time.time() - t0:.2f}s"
        )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
