"""Train a binary gate for a policy-head adapter delta."""

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

from adaptive_command_gate import CommandGateNetwork
from adaptive_common import ADAPTIVE_MOVE_PLANES, parse_grid_sizes
from adaptive_network import load_or_create_adaptive_network
from evaluate_adaptive_policy import policy_adapter_gate_features
from generals.agents.ppo_policy_agent import parse_policy_channels
from train_adaptive import OUTCOME_WIN

DATASET_FORMATS = ("strategy", "plan-q-prefix", "online-search")
ONLINE_POSITIVE_FIELDS = (
    "search_converts_to_win",
    "search_converts_draw_to_win",
    "search_improves_continuation",
    "adapter_converts_to_win",
    "adapter_converts_draw_to_win",
    "adapter_improves_continuation",
)
ONLINE_ACTION_FIELDS = ("search_action_index", "teacher_action_index", "adapter_action_index")
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


def expand_dataset_paths(patterns: list[str]) -> list[Path]:
    """Expand explicit paths and glob patterns into a stable shard list."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    unique = sorted(dict.fromkeys(paths))
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset shard not found: {missing[0]}")
    return unique


def _compute_adapter_features(
    base_network,
    adapter_network,
    feature_network,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    active: np.ndarray,
    seats: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run both policies and build adapter-gate features in batches."""
    feature_chunks = []
    policy_index_chunks = []
    adapter_index_chunks = []
    pad_size = active.shape[-1]
    for start in range(0, obs.shape[0], batch_size):
        end = min(start + batch_size, obs.shape[0])
        obs_batch = jnp.asarray(obs[start:end])
        legal_batch = jnp.asarray(legal_mask[start:end])
        active_batch = jnp.asarray(active[start:end])
        seat_batch = jnp.asarray(seats[start:end])
        policy_logits = jax.vmap(lambda o, m, a: base_network.logits_value(o, m, a)[0])(
            obs_batch,
            legal_batch,
            active_batch,
        )
        adapter_logits = jax.vmap(lambda o, m, a: adapter_network.logits_value(o, m, a)[0])(
            obs_batch,
            legal_batch,
            active_batch,
        )
        aux_network = feature_network if feature_network is not None else adapter_network
        adapter_aux = jax.vmap(lambda o, m, a: aux_network.strategy_auxiliary(o, m, a))(
            obs_batch,
            legal_batch,
            active_batch,
        )
        if aux_network.outcome_head:
            outcome_logits = jax.vmap(lambda o, m, a: aux_network.logits_value_auxiliary(o, m, a)[3])(
                obs_batch,
                legal_batch,
                active_batch,
            )
        else:
            outcome_logits = jnp.zeros((obs_batch.shape[0], 3), dtype=obs_batch.dtype)
        features = jax.vmap(policy_adapter_gate_features, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))(
            obs_batch,
            policy_logits,
            adapter_logits,
            adapter_aux.finish_logits,
            outcome_logits,
            active_batch,
            seat_batch,
            pad_size,
            len(POLICY_ADAPTER_GATE_FEATURE_NAMES),
        )
        legal = policy_logits > -1.0e8
        policy_indices = jnp.argmax(jnp.where(legal, policy_logits, -1.0e9), axis=-1).astype(jnp.int32)
        adapter_indices = jnp.argmax(jnp.where(legal, adapter_logits, -1.0e9), axis=-1).astype(jnp.int32)
        feature_chunks.append(np.asarray(features))
        policy_index_chunks.append(np.asarray(policy_indices))
        adapter_index_chunks.append(np.asarray(adapter_indices))
    return (
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(policy_index_chunks, axis=0),
        np.concatenate(adapter_index_chunks, axis=0),
    )


def _balanced_binary_weights(labels: np.ndarray) -> np.ndarray:
    """Give positives and negatives equal total weight."""
    positive = labels > 0.5
    pos_count = max(int(np.sum(positive)), 1)
    neg_count = max(int(np.sum(~positive)), 1)
    return (np.where(positive, 0.5 / pos_count, 0.5 / neg_count).astype(np.float32) * labels.shape[0])


def build_gate_examples(
    paths: list[Path],
    base_network,
    adapter_network,
    feature_network,
    feature_batch_size: int,
    positive_path_contains: tuple[str, ...],
    require_search_best_win: bool,
    include_finish250_positives: bool,
    include_finish500_positives: bool,
    keep_unchanged_negatives: bool,
    max_examples: int | None,
    seed: int,
) -> dict[str, object]:
    """Construct changed-action adapter-gate examples from strategy shards."""
    rng = np.random.default_rng(seed)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    stats = {
        "rows": 0,
        "changed": 0,
        "positive": 0,
        "teacher_match": 0,
        "decisive": 0,
        "positive_domain_rows": 0,
    }
    for path in paths:
        shard = np.load(path)
        obs = shard["obs"].astype(np.float32)
        legal_mask = shard["legal_mask"].astype(np.bool_)
        active = shard["active"].astype(np.bool_)
        seats = shard["seat"].astype(np.float32)
        features, policy_indices, adapter_indices = _compute_adapter_features(
            base_network,
            adapter_network,
            feature_network,
            obs,
            legal_mask,
            active,
            seats,
            feature_batch_size,
        )
        pass_index = ADAPTIVE_MOVE_PLANES * active.shape[-1] * active.shape[-1]
        teacher_actions = shard["teacher_action_index"].astype(np.int32)
        changed = (adapter_indices != policy_indices) & (adapter_indices != pass_index)
        teacher_match = adapter_indices == teacher_actions
        positive_domain = np.ones((obs.shape[0],), dtype=np.bool_)
        if positive_path_contains:
            path_text = str(path)
            positive_domain = np.full(
                (obs.shape[0],),
                any(part in path_text for part in positive_path_contains),
                dtype=np.bool_,
            )
        if require_search_best_win and "search_best_outcome" in shard:
            decisive = shard["search_best_outcome"].astype(np.int32) == OUTCOME_WIN
            if include_finish250_positives and "finish_within_250" in shard:
                decisive |= shard["finish_within_250"].astype(np.float32) > 0.5
            if include_finish500_positives and "finish_within_500" in shard:
                decisive |= shard["finish_within_500"].astype(np.float32) > 0.5
        else:
            decisive = np.ones((obs.shape[0],), dtype=np.bool_)
        labels = (changed & teacher_match & decisive & positive_domain).astype(np.float32)
        keep = np.ones_like(changed, dtype=np.bool_) if keep_unchanged_negatives else changed
        if not np.any(keep):
            continue
        kept_features = features[keep].astype(np.float32)
        kept_labels = labels[keep].astype(np.float32)
        weights = _balanced_binary_weights(kept_labels)
        feature_chunks.append(kept_features)
        label_chunks.append(kept_labels)
        weight_chunks.append(weights)
        stats["rows"] += int(obs.shape[0])
        stats["changed"] += int(np.sum(changed))
        stats["positive"] += int(np.sum(labels[keep] > 0.5))
        stats["teacher_match"] += int(np.sum(changed & teacher_match))
        stats["decisive"] += int(np.sum(decisive))
        stats["positive_domain_rows"] += int(np.sum(positive_domain))

    if not feature_chunks:
        raise ValueError("No policy-adapter gate examples selected")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    weights = np.concatenate(weight_chunks, axis=0)
    if max_examples is not None and features.shape[0] > max_examples:
        indices = np.sort(rng.choice(features.shape[0], size=max_examples, replace=False))
        features = features[indices]
        labels = labels[indices]
        weights = weights[indices]
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(features.std(axis=0).astype(np.float32), 1.0e-6)
    return {
        "features": jnp.asarray(features),
        "labels": jnp.asarray(labels),
        "weights": jnp.asarray(weights),
        "feature_mean": jnp.asarray(feature_mean),
        "feature_std": jnp.asarray(feature_std),
        "stats": stats,
    }


def build_prefix_gate_examples(
    paths: list[Path],
    base_network,
    adapter_network,
    feature_network,
    feature_batch_size: int,
    keep_unchanged_negatives: bool,
    max_examples: int | None,
    seed: int,
    min_plan_advantage: float,
    require_plan_win: bool,
    require_base_not_win: bool,
    max_plan_time_to_terminal: int | None,
    max_prefix_step: int | None,
) -> dict[str, object]:
    """Construct gate examples from executed accepted-prefix Plan-Q shards."""
    rng = np.random.default_rng(seed)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    stats = {
        "rows": 0,
        "prefix_rows": 0,
        "valid_prefix": 0,
        "accepted_prefix": 0,
        "label_domain": 0,
        "changed": 0,
        "positive": 0,
        "prefix_action_match": 0,
    }
    for path in paths:
        shard = np.load(path)
        if "worker_prefix_obs" not in shard:
            raise KeyError(f"{path} does not contain worker_prefix_obs; use --dataset-format strategy for strategy shards")
        prefix_obs = shard["worker_prefix_obs"].astype(np.float32)
        num_rows, prefix_len = prefix_obs.shape[:2]
        obs = prefix_obs.reshape((-1,) + prefix_obs.shape[2:])
        legal_mask = shard["worker_prefix_legal_mask"].astype(np.bool_).reshape((-1,) + shard["worker_prefix_legal_mask"].shape[2:])
        active = shard["worker_prefix_active"].astype(np.bool_).reshape((-1,) + shard["worker_prefix_active"].shape[2:])
        seats = np.broadcast_to(shard["seat"][:, None], (num_rows, prefix_len)).reshape(-1).astype(np.float32)
        prefix_actions = shard["worker_prefix_action_index"].astype(np.int32).reshape(-1)
        valid = shard["worker_prefix_valid"].astype(np.bool_).reshape(-1)
        accepted = shard["accepted_prefix_mask"].astype(np.bool_).reshape(-1) if "accepted_prefix_mask" in shard else valid
        plan_outcomes = shard["worker_prefix_plan_outcome"].astype(np.int32).reshape(-1)
        plan_advantages = shard["worker_prefix_plan_advantage"].astype(np.float32).reshape(-1)
        base_outcomes = (
            shard["worker_prefix_base_outcome"].astype(np.int32).reshape(-1)
            if "worker_prefix_base_outcome" in shard
            else np.full((prefix_actions.shape[0],), -1, dtype=np.int32)
        )
        plan_times = (
            shard["worker_prefix_plan_time_to_terminal"].astype(np.int32).reshape(-1)
            if "worker_prefix_plan_time_to_terminal" in shard
            else np.full((prefix_actions.shape[0],), 1_000_000, dtype=np.int32)
        )
        prefix_steps = (
            shard["worker_prefix_step_index"].astype(np.int32).reshape(-1)
            if "worker_prefix_step_index" in shard
            else np.tile(np.arange(prefix_len, dtype=np.int32), num_rows)
        )
        pass_index = ADAPTIVE_MOVE_PLANES * active.shape[-1] * active.shape[-1]
        non_pass = prefix_actions != pass_index
        label_domain = accepted & (plan_advantages >= min_plan_advantage)
        if require_plan_win:
            label_domain &= plan_outcomes == OUTCOME_WIN
        if require_base_not_win:
            label_domain &= base_outcomes != OUTCOME_WIN
        if max_plan_time_to_terminal is not None:
            label_domain &= plan_times <= max_plan_time_to_terminal
        if max_prefix_step is not None:
            label_domain &= prefix_steps <= max_prefix_step
        feature_keep = valid & non_pass
        if not np.any(feature_keep):
            continue
        features, policy_indices, adapter_indices = _compute_adapter_features(
            base_network,
            adapter_network,
            feature_network,
            obs[feature_keep],
            legal_mask[feature_keep],
            active[feature_keep],
            seats[feature_keep],
            feature_batch_size,
        )
        kept_prefix_actions = prefix_actions[feature_keep]
        kept_label_domain = label_domain[feature_keep]
        changed = (adapter_indices != policy_indices) & (adapter_indices != pass_index)
        prefix_match = adapter_indices == kept_prefix_actions
        labels = (changed & prefix_match & kept_label_domain).astype(np.float32)
        keep = np.ones_like(changed, dtype=np.bool_) if keep_unchanged_negatives else changed
        if not np.any(keep):
            continue
        kept_features = features[keep].astype(np.float32)
        kept_labels = labels[keep].astype(np.float32)
        weights = _balanced_binary_weights(kept_labels)
        feature_chunks.append(kept_features)
        label_chunks.append(kept_labels)
        weight_chunks.append(weights)
        stats["rows"] += int(num_rows)
        stats["prefix_rows"] += int(prefix_actions.shape[0])
        stats["valid_prefix"] += int(np.sum(valid & non_pass))
        stats["accepted_prefix"] += int(np.sum(accepted & valid & non_pass))
        stats["label_domain"] += int(np.sum(label_domain & valid & non_pass))
        stats["changed"] += int(np.sum(changed))
        stats["positive"] += int(np.sum(labels[keep] > 0.5))
        stats["prefix_action_match"] += int(np.sum(changed & prefix_match))

    if not feature_chunks:
        raise ValueError("No plan-q-prefix policy-adapter gate examples selected")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    weights = np.concatenate(weight_chunks, axis=0)
    if max_examples is not None and features.shape[0] > max_examples:
        indices = np.sort(rng.choice(features.shape[0], size=max_examples, replace=False))
        features = features[indices]
        labels = labels[indices]
        weights = weights[indices]
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(features.std(axis=0).astype(np.float32), 1.0e-6)
    return {
        "features": jnp.asarray(features),
        "labels": jnp.asarray(labels),
        "weights": jnp.asarray(weights),
        "feature_mean": jnp.asarray(feature_mean),
        "feature_std": jnp.asarray(feature_std),
        "stats": stats,
    }


def build_online_search_gate_examples(
    paths: list[Path],
    base_network,
    adapter_network,
    feature_network,
    feature_batch_size: int,
    keep_unchanged_negatives: bool,
    max_examples: int | None,
    seed: int,
    positive_field: str,
    action_field: str,
    min_score_delta: float,
    require_search_used: bool,
    require_adapter_match: bool,
) -> dict[str, object]:
    """Construct gate examples from online-search continuation/conversion shards."""
    rng = np.random.default_rng(seed)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    stats = {
        "rows": 0,
        "changed": 0,
        "positive": 0,
        "positive_signal": 0,
        "adapter_search_match": 0,
        "search_used": 0,
        "score_delta_pass": 0,
    }
    for path in paths:
        shard = np.load(path)
        if positive_field not in shard:
            raise KeyError(f"{path} does not contain {positive_field}; collect online-search conversion labels first")
        if action_field not in shard:
            raise KeyError(f"{path} does not contain {action_field}")
        obs = shard["obs"].astype(np.float32)
        legal_mask = shard["legal_mask"].astype(np.bool_)
        active = shard["active"].astype(np.bool_)
        seats = shard["seat"].astype(np.float32)
        positive_actions = shard[action_field].astype(np.int32)
        positive_signal = shard[positive_field].astype(np.bool_)
        if min_score_delta > 0.0:
            if "search_continuation_score_delta" not in shard:
                raise KeyError(f"{path} does not contain search_continuation_score_delta")
            score_delta_pass = shard["search_continuation_score_delta"].astype(np.float32) >= min_score_delta
            positive_signal &= score_delta_pass
        else:
            score_delta_pass = np.ones((obs.shape[0],), dtype=np.bool_)
        if require_search_used and "search_used" in shard:
            search_used = shard["search_used"].astype(np.bool_)
            positive_signal &= search_used
        else:
            search_used = np.ones((obs.shape[0],), dtype=np.bool_)

        features, policy_indices, adapter_indices = _compute_adapter_features(
            base_network,
            adapter_network,
            feature_network,
            obs,
            legal_mask,
            active,
            seats,
            feature_batch_size,
        )
        pass_index = ADAPTIVE_MOVE_PLANES * active.shape[-1] * active.shape[-1]
        changed = (adapter_indices != policy_indices) & (adapter_indices != pass_index)
        adapter_action_match = adapter_indices == positive_actions
        match_domain = adapter_action_match if require_adapter_match else np.ones_like(adapter_action_match)
        labels = (changed & positive_signal & match_domain).astype(np.float32)
        keep = np.ones_like(changed, dtype=np.bool_) if keep_unchanged_negatives else changed
        if not np.any(keep):
            continue
        kept_features = features[keep].astype(np.float32)
        kept_labels = labels[keep].astype(np.float32)
        weights = _balanced_binary_weights(kept_labels)
        feature_chunks.append(kept_features)
        label_chunks.append(kept_labels)
        weight_chunks.append(weights)
        stats["rows"] += int(obs.shape[0])
        stats["changed"] += int(np.sum(changed))
        stats["positive"] += int(np.sum(labels[keep] > 0.5))
        stats["positive_signal"] += int(np.sum(positive_signal))
        stats["adapter_search_match"] += int(np.sum(changed & adapter_action_match))
        stats["search_used"] += int(np.sum(search_used))
        stats["score_delta_pass"] += int(np.sum(score_delta_pass))

    if not feature_chunks:
        raise ValueError("No online-search policy-adapter gate examples selected")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    weights = np.concatenate(weight_chunks, axis=0)
    if max_examples is not None and features.shape[0] > max_examples:
        indices = np.sort(rng.choice(features.shape[0], size=max_examples, replace=False))
        features = features[indices]
        labels = labels[indices]
        weights = weights[indices]
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(features.std(axis=0).astype(np.float32), 1.0e-6)
    return {
        "features": jnp.asarray(features),
        "labels": jnp.asarray(labels),
        "weights": jnp.asarray(weights),
        "feature_mean": jnp.asarray(feature_mean),
        "feature_std": jnp.asarray(feature_std),
        "stats": stats,
    }


@eqx.filter_jit
def train_step(gate, opt_state, batch, optimizer):
    features, labels, weights = batch

    def loss_fn(model):
        logits = jax.vmap(model)(features)
        losses = optax.sigmoid_binary_cross_entropy(logits, labels)
        normalizer = jnp.maximum(jnp.sum(weights), 1.0)
        loss = jnp.sum(losses * weights) / normalizer
        probs = jax.nn.sigmoid(logits)
        preds = probs >= 0.5
        accuracy = jnp.sum((preds == (labels >= 0.5)).astype(jnp.float32) * weights) / normalizer
        positive_prob = jnp.sum(probs * labels * weights) / jnp.maximum(jnp.sum(labels * weights), 1.0)
        negative_prob = jnp.sum(probs * (1.0 - labels) * weights) / jnp.maximum(
            jnp.sum((1.0 - labels) * weights),
            1.0,
        )
        return loss, {
            "accuracy": accuracy,
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "mean_prob": jnp.mean(probs),
        }

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(gate)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(gate, eqx.is_inexact_array))
    return eqx.apply_updates(gate, updates), opt_state, loss, metrics


def train_epoch(gate, opt_state, dataset, optimizer, key, minibatch_size: int):
    num_examples = dataset["features"].shape[0]
    permutation = jrandom.permutation(key, num_examples)
    num_batches = max(num_examples // minibatch_size, 1)
    metrics_sum = None
    loss_sum = 0.0
    for batch_index in range(num_batches):
        start = batch_index * minibatch_size
        end = min(start + minibatch_size, num_examples)
        idx = permutation[start:end]
        batch = (dataset["features"][idx], dataset["labels"][idx], dataset["weights"][idx])
        gate, opt_state, loss, metrics = train_step(gate, opt_state, batch, optimizer)
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return gate, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a learned gate for policy-adapter deltas.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--dataset-format", choices=DATASET_FORMATS, default="strategy")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-model-path", required=True)
    parser.add_argument(
        "--feature-model-path",
        default=None,
        help="Optional strategy-aux model used only for gate features; adapter-model still supplies policy deltas.",
    )
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="unet")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--input-channels", type=int, default=35)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--value-head-sizes", default="8,12,16")
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument(
        "--base-outcome-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Outcome-head schema for the base model. Defaults to --outcome-head.",
    )
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument(
        "--base-strategy-aux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Strategy-head schema for the base model. Defaults to --strategy-aux.",
    )
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument(
        "--base-strategy-spatial-aux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Spatial strategy-head schema for the base model. Defaults to --strategy-spatial-aux.",
    )
    parser.add_argument("--strategy-finish-outputs", type=int, default=3)
    parser.add_argument(
        "--base-strategy-finish-outputs",
        type=int,
        default=None,
        help="Strategy finish output count for the base model. Defaults to --strategy-finish-outputs.",
    )
    parser.add_argument("--positive-path-contains", action="append", default=[])
    parser.add_argument(
        "--drop-mismatched-init-leaves",
        action="store_true",
        help="Load matching checkpoint leaves and reinitialize shape-mismatched legacy leaves.",
    )
    parser.add_argument("--allow-nondecisive-positives", action="store_true")
    parser.add_argument("--include-finish250-positives", action="store_true")
    parser.add_argument("--include-finish500-positives", action="store_true")
    parser.add_argument(
        "--min-prefix-plan-advantage",
        type=float,
        default=0.25,
        help="For --dataset-format plan-q-prefix, require this plan advantage for positive labels.",
    )
    parser.add_argument(
        "--allow-prefix-nonwin",
        action="store_true",
        help="For --dataset-format plan-q-prefix, allow non-winning plan outcomes to form positive labels.",
    )
    parser.add_argument(
        "--require-prefix-base-not-win",
        action="store_true",
        help="For --dataset-format plan-q-prefix, positives require base continuation not already winning.",
    )
    parser.add_argument(
        "--max-prefix-plan-time-to-terminal",
        type=int,
        default=None,
        help="For --dataset-format plan-q-prefix, positives require plan terminal time at or below this value.",
    )
    parser.add_argument(
        "--max-prefix-step",
        type=int,
        default=None,
        help="For --dataset-format plan-q-prefix, positives use only early executed-prefix steps up to this index.",
    )
    parser.add_argument(
        "--online-positive-field",
        choices=ONLINE_POSITIVE_FIELDS,
        default="search_converts_to_win",
        help="For --dataset-format online-search, shard boolean field that marks positive conversion/improvement rows.",
    )
    parser.add_argument(
        "--online-action-field",
        choices=ONLINE_ACTION_FIELDS,
        default="search_action_index",
        help="For --dataset-format online-search, action-index field the adapter top action should match for positives.",
    )
    parser.add_argument(
        "--min-online-score-delta",
        type=float,
        default=0.0,
        help="For --dataset-format online-search, require this continuation-score improvement for positives.",
    )
    parser.add_argument(
        "--require-online-search-used",
        action="store_true",
        help="For --dataset-format online-search, positives require rows where the online search action was used.",
    )
    parser.add_argument(
        "--allow-online-adapter-mismatch",
        action="store_true",
        help="For --dataset-format online-search, allow positives even if the adapter top action differs from search action.",
    )
    parser.add_argument(
        "--keep-unchanged-negatives",
        action="store_true",
        help="Keep rows where the adapter and base greedy actions match as negative examples.",
    )
    parser.add_argument("--model-path", default="runs/adaptive-policy-adapter-gate/generals-adaptive-policy-adapter-gate.eqx")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.channels = parse_policy_channels(args.channels)
        args.value_head_sizes = parse_grid_sizes(args.value_head_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.value_loss == "hl-gauss" and args.value_bins <= 1:
        parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
    if args.strategy_finish_outputs <= 0:
        parser.error("--strategy-finish-outputs must be positive")
    if args.base_strategy_finish_outputs is not None and args.base_strategy_finish_outputs <= 0:
        parser.error("--base-strategy-finish-outputs must be positive")
    if not args.strategy_aux:
        parser.error("policy-adapter gate features require --strategy-aux")
    if (args.base_strategy_spatial_aux if args.base_strategy_spatial_aux is not None else args.strategy_spatial_aux) and not (
        args.base_strategy_aux if args.base_strategy_aux is not None else args.strategy_aux
    ):
        parser.error("--base-strategy-spatial-aux requires base strategy aux")
    if args.hidden_dim <= 0 or args.num_epochs <= 0 or args.minibatch_size <= 0 or args.feature_batch_size <= 0:
        parser.error("hidden dim, epochs, minibatch, and feature batch must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if args.min_prefix_plan_advantage < 0.0:
        parser.error("--min-prefix-plan-advantage must be non-negative")
    if args.max_prefix_plan_time_to_terminal is not None and args.max_prefix_plan_time_to_terminal <= 0:
        parser.error("--max-prefix-plan-time-to-terminal must be positive")
    if args.max_prefix_step is not None and args.max_prefix_step < 0:
        parser.error("--max-prefix-step must be non-negative")
    if args.min_online_score_delta < 0.0:
        parser.error("--min-online-score-delta must be non-negative")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    key = jrandom.PRNGKey(args.seed)
    key, base_key, adapter_key, gate_key = jrandom.split(key, 4)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    base_outcome_head = args.outcome_head if args.base_outcome_head is None else args.base_outcome_head
    base_strategy_aux = args.strategy_aux if args.base_strategy_aux is None else args.base_strategy_aux
    base_strategy_spatial_aux = (
        args.strategy_spatial_aux if args.base_strategy_spatial_aux is None else args.base_strategy_spatial_aux
    )
    base_strategy_finish_outputs = (
        args.strategy_finish_outputs
        if args.base_strategy_finish_outputs is None
        else args.base_strategy_finish_outputs
    )
    base_network = load_or_create_adaptive_network(
        base_key,
        pad_size=16,
        init_model_path=args.base_model_path,
        channels=args.channels,
        input_channels=args.input_channels,
        init_input_channels=args.input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        init_value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        value_bins=value_bins,
        init_value_bins=value_bins,
        outcome_head=base_outcome_head,
        init_outcome_head=base_outcome_head,
        strategy_aux=base_strategy_aux,
        init_strategy_aux=base_strategy_aux,
        strategy_spatial_aux=base_strategy_spatial_aux,
        init_strategy_spatial_aux=base_strategy_spatial_aux,
        strategy_finish_outputs=base_strategy_finish_outputs,
        init_strategy_finish_outputs=base_strategy_finish_outputs,
        global_context=args.global_context,
        init_global_context=args.global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
        drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
    )
    adapter_network = load_or_create_adaptive_network(
        adapter_key,
        pad_size=16,
        init_model_path=args.adapter_model_path,
        channels=args.channels,
        input_channels=args.input_channels,
        init_input_channels=args.input_channels,
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
        strategy_finish_outputs=args.strategy_finish_outputs,
        init_strategy_finish_outputs=args.strategy_finish_outputs,
        global_context=args.global_context,
        init_global_context=args.global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
        drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
    )
    feature_network = None
    if args.feature_model_path is not None:
        feature_network = load_or_create_adaptive_network(
            adapter_key,
            pad_size=16,
            init_model_path=args.feature_model_path,
            channels=args.channels,
            input_channels=args.input_channels,
            init_input_channels=args.input_channels,
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
            strategy_finish_outputs=args.strategy_finish_outputs,
            init_strategy_finish_outputs=args.strategy_finish_outputs,
            global_context=args.global_context,
            init_global_context=args.global_context,
            network_arch=args.network_arch,
            init_network_arch=args.network_arch,
            drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
        )
    if args.dataset_format == "strategy":
        dataset = build_gate_examples(
            paths,
            base_network,
            adapter_network,
            feature_network,
            args.feature_batch_size,
            tuple(args.positive_path_contains),
            not args.allow_nondecisive_positives,
            args.include_finish250_positives,
            args.include_finish500_positives,
            args.keep_unchanged_negatives,
            args.max_examples,
            args.seed,
        )
    elif args.dataset_format == "plan-q-prefix":
        dataset = build_prefix_gate_examples(
            paths,
            base_network,
            adapter_network,
            feature_network,
            args.feature_batch_size,
            args.keep_unchanged_negatives,
            args.max_examples,
            args.seed,
            args.min_prefix_plan_advantage,
            not args.allow_prefix_nonwin,
            args.require_prefix_base_not_win,
            args.max_prefix_plan_time_to_terminal,
            args.max_prefix_step,
        )
    else:
        dataset = build_online_search_gate_examples(
            paths,
            base_network,
            adapter_network,
            feature_network,
            args.feature_batch_size,
            args.keep_unchanged_negatives,
            args.max_examples,
            args.seed,
            args.online_positive_field,
            args.online_action_field,
            args.min_online_score_delta,
            args.require_online_search_used,
            not args.allow_online_adapter_mismatch,
        )
    labels = np.asarray(dataset["labels"])
    stats = dataset["stats"]
    print("Adaptive policy-adapter gate supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Data format:   {args.dataset_format}")
    print(f"Examples:      {dataset['features'].shape[0]}")
    print(f"Positive:      {float(np.mean(labels)) * 100:.2f}%")
    print(f"Rows/changed:  {stats['rows']} / {stats['changed']}")
    if args.dataset_format == "strategy":
        print(f"Teacher match: {stats['teacher_match']}")
        print(f"Decisive:      {stats['decisive']}")
    elif args.dataset_format == "plan-q-prefix":
        print(f"Prefix rows:   {stats['valid_prefix']} valid / {stats['accepted_prefix']} accepted")
        print(f"Label domain:  {stats['label_domain']}")
        print(f"Prefix match:  {stats['prefix_action_match']}")
    else:
        print(f"Positive sig:  {stats['positive_signal']}")
        print(f"Adapter match: {stats['adapter_search_match']}")
        print(f"Search used:   {stats['search_used']}")
    print(f"Base:          {args.base_model_path}")
    print(f"Adapter:       {args.adapter_model_path}")
    print(
        "Base schema:   "
        f"outcome={base_outcome_head}, strategy_aux={base_strategy_aux}, "
        f"spatial={base_strategy_spatial_aux}, finish_outputs={base_strategy_finish_outputs}"
    )
    if args.feature_model_path is not None:
        print(f"Feature model: {args.feature_model_path}")
    print(f"Output:        {args.model_path}")
    print(f"Features:      {', '.join(POLICY_ADAPTER_GATE_FEATURE_NAMES)}")
    print()

    gate = CommandGateNetwork(
        gate_key,
        input_dim=len(POLICY_ADAPTER_GATE_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        feature_mean=dataset["feature_mean"],
        feature_std=dataset["feature_std"],
    )
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(gate, eqx.is_inexact_array))
    epoch_history = []
    for epoch in range(1, args.num_epochs + 1):
        key, epoch_key = jrandom.split(key)
        t0 = time.time()
        gate, opt_state, loss, metrics = train_epoch(gate, opt_state, dataset, optimizer, epoch_key, args.minibatch_size)
        elapsed = time.time() - t0
        epoch_history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "accuracy": float(metrics["accuracy"]),
                "positive_prob": float(metrics["positive_prob"]),
                "negative_prob": float(metrics["negative_prob"]),
                "mean_prob": float(metrics["mean_prob"]),
                "seconds": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Acc {float(metrics['accuracy']) * 100:5.1f}% | "
            f"P+ {float(metrics['positive_prob']):.3f} | "
            f"P- {float(metrics['negative_prob']):.3f} | "
            f"Pmean {float(metrics['mean_prob']):.3f} | "
            f"Time {elapsed:.2f}s"
        )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(model_path, gate)
    sidecar = {
        "feature_names": list(POLICY_ADAPTER_GATE_FEATURE_NAMES),
        "feature_mean": np.asarray(gate.feature_mean).tolist(),
        "feature_std": np.asarray(gate.feature_std).tolist(),
        "hidden_dim": args.hidden_dim,
        "dataset_format": args.dataset_format,
        "examples": int(dataset["features"].shape[0]),
        "positive_fraction": float(np.mean(labels)),
        "stats": stats,
        "positive_path_contains": list(args.positive_path_contains),
        "require_search_best_win": not args.allow_nondecisive_positives,
        "include_finish250_positives": args.include_finish250_positives,
        "include_finish500_positives": args.include_finish500_positives,
        "keep_unchanged_negatives": args.keep_unchanged_negatives,
        "min_prefix_plan_advantage": args.min_prefix_plan_advantage,
        "allow_prefix_nonwin": args.allow_prefix_nonwin,
        "require_prefix_base_not_win": args.require_prefix_base_not_win,
        "max_prefix_plan_time_to_terminal": args.max_prefix_plan_time_to_terminal,
        "max_prefix_step": args.max_prefix_step,
        "online_positive_field": args.online_positive_field,
        "online_action_field": args.online_action_field,
        "min_online_score_delta": args.min_online_score_delta,
        "require_online_search_used": args.require_online_search_used,
        "allow_online_adapter_mismatch": args.allow_online_adapter_mismatch,
        "base_model_path": args.base_model_path,
        "base_outcome_head": base_outcome_head,
        "base_strategy_aux": base_strategy_aux,
        "base_strategy_spatial_aux": base_strategy_spatial_aux,
        "base_strategy_finish_outputs": base_strategy_finish_outputs,
        "adapter_model_path": args.adapter_model_path,
        "feature_model_path": args.feature_model_path,
        "datasets": [str(path) for path in paths],
        "epoch_history": epoch_history,
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
