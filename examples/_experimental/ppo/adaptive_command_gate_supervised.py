"""Train a binary command-acceptance gate from Plan-Q shards."""

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

from adaptive_command_gate import COMMAND_GATE_FEATURE_DIM, COMMAND_GATE_FEATURE_NAMES, CommandGateNetwork
from adaptive_common import ADAPTIVE_MOVE_PLANES, parse_grid_sizes
from adaptive_network import load_or_create_adaptive_network
from evaluate_adaptive_policy import (
    PLAN_WORKER_COMMAND_SOURCE_NAMES,
    PLAN_WORKER_COMMAND_SOURCE_TO_ID,
    command_gate_features,
    strategy_plan_worker_obs,
)
from generals.agents.ppo_policy_agent import parse_policy_channels
from train_adaptive import OUTCOME_WIN


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


def _gather_cells(map_values: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    flat = map_values.reshape(map_values.shape[0], -1)
    return flat[np.arange(flat.shape[0])[:, None], flat_indices]


def _compute_network_outputs(network, obs: np.ndarray, legal_mask: np.ndarray, active: np.ndarray, batch_size: int):
    """Run the feature network on shard observations in small batches."""
    logits_chunks = []
    q_chunks = []
    source_chunks = []
    target_chunks = []
    finish_chunks = []
    for start in range(0, obs.shape[0], batch_size):
        end = min(start + batch_size, obs.shape[0])
        obs_batch = jnp.asarray(obs[start:end])
        legal_batch = jnp.asarray(legal_mask[start:end])
        active_batch = jnp.asarray(active[start:end])
        logits = jax.vmap(lambda o, m, a: network.logits_value(o, m, a)[0])(obs_batch, legal_batch, active_batch)
        aux = jax.vmap(lambda o, m, a: network.strategy_auxiliary(o, m, a))(obs_batch, legal_batch, active_batch)
        logits_chunks.append(np.asarray(logits))
        q_chunks.append(np.asarray(aux.action_q_values))
        source_chunks.append(np.asarray(aux.source_logits))
        target_chunks.append(np.asarray(aux.target_logits))
        if aux.finish_logits.shape[-1] == 1:
            finish = jax.nn.sigmoid(aux.finish_logits[:, 0])
        elif aux.finish_logits.shape[-1] == 2:
            finish = jax.nn.softmax(aux.finish_logits, axis=-1)[:, 1]
        else:
            finish = jax.nn.sigmoid(aux.finish_logits[:, -1])
        finish_chunks.append(np.asarray(finish))
    return (
        np.concatenate(logits_chunks, axis=0),
        np.concatenate(q_chunks, axis=0),
        np.concatenate(source_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(finish_chunks, axis=0),
    )


def _compute_strategy_worker_gate_outputs(
    network,
    plan_worker_network,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    active: np.ndarray,
    seats: np.ndarray,
    batch_size: int,
    command_source: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build gate features for the learned Worker top action on strategy shards."""
    feature_chunks = []
    current_chunks = []
    worker_chunks = []
    target_chunks = []
    pad_size = active.shape[-1]
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    for start in range(0, obs.shape[0], batch_size):
        end = min(start + batch_size, obs.shape[0])
        obs_batch = jnp.asarray(obs[start:end])
        legal_batch = jnp.asarray(legal_mask[start:end])
        active_batch = jnp.asarray(active[start:end])
        seat_batch = jnp.asarray(seats[start:end])

        logits = jax.vmap(lambda o, m, a: network.logits_value(o, m, a)[0])(obs_batch, legal_batch, active_batch)
        aux = jax.vmap(lambda o, m, a: network.strategy_auxiliary(o, m, a))(obs_batch, legal_batch, active_batch)
        if command_source == PLAN_WORKER_COMMAND_SOURCE_TO_ID["belief-main-stack"]:
            source_logits = jnp.zeros_like(aux.enemy_general_logits)
            target_logits = aux.enemy_general_logits
        else:
            source_logits = aux.source_logits
            target_logits = aux.target_logits
        worker_obs = jax.vmap(strategy_plan_worker_obs, in_axes=(0, 0, 0, 0, 0, None))(
            obs_batch,
            legal_batch,
            active_batch,
            source_logits,
            target_logits,
            pad_size,
        )
        worker_logits = jax.vmap(lambda o, m, a: plan_worker_network.logits_value(o, m, a)[0])(
            worker_obs,
            legal_batch,
            active_batch,
        )
        legal_worker_logits = jnp.where(logits > -1.0e8, worker_logits, -1.0e9)
        current_indices = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        worker_indices = jnp.argmax(legal_worker_logits, axis=-1).astype(jnp.int32)
        source_indices = (jnp.minimum(worker_indices, pass_index - 1) % (pad_size * pad_size)).astype(jnp.int32)
        target_indices = jnp.argmax(jnp.where(active_batch, target_logits, -1.0e9).reshape(target_logits.shape[0], -1), axis=-1)
        target_indices = target_indices.astype(jnp.int32)
        features = jax.vmap(command_gate_features, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))(
            obs_batch,
            logits,
            aux.action_q_values,
            aux.finish_logits,
            source_logits,
            target_logits,
            source_indices,
            target_indices,
            worker_indices,
            current_indices,
            seat_batch,
            pad_size,
        )
        feature_chunks.append(np.asarray(features))
        current_chunks.append(np.asarray(current_indices))
        worker_chunks.append(np.asarray(worker_indices))
        target_chunks.append(np.asarray(target_indices))
    return (
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(current_chunks, axis=0),
        np.concatenate(worker_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
    )


def build_gate_examples(
    paths: list[Path],
    network,
    feature_batch_size: int,
    score_margin: float,
    include_noncomparable_negatives: bool,
    max_examples: int | None,
    seed: int,
) -> dict[str, jnp.ndarray]:
    """Construct command-gate features and labels from Plan-Q candidate plans."""
    rng = np.random.default_rng(seed)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for path in paths:
        shard = np.load(path)
        obs = shard["obs"].astype(np.float32)
        legal_mask = shard["legal_mask"].astype(np.bool_)
        active = shard["active"].astype(np.bool_)
        logits, action_q, source_logits, target_logits, finish_prob = _compute_network_outputs(
            network,
            obs,
            legal_mask,
            active,
            feature_batch_size,
        )
        plan_actions = shard["plan_action_indices"].astype(np.int32)
        plan_scores = shard["plan_scores"].astype(np.float32)
        plan_outcomes = shard["plan_outcomes"].astype(np.float32)
        source_indices = shard["source_indices"].astype(np.int32)
        target_indices = shard["target_indices"].astype(np.int32)
        teacher_actions = shard["teacher_action_index"].astype(np.int32)
        seats = shard["seat"].astype(np.float32)
        source_count = source_indices.shape[1]
        target_count = target_indices.shape[1]
        num_rows = plan_actions.shape[0]
        flat_actions = plan_actions.reshape(num_rows, -1)
        flat_scores = plan_scores.reshape(num_rows, -1)
        flat_outcomes = plan_outcomes.reshape(num_rows, -1)
        source_grid = np.broadcast_to(source_indices[:, :, None], (num_rows, source_count, target_count)).reshape(
            num_rows,
            -1,
        )
        target_grid = np.broadcast_to(target_indices[:, None, :], (num_rows, source_count, target_count)).reshape(
            num_rows,
            -1,
        )
        teacher_matches = flat_actions == teacher_actions[:, None]
        has_teacher = np.any(teacher_matches, axis=1)
        teacher_pos = np.argmax(teacher_matches.astype(np.int32), axis=1)
        row_ids = np.arange(num_rows)
        teacher_scores = flat_scores[row_ids, teacher_pos]
        teacher_outcomes = flat_outcomes[row_ids, teacher_pos]
        switched = flat_actions != teacher_actions[:, None]
        outcome_improved = flat_outcomes > teacher_outcomes[:, None]
        score_improved = (flat_outcomes == teacher_outcomes[:, None]) & (
            (flat_scores - teacher_scores[:, None]) >= score_margin
        )
        labels = (has_teacher[:, None] & switched & (outcome_improved | score_improved)).astype(np.float32)
        comparable = has_teacher[:, None] & switched
        if include_noncomparable_negatives:
            comparable = comparable | switched
        pass_index = ADAPTIVE_MOVE_PLANES * active.shape[-1] * active.shape[-1]
        keep = comparable & (flat_actions != pass_index)
        candidate_rows, candidate_pos = np.nonzero(keep)
        if candidate_rows.shape[0] == 0:
            continue
        candidate_actions = flat_actions[candidate_rows, candidate_pos]
        current_actions = teacher_actions[candidate_rows]
        candidate_sources = source_grid[candidate_rows, candidate_pos]
        candidate_targets = target_grid[candidate_rows, candidate_pos]
        source_rows = candidate_sources // active.shape[-1]
        source_cols = candidate_sources % active.shape[-1]
        target_rows = candidate_targets // active.shape[-1]
        target_cols = candidate_targets % active.shape[-1]
        route_distance = np.abs(source_rows - target_rows) + np.abs(source_cols - target_cols)
        route_distance = route_distance.astype(np.float32) / max(2 * (active.shape[-1] - 1), 1)
        source_army = np.log1p(np.maximum(obs[candidate_rows, 0, source_rows, source_cols], 0.0))
        candidate_policy = logits[candidate_rows, candidate_actions]
        current_policy = logits[candidate_rows, current_actions]
        candidate_q = action_q[candidate_rows, candidate_actions]
        current_q = action_q[candidate_rows, current_actions]
        source_values = source_logits.reshape(num_rows, -1)[candidate_rows, candidate_sources]
        target_values = target_logits.reshape(num_rows, -1)[candidate_rows, candidate_targets]
        features = np.stack(
            [
                candidate_policy - current_policy,
                candidate_q - current_q,
                source_values,
                target_values,
                finish_prob[candidate_rows],
                source_army,
                route_distance,
                candidate_policy,
                current_policy,
                candidate_q,
                current_q,
                seats[candidate_rows],
            ],
            axis=1,
        ).astype(np.float32)
        labels_flat = labels[candidate_rows, candidate_pos].astype(np.float32)
        # Balance positives and negatives while preserving every selected example.
        positive = labels_flat > 0.5
        pos_count = max(int(np.sum(positive)), 1)
        neg_count = max(int(np.sum(~positive)), 1)
        weights = np.where(positive, 0.5 / pos_count, 0.5 / neg_count).astype(np.float32) * labels_flat.shape[0]
        feature_chunks.append(features)
        label_chunks.append(labels_flat)
        weight_chunks.append(weights)

    if not feature_chunks:
        raise ValueError("No command-gate examples selected")
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
    }


def build_strategy_worker_gate_examples(
    paths: list[Path],
    network,
    plan_worker_network,
    feature_batch_size: int,
    command_source: int,
    require_search_win: bool,
    include_finish250: bool,
    max_examples: int | None,
    seed: int,
) -> dict[str, jnp.ndarray]:
    """Construct gate examples for replacing the base action with Worker top-1.

    This uses strategy shards as a fast proxy for online replacement scoring:
    accept the Worker only when it reproduces the rollout-search teacher action
    on a state whose best searched continuation is decisive. Everything else
    where the Worker wants to change the base greedy action is treated as a
    conservative negative.
    """
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
    }
    for path in paths:
        shard = np.load(path)
        obs = shard["obs"].astype(np.float32)
        legal_mask = shard["legal_mask"].astype(np.bool_)
        active = shard["active"].astype(np.bool_)
        seats = shard["seat"].astype(np.float32)
        features, current_indices, worker_indices, _ = _compute_strategy_worker_gate_outputs(
            network,
            plan_worker_network,
            obs,
            legal_mask,
            active,
            seats,
            feature_batch_size,
            command_source,
        )
        teacher_actions = shard["teacher_action_index"].astype(np.int32)
        if "search_best_outcome" not in shard:
            decisive = np.ones((obs.shape[0],), dtype=np.bool_)
        else:
            if require_search_win:
                decisive = shard["search_best_outcome"].astype(np.int32) == OUTCOME_WIN
                if include_finish250 and "finish_within_250" in shard:
                    decisive |= shard["finish_within_250"].astype(np.float32) > 0.5
            else:
                decisive = np.ones((obs.shape[0],), dtype=np.bool_)
        pass_index = ADAPTIVE_MOVE_PLANES * active.shape[-1] * active.shape[-1]
        changed = (worker_indices != current_indices) & (worker_indices != pass_index)
        teacher_match = worker_indices == teacher_actions
        labels = (changed & teacher_match & decisive).astype(np.float32)
        keep = changed
        if not np.any(keep):
            continue
        kept_features = features[keep].astype(np.float32)
        kept_labels = labels[keep].astype(np.float32)
        positive = kept_labels > 0.5
        pos_count = max(int(np.sum(positive)), 1)
        neg_count = max(int(np.sum(~positive)), 1)
        weights = np.where(positive, 0.5 / pos_count, 0.5 / neg_count).astype(np.float32) * kept_labels.shape[0]
        feature_chunks.append(kept_features)
        label_chunks.append(kept_labels)
        weight_chunks.append(weights)
        stats["rows"] += int(obs.shape[0])
        stats["changed"] += int(np.sum(changed))
        stats["positive"] += int(np.sum(labels[keep] > 0.5))
        stats["teacher_match"] += int(np.sum(changed & teacher_match))
        stats["decisive"] += int(np.sum(decisive))

    if not feature_chunks:
        raise ValueError("No strategy-worker gate examples selected")
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
    parser = argparse.ArgumentParser(description="Train a command-acceptance gate from Plan-Q shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--dataset-format", choices=("plan-q", "strategy-worker"), default="plan-q")
    parser.add_argument("--feature-model-path", required=True)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="unet")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--input-channels", type=int, default=35)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--value-head-sizes", default="8,12,16")
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--strategy-finish-outputs", type=int, default=2)
    parser.add_argument("--plan-worker-path", default=None)
    parser.add_argument("--plan-worker-network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--plan-worker-channels", default=None)
    parser.add_argument("--plan-worker-input-channels", type=int, default=None)
    parser.add_argument(
        "--plan-worker-command-source",
        choices=PLAN_WORKER_COMMAND_SOURCE_NAMES,
        default="belief-main-stack",
    )
    parser.add_argument("--allow-nondecisive-worker-positives", action="store_true")
    parser.add_argument("--include-finish250-worker-positives", action="store_true")
    parser.add_argument("--model-path", default="runs/adaptive-command-gate/generals-adaptive-command-gate.eqx")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--score-margin", type=float, default=25.0)
    parser.add_argument("--include-noncomparable-negatives", action="store_true")
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
    if args.dataset_format == "plan-q" and not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("command gate features require --strategy-aux --strategy-spatial-aux")
    if args.dataset_format == "strategy-worker":
        if not args.strategy_aux:
            parser.error("--dataset-format strategy-worker requires --strategy-aux")
        if args.plan_worker_path is None:
            parser.error("--dataset-format strategy-worker requires --plan-worker-path")
        if args.plan_worker_command_source == "spatial" and not args.strategy_spatial_aux:
            parser.error("--plan-worker-command-source spatial requires --strategy-spatial-aux")
    if args.hidden_dim <= 0 or args.num_epochs <= 0 or args.minibatch_size <= 0 or args.feature_batch_size <= 0:
        parser.error("hidden dim, epochs, minibatch, and feature batch must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.score_margin < 0.0:
        parser.error("--score-margin must be non-negative")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    try:
        args.plan_worker_channels = parse_policy_channels(args.plan_worker_channels)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_worker_input_channels is not None and args.plan_worker_input_channels <= 0:
        parser.error("--plan-worker-input-channels must be positive")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    key = jrandom.PRNGKey(args.seed)
    key, net_key, gate_key = jrandom.split(key, 3)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    network = load_or_create_adaptive_network(
        net_key,
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
    )
    if args.dataset_format == "strategy-worker":
        plan_worker_input_channels = args.plan_worker_input_channels or (args.input_channels + 3)
        plan_worker_network = load_or_create_adaptive_network(
            net_key,
            pad_size=16,
            init_model_path=args.plan_worker_path,
            channels=args.plan_worker_channels,
            input_channels=plan_worker_input_channels,
            init_input_channels=plan_worker_input_channels,
            network_arch=args.plan_worker_network_arch,
            init_network_arch=args.plan_worker_network_arch,
        )
        dataset = build_strategy_worker_gate_examples(
            paths,
            network,
            plan_worker_network,
            args.feature_batch_size,
            PLAN_WORKER_COMMAND_SOURCE_TO_ID[args.plan_worker_command_source],
            not args.allow_nondecisive_worker_positives,
            args.include_finish250_worker_positives,
            args.max_examples,
            args.seed,
        )
    else:
        dataset = build_gate_examples(
            paths,
            network,
            args.feature_batch_size,
            args.score_margin,
            args.include_noncomparable_negatives,
            args.max_examples,
            args.seed,
        )
    labels = np.asarray(dataset["labels"])
    print("Adaptive command-gate supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Examples:      {dataset['features'].shape[0]}")
    print(f"Positive:      {float(np.mean(labels)) * 100:.2f}%")
    print(f"Format:        {args.dataset_format}")
    print(f"Feature model: {args.feature_model_path}")
    if args.dataset_format == "strategy-worker":
        print(f"Plan worker:   {args.plan_worker_path}")
        print(f"Command src:   {args.plan_worker_command_source}")
        stats = dataset.get("stats", {})
        if stats:
            print(
                "Worker stats:  "
                f"rows={stats['rows']}, changed={stats['changed']}, "
                f"teacher_match={stats['teacher_match']}, decisive={stats['decisive']}"
            )
    print(f"Output:        {args.model_path}")
    print(f"Features:      {', '.join(COMMAND_GATE_FEATURE_NAMES)}")
    print()

    gate = CommandGateNetwork(
        gate_key,
        input_dim=COMMAND_GATE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        feature_mean=dataset["feature_mean"],
        feature_std=dataset["feature_std"],
    )
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(gate, eqx.is_inexact_array))
    for epoch in range(1, args.num_epochs + 1):
        key, epoch_key = jrandom.split(key)
        t0 = time.time()
        gate, opt_state, loss, metrics = train_epoch(gate, opt_state, dataset, optimizer, epoch_key, args.minibatch_size)
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Acc {float(metrics['accuracy']) * 100:5.1f}% | "
            f"P+ {float(metrics['positive_prob']):.3f} | "
            f"P- {float(metrics['negative_prob']):.3f} | "
            f"Pmean {float(metrics['mean_prob']):.3f} | "
            f"Time {time.time() - t0:.2f}s"
        )
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(model_path, gate)
    sidecar = {
        "feature_names": list(COMMAND_GATE_FEATURE_NAMES),
        "feature_mean": np.asarray(gate.feature_mean).tolist(),
        "feature_std": np.asarray(gate.feature_std).tolist(),
        "hidden_dim": args.hidden_dim,
        "score_margin": args.score_margin,
        "examples": int(dataset["features"].shape[0]),
        "positive_fraction": float(np.mean(labels)),
        "dataset_format": args.dataset_format,
        "feature_model_path": args.feature_model_path,
        "plan_worker_path": args.plan_worker_path,
        "plan_worker_command_source": args.plan_worker_command_source,
        "datasets": [str(path) for path in paths],
    }
    if "stats" in dataset:
        sidecar["stats"] = dataset["stats"]
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
