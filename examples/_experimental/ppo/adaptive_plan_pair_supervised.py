"""Train an explicit source-target plan pair scorer from Plan-Q shards."""

from __future__ import annotations

import argparse
import copy
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
from adaptive_plan_pair_scorer import PlanPairScorerNetwork, plan_pair_feature_names
from adaptive_plan_q_supervised import plan_value_targets
from generals.agents.ppo_policy_agent import parse_policy_channels


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


def _compute_network_outputs(network, obs: np.ndarray, legal_mask: np.ndarray, active: np.ndarray, batch_size: int):
    """Run the adaptive policy once to get inference-time pair features."""
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
        finish_chunks.append(np.asarray(jax.nn.softmax(aux.finish_logits, axis=-1)[:, 1]))
    return (
        np.concatenate(logits_chunks, axis=0),
        np.concatenate(q_chunks, axis=0),
        np.concatenate(source_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(finish_chunks, axis=0),
    )


def _safe_action_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    safe_indices = np.clip(indices, 0, values.shape[1] - 1)
    return values[np.arange(values.shape[0])[:, None], safe_indices]


def build_pair_dataset(
    paths: list[Path],
    network,
    feature_batch_size: int,
    q_target_outcome_weight: float,
    gap_weighting: bool,
    validation_fraction: float,
    max_rows: int | None,
    seed: int,
) -> dict[str, jnp.ndarray | int | float]:
    """Construct row-wise source-target ranking examples from Plan-Q shards."""
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    source_pos_rows: list[np.ndarray] = []
    target_pos_rows: list[np.ndarray] = []
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
        source_indices = shard["source_indices"].astype(np.int32)
        target_indices = shard["target_indices"].astype(np.int32)
        plan_actions = shard["plan_action_indices"].astype(np.int32)
        plan_q = shard["plan_q"].astype(np.float32)
        plan_outcomes = shard["plan_outcomes"].astype(np.int32)
        plan_values = np.asarray(
            plan_value_targets(jnp.asarray(plan_q), jnp.asarray(plan_outcomes), q_target_outcome_weight)
        )
        teacher_actions = shard["teacher_action_index"].astype(np.int32)
        seats = shard["seat"].astype(np.float32)
        grid_sizes = shard["grid_size"].astype(np.float32)
        turns = shard["time"].astype(np.float32)
        plan_gap = shard["plan_q_gap"].astype(np.float32)
        rows = obs.shape[0]
        pad_size = obs.shape[-1]
        source_count = source_indices.shape[1]
        target_count = target_indices.shape[1]
        pair_count = source_count * target_count

        flat_actions = plan_actions.reshape(rows, pair_count)
        flat_values = plan_values.reshape(rows, pair_count)
        source_grid = np.broadcast_to(source_indices[:, :, None], (rows, source_count, target_count)).reshape(
            rows,
            pair_count,
        )
        target_grid = np.broadcast_to(target_indices[:, None, :], (rows, source_count, target_count)).reshape(
            rows,
            pair_count,
        )
        candidate_policy = _safe_action_values(logits, flat_actions)
        candidate_q = _safe_action_values(action_q, flat_actions)
        current_policy = logits[np.arange(rows), teacher_actions][:, None]
        current_q = action_q[np.arange(rows), teacher_actions][:, None]
        flat_source_logits = source_logits.reshape(rows, -1)
        flat_target_logits = target_logits.reshape(rows, -1)
        source_values = flat_source_logits[np.arange(rows)[:, None], source_grid]
        target_values = flat_target_logits[np.arange(rows)[:, None], target_grid]

        source_r = source_grid // pad_size
        source_c = source_grid % pad_size
        target_r = target_grid // pad_size
        target_c = target_grid % pad_size
        route_distance = np.abs(source_r - target_r) + np.abs(source_c - target_c)
        route_distance = route_distance.astype(np.float32) / max(2 * (pad_size - 1), 1)
        source_army = np.log1p(np.maximum(obs[np.arange(rows)[:, None], 0, source_r, source_c], 0.0))
        target_army = np.log1p(np.maximum(obs[np.arange(rows)[:, None], 0, target_r, target_c], 0.0))
        denom = np.maximum(grid_sizes[:, None] - 1.0, 1.0)
        source_r_norm = source_r.astype(np.float32) / denom
        source_c_norm = source_c.astype(np.float32) / denom
        target_r_norm = target_r.astype(np.float32) / denom
        target_c_norm = target_c.astype(np.float32) / denom
        row_delta = (target_r - source_r).astype(np.float32) / denom
        col_delta = (target_c - source_c).astype(np.float32) / denom
        grid_norm = grid_sizes[:, None] / pad_size
        turn_norm = turns[:, None] / 750.0
        seat_feature = seats[:, None]

        row_ids = np.arange(rows)[:, None]
        source_cell_features = obs[row_ids, :, source_r, source_c]
        target_cell_features = obs[row_ids, :, target_r, target_c]
        base_features = np.stack(
            [
                candidate_policy - current_policy,
                candidate_q - current_q,
                source_values,
                target_values,
                np.broadcast_to(finish_prob[:, None], (rows, pair_count)),
                source_army,
                target_army,
                route_distance,
                candidate_policy,
                np.broadcast_to(current_policy, (rows, pair_count)),
                candidate_q,
                np.broadcast_to(current_q, (rows, pair_count)),
                source_r_norm,
                source_c_norm,
                target_r_norm,
                target_c_norm,
                row_delta,
                col_delta,
                np.broadcast_to(grid_norm, (rows, pair_count)),
                np.broadcast_to(turn_norm, (rows, pair_count)),
                np.broadcast_to(seat_feature, (rows, pair_count)),
            ],
            axis=2,
        )
        features = np.concatenate([base_features, source_cell_features, target_cell_features], axis=2)
        weights = np.maximum(plan_gap, 0.05).astype(np.float32) if gap_weighting else np.ones((rows,), dtype=np.float32)
        target_best = np.argmax(flat_values, axis=1)
        feature_rows.append(features.astype(np.float32))
        target_rows.append(flat_values.astype(np.float32))
        weight_rows.append(weights)
        source_pos_rows.append((target_best // target_count).astype(np.int32))
        target_pos_rows.append((target_best % target_count).astype(np.int32))

    features = np.concatenate(feature_rows, axis=0)
    targets = np.concatenate(target_rows, axis=0)
    weights = np.concatenate(weight_rows, axis=0)
    source_pos = np.concatenate(source_pos_rows, axis=0)
    target_pos = np.concatenate(target_pos_rows, axis=0)
    if max_rows is not None and features.shape[0] > max_rows:
        indices = np.sort(rng.choice(features.shape[0], size=max_rows, replace=False))
        features = features[indices]
        targets = targets[indices]
        weights = weights[indices]
        source_pos = source_pos[indices]
        target_pos = target_pos[indices]

    permutation = rng.permutation(features.shape[0])
    val_count = int(round(features.shape[0] * validation_fraction))
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]
    if train_indices.shape[0] == 0 or val_indices.shape[0] == 0:
        raise ValueError("validation split must leave at least one train and one validation row")
    train_flat = features[train_indices].reshape(-1, features.shape[-1])
    feature_mean = train_flat.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(train_flat.std(axis=0).astype(np.float32), 1.0e-6)
    return {
        "train_features": jnp.asarray(features[train_indices]),
        "train_targets": jnp.asarray(targets[train_indices]),
        "train_weights": jnp.asarray(weights[train_indices]),
        "train_source_pos": jnp.asarray(source_pos[train_indices]),
        "train_target_pos": jnp.asarray(target_pos[train_indices]),
        "val_features": jnp.asarray(features[val_indices]),
        "val_targets": jnp.asarray(targets[val_indices]),
        "val_weights": jnp.asarray(weights[val_indices]),
        "val_source_pos": jnp.asarray(source_pos[val_indices]),
        "val_target_pos": jnp.asarray(target_pos[val_indices]),
        "feature_mean": jnp.asarray(feature_mean),
        "feature_std": jnp.asarray(feature_std),
        "input_dim": int(features.shape[-1]),
        "pair_count": int(features.shape[1]),
        "target_count": int(target_count),
        "rows": int(features.shape[0]),
    }


def _score_pairs(model, features: jnp.ndarray) -> jnp.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    scores = jax.vmap(model)(flat)
    return scores.reshape(features.shape[0], features.shape[1])


def pair_rank_metrics(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
    source_pos: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_count: int,
    temperature: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    target_probs = jax.nn.softmax(jax.lax.stop_gradient(targets) / temperature, axis=1)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    losses = -jnp.sum(target_probs * log_probs, axis=1)
    normalizer = jnp.maximum(jnp.sum(weights), 1.0)
    loss = jnp.sum(losses * weights) / normalizer
    pred_best = jnp.argmax(logits, axis=1)
    target_best = jnp.argmax(targets, axis=1)
    pred_source = pred_best // target_count
    pred_target = pred_best % target_count
    pair_accuracy = jnp.sum((pred_best == target_best).astype(jnp.float32) * weights) / normalizer
    source_accuracy = jnp.sum((pred_source == source_pos).astype(jnp.float32) * weights) / normalizer
    target_accuracy = jnp.sum((pred_target == target_pos).astype(jnp.float32) * weights) / normalizer
    pred_centered = logits - jnp.mean(logits, axis=1, keepdims=True)
    target_centered = targets - jnp.mean(targets, axis=1, keepdims=True)
    covariance = jnp.mean(pred_centered * target_centered, axis=1)
    pred_std = jnp.sqrt(jnp.mean(pred_centered**2, axis=1) + 1.0e-6)
    target_std = jnp.sqrt(jnp.mean(target_centered**2, axis=1) + 1.0e-6)
    correlation = jnp.sum((covariance / (pred_std * target_std)) * weights) / normalizer
    margin = jnp.mean(jax.lax.top_k(logits, 2)[0][:, 0] - jax.lax.top_k(logits, 2)[0][:, 1])
    return loss, {
        "pair_accuracy": pair_accuracy,
        "source_accuracy": source_accuracy,
        "target_accuracy": target_accuracy,
        "correlation": correlation,
        "margin": margin,
    }


@eqx.filter_jit
def train_step(model, opt_state, batch, optimizer, target_count: int, temperature: float):
    features, targets, weights, source_pos, target_pos = batch

    def loss_fn(candidate_model):
        logits = _score_pairs(candidate_model, features)
        return pair_rank_metrics(logits, targets, weights, source_pos, target_pos, target_count, temperature)

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss, metrics


@eqx.filter_jit
def evaluate_model(model, features, targets, weights, source_pos, target_pos, target_count: int, temperature: float):
    logits = _score_pairs(model, features)
    return pair_rank_metrics(logits, targets, weights, source_pos, target_pos, target_count, temperature)


def train_epoch(model, opt_state, dataset, optimizer, key, minibatch_size: int, temperature: float):
    num_rows = dataset["train_features"].shape[0]
    permutation = jrandom.permutation(key, num_rows)
    num_batches = max(num_rows // minibatch_size, 1)
    metrics_sum = None
    loss_sum = 0.0
    for batch_index in range(num_batches):
        start = batch_index * minibatch_size
        end = min(start + minibatch_size, num_rows)
        idx = permutation[start:end]
        batch = (
            dataset["train_features"][idx],
            dataset["train_targets"][idx],
            dataset["train_weights"][idx],
            dataset["train_source_pos"][idx],
            dataset["train_target_pos"][idx],
        )
        model, opt_state, loss, metrics = train_step(
            model,
            opt_state,
            batch,
            optimizer,
            dataset["target_count"],
            temperature,
        )
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return model, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an explicit Plan-Q source-target pair scorer.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
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
    parser.add_argument("--model-path", default="runs/adaptive-plan-pair-scorer/generals-adaptive-plan-pair-scorer.eqx")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-target-outcome-weight", type=float, default=0.65)
    parser.add_argument("--q-rank-temperature", type=float, default=0.05)
    parser.add_argument("--gap-weighting", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None)
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
    if not (args.strategy_aux and args.strategy_spatial_aux):
        parser.error("pair scorer features require --strategy-aux --strategy-spatial-aux")
    if args.hidden_dim <= 0 or args.num_epochs <= 0 or args.minibatch_size <= 0 or args.feature_batch_size <= 0:
        parser.error("hidden dim, epochs, minibatch, and feature batch must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if not (0.0 <= args.q_target_outcome_weight <= 1.0):
        parser.error("--q-target-outcome-weight must be in [0, 1]")
    if args.q_rank_temperature <= 0.0:
        parser.error("--q-rank-temperature must be positive")
    if not (0.0 < args.validation_fraction < 1.0):
        parser.error("--validation-fraction must be between 0 and 1")
    if args.max_rows is not None and args.max_rows <= 1:
        parser.error("--max-rows must be greater than 1")
    return args


def _metric_text(prefix: str, loss, metrics: dict[str, jnp.ndarray]) -> str:
    return (
        f"{prefix} {float(loss):.4f}/"
        f"{float(metrics['pair_accuracy']) * 100:5.1f}%/"
        f"S{float(metrics['source_accuracy']) * 100:5.1f}%/"
        f"T{float(metrics['target_accuracy']) * 100:5.1f}%/"
        f"{float(metrics['correlation']):+.3f}/"
        f"M{float(metrics['margin']):.3f}"
    )


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    key = jrandom.PRNGKey(args.seed)
    key, net_key, scorer_key = jrandom.split(key, 3)
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
        global_context=args.global_context,
        init_global_context=args.global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
    )
    dataset = build_pair_dataset(
        paths,
        network,
        args.feature_batch_size,
        args.q_target_outcome_weight,
        args.gap_weighting,
        args.validation_fraction,
        args.max_rows,
        args.seed,
    )
    print("Adaptive Plan-Q pair-scorer supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Rows:          {dataset['rows']}")
    print(f"Train/val:     {dataset['train_features'].shape[0]} / {dataset['val_features'].shape[0]}")
    print(f"Pairs/row:     {dataset['pair_count']}")
    print(f"Input dim:     {dataset['input_dim']}")
    print(f"Feature model: {args.feature_model_path}")
    print(f"Output:        {args.model_path}")
    print(
        "Q-target:     "
        f"outcome_weight={args.q_target_outcome_weight:g}, rank_temp={args.q_rank_temperature:g}, "
        f"gap_weighting={args.gap_weighting}"
    )
    print()

    scorer = PlanPairScorerNetwork(
        scorer_key,
        input_dim=dataset["input_dim"],
        hidden_dim=args.hidden_dim,
        feature_mean=dataset["feature_mean"],
        feature_std=dataset["feature_std"],
    )
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(scorer, eqx.is_inexact_array))
    best_scorer = copy.deepcopy(scorer)
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_metrics: dict[str, float] = {
        "pair_accuracy": -1.0,
        "source_accuracy": 0.0,
        "target_accuracy": 0.0,
        "correlation": 0.0,
        "margin": 0.0,
    }
    for epoch in range(1, args.num_epochs + 1):
        key, epoch_key = jrandom.split(key)
        t0 = time.time()
        scorer, opt_state, train_loss, train_metrics = train_epoch(
            scorer,
            opt_state,
            dataset,
            optimizer,
            epoch_key,
            args.minibatch_size,
            args.q_rank_temperature,
        )
        val_loss, val_metrics = evaluate_model(
            scorer,
            dataset["val_features"],
            dataset["val_targets"],
            dataset["val_weights"],
            dataset["val_source_pos"],
            dataset["val_target_pos"],
            dataset["target_count"],
            args.q_rank_temperature,
        )
        val_pair_accuracy = float(val_metrics["pair_accuracy"])
        val_loss_float = float(val_loss)
        if val_pair_accuracy > best_val_metrics["pair_accuracy"]:
            best_scorer = copy.deepcopy(scorer)
            best_epoch = epoch
            best_val_loss = val_loss_float
            best_val_metrics = {name: float(value) for name, value in val_metrics.items()}
        print(
            f"Epoch {epoch:03d} | "
            f"{_metric_text('Train', train_loss, train_metrics)} | "
            f"{_metric_text('Val', val_loss, val_metrics)} | "
            f"Time {time.time() - t0:.2f}s"
        )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(model_path, scorer)
    best_model_path = model_path.with_name(f"{model_path.stem}.best{model_path.suffix}")
    eqx.tree_serialise_leaves(best_model_path, best_scorer)
    feature_names = plan_pair_feature_names(args.input_channels)
    sidecar = {
        "feature_names": list(feature_names),
        "feature_mean": np.asarray(scorer.feature_mean).tolist(),
        "feature_std": np.asarray(scorer.feature_std).tolist(),
        "hidden_dim": args.hidden_dim,
        "input_dim": dataset["input_dim"],
        "pair_count": dataset["pair_count"],
        "target_count": dataset["target_count"],
        "rows": dataset["rows"],
        "train_rows": int(dataset["train_features"].shape[0]),
        "val_rows": int(dataset["val_features"].shape[0]),
        "q_target_outcome_weight": args.q_target_outcome_weight,
        "q_rank_temperature": args.q_rank_temperature,
        "gap_weighting": args.gap_weighting,
        "best_model_path": str(best_model_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "feature_model_path": args.feature_model_path,
        "datasets": [str(path) for path in paths],
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best_model_path.with_suffix(".json").write_text(
        json.dumps({**sidecar, "model_path": str(best_model_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nModel saved to: {model_path}")
    print(
        "Best validation: "
        f"epoch {best_epoch}, loss {best_val_loss:.4f}, "
        f"pair {best_val_metrics['pair_accuracy'] * 100:.1f}%, "
        f"source {best_val_metrics['source_accuracy'] * 100:.1f}%, "
        f"target {best_val_metrics['target_accuracy'] * 100:.1f}%, "
        f"corr {best_val_metrics['correlation']:+.3f}"
    )
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
