"""Train a candidate-action scorer from online-search trace shards."""

from __future__ import annotations

import argparse
import glob
import json
import sys
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

from adaptive_common import ADAPTIVE_MOVE_PLANES

DIRECTIONS = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
POSITIVE_FIELDS = (
    "search_converts_to_win",
    "search_converts_draw_to_win",
    "search_improves_continuation",
)
BASE_FEATURE_NAMES = (
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
    """Normalized MLP that scores one candidate primitive action."""

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
        # These normalization statistics are dataset metadata, not trainable
        # parameters.  Keep them as leaves so checkpoints are self-contained,
        # but stop gradients so Optax cannot drift or invert the scale.
        feature_mean = jax.lax.stop_gradient(self.feature_mean)
        feature_std = jnp.maximum(jax.lax.stop_gradient(self.feature_std), 1.0e-6)
        x = (features - feature_mean) / feature_std
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)[0]


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


def _take2d(values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0])[:, None], rows, cols]


def _candidate_geometry(indices: np.ndarray, pad_size: int) -> dict[str, np.ndarray]:
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    is_pass = indices == pass_index
    safe = np.minimum(np.clip(indices, 0, pass_index), pass_index - 1)
    plane = safe // (pad_size * pad_size)
    position = safe % (pad_size * pad_size)
    source_row = position // pad_size
    source_col = position % pad_size
    direction = plane % 4
    is_half = plane >= 4
    dest_row = source_row + DIRECTIONS[direction, 0]
    dest_col = source_col + DIRECTIONS[direction, 1]
    dest_in_bounds = (dest_row >= 0) & (dest_row < pad_size) & (dest_col >= 0) & (dest_col < pad_size)
    return {
        "is_pass": is_pass,
        "source_row": np.clip(source_row, 0, pad_size - 1),
        "source_col": np.clip(source_col, 0, pad_size - 1),
        "direction": direction,
        "is_half": is_half,
        "dest_row": np.clip(dest_row, 0, pad_size - 1),
        "dest_col": np.clip(dest_col, 0, pad_size - 1),
        "dest_in_bounds": dest_in_bounds,
    }


def build_candidate_features_from_shard(
    shard: np.lib.npyio.NpzFile,
    max_steps: int,
    local_channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return row-candidate features and search score targets for one trace shard."""
    obs = shard["obs"].astype(np.float32)
    active = shard["active"].astype(np.float32)
    candidates = shard["search_candidate_indices"].astype(np.int32)
    prior_scores = shard["search_prior_scores"].astype(np.float32)
    search_scores = shard["search_scores"].astype(np.float32)
    rows, top_k = candidates.shape
    pad_size = obs.shape[-1]
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    geometry = _candidate_geometry(candidates, pad_size)
    source_row = geometry["source_row"]
    source_col = geometry["source_col"]
    dest_row = geometry["dest_row"]
    dest_col = geometry["dest_col"]
    direction = geometry["direction"]
    is_pass = geometry["is_pass"]

    base_action = shard["base_action_index"].astype(np.int32)[:, None]
    safe_base = np.clip(base_action[:, 0], 0, shard["teacher_logits"].shape[1] - 1)
    base_prior = shard["teacher_logits"].astype(np.float32)[np.arange(rows), safe_base][:, None]
    source_active = _take2d(active, source_row, source_col)
    dest_active = _take2d(active, dest_row, dest_col) * geometry["dest_in_bounds"].astype(np.float32)
    legal_dir = shard["legal_mask"].astype(np.float32)[
        np.arange(rows)[:, None],
        source_row,
        source_col,
        direction,
    ]

    row_index = np.arange(rows)[:, None]
    source_planes = obs[row_index, :, source_row, source_col]
    dest_planes = obs[row_index, :, dest_row, dest_col]

    source_army = source_planes[:, :, 0]
    dest_army = dest_planes[:, :, 0]
    move_all = np.maximum(np.expm1(np.maximum(source_army, 0.0)) - 1.0, 0.0)
    move_half = np.floor(np.maximum(np.expm1(np.maximum(source_army, 0.0)), 0.0) / 2.0)
    dest_army_raw = np.maximum(np.expm1(np.maximum(dest_army, 0.0)), 0.0)

    dir_one_hot = np.eye(4, dtype=np.float32)[direction]
    grid_size = shard["grid_size"].astype(np.float32)[:, None]
    feature_parts = [
        prior_scores[:, :, None],
        np.broadcast_to(
            np.arange(top_k, dtype=np.float32)[None, :, None] / max(top_k - 1, 1),
            (rows, top_k, 1),
        ),
        (prior_scores - base_prior)[:, :, None],
        (candidates == base_action).astype(np.float32)[:, :, None],
        is_pass.astype(np.float32)[:, :, None],
        geometry["is_half"].astype(np.float32)[:, :, None],
        dir_one_hot,
        (source_row.astype(np.float32) / max(pad_size - 1, 1))[:, :, None],
        (source_col.astype(np.float32) / max(pad_size - 1, 1))[:, :, None],
        (dest_row.astype(np.float32) / max(pad_size - 1, 1))[:, :, None],
        (dest_col.astype(np.float32) / max(pad_size - 1, 1))[:, :, None],
        source_active[:, :, None],
        dest_active[:, :, None],
        legal_dir[:, :, None],
        source_army[:, :, None],
        dest_army[:, :, None],
        source_planes[:, :, 5:6],
        dest_planes[:, :, 5:6],
        dest_planes[:, :, 6:7],
        dest_planes[:, :, 4:5],
        dest_planes[:, :, 2:3],
        dest_planes[:, :, 7:8],
        dest_planes[:, :, 8:9],
        (source_planes[:, :, 1:2] * source_planes[:, :, 5:6]),
        (dest_planes[:, :, 1:2] * dest_planes[:, :, 6:7]),
        np.log1p(np.maximum(move_all - dest_army_raw, 0.0))[:, :, None],
        np.log1p(np.maximum(move_half - dest_army_raw, 0.0))[:, :, None],
        (shard["time"].astype(np.float32)[:, None, None] / float(max(max_steps, 1))).repeat(top_k, axis=1),
        shard["seat"].astype(np.float32)[:, None, None].repeat(top_k, axis=1),
        (grid_size[:, :, None] / float(max(pad_size, 1))).repeat(top_k, axis=1),
        active.reshape(rows, -1).mean(axis=1)[:, None, None].repeat(top_k, axis=1),
        shard["visible_enemy_density"].astype(np.float32)[:, None, None].repeat(top_k, axis=1),
        shard["contact"].astype(np.float32)[:, None, None].repeat(top_k, axis=1),
    ]
    kept_channels = min(local_channels, obs.shape[1])
    if kept_channels > 0:
        feature_parts.extend(
            [
                source_planes[:, :, :kept_channels],
                dest_planes[:, :, :kept_channels],
                source_planes[:, :, :kept_channels] - dest_planes[:, :, :kept_channels],
            ]
        )
    features = np.concatenate(feature_parts, axis=-1).astype(np.float32)
    valid = np.isfinite(search_scores) & (search_scores > -9990.0) & (candidates >= 0) & (candidates <= pass_index)
    stats = {
        "rows": rows,
        "changed": int(np.sum(shard["search_action_changed"].astype(np.bool_))),
        "converts": int(np.sum(shard["search_converts_to_win"].astype(np.bool_))),
        "improves": int(np.sum(shard["search_improves_continuation"].astype(np.bool_))),
    }
    return features, search_scores.astype(np.float32), valid.astype(np.bool_), prior_scores, stats


def load_dataset(
    paths: list[Path],
    require_search_used: bool,
    require_action_changed: bool,
    min_score_gap: float,
    positive_field: str | None,
    max_steps: int,
    local_channels: int,
    max_rows: int | None,
    seed: int,
) -> tuple[dict[str, jnp.ndarray], dict[str, object]]:
    """Load trace shards into row-wise candidate scoring tensors."""
    rng = np.random.default_rng(seed)
    feature_chunks = []
    score_chunks = []
    valid_chunks = []
    prior_chunks = []
    stats = {
        "paths": [str(path) for path in paths],
        "raw_rows": 0,
        "kept_rows": 0,
        "changed": 0,
        "converts": 0,
        "improves": 0,
    }
    for path in paths:
        with np.load(path) as shard:
            features, scores, valid, priors, shard_stats = build_candidate_features_from_shard(
                shard,
                max_steps=max_steps,
                local_channels=local_channels,
            )
            keep = np.ones((features.shape[0],), dtype=np.bool_)
            if require_search_used:
                keep &= shard["search_used"].astype(np.bool_)
            if require_action_changed:
                keep &= shard["search_action_changed"].astype(np.bool_)
            if min_score_gap > 0.0:
                keep &= shard["search_score_gap"].astype(np.float32) >= min_score_gap
            if positive_field is not None:
                keep &= shard[positive_field].astype(np.bool_)
            keep &= np.sum(valid, axis=1) >= 2
            stats["raw_rows"] += int(shard_stats["rows"])
            stats["changed"] += int(shard_stats["changed"])
            stats["converts"] += int(shard_stats["converts"])
            stats["improves"] += int(shard_stats["improves"])
            if not np.any(keep):
                continue
            feature_chunks.append(features[keep])
            score_chunks.append(scores[keep])
            valid_chunks.append(valid[keep])
            prior_chunks.append(priors[keep])
            stats["kept_rows"] += int(np.sum(keep))
    if not feature_chunks:
        raise ValueError("No candidate-scorer examples selected")
    features = np.concatenate(feature_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)
    valid = np.concatenate(valid_chunks, axis=0)
    priors = np.concatenate(prior_chunks, axis=0)
    if max_rows is not None and features.shape[0] > max_rows:
        indices = np.sort(rng.choice(features.shape[0], size=max_rows, replace=False))
        features = features[indices]
        scores = scores[indices]
        valid = valid[indices]
        priors = priors[indices]
    flat_valid_features = features[valid]
    feature_mean = flat_valid_features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(flat_valid_features.std(axis=0).astype(np.float32), 1.0e-6)
    stats["rows"] = int(features.shape[0])
    stats["top_k"] = int(features.shape[1])
    stats["feature_dim"] = int(features.shape[2])
    stats["prior_top1"] = float(np.mean(np.argmax(priors, axis=1) == np.argmax(scores, axis=1)))
    stats["prior_top2"] = float(np.mean([np.argmax(scores[i]) in np.argsort(priors[i])[-2:] for i in range(scores.shape[0])]))
    return (
        {
            "features": jnp.asarray(features),
            "scores": jnp.asarray(scores),
            "valid": jnp.asarray(valid),
            "priors": jnp.asarray(priors),
            "feature_mean": jnp.asarray(feature_mean),
            "feature_std": jnp.asarray(feature_std),
        },
        stats,
    )


def split_dataset(dataset: dict[str, jnp.ndarray], val_fraction: float, seed: int):
    rows = dataset["features"].shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(rows)
    val_rows = max(1, int(round(rows * val_fraction)))
    val_rows = min(val_rows, rows - 1)
    val_idx = np.sort(permutation[:val_rows])
    train_idx = np.sort(permutation[val_rows:])
    return (
        {name: value[train_idx] for name, value in dataset.items() if name not in {"feature_mean", "feature_std"}},
        {name: value[val_idx] for name, value in dataset.items() if name not in {"feature_mean", "feature_std"}},
        train_idx,
        val_idx,
    )


def predict_scores(model, features: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(jax.vmap(model))(features)


def rank_metrics(predicted: jnp.ndarray, targets: jnp.ndarray, valid: jnp.ndarray, priors: jnp.ndarray) -> dict[str, jnp.ndarray]:
    masked_pred = jnp.where(valid, predicted, -1.0e9)
    masked_targets = jnp.where(valid, targets, -1.0e9)
    masked_priors = jnp.where(valid, priors, -1.0e9)
    target_best = jnp.argmax(masked_targets, axis=1)
    pred_best = jnp.argmax(masked_pred, axis=1)
    prior_best = jnp.argmax(masked_priors, axis=1)
    pred_top2 = jnp.argsort(masked_pred, axis=1)[:, -2:]
    prior_top2 = jnp.argsort(masked_priors, axis=1)[:, -2:]
    target_best_col = target_best[:, None]
    target_diff = masked_targets[:, :, None] - masked_targets[:, None, :]
    pred_diff = masked_pred[:, :, None] - masked_pred[:, None, :]
    pair_mask = valid[:, :, None] & valid[:, None, :] & (jnp.abs(target_diff) > 1.0e-6)
    pair_acc = jnp.sum(((target_diff > 0.0) == (pred_diff > 0.0)) * pair_mask) / jnp.maximum(jnp.sum(pair_mask), 1)
    return {
        "top1": jnp.mean((pred_best == target_best).astype(jnp.float32)),
        "top2": jnp.mean(jnp.any(pred_top2 == target_best_col, axis=1).astype(jnp.float32)),
        "pair": pair_acc,
        "prior_top1": jnp.mean((prior_best == target_best).astype(jnp.float32)),
        "prior_top2": jnp.mean(jnp.any(prior_top2 == target_best_col, axis=1).astype(jnp.float32)),
        "mean_gap": jnp.mean(jnp.max(masked_targets, axis=1) - jnp.partition(masked_targets, -2, axis=1)[:, -2]),
    }


@eqx.filter_jit
def train_step(
    model,
    opt_state,
    batch,
    optimizer,
    temperature: float,
    gap_weighting: bool,
    hard_best_weight: float,
):
    features, scores, valid, priors = batch

    def loss_fn(candidate_model):
        predicted = predict_scores(candidate_model, features)
        target_logits = jnp.where(valid, scores / temperature, -1.0e9)
        target_probs = jax.nn.softmax(target_logits, axis=1)
        pred_logits = jnp.where(valid, predicted, -1.0e9)
        soft_rank_loss = -jnp.sum(target_probs * jax.nn.log_softmax(pred_logits, axis=1), axis=1)
        target_best = jnp.argmax(jnp.where(valid, scores, -1.0e9), axis=1)
        hard_best_loss = optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_best)
        per_row = soft_rank_loss + hard_best_weight * hard_best_loss
        if gap_weighting:
            best = jnp.max(jnp.where(valid, scores, -1.0e9), axis=1)
            second = jnp.partition(jnp.where(valid, scores, -1.0e9), -2, axis=1)[:, -2]
            weights = jnp.clip((best - second) / 25.0, 0.25, 4.0)
            per_row = per_row * weights
        return jnp.mean(per_row), rank_metrics(predicted, scores, valid, priors)

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss, metrics


@eqx.filter_jit
def evaluate_loss(model, batch, temperature: float, hard_best_weight: float):
    features, scores, valid, priors = batch
    predicted = predict_scores(model, features)
    target_logits = jnp.where(valid, scores / temperature, -1.0e9)
    target_probs = jax.nn.softmax(target_logits, axis=1)
    pred_logits = jnp.where(valid, predicted, -1.0e9)
    soft_rank_loss = -jnp.sum(target_probs * jax.nn.log_softmax(pred_logits, axis=1), axis=1)
    target_best = jnp.argmax(jnp.where(valid, scores, -1.0e9), axis=1)
    hard_best_loss = optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_best)
    loss = jnp.mean(soft_rank_loss + hard_best_weight * hard_best_loss)
    return loss, rank_metrics(predicted, scores, valid, priors)


def train_epoch(
    model,
    opt_state,
    train_data,
    optimizer,
    key,
    minibatch_size: int,
    temperature: float,
    gap_weighting: bool,
    hard_best_weight: float,
):
    rows = train_data["features"].shape[0]
    permutation = jrandom.permutation(key, rows)
    loss_sum = 0.0
    metrics_sum = None
    batches = 0
    for start in range(0, rows, minibatch_size):
        idx = permutation[start : min(start + minibatch_size, rows)]
        batch = (
            train_data["features"][idx],
            train_data["scores"][idx],
            train_data["valid"][idx],
            train_data["priors"][idx],
        )
        model, opt_state, loss, metrics = train_step(
            model,
            opt_state,
            batch,
            optimizer,
            temperature,
            gap_weighting,
            hard_best_weight,
        )
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
        batches += 1
    return model, opt_state, loss_sum / batches, jax.tree.map(lambda value: value / batches, metrics_sum)


def as_batch(data: dict[str, jnp.ndarray]):
    return data["features"], data["scores"], data["valid"], data["priors"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a candidate scorer from online-search trace shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument(
        "--val-dataset",
        action="append",
        default=None,
        help="Optional independent validation NPZ shard path or glob. Repeatable.",
    )
    parser.add_argument("--require-search-used", action="store_true")
    parser.add_argument("--require-action-changed", action="store_true")
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--positive-field", choices=POSITIVE_FIELDS, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--local-channels", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument(
        "--hard-best-weight",
        type=float,
        default=0.0,
        help="Additional CE weight for the single best rollout-search candidate.",
    )
    parser.add_argument("--gap-weighting", action="store_true")
    parser.add_argument(
        "--model-path",
        default="runs/adaptive-online-search-candidate-scorer/generals-adaptive-online-search-candidate-scorer.eqx",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.local_channels < 0:
        parser.error("--local-channels must be non-negative")
    if args.max_rows is not None and args.max_rows <= 1:
        parser.error("--max-rows must leave at least two rows")
    if args.max_val_rows is not None and args.max_val_rows <= 0:
        parser.error("--max-val-rows must be positive")
    if not (0.0 < args.val_fraction < 1.0):
        parser.error("--val-fraction must be between 0 and 1")
    if args.hidden_dim <= 0 or args.num_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("--hidden-dim, --num-epochs, and --minibatch-size must be positive")
    if args.lr <= 0.0 or args.temperature <= 0.0:
        parser.error("--lr and --temperature must be positive")
    if args.hard_best_weight < 0.0:
        parser.error("--hard-best-weight must be non-negative")
    if args.min_score_gap < 0.0:
        parser.error("--min-score-gap must be non-negative")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    dataset, stats = load_dataset(
        paths,
        args.require_search_used,
        args.require_action_changed,
        args.min_score_gap,
        args.positive_field,
        args.max_steps,
        args.local_channels,
        args.max_rows,
        args.seed,
    )
    val_stats = None
    if args.val_dataset:
        val_paths = expand_dataset_paths(args.val_dataset)
        val_dataset, val_stats = load_dataset(
            val_paths,
            args.require_search_used,
            args.require_action_changed,
            args.min_score_gap,
            args.positive_field,
            args.max_steps,
            args.local_channels,
            args.max_val_rows,
            args.seed + 1,
        )
        if val_dataset["features"].shape[1:] != dataset["features"].shape[1:]:
            raise ValueError(
                "--val-dataset must have the same top-k and feature dimensions as --dataset "
                f"({val_dataset['features'].shape[1:]} != {dataset['features'].shape[1:]})"
            )
        train_data = {name: value for name, value in dataset.items() if name not in {"feature_mean", "feature_std"}}
        val_data = {name: value for name, value in val_dataset.items() if name not in {"feature_mean", "feature_std"}}
        train_idx = np.arange(dataset["features"].shape[0])
        val_idx = np.arange(val_dataset["features"].shape[0])
    else:
        train_data, val_data, train_idx, val_idx = split_dataset(dataset, args.val_fraction, args.seed)
    key = jrandom.PRNGKey(args.seed)
    key, model_key = jrandom.split(key)
    model = OnlineSearchCandidateScorer(
        model_key,
        input_dim=int(dataset["features"].shape[-1]),
        hidden_dim=args.hidden_dim,
        feature_mean=dataset["feature_mean"],
        feature_std=dataset["feature_std"],
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    print("Adaptive online-search candidate scorer")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Rows:          raw={stats['raw_rows']} kept={stats['rows']} train={len(train_idx)} val={len(val_idx)}")
    if val_stats is not None:
        print(f"Val rows:      raw={val_stats['raw_rows']} kept={val_stats['rows']} independent_shards=true")
    print(f"Top-k/features:{stats['top_k']} / {stats['feature_dim']}")
    print(f"Prior base:    top1={stats['prior_top1']*100:.2f}% top2={stats['prior_top2']*100:.2f}%")
    if args.positive_field is not None:
        print(f"Filter:        {args.positive_field}")
    print()

    best_val_top1 = -1.0
    best_model = model
    best_epoch = 0
    best_metrics = None
    for epoch in range(1, args.num_epochs + 1):
        key, epoch_key = jrandom.split(key)
        model, opt_state, train_loss, train_metrics = train_epoch(
            model,
            opt_state,
            train_data,
            optimizer,
            epoch_key,
            args.minibatch_size,
            args.temperature,
            args.gap_weighting,
            args.hard_best_weight,
        )
        val_loss, val_metrics = evaluate_loss(model, as_batch(val_data), args.temperature, args.hard_best_weight)
        val_top1 = float(val_metrics["top1"])
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_model = model
            best_epoch = epoch
            best_metrics = val_metrics
        if epoch == 1 or epoch == args.num_epochs or epoch % max(args.num_epochs // 10, 1) == 0:
            print(
                f"epoch {epoch:03d} | "
                f"train_loss={float(train_loss):.4f} top1={float(train_metrics['top1'])*100:.1f}% "
                f"pair={float(train_metrics['pair'])*100:.1f}% | "
                f"val_loss={float(val_loss):.4f} top1={float(val_metrics['top1'])*100:.1f}% "
                f"top2={float(val_metrics['top2'])*100:.1f}% pair={float(val_metrics['pair'])*100:.1f}% "
                f"prior={float(val_metrics['prior_top1'])*100:.1f}%"
            )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(model_path, model)
    best_path = model_path.with_name(model_path.stem + ".best" + model_path.suffix)
    eqx.tree_serialise_leaves(best_path, best_model)
    final_loss, final_metrics = evaluate_loss(model, as_batch(val_data), args.temperature, args.hard_best_weight)
    actual_local_channels = max(0, (int(dataset["features"].shape[-1]) - len(BASE_FEATURE_NAMES)) // 3)
    sidecar = {
        "feature_names": list(BASE_FEATURE_NAMES)
        + [f"source_ch{idx}" for idx in range(actual_local_channels)]
        + [f"dest_ch{idx}" for idx in range(actual_local_channels)]
        + [f"source_minus_dest_ch{idx}" for idx in range(actual_local_channels)],
        "feature_mean": np.asarray(model.feature_mean).tolist(),
        "feature_std": np.asarray(model.feature_std).tolist(),
        "hidden_dim": args.hidden_dim,
        "temperature": args.temperature,
        "hard_best_weight": args.hard_best_weight,
        "gap_weighting": args.gap_weighting,
        "positive_field": args.positive_field,
        "filters": {
            "require_search_used": args.require_search_used,
            "require_action_changed": args.require_action_changed,
            "min_score_gap": args.min_score_gap,
            "local_channels": args.local_channels,
        },
        "dataset": stats,
        "validation_dataset": val_stats,
        "best_epoch": best_epoch,
        "best_val_metrics": {name: float(value) for name, value in best_metrics.items()},
        "final_val_loss": float(final_loss),
        "final_val_metrics": {name: float(value) for name, value in final_metrics.items()},
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best epoch: {best_epoch} val_top1={best_val_top1*100:.2f}%")
    print(f"Saved: {model_path}")
    print(f"Best:  {best_path}")


if __name__ == "__main__":
    main()
