"""Train a small gate that predicts when to accept an online-search action."""

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

from adaptive_command_gate import CommandGateNetwork

SEARCH_GATE_FEATURE_NAMES = (
    "best_score",
    "second_score",
    "score_gap",
    "mean_score",
    "std_score",
    "best_prior",
    "second_prior",
    "prior_gap",
    "best_prior_rank",
    "fallback_prior",
    "best_minus_fallback_prior",
    "search_action_changed",
    "time_norm",
    "seat",
    "active_fraction",
    "visible_enemy_density",
    "contact",
)
POSITIVE_FIELDS = (
    "search_converts_to_win",
    "search_converts_draw_to_win",
    "search_improves_continuation",
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


def _safe_take_logits(logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return logits at flat action indices, clipping invalid indices defensively."""
    clipped = np.clip(indices.astype(np.int64), 0, logits.shape[1] - 1)
    return logits[np.arange(logits.shape[0]), clipped].astype(np.float32)


def build_features_from_shard(shard: np.lib.npyio.NpzFile, max_steps: int) -> np.ndarray:
    """Build inference-available online-search gate features from one trace shard."""
    scores = shard["search_scores"].astype(np.float32)
    prior = shard["search_prior_scores"].astype(np.float32)
    best_pos = shard["search_best_position"].astype(np.int64)
    rows = scores.shape[0]
    top_k = scores.shape[1]

    best_score = scores[np.arange(rows), best_pos]
    sorted_scores = np.sort(scores, axis=1)
    second_score = sorted_scores[:, -2] if top_k >= 2 else best_score
    mean_score = scores.mean(axis=1)
    std_score = scores.std(axis=1)

    best_prior = prior[np.arange(rows), best_pos]
    sorted_prior = np.sort(prior, axis=1)
    second_prior = sorted_prior[:, -2] if top_k >= 2 else best_prior
    prior_gap = sorted_prior[:, -1] - second_prior

    fallback_prior = _safe_take_logits(shard["teacher_logits"].astype(np.float32), shard["base_action_index"])
    active_fraction = shard["active"].astype(np.float32).reshape(rows, -1).mean(axis=1)
    denom = max(top_k - 1, 1)
    features = np.stack(
        [
            best_score,
            second_score,
            shard["search_score_gap"].astype(np.float32),
            mean_score,
            std_score,
            best_prior,
            second_prior,
            prior_gap,
            best_pos.astype(np.float32) / float(denom),
            fallback_prior,
            best_prior - fallback_prior,
            shard["search_action_changed"].astype(np.float32),
            shard["time"].astype(np.float32) / float(max(max_steps, 1)),
            shard["seat"].astype(np.float32),
            active_fraction,
            shard["visible_enemy_density"].astype(np.float32),
            shard["contact"].astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=1.0e4, neginf=-1.0e4)


def balanced_binary_weights(labels: np.ndarray) -> np.ndarray:
    """Weight positive and negative classes equally."""
    positive = labels > 0.5
    pos_count = max(int(np.sum(positive)), 1)
    neg_count = max(int(np.sum(~positive)), 1)
    return (np.where(positive, 0.5 / pos_count, 0.5 / neg_count).astype(np.float32) * labels.shape[0])


def load_dataset(
    paths: list[Path],
    positive_field: str,
    require_search_used: bool,
    require_action_changed: bool,
    min_score_gap: float,
    max_steps: int,
    max_examples: int | None,
    seed: int,
) -> tuple[dict[str, jnp.ndarray], dict[str, object]]:
    """Load online-search NPZ shards into normalized MLP features."""
    rng = np.random.default_rng(seed)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    stats = {"rows": 0, "kept": 0, "positive": 0, "changed": 0, "paths": [str(path) for path in paths]}
    for path in paths:
        with np.load(path) as shard:
            features = build_features_from_shard(shard, max_steps)
            labels = shard[positive_field].astype(np.float32)
            keep = np.ones((features.shape[0],), dtype=np.bool_)
            if require_search_used:
                keep &= shard["search_used"].astype(np.bool_)
            if require_action_changed:
                keep &= shard["search_action_changed"].astype(np.bool_)
            if min_score_gap > 0.0:
                keep &= shard["search_score_gap"].astype(np.float32) >= min_score_gap
            stats["rows"] += int(features.shape[0])
            stats["changed"] += int(np.sum(shard["search_action_changed"].astype(np.bool_)))
            if not np.any(keep):
                continue
            feature_chunks.append(features[keep])
            label_chunks.append(labels[keep])
            stats["kept"] += int(np.sum(keep))
            stats["positive"] += int(np.sum(labels[keep] > 0.5))
    if not feature_chunks:
        raise ValueError("No online-search gate examples selected")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    if max_examples is not None and features.shape[0] > max_examples:
        indices = np.sort(rng.choice(features.shape[0], size=max_examples, replace=False))
        features = features[indices]
        labels = labels[indices]
    weights = balanced_binary_weights(labels)
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(features.std(axis=0).astype(np.float32), 1.0e-6)
    stats["examples"] = int(features.shape[0])
    stats["positive_examples"] = int(np.sum(labels > 0.5))
    stats["positive_rate"] = float(np.mean(labels > 0.5))
    return (
        {
            "features": jnp.asarray(features),
            "labels": jnp.asarray(labels),
            "weights": jnp.asarray(weights),
            "feature_mean": jnp.asarray(feature_mean),
            "feature_std": jnp.asarray(feature_std),
        },
        stats,
    )


@eqx.filter_jit
def train_step(gate, opt_state, batch, optimizer):
    features, labels, weights = batch

    def loss_fn(model):
        logits = jax.vmap(model)(features)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels) * weights
        probabilities = jax.nn.sigmoid(logits)
        predictions = probabilities >= 0.5
        positives = labels > 0.5
        true_positive = jnp.sum(predictions & positives)
        predicted_positive = jnp.maximum(jnp.sum(predictions), 1)
        actual_positive = jnp.maximum(jnp.sum(positives), 1)
        positive_count = jnp.maximum(jnp.sum(positives), 1)
        negative_count = jnp.maximum(jnp.sum(~positives), 1)
        metrics = {
            "acc": jnp.mean((predictions == positives).astype(jnp.float32)),
            "precision": true_positive / predicted_positive,
            "recall": true_positive / actual_positive,
            "p_pos": jnp.sum(jnp.where(positives, probabilities, 0.0)) / positive_count,
            "p_neg": jnp.sum(jnp.where(~positives, probabilities, 0.0)) / negative_count,
        }
        return jnp.mean(loss), metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(gate)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(gate, eqx.is_inexact_array))
    return eqx.apply_updates(gate, updates), opt_state, loss, metrics


def train_epoch(gate, opt_state, dataset, optimizer, key, minibatch_size: int):
    examples = dataset["features"].shape[0]
    permutation = jrandom.permutation(key, examples)
    loss_sum = 0.0
    metric_sum = None
    batches = 0
    for start in range(0, examples, minibatch_size):
        indices = permutation[start : min(start + minibatch_size, examples)]
        batch = (dataset["features"][indices], dataset["labels"][indices], dataset["weights"][indices])
        gate, opt_state, loss, metrics = train_step(gate, opt_state, batch, optimizer)
        loss_sum += loss
        metric_sum = metrics if metric_sum is None else jax.tree.map(lambda a, b: a + b, metric_sum, metrics)
        batches += 1
    return gate, opt_state, loss_sum / batches, jax.tree.map(lambda value: value / batches, metric_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an online-search accept/reject gate from trace shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--positive-field", choices=POSITIVE_FIELDS, default="search_converts_to_win")
    parser.add_argument("--require-search-used", action="store_true")
    parser.add_argument("--require-action-changed", action="store_true")
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-path", default="runs/adaptive-online-search-gate/generals-adaptive-online-search-gate.eqx")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.hidden_dim <= 0 or args.num_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("--hidden-dim, --num-epochs, and --minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.min_score_gap < 0.0:
        parser.error("--min-score-gap must be non-negative")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    dataset, stats = load_dataset(
        paths,
        args.positive_field,
        args.require_search_used,
        args.require_action_changed,
        args.min_score_gap,
        args.max_steps,
        args.max_examples,
        args.seed,
    )
    key = jrandom.PRNGKey(args.seed)
    key, gate_key = jrandom.split(key)
    gate = CommandGateNetwork(
        gate_key,
        input_dim=len(SEARCH_GATE_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        feature_mean=dataset["feature_mean"],
        feature_std=dataset["feature_std"],
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(gate, eqx.is_inexact_array))
    print("Adaptive online-search gate supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Rows:          {stats['rows']} kept={stats['kept']}")
    print(f"Examples:      {stats['examples']} positive={stats['positive_examples']} ({stats['positive_rate']*100:.2f}%)")
    print(f"Positive:      {args.positive_field}")
    print(f"Changed only:  {args.require_action_changed}")
    print()
    last_loss = None
    last_metrics = None
    for epoch in range(1, args.num_epochs + 1):
        key, epoch_key = jrandom.split(key)
        gate, opt_state, loss, metrics = train_epoch(gate, opt_state, dataset, optimizer, epoch_key, args.minibatch_size)
        last_loss = loss
        last_metrics = metrics
        if epoch == 1 or epoch == args.num_epochs or epoch % max(args.num_epochs // 10, 1) == 0:
            print(
                f"epoch {epoch:03d} | loss={float(loss):.4f} "
                f"acc={float(metrics['acc'])*100:.1f}% "
                f"prec={float(metrics['precision'])*100:.1f}% "
                f"recall={float(metrics['recall'])*100:.1f}% "
                f"p+={float(metrics['p_pos']):.3f} p-={float(metrics['p_neg']):.3f}"
            )
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(model_path, gate)
    sidecar = {
        "feature_names": list(SEARCH_GATE_FEATURE_NAMES),
        "feature_mean": np.asarray(gate.feature_mean).tolist(),
        "feature_std": np.asarray(gate.feature_std).tolist(),
        "hidden_dim": args.hidden_dim,
        "positive_field": args.positive_field,
        "require_search_used": args.require_search_used,
        "require_action_changed": args.require_action_changed,
        "min_score_gap": args.min_score_gap,
        "dataset": stats,
        "final_loss": float(last_loss),
        "final_metrics": {name: float(value) for name, value in last_metrics.items()},
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
