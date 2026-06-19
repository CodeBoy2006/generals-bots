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

POLICY_ADAPTER_GATE_FEATURE_NAMES = (
    "adapter_delta_at_adapter_top",
    "adapter_delta_at_policy_top",
    "policy_support_for_adapter_top",
    "adapter_top_margin",
    "policy_top_margin",
    "adapter_finish_probability",
    "visible_enemy_density",
    "visible_enemy_army_log_density",
    "owned_army_log_density",
    "active_fraction",
    "adapter_changes_action",
    "seat",
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
        adapter_aux = jax.vmap(lambda o, m, a: adapter_network.strategy_auxiliary(o, m, a))(
            obs_batch,
            legal_batch,
            active_batch,
        )
        features = jax.vmap(policy_adapter_gate_features, in_axes=(0, 0, 0, 0, 0, 0, None))(
            obs_batch,
            policy_logits,
            adapter_logits,
            adapter_aux.finish_logits,
            active_batch,
            seat_batch,
            pad_size,
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


def build_gate_examples(
    paths: list[Path],
    base_network,
    adapter_network,
    feature_batch_size: int,
    positive_path_contains: tuple[str, ...],
    require_search_best_win: bool,
    include_finish250_positives: bool,
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
        else:
            decisive = np.ones((obs.shape[0],), dtype=np.bool_)
        labels = (changed & teacher_match & decisive & positive_domain).astype(np.float32)
        keep = np.ones_like(changed, dtype=np.bool_) if keep_unchanged_negatives else changed
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
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-model-path", required=True)
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
    parser.add_argument("--strategy-finish-outputs", type=int, default=3)
    parser.add_argument("--positive-path-contains", action="append", default=[])
    parser.add_argument("--allow-nondecisive-positives", action="store_true")
    parser.add_argument("--include-finish250-positives", action="store_true")
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
    if not args.strategy_aux:
        parser.error("policy-adapter gate features require --strategy-aux")
    if args.hidden_dim <= 0 or args.num_epochs <= 0 or args.minibatch_size <= 0 or args.feature_batch_size <= 0:
        parser.error("hidden dim, epochs, minibatch, and feature batch must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    key = jrandom.PRNGKey(args.seed)
    key, base_key, adapter_key, gate_key = jrandom.split(key, 4)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
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
    )
    dataset = build_gate_examples(
        paths,
        base_network,
        adapter_network,
        args.feature_batch_size,
        tuple(args.positive_path_contains),
        not args.allow_nondecisive_positives,
        args.include_finish250_positives,
        args.keep_unchanged_negatives,
        args.max_examples,
        args.seed,
    )
    labels = np.asarray(dataset["labels"])
    stats = dataset["stats"]
    print("Adaptive policy-adapter gate supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Examples:      {dataset['features'].shape[0]}")
    print(f"Positive:      {float(np.mean(labels)) * 100:.2f}%")
    print(f"Rows/changed:  {stats['rows']} / {stats['changed']}")
    print(f"Teacher match: {stats['teacher_match']}")
    print(f"Decisive:      {stats['decisive']}")
    print(f"Base:          {args.base_model_path}")
    print(f"Adapter:       {args.adapter_model_path}")
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
        "feature_names": list(POLICY_ADAPTER_GATE_FEATURE_NAMES),
        "feature_mean": np.asarray(gate.feature_mean).tolist(),
        "feature_std": np.asarray(gate.feature_std).tolist(),
        "hidden_dim": args.hidden_dim,
        "examples": int(dataset["features"].shape[0]),
        "positive_fraction": float(np.mean(labels)),
        "stats": stats,
        "positive_path_contains": list(args.positive_path_contains),
        "require_search_best_win": not args.allow_nondecisive_positives,
        "include_finish250_positives": args.include_finish250_positives,
        "keep_unchanged_negatives": args.keep_unchanged_negatives,
        "base_model_path": args.base_model_path,
        "adapter_model_path": args.adapter_model_path,
        "datasets": [str(path) for path in paths],
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
