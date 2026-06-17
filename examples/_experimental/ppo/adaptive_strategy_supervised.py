"""Offline supervised training for adaptive strategy auxiliary heads."""

from __future__ import annotations

import argparse
import glob
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


def load_strategy_dataset(paths: list[Path], max_samples: int | None = None) -> dict[str, jnp.ndarray]:
    """Load the subset of NPZ fields needed by the frozen-head trainer."""
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
    }
    for path in paths:
        shard = np.load(path)
        chunks["obs"].append(shard["obs"].astype(np.float32))
        chunks["legal_mask"].append(shard["legal_mask"].astype(np.bool_))
        chunks["active"].append(shard["active"].astype(np.bool_))
        chunks["intent"].append(shard["intent"].astype(np.int32))
        chunks["finish"].append((shard["finish_within_250"] > 0.5).astype(np.int32))
        chunks["finish_weight"].append(shard["outcome_known"].astype(np.float32))
        chunks["outcome"].append(shard["outcome"].astype(np.int32))
        chunks["outcome_weight"].append(shard["outcome_known"].astype(np.float32))
        chunks["enemy_general"].append(shard["enemy_general_heatmap"].astype(np.float32))

    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_samples is not None:
        arrays = {name: value[:max_samples] for name, value in arrays.items()}
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def mask_strategy_supervised_grads(grads, keep_outcome: bool):
    """Keep gradients only for strategy auxiliary heads and optional outcome head."""
    masked = jax.tree.map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, grads)
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
    return masked


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
):
    """Train one minibatch of frozen-trunk strategy auxiliary losses."""
    obs, masks, active, intent_targets, finish_targets, finish_weights, outcome_targets, outcome_weights, enemy_general = batch

    def loss_fn(net):
        outputs = jax.vmap(lambda o, m, a: net.strategy_auxiliary(o, m, a))(obs, masks, active)

        intent_log_probs = jax.nn.log_softmax(outputs.intent_logits, axis=-1)
        intent_losses = -intent_log_probs[jnp.arange(intent_log_probs.shape[0]), intent_targets]
        intent_loss = jnp.mean(intent_losses)
        intent_accuracy = jnp.mean((jnp.argmax(outputs.intent_logits, axis=-1) == intent_targets).astype(jnp.float32))

        finish_log_probs = jax.nn.log_softmax(outputs.finish_logits, axis=-1)
        finish_losses = -finish_log_probs[jnp.arange(finish_log_probs.shape[0]), finish_targets]
        finish_normalizer = jnp.maximum(jnp.sum(finish_weights), 1.0)
        finish_loss = jnp.sum(finish_losses * finish_weights) / finish_normalizer
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
            outcome_loss = jnp.sum(outcome_losses * outcome_weights) / outcome_normalizer
            outcome_accuracy = jnp.sum(
                (jnp.argmax(outcome_logits, axis=-1) == outcome_targets).astype(jnp.float32) * outcome_weights
            )
            outcome_accuracy = outcome_accuracy / outcome_normalizer

        loss = (
            intent_weight * intent_loss
            + finish_weight * finish_loss
            + belief_weight * belief_loss
            + outcome_weight * outcome_loss
        )
        metrics = {
            "intent_loss": intent_loss,
            "finish_loss": finish_loss,
            "belief_loss": belief_loss,
            "outcome_loss": outcome_loss,
            "intent_accuracy": intent_accuracy,
            "finish_accuracy": finish_accuracy,
            "outcome_accuracy": outcome_accuracy,
            "finish_weight_mean": jnp.mean(finish_weights),
            "outcome_weight_mean": jnp.mean(outcome_weights),
        }
        return loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    grads = mask_strategy_supervised_grads(grads, outcome_weight > 0.0)
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
        )
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return network, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train adaptive strategy auxiliary heads from NPZ shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--max-samples", type=int, default=None)
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
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--model-path", default="runs/adaptive-strategy-supervised/generals-adaptive-strategy-supervised.eqx")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--intent-weight", type=float, default=0.2)
    parser.add_argument("--finish-weight", type=float, default=0.4)
    parser.add_argument("--belief-weight", type=float, default=0.3)
    parser.add_argument("--outcome-weight", type=float, default=0.0)
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
    if args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.num_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("--num-epochs and --minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.weight_decay != 0.0:
        parser.error("--weight-decay must stay 0 because this trainer freezes most parameters with a gradient mask")
    if args.value_loss == "hl-gauss" and args.value_bins <= 1:
        parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
    if args.init_value_loss == "hl-gauss":
        init_bins = args.value_bins if args.init_value_bins is None else args.init_value_bins
        if init_bins <= 1:
            parser.error("--init-value-bins must be greater than 1 for --init-value-loss hl-gauss")
    elif args.init_value_bins is not None:
        parser.error("--init-value-bins requires --init-value-loss hl-gauss")
    if any(weight < 0.0 for weight in (args.intent_weight, args.finish_weight, args.belief_weight, args.outcome_weight)):
        parser.error("loss weights must be non-negative")
    if args.outcome_weight > 0.0 and not args.outcome_head:
        parser.error("--outcome-weight requires --outcome-head")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    dataset = load_strategy_dataset(paths, args.max_samples)
    key = jrandom.PRNGKey(args.seed)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )

    print("Adaptive strategy supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Samples:       {dataset['obs'].shape[0]}")
    print(f"Network arch:  {args.network_arch}")
    print(f"Warm start:    {args.init_model_path}")
    print(
        "Loss weights:  "
        f"intent={args.intent_weight:g}, finish={args.finish_weight:g}, "
        f"belief={args.belief_weight:g}, outcome={args.outcome_weight:g}"
    )
    print("Update scope:  strategy auxiliary heads" + (" + outcome head" if args.outcome_weight > 0.0 else ""))
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
        )
        jax.block_until_ready(network)
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Intent {float(metrics['intent_loss']):.4f}/{float(metrics['intent_accuracy']) * 100:5.1f}% | "
            f"Finish {float(metrics['finish_loss']):.4f}/{float(metrics['finish_accuracy']) * 100:5.1f}% | "
            f"Belief {float(metrics['belief_loss']):.4f} | "
            f"Outcome {float(metrics['outcome_loss']):.4f}/{float(metrics['outcome_accuracy']) * 100:5.1f}% | "
            f"Time {time.time() - t0:.2f}s"
        )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
