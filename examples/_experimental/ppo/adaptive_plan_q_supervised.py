"""Train adaptive strategy heads from Plan-Q source-target shards."""

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


def load_plan_q_dataset(
    paths: list[Path],
    max_samples: int | None = None,
    max_samples_per_shard: int | None = None,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Load only the fields needed for source/target Q-map supervision."""
    rng = np.random.default_rng(seed)
    chunks: dict[str, list[np.ndarray]] = {
        "obs": [],
        "legal_mask": [],
        "active": [],
        "source_indices": [],
        "target_indices": [],
        "source_probs": [],
        "target_probs": [],
        "teacher_logits": [],
        "teacher_action": [],
        "plan_action_indices": [],
        "plan_q": [],
        "plan_q_gap": [],
    }
    for path in paths:
        shard = np.load(path)
        shard_samples = shard["obs"].shape[0]
        indices = np.arange(shard_samples)
        if max_samples_per_shard is not None and shard_samples > max_samples_per_shard:
            indices = np.sort(rng.choice(shard_samples, size=max_samples_per_shard, replace=False))
        chunks["obs"].append(shard["obs"][indices].astype(np.float32))
        chunks["legal_mask"].append(shard["legal_mask"][indices].astype(np.bool_))
        chunks["active"].append(shard["active"][indices].astype(np.bool_))
        chunks["source_indices"].append(shard["source_indices"][indices].astype(np.int32))
        chunks["target_indices"].append(shard["target_indices"][indices].astype(np.int32))
        chunks["source_probs"].append(shard["source_score_probs"][indices].astype(np.float32))
        chunks["target_probs"].append(shard["target_score_probs"][indices].astype(np.float32))
        chunks["teacher_logits"].append(shard["teacher_logits"][indices].astype(np.float32))
        chunks["teacher_action"].append(shard["teacher_action_index"][indices].astype(np.int32))
        chunks["plan_action_indices"].append(shard["plan_action_indices"][indices].astype(np.int32))
        chunks["plan_q"].append(shard["plan_q"][indices].astype(np.float32))
        chunks["plan_q_gap"].append(shard["plan_q_gap"][indices].astype(np.float32))

    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_samples is not None:
        arrays = {name: value[:max_samples] for name, value in arrays.items()}
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def mask_plan_q_grads(grads, keep_outcome: bool = False):
    """Keep gradients only for Plan-Q strategy heads and optional outcome head."""
    masked = jax.tree.map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, grads)
    if keep_outcome and grads.outcome_linear2 is not None:
        masked = eqx.tree_at(lambda net: net.outcome_linear2, masked, grads.outcome_linear2)
    if grads.strategy_q_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_q_conv, masked, grads.strategy_q_conv)
    if grads.strategy_q_pass_linear is not None:
        masked = eqx.tree_at(lambda net: net.strategy_q_pass_linear, masked, grads.strategy_q_pass_linear)
    if grads.strategy_source_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_source_conv, masked, grads.strategy_source_conv)
    if grads.strategy_target_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_target_conv, masked, grads.strategy_target_conv)
    return masked


def indexed_spatial_ce(
    logits: jnp.ndarray,
    active: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    candidate_probs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cross-entropy against a sparse candidate distribution over spatial cells."""
    masked_logits = jnp.where(active, logits, -1.0e9).reshape(logits.shape[0], -1)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    candidate_log_probs = jnp.take_along_axis(log_probs, candidate_indices, axis=1)
    candidate_probs = candidate_probs / jnp.maximum(jnp.sum(candidate_probs, axis=1, keepdims=True), 1.0e-6)
    losses = -jnp.sum(candidate_probs * candidate_log_probs, axis=1)
    target_pos = jnp.argmax(candidate_probs, axis=1)
    target_indices = jnp.take_along_axis(candidate_indices, target_pos[:, None], axis=1)[:, 0]
    predictions = jnp.argmax(masked_logits, axis=1)
    global_accuracy = jnp.mean((predictions == target_indices).astype(jnp.float32))
    candidate_accuracy = jnp.mean((jnp.argmax(candidate_log_probs, axis=1) == target_pos).astype(jnp.float32))
    entropy = -jnp.mean(jnp.sum(candidate_probs * jnp.log(jnp.clip(candidate_probs, 1.0e-8, 1.0)), axis=1))
    return jnp.mean(losses), global_accuracy, candidate_accuracy, entropy


@eqx.filter_jit
def train_step(
    network,
    opt_state,
    batch,
    optimizer,
    source_weight: float,
    target_weight: float,
    policy_kl_weight: float,
    action_ce_weight: float,
    action_q_weight: float,
    action_q_mse_weight: float,
    action_q_temperature: float,
    gap_weighting: bool,
    freeze_base: bool,
):
    """Train one Plan-Q supervised minibatch."""
    (
        obs,
        masks,
        active,
        source_indices,
        target_indices,
        source_probs,
        target_probs,
        teacher_logits,
        teacher_actions,
        plan_action_indices,
        plan_q,
        plan_q_gap,
    ) = batch

    def loss_fn(net):
        outputs = jax.vmap(lambda o, m, a: net.strategy_auxiliary(o, m, a))(obs, masks, active)
        sample_weight = jnp.ones_like(plan_q_gap)
        if gap_weighting:
            sample_weight = jnp.clip(plan_q_gap / jnp.maximum(jnp.mean(plan_q_gap), 1.0e-6), 0.25, 4.0)
            sample_weight = jax.lax.stop_gradient(sample_weight)

        source_loss = jnp.asarray(0.0, dtype=jnp.float32)
        source_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        source_candidate_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        source_entropy = jnp.asarray(0.0, dtype=jnp.float32)
        target_loss = jnp.asarray(0.0, dtype=jnp.float32)
        target_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        target_candidate_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        target_entropy = jnp.asarray(0.0, dtype=jnp.float32)
        if source_weight > 0.0 or target_weight > 0.0:
            if outputs.source_logits is None or outputs.target_logits is None:
                raise ValueError("source/target Plan-Q supervision requires strategy_spatial_aux")
            source_loss, source_accuracy, source_candidate_accuracy, source_entropy = indexed_spatial_ce(
                outputs.source_logits,
                active,
                source_indices,
                source_probs,
            )
            target_loss, target_accuracy, target_candidate_accuracy, target_entropy = indexed_spatial_ce(
                outputs.target_logits,
                active,
                target_indices,
                target_probs,
            )
            if gap_weighting:
                # Recompute weighted losses directly when gap weighting is enabled.
                source_loss = weighted_indexed_spatial_ce(
                    outputs.source_logits,
                    active,
                    source_indices,
                    source_probs,
                    sample_weight,
                )
                target_loss = weighted_indexed_spatial_ce(
                    outputs.target_logits,
                    active,
                    target_indices,
                    target_probs,
                    sample_weight,
                )

        policy_kl = jnp.asarray(0.0, dtype=jnp.float32)
        action_ce = jnp.asarray(0.0, dtype=jnp.float32)
        teacher_action_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if policy_kl_weight > 0.0 or action_ce_weight > 0.0:
            teacher_legal = teacher_logits > -9999.0
            student_logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs, masks, active)
            masked_teacher_logits = jnp.where(teacher_legal, teacher_logits, -1.0e9)
            teacher_log_probs = jax.nn.log_softmax(masked_teacher_logits, axis=-1)
            teacher_probs = jnp.exp(teacher_log_probs)
            student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
            policy_kl = jnp.mean(jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1))
            action_ce = jnp.mean(-student_log_probs[jnp.arange(student_log_probs.shape[0]), teacher_actions])
            teacher_action_accuracy = jnp.mean(
                (jnp.argmax(student_logits, axis=-1) == teacher_actions).astype(jnp.float32)
            )

        action_q_rank_loss = jnp.asarray(0.0, dtype=jnp.float32)
        action_q_mse_loss = jnp.asarray(0.0, dtype=jnp.float32)
        action_q_candidate_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        action_q_target_entropy = jnp.asarray(0.0, dtype=jnp.float32)
        action_q_pred_gap = jnp.asarray(0.0, dtype=jnp.float32)
        if action_q_weight > 0.0 or action_q_mse_weight > 0.0:
            action_q_rank_loss, action_q_mse_loss, action_q_candidate_accuracy, action_q_target_entropy, action_q_pred_gap = (
                plan_action_q_losses(
                    outputs.action_q_values,
                    masks,
                    plan_action_indices,
                    plan_q,
                    action_q_temperature,
                    sample_weight,
                )
            )

        loss = (
            source_weight * source_loss
            + target_weight * target_loss
            + policy_kl_weight * policy_kl
            + action_ce_weight * action_ce
            + action_q_weight * action_q_rank_loss
            + action_q_mse_weight * action_q_mse_loss
        )
        metrics = {
            "source_loss": source_loss,
            "target_loss": target_loss,
            "source_accuracy": source_accuracy,
            "target_accuracy": target_accuracy,
            "source_candidate_accuracy": source_candidate_accuracy,
            "target_candidate_accuracy": target_candidate_accuracy,
            "source_entropy": source_entropy,
            "target_entropy": target_entropy,
            "policy_kl": policy_kl,
            "action_ce": action_ce,
            "teacher_action_accuracy": teacher_action_accuracy,
            "action_q_rank_loss": action_q_rank_loss,
            "action_q_mse_loss": action_q_mse_loss,
            "action_q_candidate_accuracy": action_q_candidate_accuracy,
            "action_q_target_entropy": action_q_target_entropy,
            "action_q_pred_gap": action_q_pred_gap,
            "mean_gap": jnp.mean(plan_q_gap),
            "mean_sample_weight": jnp.mean(sample_weight),
        }
        return loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    if freeze_base:
        grads = mask_plan_q_grads(grads)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss, metrics


def weighted_indexed_spatial_ce(
    logits: jnp.ndarray,
    active: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    candidate_probs: jnp.ndarray,
    sample_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted sparse spatial CE for emphasizing high-gap plan rows."""
    masked_logits = jnp.where(active, logits, -1.0e9).reshape(logits.shape[0], -1)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    candidate_log_probs = jnp.take_along_axis(log_probs, candidate_indices, axis=1)
    candidate_probs = candidate_probs / jnp.maximum(jnp.sum(candidate_probs, axis=1, keepdims=True), 1.0e-6)
    losses = -jnp.sum(candidate_probs * candidate_log_probs, axis=1)
    normalizer = jnp.maximum(jnp.sum(sample_weight), 1.0)
    return jnp.sum(losses * sample_weight) / normalizer


def plan_action_q_losses(
    action_q_values: jnp.ndarray,
    legal_mask: jnp.ndarray,
    plan_action_indices: jnp.ndarray,
    plan_q: jnp.ndarray,
    temperature: float,
    sample_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit action-Q values to plan-level counterfactual ranking targets."""
    flat_action_indices = plan_action_indices.reshape(plan_action_indices.shape[0], -1)
    flat_plan_q = jax.lax.stop_gradient(plan_q.reshape(plan_q.shape[0], -1))
    batch_indices = jnp.broadcast_to(jnp.arange(action_q_values.shape[0])[:, None], flat_action_indices.shape)
    candidate_q = jnp.take_along_axis(action_q_values, flat_action_indices, axis=1)
    target_plan_probs = jax.nn.softmax(flat_plan_q / temperature, axis=1)
    target_action_probs = jnp.zeros_like(action_q_values).at[batch_indices, flat_action_indices].add(target_plan_probs)

    move_mask = jnp.transpose(legal_mask, (0, 3, 1, 2)).reshape(legal_mask.shape[0], -1)
    full_legal = jnp.concatenate(
        [move_mask, move_mask, jnp.ones((move_mask.shape[0], 1), dtype=bool)],
        axis=1,
    )
    masked_action_q = jnp.where(full_legal, action_q_values / temperature, -1.0e9)
    action_log_probs = jax.nn.log_softmax(masked_action_q, axis=1)
    rank_losses = -jnp.sum(target_action_probs * action_log_probs, axis=1)
    mse_losses = jnp.mean((candidate_q - flat_plan_q) ** 2, axis=1)
    normalizer = jnp.maximum(jnp.sum(sample_weight), 1.0)
    rank_loss = jnp.sum(rank_losses * sample_weight) / normalizer
    mse_loss = jnp.sum(mse_losses * sample_weight) / normalizer
    target_action = jnp.argmax(target_action_probs, axis=1)
    pred_action = jnp.argmax(masked_action_q, axis=1)
    action_accuracy = jnp.sum((pred_action == target_action).astype(jnp.float32) * sample_weight) / normalizer
    target_entropy = -jnp.mean(
        jnp.sum(target_action_probs * jnp.log(jnp.clip(target_action_probs, 1.0e-8, 1.0)), axis=1)
    )
    legal_count = jnp.maximum(jnp.sum(full_legal, axis=1), 1)
    legal_mean = jnp.sum(jnp.where(full_legal, action_q_values, 0.0), axis=1) / legal_count
    pred_gap = jnp.mean(jnp.max(jnp.where(full_legal, action_q_values, -1.0e9), axis=1) - legal_mean)
    return rank_loss, mse_loss, action_accuracy, target_entropy, pred_gap


def train_epoch(
    network,
    opt_state,
    dataset,
    optimizer,
    key,
    minibatch_size: int,
    source_weight: float,
    target_weight: float,
    policy_kl_weight: float,
    action_ce_weight: float,
    action_q_weight: float,
    action_q_mse_weight: float,
    action_q_temperature: float,
    gap_weighting: bool,
    freeze_base: bool,
):
    """Shuffle one full pass over loaded Plan-Q rows."""
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
            dataset["source_indices"][idx],
            dataset["target_indices"][idx],
            dataset["source_probs"][idx],
            dataset["target_probs"][idx],
            dataset["teacher_logits"][idx],
            dataset["teacher_action"][idx],
            dataset["plan_action_indices"][idx],
            dataset["plan_q"][idx],
            dataset["plan_q_gap"][idx],
        )
        network, opt_state, loss, metrics = train_step(
            network,
            opt_state,
            batch,
            optimizer,
            source_weight,
            target_weight,
            policy_kl_weight,
            action_ce_weight,
            action_q_weight,
            action_q_mse_weight,
            action_q_temperature,
            gap_weighting,
            freeze_base,
        )
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return network, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train adaptive source/target heads from Plan-Q shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
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
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument("--init-strategy-aux", action="store_true")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--model-path", default="runs/adaptive-plan-q-supervised/generals-adaptive-plan-q-supervised.eqx")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--update-scope", choices=("strategy-heads", "all"), default="strategy-heads")
    parser.add_argument("--source-weight", type=float, default=0.5)
    parser.add_argument("--target-weight", type=float, default=0.5)
    parser.add_argument("--policy-kl-weight", type=float, default=0.0)
    parser.add_argument("--action-ce-weight", type=float, default=0.0)
    parser.add_argument("--action-q-weight", type=float, default=0.0)
    parser.add_argument("--action-q-mse-weight", type=float, default=0.0)
    parser.add_argument("--action-q-temperature", type=float, default=0.25)
    parser.add_argument("--gap-weighting", action="store_true")
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
    if args.weight_decay != 0.0 and args.update_scope == "strategy-heads":
        parser.error("--weight-decay must stay 0 when most parameters are frozen by gradient mask")
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
            args.source_weight,
            args.target_weight,
            args.policy_kl_weight,
            args.action_ce_weight,
            args.action_q_weight,
            args.action_q_mse_weight,
        )
    ):
        parser.error("loss weights must be non-negative")
    if args.action_q_temperature <= 0.0:
        parser.error("--action-q-temperature must be positive")
    if args.update_scope == "all" and args.policy_kl_weight <= 0.0:
        parser.error("--update-scope all requires a positive --policy-kl-weight to anchor policy drift")
    if not args.strategy_aux:
        parser.error("Plan-Q supervision requires --strategy-aux")
    if (args.source_weight > 0.0 or args.target_weight > 0.0) and not args.strategy_spatial_aux:
        parser.error("source/target Plan-Q supervision requires --strategy-spatial-aux")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    dataset = load_plan_q_dataset(paths, args.max_samples, args.max_samples_per_shard, args.seed)
    key = jrandom.PRNGKey(args.seed)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )

    print("Adaptive Plan-Q supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Samples:       {dataset['obs'].shape[0]}")
    print(f"Network arch:  {args.network_arch}")
    print(f"Warm start:    {args.init_model_path}")
    print(
        "Loss weights:  "
        f"source={args.source_weight:g}, target={args.target_weight:g}, "
        f"policy_kl={args.policy_kl_weight:g}, action_ce={args.action_ce_weight:g}, "
        f"action_q={args.action_q_weight:g}, action_q_mse={args.action_q_mse_weight:g}"
    )
    print(f"Action-Q temp: {args.action_q_temperature:g}")
    print(f"Update scope:  {args.update_scope}")
    print(f"Gap weighting: {args.gap_weighting}")
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
        strategy_aux=args.strategy_aux,
        init_strategy_aux=args.init_strategy_aux,
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
            args.source_weight,
            args.target_weight,
            args.policy_kl_weight,
            args.action_ce_weight,
            args.action_q_weight,
            args.action_q_mse_weight,
            args.action_q_temperature,
            args.gap_weighting,
            args.update_scope == "strategy-heads",
        )
        jax.block_until_ready(network)
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Src {float(metrics['source_loss']):.4f}/"
            f"{float(metrics['source_candidate_accuracy']) * 100:5.1f}%cand/"
            f"{float(metrics['source_accuracy']) * 100:5.1f}%grid | "
            f"Tgt {float(metrics['target_loss']):.4f}/"
            f"{float(metrics['target_candidate_accuracy']) * 100:5.1f}%cand/"
            f"{float(metrics['target_accuracy']) * 100:5.1f}%grid | "
            f"SrcH {float(metrics['source_entropy']):.3f} | "
            f"TgtH {float(metrics['target_entropy']):.3f} | "
            f"KL {float(metrics['policy_kl']):.4f} | "
            f"ActCE {float(metrics['action_ce']):.4f}/{float(metrics['teacher_action_accuracy']) * 100:5.1f}% | "
            f"AQ {float(metrics['action_q_rank_loss']):.4f}/"
            f"{float(metrics['action_q_mse_loss']):.4f}/"
            f"{float(metrics['action_q_candidate_accuracy']) * 100:5.1f}% | "
            f"AQgap {float(metrics['action_q_pred_gap']):.3f} | "
            f"Gap {float(metrics['mean_gap']):.4f} | "
            f"W {float(metrics['mean_sample_weight']):.3f} | "
            f"Time {time.time() - t0:.2f}s"
        )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
