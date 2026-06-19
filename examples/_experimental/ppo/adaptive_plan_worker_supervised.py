"""Train a target-conditioned Worker from Plan-Q or strategy source-target shards."""

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

from adaptive_common import ADAPTIVE_MOVE_PLANES, adaptive_action_space_size
from adaptive_network import load_or_create_adaptive_network
from adaptive_worker_pretrain import worker_source_direction_logits, worker_source_direction_targets
from generals.agents.ppo_policy_agent import parse_policy_channels
from train_adaptive import OUTCOME_WIN

PLAN_WORKER_EXTRA_CHANNELS = 3


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


def plan_command_planes(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    active: np.ndarray,
    pad_size: int,
) -> np.ndarray:
    """Build source, target, and target-closeness command planes."""
    num_examples = source_indices.shape[0]
    rows = np.arange(pad_size, dtype=np.float32)[:, None]
    cols = np.arange(pad_size, dtype=np.float32)[None, :]
    target_rows = (target_indices // pad_size).astype(np.float32)
    target_cols = (target_indices % pad_size).astype(np.float32)
    source = np.zeros((num_examples, pad_size * pad_size), dtype=np.float32)
    target = np.zeros_like(source)
    source[np.arange(num_examples), source_indices] = 1.0
    target[np.arange(num_examples), target_indices] = 1.0
    source = source.reshape(num_examples, pad_size, pad_size)
    target = target.reshape(num_examples, pad_size, pad_size)
    distance = np.abs(rows[None, :, :] - target_rows[:, None, None]) + np.abs(
        cols[None, :, :] - target_cols[:, None, None]
    )
    max_distance = max(2 * (pad_size - 1), 1)
    route_potential = 1.0 - np.minimum(distance, max_distance) / max_distance
    active_f = active.astype(np.float32)
    return np.stack(
        [
            source * active_f,
            target * active_f,
            route_potential.astype(np.float32) * active_f,
        ],
        axis=1,
    )


def select_plan_examples(
    shard: np.lib.npyio.NpzFile,
    selection: str,
    score_temperature: float,
    drop_pass_labels: bool,
    accepted_score_margin: float,
    mixed_best_weight: float,
    accepted_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return selected row ids, source ids, target ids, labels, and weights."""
    plan_actions = shard["plan_action_indices"].astype(np.int32)
    plan_scores = shard["plan_scores"].astype(np.float32)
    plan_q = shard["plan_q"].astype(np.float32)
    plan_outcomes = shard["plan_outcomes"].astype(np.float32)
    source_indices = shard["source_indices"].astype(np.int32)
    target_indices = shard["target_indices"].astype(np.int32)
    num_rows, source_count, target_count = plan_actions.shape
    pass_index = ADAPTIVE_MOVE_PLANES * shard["active"].shape[-1] * shard["active"].shape[-1]

    flat_actions = plan_actions.reshape(num_rows, -1)
    flat_q = plan_q.reshape(num_rows, -1)
    flat_scores = plan_scores.reshape(num_rows, -1)
    flat_outcomes = plan_outcomes.reshape(num_rows, -1)
    teacher_actions = shard["teacher_action_index"].astype(np.int32)
    source_grid = np.broadcast_to(source_indices[:, :, None], (num_rows, source_count, target_count)).reshape(
        num_rows,
        -1,
    )
    target_grid = np.broadcast_to(target_indices[:, None, :], (num_rows, source_count, target_count)).reshape(
        num_rows,
        -1,
    )

    best_key = flat_outcomes * 100000.0 + flat_scores
    best_pos = np.argmax(best_key, axis=1)
    row_ids = np.arange(num_rows, dtype=np.int32)

    teacher_matches = flat_actions == teacher_actions[:, None]
    has_teacher_plan = np.any(teacher_matches, axis=1)
    teacher_pos = np.argmax(teacher_matches.astype(np.int32), axis=1)
    teacher_scores = flat_scores[row_ids, teacher_pos]
    teacher_outcomes = flat_outcomes[row_ids, teacher_pos]
    switched = flat_actions != teacher_actions[:, None]
    outcome_improved = flat_outcomes > teacher_outcomes[:, None]
    score_improved = (flat_outcomes == teacher_outcomes[:, None]) & (
        (flat_scores - teacher_scores[:, None]) >= accepted_score_margin
    )
    accepted = has_teacher_plan[:, None] & switched & (outcome_improved | score_improved)

    if selection == "best":
        selected_sources = source_grid[row_ids, best_pos]
        selected_targets = target_grid[row_ids, best_pos]
        labels = flat_actions[row_ids, best_pos]
        weights = np.ones((num_rows,), dtype=np.float32)
    elif selection == "all":
        row_ids = np.repeat(np.arange(num_rows, dtype=np.int32), source_count * target_count)
        selected_sources = source_grid.reshape(-1)
        selected_targets = target_grid.reshape(-1)
        labels = flat_actions.reshape(-1)
        centered_q = flat_q - np.max(flat_q, axis=1, keepdims=True)
        probs = np.exp(centered_q / score_temperature)
        probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1.0e-6)
        weights = (probs * source_count * target_count).reshape(-1).astype(np.float32)
    elif selection == "accepted":
        accepted_rows, accepted_pos = np.nonzero(accepted)
        row_ids = accepted_rows.astype(np.int32)
        selected_sources = source_grid[row_ids, accepted_pos]
        selected_targets = target_grid[row_ids, accepted_pos]
        labels = flat_actions[row_ids, accepted_pos]
        weights = np.full((row_ids.shape[0],), accepted_weight, dtype=np.float32)
    elif selection == "mixed":
        accepted_rows, accepted_pos = np.nonzero(accepted)
        best_rows = row_ids
        row_ids = np.concatenate([best_rows, accepted_rows.astype(np.int32)], axis=0)
        selected_sources = np.concatenate(
            [source_grid[best_rows, best_pos], source_grid[accepted_rows, accepted_pos]],
            axis=0,
        )
        selected_targets = np.concatenate(
            [target_grid[best_rows, best_pos], target_grid[accepted_rows, accepted_pos]],
            axis=0,
        )
        labels = np.concatenate([flat_actions[best_rows, best_pos], flat_actions[accepted_rows, accepted_pos]], axis=0)
        weights = np.concatenate(
            [
                np.full((best_rows.shape[0],), mixed_best_weight, dtype=np.float32),
                np.full((accepted_rows.shape[0],), accepted_weight, dtype=np.float32),
            ],
            axis=0,
        )
    else:
        raise ValueError(f"unknown selection mode: {selection}")

    if drop_pass_labels:
        keep = labels != pass_index
        row_ids = row_ids[keep]
        selected_sources = selected_sources[keep]
        selected_targets = selected_targets[keep]
        labels = labels[keep]
        weights = weights[keep]
    return row_ids, selected_sources, selected_targets, labels.astype(np.int32), weights.astype(np.float32)


def load_plan_worker_dataset(
    paths: list[Path],
    selection: str,
    score_temperature: float,
    drop_pass_labels: bool,
    accepted_score_margin: float,
    mixed_best_weight: float,
    accepted_weight: float,
    max_examples: int | None,
    seed: int,
) -> dict[str, jnp.ndarray]:
    """Load Plan-Q shards and construct Worker command observations."""
    chunks: dict[str, list[np.ndarray]] = {
        "obs": [],
        "legal_mask": [],
        "active": [],
        "labels": [],
        "weights": [],
    }
    for path in paths:
        shard = np.load(path)
        row_ids, sources, targets, labels, weights = select_plan_examples(
            shard,
            selection,
            score_temperature,
            drop_pass_labels,
            accepted_score_margin,
            mixed_best_weight,
            accepted_weight,
        )
        if row_ids.shape[0] == 0:
            continue
        base_obs = shard["obs"][row_ids].astype(np.float32)
        active = shard["active"][row_ids].astype(np.bool_)
        command = plan_command_planes(sources, targets, active, base_obs.shape[-1])
        chunks["obs"].append(np.concatenate([base_obs, command], axis=1).astype(np.float32))
        chunks["legal_mask"].append(shard["legal_mask"][row_ids].astype(np.bool_))
        chunks["active"].append(active)
        chunks["labels"].append(labels)
        chunks["weights"].append(weights)

    if not chunks["obs"]:
        raise ValueError("No Plan-Worker examples selected; relax selection or accepted margins")
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_examples is not None and arrays["obs"].shape[0] > max_examples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(arrays["obs"].shape[0], size=max_examples, replace=False))
        arrays = {name: value[indices] for name, value in arrays.items()}
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def _heatmap_argmax_indices(heatmap: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return active-cell argmax indices for spatial supervision maps."""
    flat_scores = heatmap.reshape(heatmap.shape[0], -1).astype(np.float32)
    flat_active = active.reshape(active.shape[0], -1)
    return np.argmax(np.where(flat_active, flat_scores, -1.0e9), axis=1).astype(np.int32)


def load_strategy_worker_dataset(
    paths: list[Path],
    drop_pass_labels: bool,
    require_outcome_win: bool,
    require_search_best_win: bool,
    require_finish_within_250: bool,
    max_examples: int | None,
    seed: int,
) -> dict[str, jnp.ndarray]:
    """Load strategy shards and construct Worker command observations.

    Strategy shards store one source heatmap, one target heatmap, and the
    rollout-search teacher action per row. This gives a cheap executor dataset
    for decisive midgame states without rerunning Plan-Q candidate generation.
    """
    chunks: dict[str, list[np.ndarray]] = {
        "obs": [],
        "legal_mask": [],
        "active": [],
        "labels": [],
        "weights": [],
    }
    stats = {"rows": 0, "kept": 0}
    for path in paths:
        shard = np.load(path)
        base_obs = shard["obs"].astype(np.float32)
        active = shard["active"].astype(np.bool_)
        labels = shard["teacher_action_index"].astype(np.int32)
        pass_index = ADAPTIVE_MOVE_PLANES * active.shape[-1] * active.shape[-1]
        keep = np.ones((base_obs.shape[0],), dtype=np.bool_)
        if require_outcome_win:
            if "outcome" not in shard:
                raise KeyError(f"{path} is missing outcome for --require-outcome-win")
            keep &= shard["outcome"].astype(np.int32) == OUTCOME_WIN
        if require_search_best_win:
            if "search_best_outcome" not in shard:
                raise KeyError(f"{path} is missing search_best_outcome for --require-search-best-win")
            keep &= shard["search_best_outcome"].astype(np.int32) == OUTCOME_WIN
        if require_finish_within_250:
            if "finish_within_250" not in shard:
                raise KeyError(f"{path} is missing finish_within_250 for --require-finish-within-250")
            keep &= shard["finish_within_250"].astype(np.float32) > 0.5
        if drop_pass_labels:
            keep &= labels != pass_index
        if "legal_mask" in shard:
            direction_legal = np.transpose(shard["legal_mask"].astype(np.bool_), (0, 3, 1, 2))
            move_legal = np.concatenate([direction_legal, direction_legal], axis=1).reshape(base_obs.shape[0], -1)
            pass_legal = np.ones((base_obs.shape[0], 1), dtype=np.bool_)
            flat_legal = np.concatenate([move_legal, pass_legal], axis=1)
            keep &= flat_legal[np.arange(base_obs.shape[0]), np.clip(labels, 0, flat_legal.shape[1] - 1)]
        stats["rows"] += int(base_obs.shape[0])
        stats["kept"] += int(np.sum(keep))
        if not np.any(keep):
            continue
        base_obs = base_obs[keep]
        active = active[keep]
        labels = labels[keep]
        source_indices = _heatmap_argmax_indices(shard["source_heatmap"][keep], active)
        target_indices = _heatmap_argmax_indices(shard["target_heatmap"][keep], active)
        command = plan_command_planes(source_indices, target_indices, active, base_obs.shape[-1])
        chunks["obs"].append(np.concatenate([base_obs, command], axis=1).astype(np.float32))
        chunks["legal_mask"].append(shard["legal_mask"][keep].astype(np.bool_))
        chunks["active"].append(active)
        chunks["labels"].append(labels)
        chunks["weights"].append(np.ones((labels.shape[0],), dtype=np.float32))

    if not chunks["obs"]:
        raise ValueError("No strategy Worker examples selected; relax filters or keep pass labels")
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_examples is not None and arrays["obs"].shape[0] > max_examples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(arrays["obs"].shape[0], size=max_examples, replace=False))
        arrays = {name: value[indices] for name, value in arrays.items()}
    dataset = {name: jnp.asarray(value) for name, value in arrays.items()}
    dataset["stats"] = stats
    return dataset


@eqx.filter_jit
def train_step(
    network,
    opt_state,
    batch,
    optimizer,
    action_weight: float,
    source_weight: float,
    direction_weight: float,
):
    """Train one Plan-Worker minibatch."""
    obs, masks, active, labels, weights = batch
    action_dim = adaptive_action_space_size(active.shape[-1])
    targets = jax.nn.one_hot(labels, action_dim, dtype=jnp.float32)
    source_targets, direction_targets = worker_source_direction_targets(targets, active.shape[-1])

    def loss_fn(net):
        logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs, masks, active)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        source_logits, direction_logits = worker_source_direction_logits(logits, active.shape[-1])
        source_log_probs = jax.nn.log_softmax(source_logits, axis=-1)
        direction_log_probs = jax.nn.log_softmax(direction_logits, axis=-1)
        action_losses = -jnp.sum(targets * log_probs, axis=-1)
        source_losses = -jnp.sum(source_targets * source_log_probs, axis=-1)
        direction_losses = -jnp.sum(direction_targets * direction_log_probs, axis=-1)
        losses = action_weight * action_losses + source_weight * source_losses + direction_weight * direction_losses
        normalizer = jnp.maximum(jnp.sum(weights), 1.0)
        loss = jnp.sum(losses * weights) / normalizer
        predictions = jnp.argmax(logits, axis=-1)
        source_predictions = jnp.argmax(source_logits, axis=-1)
        direction_predictions = jnp.argmax(direction_logits, axis=-1)
        source_labels = jnp.argmax(source_targets, axis=-1)
        direction_labels = jnp.argmax(direction_targets, axis=-1)
        action_accuracy = jnp.sum((predictions == labels).astype(jnp.float32) * weights) / normalizer
        source_accuracy = jnp.sum((source_predictions == source_labels).astype(jnp.float32) * weights) / normalizer
        direction_accuracy = jnp.sum((direction_predictions == direction_labels).astype(jnp.float32) * weights) / normalizer
        predicted_mass = jnp.take_along_axis(targets, predictions[:, None], axis=1)[:, 0]
        useful_accuracy = jnp.sum((predicted_mass > 0.0).astype(jnp.float32) * weights) / normalizer
        return loss, (action_accuracy, source_accuracy, direction_accuracy, useful_accuracy)

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
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
    action_weight: float,
    source_weight: float,
    direction_weight: float,
):
    """Shuffle one full pass over Plan-Worker examples."""
    num_examples = dataset["obs"].shape[0]
    permutation = jrandom.permutation(key, num_examples)
    num_batches = max((num_examples + minibatch_size - 1) // minibatch_size, 1)
    loss_sum = 0.0
    metric_sum = None
    for batch_index in range(num_batches):
        start = batch_index * minibatch_size
        end = min(start + minibatch_size, num_examples)
        idx = permutation[start:end]
        batch = (
            dataset["obs"][idx],
            dataset["legal_mask"][idx],
            dataset["active"][idx],
            dataset["labels"][idx],
            dataset["weights"][idx],
        )
        network, opt_state, loss, metrics = train_step(
            network,
            opt_state,
            batch,
            optimizer,
            action_weight,
            source_weight,
            direction_weight,
        )
        loss_sum += loss
        metric_sum = metrics if metric_sum is None else jax.tree.map(lambda a, b: a + b, metric_sum, metrics)
    return network, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metric_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a target-conditioned Worker from Plan-Q shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument("--dataset-format", choices=("plan-q", "strategy"), default="plan-q")
    parser.add_argument("--selection", choices=("best", "all", "accepted", "mixed"), default="best")
    parser.add_argument("--score-temperature", type=float, default=0.25)
    parser.add_argument("--accepted-score-margin", type=float, default=25.0)
    parser.add_argument("--mixed-best-weight", type=float, default=0.5)
    parser.add_argument("--accepted-weight", type=float, default=1.0)
    parser.add_argument("--keep-pass-labels", action="store_true")
    parser.add_argument("--require-outcome-win", action="store_true")
    parser.add_argument("--require-search-best-win", action="store_true")
    parser.add_argument("--require-finish-within-250", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--init-input-channels", type=int, default=None)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/adaptive-plan-worker-supervised/generals-adaptive-plan-worker.eqx")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.2)
    parser.add_argument("--source-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.score_temperature <= 0.0:
        parser.error("--score-temperature must be positive")
    if args.accepted_score_margin < 0.0:
        parser.error("--accepted-score-margin must be non-negative")
    if args.mixed_best_weight < 0.0 or args.accepted_weight < 0.0:
        parser.error("--mixed-best-weight and --accepted-weight must be non-negative")
    if args.selection == "mixed" and args.mixed_best_weight + args.accepted_weight <= 0.0:
        parser.error("mixed selection requires a positive best or accepted weight")
    if args.selection == "accepted" and args.accepted_weight <= 0.0:
        parser.error("accepted selection requires --accepted-weight > 0")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if args.pad_to <= 0:
        parser.error("--pad-to must be positive")
    if args.num_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("--num-epochs and --minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if min(args.action_loss_weight, args.source_loss_weight, args.direction_loss_weight) < 0.0:
        parser.error("loss weights must be non-negative")
    if args.action_loss_weight + args.source_loss_weight + args.direction_loss_weight <= 0.0:
        parser.error("at least one loss weight must be positive")
    try:
        args.channels = parse_policy_channels(args.channels)
        args.init_channels = parse_policy_channels(args.init_channels) if args.init_channels is not None else None
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    if args.dataset_format == "strategy":
        dataset = load_strategy_worker_dataset(
            paths,
            not args.keep_pass_labels,
            args.require_outcome_win,
            args.require_search_best_win,
            args.require_finish_within_250,
            args.max_examples,
            args.seed,
        )
    else:
        dataset = load_plan_worker_dataset(
            paths,
            args.selection,
            args.score_temperature,
            not args.keep_pass_labels,
            args.accepted_score_margin,
            args.mixed_best_weight,
            args.accepted_weight,
            args.max_examples,
            args.seed,
        )
    input_channels = args.input_channels or int(dataset["obs"].shape[1])
    if input_channels != int(dataset["obs"].shape[1]):
        raise ValueError(f"--input-channels {input_channels} does not match dataset channels {dataset['obs'].shape[1]}")

    print("Adaptive Plan-Worker supervised training")
    print(f"Device:       {jax.devices()[0]}")
    print(f"Datasets:     {len(paths)} shard(s)")
    print(f"Examples:     {dataset['obs'].shape[0]}")
    print(f"Format:       {args.dataset_format}")
    if args.dataset_format == "strategy":
        stats = dataset.get("stats", {})
        if stats:
            print(f"Rows kept:    {stats['kept']} / {stats['rows']}")
        filters = []
        if args.require_outcome_win:
            filters.append("outcome=win")
        if args.require_search_best_win:
            filters.append("search_best=win")
        if args.require_finish_within_250:
            filters.append("finish<=250")
        if filters:
            print(f"Filters:      {', '.join(filters)}")
    if args.dataset_format == "plan-q":
        print(f"Selection:    {args.selection}")
        if args.selection in ("accepted", "mixed"):
            print(f"Accepted:     score_margin={args.accepted_score_margin:g}, weight={args.accepted_weight:g}")
        if args.selection == "mixed":
            print(f"Mixed best:   weight={args.mixed_best_weight:g}")
    print(f"Input chans:  {input_channels}")
    print(f"Arch:         {args.network_arch}")
    print(
        "Loss weights: "
        f"action={args.action_loss_weight:g}, source={args.source_loss_weight:g}, "
        f"direction={args.direction_loss_weight:g}"
    )
    if args.init_model_path is not None:
        print(f"Warm start:   {args.init_model_path}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, net_key = jrandom.split(key)
    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.init_model_path,
        channels=args.channels,
        init_channels=args.init_channels,
        input_channels=input_channels,
        init_input_channels=args.init_input_channels,
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
            args.action_loss_weight,
            args.source_loss_weight,
            args.direction_loss_weight,
        )
        jax.block_until_ready(network)
        action_accuracy, source_accuracy, direction_accuracy, useful_accuracy = metrics
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Act {float(action_accuracy) * 100:5.1f}% | "
            f"Src {float(source_accuracy) * 100:5.1f}% | "
            f"Dir {float(direction_accuracy) * 100:5.1f}% | "
            f"Useful {float(useful_accuracy) * 100:5.1f}% | "
            f"Time {time.time() - t0:.2f}s"
        )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
