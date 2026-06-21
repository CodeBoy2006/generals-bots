"""Evaluate additive and explicit Plan-Q source-target pair scorers."""

from __future__ import annotations

import argparse
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

from adaptive_common import parse_grid_sizes
from adaptive_network import load_or_create_adaptive_network
from adaptive_plan_pair_scorer import PlanPairScorerNetwork
from adaptive_plan_pair_supervised import build_pair_dataset, expand_dataset_paths, evaluate_model, pair_rank_metrics
from generals.agents.ppo_policy_agent import parse_policy_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Plan-Q source-target pair ranking metrics.")
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
    parser.add_argument("--scorer-path", default=None)
    parser.add_argument("--scorer-json", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--q-target-outcome-weight", type=float, default=0.65)
    parser.add_argument("--q-rank-temperature", type=float, default=0.05)
    parser.add_argument("--gap-weighting", action="store_true")
    parser.add_argument("--min-plan-gap", type=float, default=0.0)
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
    if args.feature_batch_size <= 0:
        parser.error("--feature-batch-size must be positive")
    if not (0.0 <= args.q_target_outcome_weight <= 1.0):
        parser.error("--q-target-outcome-weight must be in [0, 1]")
    if args.q_rank_temperature <= 0.0:
        parser.error("--q-rank-temperature must be positive")
    if args.min_plan_gap < 0.0:
        parser.error("--min-plan-gap must be non-negative")
    if not (0.0 < args.validation_fraction < 1.0):
        parser.error("--validation-fraction must be between 0 and 1")
    if args.max_rows is not None and args.max_rows <= 1:
        parser.error("--max-rows must be greater than 1")
    if (args.scorer_path is None) != (args.scorer_json is None):
        parser.error("--scorer-path and --scorer-json must be provided together")
    if args.scorer_path is not None and args.hidden_dim is not None:
        parser.error("--hidden-dim is read from --scorer-json when loading a scorer")
    return args


def metric_line(name: str, split: str, loss, metrics: dict[str, jnp.ndarray]) -> str:
    return (
        f"{name:8s} {split:5s} "
        f"loss={float(loss):.4f} "
        f"pair@1={float(metrics['pair_accuracy']) * 100:5.1f}% "
        f"pair@2={float(metrics['pair_top2_accuracy']) * 100:5.1f}% "
        f"pair@4={float(metrics['pair_top4_accuracy']) * 100:5.1f}% "
        f"source={float(metrics['source_accuracy']) * 100:5.1f}% "
        f"target={float(metrics['target_accuracy']) * 100:5.1f}% "
        f"corr={float(metrics['correlation']):+.3f} "
        f"margin={float(metrics['margin']):.3f}"
    )


def additive_scores(features: jnp.ndarray) -> jnp.ndarray:
    """Use the decomposed source and target logits as an additive pair score."""
    return features[:, :, 2] + features[:, :, 3]


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    network = load_or_create_adaptive_network(
        jrandom.PRNGKey(args.seed),
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
        args.min_plan_gap,
        args.validation_fraction,
        args.max_rows,
        args.seed,
    )
    print("Adaptive Plan-Q pair-scorer evaluation")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Rows:          {dataset['rows']}")
    print(f"Train/val:     {dataset['train_features'].shape[0]} / {dataset['val_features'].shape[0]}")
    print(f"Pairs/row:     {dataset['pair_count']}")
    print(f"Feature model: {args.feature_model_path}")
    print()

    for split in ("train", "val"):
        logits = additive_scores(dataset[f"{split}_features"])
        loss, metrics = pair_rank_metrics(
            logits,
            dataset[f"{split}_targets"],
            dataset[f"{split}_weights"],
            dataset[f"{split}_source_pos"],
            dataset[f"{split}_target_pos"],
            dataset["target_count"],
            args.q_rank_temperature,
        )
        print(metric_line("additive", split, loss, metrics))

    if args.scorer_path is None:
        return
    scorer_metadata = json.loads(Path(args.scorer_json).read_text(encoding="utf-8"))
    scorer = PlanPairScorerNetwork(
        jrandom.PRNGKey(args.seed + 1),
        input_dim=int(scorer_metadata["input_dim"]),
        hidden_dim=int(scorer_metadata["hidden_dim"]),
        feature_mean=jnp.asarray(scorer_metadata["feature_mean"]),
        feature_std=jnp.asarray(scorer_metadata["feature_std"]),
    )
    scorer = eqx.tree_deserialise_leaves(args.scorer_path, scorer)
    for split in ("train", "val"):
        loss, metrics = evaluate_model(
            scorer,
            dataset[f"{split}_features"],
            dataset[f"{split}_targets"],
            dataset[f"{split}_weights"],
            dataset[f"{split}_source_pos"],
            dataset[f"{split}_target_pos"],
            dataset["target_count"],
            args.q_rank_temperature,
        )
        print(metric_line("scorer", split, loss, metrics))


if __name__ == "__main__":
    main()
