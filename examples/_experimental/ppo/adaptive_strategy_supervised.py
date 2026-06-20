"""Offline supervised training for adaptive strategy auxiliary heads."""

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

from adaptive_common import parse_grid_sizes
from adaptive_network import hl_gauss_value_loss, load_or_create_adaptive_network
from adaptive_search_distill import binary_cross_entropy_with_logits
from adaptive_strategy_aux import STRATEGY_INTENT_FINISH
from generals.agents.ppo_policy_agent import parse_policy_channels

OUTCOME_LOSS = 0
OUTCOME_DRAW = 1
OUTCOME_WIN = 2
DATASET_FORMAT_MODES = ("strategy", "plan-q-prefix", "online-search")
ACTION_CE_WEIGHT_MODES = (
    "all",
    "non-draw",
    "wins",
    "search-best-win",
    "search-used",
    "search-changed",
    "search-continuation-win",
    "search-improves-continuation",
    "search-converts-win",
)
BALANCE_STRATA_MODES = (
    "none",
    "size-seat",
    "size-seat-domain",
    "size-seat-oversample",
    "size-seat-domain-oversample",
)
LABEL_SOURCE_MODES = (
    "trajectory",
    "search-best",
    "search-best-or-trajectory",
    "search-continuation",
    "search-continuation-or-trajectory",
)


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


def dataset_domain_name(path: Path) -> str:
    """Return a coarse data-domain label for balancing mixed offline shards."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        opponent_policy_path = metadata.get("opponent_policy_path")
        if opponent_policy_path:
            return f"policy:{Path(opponent_policy_path).name}"
        opponent = metadata.get("opponent")
        if opponent:
            return f"opponent:{opponent}"
    name = path.parent.name
    if "fixed-v5" in name:
        return "policy:fixed-v5"
    if "expander" in name:
        return "opponent:expander"
    return f"path:{name}"


def load_strategy_dataset(
    paths: list[Path],
    max_samples: int | None = None,
    max_samples_per_shard: int | None = None,
    seed: int = 0,
    finish_head_mode: str = "binary",
    action_ce_weight_mode: str = "all",
    label_source: str = "trajectory",
    action_ce_path_contains: tuple[str, ...] = (),
    min_row_turn: int = 0,
    max_row_turn: int | None = None,
    require_contact: bool = False,
    min_visible_enemy_cells: int = 0,
    min_visible_enemy_density: float = 0.0,
    require_outcome_win: bool = False,
    require_outcome_draw: bool = False,
    require_outcome_nonwin: bool = False,
    require_search_best_win: bool = False,
    require_search_used: bool = False,
    require_search_action_changed: bool = False,
    require_search_improves_continuation: bool = False,
    require_search_converts_to_win: bool = False,
    require_finish_within_250: bool = False,
    require_win_or_finish_within_250: bool = False,
    min_search_score_gap: float = 0.0,
    return_stats: bool = False,
) -> dict[str, jnp.ndarray] | tuple[dict[str, jnp.ndarray], dict]:
    """Load the subset of NPZ fields needed by the frozen-head trainer."""
    rng = np.random.default_rng(seed)
    search_candidate_count = 1
    for path in paths:
        with np.load(path) as shard:
            if "search_candidate_indices" in shard:
                search_candidate_count = max(search_candidate_count, int(shard["search_candidate_indices"].shape[1]))
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
        "source_heatmap": [],
        "target_heatmap": [],
        "teacher_logits": [],
        "teacher_action": [],
        "action_weight": [],
        "grid_size": [],
        "seat": [],
        "domain": [],
        "search_candidate_indices": [],
        "search_prior_scores": [],
        "search_scores": [],
        "search_outcomes": [],
        "search_score_gap": [],
        "prefix_weight": [],
        "prefix_base_action": [],
    }
    load_stats = {
        "rows": 0,
        "kept": 0,
        "sampled": 0,
        "filters": [],
    }
    domain_to_id: dict[str, int] = {}

    def require_field(shard, path: Path, name: str) -> np.ndarray:
        if name not in shard:
            raise KeyError(f"{path} is missing {name} for active row filters")
        return shard[name]

    for path in paths:
        shard = np.load(path)
        domain_name = dataset_domain_name(path)
        domain_id = domain_to_id.setdefault(domain_name, len(domain_to_id))
        shard_samples = shard["obs"].shape[0]
        row_keep = np.ones((shard_samples,), dtype=np.bool_)

        def add_row_filter(name: str, mask: np.ndarray) -> None:
            nonlocal row_keep
            bool_mask = np.asarray(mask, dtype=np.bool_)
            load_stats["filters"].append({"name": name, "matches": int(np.sum(bool_mask))})
            row_keep &= bool_mask

        if min_row_turn > 0:
            add_row_filter(f"time>={min_row_turn}", require_field(shard, path, "time").astype(np.int32) >= min_row_turn)
        if max_row_turn is not None:
            add_row_filter(f"time<={max_row_turn}", require_field(shard, path, "time").astype(np.int32) <= max_row_turn)
        if require_contact:
            add_row_filter("contact", require_field(shard, path, "contact").astype(np.float32) > 0.5)
        if min_visible_enemy_cells > 0:
            add_row_filter(
                f"visible_enemy_cells>={min_visible_enemy_cells}",
                require_field(shard, path, "visible_enemy_count").astype(np.int32) >= min_visible_enemy_cells,
            )
        if min_visible_enemy_density > 0.0:
            add_row_filter(
                f"visible_enemy_density>={min_visible_enemy_density:g}",
                require_field(shard, path, "visible_enemy_density").astype(np.float32) >= min_visible_enemy_density,
            )
        if require_outcome_win:
            known = require_field(shard, path, "outcome_known").astype(np.float32) > 0.0
            add_row_filter("outcome=win", (shard["outcome"].astype(np.int32) == OUTCOME_WIN) & known)
        if require_outcome_draw:
            known = require_field(shard, path, "outcome_known").astype(np.float32) > 0.0
            add_row_filter("outcome=draw", (shard["outcome"].astype(np.int32) == OUTCOME_DRAW) & known)
        if require_outcome_nonwin:
            known = require_field(shard, path, "outcome_known").astype(np.float32) > 0.0
            add_row_filter("outcome!=win", (shard["outcome"].astype(np.int32) != OUTCOME_WIN) & known)
        if require_search_best_win:
            add_row_filter(
                "search_best=win",
                require_field(shard, path, "search_best_outcome").astype(np.int32) == OUTCOME_WIN,
            )
        if require_search_used:
            add_row_filter("search_used", require_field(shard, path, "search_used").astype(np.bool_))
        if require_search_action_changed:
            add_row_filter(
                "search_action_changed",
                require_field(shard, path, "search_action_changed").astype(np.bool_),
            )
        if require_search_improves_continuation:
            add_row_filter(
                "search_improves_continuation",
                require_field(shard, path, "search_improves_continuation").astype(np.bool_),
            )
        if require_search_converts_to_win:
            add_row_filter(
                "search_converts_to_win",
                require_field(shard, path, "search_converts_to_win").astype(np.bool_),
            )
        if require_finish_within_250:
            add_row_filter(
                "finish<=250",
                require_field(shard, path, "finish_within_250").astype(np.float32) > 0.5,
            )
        if require_win_or_finish_within_250:
            known = require_field(shard, path, "outcome_known").astype(np.float32) > 0.0
            wins = (shard["outcome"].astype(np.int32) == OUTCOME_WIN) & known
            finish250 = require_field(shard, path, "finish_within_250").astype(np.float32) > 0.5
            add_row_filter("win_or_finish<=250", wins | finish250)
        if min_search_score_gap > 0.0:
            add_row_filter(
                f"search_gap>={min_search_score_gap:g}",
                require_field(shard, path, "search_score_gap").astype(np.float32) >= min_search_score_gap,
            )

        load_stats["rows"] += int(shard_samples)
        load_stats["kept"] += int(np.sum(row_keep))
        shard_indices = np.flatnonzero(row_keep)
        if shard_indices.size == 0:
            continue
        if max_samples_per_shard is not None and shard_indices.size > max_samples_per_shard:
            shard_indices = np.sort(rng.choice(shard_indices, size=max_samples_per_shard, replace=False))
        load_stats["sampled"] += int(shard_indices.shape[0])
        chunks["obs"].append(shard["obs"][shard_indices].astype(np.float32))
        chunks["legal_mask"].append(shard["legal_mask"][shard_indices].astype(np.bool_))
        chunks["active"].append(shard["active"][shard_indices].astype(np.bool_))
        chunks["intent"].append(shard["intent"][shard_indices].astype(np.int32))
        trajectory_outcome = shard["outcome"][shard_indices].astype(np.int32)
        trajectory_outcome_weight = shard["outcome_known"][shard_indices].astype(np.float32)
        if "search_best_outcome" in shard:
            search_best_outcome = shard["search_best_outcome"][shard_indices].astype(np.int32)
        else:
            search_best_outcome = np.full((shard_indices.shape[0],), -1, dtype=np.int32)
        if "search_continuation_outcome" in shard:
            search_continuation_outcome = shard["search_continuation_outcome"][shard_indices].astype(np.int32)
        else:
            search_continuation_outcome = np.full((shard_indices.shape[0],), -1, dtype=np.int32)
        if label_source in ("search-best", "search-best-or-trajectory"):
            search_label = search_best_outcome
            label_known = search_label >= 0
            fallback_to_trajectory = label_source == "search-best-or-trajectory"
        elif label_source in ("search-continuation", "search-continuation-or-trajectory"):
            search_label = search_continuation_outcome
            label_known = search_label >= 0
            fallback_to_trajectory = label_source == "search-continuation-or-trajectory"
        else:
            search_label = np.full_like(search_best_outcome, -1)
            label_known = np.zeros_like(search_best_outcome, dtype=np.bool_)
            fallback_to_trajectory = False
        if label_source != "trajectory":
            if np.any(label_known):
                if fallback_to_trajectory:
                    outcome_target = np.where(label_known, search_label, trajectory_outcome).astype(np.int32)
                    outcome_weight = np.where(label_known, 1.0, trajectory_outcome_weight).astype(np.float32)
                    finish_target = np.where(
                        label_known,
                        search_label == OUTCOME_WIN,
                        shard["finish_within_250"][shard_indices].astype(np.float32) > 0.5,
                    ).astype(np.float32)
                else:
                    outcome_target = np.where(label_known, search_label, OUTCOME_DRAW).astype(np.int32)
                    outcome_weight = label_known.astype(np.float32)
                    finish_target = (search_label == OUTCOME_WIN).astype(np.float32)
            else:
                shard_count = shard_indices.shape[0]
                if fallback_to_trajectory:
                    outcome_target = trajectory_outcome
                    outcome_weight = trajectory_outcome_weight
                    finish_target = shard["finish_within_250"][shard_indices].astype(np.float32)
                else:
                    outcome_target = np.full((shard_count,), OUTCOME_DRAW, dtype=np.int32)
                    outcome_weight = np.zeros((shard_count,), dtype=np.float32)
                    finish_target = np.zeros((shard_count,), dtype=np.float32)
        else:
            outcome_target = trajectory_outcome
            outcome_weight = trajectory_outcome_weight
            finish_target = shard["finish_within_250"][shard_indices].astype(np.float32)
        if finish_head_mode == "multi-horizon":
            trajectory_finish_targets = np.stack(
                [
                    shard["finish_within_50"][shard_indices],
                    shard["finish_within_100"][shard_indices],
                    shard["finish_within_250"][shard_indices],
                ],
                axis=-1,
            ).astype(np.float32)
            if label_source == "search-best-or-trajectory":
                # Search-best labels are horizon-free; repeat them only for rows
                # that actually have local search labels. Ordinary contrast rows
                # keep their trajectory horizon labels.
                search_finish_targets = np.repeat(finish_target[:, None], 3, axis=1)
                finish_targets = np.where(label_known[:, None], search_finish_targets, trajectory_finish_targets)
            elif label_source == "search-best":
                # Search-best labels are horizon-free; repeat the same target so
                # multi-output checkpoints can still learn the search win signal.
                finish_targets = np.repeat(finish_target[:, None], 3, axis=1)
            else:
                finish_targets = trajectory_finish_targets
            chunks["finish"].append(finish_targets.astype(np.float32))
        else:
            chunks["finish"].append((finish_target > 0.5).astype(np.int32))
        chunks["finish_weight"].append(outcome_weight.astype(np.float32))
        chunks["outcome"].append(outcome_target.astype(np.int32))
        chunks["outcome_weight"].append(outcome_weight.astype(np.float32))
        chunks["enemy_general"].append(shard["enemy_general_heatmap"][shard_indices].astype(np.float32))
        chunks["source_heatmap"].append(shard["source_heatmap"][shard_indices].astype(np.float32))
        chunks["target_heatmap"].append(shard["target_heatmap"][shard_indices].astype(np.float32))
        chunks["teacher_logits"].append(shard["teacher_logits"][shard_indices].astype(np.float32))
        chunks["teacher_action"].append(shard["teacher_action_index"][shard_indices].astype(np.int32))
        chunks["grid_size"].append(shard["grid_size"][shard_indices].astype(np.int32))
        chunks["seat"].append(shard["seat"][shard_indices].astype(np.int32))
        chunks["domain"].append(np.full((shard_indices.shape[0],), domain_id, dtype=np.int32))
        shard_count = shard_indices.shape[0]
        if "search_candidate_indices" in shard:
            candidate_indices = shard["search_candidate_indices"][shard_indices].astype(np.int32)
            prior_scores = shard["search_prior_scores"][shard_indices].astype(np.float32)
            search_scores = shard["search_scores"][shard_indices].astype(np.float32)
            if "search_outcomes" in shard:
                search_outcomes = shard["search_outcomes"][shard_indices].astype(np.int32)
            else:
                search_outcomes = np.full_like(candidate_indices, -1, dtype=np.int32)
            if candidate_indices.shape[1] < search_candidate_count:
                pad_width = search_candidate_count - candidate_indices.shape[1]
                candidate_indices = np.pad(candidate_indices, ((0, 0), (0, pad_width)), constant_values=0)
                prior_scores = np.pad(prior_scores, ((0, 0), (0, pad_width)), constant_values=-1.0e4)
                search_scores = np.pad(search_scores, ((0, 0), (0, pad_width)), constant_values=-1.0e4)
                search_outcomes = np.pad(search_outcomes, ((0, 0), (0, pad_width)), constant_values=-1)
            chunks["search_candidate_indices"].append(candidate_indices[:, :search_candidate_count])
            chunks["search_prior_scores"].append(prior_scores[:, :search_candidate_count])
            chunks["search_scores"].append(search_scores[:, :search_candidate_count])
            chunks["search_outcomes"].append(search_outcomes[:, :search_candidate_count])
            chunks["search_score_gap"].append(shard["search_score_gap"][shard_indices].astype(np.float32))
        else:
            chunks["search_candidate_indices"].append(np.zeros((shard_count, search_candidate_count), dtype=np.int32))
            chunks["search_prior_scores"].append(
                np.full((shard_count, search_candidate_count), -1.0e4, dtype=np.float32)
            )
            chunks["search_scores"].append(np.full((shard_count, search_candidate_count), -1.0e4, dtype=np.float32))
            chunks["search_outcomes"].append(np.full((shard_count, search_candidate_count), -1, dtype=np.int32))
            chunks["search_score_gap"].append(np.zeros((shard_count,), dtype=np.float32))
        outcome = trajectory_outcome
        outcome_known = trajectory_outcome_weight > 0.0
        if action_ce_weight_mode == "non-draw":
            action_weight = ~(outcome_known & (outcome == OUTCOME_DRAW))
        elif action_ce_weight_mode == "wins":
            action_weight = outcome_known & (outcome == OUTCOME_WIN)
        elif action_ce_weight_mode == "search-best-win":
            action_weight = search_best_outcome == OUTCOME_WIN
        elif action_ce_weight_mode == "search-used":
            action_weight = require_field(shard, path, "search_used")[shard_indices].astype(np.bool_)
        elif action_ce_weight_mode == "search-changed":
            action_weight = require_field(shard, path, "search_action_changed")[shard_indices].astype(np.bool_)
        elif action_ce_weight_mode == "search-continuation-win":
            action_weight = search_continuation_outcome == OUTCOME_WIN
        elif action_ce_weight_mode == "search-improves-continuation":
            action_weight = require_field(shard, path, "search_improves_continuation")[shard_indices].astype(np.bool_)
        elif action_ce_weight_mode == "search-converts-win":
            action_weight = require_field(shard, path, "search_converts_to_win")[shard_indices].astype(np.bool_)
        else:
            action_weight = np.ones_like(outcome, dtype=np.bool_)
        if action_ce_path_contains and not any(token in str(path) for token in action_ce_path_contains):
            action_weight = np.zeros_like(action_weight, dtype=np.bool_)
        chunks["action_weight"].append(action_weight.astype(np.float32))
        chunks["prefix_weight"].append(action_weight.astype(np.float32))
        if "base_action_index" in shard:
            chunks["prefix_base_action"].append(shard["base_action_index"][shard_indices].astype(np.int32))
        else:
            chunks["prefix_base_action"].append(shard["teacher_action_index"][shard_indices].astype(np.int32))

    if not chunks["obs"]:
        raise ValueError("row filters kept no strategy-supervision samples")
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_samples is not None:
        arrays = {name: value[:max_samples] for name, value in arrays.items()}
    dataset = {name: jnp.asarray(value) for name, value in arrays.items()}
    if return_stats:
        return dataset, load_stats
    return dataset


def heatmaps_from_indices(indices: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Build one-hot spatial maps from flattened cell indices."""
    indices = indices.astype(np.int32)
    num_samples, height, width = active.shape
    heatmaps = np.zeros((num_samples, height * width), dtype=np.float32)
    valid = (indices >= 0) & (indices < height * width)
    if np.any(valid):
        heatmaps[np.arange(num_samples)[valid], indices[valid]] = 1.0
    return heatmaps.reshape(num_samples, height, width) * active.astype(np.float32)


def load_plan_q_prefix_strategy_dataset(
    paths: list[Path],
    max_samples: int | None = None,
    max_samples_per_shard: int | None = None,
    seed: int = 0,
    finish_head_mode: str = "binary",
    action_ce_weight_mode: str = "all",
    action_ce_path_contains: tuple[str, ...] = (),
    min_row_turn: int = 0,
    max_row_turn: int | None = None,
    require_outcome_win: bool = False,
    require_outcome_draw: bool = False,
    require_outcome_nonwin: bool = False,
    min_search_score_gap: float = 0.0,
    min_teacher_action_logit_margin: float | None = None,
    require_plan_advantage: float = 0.0,
    prefix_advantage_weighting: bool = False,
    prefix_step_decay: float = 0.0,
    drop_pass_labels: bool = True,
    return_stats: bool = False,
) -> dict[str, jnp.ndarray] | tuple[dict[str, jnp.ndarray], dict]:
    """Load best-command prefix rows as main-policy supervised examples.

    Plan-Q prefix shards store states reached while executing the best scored
    source-target command.  This adapter turns those prefix states into the
    same schema as strategy shards, using saved base-policy logits as the KL
    anchor and the executed worker action as the small imitation target.
    """
    rng = np.random.default_rng(seed)
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
        "source_heatmap": [],
        "target_heatmap": [],
        "teacher_logits": [],
        "teacher_action": [],
        "action_weight": [],
        "grid_size": [],
        "seat": [],
        "domain": [],
        "search_candidate_indices": [],
        "search_prior_scores": [],
        "search_scores": [],
        "search_outcomes": [],
        "search_score_gap": [],
        "prefix_weight": [],
        "prefix_base_action": [],
    }
    load_stats = {
        "rows": 0,
        "kept": 0,
        "sampled": 0,
        "filters": [],
    }
    domain_to_id: dict[str, int] = {}
    required = (
        "worker_prefix_obs",
        "worker_prefix_legal_mask",
        "worker_prefix_active",
        "worker_prefix_teacher_logits",
        "worker_prefix_action_index",
        "worker_prefix_valid",
        "worker_prefix_time",
        "worker_prefix_source_index",
        "worker_prefix_target_index",
        "worker_prefix_plan_outcome",
        "worker_prefix_plan_q",
        "grid_size",
        "seat",
    )

    for path in paths:
        shard = np.load(path)
        missing = [name for name in required if name not in shard]
        if missing:
            raise KeyError(f"{path} is missing Plan-Q prefix field {missing[0]}")
        base_obs = shard["worker_prefix_obs"].astype(np.float32)
        if base_obs.shape[1] == 0:
            continue
        num_rows, prefix_steps = base_obs.shape[:2]
        total_prefix = num_rows * prefix_steps
        active = shard["worker_prefix_active"].astype(np.bool_)
        labels = shard["worker_prefix_action_index"].astype(np.int32)
        valid = shard["worker_prefix_valid"].astype(np.bool_)
        times = shard["worker_prefix_time"].astype(np.int32)
        outcomes = shard["worker_prefix_plan_outcome"].astype(np.int32)
        logits = shard["worker_prefix_teacher_logits"].astype(np.float32)
        pass_index = logits.shape[-1] - 1
        keep = valid.copy()

        def add_prefix_filter(name: str, mask: np.ndarray) -> None:
            nonlocal keep
            bool_mask = np.asarray(mask, dtype=np.bool_)
            load_stats["filters"].append({"name": name, "matches": int(np.sum(bool_mask))})
            keep &= bool_mask

        if drop_pass_labels:
            add_prefix_filter("non-pass-prefix-action", labels != pass_index)
        if min_row_turn > 0:
            add_prefix_filter(f"time>={min_row_turn}", times >= min_row_turn)
        if max_row_turn is not None:
            add_prefix_filter(f"time<={max_row_turn}", times <= max_row_turn)
        if require_outcome_win:
            add_prefix_filter("plan_outcome=win", outcomes == OUTCOME_WIN)
        if require_outcome_draw:
            add_prefix_filter("plan_outcome=draw", outcomes == OUTCOME_DRAW)
        if require_outcome_nonwin:
            add_prefix_filter("plan_outcome!=win", outcomes != OUTCOME_WIN)
        if min_search_score_gap > 0.0:
            if "plan_q_gap" not in shard:
                raise KeyError(f"{path} is missing plan_q_gap for --min-search-score-gap")
            gap = np.repeat(shard["plan_q_gap"].astype(np.float32)[:, None], prefix_steps, axis=1)
            add_prefix_filter(f"plan_q_gap>={min_search_score_gap:g}", gap >= min_search_score_gap)
        if require_plan_advantage > 0.0:
            if "worker_prefix_plan_advantage" in shard:
                prefix_advantage_for_filter = shard["worker_prefix_plan_advantage"].astype(np.float32)
            elif "plan_advantage" in shard:
                prefix_advantage_for_filter = np.repeat(
                    shard["plan_advantage"].astype(np.float32)[:, None],
                    prefix_steps,
                    axis=1,
                )
            else:
                raise KeyError(f"{path} is missing plan_advantage for --require-plan-advantage")
            add_prefix_filter(
                f"plan_advantage>={require_plan_advantage:g}",
                prefix_advantage_for_filter >= require_plan_advantage,
            )
        if min_teacher_action_logit_margin is not None:
            flat_logits_for_margin = logits.reshape(total_prefix, logits.shape[-1])
            flat_labels_for_margin = labels.reshape(-1)
            chosen_logits = flat_logits_for_margin[
                np.arange(total_prefix),
                np.clip(flat_labels_for_margin, 0, logits.shape[-1] - 1),
            ].reshape(num_rows, prefix_steps)
            top_logits = np.max(flat_logits_for_margin, axis=-1).reshape(num_rows, prefix_steps)
            margin = chosen_logits - top_logits
            add_prefix_filter(
                f"teacher_action_margin>={min_teacher_action_logit_margin:g}",
                margin >= min_teacher_action_logit_margin,
            )

        load_stats["rows"] += int(total_prefix)
        load_stats["kept"] += int(np.sum(keep))
        flat_keep = keep.reshape(-1)
        if not np.any(flat_keep):
            continue
        selected = np.flatnonzero(flat_keep)
        if max_samples_per_shard is not None and selected.size > max_samples_per_shard:
            selected = np.sort(rng.choice(selected, size=max_samples_per_shard, replace=False))
        load_stats["sampled"] += int(selected.shape[0])

        flat_obs = base_obs.reshape(total_prefix, *base_obs.shape[2:])
        flat_mask = shard["worker_prefix_legal_mask"].astype(np.bool_).reshape(
            total_prefix,
            *shard["worker_prefix_legal_mask"].shape[2:],
        )
        flat_active = active.reshape(total_prefix, *active.shape[2:])
        flat_logits = logits.reshape(total_prefix, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_outcomes = outcomes.reshape(-1)
        flat_sources = shard["worker_prefix_source_index"].astype(np.int32).reshape(-1)
        flat_targets = shard["worker_prefix_target_index"].astype(np.int32).reshape(-1)
        row_grid_size = np.repeat(shard["grid_size"].astype(np.int32)[:, None], prefix_steps, axis=1).reshape(-1)
        row_seat = np.repeat(shard["seat"].astype(np.int32)[:, None], prefix_steps, axis=1).reshape(-1)
        if "plan_q_gap" in shard:
            row_gap = np.repeat(shard["plan_q_gap"].astype(np.float32)[:, None], prefix_steps, axis=1).reshape(-1)
        else:
            row_gap = np.zeros((total_prefix,), dtype=np.float32)
        if "worker_prefix_plan_advantage" in shard:
            row_advantage = shard["worker_prefix_plan_advantage"].astype(np.float32).reshape(-1)
        elif "plan_advantage" in shard:
            row_advantage = np.repeat(shard["plan_advantage"].astype(np.float32)[:, None], prefix_steps, axis=1).reshape(-1)
        else:
            row_advantage = row_gap
        if "worker_prefix_step_index" in shard:
            row_step_index = shard["worker_prefix_step_index"].astype(np.float32).reshape(-1)
        else:
            row_step_index = np.tile(np.arange(prefix_steps, dtype=np.float32), num_rows)

        active_selected = flat_active[selected]
        outcome_selected = flat_outcomes[selected]
        finish_binary = (outcome_selected == OUTCOME_WIN).astype(np.float32)
        if finish_head_mode == "multi-horizon":
            finish_targets = np.repeat(finish_binary[:, None], 3, axis=1).astype(np.float32)
        else:
            finish_targets = finish_binary.astype(np.int32)

        outcome_known = np.ones((selected.shape[0],), dtype=np.float32)
        if action_ce_weight_mode == "non-draw":
            action_weight = outcome_selected != OUTCOME_DRAW
        elif action_ce_weight_mode in ("wins", "search-best-win"):
            action_weight = outcome_selected == OUTCOME_WIN
        else:
            action_weight = np.ones_like(outcome_selected, dtype=np.bool_)
        if action_ce_path_contains and not any(token in str(path) for token in action_ce_path_contains):
            action_weight = np.zeros_like(action_weight, dtype=np.bool_)
        prefix_weight = action_weight.astype(np.float32)
        selected_advantage = row_advantage[selected].astype(np.float32)
        if prefix_advantage_weighting:
            prefix_weight *= np.clip(selected_advantage / 0.5, 0.5, 4.0).astype(np.float32)
        if prefix_step_decay > 0.0:
            selected_step = row_step_index[selected].astype(np.float32)
            prefix_weight *= np.exp(-np.log(2.0) * selected_step / prefix_step_decay).astype(np.float32)
        flat_base_actions = np.argmax(np.where(flat_logits > -9999.0, flat_logits, -1.0e9), axis=1).astype(np.int32)

        domain_name = dataset_domain_name(path)
        domain_id = domain_to_id.setdefault(domain_name, len(domain_to_id))
        chunks["obs"].append(flat_obs[selected].astype(np.float32))
        chunks["legal_mask"].append(flat_mask[selected])
        chunks["active"].append(active_selected)
        chunks["intent"].append(
            np.full((selected.shape[0],), STRATEGY_INTENT_FINISH, dtype=np.int32)
        )
        chunks["finish"].append(finish_targets)
        chunks["finish_weight"].append(outcome_known)
        chunks["outcome"].append(outcome_selected.astype(np.int32))
        chunks["outcome_weight"].append(outcome_known)
        chunks["enemy_general"].append(np.zeros(active_selected.shape, dtype=np.float32))
        chunks["source_heatmap"].append(heatmaps_from_indices(flat_sources[selected], active_selected))
        chunks["target_heatmap"].append(heatmaps_from_indices(flat_targets[selected], active_selected))
        chunks["teacher_logits"].append(flat_logits[selected].astype(np.float32))
        chunks["teacher_action"].append(flat_labels[selected].astype(np.int32))
        chunks["action_weight"].append(prefix_weight.astype(np.float32))
        chunks["grid_size"].append(row_grid_size[selected].astype(np.int32))
        chunks["seat"].append(row_seat[selected].astype(np.int32))
        chunks["domain"].append(np.full((selected.shape[0],), domain_id, dtype=np.int32))
        chunks["search_candidate_indices"].append(np.zeros((selected.shape[0], 1), dtype=np.int32))
        chunks["search_prior_scores"].append(np.full((selected.shape[0], 1), -1.0e4, dtype=np.float32))
        chunks["search_scores"].append(np.full((selected.shape[0], 1), -1.0e4, dtype=np.float32))
        chunks["search_outcomes"].append(np.full((selected.shape[0], 1), -1, dtype=np.int32))
        chunks["search_score_gap"].append(row_gap[selected].astype(np.float32))
        chunks["prefix_weight"].append(prefix_weight.astype(np.float32))
        chunks["prefix_base_action"].append(flat_base_actions[selected].astype(np.int32))

    if not chunks["obs"]:
        raise ValueError("row filters kept no Plan-Q prefix strategy samples")
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if max_samples is not None and arrays["obs"].shape[0] > max_samples:
        indices = np.sort(rng.choice(arrays["obs"].shape[0], size=max_samples, replace=False))
        arrays = {name: value[indices] for name, value in arrays.items()}
    dataset = {name: jnp.asarray(value) for name, value in arrays.items()}
    if return_stats:
        return dataset, load_stats
    return dataset


def concatenate_strategy_datasets(datasets: list[dict[str, jnp.ndarray]]) -> dict[str, jnp.ndarray]:
    """Concatenate loaded strategy datasets, padding variable top-k search fields."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    if len(datasets) == 1:
        return datasets[0]

    search_2d_fields = (
        "search_candidate_indices",
        "search_prior_scores",
        "search_scores",
        "search_outcomes",
    )
    max_search_count = max(int(dataset["search_candidate_indices"].shape[1]) for dataset in datasets)
    output: dict[str, list[jnp.ndarray]] = {name: [] for name in datasets[0]}
    domain_offset = 0
    for dataset in datasets:
        for name, value in dataset.items():
            if name == "domain":
                output[name].append(value + domain_offset)
            elif name in search_2d_fields and value.shape[1] < max_search_count:
                pad_width = max_search_count - value.shape[1]
                constant = 0 if name == "search_candidate_indices" else (-1 if name == "search_outcomes" else -1.0e4)
                output[name].append(jnp.pad(value, ((0, 0), (0, pad_width)), constant_values=constant))
            else:
                output[name].append(value)
        if "domain" in dataset:
            domain_offset += int(jnp.max(dataset["domain"])) + 1
    return {name: jnp.concatenate(values, axis=0) for name, values in output.items()}


def balance_strategy_dataset(dataset: dict[str, jnp.ndarray], mode: str, seed: int) -> dict[str, jnp.ndarray]:
    """Balance strategy rows to equal task strata before JAX training."""
    if mode == "none":
        return dataset
    if mode not in BALANCE_STRATA_MODES:
        raise ValueError(f"unknown balance mode: {mode}")
    include_domain = mode in ("size-seat-domain", "size-seat-domain-oversample")
    oversample = mode in ("size-seat-oversample", "size-seat-domain-oversample")
    grid_size = np.asarray(dataset["grid_size"])
    seat = np.asarray(dataset["seat"])
    domain = np.asarray(dataset["domain"])
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    for size in sorted(np.unique(grid_size)):
        for player in sorted(np.unique(seat)):
            if include_domain:
                for domain_id in sorted(np.unique(domain)):
                    indices = np.flatnonzero((grid_size == size) & (seat == player) & (domain == domain_id))
                    if indices.size > 0:
                        groups.append(indices)
            else:
                indices = np.flatnonzero((grid_size == size) & (seat == player))
                if indices.size > 0:
                    groups.append(indices)
    if not groups:
        return dataset
    target_count = max(group.size for group in groups) if oversample else min(group.size for group in groups)
    selected = np.concatenate(
        [rng.choice(group, size=target_count, replace=oversample and group.size < target_count) for group in groups],
        axis=0,
    )
    rng.shuffle(selected)
    return {name: value[selected] for name, value in dataset.items()}


def mask_strategy_supervised_grads(
    grads,
    keep_outcome: bool,
    keep_value_bottleneck: bool = False,
    keep_value_heads: bool = False,
    keep_policy_head: bool = False,
):
    """Keep gradients only for selected supervised heads and optional pooled bottleneck."""
    masked = jax.tree.map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, grads)
    if keep_policy_head:
        masked = eqx.tree_at(lambda net: net.policy_conv, masked, grads.policy_conv)
        masked = eqx.tree_at(lambda net: net.pass_linear, masked, grads.pass_linear)
    if keep_value_bottleneck or keep_value_heads:
        masked = eqx.tree_at(lambda net: net.value_linear1, masked, grads.value_linear1)
    if keep_value_heads:
        masked = eqx.tree_at(lambda net: net.value_linear2, masked, grads.value_linear2)
        if grads.categorical_value_linear2 is not None:
            masked = eqx.tree_at(lambda net: net.categorical_value_linear2, masked, grads.categorical_value_linear2)
        if grads.size_value_linear1:
            masked = eqx.tree_at(lambda net: net.size_value_linear1, masked, grads.size_value_linear1)
        if grads.size_value_linear2:
            masked = eqx.tree_at(lambda net: net.size_value_linear2, masked, grads.size_value_linear2)
        if grads.size_categorical_value_linear2:
            masked = eqx.tree_at(
                lambda net: net.size_categorical_value_linear2,
                masked,
                grads.size_categorical_value_linear2,
            )
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
    if grads.strategy_source_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_source_conv, masked, grads.strategy_source_conv)
    if grads.strategy_target_conv is not None:
        masked = eqx.tree_at(lambda net: net.strategy_target_conv, masked, grads.strategy_target_conv)
    return masked


def spatial_ce_metrics(logits: jnp.ndarray, targets: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cross-entropy and argmax accuracy for one-hot or soft spatial heatmap targets."""
    active_f = active.astype(jnp.float32)
    target_mass = jnp.sum(targets * active_f, axis=(1, 2))
    valid = target_mass > 1.0e-6
    target_probs = targets * active_f / jnp.maximum(target_mass[:, None, None], 1.0e-6)
    masked_logits = jnp.where(active, logits, -1.0e9).reshape(logits.shape[0], -1)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1).reshape(logits.shape)
    per_sample_loss = -jnp.sum(target_probs * log_probs, axis=(1, 2))
    normalizer = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    loss = jnp.sum(per_sample_loss * valid.astype(jnp.float32)) / normalizer

    predicted = jnp.argmax(masked_logits, axis=-1)
    target_index = jnp.argmax(target_probs.reshape(targets.shape[0], -1), axis=-1)
    accuracy = jnp.sum((predicted == target_index).astype(jnp.float32) * valid.astype(jnp.float32)) / normalizer
    return loss, accuracy


def search_q_rank_metrics(
    action_q_values: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    prior_scores: jnp.ndarray,
    search_scores: jnp.ndarray,
    score_gaps: jnp.ndarray,
    temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit candidate action values or policy logits to search top-k score rankings."""
    action_count = action_q_values.shape[1]
    valid = (candidate_indices >= 0) & (candidate_indices < action_count) & (prior_scores > -9999.0)
    valid_count = jnp.sum(valid.astype(jnp.float32), axis=1)
    sample_weight = ((valid_count > 1.0) & (score_gaps > 0.0)).astype(jnp.float32)
    safe_indices = jnp.clip(candidate_indices, 0, action_count - 1)
    candidate_q = jnp.take_along_axis(action_q_values, safe_indices, axis=1)
    target_logits = jnp.where(valid, search_scores / temperature, -1.0e9)
    target_log_probs = jax.nn.log_softmax(target_logits, axis=1)
    target_probs = jnp.exp(target_log_probs)
    q_log_probs = jax.nn.log_softmax(jnp.where(valid, candidate_q, -1.0e9), axis=1)
    per_sample_loss = -jnp.sum(target_probs * q_log_probs, axis=1)
    normalizer = jnp.maximum(jnp.sum(sample_weight), 1.0)
    loss = jnp.sum(per_sample_loss * sample_weight) / normalizer
    pred_best = jnp.argmax(jnp.where(valid, candidate_q, -1.0e9), axis=1)
    target_best = jnp.argmax(target_logits, axis=1)
    accuracy = jnp.sum((pred_best == target_best).astype(jnp.float32) * sample_weight) / normalizer
    entropy = -jnp.sum(target_probs * jnp.log(jnp.clip(target_probs, 1.0e-8, 1.0)), axis=1)
    entropy = jnp.sum(entropy * sample_weight) / normalizer
    weight_mean = jnp.mean(sample_weight)
    return loss, accuracy, entropy, weight_mean


def search_q_value_metrics(
    action_q_values: jnp.ndarray,
    candidate_indices: jnp.ndarray,
    prior_scores: jnp.ndarray,
    search_scores: jnp.ndarray,
    search_outcomes: jnp.ndarray,
    score_gaps: jnp.ndarray,
    score_scale: float,
    outcome_score_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Regress candidate action-Q values to search outcome values."""
    action_count = action_q_values.shape[1]
    valid = (
        (candidate_indices >= 0)
        & (candidate_indices < action_count)
        & (prior_scores > -9999.0)
        & (search_outcomes >= OUTCOME_LOSS)
        & (search_outcomes <= OUTCOME_WIN)
    )
    valid_count = jnp.sum(valid.astype(jnp.float32), axis=1)
    sample_weight = ((valid_count > 0.0) & (score_gaps > 0.0)).astype(jnp.float32)
    safe_indices = jnp.clip(candidate_indices, 0, action_count - 1)
    candidate_q = jnp.take_along_axis(action_q_values, safe_indices, axis=1)
    outcome_targets = search_outcomes.astype(jnp.float32) - float(OUTCOME_DRAW)
    score_targets = jnp.tanh(search_scores / score_scale)
    targets = outcome_targets + outcome_score_weight * score_targets
    squared_error = jnp.square(candidate_q - targets)
    per_sample_loss = jnp.sum(jnp.where(valid, squared_error, 0.0), axis=1) / jnp.maximum(valid_count, 1.0)
    normalizer = jnp.maximum(jnp.sum(sample_weight), 1.0)
    loss = jnp.sum(per_sample_loss * sample_weight) / normalizer
    pred_best = jnp.argmax(jnp.where(valid, candidate_q, -1.0e9), axis=1)
    target_best = jnp.argmax(jnp.where(valid, targets, -1.0e9), axis=1)
    accuracy = jnp.sum((pred_best == target_best).astype(jnp.float32) * sample_weight) / normalizer
    return loss, accuracy, jnp.mean(sample_weight)


def binary_balance_weights(labels: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Return per-label weights that give positive and negative labels equal mass."""
    labels_f = labels.astype(jnp.float32)
    weights_f = weights.astype(jnp.float32)
    positives = jnp.sum(weights_f * labels_f)
    negatives = jnp.sum(weights_f * (1.0 - labels_f))
    total = positives + negatives
    positive_scale = total / jnp.maximum(2.0 * positives, 1.0)
    negative_scale = total / jnp.maximum(2.0 * negatives, 1.0)
    return jnp.where(labels_f > 0.5, positive_scale, negative_scale)


def class_balance_weights(targets: jnp.ndarray, weights: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Return inverse-frequency per-sample class weights for present classes."""
    weights_f = weights.astype(jnp.float32)
    class_ids = jnp.arange(num_classes)
    counts = jnp.sum((targets[:, None] == class_ids[None, :]).astype(jnp.float32) * weights_f[:, None], axis=0)
    present = counts > 0.0
    present_count = jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)
    total = jnp.sum(counts)
    scales = jnp.where(present, total / jnp.maximum(present_count * counts, 1.0), 0.0)
    return scales[targets]


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
    value_target_weight: float,
    policy_kl_weight: float,
    action_ce_weight: float,
    search_policy_rank_weight: float,
    prefix_pairwise_margin_weight: float,
    prefix_pairwise_margin: float,
    q_kl_weight: float,
    q_action_ce_weight: float,
    search_q_rank_weight: float,
    search_q_temperature: float,
    search_q_value_weight: float,
    search_q_score_scale: float,
    search_q_outcome_score_weight: float,
    source_weight: float,
    target_weight: float,
    balance_finish_labels: bool,
    balance_outcome_labels: bool,
    freeze_base: bool,
    train_value_bottleneck: bool,
    train_value_heads: bool,
    train_policy_head: bool,
    multi_horizon_finish: bool,
):
    """Train one minibatch of frozen-trunk strategy auxiliary losses."""
    (
        obs,
        masks,
        active,
        intent_targets,
        finish_targets,
        finish_weights,
        outcome_targets,
        outcome_weights,
        enemy_general,
        source_heatmap,
        target_heatmap,
        teacher_logits,
        teacher_actions,
        action_weights,
        prefix_weights,
        prefix_base_actions,
        search_candidate_indices,
        search_prior_scores,
        search_scores,
        search_outcomes,
        search_score_gaps,
    ) = batch

    def loss_fn(net):
        teacher_legal = teacher_logits > -9999.0

        outputs = None
        intent_loss = jnp.asarray(0.0, dtype=jnp.float32)
        intent_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        finish_loss = jnp.asarray(0.0, dtype=jnp.float32)
        finish_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        belief_loss = jnp.asarray(0.0, dtype=jnp.float32)
        if net.strategy_aux:
            outputs = jax.vmap(lambda o, m, a: net.strategy_auxiliary(o, m, a))(obs, masks, active)

            intent_log_probs = jax.nn.log_softmax(outputs.intent_logits, axis=-1)
            intent_losses = -intent_log_probs[jnp.arange(intent_log_probs.shape[0]), intent_targets]
            intent_loss = jnp.mean(intent_losses)
            intent_accuracy = jnp.mean((jnp.argmax(outputs.intent_logits, axis=-1) == intent_targets).astype(jnp.float32))

            finish_normalizer = jnp.maximum(jnp.sum(finish_weights), 1.0)
            if multi_horizon_finish:
                finish_losses = binary_cross_entropy_with_logits(outputs.finish_logits, finish_targets)
                finish_label_weights = jnp.where(
                    balance_finish_labels,
                    binary_balance_weights(finish_targets, finish_weights[:, None]),
                    1.0,
                )
                weighted_finish = finish_losses * finish_weights[:, None] * finish_label_weights
                finish_loss = jnp.sum(weighted_finish)
                finish_loss = finish_loss / jnp.maximum(jnp.sum(finish_weights[:, None] * finish_label_weights), 1.0)
                finish_predictions = (jax.nn.sigmoid(outputs.finish_logits) >= 0.5).astype(jnp.float32)
                finish_accuracy = jnp.sum(
                    (finish_predictions == finish_targets).astype(jnp.float32) * finish_weights[:, None]
                )
                finish_accuracy = finish_accuracy / jnp.maximum(finish_normalizer * finish_targets.shape[-1], 1.0)
            else:
                finish_log_probs = jax.nn.log_softmax(outputs.finish_logits, axis=-1)
                finish_losses = -finish_log_probs[jnp.arange(finish_log_probs.shape[0]), finish_targets]
                finish_label_weights = jnp.where(
                    balance_finish_labels,
                    binary_balance_weights(finish_targets, finish_weights),
                    1.0,
                )
                finish_loss = jnp.sum(finish_losses * finish_weights * finish_label_weights)
                finish_loss = finish_loss / jnp.maximum(jnp.sum(finish_weights * finish_label_weights), 1.0)
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
            outcome_label_weights = jnp.where(
                balance_outcome_labels,
                class_balance_weights(outcome_targets, outcome_weights, 3),
                1.0,
            )
            outcome_loss = jnp.sum(outcome_losses * outcome_weights * outcome_label_weights)
            outcome_loss = outcome_loss / jnp.maximum(jnp.sum(outcome_weights * outcome_label_weights), 1.0)
            outcome_accuracy = jnp.sum(
                (jnp.argmax(outcome_logits, axis=-1) == outcome_targets).astype(jnp.float32) * outcome_weights
            )
            outcome_accuracy = outcome_accuracy / outcome_normalizer

        value_target_loss = jnp.asarray(0.0, dtype=jnp.float32)
        value_target_mae = jnp.asarray(0.0, dtype=jnp.float32)
        if value_target_weight > 0.0:
            _, values, value_logits, _ = jax.vmap(lambda o, m, a: net.logits_value_auxiliary(o, m, a))(
                obs,
                masks,
                active,
            )
            value_targets = outcome_targets.astype(jnp.float32) - float(OUTCOME_DRAW)
            if value_logits is not None:
                value_losses = hl_gauss_value_loss(
                    value_logits,
                    value_targets,
                    net.value_bins,
                    net.value_min,
                    net.value_max,
                    net.value_sigma,
                )
            else:
                value_losses = 0.5 * jnp.square(values - value_targets)
            value_normalizer = jnp.maximum(jnp.sum(outcome_weights), 1.0)
            value_target_loss = jnp.sum(value_losses * outcome_weights) / value_normalizer
            value_target_mae = jnp.sum(jnp.abs(values - value_targets) * outcome_weights) / value_normalizer

        policy_kl = jnp.asarray(0.0, dtype=jnp.float32)
        action_ce = jnp.asarray(0.0, dtype=jnp.float32)
        teacher_action_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        prefix_pairwise_margin_loss = jnp.asarray(0.0, dtype=jnp.float32)
        prefix_pairwise_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        search_policy_rank_loss = jnp.asarray(0.0, dtype=jnp.float32)
        search_policy_rank_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        search_policy_target_entropy = jnp.asarray(0.0, dtype=jnp.float32)
        search_policy_weight_mean = jnp.asarray(0.0, dtype=jnp.float32)
        if (
            policy_kl_weight > 0.0
            or action_ce_weight > 0.0
            or prefix_pairwise_margin_weight > 0.0
            or search_policy_rank_weight > 0.0
        ):
            student_logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs, masks, active)
            masked_teacher_logits = jnp.where(teacher_legal, teacher_logits, -1.0e9)
            teacher_log_probs = jax.nn.log_softmax(masked_teacher_logits, axis=-1)
            teacher_probs = jnp.exp(teacher_log_probs)
            student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
            policy_kl_per_sample = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
            policy_kl = jnp.mean(policy_kl_per_sample)

            action_ce_losses = -student_log_probs[jnp.arange(student_log_probs.shape[0]), teacher_actions]
            action_normalizer = jnp.maximum(jnp.sum(action_weights), 1.0)
            action_ce = jnp.sum(action_ce_losses * action_weights) / action_normalizer
            teacher_action_accuracy = jnp.sum(
                (jnp.argmax(student_logits, axis=-1) == teacher_actions).astype(jnp.float32) * action_weights
            ) / action_normalizer
            if prefix_pairwise_margin_weight > 0.0:
                action_count = student_logits.shape[-1]
                safe_teacher_actions = jnp.clip(teacher_actions, 0, action_count - 1)
                safe_base_actions = jnp.clip(prefix_base_actions, 0, action_count - 1)
                accepted_logits = student_logits[jnp.arange(student_logits.shape[0]), safe_teacher_actions]
                base_logits = student_logits[jnp.arange(student_logits.shape[0]), safe_base_actions]
                pair_valid = (safe_teacher_actions != safe_base_actions).astype(jnp.float32)
                pair_weights = prefix_weights * pair_valid
                pair_losses = jax.nn.softplus(prefix_pairwise_margin - (accepted_logits - base_logits))
                pair_normalizer = jnp.maximum(jnp.sum(pair_weights), 1.0)
                prefix_pairwise_margin_loss = jnp.sum(pair_losses * pair_weights) / pair_normalizer
                prefix_pairwise_accuracy = jnp.sum(
                    ((accepted_logits > base_logits).astype(jnp.float32)) * pair_weights
                ) / pair_normalizer
            if search_policy_rank_weight > 0.0:
                (
                    search_policy_rank_loss,
                    search_policy_rank_accuracy,
                    search_policy_target_entropy,
                    search_policy_weight_mean,
                ) = search_q_rank_metrics(
                    student_logits,
                    search_candidate_indices,
                    search_prior_scores,
                    search_scores,
                    search_score_gaps,
                    search_q_temperature,
                )

        q_policy_kl = jnp.asarray(0.0, dtype=jnp.float32)
        q_action_ce = jnp.asarray(0.0, dtype=jnp.float32)
        q_action_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if q_kl_weight > 0.0 or q_action_ce_weight > 0.0:
            masked_teacher_logits = jnp.where(teacher_legal, teacher_logits, -1.0e9)
            teacher_log_probs = jax.nn.log_softmax(masked_teacher_logits, axis=-1)
            teacher_probs = jnp.exp(teacher_log_probs)
            q_logits = jnp.where(teacher_legal, outputs.action_q_values, -1.0e9)
            q_log_probs = jax.nn.log_softmax(q_logits, axis=-1)
            q_policy_kl_per_sample = jnp.sum(teacher_probs * (teacher_log_probs - q_log_probs), axis=-1)
            q_policy_kl = jnp.mean(q_policy_kl_per_sample)

            q_action_ce_losses = -q_log_probs[jnp.arange(q_log_probs.shape[0]), teacher_actions]
            action_normalizer = jnp.maximum(jnp.sum(action_weights), 1.0)
            q_action_ce = jnp.sum(q_action_ce_losses * action_weights) / action_normalizer
            q_action_accuracy = jnp.sum(
                (jnp.argmax(q_logits, axis=-1) == teacher_actions).astype(jnp.float32) * action_weights
            ) / action_normalizer

        search_q_rank_loss = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_rank_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_target_entropy = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_weight_mean = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_value_loss = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_value_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        search_q_value_weight_mean = jnp.asarray(0.0, dtype=jnp.float32)
        if search_q_rank_weight > 0.0:
            (
                search_q_rank_loss,
                search_q_rank_accuracy,
                search_q_target_entropy,
                search_q_weight_mean,
            ) = search_q_rank_metrics(
                outputs.action_q_values,
                search_candidate_indices,
                search_prior_scores,
                search_scores,
                search_score_gaps,
                search_q_temperature,
            )
        if search_q_value_weight > 0.0:
            (
                search_q_value_loss,
                search_q_value_accuracy,
                search_q_value_weight_mean,
            ) = search_q_value_metrics(
                outputs.action_q_values,
                search_candidate_indices,
                search_prior_scores,
                search_scores,
                search_outcomes,
                search_score_gaps,
                search_q_score_scale,
                search_q_outcome_score_weight,
            )

        source_loss = jnp.asarray(0.0, dtype=jnp.float32)
        source_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        target_loss = jnp.asarray(0.0, dtype=jnp.float32)
        target_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        if source_weight > 0.0 or target_weight > 0.0:
            if outputs.source_logits is None or outputs.target_logits is None:
                raise ValueError("source/target losses require strategy_spatial_aux")
            source_loss, source_accuracy = spatial_ce_metrics(outputs.source_logits, source_heatmap, active)
            target_loss, target_accuracy = spatial_ce_metrics(outputs.target_logits, target_heatmap, active)

        loss = (
            intent_weight * intent_loss
            + finish_weight * finish_loss
            + belief_weight * belief_loss
            + outcome_weight * outcome_loss
            + value_target_weight * value_target_loss
            + policy_kl_weight * policy_kl
            + action_ce_weight * action_ce
            + search_policy_rank_weight * search_policy_rank_loss
            + prefix_pairwise_margin_weight * prefix_pairwise_margin_loss
            + q_kl_weight * q_policy_kl
            + q_action_ce_weight * q_action_ce
            + search_q_rank_weight * search_q_rank_loss
            + search_q_value_weight * search_q_value_loss
            + source_weight * source_loss
            + target_weight * target_loss
        )
        metrics = {
            "intent_loss": intent_loss,
            "finish_loss": finish_loss,
            "belief_loss": belief_loss,
            "outcome_loss": outcome_loss,
            "value_target_loss": value_target_loss,
            "policy_kl": policy_kl,
            "action_ce": action_ce,
            "search_policy_rank_loss": search_policy_rank_loss,
            "prefix_pairwise_margin_loss": prefix_pairwise_margin_loss,
            "q_policy_kl": q_policy_kl,
            "q_action_ce": q_action_ce,
            "search_q_rank_loss": search_q_rank_loss,
            "search_q_value_loss": search_q_value_loss,
            "source_loss": source_loss,
            "target_loss": target_loss,
            "intent_accuracy": intent_accuracy,
            "finish_accuracy": finish_accuracy,
            "outcome_accuracy": outcome_accuracy,
            "value_target_mae": value_target_mae,
            "teacher_action_accuracy": teacher_action_accuracy,
            "search_policy_rank_accuracy": search_policy_rank_accuracy,
            "prefix_pairwise_accuracy": prefix_pairwise_accuracy,
            "q_action_accuracy": q_action_accuracy,
            "search_q_rank_accuracy": search_q_rank_accuracy,
            "search_q_value_accuracy": search_q_value_accuracy,
            "search_q_target_entropy": search_q_target_entropy,
            "search_policy_target_entropy": search_policy_target_entropy,
            "source_accuracy": source_accuracy,
            "target_accuracy": target_accuracy,
            "finish_weight_mean": jnp.mean(finish_weights),
            "outcome_weight_mean": jnp.mean(outcome_weights),
            "value_target_weight_mean": jnp.mean(outcome_weights),
            "action_weight_mean": jnp.mean(action_weights),
            "search_policy_weight_mean": search_policy_weight_mean,
            "prefix_weight_mean": jnp.mean(prefix_weights),
            "search_q_weight_mean": search_q_weight_mean,
            "search_q_value_weight_mean": search_q_value_weight_mean,
        }
        return loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    if freeze_base:
        grads = mask_strategy_supervised_grads(
            grads,
            outcome_weight > 0.0,
            train_value_bottleneck,
            train_value_heads,
            train_policy_head,
        )
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
    value_target_weight: float,
    policy_kl_weight: float,
    action_ce_weight: float,
    search_policy_rank_weight: float,
    prefix_pairwise_margin_weight: float,
    prefix_pairwise_margin: float,
    q_kl_weight: float,
    q_action_ce_weight: float,
    search_q_rank_weight: float,
    search_q_temperature: float,
    search_q_value_weight: float,
    search_q_score_scale: float,
    search_q_outcome_score_weight: float,
    source_weight: float,
    target_weight: float,
    balance_finish_labels: bool,
    balance_outcome_labels: bool,
    freeze_base: bool,
    train_value_bottleneck: bool,
    train_value_heads: bool,
    train_policy_head: bool,
    multi_horizon_finish: bool,
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
            dataset["source_heatmap"][idx],
            dataset["target_heatmap"][idx],
            dataset["teacher_logits"][idx],
            dataset["teacher_action"][idx],
            dataset["action_weight"][idx],
            dataset["prefix_weight"][idx],
            dataset["prefix_base_action"][idx],
            dataset["search_candidate_indices"][idx],
            dataset["search_prior_scores"][idx],
            dataset["search_scores"][idx],
            dataset["search_outcomes"][idx],
            dataset["search_score_gap"][idx],
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
            value_target_weight,
            policy_kl_weight,
            action_ce_weight,
            search_policy_rank_weight,
            prefix_pairwise_margin_weight,
            prefix_pairwise_margin,
            q_kl_weight,
            q_action_ce_weight,
            search_q_rank_weight,
            search_q_temperature,
            search_q_value_weight,
            search_q_score_scale,
            search_q_outcome_score_weight,
            source_weight,
            target_weight,
            balance_finish_labels,
            balance_outcome_labels,
            freeze_base,
            train_value_bottleneck,
            train_value_heads,
            train_policy_head,
            multi_horizon_finish,
        )
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return network, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Train adaptive strategy auxiliary heads from NPZ shards.")
    parser.add_argument("--dataset", action="append", required=True, help="NPZ shard path or glob. Repeatable.")
    parser.add_argument(
        "--dataset-format",
        choices=DATASET_FORMAT_MODES,
        default="strategy",
        help="Read ordinary strategy shards, Plan-Q best-command prefix rows, or online-search trace rows.",
    )
    parser.add_argument(
        "--extra-plan-q-prefix-dataset",
        action="append",
        default=[],
        help="Additional Plan-Q prefix NPZ shard path or glob to mix with the main dataset. Repeatable.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--min-row-turn", type=int, default=0)
    parser.add_argument("--max-row-turn", type=int, default=None)
    parser.add_argument("--require-contact", action="store_true")
    parser.add_argument("--min-visible-enemy-cells", type=int, default=0)
    parser.add_argument("--min-visible-enemy-density", type=float, default=0.0)
    parser.add_argument("--require-outcome-win", action="store_true")
    parser.add_argument("--require-outcome-draw", action="store_true")
    parser.add_argument("--require-outcome-nonwin", action="store_true")
    parser.add_argument("--require-search-best-win", action="store_true")
    parser.add_argument("--require-search-used", action="store_true")
    parser.add_argument("--require-search-action-changed", action="store_true")
    parser.add_argument("--require-search-improves-continuation", action="store_true")
    parser.add_argument("--require-search-converts-to-win", action="store_true")
    parser.add_argument("--require-finish-within-250", action="store_true")
    parser.add_argument("--require-win-or-finish-within-250", action="store_true")
    parser.add_argument("--min-search-score-gap", type=float, default=0.0)
    parser.add_argument(
        "--min-teacher-action-logit-margin",
        type=float,
        default=None,
        help="For Plan-Q prefix rows, keep only labels whose saved teacher logit is within this margin of top.",
    )
    parser.add_argument("--balance-strata", choices=BALANCE_STRATA_MODES, default="none")
    parser.add_argument(
        "--label-source",
        choices=LABEL_SOURCE_MODES,
        default="trajectory",
        help=(
            "Use trajectory labels, rollout-search best-action labels, or search labels "
            "with trajectory fallback for finish/outcome/value heads."
        ),
    )
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
    parser.add_argument(
        "--strategy-aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strategy auxiliary heads on the trained model. Use --no-strategy-aux for pure policy adapters.",
    )
    parser.add_argument("--init-strategy-aux", action="store_true")
    parser.add_argument("--finish-head-mode", choices=("binary", "multi-horizon"), default="binary")
    parser.add_argument("--init-finish-head-mode", choices=("binary", "multi-horizon"), default="binary")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-strategy-spatial-aux", action="store_true")
    parser.add_argument(
        "--drop-mismatched-init-leaves",
        action="store_true",
        help="Load matching checkpoint leaves and reinitialize shape-mismatched legacy leaves.",
    )
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--model-path", default="runs/adaptive-strategy-supervised/generals-adaptive-strategy-supervised.eqx")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--update-scope",
        choices=("strategy-heads", "strategy-value-heads", "policy-heads", "all"),
        default="strategy-heads",
    )
    parser.add_argument("--intent-weight", type=float, default=0.2)
    parser.add_argument("--finish-weight", type=float, default=0.4)
    parser.add_argument("--belief-weight", type=float, default=0.3)
    parser.add_argument("--outcome-weight", type=float, default=0.0)
    parser.add_argument(
        "--value-target-weight",
        type=float,
        default=0.0,
        help="Fit the PPO value head to loss/draw/win targets from the selected label source.",
    )
    parser.add_argument("--balance-finish-labels", action="store_true")
    parser.add_argument("--balance-outcome-labels", action="store_true")
    parser.add_argument("--policy-kl-weight", type=float, default=0.0)
    parser.add_argument("--action-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--search-policy-rank-weight",
        type=float,
        default=0.0,
        help="Fit policy logits directly to rollout-search top-k soft score rankings.",
    )
    parser.add_argument(
        "--prefix-pairwise-margin-weight",
        type=float,
        default=0.0,
        help="For Plan-Q prefix rows, rank accepted executed-prefix action above base-policy action.",
    )
    parser.add_argument("--prefix-pairwise-margin", type=float, default=1.0)
    parser.add_argument(
        "--prefix-advantage-weighting",
        action="store_true",
        help="Scale Plan-Q prefix imitation weights by clipped plan_advantage / 0.5.",
    )
    parser.add_argument(
        "--prefix-step-decay",
        type=float,
        default=0.0,
        help="Half-life in prefix steps for weighting earlier executed-prefix actions; 0 disables.",
    )
    parser.add_argument(
        "--prefix-negative-weight",
        type=float,
        default=0.0,
        help="Reserved for rejected-prefix labels; currently accepted for command compatibility and kept at zero effect.",
    )
    parser.add_argument(
        "--require-plan-advantage",
        type=float,
        default=0.0,
        help="For Plan-Q prefix rows, keep only rows with saved plan_advantage at least this value.",
    )
    parser.add_argument("--action-ce-weight-mode", choices=ACTION_CE_WEIGHT_MODES, default="all")
    parser.add_argument(
        "--action-ce-path-contains",
        action="append",
        default=[],
        help="Only shards whose path contains this token contribute action CE. Repeatable.",
    )
    parser.add_argument("--q-kl-weight", type=float, default=0.0)
    parser.add_argument("--q-action-ce-weight", type=float, default=0.0)
    parser.add_argument("--search-q-rank-weight", type=float, default=0.0)
    parser.add_argument(
        "--search-q-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for rollout-search top-k score targets.",
    )
    parser.add_argument(
        "--search-q-value-weight",
        type=float,
        default=0.0,
        help="MSE weight for fitting strategy-Q values to rollout-search candidate outcomes.",
    )
    parser.add_argument(
        "--search-q-score-scale",
        type=float,
        default=1000.0,
        help="Scale for optional tanh(search_score / scale) tie-break in search-Q value targets.",
    )
    parser.add_argument(
        "--search-q-outcome-score-weight",
        type=float,
        default=0.0,
        help="Optional shaped-score tie-break weight added to loss/draw/win outcome targets.",
    )
    parser.add_argument("--source-weight", type=float, default=0.0)
    parser.add_argument("--target-weight", type=float, default=0.0)
    parser.add_argument(
        "--keep-pass-prefix-labels",
        action="store_true",
        help="For --dataset-format plan-q-prefix, keep pass labels instead of dropping them.",
    )
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
    if args.min_row_turn < 0:
        parser.error("--min-row-turn must be non-negative")
    if args.max_row_turn is not None and args.max_row_turn < args.min_row_turn:
        parser.error("--max-row-turn must be greater than or equal to --min-row-turn")
    if args.min_visible_enemy_cells < 0:
        parser.error("--min-visible-enemy-cells must be non-negative")
    if not (0.0 <= args.min_visible_enemy_density <= 1.0):
        parser.error("--min-visible-enemy-density must be between 0 and 1")
    if args.min_search_score_gap < 0.0:
        parser.error("--min-search-score-gap must be non-negative")
    if args.require_outcome_win and args.require_win_or_finish_within_250:
        parser.error("--require-outcome-win conflicts with --require-win-or-finish-within-250")
    if args.require_outcome_win and (args.require_outcome_draw or args.require_outcome_nonwin):
        parser.error("--require-outcome-win conflicts with draw/non-win outcome filters")
    if args.require_outcome_draw and args.require_win_or_finish_within_250:
        parser.error("--require-outcome-draw conflicts with --require-win-or-finish-within-250")
    if args.dataset_format == "plan-q-prefix":
        if args.require_search_best_win:
            parser.error("--require-search-best-win is not available for --dataset-format plan-q-prefix")
        if args.require_search_used or args.require_search_action_changed:
            parser.error("search-used/action-changed filters are not available for --dataset-format plan-q-prefix")
        if args.require_search_improves_continuation or args.require_search_converts_to_win:
            parser.error("continuation filters are not available for --dataset-format plan-q-prefix")
        if args.require_finish_within_250 or args.require_win_or_finish_within_250:
            parser.error("finish-window filters are not available for --dataset-format plan-q-prefix")
        if args.require_contact or args.min_visible_enemy_cells > 0 or args.min_visible_enemy_density > 0.0:
            parser.error("contact/visible-enemy filters are not available for --dataset-format plan-q-prefix")
        if args.label_source != "trajectory":
            parser.error("--dataset-format plan-q-prefix uses plan outcomes directly; keep --label-source trajectory")
    if args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.num_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("--num-epochs and --minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.weight_decay != 0.0 and args.update_scope != "all":
        parser.error("--weight-decay must stay 0 because this trainer freezes most parameters with a gradient mask")
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
            args.intent_weight,
            args.finish_weight,
            args.belief_weight,
            args.outcome_weight,
            args.value_target_weight,
            args.policy_kl_weight,
            args.action_ce_weight,
            args.search_policy_rank_weight,
            args.prefix_pairwise_margin_weight,
            args.prefix_negative_weight,
            args.q_kl_weight,
            args.q_action_ce_weight,
            args.search_q_rank_weight,
            args.search_q_value_weight,
            args.source_weight,
            args.target_weight,
        )
    ):
        parser.error("loss weights must be non-negative")
    if not args.strategy_aux:
        if args.strategy_spatial_aux:
            parser.error("--strategy-spatial-aux requires --strategy-aux")
        if args.update_scope in ("strategy-heads", "strategy-value-heads"):
            parser.error("--no-strategy-aux requires --update-scope policy-heads or all")
        disabled_aux_weights = (
            args.intent_weight,
            args.finish_weight,
            args.belief_weight,
            args.q_kl_weight,
            args.q_action_ce_weight,
            args.search_q_rank_weight,
            args.search_q_value_weight,
            args.source_weight,
            args.target_weight,
        )
        if any(weight > 0.0 for weight in disabled_aux_weights):
            parser.error("--no-strategy-aux requires auxiliary/Q/source/target loss weights to be 0")
    if args.search_q_temperature <= 0.0:
        parser.error("--search-q-temperature must be positive")
    if args.prefix_pairwise_margin_weight < 0.0:
        parser.error("--prefix-pairwise-margin-weight must be non-negative")
    if args.prefix_pairwise_margin < 0.0:
        parser.error("--prefix-pairwise-margin must be non-negative")
    if args.prefix_step_decay < 0.0:
        parser.error("--prefix-step-decay must be non-negative")
    if args.prefix_negative_weight < 0.0:
        parser.error("--prefix-negative-weight must be non-negative")
    if args.require_plan_advantage < 0.0:
        parser.error("--require-plan-advantage must be non-negative")
    if args.search_q_score_scale <= 0.0:
        parser.error("--search-q-score-scale must be positive")
    if args.search_q_outcome_score_weight < 0.0:
        parser.error("--search-q-outcome-score-weight must be non-negative")
    if args.outcome_weight > 0.0 and not args.outcome_head:
        parser.error("--outcome-weight requires --outcome-head")
    if args.value_target_weight > 0.0 and args.value_loss == "hl-gauss" and args.value_bins <= 1:
        parser.error("--value-target-weight with --value-loss hl-gauss requires --value-bins > 1")
    if args.update_scope in ("policy-heads", "all") and args.policy_kl_weight <= 0.0:
        parser.error("--update-scope policy-heads/all requires a positive --policy-kl-weight to anchor policy drift")
    if (args.source_weight > 0.0 or args.target_weight > 0.0) and not args.strategy_spatial_aux:
        parser.error("--source-weight/--target-weight require --strategy-spatial-aux")
    return args


def main():
    args = parse_args()
    paths = expand_dataset_paths(args.dataset)
    if args.dataset_format == "plan-q-prefix":
        dataset, load_stats = load_plan_q_prefix_strategy_dataset(
            paths,
            args.max_samples,
            args.max_samples_per_shard,
            args.seed,
            args.finish_head_mode,
            args.action_ce_weight_mode,
            tuple(args.action_ce_path_contains),
            args.min_row_turn,
            args.max_row_turn,
            args.require_outcome_win,
            args.require_outcome_draw,
            args.require_outcome_nonwin,
            args.min_search_score_gap,
            args.min_teacher_action_logit_margin,
            args.require_plan_advantage,
            args.prefix_advantage_weighting,
            args.prefix_step_decay,
            not args.keep_pass_prefix_labels,
            True,
        )
    else:
        dataset, load_stats = load_strategy_dataset(
            paths,
            args.max_samples,
            args.max_samples_per_shard,
            args.seed,
            args.finish_head_mode,
            args.action_ce_weight_mode,
            args.label_source,
            tuple(args.action_ce_path_contains),
            args.min_row_turn,
            args.max_row_turn,
            args.require_contact,
            args.min_visible_enemy_cells,
            args.min_visible_enemy_density,
            args.require_outcome_win,
            args.require_outcome_draw,
            args.require_outcome_nonwin,
            args.require_search_best_win,
            args.require_search_used,
            args.require_search_action_changed,
            args.require_search_improves_continuation,
            args.require_search_converts_to_win,
            args.require_finish_within_250,
            args.require_win_or_finish_within_250,
            args.min_search_score_gap,
            True,
        )
    extra_load_stats = []
    if args.extra_plan_q_prefix_dataset:
        extra_paths = expand_dataset_paths(args.extra_plan_q_prefix_dataset)
        extra_dataset, extra_stats = load_plan_q_prefix_strategy_dataset(
            extra_paths,
            args.max_samples,
            args.max_samples_per_shard,
            args.seed + 104729,
            args.finish_head_mode,
            args.action_ce_weight_mode,
            tuple(args.action_ce_path_contains),
            args.min_row_turn,
            args.max_row_turn,
            False,
            False,
            False,
            args.min_search_score_gap,
            args.min_teacher_action_logit_margin,
            args.require_plan_advantage,
            args.prefix_advantage_weighting,
            args.prefix_step_decay,
            not args.keep_pass_prefix_labels,
            True,
        )
        dataset = concatenate_strategy_datasets([dataset, extra_dataset])
        extra_load_stats.append({"paths": len(extra_paths), **extra_stats})
    dataset = balance_strategy_dataset(dataset, args.balance_strata, args.seed)
    if args.label_source == "search-best" and float(jnp.sum(dataset["outcome_weight"])) <= 0.0:
        raise ValueError("--label-source search-best requires shards with search_best_outcome labels")
    key = jrandom.PRNGKey(args.seed)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )
    finish_outputs = 3 if args.finish_head_mode == "multi-horizon" else 2
    init_finish_outputs = 3 if args.init_finish_head_mode == "multi-horizon" else 2

    print("Adaptive strategy supervised training")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Shards:        {len(paths)}")
    print(f"Data format:   {args.dataset_format}")
    if extra_load_stats:
        print(f"Extra prefix:  {sum(item['paths'] for item in extra_load_stats)} shards")
    print(f"Samples:       {dataset['obs'].shape[0]}")
    print(
        "Row filters:   "
        f"kept={load_stats['kept']}/{load_stats['rows']}, sampled={load_stats['sampled']}"
    )
    if load_stats["filters"]:
        labels = [item["name"] for item in load_stats["filters"]]
        print(f"Filters:       {', '.join(dict.fromkeys(labels))}")
    if args.max_samples_per_shard is not None:
        print(f"Shard cap:     {args.max_samples_per_shard}")
    if args.balance_strata != "none":
        print(f"Balance:       {args.balance_strata}")
    print(f"Network arch:  {args.network_arch}")
    print(f"Warm start:    {args.init_model_path}")
    print(f"Finish head:   {args.finish_head_mode} ({finish_outputs} logits)")
    print(f"Label source:  {args.label_source}")
    print(
        "Loss weights:  "
        f"intent={args.intent_weight:g}, finish={args.finish_weight:g}, "
        f"belief={args.belief_weight:g}, outcome={args.outcome_weight:g}, "
        f"value={args.value_target_weight:g}, "
        f"balance_finish={args.balance_finish_labels}, balance_outcome={args.balance_outcome_labels}, "
        f"policy_kl={args.policy_kl_weight:g}, action_ce={args.action_ce_weight:g}, "
        f"search_policy_rank={args.search_policy_rank_weight:g}, "
        f"action_ce_mode={args.action_ce_weight_mode}, "
        f"prefix_pair={args.prefix_pairwise_margin_weight:g}, "
        f"prefix_adv_weight={args.prefix_advantage_weighting}, prefix_step_decay={args.prefix_step_decay:g}, "
        f"q_kl={args.q_kl_weight:g}, q_action_ce={args.q_action_ce_weight:g}, "
        f"search_q_rank={args.search_q_rank_weight:g}, search_q_temp={args.search_q_temperature:g}, "
        f"search_q_value={args.search_q_value_weight:g}, search_q_score_scale={args.search_q_score_scale:g}, "
        f"search_q_outcome_score={args.search_q_outcome_score_weight:g}, "
        f"source={args.source_weight:g}, target={args.target_weight:g}"
    )
    if args.update_scope == "strategy-heads":
        scope_label = "strategy auxiliary heads" + (" + outcome head" if args.outcome_weight > 0.0 else "")
    elif args.update_scope == "strategy-value-heads":
        scope_label = "strategy auxiliary heads + pooled value bottleneck"
        if args.outcome_weight > 0.0:
            scope_label += " + outcome head"
        if args.value_target_weight > 0.0:
            scope_label += " + value heads"
    elif args.update_scope == "policy-heads":
        scope_label = "policy output head + strategy auxiliary heads"
        if args.outcome_weight > 0.0:
            scope_label += " + outcome head"
    else:
        scope_label = "all trainable network weights with policy KL anchor"
    print(f"Update scope:  {scope_label}")
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
        strategy_finish_outputs=finish_outputs,
        init_strategy_finish_outputs=init_finish_outputs,
        strategy_spatial_aux=args.strategy_spatial_aux,
        init_strategy_spatial_aux=args.init_strategy_spatial_aux,
        global_context=args.global_context,
        init_global_context=args.global_context,
        network_arch=args.network_arch,
        init_network_arch=args.network_arch,
        drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
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
            args.value_target_weight,
            args.policy_kl_weight,
            args.action_ce_weight,
            args.search_policy_rank_weight,
            args.prefix_pairwise_margin_weight,
            args.prefix_pairwise_margin,
            args.q_kl_weight,
            args.q_action_ce_weight,
            args.search_q_rank_weight,
            args.search_q_temperature,
            args.search_q_value_weight,
            args.search_q_score_scale,
            args.search_q_outcome_score_weight,
            args.source_weight,
            args.target_weight,
            args.balance_finish_labels,
            args.balance_outcome_labels,
            args.update_scope in ("strategy-heads", "strategy-value-heads", "policy-heads"),
            args.update_scope == "strategy-value-heads",
            args.value_target_weight > 0.0,
            args.update_scope == "policy-heads",
            args.finish_head_mode == "multi-horizon",
        )
        jax.block_until_ready(network)
        print(
            f"Epoch {epoch:03d} | Loss {float(loss):.4f} | "
            f"Intent {float(metrics['intent_loss']):.4f}/{float(metrics['intent_accuracy']) * 100:5.1f}% | "
            f"Finish {float(metrics['finish_loss']):.4f}/{float(metrics['finish_accuracy']) * 100:5.1f}% | "
            f"Belief {float(metrics['belief_loss']):.4f} | "
            f"Outcome {float(metrics['outcome_loss']):.4f}/{float(metrics['outcome_accuracy']) * 100:5.1f}% | "
            f"Value {float(metrics['value_target_loss']):.4f}/MAE {float(metrics['value_target_mae']):.3f} | "
            f"KL {float(metrics['policy_kl']):.4f} | "
            f"ActCE {float(metrics['action_ce']):.4f}/{float(metrics['teacher_action_accuracy']) * 100:5.1f}% | "
            f"ActW {float(metrics['action_weight_mean']):.3f} | "
            f"SP {float(metrics['search_policy_rank_loss']):.4f}/{float(metrics['search_policy_rank_accuracy']) * 100:5.1f}% | "
            f"SPw {float(metrics['search_policy_weight_mean']):.3f} | "
            f"Pair {float(metrics['prefix_pairwise_margin_loss']):.4f}/{float(metrics['prefix_pairwise_accuracy']) * 100:5.1f}% | "
            f"PW {float(metrics['prefix_weight_mean']):.3f} | "
            f"QKL {float(metrics['q_policy_kl']):.4f} | "
            f"QCE {float(metrics['q_action_ce']):.4f}/{float(metrics['q_action_accuracy']) * 100:5.1f}% | "
            f"SQ {float(metrics['search_q_rank_loss']):.4f}/{float(metrics['search_q_rank_accuracy']) * 100:5.1f}% | "
            f"SQw {float(metrics['search_q_weight_mean']):.3f} | "
            f"SQV {float(metrics['search_q_value_loss']):.4f}/{float(metrics['search_q_value_accuracy']) * 100:5.1f}% | "
            f"SQVw {float(metrics['search_q_value_weight_mean']):.3f} | "
            f"Src {float(metrics['source_loss']):.4f}/{float(metrics['source_accuracy']) * 100:5.1f}% | "
            f"Tgt {float(metrics['target_loss']):.4f}/{float(metrics['target_accuracy']) * 100:5.1f}% | "
            f"Time {time.time() - t0:.2f}s"
        )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
