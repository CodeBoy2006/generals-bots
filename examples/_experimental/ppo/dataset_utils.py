"""Shared shard I/O and label weights for offline policy datasets."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


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


def balanced_binary_weights(labels: np.ndarray) -> np.ndarray:
    """Weight positive and negative classes equally."""
    positive = labels > 0.5
    pos_count = max(int(np.sum(positive)), 1)
    neg_count = max(int(np.sum(~positive)), 1)
    return (np.where(positive, 0.5 / pos_count, 0.5 / neg_count).astype(np.float32) * labels.shape[0])


def save_shard(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write one compressed NPZ shard and sidecar metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
