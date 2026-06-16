"""Shared helpers for adaptive multisize PPO training."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents._heuristic_logic import HEURISTIC_NAMES, heuristic_action
from generals.agents.ppo_policy_agent import obs_to_array
from generals.core import game
from generals.core.action import DIRECTIONS
from generals.core.grid import generate_grid

ADAPTIVE_INPUT_CHANNELS = 15
ADAPTIVE_MOVE_PLANES = 8


class AdaptiveStatePool(NamedTuple):
    states: game.GameState
    effective_sizes: jnp.ndarray


def parse_grid_sizes(value: str) -> tuple[int, ...]:
    """Parse comma-separated effective board sizes for adaptive training."""
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("at least one grid size is required")
    if any(size < 4 for size in sizes):
        raise ValueError("grid sizes must be at least 4")
    if len(set(sizes)) != len(sizes):
        raise ValueError("grid sizes must be unique")
    return sizes


def min_distance_for_size(size: int) -> int:
    """Return the default generated-map general spacing for one effective size."""
    return {8: 5, 12: 7, 16: 9}.get(size, max(3, size // 2))


def active_cells_for_size(effective_size: int, pad_size: int) -> jnp.ndarray:
    """Return a mask of real board cells inside a padded square canvas."""
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    return (rows < effective_size) & (cols < effective_size)


def adaptive_obs_to_array(obs, effective_size: int, pad_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert an observation to adaptive policy channels plus active-cell mask."""
    base = obs_to_array(obs)
    active = active_cells_for_size(effective_size, pad_size)
    padding = ~active
    rows = jnp.arange(pad_size, dtype=jnp.float32)[:, None]
    cols = jnp.arange(pad_size, dtype=jnp.float32)[None, :]
    denom = jnp.maximum(jnp.asarray(effective_size - 1, dtype=jnp.float32), 1.0)
    row_coord = jnp.where(active, rows / denom, 0.0)
    col_coord = jnp.where(active, cols / denom, 0.0)
    size_channel = jnp.ones((pad_size, pad_size), dtype=jnp.float32) * (effective_size / pad_size)
    area_channel = jnp.ones((pad_size, pad_size), dtype=jnp.float32) * (
        (effective_size * effective_size) / (pad_size * pad_size)
    )
    normalized_base = base.at[0].set(jnp.log1p(jnp.maximum(base[0], 0.0)))
    extra = jnp.stack(
        [
            active.astype(jnp.float32),
            padding.astype(jnp.float32),
            row_coord,
            col_coord,
            size_channel,
            area_channel,
        ],
        axis=0,
    )
    return jnp.concatenate([normalized_base, extra], axis=0), active


def compute_adaptive_valid_move_mask(
    armies: jnp.ndarray,
    owned_cells: jnp.ndarray,
    mountains: jnp.ndarray,
    effective_size: int,
    pad_size: int,
) -> jnp.ndarray:
    """Compute valid movement mask while excluding padded cells semantically."""
    active = active_cells_for_size(effective_size, pad_size)
    can_move_from = active & owned_cells & (armies > 1)
    passable = active & ~mountains
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    dest_i = rows[:, :, None] + DIRECTIONS[None, None, :, 0]
    dest_j = cols[:, :, None] + DIRECTIONS[None, None, :, 1]
    in_bounds = (dest_i >= 0) & (dest_i < effective_size) & (dest_j >= 0) & (dest_j < effective_size)
    safe_i = jnp.clip(dest_i, 0, pad_size - 1)
    safe_j = jnp.clip(dest_j, 0, pad_size - 1)
    dest_passable = passable[safe_i, safe_j]
    return can_move_from[:, :, None] & in_bounds & dest_passable


def adaptive_action_space_size(pad_size: int) -> int:
    """Return adaptive action count: eight movement planes plus one pass."""
    return ADAPTIVE_MOVE_PLANES * pad_size * pad_size + 1


def adaptive_action_to_index(action: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Encode one public action into the adaptive flattened policy index."""
    is_pass, row, col, direction, is_half = action
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    plane = direction + jnp.where(is_half > 0, 4, 0)
    move_index = plane * pad_size * pad_size + row * pad_size + col
    return jnp.where(is_pass > 0, pass_index, move_index).astype(jnp.int32)


def adaptive_index_to_action(index: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Decode an adaptive flattened policy index into the public action format."""
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    is_pass = index == pass_index
    safe_index = jnp.minimum(index, pass_index - 1)
    plane = safe_index // (pad_size * pad_size)
    position = safe_index % (pad_size * pad_size)
    row = position // pad_size
    col = position % pad_size
    direction = plane % 4
    is_half = plane >= 4
    return jnp.where(
        is_pass,
        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.array([0, row, col, direction, is_half], dtype=jnp.int32),
    )


def adaptive_action_to_target_probs(action: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Return one-hot target probabilities for one adaptive action."""
    index = adaptive_action_to_index(action, pad_size)
    return jax.nn.one_hot(index, adaptive_action_space_size(pad_size), dtype=jnp.float32)
