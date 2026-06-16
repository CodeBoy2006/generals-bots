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
ADAPTIVE_GLOBAL_INPUT_CHANNELS = 20
ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS = 5
ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS = 10
ADAPTIVE_HISTORY_INPUT_CHANNELS = ADAPTIVE_GLOBAL_INPUT_CHANNELS + ADAPTIVE_SCOREBOARD_HISTORY_CHANNELS
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


def parse_grid_size_weights(value: str | None, grid_sizes: tuple[int, ...]) -> tuple[float, ...] | None:
    """Parse size:weight pairs aligned to configured adaptive grid sizes."""
    if value is None or not value.strip():
        return None
    parsed: dict[int, float] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--grid-size-weights entries must use size:weight")
        size_text, weight_text = part.split(":", 1)
        size = int(size_text.strip())
        weight = float(weight_text.strip())
        if size in parsed:
            raise ValueError("--grid-size-weights cannot repeat a grid size")
        if weight <= 0.0:
            raise ValueError("--grid-size-weights values must be positive")
        parsed[size] = weight
    if set(parsed) != set(grid_sizes):
        raise ValueError("--grid-size-weights must specify exactly the same sizes as --grid-sizes")
    return tuple(parsed[size] for size in grid_sizes)


def min_distance_for_size(size: int) -> int:
    """Return the default generated-map general spacing for one effective size."""
    return {8: 5, 12: 7, 16: 9}.get(size, max(3, size // 2))


def active_cells_for_size(effective_size: int, pad_size: int) -> jnp.ndarray:
    """Return a mask of real board cells inside a padded square canvas."""
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    return (rows < effective_size) & (cols < effective_size)


def adaptive_scoreboard_features(obs, effective_size: int) -> jnp.ndarray:
    """Return normalized land/army/time features for one fogged observation."""
    active_area = jnp.maximum(jnp.asarray(effective_size * effective_size, dtype=jnp.float32), 1.0)
    army_scale = jnp.log1p(jnp.maximum(active_area * 100.0, 1.0))
    owned_land = jnp.asarray(obs.owned_land_count, dtype=jnp.float32)
    owned_army = jnp.asarray(obs.owned_army_count, dtype=jnp.float32)
    opponent_land = jnp.asarray(obs.opponent_land_count, dtype=jnp.float32)
    opponent_army = jnp.asarray(obs.opponent_army_count, dtype=jnp.float32)
    timestep = jnp.asarray(obs.timestep, dtype=jnp.float32)
    return jnp.stack(
        [
            owned_land / active_area,
            jnp.log1p(jnp.maximum(owned_army, 0.0)) / army_scale,
            opponent_land / active_area,
            jnp.log1p(jnp.maximum(opponent_army, 0.0)) / army_scale,
            timestep / 750.0,
        ]
    )


def adaptive_scoreboard_history_context(previous: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
    """Return previous scoreboard features plus one-step feature deltas."""
    return jnp.concatenate([previous, current - previous], axis=-1)


def reset_adaptive_scoreboard_history(current: jnp.ndarray, dones: jnp.ndarray) -> jnp.ndarray:
    """Keep current scoreboard features for continuing rows and clear finished rows."""
    return jnp.where(dones[..., None], jnp.zeros_like(current), current)


def adaptive_obs_to_array(
    obs,
    effective_size: int,
    pad_size: int,
    include_global_context: bool = False,
    scoreboard_history: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    adaptive = jnp.concatenate([normalized_base, extra], axis=0)
    if not include_global_context:
        return adaptive, active

    active_f = active.astype(jnp.float32)
    global_values = adaptive_scoreboard_features(obs, effective_size)
    if scoreboard_history is not None:
        global_values = jnp.concatenate([global_values, scoreboard_history], axis=0)
    global_planes = global_values[:, None, None] * active_f[None, :, :]
    return jnp.concatenate([adaptive, global_planes], axis=0), active


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


def make_simple_general_grid(key, grid_size: int, pad_size: int) -> jnp.ndarray:
    """Create a padded simple map with two random generals."""
    grid = jnp.full((pad_size, pad_size), -2, dtype=jnp.int32)
    grid = grid.at[:grid_size, :grid_size].set(0)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    return grid.at[pos_a].set(1).at[pos_b].set(2)


def _pool_counts(pool_size: int, grid_sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Split pool slots across sizes, assigning remainders to larger sizes."""
    base = pool_size // len(grid_sizes)
    remainder = pool_size % len(grid_sizes)
    counts = [base] * len(grid_sizes)
    for index in range(remainder):
        counts[len(counts) - 1 - index] += 1
    return tuple(counts)


def _weighted_pool_counts(pool_size: int, grid_sizes: tuple[int, ...], weights: tuple[float, ...]) -> tuple[int, ...]:
    """Split pool slots proportionally, assigning ties to larger sizes."""
    if len(weights) != len(grid_sizes):
        raise ValueError("grid size weights must align with grid sizes")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("grid size weights must be positive")

    if pool_size >= len(grid_sizes):
        counts = [1] * len(grid_sizes)
        remaining = pool_size - len(grid_sizes)
    else:
        counts = [0] * len(grid_sizes)
        remaining = pool_size

    total_weight = sum(weights)
    quotas = [remaining * weight / total_weight for weight in weights]
    floors = [int(quota) for quota in quotas]
    counts = [count + floor for count, floor in zip(counts, floors, strict=True)]
    remainder = remaining - sum(floors)
    order = sorted(
        range(len(grid_sizes)),
        key=lambda index: (quotas[index] - floors[index], grid_sizes[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def make_adaptive_state_pool(
    key,
    pool_size: int,
    grid_sizes: tuple[int, ...],
    pad_size: int,
    map_generator: str,
    mountain_density_range: tuple[float, float],
    num_cities_range: tuple[int, int],
    max_generals_distance: int | None,
    castle_val_range: tuple[int, int],
    grid_size_weights: tuple[float, ...] | None = None,
) -> AdaptiveStatePool:
    """Generate a size-balanced padded state pool for adaptive rollouts."""
    keys = jrandom.split(key, pool_size + 1)
    shuffle_key = keys[0]
    offset = 1
    pools = []
    sizes = []
    counts = (
        _pool_counts(pool_size, grid_sizes)
        if grid_size_weights is None
        else _weighted_pool_counts(pool_size, grid_sizes, grid_size_weights)
    )
    for grid_size, count in zip(grid_sizes, counts, strict=True):
        combo_keys = keys[offset : offset + count]
        offset += count
        if map_generator == "simple":
            grids = jax.vmap(lambda k, s=grid_size: make_simple_general_grid(k, s, pad_size))(combo_keys)
        else:
            grids = jax.vmap(
                lambda k, s=grid_size: generate_grid(
                    k,
                    grid_dims=(s, s),
                    pad_to=pad_size,
                    mountain_density_range=mountain_density_range,
                    num_cities_range=num_cities_range,
                    min_generals_distance=min_distance_for_size(s),
                    max_generals_distance=max_generals_distance,
                    castle_val_range=castle_val_range,
                )
            )(combo_keys)
        pools.append(jax.vmap(game.create_initial_state)(grids))
        sizes.append(jnp.full((count,), grid_size, dtype=jnp.int32))

    states = jax.tree.map(lambda *xs: jnp.concatenate(xs), *pools)
    effective_sizes = jnp.concatenate(sizes)
    permutation = jrandom.permutation(shuffle_key, pool_size)
    return AdaptiveStatePool(
        states=jax.tree.map(lambda x: x[permutation], states),
        effective_sizes=effective_sizes[permutation],
    )


def make_adaptive_initial_states(pool: AdaptiveStatePool, num_envs: int) -> tuple[game.GameState, jnp.ndarray]:
    """Take initial states and matching effective sizes from an adaptive pool."""
    states = jax.tree.map(lambda x: x[:num_envs], pool.states)
    sizes = pool.effective_sizes[:num_envs]
    pool_size = pool.states.armies.shape[0]
    pool_idx = (jnp.arange(num_envs, dtype=jnp.int32) + num_envs) % pool_size
    return states._replace(pool_idx=pool_idx), sizes


def adaptive_expander_target_probs(obs, effective_size: int, pad_size: int) -> jnp.ndarray:
    """Return a soft Expander target distribution over adaptive action indices."""
    valid_mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    target = jnp.zeros(adaptive_action_space_size(pad_size), dtype=jnp.float32)
    rows = jnp.arange(pad_size)[:, None, None]
    cols = jnp.arange(pad_size)[None, :, None]
    dest_i = jnp.clip(rows + DIRECTIONS[None, None, :, 0], 0, pad_size - 1)
    dest_j = jnp.clip(cols + DIRECTIONS[None, None, :, 1], 0, pad_size - 1)

    source_armies = obs.armies[:, :, None]
    dest_armies = obs.armies[dest_i, dest_j]
    is_opponent = obs.opponent_cells[dest_i, dest_j]
    is_neutral = obs.neutral_cells[dest_i, dest_j]
    is_owned = obs.owned_cells[dest_i, dest_j]

    can_capture = source_armies > dest_armies + 1
    is_expansion = ~is_owned & (is_opponent | is_neutral)
    scores = source_armies.astype(jnp.float32)
    scores = jnp.where(is_expansion & can_capture, scores * jnp.where(is_opponent, 20.0, 10.0), scores)
    scores = jnp.where(valid_mask & can_capture, scores, 0.0)

    score_sum = jnp.sum(scores)
    num_valid = jnp.sum(valid_mask)
    fallback = valid_mask.astype(jnp.float32) / jnp.maximum(num_valid, 1)
    move_probs = jnp.where(score_sum > 0, scores / jnp.maximum(score_sum, 1e-8), fallback)
    move_probs = jnp.where(num_valid > 0, move_probs, jnp.zeros_like(move_probs))
    full_planes = jnp.transpose(move_probs, (2, 0, 1)).reshape(4 * pad_size * pad_size)
    target = target.at[: 4 * pad_size * pad_size].set(full_planes)
    target = target.at[-1].set(jnp.where(num_valid == 0, 1.0, 0.0))
    return target
