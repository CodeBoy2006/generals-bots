"""Observation-only Worker commands and action marginals."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS, active_cells_for_size, adaptive_obs_to_array
from generals.core.rewards import shortest_path_distance_map

WORKER_EXTRA_CHANNELS = 3
WORKER_INPUT_CHANNELS = ADAPTIVE_INPUT_CHANNELS + WORKER_EXTRA_CHANNELS
WORKER_COMMAND_NAMES = ("auto", "visible-general", "city", "frontier")
WORKER_COMMAND_NAME_TO_ID = {name: index for index, name in enumerate(WORKER_COMMAND_NAMES)}

PLAN_WORKER_COMMAND_SOURCE_NAMES = ("spatial", "belief-main-stack", "main-stack-heuristic")
PLAN_WORKER_COMMAND_SOURCE_TO_ID = {name: index for index, name in enumerate(PLAN_WORKER_COMMAND_SOURCE_NAMES)}


def worker_command_target_mask(obs, effective_size: int, pad_size: int, command_mode: int) -> jnp.ndarray:
    """Build a command target mask from fogged observation only."""
    active = active_cells_for_size(effective_size, pad_size)
    visible_general = obs.generals & obs.opponent_cells & active
    city_targets = ((obs.cities & ~obs.owned_cells) | obs.structures_in_fog) & active
    frontier_targets = (obs.opponent_cells | obs.neutral_cells | obs.fog_cells | obs.structures_in_fog) & active
    fallback = jnp.where(jnp.any(frontier_targets), frontier_targets, active & ~obs.owned_cells)
    general_or_frontier = jnp.where(jnp.any(visible_general), visible_general, fallback)
    city_or_frontier = jnp.where(jnp.any(city_targets), city_targets, fallback)
    auto_targets = jnp.where(jnp.any(visible_general), visible_general, city_or_frontier)
    options = jnp.stack([auto_targets, general_or_frontier, city_or_frontier, fallback], axis=0)
    return options[command_mode]


def worker_command_obs_to_array(
    obs,
    effective_size: int,
    pad_size: int,
    command_mode: int,
    min_army: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Append observation-only Worker command channels for policy execution."""
    base, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    target_mask = worker_command_target_mask(obs, effective_size, pad_size, command_mode)
    passable = active & ~obs.mountains
    distance = shortest_path_distance_map(passable, target_mask)
    eligible = obs.owned_cells & active & (obs.armies >= min_army)
    army_scale = jnp.maximum(jnp.log1p(jnp.max(obs.armies.astype(jnp.float32))), 1.0)
    source_heatmap = jnp.where(eligible, jnp.log1p(obs.armies.astype(jnp.float32)) / army_scale, 0.0)
    max_distance = jnp.maximum(jnp.asarray(effective_size * effective_size, dtype=jnp.float32), 1.0)
    route_potential = 1.0 - jnp.minimum(distance.astype(jnp.float32), max_distance) / max_distance
    route_potential = jnp.where(active, route_potential, 0.0)
    extra = jnp.stack(
        [
            target_mask.astype(jnp.float32) * active.astype(jnp.float32),
            source_heatmap,
            route_potential,
        ],
        axis=0,
    )
    return jnp.concatenate([base, extra], axis=0), active


def worker_source_direction_targets(targets: jnp.ndarray, pad_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Marginalize adaptive action targets into source-cell and direction targets."""
    leading_shape = targets.shape[:-1]
    move_planes = targets[..., : 8 * pad_size * pad_size].reshape(*leading_shape, 8, pad_size, pad_size)
    source_targets = jnp.sum(move_planes, axis=-3).reshape(*leading_shape, pad_size * pad_size)
    full_direction_targets = jnp.sum(move_planes[..., :4, :, :], axis=(-1, -2))
    half_direction_targets = jnp.sum(move_planes[..., 4:8, :, :], axis=(-1, -2))
    direction_targets = full_direction_targets + half_direction_targets
    return source_targets, direction_targets


def worker_source_direction_logits(logits: jnp.ndarray, pad_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert flat adaptive logits into source-cell and direction logits."""
    leading_shape = logits.shape[:-1]
    move_logits = logits[..., : 8 * pad_size * pad_size].reshape(*leading_shape, 8, pad_size, pad_size)
    source_logits = jax.nn.logsumexp(move_logits, axis=-3).reshape(*leading_shape, pad_size * pad_size)
    direction_logits = jax.nn.logsumexp(
        jnp.stack([move_logits[..., :4, :, :], move_logits[..., 4:8, :, :]], axis=-1),
        axis=(-1, -2, -3),
    )
    return source_logits, direction_logits


def strategy_plan_worker_obs(
    obs_arr: jnp.ndarray,
    legal_mask: jnp.ndarray,
    active: jnp.ndarray,
    source_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    pad_size: int,
) -> jnp.ndarray:
    """Append source/target command planes for a learned Plan-Worker."""
    coords = jnp.arange(pad_size)
    rows = coords[:, None]
    cols = coords[None, :]
    target_scores = jnp.where(active, target_logits, -1.0e9)
    target_index = jnp.argmax(target_scores.reshape(-1))
    target_row = target_index // pad_size
    target_col = target_index % pad_size

    movable = jnp.any(legal_mask, axis=-1)
    army_score = 0.25 * obs_arr[0]
    route_distance = jnp.abs(rows - target_row) + jnp.abs(cols - target_col)
    source_scores = source_logits + army_score - 0.05 * route_distance.astype(jnp.float32)
    source_index = jnp.argmax(jnp.where(movable, source_scores, -1.0e9).reshape(-1))
    source_row = source_index // pad_size
    source_col = source_index % pad_size

    source_plane = jnp.zeros((pad_size * pad_size,), dtype=obs_arr.dtype).at[source_index].set(1.0)
    target_plane = jnp.zeros((pad_size * pad_size,), dtype=obs_arr.dtype).at[target_index].set(1.0)
    max_distance = jnp.maximum(jnp.asarray(2 * (pad_size - 1), dtype=jnp.float32), 1.0)
    route_potential = 1.0 - jnp.minimum(route_distance.astype(jnp.float32), max_distance) / max_distance
    command = jnp.stack(
        [
            source_plane.reshape(pad_size, pad_size),
            target_plane.reshape(pad_size, pad_size),
            route_potential * active.astype(jnp.float32),
        ],
        axis=0,
    )
    del source_row, source_col  # Kept by source_index; names make the command construction easier to audit.
    return jnp.concatenate([obs_arr, command], axis=0)
