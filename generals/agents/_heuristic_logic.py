"""JAX-compatible heuristic agent logic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.action import DIRECTIONS, compute_valid_move_mask_obs

HEURISTIC_EXPANDER = 0
HEURISTIC_CITY_RUSH = 1
HEURISTIC_GENERAL_HUNTER = 2
HEURISTIC_DEFENSIVE_EXPANDER = 3
HEURISTIC_BALANCED = 4
HEURISTIC_MIXED = 5

HEURISTIC_NAMES = (
    "expander",
    "city-rush",
    "general-hunter",
    "defensive-expander",
    "balanced",
    "mixed",
)
HEURISTIC_NAME_TO_ID = {name: idx for idx, name in enumerate(HEURISTIC_NAMES)}


def _map_scale(observation) -> jnp.ndarray:
    """Return a floating-point scale factor that grows with the map size."""
    h, w = observation.armies.shape
    return jnp.array(jnp.maximum(h, w), dtype=jnp.float32)


def _land_ratio(observation) -> jnp.ndarray:
    """Return the fraction of the map currently owned by the player."""
    total_cells = jnp.maximum(observation.armies.size, 1)
    return observation.owned_land_count.astype(jnp.float32) / total_cells


def _large_map_factor(scale: jnp.ndarray) -> jnp.ndarray:
    """Return a 0-1 factor that activates on larger maps."""
    return jnp.clip((scale - 8.0) / 8.0, 0.0, 1.0)


def _frontier_mask(owned_cells: jnp.ndarray) -> jnp.ndarray:
    """Cells on the edge of owned territory."""
    padded = jnp.pad(owned_cells, 1, mode="constant", constant_values=False)
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    return owned_cells & (~north | ~south | ~west | ~east)


def _distance_map(mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a map of Manhattan distance to the nearest True cell in mask."""
    h, w = mask.shape
    coords = jnp.argwhere(mask, size=h * w, fill_value=-1)
    valid = jnp.all(coords >= 0, axis=-1)

    rows = jnp.arange(h, dtype=jnp.int32)[:, None, None]
    cols = jnp.arange(w, dtype=jnp.int32)[None, :, None]
    target_rows = coords[:, 0]
    target_cols = coords[:, 1]

    large = jnp.array(h + w + 1, dtype=jnp.int32)
    dist = jnp.abs(rows - target_rows[None, None, :]) + jnp.abs(cols - target_cols[None, None, :])
    dist = jnp.where(valid[None, None, :], dist, large)
    return jnp.min(dist, axis=-1), jnp.any(mask)


def _valid_positions(observation) -> tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
    valid_mask = compute_valid_move_mask_obs(observation)
    h, w = observation.armies.shape
    max_moves = h * w * 4
    positions = jnp.argwhere(valid_mask, size=max_moves, fill_value=-1)
    num_valid = jnp.sum(jnp.all(positions >= 0, axis=-1))
    return positions, num_valid, h, w, max_moves


def _select_sampled_action(
    key: jnp.ndarray,
    positions: jnp.ndarray,
    scores: jnp.ndarray,
    splits: jnp.ndarray,
    num_valid: jnp.ndarray,
    max_moves: int,
) -> jnp.ndarray:
    valid_slots = jnp.arange(max_moves) < num_valid
    positive_scores = jnp.where(valid_slots, jnp.maximum(scores, 0.0), 0.0)
    score_sum = jnp.sum(positive_scores)
    fallback_slots = jnp.arange(max_moves) < jnp.maximum(num_valid, 1)
    fallback_probs = fallback_slots.astype(jnp.float32)
    probs = jnp.where(score_sum > 0.0, positive_scores, fallback_probs)
    probs = probs / (jnp.sum(probs) + 1e-8)

    move_idx = jrandom.choice(key, max_moves, p=probs)
    selected_move = positions[move_idx]
    should_pass = num_valid == 0
    selected_move = jnp.where(should_pass, jnp.array([0, 0, 0], dtype=jnp.int32), selected_move)
    selected_split = jnp.where(should_pass, jnp.int32(0), splits[move_idx].astype(jnp.int32))
    return jnp.array(
        [
            should_pass.astype(jnp.int32),
            selected_move[0],
            selected_move[1],
            selected_move[2],
            selected_split,
        ],
        dtype=jnp.int32,
    )


def _select_greedy_action(
    positions: jnp.ndarray,
    scores: jnp.ndarray,
    splits: jnp.ndarray,
    num_valid: jnp.ndarray,
    max_moves: int,
) -> jnp.ndarray:
    valid_slots = jnp.arange(max_moves) < num_valid
    ranked_scores = jnp.where(valid_slots, scores, -jnp.inf)
    move_idx = jnp.argmax(ranked_scores)
    selected_move = positions[move_idx]
    should_pass = num_valid == 0
    selected_move = jnp.where(should_pass, jnp.array([0, 0, 0], dtype=jnp.int32), selected_move)
    selected_split = jnp.where(should_pass, jnp.int32(0), splits[move_idx].astype(jnp.int32))
    return jnp.array(
        [
            should_pass.astype(jnp.int32),
            selected_move[0],
            selected_move[1],
            selected_move[2],
            selected_split,
        ],
        dtype=jnp.int32,
    )


def _move_parts(observation, move: jnp.ndarray, h: int, w: int):
    is_valid = jnp.all(move >= 0)
    src_row, src_col, direction = move[0], move[1], move[2]
    offset = DIRECTIONS[direction]
    dest_row = jnp.clip(src_row + offset[0], 0, h - 1)
    dest_col = jnp.clip(src_col + offset[1], 0, w - 1)

    source_armies = observation.armies[src_row, src_col]
    dest_armies = observation.armies[dest_row, dest_col]
    dest_owned = observation.owned_cells[dest_row, dest_col]
    dest_neutral = observation.neutral_cells[dest_row, dest_col]
    dest_opponent = observation.opponent_cells[dest_row, dest_col]
    dest_city = observation.cities[dest_row, dest_col]
    dest_general = observation.generals[dest_row, dest_col] & dest_opponent
    dest_unknown = observation.fog_cells[dest_row, dest_col] | observation.structures_in_fog[dest_row, dest_col]
    source_general = observation.generals[src_row, src_col] & observation.owned_cells[src_row, src_col]

    move_all_army = jnp.maximum(source_armies - 1, 0)
    move_half_army = source_armies // 2
    can_capture_all = move_all_army > dest_armies
    can_capture_half = move_half_army > dest_armies

    return (
        is_valid,
        src_row,
        src_col,
        dest_row,
        dest_col,
        source_armies,
        dest_armies,
        dest_owned,
        dest_neutral,
        dest_opponent,
        dest_city,
        dest_general,
        dest_unknown,
        source_general,
        can_capture_all,
        can_capture_half,
    )


def _expansion_score(
    source_armies: jnp.ndarray,
    dest_owned: jnp.ndarray,
    dest_neutral: jnp.ndarray,
    dest_opponent: jnp.ndarray,
    can_capture: jnp.ndarray,
    neutral_weight: float = 10.0,
    opponent_weight: float = 20.0,
) -> jnp.ndarray:
    """Expander-style baseline score for capturing new cells."""
    expansion_weight = jnp.where(dest_opponent, opponent_weight, neutral_weight)
    is_expansion = ~dest_owned & (dest_neutral | dest_opponent)
    return jnp.where(
        is_expansion & can_capture,
        source_armies.astype(jnp.float32) * expansion_weight,
        0.0,
    )


def _score_from_distance_map(
    distance_map: jnp.ndarray,
    has_target: jnp.ndarray,
    src_row: jnp.ndarray,
    src_col: jnp.ndarray,
    dest_row: jnp.ndarray,
    dest_col: jnp.ndarray,
    scale: jnp.ndarray,
    improvement_weight: float,
    closeness_weight: float,
) -> jnp.ndarray:
    """Score moves that reduce the distance to a global target map."""
    source_dist = distance_map[src_row, src_col]
    dest_dist = distance_map[dest_row, dest_col]
    improvement = jnp.maximum(source_dist - dest_dist, 0).astype(jnp.float32)
    normalized_dest = dest_dist.astype(jnp.float32) / jnp.maximum(scale, 1.0)
    closeness = 1.0 / (normalized_dest + 1.0)
    return jnp.where(has_target, improvement * improvement_weight + closeness * closeness_weight, 0.0)


def _distance_to_mask(mask: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return whether a mask is nonempty and the distance from one cell to it."""
    distance_map, has_target = _distance_map(mask)
    return has_target, distance_map[row, col]


def _expander_scores(observation):
    positions, num_valid, h, w, max_moves = _valid_positions(observation)
    scale = _map_scale(observation)
    large_map = _large_map_factor(scale)
    own_frontier = _frontier_mask(observation.owned_cells)
    frontier_dist, frontier_exists = _distance_map(own_frontier)

    def evaluate(idx):
        move = positions[idx]
        (
            is_valid,
            src_row,
            src_col,
            dest_row,
            dest_col,
            source_armies,
            _dest_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            _dest_city,
            _dest_general,
            _dest_unknown,
            _source_general,
            can_capture_all,
            _can_capture_half,
        ) = _move_parts(observation, move, h, w)

        is_expansion = ~dest_owned & (dest_opponent | dest_neutral)
        opponent_multiplier = jnp.where(dest_opponent, 2.0, 1.0)
        base_score = source_armies.astype(jnp.float32)
        frontier_score = _score_from_distance_map(
            frontier_dist, frontier_exists, src_row, src_col, dest_row, dest_col, scale, 4.0, 2.0
        )
        score = jnp.where(
            is_expansion & can_capture_all,
            base_score * (10.0 + 6.0 * large_map) * opponent_multiplier,
            base_score + frontier_score,
        )
        score = jnp.where(is_valid & can_capture_all, score, 0.0)
        return score, jnp.int32(0)

    scores, splits = jax.vmap(evaluate)(jnp.arange(max_moves))
    return positions, scores, splits, num_valid, max_moves


@jax.jit
def expander_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    positions, scores, splits, num_valid, max_moves = _expander_scores(observation)
    return _select_sampled_action(key, positions, scores, splits, num_valid, max_moves)


@jax.jit
def expander_greedy_action(observation) -> jnp.ndarray:
    positions, scores, splits, num_valid, max_moves = _expander_scores(observation)
    return _select_greedy_action(positions, scores, splits, num_valid, max_moves)


@jax.jit
def city_rush_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    positions, num_valid, h, w, max_moves = _valid_positions(observation)
    scale = _map_scale(observation)
    large_map = _large_map_factor(scale)
    opening = _land_ratio(observation) < (0.10 + 0.04 * large_map)
    city_targets = (observation.cities & ~observation.owned_cells) | observation.structures_in_fog
    city_dist, city_exists = _distance_map(city_targets)

    def evaluate(idx):
        move = positions[idx]
        (
            is_valid,
            src_row,
            src_col,
            dest_row,
            dest_col,
            source_armies,
            _dest_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            dest_city,
            dest_general,
            dest_unknown,
            _source_general,
            can_capture_all,
            can_capture_half,
        ) = _move_parts(observation, move, h, w)

        direct_city = dest_city & ~dest_owned
        direct_structure_fog = observation.structures_in_fog[dest_row, dest_col]
        attack_target = (dest_neutral | dest_opponent | direct_city) & ~dest_owned
        safe_attack = can_capture_all | dest_unknown
        approach = _score_from_distance_map(city_dist, city_exists, src_row, src_col, dest_row, dest_col, scale, 20.0, 12.0)
        expansion = _expansion_score(source_armies, dest_owned, dest_neutral, dest_opponent, can_capture_all)

        capture_score = expansion
        capture_score = capture_score + jnp.where(dest_neutral & can_capture_all, 55.0 + 15.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_opponent & can_capture_all, 90.0 + 20.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(
            direct_city & can_capture_all, 260.0 + source_armies.astype(jnp.float32) + 60.0 * large_map, 0.0
        )
        capture_score = capture_score + jnp.where(direct_structure_fog, 12.0 + 10.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_general & can_capture_all, 1200.0 + 120.0 * large_map, 0.0)
        capture_score = jnp.where(attack_target & ~safe_attack, capture_score * 0.05, capture_score)
        approach_score = source_armies.astype(jnp.float32) * (0.8 + 0.2 * large_map) + approach
        approach_score = jnp.where(dest_owned | dest_unknown, approach_score, approach_score * (0.25 + 0.15 * large_map))
        split = (dest_owned | (~opening & can_capture_half & direct_structure_fog & (source_armies > 14))).astype(jnp.int32)
        return jnp.where(is_valid, capture_score, 0.0), jnp.where(is_valid, approach_score, 0.0), split

    capture_scores, approach_scores, splits = jax.vmap(evaluate)(jnp.arange(max_moves))
    scores = jnp.where(jnp.sum(jnp.maximum(capture_scores, 0.0)) > 0.0, capture_scores, approach_scores)
    return _select_sampled_action(key, positions, scores, splits, num_valid, max_moves)


@jax.jit
def general_hunter_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    positions, num_valid, h, w, max_moves = _valid_positions(observation)
    scale = _map_scale(observation)
    large_map = _large_map_factor(scale)
    opening = _land_ratio(observation) < (0.10 + 0.04 * large_map)
    visible_general = observation.generals & observation.opponent_cells
    visible_enemy = observation.opponent_cells
    exploration = observation.fog_cells | observation.structures_in_fog
    has_general = jnp.any(visible_general)
    has_enemy = jnp.any(visible_enemy)
    target_mask = jnp.where(has_general, visible_general, jnp.where(has_enemy, visible_enemy, exploration))
    target_dist, target_exists = _distance_map(target_mask)

    def evaluate(idx):
        move = positions[idx]
        (
            is_valid,
            src_row,
            src_col,
            dest_row,
            dest_col,
            source_armies,
            _dest_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            dest_city,
            dest_general,
            dest_unknown,
            _source_general,
            can_capture_all,
            can_capture_half,
        ) = _move_parts(observation, move, h, w)

        approach = _score_from_distance_map(target_dist, target_exists, src_row, src_col, dest_row, dest_col, scale, 32.0, 18.0)
        attack_target = (dest_neutral | dest_opponent | (dest_city & ~dest_owned)) & ~dest_owned
        safe_attack = can_capture_all | dest_unknown
        expansion = _expansion_score(source_armies, dest_owned, dest_neutral, dest_opponent, can_capture_all)

        capture_score = expansion
        capture_score = capture_score + jnp.where(dest_general & can_capture_all, 2000.0 + 180.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_opponent & can_capture_all, 180.0 + 20.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_city & ~dest_owned & can_capture_all, 90.0 + 25.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_neutral & can_capture_all, 28.0 + 12.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_unknown, 10.0 + 6.0 * large_map, 0.0)
        capture_score = jnp.where(attack_target & ~safe_attack, capture_score * 0.08, capture_score)
        approach_score = source_armies.astype(jnp.float32) * (0.9 + 0.15 * large_map) + approach
        approach_score = jnp.where(dest_owned | dest_unknown, approach_score, approach_score * (0.25 + 0.15 * large_map))
        split = (dest_owned | (~opening & can_capture_half & dest_unknown & (source_armies > 16))).astype(jnp.int32)
        return jnp.where(is_valid, capture_score, 0.0), jnp.where(is_valid, approach_score, 0.0), split

    capture_scores, approach_scores, splits = jax.vmap(evaluate)(jnp.arange(max_moves))
    scores = jnp.where(jnp.sum(jnp.maximum(capture_scores, 0.0)) > 0.0, capture_scores, approach_scores)
    return _select_sampled_action(key, positions, scores, splits, num_valid, max_moves)


@jax.jit
def defensive_expander_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    positions, num_valid, h, w, max_moves = _valid_positions(observation)
    scale = _map_scale(observation)
    large_map = _large_map_factor(scale)
    opening = _land_ratio(observation) < (0.10 + 0.04 * large_map)
    own_general = observation.generals & observation.owned_cells
    own_general_dist, has_own_general = _distance_map(own_general)
    frontier = observation.neutral_cells | observation.opponent_cells | observation.fog_cells | observation.structures_in_fog
    frontier_dist, frontier_exists = _distance_map(frontier)

    def evaluate(idx):
        move = positions[idx]
        (
            is_valid,
            src_row,
            src_col,
            dest_row,
            dest_col,
            source_armies,
            _dest_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            dest_city,
            dest_general,
            dest_unknown,
            source_general,
            can_capture_all,
            can_capture_half,
        ) = _move_parts(observation, move, h, w)

        source_general_dist = own_general_dist[src_row, src_col]
        dest_general_dist = own_general_dist[dest_row, dest_col]
        near_general_source = has_own_general & (source_general_dist <= 2)
        moving_toward_general = has_own_general & (dest_general_dist < source_general_dist)
        moving_away_from_general = has_own_general & (dest_general_dist > source_general_dist)
        approach_frontier = _score_from_distance_map(frontier_dist, frontier_exists, src_row, src_col, dest_row, dest_col, scale, 10.0, 6.0)

        prefer_half = dest_owned | (near_general_source & dest_owned) | (
            ~opening & can_capture_half & (source_armies > 18 + 6 * large_map) & ~dest_general & dest_unknown
        )
        selected_can_capture = jnp.where(prefer_half, can_capture_half, can_capture_all)
        attack_target = (dest_neutral | dest_opponent | (dest_city & ~dest_owned)) & ~dest_owned
        expansion = _expansion_score(
            source_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            selected_can_capture,
            8.0 + 2.0 * large_map,
            16.0 + 6.0 * large_map,
        )

        capture_score = expansion
        capture_score = capture_score + jnp.where(dest_neutral & selected_can_capture, 45.0 + 10.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_opponent & selected_can_capture, 85.0 + 15.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_city & ~dest_owned & selected_can_capture, 105.0 + 20.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_general & selected_can_capture, 1500.0 + 120.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_unknown, 8.0 + 4.0 * large_map, 0.0)
        capture_score = jnp.where(attack_target & ~selected_can_capture & ~dest_unknown, capture_score * 0.05, capture_score)
        capture_score = jnp.where(source_general & ~prefer_half & (source_armies < 8 + 4 * large_map), capture_score * 0.35, capture_score)
        capture_score = jnp.where(
            near_general_source & moving_away_from_general & (observation.owned_land_count < 4 + 2 * large_map),
            capture_score * 0.65,
            capture_score,
        )
        approach_score = source_armies.astype(jnp.float32) * (0.7 + 0.15 * large_map) + approach_frontier
        approach_score = approach_score + jnp.where(dest_owned & moving_toward_general, 55.0 + 20.0 * large_map, 0.0)
        approach_score = approach_score + jnp.where(dest_owned & near_general_source, 25.0 + 10.0 * large_map, 0.0)
        approach_score = jnp.where(dest_owned | dest_unknown, approach_score, approach_score * (0.25 + 0.15 * large_map))
        split = prefer_half.astype(jnp.int32)
        return jnp.where(is_valid, capture_score, 0.0), jnp.where(is_valid, approach_score, 0.0), split

    capture_scores, approach_scores, splits = jax.vmap(evaluate)(jnp.arange(max_moves))
    scores = jnp.where(jnp.sum(jnp.maximum(capture_scores, 0.0)) > 0.0, capture_scores, approach_scores)
    return _select_sampled_action(key, positions, scores, splits, num_valid, max_moves)


@jax.jit
def balanced_strategic_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    positions, num_valid, h, w, max_moves = _valid_positions(observation)
    scale = _map_scale(observation)
    large_map = _large_map_factor(scale)
    opening = _land_ratio(observation) < (0.10 + 0.04 * large_map)
    visible_general = observation.generals & observation.opponent_cells
    city_targets = (observation.cities & ~observation.owned_cells) | observation.structures_in_fog
    enemy_targets = observation.opponent_cells
    exploration = observation.fog_cells | observation.structures_in_fog
    own_general = observation.generals & observation.owned_cells
    own_general_dist, has_own_general = _distance_map(own_general)
    city_dist, city_exists = _distance_map(city_targets)
    enemy_dist, enemy_exists = _distance_map(enemy_targets)
    explore_dist, explore_exists = _distance_map(exploration)
    general_dist, general_exists = _distance_map(visible_general)

    def evaluate(idx):
        move = positions[idx]
        (
            is_valid,
            src_row,
            src_col,
            dest_row,
            dest_col,
            source_armies,
            _dest_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            dest_city,
            dest_general,
            dest_unknown,
            source_general,
            can_capture_all,
            can_capture_half,
        ) = _move_parts(observation, move, h, w)

        general_approach = _score_from_distance_map(general_dist, general_exists, src_row, src_col, dest_row, dest_col, scale, 50.0 + 12.0 * large_map, 28.0 + 6.0 * large_map)
        city_approach = _score_from_distance_map(city_dist, city_exists, src_row, src_col, dest_row, dest_col, scale, 18.0 + 8.0 * large_map, 10.0 + 4.0 * large_map)
        enemy_approach = _score_from_distance_map(enemy_dist, enemy_exists, src_row, src_col, dest_row, dest_col, scale, 20.0 + 8.0 * large_map, 12.0 + 4.0 * large_map)
        explore_approach = _score_from_distance_map(explore_dist, explore_exists, src_row, src_col, dest_row, dest_col, scale, 6.0 + 4.0 * large_map, 4.0 + 2.0 * large_map)
        source_general_dist = own_general_dist[src_row, src_col]
        dest_general_dist = own_general_dist[dest_row, dest_col]

        army_ratio = observation.owned_army_count.astype(jnp.float32) / (
            observation.opponent_army_count.astype(jnp.float32) + 1.0
        )
        aggression = jnp.clip(army_ratio, 0.65, 1.8)
        near_general_source = has_own_general & (source_general_dist <= 2)
        prefer_half = dest_owned | (
            ~opening
            & can_capture_half
            & (source_armies > 10 + 4 * large_map)
            & ~dest_general
            & ~(dest_city & ~dest_owned)
            & ~(dest_opponent & (army_ratio < 1.1))
        )
        selected_can_capture = jnp.where(prefer_half, can_capture_half, can_capture_all)
        attack_target = (dest_neutral | dest_opponent | (dest_city & ~dest_owned)) & ~dest_owned
        moving_away_from_general = has_own_general & (dest_general_dist > source_general_dist)
        expansion = _expansion_score(
            source_armies,
            dest_owned,
            dest_neutral,
            dest_opponent,
            selected_can_capture,
            neutral_weight=10.0,
            opponent_weight=20.0 * aggression,
        )

        capture_score = expansion
        capture_score = capture_score + jnp.where(dest_general & selected_can_capture, 2200.0 + 220.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_opponent & selected_can_capture, 150.0 * aggression + 25.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_city & ~dest_owned & selected_can_capture, 190.0 + 30.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_neutral & selected_can_capture, 62.0 + 18.0 * large_map, 0.0)
        capture_score = capture_score + jnp.where(dest_unknown, 10.0 + 6.0 * large_map, 0.0)
        capture_score = jnp.where(attack_target & ~selected_can_capture & ~dest_unknown, capture_score * 0.05, capture_score)
        capture_score = jnp.where(source_general & ~prefer_half & (source_armies < 8 + 4 * large_map), capture_score * 0.35, capture_score)
        capture_score = jnp.where(
            near_general_source & moving_away_from_general & (observation.owned_land_count < 4 + 2 * large_map),
            capture_score * 0.65,
            capture_score,
        )
        approach_score = source_armies.astype(jnp.float32) * (0.85 + 0.15 * large_map)
        approach_score = approach_score + general_approach + city_approach + enemy_approach + explore_approach
        approach_score = approach_score + jnp.where(dest_owned & (dest_general_dist < source_general_dist), 28.0 + 10.0 * large_map, 0.0)
        approach_score = jnp.where(dest_owned | dest_unknown, approach_score, approach_score * (0.25 + 0.15 * large_map))
        split = prefer_half.astype(jnp.int32)
        return jnp.where(is_valid, capture_score, 0.0), jnp.where(is_valid, approach_score, 0.0), split

    capture_scores, approach_scores, splits = jax.vmap(evaluate)(jnp.arange(max_moves))
    scores = jnp.where(jnp.sum(jnp.maximum(capture_scores, 0.0)) > 0.0, capture_scores, approach_scores)
    return _select_sampled_action(key, positions, scores, splits, num_valid, max_moves)


@jax.jit
def mixed_heuristic_action(key: jnp.ndarray, observation) -> jnp.ndarray:
    selector_key, action_key = jrandom.split(key)
    scale = _map_scale(observation)
    large_map = _large_map_factor(scale)
    weights = jnp.array(
        [
            0.30 - 0.02 * large_map,
            0.18 - 0.02 * large_map,
            0.16 + 0.02 * large_map,
            0.12 - 0.04 * large_map,
            0.24 + 0.06 * large_map,
        ],
        dtype=jnp.float32,
    )
    weights = weights / jnp.sum(weights)
    selected = jrandom.choice(selector_key, 5, p=weights)
    branches = (
        expander_action,
        city_rush_action,
        general_hunter_action,
        defensive_expander_action,
        balanced_strategic_action,
    )
    return jax.lax.switch(selected, branches, action_key, observation)


@jax.jit
def heuristic_action(heuristic_id: jnp.ndarray, key: jnp.ndarray, observation) -> jnp.ndarray:
    branches = (
        expander_action,
        city_rush_action,
        general_hunter_action,
        defensive_expander_action,
        balanced_strategic_action,
        mixed_heuristic_action,
    )
    return jax.lax.switch(heuristic_id, branches, key, observation)
