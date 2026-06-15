import jax
import jax.numpy as jnp

from generals.core.game import GameState
from generals.core.observation import Observation


def compute_num_cities_owned(observation: Observation) -> jnp.ndarray:
    """Count number of cities owned by the agent."""
    owned_cities_mask = observation.cities & observation.owned_cells
    num_cities_owned = jnp.sum(owned_cities_mask)
    return num_cities_owned.astype(jnp.float32)


def compute_num_generals_owned(observation: Observation) -> jnp.ndarray:
    """Count number of generals owned by the agent."""
    owned_generals_mask = observation.generals & observation.owned_cells
    num_generals_owned = jnp.sum(owned_generals_mask)
    return num_generals_owned.astype(jnp.float32)


@jax.jit
def general_target_potential(
    state: GameState,
    player: int,
    max_distance: int = 16,
    min_army: int = 2,
) -> jnp.ndarray:
    """Return attack pressure potential toward the opponent general."""
    opponent = 1 - player
    target = state.general_positions[opponent]
    height, width = state.armies.shape
    rows = jnp.arange(height)[:, None]
    cols = jnp.arange(width)[None, :]
    distances = jnp.abs(rows - target[0]) + jnp.abs(cols - target[1])
    max_distance_f = jnp.maximum(max_distance, 1).astype(jnp.float32)
    eligible = state.ownership[player] & state.passable & (state.armies >= min_army)
    nearest = jnp.min(jnp.where(eligible, distances.astype(jnp.float32), max_distance_f))
    nearest = jnp.minimum(nearest, max_distance_f)
    potential = (max_distance_f - nearest) / max_distance_f
    return jnp.where(jnp.any(eligible), potential, 0.0)


@jax.jit
def general_target_reward_fn(
    prior_state: GameState,
    state: GameState,
    player: int,
    scale: float = 0.0,
    max_distance: int = 16,
    min_army: int = 2,
) -> jnp.ndarray:
    """Reward progress by strong owned cells toward the opponent general."""
    prior_potential = general_target_potential(prior_state, player, max_distance, min_army)
    current_potential = general_target_potential(state, player, max_distance, min_army)
    active = (prior_state.winner < 0) & (state.winner < 0)
    return jnp.where(active, scale * (current_potential - prior_potential), 0.0)


@jax.jit
def shortest_path_distance_map(
    passable: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """Return cardinal shortest-path distances to the nearest target."""
    unreachable = passable.size + 1
    distances = jnp.where(passable & targets, 0, unreachable).astype(jnp.int32)

    def relax(_, current):
        padded = jnp.pad(current, 1, mode="constant", constant_values=unreachable)
        neighbor_min = jnp.min(
            jnp.stack(
                [
                    padded[:-2, 1:-1],
                    padded[2:, 1:-1],
                    padded[1:-1, :-2],
                    padded[1:-1, 2:],
                ]
            ),
            axis=0,
        )
        relaxed = jnp.minimum(current, neighbor_min + 1)
        return jnp.where(passable, relaxed, unreachable)

    return jax.lax.fori_loop(0, passable.size, relax, distances)


@jax.jit
def path_target_distance_cache(state: GameState, player: int) -> jnp.ndarray:
    """Cache shortest-path distance maps for assignment target families."""
    opponent = 1 - player
    target = state.general_positions[opponent]
    general_targets = jnp.zeros_like(state.passable).at[target[0], target[1]].set(True)
    city_targets = state.cities & ~state.ownership[player] & state.passable
    frontier_targets = ~state.ownership[player] & state.passable
    return jnp.stack(
        [
            shortest_path_distance_map(state.passable, general_targets),
            shortest_path_distance_map(state.passable, city_targets),
            shortest_path_distance_map(state.passable, frontier_targets),
        ]
    )


@jax.jit
def path_assignment_potential(
    state: GameState,
    player: int,
    max_distance: int = 64,
    min_army: int = 2,
    general_weight: float = 1.0,
    city_weight: float = 0.8,
    frontier_weight: float = 0.25,
) -> jnp.ndarray:
    """Return weighted transport potential from strong owned cells to assigned targets."""
    distance_cache = path_target_distance_cache(state, player)
    max_distance_f = jnp.maximum(jnp.asarray(max_distance, dtype=jnp.float32), 1.0)
    scores = (max_distance_f - jnp.minimum(distance_cache.astype(jnp.float32), max_distance_f)) / max_distance_f
    target_weights = jnp.asarray(
        [general_weight, city_weight, frontier_weight],
        dtype=jnp.float32,
    ).reshape((3, 1, 1))
    assigned_scores = jnp.max(scores * target_weights, axis=0)

    eligible = state.ownership[player] & state.passable & (state.armies >= min_army)
    army_weights = jnp.where(eligible, jnp.log1p(state.armies.astype(jnp.float32)), 0.0)
    total_weight = jnp.sum(army_weights)
    potential = jnp.sum(assigned_scores * army_weights) / jnp.maximum(total_weight, 1.0)
    return jnp.where(total_weight > 0.0, potential, 0.0)


@jax.jit
def path_assignment_reward_fn(
    prior_state: GameState,
    state: GameState,
    player: int,
    scale: float = 0.0,
    max_distance: int = 64,
    min_army: int = 2,
    general_weight: float = 1.0,
    city_weight: float = 0.8,
    frontier_weight: float = 0.25,
) -> jnp.ndarray:
    """Reward assigned transport progress along passable shortest paths."""
    prior_potential = path_assignment_potential(
        prior_state,
        player,
        max_distance,
        min_army,
        general_weight,
        city_weight,
        frontier_weight,
    )
    current_potential = path_assignment_potential(
        state,
        player,
        max_distance,
        min_army,
        general_weight,
        city_weight,
        frontier_weight,
    )
    active = (prior_state.winner < 0) & (state.winner < 0)
    return jnp.where(active, scale * (current_potential - prior_potential), 0.0)


@jax.jit
def calculate_army_size(castles: jnp.ndarray, ownership: jnp.ndarray) -> jnp.ndarray:
    """Calculate total army size in castles (cities/generals) owned by the player."""
    return jnp.sum(castles * ownership).astype(jnp.float32)


@jax.jit
def city_reward_fn(
    prior_obs: Observation, 
    prior_action: jnp.ndarray, 
    obs: Observation,
    shaping_weight: float = 0.3
) -> jnp.ndarray:
    """
    Reward function that shapes the reward based on the number of cities owned.
    
    Args:
        shaping_weight: Weight for city change shaping term
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # If game is done, don't shape the reward
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    
    city_now = calculate_army_size(obs.cities, obs.owned_cells)
    city_prev = calculate_army_size(prior_obs.cities, prior_obs.owned_cells)
    city_change = city_now - city_prev
    
    shaped_reward = original_reward + shaping_weight * city_change
    
    return jnp.where(game_done, original_reward, shaped_reward)


@jax.jit
def ratio_reward_fn(
    prior_obs: Observation, 
    prior_action: jnp.ndarray, 
    obs: Observation,
    clip_value: float = 1.5,
    shaping_weight: float = 0.5
) -> jnp.ndarray:
    """
    Reward function that shapes based on army ratio between player and opponent.
    
    Args:
        clip_value: Maximum ratio for clipping
        shaping_weight: Weight for ratio shaping term
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # If game is done, don't shape the reward
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    
    def calculate_ratio_reward(my_army: jnp.ndarray, opponent_army: jnp.ndarray) -> jnp.ndarray:
        ratio = my_army / jnp.maximum(opponent_army, 1.0)  # Avoid division by zero
        ratio = jnp.log(ratio) / jnp.log(clip_value)
        return jnp.clip(ratio, -1.0, 1.0)
    
    prev_ratio_reward = calculate_ratio_reward(
        prior_obs.owned_army_count.astype(jnp.float32), 
        prior_obs.opponent_army_count.astype(jnp.float32)
    )
    current_ratio_reward = calculate_ratio_reward(
        obs.owned_army_count.astype(jnp.float32), 
        obs.opponent_army_count.astype(jnp.float32)
    )
    ratio_reward = current_ratio_reward - prev_ratio_reward
    
    shaped_reward = original_reward + shaping_weight * ratio_reward
    
    return jnp.where(game_done, original_reward, shaped_reward)


@jax.jit
def win_lose_reward_fn(
    prior_obs: Observation, 
    prior_action: jnp.ndarray, 
    obs: Observation
) -> jnp.ndarray:
    """
    Simple reward function based on generals owned with small bonus for splitting.
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # Encourage splitting a bit
    split_bonus = jnp.where(prior_action[4] == 1, 0.0015, 0.0)
    
    return original_reward + split_bonus


@jax.jit
def composite_reward_fn(
    prior_obs: Observation, 
    prior_action: jnp.ndarray, 
    obs: Observation,
    city_weight: float = 0.4,
    ratio_weight: float = 0.3,
    maximum_army_ratio: float = 1.6,
    maximum_land_ratio: float = 1.3
) -> jnp.ndarray:
    """
    Composite reward function combining multiple reward signals.
    
    Combines:
    - Base win/lose reward (generals owned)
    - Army ratio reward
    - Land ratio reward
    - City capture reward
    
    Args:
        city_weight: Weight for city reward
        ratio_weight: Weight for ratio rewards (army and land)
        maximum_army_ratio: Maximum army ratio for clipping
        maximum_land_ratio: Maximum land ratio for clipping
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # If game is done, don't shape the reward (except split bonus)
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    
    def calculate_ratio_reward(
        mine: jnp.ndarray, 
        opponents: jnp.ndarray, 
        max_ratio: float
    ) -> jnp.ndarray:
        ratio = mine / jnp.maximum(opponents, 1.0)  # Avoid division by zero
        ratio = jnp.log(ratio) / jnp.log(max_ratio)
        return jnp.clip(ratio, -1.0, 1.0)
    
    # Army ratio reward
    previous_army_ratio = calculate_ratio_reward(
        prior_obs.owned_army_count.astype(jnp.float32),
        prior_obs.opponent_army_count.astype(jnp.float32),
        maximum_army_ratio
    )
    current_army_ratio = calculate_ratio_reward(
        obs.owned_army_count.astype(jnp.float32),
        obs.opponent_army_count.astype(jnp.float32),
        maximum_army_ratio
    )
    army_reward = current_army_ratio - previous_army_ratio
    
    # Land ratio reward
    previous_land_ratio = calculate_ratio_reward(
        prior_obs.owned_land_count.astype(jnp.float32),
        prior_obs.opponent_land_count.astype(jnp.float32),
        maximum_land_ratio
    )
    current_land_ratio = calculate_ratio_reward(
        obs.owned_land_count.astype(jnp.float32),
        obs.opponent_land_count.astype(jnp.float32),
        maximum_land_ratio
    )
    land_reward = current_land_ratio - previous_land_ratio
    
    # City reward
    city_reward = compute_num_cities_owned(obs) - compute_num_cities_owned(prior_obs)

    # Combine all rewards
    shaped_reward = (
        original_reward
        + ratio_weight * army_reward
        + city_weight * city_reward
        + ratio_weight * land_reward
    )

    # If game done, only return original reward
    return jnp.where(game_done, original_reward, shaped_reward)
