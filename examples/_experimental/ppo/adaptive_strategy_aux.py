"""Weak strategic auxiliary labels for adaptive Generals.io training."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

STRATEGY_INTENT_EXPAND = 0
STRATEGY_INTENT_GATHER = 1
STRATEGY_INTENT_ATTACK = 2
STRATEGY_INTENT_DEFEND = 3
STRATEGY_INTENT_CAPTURE_CITY = 4
STRATEGY_INTENT_PROBE = 5
STRATEGY_INTENT_FINISH = 6
STRATEGY_INTENT_ATTRITION = 7
STRATEGY_INTENT_COUNT = 8

OUTCOME_WIN = 2


class StrategyAuxTargets(NamedTuple):
    """Training labels for the strategic auxiliary heads."""

    intent: jnp.ndarray
    finish: jnp.ndarray
    enemy_general_heatmap: jnp.ndarray


def enemy_general_heatmap_from_state(state, learner_player: int, pad_size: int) -> jnp.ndarray:
    """Return a one-hot map of the opponent general from the full state."""
    opponent = 1 - learner_player
    row, col = state.general_positions[opponent]
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    return ((rows == row) & (cols == col)).astype(jnp.float32)


def weak_intent_label(obs, state, learner_player: int, search_outcomes: jnp.ndarray) -> jnp.ndarray:
    """Generate a cheap strategic intent label from visible state and search finish signal."""
    opponent = 1 - learner_player
    visible_enemy = jnp.any(obs.opponent_cells)
    finish_available = jnp.any(search_outcomes == OUTCOME_WIN)
    own_general = state.general_positions[learner_player]
    enemy_owned = state.ownership[opponent]
    rows = jnp.arange(state.armies.shape[0])[:, None]
    cols = jnp.arange(state.armies.shape[1])[None, :]
    distance_to_own_general = jnp.abs(rows - own_general[0]) + jnp.abs(cols - own_general[1])
    enemy_near_general = jnp.any(enemy_owned & (distance_to_own_general <= 2))
    visible_city = jnp.any(obs.cities & ~obs.owned_cells)
    early = obs.timestep < 25
    midgame = (obs.timestep >= 25) & (obs.timestep < 75)

    intent = jnp.asarray(STRATEGY_INTENT_ATTRITION, dtype=jnp.int32)
    intent = jnp.where(~visible_enemy, STRATEGY_INTENT_PROBE, intent)
    intent = jnp.where(midgame & ~visible_enemy, STRATEGY_INTENT_GATHER, intent)
    intent = jnp.where(early & ~visible_enemy, STRATEGY_INTENT_EXPAND, intent)
    intent = jnp.where(visible_enemy, STRATEGY_INTENT_ATTACK, intent)
    intent = jnp.where(visible_city, STRATEGY_INTENT_CAPTURE_CITY, intent)
    intent = jnp.where(enemy_near_general, STRATEGY_INTENT_DEFEND, intent)
    intent = jnp.where(finish_available, STRATEGY_INTENT_FINISH, intent)
    return intent.astype(jnp.int32)


def strategy_aux_targets(
    state,
    obs,
    learner_player: int,
    effective_size: int,
    pad_size: int,
    search_scores: jnp.ndarray,
    search_outcomes: jnp.ndarray,
    search_value_scale: float = 100.0,
) -> StrategyAuxTargets:
    """Build strategic auxiliary targets for one learner state."""
    del effective_size
    finish = jnp.any(search_outcomes == OUTCOME_WIN).astype(jnp.int32)
    del search_scores, search_value_scale
    return StrategyAuxTargets(
        intent=weak_intent_label(obs, state, learner_player, search_outcomes),
        finish=finish,
        enemy_general_heatmap=enemy_general_heatmap_from_state(state, learner_player, pad_size),
    )
