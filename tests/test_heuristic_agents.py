"""Tests for JAX-compatible heuristic agents."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from generals.agents._heuristic_logic import HEURISTIC_NAMES, heuristic_action
from generals.core import game
from generals.core.action import compute_valid_move_mask_obs


def create_medium_grid():
    grid = jnp.zeros((8, 8), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[7, 7].set(2)
    grid = grid.at[2, 3].set(45)
    grid = grid.at[5, 4].set(42)
    grid = grid.at[3, 1].set(-2)
    grid = grid.at[4, 6].set(-2)
    return grid


def assert_action_shape_and_legality(action, obs):
    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert int(action[0]) in (0, 1)
    assert int(action[4]) in (0, 1)
    if int(action[0]) == 0:
        row, col, direction = map(int, action[1:4])
        assert compute_valid_move_mask_obs(obs)[row, col, direction]
    else:
        assert int(action[1]) == 0
        assert int(action[2]) == 0


@pytest.mark.parametrize("heuristic_id", range(len(HEURISTIC_NAMES)))
def test_heuristic_action_is_valid_or_passes(heuristic_id):
    state = game.create_initial_state(create_medium_grid())
    state = state._replace(armies=state.armies.at[0, 0].set(25))
    obs = game.get_observation(state, 0)
    action = heuristic_action(jnp.int32(heuristic_id), jrandom.PRNGKey(heuristic_id), obs)
    assert_action_shape_and_legality(action, obs)


@pytest.mark.parametrize("heuristic_id", range(len(HEURISTIC_NAMES)))
def test_heuristic_action_jit_compiles(heuristic_id):
    state = game.create_initial_state(create_medium_grid())
    state = state._replace(armies=state.armies.at[0, 0].set(25))
    obs = game.get_observation(state, 0)
    action = jax.jit(heuristic_action)(jnp.int32(heuristic_id), jrandom.PRNGKey(heuristic_id), obs)
    assert_action_shape_and_legality(action, obs)


def test_heuristic_action_vmapped_batch():
    state = game.create_initial_state(create_medium_grid())
    state = state._replace(armies=state.armies.at[0, 0].set(25))
    states = jax.tree.map(lambda x: jnp.stack([x, x, x]), state)
    obs = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    keys = jrandom.split(jrandom.PRNGKey(123), 3)
    heuristic_ids = jnp.array([0, 3, 5], dtype=jnp.int32)
    actions = jax.vmap(heuristic_action)(heuristic_ids, keys, obs)
    assert actions.shape == (3, 5)
    for idx in range(3):
        assert_action_shape_and_legality(actions[idx], jax.tree.map(lambda x: x[idx], obs))
