import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.common import (
    POLICY_INPUT_NAME_TO_ID,
    augmented_full_state_to_array,
    full_state_to_array,
    obs_to_array,
    policy_state_action,
)
from generals.agents.ppo_policy_agent import PolicyValueNetwork
from generals.core import game


def test_full_state_to_array_exposes_hidden_opponent_cells():
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    obs = game.get_observation(state, 0)

    full = full_state_to_array(state, 0)

    assert not bool(obs.opponent_cells[3, 3])
    assert bool(full[6, 3, 3])
    assert bool(full[1, 3, 3])


def test_policy_state_action_supports_full_state_input():
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(6))
    obs = game.get_observation(state, 0)

    action = policy_state_action(
        network,
        jrandom.PRNGKey(1),
        state,
        obs,
        player=0,
        policy_mode=1,
        policy_input=POLICY_INPUT_NAME_TO_ID["full-state"],
    )

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert int(action[0]) in (0, 1)


def test_augmented_full_state_keeps_observation_channels_first():
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    obs = game.get_observation(state, 0)

    augmented = augmented_full_state_to_array(state, obs, 0)

    assert augmented.shape == (18, 4, 4)
    assert jnp.array_equal(augmented[:9], obs_to_array(obs))
    assert bool(augmented[9 + 6, 3, 3])
