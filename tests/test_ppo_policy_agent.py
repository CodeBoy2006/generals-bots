import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import PPOPolicyAgent, PolicyValueNetwork
from generals.core import game


def make_checkpoint(tmp_path, grid_size=4):
    model_path = tmp_path / "policy.eqx"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=grid_size)
    eqx.tree_serialise_leaves(model_path, network)
    return model_path


def make_observation():
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(5))
    return game.get_observation(state, 0)


def test_ppo_policy_agent_loads_checkpoint_and_returns_action(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, policy_mode="greedy")

    action = agent.act(make_observation(), jrandom.PRNGKey(1))

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert int(action[0]) in (0, 1)
    assert int(action[4]) in (0, 1)


def test_ppo_policy_agent_rejects_observation_size_mismatch(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, policy_mode="greedy")
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[4, 4].set(2)
    obs = game.get_observation(game.create_initial_state(grid), 0)

    try:
        agent.act(obs, jrandom.PRNGKey(1))
    except ValueError as exc:
        assert "expects 4x4" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched observation shape")
