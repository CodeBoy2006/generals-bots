import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import PPOPolicyAgent, PolicyValueNetwork, load_policy_network, parse_policy_channels
from generals.core import game


def make_checkpoint(tmp_path, grid_size=4):
    model_path = tmp_path / "policy.eqx"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=grid_size)
    eqx.tree_serialise_leaves(model_path, network)
    return model_path


def make_checkpoint_with_channels(tmp_path, grid_size=4, channels=(16, 16, 16, 8)):
    model_path = tmp_path / "wide-policy.eqx"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=grid_size, channels=channels)
    eqx.tree_serialise_leaves(model_path, network)
    return model_path, network


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


def test_ppo_policy_agent_accepts_agent_id_keyword(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, agent_id="Model")

    assert agent.id == "Model"


def test_ppo_policy_agent_keeps_legacy_id_keyword(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, id="Legacy")

    assert agent.id == "Legacy"


def test_ppo_policy_agent_rejects_conflicting_identifier_keywords(tmp_path):
    try:
        PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, agent_id="Model", id="Legacy")
    except TypeError as exc:
        assert "agent_id" in str(exc)
        assert "id" in str(exc)
    else:
        raise AssertionError("Expected TypeError for conflicting identifier keywords")


def test_parse_policy_channels_accepts_four_positive_integers():
    assert parse_policy_channels("64,64,64,32") == (64, 64, 64, 32)
    assert parse_policy_channels((16, 16, 16, 8)) == (16, 16, 16, 8)


def test_load_policy_network_accepts_custom_channels(tmp_path):
    model_path, saved = make_checkpoint_with_channels(tmp_path)

    loaded = load_policy_network(model_path, grid_size=4, channels=(16, 16, 16, 8))

    obs = jnp.zeros((9, 4, 4), dtype=jnp.float32)
    mask = jnp.ones((4, 4, 4), dtype=bool)
    saved_logits, saved_value = saved.logits_value(obs, mask)
    loaded_logits, loaded_value = loaded.logits_value(obs, mask)

    assert jnp.allclose(loaded_logits, saved_logits)
    assert jnp.allclose(loaded_value, saved_value)
