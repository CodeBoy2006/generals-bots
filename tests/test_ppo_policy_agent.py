import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import PPOPolicyAgent, PolicyValueNetwork, load_policy_network, parse_policy_channels
from generals.core import game


def make_checkpoint(tmp_path, grid_size=4, input_channels=9):
    model_path = tmp_path / "policy.eqx"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=grid_size, input_channels=input_channels)
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


def make_state():
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    state = game.create_initial_state(grid)
    return state._replace(armies=state.armies.at[0, 0].set(5))


def test_ppo_policy_agent_loads_checkpoint_and_returns_action(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, policy_mode="greedy")

    action = agent.act(make_observation(), jrandom.PRNGKey(1))

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert int(action[0]) in (0, 1)
    assert int(action[4]) in (0, 1)


def test_ppo_policy_agent_loads_augmented_full_state_checkpoint_and_returns_action(tmp_path):
    agent = PPOPolicyAgent(
        make_checkpoint(tmp_path, input_channels=18),
        grid_size=4,
        policy_mode="sample",
        policy_input="augmented-full-state",
    )

    action = agent.act_for_state(make_state(), player=0, key=jrandom.PRNGKey(1))
    preview = agent.explain_for_state(make_state(), player=0, top_k=3)

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert int(action[0]) in (0, 1)
    assert 0 < len(preview.candidates) <= 3
    assert preview.policy_mode == "sample"


def test_ppo_policy_agent_auto_detects_augmented_full_state_checkpoint(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path, input_channels=18), grid_size=4, policy_mode="sample")

    action = agent.act_for_state(make_state(), player=0, key=jrandom.PRNGKey(1))

    assert agent.policy_input == "augmented-full-state"
    assert agent.input_channels == 18
    assert action.shape == (5,)


def test_ppo_policy_agent_auto_detects_observation_checkpoint(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, policy_mode="greedy")

    action = agent.act(make_observation(), jrandom.PRNGKey(1))

    assert agent.policy_input == "observation"
    assert agent.input_channels == 9
    assert action.shape == (5,)


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


def test_ppo_policy_agent_explain_returns_ordered_candidates(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4, policy_mode="sample")

    preview = agent.explain(make_observation(), top_k=5)

    assert preview.policy_mode == "sample"
    assert isinstance(preview.value, float)
    assert 0 < len(preview.candidates) <= 5
    probabilities = [candidate.probability for candidate in preview.candidates]
    assert probabilities == sorted(probabilities, reverse=True)
    for candidate in preview.candidates:
        assert len(candidate.action) == 5
        assert 0.0 <= candidate.probability <= 1.0
        assert candidate.action[0] in (0, 1)
        assert candidate.action[3] in (0, 1, 2, 3)
        assert candidate.action[4] in (0, 1)
        if candidate.is_pass:
            assert candidate.source is None
            assert candidate.target is None
        else:
            assert candidate.source is not None
            assert candidate.target is not None


def test_ppo_policy_agent_explain_merges_pass_actions(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4)

    preview = agent.explain(make_observation(), top_k=20)

    pass_candidates = [candidate for candidate in preview.candidates if candidate.is_pass]
    assert len(pass_candidates) <= 1


def test_ppo_policy_agent_explain_rejects_observation_size_mismatch(tmp_path):
    agent = PPOPolicyAgent(make_checkpoint(tmp_path), grid_size=4)
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[4, 4].set(2)
    obs = game.get_observation(game.create_initial_state(grid), 0)

    try:
        agent.explain(obs)
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
