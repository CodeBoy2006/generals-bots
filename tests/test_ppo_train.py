import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.common import policy_network_action
from examples._experimental.ppo.evaluate_policy import summarize_policy_results
from examples._experimental.ppo.train import apply_terminal_reward, load_or_create_network, stack_learner_actions
from generals.agents.ppo_policy_agent import PolicyValueNetwork, greedy_policy_action
from generals.core import game
from generals.core.game import GameInfo


def test_load_or_create_network_restores_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "policy.eqx"
    saved = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    eqx.tree_serialise_leaves(checkpoint_path, saved)

    loaded = load_or_create_network(jrandom.PRNGKey(1), grid_size=4, init_model_path=checkpoint_path)

    obs = jnp.zeros((9, 4, 4), dtype=jnp.float32)
    mask = jnp.ones((4, 4, 4), dtype=bool)
    saved_logits, saved_value = saved.logits_value(obs, mask)
    loaded_logits, loaded_value = loaded.logits_value(obs, mask)

    assert jnp.allclose(loaded_logits, saved_logits)
    assert jnp.allclose(loaded_value, saved_value)


def test_load_or_create_network_restores_custom_channel_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "wide-policy.eqx"
    channels = (16, 16, 16, 8)
    saved = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4, channels=channels)
    eqx.tree_serialise_leaves(checkpoint_path, saved)

    loaded = load_or_create_network(
        jrandom.PRNGKey(1),
        grid_size=4,
        init_model_path=checkpoint_path,
        channels=channels,
    )

    obs = jnp.zeros((9, 4, 4), dtype=jnp.float32)
    mask = jnp.ones((4, 4, 4), dtype=bool)
    saved_logits, saved_value = saved.logits_value(obs, mask)
    loaded_logits, loaded_value = loaded.logits_value(obs, mask)

    assert jnp.allclose(loaded_logits, saved_logits)
    assert jnp.allclose(loaded_value, saved_value)


def test_load_or_create_network_rejects_missing_checkpoint(tmp_path):
    missing_path = tmp_path / "missing.eqx"

    try:
        load_or_create_network(jrandom.PRNGKey(1), grid_size=4, init_model_path=missing_path)
    except FileNotFoundError as exc:
        assert str(missing_path) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing warm-start checkpoint")


def test_summarize_policy_results_counts_wins_for_selected_player():
    info = GameInfo(
        army=jnp.zeros((4, 2), dtype=jnp.int32),
        land=jnp.zeros((4, 2), dtype=jnp.int32),
        is_done=jnp.array([True, True, True, False]),
        winner=jnp.array([0, 1, 1, -1], dtype=jnp.int32),
        time=jnp.array([10, 20, 30, 40], dtype=jnp.int32),
    )

    summary = summarize_policy_results(info, policy_player=1, num_games=4)

    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["win_rate"] == 0.5


def test_policy_network_action_dispatches_greedy_mode():
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(6))
    obs = game.get_observation(state, 0)

    action = policy_network_action(network, jrandom.PRNGKey(1), obs, 0)

    assert jnp.array_equal(action, greedy_policy_action(network, obs))


def test_policy_network_action_dispatches_sample_mode():
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(6))
    obs = game.get_observation(state, 0)

    action = policy_network_action(network, jrandom.PRNGKey(1), obs, 1)

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert int(action[0]) in (0, 1)


def test_apply_terminal_reward_only_adjusts_decisive_terminal_games():
    info = GameInfo(
        army=jnp.zeros((4, 2), dtype=jnp.int32),
        land=jnp.zeros((4, 2), dtype=jnp.int32),
        is_done=jnp.array([True, True, True, False]),
        winner=jnp.array([0, 1, -1, 0], dtype=jnp.int32),
        time=jnp.array([10, 20, 30, 40], dtype=jnp.int32),
    )
    rewards = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)

    adjusted = apply_terminal_reward(rewards, info, learner_player=1, terminal_reward_scale=2.0)

    assert jnp.allclose(adjusted, jnp.array([-1.9, 2.2, 0.3, 0.4], dtype=jnp.float32))


def test_stack_learner_actions_places_actions_in_selected_player_slot():
    learner_actions = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=jnp.int32)
    opponent_actions = jnp.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]], dtype=jnp.int32)

    as_player0 = stack_learner_actions(learner_actions, opponent_actions, learner_player=0)
    as_player1 = stack_learner_actions(learner_actions, opponent_actions, learner_player=1)

    assert jnp.array_equal(as_player0[:, 0], learner_actions)
    assert jnp.array_equal(as_player0[:, 1], opponent_actions)
    assert jnp.array_equal(as_player1[:, 0], opponent_actions)
    assert jnp.array_equal(as_player1[:, 1], learner_actions)
