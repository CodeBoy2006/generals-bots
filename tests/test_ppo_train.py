import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.evaluate_policy import summarize_policy_results
from examples._experimental.ppo.train import load_or_create_network
from generals.core.game import GameInfo
from generals.agents.ppo_policy_agent import PolicyValueNetwork


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
