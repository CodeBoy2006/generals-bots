import jax.numpy as jnp
import jax.random as jrandom


def test_make_policy_agent_uses_adaptive_agent_when_configured(monkeypatch):
    from generals.agents import ppo_runtime
    from generals.agents.ppo_runtime import AdaptiveRuntimeConfig, SearchConfig

    captured = {}

    class DummyAdaptiveAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.id = kwargs["agent_id"]

        def act_for_state(self, state, player, key):
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

    monkeypatch.setattr(ppo_runtime, "AdaptiveWebPolicyAgent", DummyAdaptiveAgent, raising=False)

    search_config = SearchConfig(
        rollout_policy_mode="sample",
        top_k=4,
        rollout_steps=16,
        rollouts_per_action=2,
        army_weight=1.0,
        land_weight=10.0,
        prior_weight=0.001,
    )
    adaptive_config = AdaptiveRuntimeConfig(
        pad_to=16,
        network_arch="unet",
        channels="64,96,128,64",
        global_context=True,
        scoreboard_history=True,
        fog_memory=True,
        value_loss="hl-gauss",
        policy_adapter_path="adapter.eqx",
        policy_adapter_scale=1.0,
        policy_adapter_mode="replace",
        policy_adapter_max_grid_size=8,
        online_search=True,
        online_search_min_turn=80,
        online_search_require_contact=True,
        online_search_opponent_path="v5.eqx",
        online_search_opponent_channels="32,32,32,16",
        online_search_opponent_input_channels=9,
    )

    agent = ppo_runtime.make_policy_agent(
        "adaptive.eqx",
        8,
        "sample",
        "Adaptive Champion",
        "auto",
        None,
        False,
        search_config,
        adaptive_config=adaptive_config,
    )

    assert isinstance(agent, DummyAdaptiveAgent)
    assert captured["model_path"] == "adaptive.eqx"
    assert captured["grid_size"] == 8
    assert captured["agent_id"] == "Adaptive Champion"
    assert captured["policy_mode"] == "sample"
    assert captured["search_config"] == search_config
    assert captured["config"] == adaptive_config


def test_make_grid_uses_separate_effective_size_and_padding_for_adaptive_maps():
    from types import SimpleNamespace

    from generals.agents.ppo_runtime import make_grid

    args = SimpleNamespace(
        grid_size=8,
        map_pad_to=16,
        map_generator="simple",
        mountain_density_min=0.12,
        mountain_density_max=0.22,
        num_cities_min=4,
        num_cities_max=8,
        effective_min_generals_distance=5,
        max_generals_distance=None,
        city_army_min=40,
        city_army_max=51,
    )

    grid = make_grid(args, jrandom.PRNGKey(0))

    assert grid.shape == (16, 16)
    assert jnp.all(grid[8:, :] == -2)
    assert jnp.all(grid[:, 8:] == -2)
    assert int(jnp.sum(grid[:8, :8] > 0)) == 2
