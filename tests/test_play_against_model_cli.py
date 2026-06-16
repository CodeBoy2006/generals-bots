import jax.numpy as jnp
import pytest

import examples.play_against_model as play_against_model
from examples.play_against_model import (
    advance_until_human_can_move,
    auto_tick_due,
    choose_human_action,
    choose_machine_actions,
    human_can_move,
    make_player_names,
    parse_args,
)
from generals.core import game


def parse_with_args(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["play_against_model.py", "policy.eqx", *args])
    return parse_args()


def parse_raw_args(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["play_against_model.py", *args])
    return parse_args()


def test_parse_args_rejects_nonpositive_min_generals_distance(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--min-generals-distance", "0")


def test_parse_args_rejects_nonpositive_max_generals_distance(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--max-generals-distance", "0")


def test_parse_args_rejects_min_generals_distance_above_max(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--min-generals-distance", "5", "--max-generals-distance", "4")


def test_parse_args_rejects_default_min_generals_distance_above_max(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--grid-size", "8", "--max-generals-distance", "3")


def test_parse_args_accepts_valid_generals_distance(monkeypatch):
    args = parse_with_args(monkeypatch, "--min-generals-distance", "3", "--max-generals-distance", "5")

    assert args.effective_min_generals_distance == 3
    assert args.max_generals_distance == 5


def test_parse_args_accepts_preview_options(monkeypatch):
    args = parse_with_args(monkeypatch, "--preview-top-k", "5", "--no-ai-preview")

    assert args.preview_top_k == 5
    assert args.ai_preview is False


def test_parse_args_defaults_to_ai_preview(monkeypatch):
    args = parse_with_args(monkeypatch)

    assert args.preview_top_k == 3
    assert args.ai_preview is True
    assert args.policy_mode == "sample"
    assert args.auto_tick is True


def test_parse_args_rejects_preview_top_k_below_range(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--preview-top-k", "0")


def test_parse_args_rejects_preview_top_k_above_range(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--preview-top-k", "6")


def test_parse_args_accepts_auto_tick_options(monkeypatch):
    args = parse_with_args(monkeypatch, "--no-auto-tick", "--tick-rate", "2.5")

    assert args.auto_tick is False
    assert args.tick_rate == 2.5


def test_parse_args_accepts_machine_vs_machine_options(monkeypatch):
    args = parse_with_args(
        monkeypatch,
        "--machine-vs-machine",
        "--opponent-model-path",
        "opponent.eqx",
        "--opponent-policy-mode",
        "greedy",
    )

    assert args.machine_vs_machine is True
    assert args.opponent_model_path == "opponent.eqx"
    assert args.opponent_policy_mode == "greedy"


def test_parse_args_accepts_rollout_search_options(monkeypatch):
    args = parse_with_args(
        monkeypatch,
        "--search-policy",
        "--opponent-search-policy",
        "--search-rollout-policy-mode",
        "greedy",
        "--search-top-k",
        "2",
        "--search-rollout-steps",
        "3",
        "--search-rollouts-per-action",
        "4",
        "--search-army-weight",
        "5.5",
        "--search-land-weight",
        "6.5",
        "--search-prior-weight",
        "0.25",
    )

    assert args.search_policy is True
    assert args.opponent_search_policy is True
    assert args.search_rollout_policy_mode == "greedy"
    assert args.search_top_k == 2
    assert args.search_rollout_steps == 3
    assert args.search_rollouts_per_action == 4
    assert args.search_army_weight == 5.5
    assert args.search_land_weight == 6.5
    assert args.search_prior_weight == 0.25


def test_parse_args_rejects_nonpositive_search_budget(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--search-policy", "--search-top-k", "0")


def test_parse_args_accepts_explicit_machine_model_paths_without_positional(monkeypatch):
    args = parse_raw_args(
        monkeypatch,
        "--machine-vs-machine",
        "--model-0-path",
        "p0.eqx",
        "--model-1-path",
        "p1.eqx",
    )

    assert args.model_path == "p0.eqx"
    assert args.opponent_model_path == "p1.eqx"


def test_parse_args_accepts_model_1_alias_for_opponent(monkeypatch):
    args = parse_with_args(monkeypatch, "--machine-vs-machine", "--model-1-path", "p1.eqx")

    assert args.model_path == "policy.eqx"
    assert args.opponent_model_path == "p1.eqx"


def test_parse_args_defaults_machine_policy_inputs_to_auto(monkeypatch):
    args = parse_raw_args(
        monkeypatch,
        "--machine-vs-machine",
        "--model-0-path",
        "generals-ppo-8x8-path-assignment-p0-v2.eqx",
        "--model-1-path",
        "generals-ppo-8x8-expander-gpu-v5.eqx",
    )

    assert args.model_0_policy_input == "auto"
    assert args.model_1_policy_input == "auto"
    assert args.model_0_input_channels is None
    assert args.model_1_input_channels is None


def test_parse_args_accepts_separate_machine_policy_inputs(monkeypatch):
    args = parse_raw_args(
        monkeypatch,
        "--machine-vs-machine",
        "--model-0-path",
        "p0.eqx",
        "--model-1-path",
        "p1.eqx",
        "--model-0-policy-input",
        "augmented-full-state",
        "--model-1-policy-input",
        "observation",
    )

    assert args.model_0_policy_input == "augmented-full-state"
    assert args.model_1_policy_input == "observation"
    assert args.model_0_input_channels == 18
    assert args.model_1_input_channels == 9


def test_parse_args_accepts_policy_input_aliases(monkeypatch):
    args = parse_with_args(
        monkeypatch,
        "--machine-vs-machine",
        "--policy-input",
        "augmented-full-state",
        "--opponent-policy-input",
        "observation",
    )

    assert args.model_0_policy_input == "augmented-full-state"
    assert args.model_1_policy_input == "observation"
    assert args.model_0_input_channels == 18
    assert args.model_1_input_channels == 9


def test_parse_args_rejects_missing_primary_model(monkeypatch):
    with pytest.raises(SystemExit):
        parse_raw_args(monkeypatch, "--machine-vs-machine")


def test_parse_args_rejects_nonpositive_tick_rate(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--tick-rate", "0")


def test_advance_until_human_can_move_skips_initial_no_move_turns():
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    state = game.create_initial_state(grid)

    assert human_can_move(state, human_player=0) is False

    warmed_state, warmed_info, auto_passes = advance_until_human_can_move(state, human_player=0)

    assert auto_passes == 2
    assert int(warmed_state.time) == 2
    assert int(warmed_info.time) == 2
    assert human_can_move(warmed_state, human_player=0) is True


def test_auto_tick_due_waits_for_interval_and_pauses_during_selection():
    assert auto_tick_due(auto_tick=True, selected_cell=None, now=10.0, last_tick=9.4, tick_rate=2.0) is True
    assert auto_tick_due(auto_tick=True, selected_cell=None, now=10.0, last_tick=9.8, tick_rate=2.0) is False
    assert auto_tick_due(auto_tick=True, selected_cell=(1, 1), now=10.0, last_tick=9.4, tick_rate=2.0) is False
    assert auto_tick_due(auto_tick=False, selected_cell=None, now=10.0, last_tick=9.4, tick_rate=2.0) is False


def test_choose_human_action_passes_only_on_auto_tick():
    manual_action = jnp.array([0, 2, 3, 1, 0], dtype=jnp.int32)

    assert choose_human_action(manual_action, auto_tick_ready=True).tolist() == manual_action.tolist()
    assert choose_human_action(None, auto_tick_ready=True).tolist() == [1, 0, 0, 0, 0]
    assert choose_human_action(None, auto_tick_ready=False) is None


def test_machine_vs_machine_player_names_do_not_include_human():
    assert make_player_names(human_player=0, machine_vs_machine=True) == ["PPO 0", "PPO 1"]


def test_choose_machine_actions_uses_both_player_observations():
    class FixedAgent:
        def __init__(self, action):
            self.action = jnp.array(action, dtype=jnp.int32)

        def act(self, observation, key):
            return self.action

    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    state = game.create_initial_state(grid)
    agents = (
        FixedAgent([1, 0, 0, 0, 0]),
        FixedAgent([0, 3, 3, 0, 0]),
    )

    actions = choose_machine_actions(state, agents, jnp.array([0, 1], dtype=jnp.uint32))

    assert actions.tolist() == [[1, 0, 0, 0, 0], [0, 3, 3, 0, 0]]


def test_rollout_search_agent_delegates_to_search_action(monkeypatch):
    calls = {}
    sentinel_network = object()

    def fake_load_policy_network(model_path, grid_size, *, channels=None, input_channels=9):
        calls["load"] = (model_path, grid_size, channels, input_channels)
        return sentinel_network

    def fake_rollout_search_action(
        network,
        state,
        key,
        player,
        top_k,
        rollout_steps,
        rollouts_per_action,
        policy_mode,
        army_weight,
        land_weight,
        prior_weight,
    ):
        calls["search"] = (
            network,
            state,
            key,
            player,
            top_k,
            rollout_steps,
            rollouts_per_action,
            policy_mode,
            army_weight,
            land_weight,
            prior_weight,
        )
        return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

    monkeypatch.setattr(play_against_model, "load_policy_network", fake_load_policy_network)
    monkeypatch.setattr(play_against_model, "rollout_search_action", fake_rollout_search_action)

    agent = play_against_model.RolloutSearchPolicyAgent(
        "policy.eqx",
        grid_size=4,
        top_k=2,
        rollout_steps=3,
        rollouts_per_action=4,
        rollout_policy_mode="greedy",
        army_weight=5.0,
        land_weight=6.0,
        prior_weight=0.2,
    )
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    key = jnp.array([0, 1], dtype=jnp.uint32)

    action = agent.act_for_state(state, 1, key)

    assert action.tolist() == [1, 0, 0, 0, 0]
    assert calls["load"] == ("policy.eqx", 4, None, 9)
    search_call = calls["search"]
    assert search_call[0] is sentinel_network
    assert search_call[1] is state
    assert search_call[2] is key
    assert search_call[3:] == (1, 2, 3, 4, 0, 5.0, 6.0, 0.2)
