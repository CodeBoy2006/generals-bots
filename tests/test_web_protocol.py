import json

import jax.numpy as jnp

from generals.agents.ppo_policy_agent import PolicyActionCandidate, PolicyPreview
from generals.core import game
from generals.web.schemas import build_snapshot, serialize_policy_preview


def _state_with_army() -> game.GameState:
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    state = game.create_initial_state(grid)
    return state._replace(armies=state.armies.at[0, 0].set(5))


def test_snapshot_serializes_grid_players_and_human_visibility():
    state = _state_with_army()
    info = game.get_info(state)

    snapshot = build_snapshot(
        state=state,
        info=info,
        names=["Human", "PPO Model"],
        colors=["#dc3737", "#285adc"],
        mode="human-vs-model",
        visibility_player=0,
        step_count=0,
        selected_cell=(0, 0),
        split_enabled=False,
        last_message="Selected: (0, 0)",
        auto_tick_enabled=True,
        tick_rate=2.0,
        policy_preview=None,
        valid_targets=[(0, 1), (1, 0)],
        reached_limit=False,
    )

    assert snapshot["type"] == "snapshot"
    assert snapshot["mode"] == "human-vs-model"
    assert snapshot["grid"]["height"] == 4
    assert snapshot["grid"]["width"] == 4
    assert snapshot["grid"]["armies"][0][0] == 5
    assert snapshot["grid"]["ownership"][0][0] == 0
    assert snapshot["grid"]["ownership"][3][3] == 1
    assert snapshot["grid"]["ownership"][1][1] == -1
    assert snapshot["grid"]["visible"][0][1] is True
    assert snapshot["grid"]["visible"][3][3] is False
    assert snapshot["players"] == [
        {"index": 0, "name": "Human", "army": 5, "land": 1, "color": "#dc3737"},
        {"index": 1, "name": "PPO Model", "army": 1, "land": 1, "color": "#285adc"},
    ]
    assert snapshot["time"] == 0
    assert snapshot["step_count"] == 0
    assert snapshot["winner"] is None
    assert snapshot["game_done"] is False
    assert snapshot["selected_cell"] == [0, 0]
    assert snapshot["valid_targets"] == [[0, 1], [1, 0]]
    assert snapshot["split_enabled"] is False
    assert snapshot["last_message"] == "Selected: (0, 0)"
    assert snapshot["auto_tick"] == {"enabled": True, "tick_rate": 2.0}
    assert snapshot["policy_preview"] is None
    json.dumps(snapshot)


def test_snapshot_full_visibility_shows_entire_machine_watch_board():
    state = _state_with_army()
    info = game.get_info(state)

    snapshot = build_snapshot(
        state=state,
        info=info,
        names=["PPO 0", "PPO 1"],
        colors=["#dc3737", "#285adc"],
        mode="machine-vs-machine",
        visibility_player=None,
        step_count=4,
        selected_cell=None,
        split_enabled=True,
        last_message="Watching",
        auto_tick_enabled=True,
        tick_rate=4.0,
        policy_preview=None,
        valid_targets=[],
        reached_limit=False,
    )

    assert all(all(row) for row in snapshot["grid"]["visible"])
    assert snapshot["selected_cell"] is None
    assert snapshot["valid_targets"] == []
    assert snapshot["split_enabled"] is True
    assert snapshot["auto_tick"]["tick_rate"] == 4.0


def test_policy_preview_serializes_move_pass_value_and_sample_note():
    move = PolicyActionCandidate(
        action=(0, 1, 1, 3, 1),
        probability=0.49,
        source=(1, 1),
        target=(1, 2),
        direction=3,
        direction_label="Right",
        is_split=True,
        is_pass=False,
    )
    pass_action = PolicyActionCandidate(
        action=(1, 0, 0, 0, 0),
        probability=0.09,
        source=None,
        target=None,
        direction=None,
        direction_label="Pass",
        is_split=False,
        is_pass=True,
    )
    preview = PolicyPreview(candidates=(move, pass_action), value=0.18, policy_mode="sample")

    serialized = serialize_policy_preview(preview)

    assert serialized == {
        "candidates": [
            {
                "action": [0, 1, 1, 3, 1],
                "probability": 0.49,
                "source": [1, 1],
                "target": [1, 2],
                "direction": 3,
                "direction_label": "Right",
                "is_split": True,
                "is_pass": False,
            },
            {
                "action": [1, 0, 0, 0, 0],
                "probability": 0.09,
                "source": None,
                "target": None,
                "direction": None,
                "direction_label": "Pass",
                "is_split": False,
                "is_pass": True,
            },
        ],
        "value": 0.18,
        "policy_mode": "sample",
    }
    json.dumps(serialized)
