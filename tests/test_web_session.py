import jax.numpy as jnp

from generals.core import game
from generals.web.session import WebGameSession


class FixedAgent:
    def __init__(self, action):
        self.action = jnp.array(action, dtype=jnp.int32)

    def act(self, observation, key):
        return self.action


def _basic_grid():
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    return grid


def _human_session() -> WebGameSession:
    session = WebGameSession.for_testing(
        grid=_basic_grid(),
        names=["Human", "PPO Model"],
        agents=(FixedAgent([1, 0, 0, 0, 0]),),
        human_player=0,
        auto_tick=True,
        tick_rate=2.0,
    )
    session.state = session.state._replace(armies=session.state.armies.at[0, 0].set(5))
    session.info = game.get_info(session.state)
    return session


def test_select_and_move_command_queues_until_tick_executes_it():
    session = _human_session()

    select_snapshot = session.submit_client_command({"type": "select", "row": 0, "col": 0})
    assert select_snapshot["selected_cell"] == [0, 0]
    assert select_snapshot["last_message"] == "Selected: (0, 0)"

    move_snapshot = session.submit_client_command({"type": "move", "source": [0, 0], "target": [0, 1], "split": False})
    assert move_snapshot["time"] == 0
    assert move_snapshot["step_count"] == 0
    assert move_snapshot["selected_cell"] == [0, 1]
    assert move_snapshot["last_message"] == "Move queued"
    assert move_snapshot["queued_moves"] == [
        {"source": [0, 0], "target": [0, 1], "split": False, "is_pass": False}
    ]
    assert move_snapshot["grid"]["ownership"][0][1] == -1

    session.last_tick = 0.0
    executed_snapshot = session.tick(now=1.0)
    assert executed_snapshot["time"] == 1
    assert executed_snapshot["step_count"] == 1
    assert executed_snapshot["queued_moves"] == []
    assert executed_snapshot["last_message"] == "Queued move executed"
    assert executed_snapshot["grid"]["ownership"][0][1] == 0


def test_invalid_source_and_invalid_target_preserve_selection_state():
    session = _human_session()

    invalid_source = session.submit_client_command({"type": "select", "row": 1, "col": 1})
    assert invalid_source["selected_cell"] is None
    assert invalid_source["last_message"] == "Invalid source"

    session.submit_client_command({"type": "select", "row": 0, "col": 0})
    invalid_target = session.submit_client_command(
        {"type": "move", "source": [0, 0], "target": [2, 2], "split": False}
    )
    assert invalid_target["selected_cell"] == [0, 0]
    assert invalid_target["last_message"] == "Invalid target"
    assert invalid_target["time"] == 0


def test_queued_moves_can_chain_from_projected_targets_and_be_edited():
    session = _human_session()
    session.submit_client_command({"type": "select", "row": 0, "col": 0})

    first = session.submit_client_command({"type": "move", "source": [0, 0], "target": [0, 1], "split": False})
    assert first["selected_cell"] == [0, 1]
    assert [0, 2] in first["valid_targets"]

    second = session.submit_client_command({"type": "move", "source": [0, 1], "target": [0, 2], "split": True})
    assert second["time"] == 0
    assert second["selected_cell"] == [0, 2]
    assert second["queued_moves"] == [
        {"source": [0, 0], "target": [0, 1], "split": False, "is_pass": False},
        {"source": [0, 1], "target": [0, 2], "split": True, "is_pass": False},
    ]

    undone = session.submit_client_command({"type": "undo_queue"})
    assert undone["selected_cell"] == [0, 1]
    assert undone["queued_moves"] == [
        {"source": [0, 0], "target": [0, 1], "split": False, "is_pass": False}
    ]
    assert undone["last_message"] == "Queued move undone"

    cleared = session.submit_client_command({"type": "clear_queue"})
    assert cleared["selected_cell"] is None
    assert cleared["queued_moves"] == []
    assert cleared["last_message"] == "Move queue cleared"


def test_split_pass_cancel_and_restart_commands_update_session_state():
    session = _human_session()

    split_snapshot = session.submit_client_command({"type": "set_split", "enabled": True})
    assert split_snapshot["split_enabled"] is True
    assert split_snapshot["last_message"] == "Split: On"

    pass_snapshot = session.submit_client_command({"type": "pass"})
    assert pass_snapshot["time"] == 0
    assert pass_snapshot["selected_cell"] is None
    assert pass_snapshot["last_message"] == "Pass queued"
    assert pass_snapshot["queued_moves"] == [{"source": None, "target": None, "split": False, "is_pass": True}]

    session.last_tick = 0.0
    executed_pass = session.tick(now=1.0)
    assert executed_pass["time"] == 1
    assert executed_pass["queued_moves"] == []
    assert executed_pass["last_message"] == "Queued pass executed"

    session.submit_client_command({"type": "select", "row": 0, "col": 0})
    cancel_snapshot = session.submit_client_command({"type": "cancel"})
    assert cancel_snapshot["selected_cell"] is None
    assert cancel_snapshot["last_message"] == "Canceled"

    restart_snapshot = session.submit_client_command({"type": "restart"})
    assert restart_snapshot["time"] == 0
    assert restart_snapshot["step_count"] == 0
    assert restart_snapshot["queued_moves"] == []
    assert restart_snapshot["last_message"] == "Restarted"


def test_auto_tick_pauses_while_source_is_selected_and_passes_when_idle():
    session = _human_session()
    session.last_tick = 0.0

    session.submit_client_command({"type": "select", "row": 0, "col": 0})
    selected_snapshot = session.tick(now=1.0)
    assert selected_snapshot["time"] == 0
    assert selected_snapshot["selected_cell"] == [0, 0]

    session.submit_client_command({"type": "cancel"})
    idle_snapshot = session.tick(now=1.0)
    assert idle_snapshot["time"] == 1
    assert idle_snapshot["last_message"] == "Auto pass"


def test_machine_tick_uses_both_agents_and_advances_time():
    session = WebGameSession.for_testing(
        grid=_basic_grid(),
        names=["PPO 0", "PPO 1"],
        agents=(FixedAgent([1, 0, 0, 0, 0]), FixedAgent([1, 0, 0, 0, 0])),
        machine_vs_machine=True,
        auto_tick=True,
        tick_rate=4.0,
    )
    session.last_tick = 0.0

    snapshot = session.tick(now=0.3)

    assert snapshot["mode"] == "machine-vs-machine"
    assert snapshot["time"] == 1
    assert snapshot["step_count"] == 1
    assert snapshot["last_message"] == "Tick"
    assert all(all(row) for row in snapshot["grid"]["visible"])
