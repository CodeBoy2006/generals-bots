import jax.numpy as jnp
import pygame

from generals.core import game
from generals.core.rendering import JaxGameAdapter
from generals.gui.event_handler import GameEventHandler
from generals.gui.properties import GuiMode, Properties


def make_handler(state):
    adapter = JaxGameAdapter(state, ["Human", "PPO"], game.get_info(state))
    properties = Properties(
        adapter,
        {"Human": {"color": (220, 55, 55)}, "PPO": {"color": (40, 90, 220)}},
        GuiMode.GAME,
        human_player=0,
    )
    return GameEventHandler(properties)


def make_state():
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[3, 3].set(2)
    state = game.create_initial_state(grid)
    return state._replace(armies=state.armies.at[0, 0].set(5))


def mouse_event(row, col, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (col * 50 + 25, row * 50 + 25), "button": button})


def key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, {"key": key})


def test_click_source_then_adjacent_target_creates_move_action():
    handler = make_handler(make_state())

    handler.handle_mouse_event(mouse_event(0, 0))
    assert handler.command.action is None
    assert handler.command.selected_cell == (0, 0)

    handler.reset_command()
    handler.handle_mouse_event(mouse_event(0, 1))

    assert handler.command.action.tolist() == [0, 0, 0, 3, 0]
    assert handler.command.selected_cell is None


def test_split_toggle_applies_to_next_move():
    handler = make_handler(make_state())

    handler.handle_key_event(key_event(pygame.K_s))
    handler.reset_command()
    handler.handle_mouse_event(mouse_event(0, 0))
    handler.reset_command()
    handler.handle_mouse_event(mouse_event(1, 0))

    assert handler.command.action.tolist() == [0, 0, 0, 1, 1]


def test_pass_and_cancel_commands_do_not_require_mouse_selection():
    handler = make_handler(make_state())

    handler.handle_key_event(key_event(pygame.K_p))
    assert handler.command.action.tolist() == [1, 0, 0, 0, 0]

    handler.reset_command()
    handler.handle_mouse_event(mouse_event(0, 0))
    assert handler.command.selected_cell == (0, 0)

    handler.reset_command()
    handler.handle_key_event(key_event(pygame.K_ESCAPE))
    assert handler.command.cancel_selection is True
    assert handler.command.selected_cell is None


def test_invalid_source_click_does_not_create_action():
    handler = make_handler(make_state())

    handler.handle_mouse_event(mouse_event(1, 1))

    assert handler.command.action is None
    assert handler.command.selected_cell is None
