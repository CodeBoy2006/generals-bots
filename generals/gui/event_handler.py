from abc import ABC, abstractmethod
from enum import Enum

import jax.numpy as jnp
import pygame
from pygame.event import Event

from generals.core.action import create_action
from generals.core.config import Dimension

from .properties import GuiMode, Properties


class Keybindings(Enum):
    ### General ###
    Q = pygame.K_q  # Quit the game

    ### Replay ###
    RIGHT = pygame.K_RIGHT  # Increase speed
    LEFT = pygame.K_LEFT  # Decrease speed
    SPACE = pygame.K_SPACE  # Pause
    R = pygame.K_r  # Restart
    L = pygame.K_l  # Move forward one frame
    H = pygame.K_h  # Move back one frame

    ### Game ###
    P = pygame.K_p  # Pass current turn
    S = pygame.K_s  # Toggle split move
    ESCAPE = pygame.K_ESCAPE  # Cancel selected source cell


class Command:
    def __init__(self):
        self.quit: bool = False


class ReplayCommand(Command):
    def __init__(self):
        super().__init__()
        self.frame_change: int = 0
        self.speed_change: float = 1.0
        self.restart: bool = False
        self.pause_toggle: bool = False


class GameCommand(Command):
    def __init__(self):
        super().__init__()
        self.action: jnp.ndarray | None = None
        self.restart: bool = False
        self.cancel_selection: bool = False
        self.selected_cell: tuple[int, int] | None = None
        self.split_enabled: bool = False


class TrainCommand(Command):
    def __init__(self):
        super().__init__()


class EventHandler(ABC):
    def __init__(self, properties: Properties):
        """
        Initialize the event handler.

        Args:
            properties: the Properties object
        """
        self.properties = properties

    @property
    @abstractmethod
    def command(self) -> Command:
        raise NotImplementedError

    @abstractmethod
    def reset_command(self):
        raise NotImplementedError

    @staticmethod
    def from_mode(mode: GuiMode, properties: Properties) -> "EventHandler":
        match mode:
            case GuiMode.TRAIN:
                return TrainEventHandler(properties)
            case GuiMode.GAME:
                return GameEventHandler(properties)
            case GuiMode.REPLAY:
                return ReplayEventHandler(properties)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def handle_events(self) -> Command:
        """
        Handle pygame GUI events
        """
        self.reset_command()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.command.quit = True
            if event.type == pygame.KEYDOWN:
                self.handle_key_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_event(event)
        return self.command

    def is_click_on_agents_row(self, x: int, y: int, i: int) -> bool:
        """
        Check if the click is on an agent's row.

        Args:
            x: int, x-coordinate of the click
            y: int, y-coordinate of the click
            i: int, index of the row
        """
        return (
            x >= self.properties.display_grid_width
            and (i + 1) * Dimension.GUI_CELL_HEIGHT.value <= y < (i + 2) * Dimension.GUI_CELL_HEIGHT.value
        )

    def toggle_player_fov(self):
        agents = self.properties.game.agents
        agent_fov = self.properties.agent_fov

        x, y = pygame.mouse.get_pos()
        for i, agent in enumerate(agents):
            if self.is_click_on_agents_row(x, y, i):
                agent_fov[agent] = not agent_fov[agent]
                break

    @abstractmethod
    def handle_key_event(self, event: Event) -> Command:
        raise NotImplementedError

    @abstractmethod
    def handle_mouse_event(self, event: Event):
        raise NotImplementedError


class ReplayEventHandler(EventHandler):
    def __init__(self, properties: Properties):
        super().__init__(properties)
        self._command = ReplayCommand()

    @property
    def command(self) -> ReplayCommand:
        return self._command

    def reset_command(self):
        self._command = ReplayCommand()

    def handle_key_event(self, event: Event) -> ReplayCommand:
        match event.key:
            case Keybindings.Q.value:
                self.command.quit = True
            case Keybindings.RIGHT.value:
                self.command.speed_change = 2.0
            case Keybindings.LEFT.value:
                self.command.speed_change = 0.5
            case Keybindings.SPACE.value:
                self.command.pause_toggle = True
            case Keybindings.R.value:
                self.command.restart = True
            case Keybindings.H.value:
                self.command.frame_change = -1
            case Keybindings.L.value:
                self.command.frame_change = 1
        return self.command

    def handle_mouse_event(self, event: Event) -> None:
        """
        Handle mouse clicks in replay mode.
        """
        self.toggle_player_fov()


class GameEventHandler(EventHandler):
    def __init__(self, properties: Properties):
        super().__init__(properties)
        self._command = GameCommand()
        self._selected_cell = properties.selected_cell
        self._split_next = properties.split_enabled
        self._sync_interaction_state()

    @property
    def command(self) -> GameCommand:
        return self._command

    def reset_command(self):
        self._command = GameCommand()
        self._sync_interaction_state()

    def handle_key_event(self, event: Event) -> GameCommand:
        match event.key:
            case Keybindings.Q.value:
                self.command.quit = True
            case Keybindings.P.value:
                self._selected_cell = None
                self.command.action = create_action(to_pass=True)
                self._sync_interaction_state("Pass queued")
            case Keybindings.S.value:
                self._split_next = not self._split_next
                self._sync_interaction_state(f"Split: {'On' if self._split_next else 'Off'}")
            case Keybindings.ESCAPE.value:
                self._selected_cell = None
                self.command.cancel_selection = True
                self._sync_interaction_state("Canceled")
            case Keybindings.R.value:
                self._selected_cell = None
                self.command.restart = True
                self._sync_interaction_state("Restart requested")
            case _:
                self._sync_interaction_state()
        return self.command

    def handle_mouse_event(self, event: Event) -> None:
        x, y = event.pos
        if x >= self.properties.display_grid_width:
            self.toggle_player_fov()
            return

        if event.button == 3:
            self._selected_cell = None
            self.command.cancel_selection = True
            self._sync_interaction_state("Canceled")
            return

        if event.button != 1:
            return

        clicked_cell = self._cell_from_pos(x, y)
        if clicked_cell is None:
            return

        if self._selected_cell is None:
            if self._is_valid_source(clicked_cell):
                self._selected_cell = clicked_cell
                self._sync_interaction_state(f"Selected: {clicked_cell}")
            else:
                self._sync_interaction_state("Invalid source")
        else:
            action = self._action_from_selection(self._selected_cell, clicked_cell)
            if action is not None:
                self.command.action = action
                self._selected_cell = None
                self._sync_interaction_state("Move queued")
            elif self._is_valid_source(clicked_cell):
                self._selected_cell = clicked_cell
                self._sync_interaction_state(f"Selected: {clicked_cell}")
            else:
                self._sync_interaction_state("Invalid target")

    def _cell_from_pos(self, x: int, y: int) -> tuple[int, int] | None:
        square_size = Dimension.SQUARE_SIZE.value
        row = y // square_size
        col = x // square_size
        if 0 <= row < self.properties.grid_height and 0 <= col < self.properties.grid_width:
            return row, col
        return None

    def _is_valid_source(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        agent_id = self.properties.game.agents[self.properties.human_player]
        channels = self.properties.game.channels
        return bool(channels.ownership[agent_id][row, col]) and int(channels.armies[row, col]) > 1

    def _action_from_selection(self, source: tuple[int, int], target: tuple[int, int]) -> jnp.ndarray | None:
        row, col = source
        target_row, target_col = target
        direction_by_delta = {
            (-1, 0): 0,
            (1, 0): 1,
            (0, -1): 2,
            (0, 1): 3,
        }
        direction = direction_by_delta.get((target_row - row, target_col - col))
        if direction is None:
            return None
        if bool(self.properties.game.channels.mountains[target_row, target_col]):
            return None
        return create_action(row=row, col=col, direction=direction, to_split=self._split_next)

    def _sync_interaction_state(self, message: str | None = None) -> None:
        self.properties.selected_cell = self._selected_cell
        self.properties.split_enabled = self._split_next
        if message is not None:
            self.properties.last_game_message = message
        self.command.selected_cell = self.properties.selected_cell
        self.command.split_enabled = self.properties.split_enabled


class TrainEventHandler(EventHandler):
    def __init__(self, properties: Properties):
        super().__init__(properties)
        self._command = TrainCommand()

    @property
    def command(self) -> TrainCommand:
        return self._command

    def reset_command(self):
        self._command = TrainCommand()

    def handle_key_event(self, event: Event) -> TrainCommand:
        if event.key == Keybindings.Q.value:
            self.command.quit = True
        return self.command

    def handle_mouse_event(self, event: Event) -> None:
        self.toggle_player_fov()
