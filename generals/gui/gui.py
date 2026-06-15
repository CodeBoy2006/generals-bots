from typing import Any

import pygame

from generals.core.game import Game

from .event_handler import (
    Command,
    EventHandler,
    ReplayCommand,
)
from .properties import GuiMode, Properties
from .rendering import Renderer


class GUI:
    def __init__(
        self,
        game: Game,
        agent_data: dict[str, dict[str, Any]],
        mode: GuiMode = GuiMode.TRAIN,
        speed_multiplier: float = 1.0,
        show_tile_types: bool = False,
        human_player: int = 0,
    ):
        pygame.init()
        pygame.display.set_caption("Generals")

        # Handle key repeats
        pygame.key.set_repeat(500, 64)

        self.properties = Properties(game, agent_data, mode, speed_multiplier, human_player=human_player)
        self.properties.show_tile_types = show_tile_types
        self.__renderer = Renderer(self.properties)
        self.__event_handler = EventHandler.from_mode(self.properties.mode, self.properties)

    def tick(self, fps: int | None = None) -> Command:
        command = self.__event_handler.handle_events()
        if command.quit:
            quit()
        if isinstance(command, ReplayCommand):
            self.properties.update_speed(command.speed_change)
            if command.frame_change != 0 or command.restart:
                self.properties.paused = True
            if command.pause_toggle:
                self.properties.paused = not self.properties.paused
        self.__renderer.render(fps)
        return command

    def set_policy_preview(self, preview: Any) -> None:
        """Set the policy preview rendered by the game HUD."""
        self.properties.policy_preview = preview

    def clear_policy_preview(self) -> None:
        """Clear the policy preview rendered by the game HUD."""
        self.properties.policy_preview = None

    def close(self):
        pygame.quit()
