"""Authoritative game sessions for the browser renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import PolicyPreview
from generals.core import game
from generals.core.action import create_action
from generals.core.game import create_initial_state

from .schemas import build_snapshot


class WebAgent(Protocol):
    def act(self, observation: Any, key: jnp.ndarray) -> jnp.ndarray:
        """Return one public Generals action."""


@dataclass
class WebSessionConfig:
    """Launch settings for one browser game session."""

    model_path: str | None = None
    model_0_path: str | None = None
    model_1_path: str | None = None
    grid_size: int = 8
    map_generator: str = "generated"
    policy_mode: str = "sample"
    opponent_policy_mode: str | None = None
    machine_vs_machine: bool = False
    human_player: int = 0
    auto_tick: bool = True
    tick_rate: float = 2.0
    max_steps: int = 500
    seed: int = 43
    preview_top_k: int = 3
    ai_preview: bool = True


class WebGameSession:
    """Mutable, single-match session used by the WebSocket server."""

    def __init__(
        self,
        *,
        initial_grid: jnp.ndarray,
        names: list[str],
        colors: list[str],
        key: jnp.ndarray,
        human_player: int = 0,
        machine_vs_machine: bool = False,
        model_agent: WebAgent | None = None,
        machine_agents: tuple[WebAgent, WebAgent] | None = None,
        policy_agent: Any | None = None,
        preview_player: int = 0,
        auto_tick: bool = True,
        tick_rate: float = 2.0,
        max_steps: int = 500,
        preview_top_k: int = 3,
        ai_preview: bool = False,
    ):
        self.initial_grid = initial_grid
        self.names = names
        self.colors = colors
        self.key = key
        self.human_player = human_player
        self.machine_vs_machine = machine_vs_machine
        self.model_player = 1 - human_player
        self.model_agent = model_agent
        self.machine_agents = machine_agents
        self.policy_agent = policy_agent or model_agent or (machine_agents[0] if machine_agents else None)
        self.preview_player = preview_player
        self.auto_tick_enabled = auto_tick
        self.tick_rate = tick_rate
        self.max_steps = max_steps
        self.preview_top_k = preview_top_k
        self.ai_preview = ai_preview

        self.selected_cell: tuple[int, int] | None = None
        self.split_enabled = False
        self.last_message = "Ready"
        self.step_count = 0
        self.last_tick = 0.0
        self.policy_preview: PolicyPreview | None = None
        self.new_game(message="Ready")

    @classmethod
    def for_testing(
        cls,
        *,
        grid: jnp.ndarray,
        names: list[str],
        agents: tuple[WebAgent, ...],
        human_player: int = 0,
        machine_vs_machine: bool = False,
        auto_tick: bool = True,
        tick_rate: float = 2.0,
    ) -> "WebGameSession":
        colors = ["#dc3737", "#285adc"]
        if machine_vs_machine:
            if len(agents) != 2:
                raise ValueError("machine-vs-machine test sessions require two agents")
            return cls(
                initial_grid=grid,
                names=names,
                colors=colors,
                key=jrandom.PRNGKey(0),
                human_player=human_player,
                machine_vs_machine=True,
                machine_agents=(agents[0], agents[1]),
                preview_player=0,
                auto_tick=auto_tick,
                tick_rate=tick_rate,
                ai_preview=False,
            )

        if len(agents) != 1:
            raise ValueError("human test sessions require one model agent")
        return cls(
            initial_grid=grid,
            names=names,
            colors=colors,
            key=jrandom.PRNGKey(0),
            human_player=human_player,
            machine_vs_machine=False,
            model_agent=agents[0],
            preview_player=1 - human_player,
            auto_tick=auto_tick,
            tick_rate=tick_rate,
            ai_preview=False,
        )

    def new_game(self, message: str = "Restarted") -> dict[str, Any]:
        """Reset this session to its initial grid."""
        self.state = create_initial_state(self.initial_grid)
        self.info = game.get_info(self.state)
        self.selected_cell = None
        self.split_enabled = False
        self.last_message = message
        self.step_count = 0
        self.policy_preview = None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for the current state."""
        self._refresh_policy_preview()
        visibility_player = None if self.machine_vs_machine else self.human_player
        return build_snapshot(
            state=self.state,
            info=self.info,
            names=self.names,
            colors=self.colors,
            mode="machine-vs-machine" if self.machine_vs_machine else "human-vs-model",
            visibility_player=visibility_player,
            step_count=self.step_count,
            selected_cell=self.selected_cell,
            split_enabled=self.split_enabled,
            last_message=self.last_message,
            auto_tick_enabled=self.auto_tick_enabled,
            tick_rate=self.tick_rate,
            policy_preview=self.policy_preview,
            valid_targets=self._valid_targets(self.selected_cell),
            reached_limit=self._reached_limit(),
        )

    def submit_client_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Apply one semantic browser command and return the resulting snapshot."""
        command_type = command.get("type")
        if command_type == "select":
            self._select((int(command["row"]), int(command["col"])))
            return self.snapshot()
        if command_type == "move":
            source = tuple(command["source"])
            target = tuple(command["target"])
            self._move((int(source[0]), int(source[1])), (int(target[0]), int(target[1])), bool(command["split"]))
            return self.snapshot()
        if command_type == "pass":
            self.selected_cell = None
            self._advance_human_turn(create_action(to_pass=True))
            self.last_message = "Pass queued"
            return self.snapshot()
        if command_type == "cancel":
            self.selected_cell = None
            self.last_message = "Canceled"
            return self.snapshot()
        if command_type == "set_split":
            self.split_enabled = bool(command["enabled"])
            self.last_message = f"Split: {'On' if self.split_enabled else 'Off'}"
            return self.snapshot()
        if command_type == "set_auto_tick":
            self.auto_tick_enabled = bool(command["enabled"])
            self.tick_rate = float(command["tick_rate"])
            self.last_message = f"Auto tick: {'On' if self.auto_tick_enabled else 'Off'}"
            return self.snapshot()
        if command_type == "restart":
            return self.new_game(message="Restarted")

        self.last_message = f"Unknown command: {command_type}"
        return self.snapshot()

    def tick(self, now: float) -> dict[str, Any]:
        """Advance an automatic tick if due, otherwise return the current snapshot."""
        if self._game_done() or not self._auto_tick_due(now):
            return self.snapshot()

        if self.machine_vs_machine:
            if self.machine_agents is None:
                self.last_message = "No machine agents"
                return self.snapshot()
            self.key, action_key = jrandom.split(self.key)
            actions = self._choose_machine_actions(action_key)
            self._advance(actions, now)
            self.last_message = "Tick"
            return self.snapshot()

        self._advance_human_turn(create_action(to_pass=True), now=now)
        self.last_message = "Auto pass"
        return self.snapshot()

    def _refresh_policy_preview(self) -> None:
        if not self.ai_preview or self._game_done() or self.policy_agent is None:
            self.policy_preview = None
            return
        if hasattr(self.policy_agent, "explain_for_state"):
            self.policy_preview = self.policy_agent.explain_for_state(
                self.state,
                self.preview_player,
                top_k=self.preview_top_k,
            )
        elif hasattr(self.policy_agent, "explain"):
            self.policy_preview = self.policy_agent.explain(
                game.get_observation(self.state, self.preview_player),
                top_k=self.preview_top_k,
            )

    def _select(self, cell: tuple[int, int]) -> None:
        if self._is_valid_source(cell):
            self.selected_cell = cell
            self.last_message = f"Selected: {cell}"
            return
        self.selected_cell = None
        self.last_message = "Invalid source"

    def _move(self, source: tuple[int, int], target: tuple[int, int], split: bool) -> None:
        if not self._is_valid_source(source):
            self.selected_cell = None
            self.last_message = "Invalid source"
            return

        direction = self._direction(source, target)
        if direction is None or target not in self._valid_targets(source):
            self.selected_cell = source
            self.last_message = "Invalid target"
            return

        self.selected_cell = None
        self._advance_human_turn(create_action(row=source[0], col=source[1], direction=direction, to_split=split))
        self.last_message = "Move queued"

    def _advance_human_turn(self, human_action: jnp.ndarray, now: float | None = None) -> None:
        if self.model_agent is None:
            self.last_message = "No model agent"
            return
        self.key, action_key = jrandom.split(self.key)
        model_action = self._choose_agent_action(self.model_agent, self.model_player, action_key)
        actions = (
            jnp.stack((human_action, model_action))
            if self.human_player == 0
            else jnp.stack((model_action, human_action))
        )
        self._advance(actions, now)

    def _advance(self, actions: jnp.ndarray, now: float | None) -> None:
        self.state, self.info = game.step(self.state, actions)
        self.step_count += 1
        if now is not None:
            self.last_tick = now

    def _choose_machine_actions(self, key: jnp.ndarray) -> jnp.ndarray:
        if self.machine_agents is None:
            return jnp.stack([create_action(to_pass=True), create_action(to_pass=True)])
        key_0, key_1 = jrandom.split(key)
        return jnp.stack(
            [
                self._choose_agent_action(self.machine_agents[0], 0, key_0),
                self._choose_agent_action(self.machine_agents[1], 1, key_1),
            ]
        )

    def _choose_agent_action(self, agent: WebAgent, player: int, key: jnp.ndarray) -> jnp.ndarray:
        if hasattr(agent, "act_for_state"):
            return agent.act_for_state(self.state, player, key)
        return agent.act(game.get_observation(self.state, player), key)

    def _is_valid_source(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        height, width = self.state.armies.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return bool(self.state.ownership[self.human_player, row, col]) and int(self.state.armies[row, col]) > 1

    def _valid_targets(self, selected_cell: tuple[int, int] | None) -> list[tuple[int, int]]:
        if selected_cell is None:
            return []
        row, col = selected_cell
        height, width = self.state.armies.shape
        targets = []
        for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target_row = row + row_delta
            target_col = col + col_delta
            if (
                0 <= target_row < height
                and 0 <= target_col < width
                and not bool(self.state.mountains[target_row, target_col])
            ):
                targets.append((target_row, target_col))
        return targets

    def _auto_tick_due(self, now: float) -> bool:
        if not self.auto_tick_enabled or self.tick_rate <= 0:
            return False
        if not self.machine_vs_machine and self.selected_cell is not None:
            return False
        return now - self.last_tick >= 1.0 / self.tick_rate

    def _game_done(self) -> bool:
        return bool(self.info.is_done) or self._reached_limit()

    def _reached_limit(self) -> bool:
        return self.step_count >= self.max_steps and int(self.info.winner) < 0

    @staticmethod
    def _direction(source: tuple[int, int], target: tuple[int, int]) -> int | None:
        row, col = source
        target_row, target_col = target
        return {
            (-1, 0): 0,
            (1, 0): 1,
            (0, -1): 2,
            (0, 1): 3,
        }.get((target_row - row, target_col - col))
