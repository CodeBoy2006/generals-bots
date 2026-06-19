"""Authoritative game sessions for the browser renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
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
    max_steps: int | None = None
    seed: int = 43
    preview_top_k: int = 3
    ai_preview: bool = True
    policy_input: str = "auto"
    model_0_policy_input: str = "auto"
    model_1_policy_input: str = "auto"
    input_channels: int | None = None
    model_0_input_channels: int | None = None
    model_1_input_channels: int | None = None
    search_policy: bool = False
    opponent_search_policy: bool = False
    search_rollout_policy_mode: str = "sample"
    search_top_k: int = 4
    search_rollout_steps: int = 16
    search_rollouts_per_action: int = 4
    search_army_weight: float = 12.0
    search_land_weight: float = 8.0
    search_prior_weight: float = 0.01
    mountain_density_min: float = 0.12
    mountain_density_max: float = 0.22
    num_cities_min: int = 4
    num_cities_max: int = 8
    min_generals_distance: int | None = None
    max_generals_distance: int | None = None
    city_army_min: int = 40
    city_army_max: int = 51


@dataclass(frozen=True, slots=True)
class QueuedMove:
    """One pending human action in the browser move queue."""

    source: tuple[int, int] | None
    target: tuple[int, int] | None
    split: bool = False
    is_pass: bool = False


class WebGameSession:
    """Mutable, single-match session used by the WebSocket server."""

    def __init__(
        self,
        *,
        initial_grid: jnp.ndarray | None = None,
        grid_factory: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
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
        max_steps: int | None = None,
        preview_top_k: int = 3,
        ai_preview: bool = False,
    ):
        if initial_grid is None and grid_factory is None:
            raise ValueError("initial_grid or grid_factory is required")
        self.initial_grid = initial_grid
        self.grid_factory = grid_factory
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
        self.move_queue: list[QueuedMove] = []
        self.split_enabled = False
        self.last_message = "Ready"
        self.step_count = 0
        self.last_tick = 0.0
        self.policy_preview: PolicyPreview | None = None
        self.new_game(message="Ready")

    @classmethod
    def from_config(cls, config: WebSessionConfig) -> "WebGameSession":
        """Build a session from CLI/server configuration."""
        from generals.agents.ppo_runtime import make_grid, make_player_names, make_policy_agent, make_search_config

        args = _config_namespace(config)
        key = jrandom.PRNGKey(config.seed)
        names = make_player_names(config.human_player, machine_vs_machine=config.machine_vs_machine)
        colors = ["#dc3737", "#285adc"]
        search_config = make_search_config(args)

        def grid_factory(map_key: jnp.ndarray) -> jnp.ndarray:
            return make_grid(args, map_key)

        if config.machine_vs_machine:
            model_path = config.model_0_path or config.model_path
            opponent_model_path = config.model_1_path or model_path
            if model_path is None:
                raise ValueError("model_path or model_0_path is required")
            if opponent_model_path is None:
                raise ValueError("model_1_path or opponent_model_path is required")
            opponent_policy_mode = config.opponent_policy_mode or config.policy_mode
            machine_agents = (
                make_policy_agent(
                    model_path,
                    config.grid_size,
                    config.policy_mode,
                    names[0],
                    config.model_0_policy_input,
                    config.model_0_input_channels,
                    config.search_policy,
                    search_config,
                ),
                make_policy_agent(
                    opponent_model_path,
                    config.grid_size,
                    opponent_policy_mode,
                    names[1],
                    config.model_1_policy_input,
                    config.model_1_input_channels,
                    config.opponent_search_policy,
                    search_config,
                ),
            )
            return cls(
                grid_factory=grid_factory,
                names=names,
                colors=colors,
                key=key,
                human_player=config.human_player,
                machine_vs_machine=True,
                machine_agents=machine_agents,
                policy_agent=machine_agents[0],
                preview_player=0,
                auto_tick=config.auto_tick,
                tick_rate=config.tick_rate,
                max_steps=config.max_steps,
                preview_top_k=config.preview_top_k,
                ai_preview=config.ai_preview,
            )

        model_path = config.model_path or config.model_0_path
        if model_path is None:
            raise ValueError("model_path or model_0_path is required")
        model_agent = make_policy_agent(
            model_path,
            config.grid_size,
            config.policy_mode,
            "PPO Model",
            config.model_0_policy_input,
            config.model_0_input_channels,
            config.search_policy,
            search_config,
        )
        return cls(
            grid_factory=grid_factory,
            names=names,
            colors=colors,
            key=key,
            human_player=config.human_player,
            machine_vs_machine=False,
            model_agent=model_agent,
            policy_agent=model_agent,
            preview_player=1 - config.human_player,
            auto_tick=config.auto_tick,
            tick_rate=config.tick_rate,
            max_steps=config.max_steps,
            preview_top_k=config.preview_top_k,
            ai_preview=config.ai_preview,
        )

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
        max_steps: int | None = None,
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
                max_steps=max_steps,
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
            max_steps=max_steps,
            ai_preview=False,
        )

    def new_game(self, message: str = "Restarted") -> dict[str, Any]:
        """Reset this session to its initial grid."""
        if self.grid_factory is None:
            grid = self.initial_grid
        else:
            self.key, map_key = jrandom.split(self.key)
            grid = self.grid_factory(map_key)
        self.state = create_initial_state(grid)
        self.info = game.get_info(self.state)
        self.selected_cell = None
        self.move_queue.clear()
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
            queued_moves=self._serialized_queue(),
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
            self._queue_move((int(source[0]), int(source[1])), (int(target[0]), int(target[1])), bool(command["split"]))
            return self.snapshot()
        if command_type == "pass":
            self.selected_cell = None
            self.move_queue.append(QueuedMove(source=None, target=None, is_pass=True))
            self.last_message = "Pass queued"
            return self.snapshot()
        if command_type == "undo_queue":
            self._undo_queue()
            return self.snapshot()
        if command_type == "clear_queue":
            self._clear_queue()
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

        queued_action = self._pop_next_queued_action()
        if queued_action is not None:
            action, is_pass = queued_action
            self._advance_human_turn(action, now=now)
            self.last_message = "Queued pass executed" if is_pass else "Queued move executed"
            return self.snapshot()

        self._advance_human_turn(create_action(to_pass=True), now=now)
        self._clear_invalid_selection()
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

    def _queue_move(self, source: tuple[int, int], target: tuple[int, int], split: bool) -> None:
        projected_state = self._projected_queue_state()
        if not self._is_valid_source_in_state(projected_state, source):
            self.selected_cell = None
            self.last_message = "Invalid source"
            return

        direction = self._direction(source, target)
        if direction is None or target not in self._valid_targets_in_state(projected_state, source):
            self.selected_cell = source
            self.last_message = "Invalid target"
            return

        self.move_queue.append(QueuedMove(source=source, target=target, split=split))
        self.selected_cell = target
        self.last_message = "Move queued"

    def _undo_queue(self) -> None:
        if not self.move_queue:
            self.last_message = "Move queue empty"
            return
        move = self.move_queue.pop()
        self.selected_cell = move.source
        self.last_message = "Queued pass undone" if move.is_pass else "Queued move undone"

    def _clear_queue(self) -> None:
        if not self.move_queue:
            self.selected_cell = None
            self.last_message = "Move queue empty"
            return
        self.move_queue.clear()
        self.selected_cell = None
        self.last_message = "Move queue cleared"

    def _clear_invalid_selection(self) -> None:
        if self.selected_cell is not None and not self._is_valid_source(self.selected_cell):
            self.selected_cell = None

    def _advance_human_turn(self, human_action: jnp.ndarray, now: float | None = None) -> None:
        if self.model_agent is None:
            self.last_message = "No model agent"
            return
        self.key, action_key = jrandom.split(self.key)
        model_action = self._choose_agent_action(self.model_agent, self.model_player, action_key)
        actions = self._stack_human_and_model_actions(human_action, model_action)
        self._advance(actions, now)

    def _advance(self, actions: jnp.ndarray, now: float | None) -> None:
        self.state, self.info = game.step(self.state, actions)
        self.step_count += 1
        if now is not None:
            self.last_tick = now

    def _stack_human_and_model_actions(self, human_action: jnp.ndarray, model_action: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.stack((human_action, model_action))
            if self.human_player == 0
            else jnp.stack((model_action, human_action))
        )

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
        return self._is_valid_source_in_state(self._projected_queue_state(), cell)

    def _is_valid_source_in_state(self, state: game.GameState, cell: tuple[int, int]) -> bool:
        row, col = cell
        height, width = state.armies.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return bool(state.ownership[self.human_player, row, col]) and int(state.armies[row, col]) > 1

    def _valid_targets(self, selected_cell: tuple[int, int] | None) -> list[tuple[int, int]]:
        return self._valid_targets_in_state(self._projected_queue_state(), selected_cell)

    def _valid_targets_in_state(self, state: game.GameState, selected_cell: tuple[int, int] | None) -> list[tuple[int, int]]:
        if selected_cell is None:
            return []
        row, col = selected_cell
        height, width = state.armies.shape
        targets = []
        for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target_row = row + row_delta
            target_col = col + col_delta
            if (
                0 <= target_row < height
                and 0 <= target_col < width
                and not bool(state.mountains[target_row, target_col])
            ):
                targets.append((target_row, target_col))
        return targets

    def _projected_queue_state(self) -> game.GameState:
        projected = self.state
        model_pass = create_action(to_pass=True)
        for move in self.move_queue:
            action = self._action_for_queued_move(move, projected)
            if action is None:
                continue
            actions = self._stack_human_and_model_actions(action, model_pass)
            projected, _ = game.step(projected, actions)
        return projected

    def _action_for_queued_move(self, move: QueuedMove, state: game.GameState) -> jnp.ndarray | None:
        if move.is_pass:
            return create_action(to_pass=True)
        if move.source is None or move.target is None:
            return None
        if not self._is_valid_source_in_state(state, move.source):
            return None
        direction = self._direction(move.source, move.target)
        if direction is None or move.target not in self._valid_targets_in_state(state, move.source):
            return None
        return create_action(
            row=move.source[0],
            col=move.source[1],
            direction=direction,
            to_split=move.split,
        )

    def _pop_next_queued_action(self) -> tuple[jnp.ndarray, bool] | None:
        while self.move_queue:
            move = self.move_queue.pop(0)
            action = self._action_for_queued_move(move, self.state)
            if action is not None:
                return action, move.is_pass
        return None

    def _serialized_queue(self) -> list[dict[str, Any]]:
        return [
            {
                "source": _cell_payload(move.source),
                "target": _cell_payload(move.target),
                "split": bool(move.split),
                "is_pass": bool(move.is_pass),
            }
            for move in self.move_queue
        ]

    def _auto_tick_due(self, now: float) -> bool:
        if not self.auto_tick_enabled or self.tick_rate <= 0:
            return False
        return now - self.last_tick >= 1.0 / self.tick_rate

    def _game_done(self) -> bool:
        return bool(self.info.is_done) or self._reached_limit()

    def _reached_limit(self) -> bool:
        return self.max_steps is not None and self.step_count >= self.max_steps and int(self.info.winner) < 0

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


def _cell_payload(cell: tuple[int, int] | None) -> list[int] | None:
    if cell is None:
        return None
    return [int(cell[0]), int(cell[1])]


def _config_namespace(config: WebSessionConfig) -> SimpleNamespace:
    effective_min_generals_distance = config.min_generals_distance
    if effective_min_generals_distance is None:
        effective_min_generals_distance = max(3, config.grid_size // 2)
    return SimpleNamespace(
        grid_size=config.grid_size,
        map_generator=config.map_generator,
        mountain_density_min=config.mountain_density_min,
        mountain_density_max=config.mountain_density_max,
        num_cities_min=config.num_cities_min,
        num_cities_max=config.num_cities_max,
        effective_min_generals_distance=effective_min_generals_distance,
        max_generals_distance=config.max_generals_distance,
        city_army_min=config.city_army_min,
        city_army_max=config.city_army_max,
        search_rollout_policy_mode=config.search_rollout_policy_mode,
        search_top_k=config.search_top_k,
        search_rollout_steps=config.search_rollout_steps,
        search_rollouts_per_action=config.search_rollouts_per_action,
        search_army_weight=config.search_army_weight,
        search_land_weight=config.search_land_weight,
        search_prior_weight=config.search_prior_weight,
    )
