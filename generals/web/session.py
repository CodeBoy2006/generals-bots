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
    adaptive_policy: bool = False
    opponent_adaptive_policy: bool = False
    pad_to: int = 16
    network_arch: str = "cnn"
    channels: str | None = None
    global_context: bool = False
    scoreboard_history: bool = False
    fog_memory: bool = False
    value_loss: str = "mse"
    value_bins: int = 128
    value_min: float = -1.0
    value_max: float = 1.0
    value_sigma: float = 0.04
    policy_adapter_path: str | None = None
    policy_adapter_scale: float = 0.0
    policy_adapter_mode: str = "delta"
    policy_adapter_min_grid_size: int = 0
    policy_adapter_max_grid_size: int = 0
    adaptive_online_search: bool = False
    online_search_min_turn: int = 0
    online_search_require_contact: bool = False
    online_search_min_grid_size: int = 0
    online_search_max_grid_size: int = 0
    online_search_terminal_score: float = 100.0
    online_search_min_score_gap: float = 0.0
    online_search_max_steps: int = 750
    adaptive_online_search_opponent_path: str | None = None
    adaptive_online_search_opponent_policy_mode: str = "sample"
    adaptive_online_search_opponent_channels: str | None = None
    adaptive_online_search_opponent_input_channels: int = 9
    mountain_density_min: float = 0.12
    mountain_density_max: float = 0.22
    num_cities_min: int = 4
    num_cities_max: int = 8
    min_generals_distance: int | None = None
    max_generals_distance: int | None = None
    city_army_min: int = 40
    city_army_max: int = 51
    model_catalog: list[dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class QueuedMove:
    """One pending human action in the browser move queue."""

    player: int
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
        agent_loader: Callable[[int, str], WebAgent] | None = None,
        policy_agent: Any | None = None,
        preview_player: int = 0,
        player_controls: tuple[str, str] | list[str] | None = None,
        active_human_player: int | None = None,
        player_model_ids: tuple[str | None, str | None] | list[str | None] | None = None,
        model_catalog: list[dict[str, Any]] | None = None,
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
        self.agent_loader = agent_loader
        self.model_agents: list[WebAgent | None] = list(machine_agents) if machine_agents else [None, None]
        if model_agent is not None:
            self.model_agents[self.model_player] = model_agent
        default_controls = ["model", "model"] if machine_vs_machine else ["model", "model"]
        if not machine_vs_machine:
            default_controls[human_player] = "human"
        self.player_controls = list(player_controls or default_controls)
        self.player_model_ids = list(player_model_ids or [None, None])
        self.model_catalog = [dict(model) for model in model_catalog or []]
        self.agent_cache: dict[tuple[int, str], WebAgent] = {}
        self.active_human_player = (
            active_human_player
            if active_human_player is not None and self.player_controls[active_human_player] == "human"
            else self._first_human_player()
        )
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
        from generals.agents.ppo_runtime import (
            AdaptiveRuntimeConfig,
            make_grid,
            make_player_names,
            make_policy_agent,
            make_search_config,
        )

        args = _config_namespace(config)
        key = jrandom.PRNGKey(config.seed)
        names = make_player_names(config.human_player, machine_vs_machine=config.machine_vs_machine)
        colors = ["#dc3737", "#285adc"]
        search_config = make_search_config(args)

        def grid_factory(map_key: jnp.ndarray) -> jnp.ndarray:
            return make_grid(args, map_key)

        primary_model_path = config.model_path or config.model_0_path
        if primary_model_path is None:
            raise ValueError("model_path or model_0_path is required")
        player_model_ids = [
            config.model_0_path or primary_model_path,
            config.model_1_path or primary_model_path,
        ]
        adaptive_model_ids = {
            model_id
            for model_id, enabled in (
                (player_model_ids[0], config.adaptive_policy),
                (player_model_ids[1], config.opponent_adaptive_policy),
            )
            if enabled and model_id is not None
        }
        model_catalog = [dict(model) for model in config.model_catalog or []]
        known_model_ids = {model["id"] for model in model_catalog}
        for model in model_catalog:
            if model["id"] in adaptive_model_ids:
                model["runtime"] = "adaptive"
        for model_id in player_model_ids:
            if model_id is not None and model_id not in known_model_ids:
                model_entry = {"id": model_id, "label": str(model_id).split("/")[-1], "path": model_id}
                if model_id in adaptive_model_ids:
                    model_entry["runtime"] = "adaptive"
                model_catalog.append(model_entry)
                known_model_ids.add(model_id)

        def agent_loader(player: int, model_id: str) -> WebAgent:
            policy_mode = config.policy_mode if player == 0 else config.opponent_policy_mode or config.policy_mode
            policy_input = config.model_0_policy_input if player == 0 else config.model_1_policy_input
            input_channels = config.model_0_input_channels if player == 0 else config.model_1_input_channels
            use_search = config.search_policy if player == 0 else config.opponent_search_policy
            model_entry = next((model for model in model_catalog if model["id"] == model_id), None)
            model_runtime = None if model_entry is None else model_entry.get("runtime")
            player_uses_adaptive = config.adaptive_policy if player == 0 else config.opponent_adaptive_policy
            use_adaptive = player_uses_adaptive or model_runtime == "adaptive"
            if use_adaptive:
                use_search = False
            adaptive_config = (
                AdaptiveRuntimeConfig(
                    pad_to=config.pad_to,
                    network_arch=config.network_arch,
                    channels=config.channels,
                    global_context=config.global_context,
                    scoreboard_history=config.scoreboard_history,
                    fog_memory=config.fog_memory,
                    value_loss=config.value_loss,
                    value_bins=config.value_bins,
                    value_min=config.value_min,
                    value_max=config.value_max,
                    value_sigma=config.value_sigma,
                    policy_adapter_path=config.policy_adapter_path,
                    policy_adapter_scale=config.policy_adapter_scale,
                    policy_adapter_mode=config.policy_adapter_mode,
                    policy_adapter_min_grid_size=config.policy_adapter_min_grid_size,
                    policy_adapter_max_grid_size=config.policy_adapter_max_grid_size,
                    online_search=config.adaptive_online_search,
                    online_search_min_turn=config.online_search_min_turn,
                    online_search_require_contact=config.online_search_require_contact,
                    online_search_min_grid_size=config.online_search_min_grid_size,
                    online_search_max_grid_size=config.online_search_max_grid_size,
                    online_search_terminal_score=config.online_search_terminal_score,
                    online_search_min_score_gap=config.online_search_min_score_gap,
                    online_search_max_steps=config.online_search_max_steps,
                    online_search_opponent_path=config.adaptive_online_search_opponent_path,
                    online_search_opponent_policy_mode=config.adaptive_online_search_opponent_policy_mode,
                    online_search_opponent_channels=config.adaptive_online_search_opponent_channels,
                    online_search_opponent_input_channels=config.adaptive_online_search_opponent_input_channels,
                )
                if use_adaptive
                else None
            )
            return make_policy_agent(
                model_id,
                config.grid_size,
                policy_mode,
                names[player],
                policy_input,
                input_channels,
                use_search,
                search_config,
                adaptive_config=adaptive_config,
            )

        if config.machine_vs_machine:
            player_controls = ["model", "model"]
            return cls(
                grid_factory=grid_factory,
                names=names,
                colors=colors,
                key=key,
                human_player=config.human_player,
                machine_vs_machine=True,
                agent_loader=agent_loader,
                player_controls=player_controls,
                active_human_player=None,
                player_model_ids=player_model_ids,
                model_catalog=model_catalog,
                preview_player=0,
                auto_tick=config.auto_tick,
                tick_rate=config.tick_rate,
                max_steps=config.max_steps,
                preview_top_k=config.preview_top_k,
                ai_preview=config.ai_preview,
            )

        player_controls = ["model", "model"]
        player_controls[config.human_player] = "human"
        return cls(
            grid_factory=grid_factory,
            names=names,
            colors=colors,
            key=key,
            human_player=config.human_player,
            machine_vs_machine=False,
            agent_loader=agent_loader,
            player_controls=player_controls,
            active_human_player=config.human_player,
            player_model_ids=player_model_ids,
            model_catalog=model_catalog,
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
        player_controls: tuple[str, str] | list[str] | None = None,
        active_human_player: int | None = None,
        player_model_ids: tuple[str | None, str | None] | list[str | None] | None = None,
        model_catalog: list[dict[str, Any]] | None = None,
        agent_loader: Callable[[int, str], WebAgent] | None = None,
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
                agent_loader=agent_loader,
                player_controls=player_controls,
                active_human_player=active_human_player,
                player_model_ids=player_model_ids,
                model_catalog=model_catalog,
                preview_player=0,
                auto_tick=auto_tick,
                tick_rate=tick_rate,
                max_steps=max_steps,
                ai_preview=False,
            )

        if len(agents) not in (1, 2):
            raise ValueError("human test sessions require one or two model agents")
        machine_agents = (agents[0], agents[1]) if len(agents) == 2 else None
        model_agent = None if len(agents) == 2 else agents[0]
        return cls(
            initial_grid=grid,
            names=names,
            colors=colors,
            key=jrandom.PRNGKey(0),
            human_player=human_player,
            machine_vs_machine=False,
            model_agent=model_agent,
            machine_agents=machine_agents,
            agent_loader=agent_loader,
            player_controls=player_controls,
            active_human_player=active_human_player,
            player_model_ids=player_model_ids,
            model_catalog=model_catalog,
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
        self._reset_loaded_agents()
        self.selected_cell = None
        self.move_queue.clear()
        self.split_enabled = False
        self.last_message = message
        self.step_count = 0
        self.policy_preview = None
        return self.snapshot()

    def _reset_loaded_agents(self) -> None:
        """Reset per-game agent memory for already-loaded model agents."""
        seen = set()
        for agent in [self.model_agent, *(self.machine_agents or ()), *self.model_agents, *self.agent_cache.values()]:
            if agent is None or id(agent) in seen or not hasattr(agent, "reset"):
                continue
            seen.add(id(agent))
            agent.reset()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for the current state."""
        self._refresh_policy_preview()
        visibility_player = self.active_human_player if self.active_human_player is not None else None
        mode = "machine-vs-machine" if self.active_human_player is None else "human-vs-model"
        return build_snapshot(
            state=self.state,
            info=self.info,
            names=self.names,
            colors=self.colors,
            mode=mode,
            visibility_player=visibility_player,
            step_count=self.step_count,
            selected_cell=self.selected_cell,
            split_enabled=self.split_enabled,
            last_message=self.last_message,
            auto_tick_enabled=self.auto_tick_enabled,
            tick_rate=self.tick_rate,
            policy_preview=self.policy_preview,
            valid_targets=self._valid_targets(self.active_human_player, self.selected_cell),
            reached_limit=self._reached_limit(),
            queued_moves=self._serialized_queue(),
            player_controls=self.player_controls,
            player_model_ids=self.player_model_ids,
            model_catalog=self.model_catalog,
            active_human_player=self.active_human_player,
        )

    def submit_client_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Apply one semantic browser command and return the resulting snapshot."""
        command_type = command.get("type")
        if command_type == "select":
            player = self._active_human_or_none()
            if player is None:
                self.last_message = "No active human player"
            else:
                self._select(player, (int(command["row"]), int(command["col"])))
            return self.snapshot()
        if command_type == "move":
            source = tuple(command["source"])
            target = tuple(command["target"])
            player = self._active_human_or_none()
            if player is None:
                self.last_message = "No active human player"
            else:
                self._queue_move(
                    player,
                    (int(source[0]), int(source[1])),
                    (int(target[0]), int(target[1])),
                    bool(command["split"]),
                )
            return self.snapshot()
        if command_type == "pass":
            player = self._active_human_or_none()
            if player is None:
                self.last_message = "No active human player"
            else:
                self.selected_cell = None
                self.move_queue.append(QueuedMove(player=player, source=None, target=None, is_pass=True))
                self.last_message = "Pass queued"
            return self.snapshot()
        if command_type == "set_player_control":
            self._set_player_control(int(command["player"]), str(command["control"]))
            return self.snapshot()
        if command_type == "set_player_model":
            self._set_player_model(int(command["player"]), str(command["model_id"]))
            return self.snapshot()
        if command_type == "set_active_human_player":
            self._set_active_human_player(int(command["player"]))
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

        self.key, action_key = jrandom.split(self.key)
        player_keys = jrandom.split(action_key, 2)
        actions = []
        messages = []
        for player in range(2):
            action, message = self._turn_action_for_player(player, player_keys[player])
            actions.append(action)
            if message:
                messages.append(message)
        self._advance(jnp.stack(actions), now)
        self._clear_invalid_selection()
        self.last_message = self._tick_message(messages)
        return self.snapshot()

    def _refresh_policy_preview(self) -> None:
        preview_player = self._preview_player()
        preview_agent = self._agent_for_player(preview_player) if preview_player is not None else None
        if not self.ai_preview or self._game_done() or preview_agent is None or preview_player is None:
            self.policy_preview = None
            return
        if hasattr(preview_agent, "explain_for_state"):
            self.policy_preview = preview_agent.explain_for_state(
                self.state,
                preview_player,
                top_k=self.preview_top_k,
            )
        elif hasattr(preview_agent, "explain"):
            self.policy_preview = preview_agent.explain(
                game.get_observation(self.state, preview_player),
                top_k=self.preview_top_k,
            )

    def _turn_action_for_player(self, player: int, key: jnp.ndarray) -> tuple[jnp.ndarray, str | None]:
        if self.player_controls[player] == "model":
            agent = self._agent_for_player(player)
            if agent is None:
                return create_action(to_pass=True), f"No model for Player {player}"
            return self._choose_agent_action(agent, player, key), "Tick"

        queued_action = self._pop_next_queued_action(player)
        if queued_action is not None:
            action, is_pass = queued_action
            return action, "Queued pass executed" if is_pass else "Queued move executed"
        return create_action(to_pass=True), "Auto pass"

    def _tick_message(self, messages: list[str]) -> str:
        for message in messages:
            if message.startswith("Queued"):
                return message
        for message in messages:
            if message.startswith("No model"):
                return message
        if "Auto pass" in messages:
            return "Auto pass"
        return "Tick"

    def _select(self, player: int, cell: tuple[int, int]) -> None:
        if self._is_valid_source(player, cell):
            self.selected_cell = cell
            self.last_message = f"Selected: {cell}"
            return
        self.selected_cell = None
        self.last_message = "Invalid source"

    def _queue_move(self, player: int, source: tuple[int, int], target: tuple[int, int], split: bool) -> None:
        projected_state = self._projected_queue_state()
        if not self._is_valid_source_in_state(projected_state, player, source):
            self.selected_cell = None
            self.last_message = "Invalid source"
            return

        direction = self._direction(source, target)
        if direction is None or target not in self._valid_targets_in_state(projected_state, source):
            self.selected_cell = source
            self.last_message = "Invalid target"
            return

        self.move_queue.append(QueuedMove(player=player, source=source, target=target, split=split))
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

    def _set_player_control(self, player: int, control: str) -> None:
        if player not in (0, 1):
            self.last_message = "Invalid player"
            return
        if control not in ("human", "model"):
            self.last_message = "Invalid control"
            return
        if control == "model" and self._agent_for_player(player) is None:
            self.last_message = f"No model for Player {player}"
            return

        self.player_controls[player] = control
        self.move_queue.clear()
        self.selected_cell = None
        if control == "human":
            self.active_human_player = player
            self.last_message = f"Player {player} controlled by human"
            return
        if self.active_human_player == player:
            self.active_human_player = self._first_human_player()
        self.last_message = f"Player {player} hosted by model"

    def _set_player_model(self, player: int, model_id: str) -> None:
        if player not in (0, 1):
            self.last_message = "Invalid player"
            return
        model = self._catalog_entry(model_id)
        if model is None:
            self.last_message = "Unknown model"
            return
        self.player_model_ids[player] = model_id
        self.model_agents[player] = None
        self.agent_cache.pop((player, model_id), None)
        if self.player_controls[player] == "model" and self._agent_for_player(player) is None:
            self.last_message = f"Could not load model for Player {player}"
            return
        self.last_message = f"Player {player} model: {model['label']}"

    def _set_active_human_player(self, player: int) -> None:
        if player not in (0, 1):
            self.last_message = "Invalid player"
            return
        if self.player_controls[player] != "human":
            self.last_message = f"Player {player} is not human-controlled"
            return
        self.active_human_player = player
        self.move_queue.clear()
        self.selected_cell = None
        self.last_message = f"Active human: Player {player}"

    def _clear_invalid_selection(self) -> None:
        player = self._active_human_or_none()
        if self.selected_cell is not None and (player is None or not self._is_valid_source(player, self.selected_cell)):
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

    def _stack_single_player_action(self, player: int, action: jnp.ndarray) -> jnp.ndarray:
        actions = [create_action(to_pass=True), create_action(to_pass=True)]
        actions[player] = action
        return jnp.stack(actions)

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

    def _is_valid_source(self, player: int, cell: tuple[int, int]) -> bool:
        return self._is_valid_source_in_state(self._projected_queue_state(), player, cell)

    def _is_valid_source_in_state(self, state: game.GameState, player: int, cell: tuple[int, int]) -> bool:
        row, col = cell
        height, width = state.armies.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return bool(state.ownership[player, row, col]) and int(state.armies[row, col]) > 1

    def _valid_targets(self, player: int | None, selected_cell: tuple[int, int] | None) -> list[tuple[int, int]]:
        if player is None:
            return []
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
        for move in self.move_queue:
            action = self._action_for_queued_move(move, projected)
            if action is None:
                continue
            actions = self._stack_single_player_action(move.player, action)
            projected, _ = game.step(projected, actions)
        return projected

    def _action_for_queued_move(self, move: QueuedMove, state: game.GameState) -> jnp.ndarray | None:
        if move.is_pass:
            return create_action(to_pass=True)
        if move.source is None or move.target is None:
            return None
        if not self._is_valid_source_in_state(state, move.player, move.source):
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

    def _pop_next_queued_action(self, player: int) -> tuple[jnp.ndarray, bool] | None:
        while self.move_queue:
            move = self.move_queue.pop(0)
            if move.player != player:
                continue
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

    def _first_human_player(self) -> int | None:
        for player, control in enumerate(self.player_controls):
            if control == "human":
                return player
        return None

    def _active_human_or_none(self) -> int | None:
        if self.active_human_player is None:
            return None
        if self.player_controls[self.active_human_player] != "human":
            return None
        return self.active_human_player

    def _catalog_entry(self, model_id: str) -> dict[str, Any] | None:
        for model in self.model_catalog:
            if model["id"] == model_id:
                return model
        return None

    def _agent_for_player(self, player: int | None) -> WebAgent | None:
        if player is None:
            return None
        cached_agent = self.model_agents[player]
        if cached_agent is not None:
            return cached_agent
        model_id = self.player_model_ids[player]
        if model_id is None:
            return None
        if (player, model_id) in self.agent_cache:
            self.model_agents[player] = self.agent_cache[(player, model_id)]
            return self.model_agents[player]
        if self.agent_loader is None:
            return None
        try:
            agent = self.agent_loader(player, model_id)
        except Exception as error:
            self.last_message = f"Model load failed: {error}"
            return None
        self.agent_cache[(player, model_id)] = agent
        self.model_agents[player] = agent
        return agent

    def _preview_player(self) -> int | None:
        active = self._active_human_or_none()
        if active is not None:
            opponent = 1 - active
            if self.player_controls[opponent] == "model":
                return opponent
        for player, control in enumerate(self.player_controls):
            if control == "model":
                return player
        return None

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
    map_pad_to = config.pad_to if config.adaptive_policy or config.opponent_adaptive_policy else config.grid_size
    return SimpleNamespace(
        grid_size=config.grid_size,
        map_pad_to=map_pad_to,
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
