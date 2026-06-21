"""Shared PPO runtime helpers for interactive frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from examples._experimental.ppo.search_policy import rollout_search_action, rollout_search_candidates
from generals.agents.ppo_policy_agent import (
    POLICY_INPUT_CHOICES,
    PPOPolicyAgent,
    PolicyInputOption,
    PolicyPreview,
    PolicyValueNetwork,
    action_tuple_to_candidate,
    load_policy_network,
    parse_policy_channels,
    policy_input_default_channels,
)
from generals.core import game
from generals.core.grid import generate_grid

POLICY_MODE_NAME_TO_ID = {"greedy": 0, "sample": 1}


class PlayAgent(Protocol):
    def act(self, observation, key: jnp.ndarray) -> jnp.ndarray:
        """Return one public Generals action."""


@dataclass(frozen=True)
class SearchConfig:
    """Rollout-search parameters shared by interactive player agents."""

    rollout_policy_mode: str
    top_k: int
    rollout_steps: int
    rollouts_per_action: int
    army_weight: float
    land_weight: float
    prior_weight: float


@dataclass(frozen=True)
class AdaptiveRuntimeConfig:
    """Adaptive checkpoint settings used by interactive browser agents."""

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
    online_search: bool = False
    online_search_min_turn: int = 0
    online_search_require_contact: bool = False
    online_search_min_grid_size: int = 0
    online_search_max_grid_size: int = 0
    online_search_terminal_score: float = 100.0
    online_search_min_score_gap: float = 0.0
    online_search_max_steps: int = 750
    online_search_opponent_path: str | None = None
    online_search_opponent_policy_mode: str = "sample"
    online_search_opponent_channels: str | None = None
    online_search_opponent_input_channels: int = 9


class RolloutSearchPolicyAgent:
    """Agent that wraps a 9-channel PPO checkpoint with rollout search."""

    def __init__(
        self,
        model_path: str,
        grid_size: int,
        *,
        agent_id: str = "PPO Search",
        policy_input: PolicyInputOption = "auto",
        input_channels: int | None = None,
        top_k: int = 4,
        rollout_steps: int = 16,
        rollouts_per_action: int = 4,
        rollout_policy_mode: str = "sample",
        army_weight: float = 12.0,
        land_weight: float = 8.0,
        prior_weight: float = 0.01,
    ):
        if policy_input not in ("auto", "observation"):
            raise ValueError("--search-policy only supports observation checkpoints")
        if input_channels not in (None, 9):
            raise ValueError("--search-policy requires a 9-channel observation checkpoint")
        if rollout_policy_mode not in POLICY_MODE_NAME_TO_ID:
            raise ValueError("rollout_policy_mode must be 'greedy' or 'sample'")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if rollouts_per_action <= 0:
            raise ValueError("rollouts_per_action must be positive")

        self.id = agent_id
        self.grid_size = grid_size
        self.network = load_policy_network(model_path, grid_size, input_channels=9)
        self.rollout_policy_mode = rollout_policy_mode
        self.rollout_policy_mode_id = POLICY_MODE_NAME_TO_ID[rollout_policy_mode]
        self.top_k = top_k
        self.rollout_steps = rollout_steps
        self.rollouts_per_action = rollouts_per_action
        self.army_weight = army_weight
        self.land_weight = land_weight
        self.prior_weight = prior_weight

    def _validate_grid_shape(self, shape: tuple[int, ...]) -> None:
        if shape != (self.grid_size, self.grid_size):
            raise ValueError(f"PPO checkpoint expects {self.grid_size}x{self.grid_size} states, got {shape}")

    def act(self, observation, key: jnp.ndarray) -> jnp.ndarray:
        raise ValueError("rollout-search requires act_for_state so it can simulate from the full game state")

    def act_for_state(self, state: game.GameState, player: int, key: jnp.ndarray) -> jnp.ndarray:
        """Return the rollout-search action for one player in the current state."""
        self._validate_grid_shape(state.armies.shape)
        return rollout_search_action(
            self.network,
            state,
            key,
            player,
            self.top_k,
            self.rollout_steps,
            self.rollouts_per_action,
            self.rollout_policy_mode_id,
            self.army_weight,
            self.land_weight,
            self.prior_weight,
        )

    def explain_for_state(self, state: game.GameState, player: int, top_k: int = 3) -> PolicyPreview:
        """Show rollout-search candidates ranked by simulated score."""
        self._validate_grid_shape(state.armies.shape)
        actions, _, _, search_scores = rollout_search_candidates(
            self.network,
            state,
            jrandom.PRNGKey(0),
            player,
            self.top_k,
            self.rollout_steps,
            self.rollouts_per_action,
            self.rollout_policy_mode_id,
            self.army_weight,
            self.land_weight,
            self.prior_weight,
        )
        scores_np = np.asarray(search_scores)
        order = np.argsort(scores_np)[::-1][:top_k]
        shifted_scores = scores_np - np.max(scores_np)
        all_probabilities = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores))
        probabilities = all_probabilities[order]
        candidates = []
        for candidate_index, probability in zip(order.tolist(), probabilities.tolist(), strict=True):
            action = tuple(int(value) for value in np.asarray(actions[candidate_index]).tolist())
            candidates.append(action_tuple_to_candidate(action, float(probability)))
        value = float(np.max(scores_np)) if scores_np.size else 0.0
        return PolicyPreview(candidates=tuple(candidates), value=value, policy_mode="rollout-search")


class AdaptiveWebPolicyAgent:
    """Interactive wrapper for adaptive checkpoints, optional adapters, and adaptive online search."""

    def __init__(
        self,
        *,
        model_path: str,
        grid_size: int,
        agent_id: str,
        policy_mode: str,
        search_config: SearchConfig,
        config: AdaptiveRuntimeConfig,
    ):
        if policy_mode not in POLICY_MODE_NAME_TO_ID:
            raise ValueError("policy_mode must be 'greedy' or 'sample'")
        if config.network_arch not in ("cnn", "unet"):
            raise ValueError("adaptive network_arch must be 'cnn' or 'unet'")
        if config.value_loss not in ("mse", "hl-gauss"):
            raise ValueError("adaptive value_loss must be 'mse' or 'hl-gauss'")
        if config.policy_adapter_mode not in ("delta", "blend", "replace"):
            raise ValueError("adaptive policy_adapter_mode must be delta, blend, or replace")
        if grid_size > config.pad_to:
            raise ValueError("adaptive pad_to must be at least grid_size")
        if config.policy_adapter_scale > 0.0 and config.policy_adapter_path is None:
            raise ValueError("policy_adapter_scale requires policy_adapter_path")
        if config.online_search and search_config.top_k <= 0:
            raise ValueError("adaptive online search requires top_k > 0")

        from examples._experimental.ppo import evaluate_adaptive_policy as adaptive_eval
        from examples._experimental.ppo.adaptive_common import (
            ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
            adaptive_input_channel_count,
            adaptive_scoreboard_features,
            empty_adaptive_fog_memory,
        )
        from examples._experimental.ppo.adaptive_network import load_or_create_adaptive_network
        from examples._experimental.ppo.common import OPPONENT_NAME_TO_ID, opponent_action, policy_network_action
        from examples._experimental.ppo.train import random_action

        self.id = agent_id
        self.grid_size = grid_size
        self.config = config
        self.search_config = search_config
        self.policy_mode_id = POLICY_MODE_NAME_TO_ID[policy_mode]
        self.opponent_policy_mode_id = POLICY_MODE_NAME_TO_ID[config.online_search_opponent_policy_mode]
        self._adaptive_eval = adaptive_eval
        self._adaptive_scoreboard_features = adaptive_scoreboard_features
        self._empty_adaptive_fog_memory = empty_adaptive_fog_memory
        self._opponent_action = opponent_action
        self._policy_network_action = policy_network_action
        self._random_action = random_action
        self._opponent_id = OPPONENT_NAME_TO_ID["expander"]
        self._scoreboard_feature_channels = ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS

        key = jrandom.PRNGKey(0)
        key, model_key, adapter_key, opponent_key = jrandom.split(key, 4)
        network_global_context = config.global_context or config.scoreboard_history
        input_channels = adaptive_input_channel_count(
            network_global_context,
            config.scoreboard_history,
            config.fog_memory,
        )
        value_bins = config.value_bins if config.value_loss == "hl-gauss" else 0
        self.network = load_or_create_adaptive_network(
            model_key,
            pad_size=config.pad_to,
            init_model_path=model_path,
            channels=config.channels,
            input_channels=input_channels,
            init_input_channels=input_channels,
            value_bins=value_bins,
            value_min=config.value_min,
            value_max=config.value_max,
            value_sigma=config.value_sigma,
            global_context=network_global_context,
            init_global_context=network_global_context,
            network_arch=config.network_arch,
            init_network_arch=config.network_arch,
        )
        self.policy_adapter_network = None
        if config.policy_adapter_path is not None:
            self.policy_adapter_network = load_or_create_adaptive_network(
                adapter_key,
                pad_size=config.pad_to,
                init_model_path=config.policy_adapter_path,
                channels=config.channels,
                input_channels=input_channels,
                init_input_channels=input_channels,
                value_bins=value_bins,
                value_min=config.value_min,
                value_max=config.value_max,
                value_sigma=config.value_sigma,
                global_context=network_global_context,
                init_global_context=network_global_context,
                network_arch=config.network_arch,
                init_network_arch=config.network_arch,
            )
        self.opponent_network = None
        if config.online_search_opponent_path is not None:
            self.opponent_network = PolicyValueNetwork(
                opponent_key,
                grid_size=grid_size,
                channels=parse_policy_channels(config.online_search_opponent_channels),
                input_channels=config.online_search_opponent_input_channels,
            )
            self.opponent_network = eqx.tree_deserialise_leaves(
                config.online_search_opponent_path,
                self.opponent_network,
            )
        self.policy_adapter_mode_id = adaptive_eval.POLICY_ADAPTER_MODE_TO_ID[config.policy_adapter_mode]
        self.reset()

    def reset(self) -> None:
        """Clear one-game adaptive memory carried by scoreboard-history and fog-memory inputs."""
        self.previous_scoreboard = jnp.zeros((self._scoreboard_feature_channels,), dtype=jnp.float32)
        memory = self._empty_adaptive_fog_memory(1, self.config.pad_to)
        self.fog_memory = type(memory)(*(value[0] for value in memory))

    def _validate_grid_shape(self, shape: tuple[int, ...]) -> int:
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Adaptive checkpoint expects square states, got {shape}")
        canvas_size = int(shape[0])
        if canvas_size not in (self.grid_size, self.config.pad_to):
            raise ValueError(
                f"Adaptive checkpoint expects {self.grid_size}x{self.grid_size} or "
                f"{self.config.pad_to}x{self.config.pad_to} states, got {shape}"
            )
        if canvas_size > self.config.pad_to:
            raise ValueError(f"Adaptive pad_to={self.config.pad_to} cannot represent {shape}")
        return self.grid_size

    def _online_search_size_allowed(self, effective_size: int) -> bool:
        return (
            (self.config.online_search_min_grid_size <= 0 or effective_size >= self.config.online_search_min_grid_size)
            and (self.config.online_search_max_grid_size <= 0 or effective_size <= self.config.online_search_max_grid_size)
        )

    def _online_search_allowed(self, state: game.GameState, player: int, effective_size: int) -> bool:
        if not self.config.online_search or not self._online_search_size_allowed(effective_size):
            return False
        if int(state.time) < self.config.online_search_min_turn:
            return False
        if not self.config.online_search_require_contact:
            return True
        obs = game.get_observation(state, player)
        return bool(jnp.any(obs.opponent_cells))

    def act(self, observation, key: jnp.ndarray) -> jnp.ndarray:
        raise ValueError("adaptive web policy requires act_for_state so it can carry memory and run online search")

    def act_for_state(self, state: game.GameState, player: int, key: jnp.ndarray) -> jnp.ndarray:
        effective_size = self._validate_grid_shape(state.armies.shape)
        obs = game.get_observation(state, player)
        fallback_action, current_scoreboard, current_memory = self._adaptive_eval.adapter_policy_action_with_memory(
            self.network,
            self.policy_adapter_network,
            obs,
            effective_size,
            key,
            self.policy_mode_id,
            self.config.pad_to,
            self.config.global_context or self.config.scoreboard_history,
            self.config.scoreboard_history,
            self.previous_scoreboard,
            self.config.fog_memory,
            self.fog_memory,
            self.config.policy_adapter_scale,
            self.policy_adapter_mode_id,
            self.config.policy_adapter_min_grid_size,
            self.config.policy_adapter_max_grid_size,
        )
        action = fallback_action
        if self._online_search_allowed(state, player, effective_size):
            key, opponent_key, search_key = jrandom.split(key, 3)
            opponent_player = 1 - player
            opponent_obs = game.get_observation(state, opponent_player)
            if self.opponent_network is not None:
                opponent_first_action = self._policy_network_action(
                    self.opponent_network,
                    opponent_key,
                    self._adaptive_eval.crop_observation(opponent_obs, effective_size),
                    self.opponent_policy_mode_id,
                )
                action = self._adaptive_eval.online_search_action_policy_opponent(
                    self.network,
                    self.policy_adapter_network,
                    None,
                    self.opponent_network,
                    state,
                    effective_size,
                    search_key,
                    fallback_action,
                    opponent_first_action,
                    player,
                    self.policy_mode_id,
                    self.opponent_policy_mode_id,
                    self.config.pad_to,
                    self.config.online_search_max_steps,
                    self.config.global_context or self.config.scoreboard_history,
                    self.config.scoreboard_history,
                    self.previous_scoreboard,
                    self.config.fog_memory,
                    current_memory,
                    self.config.policy_adapter_scale,
                    self.policy_adapter_mode_id,
                    self.config.policy_adapter_min_grid_size,
                    self.config.policy_adapter_max_grid_size,
                    self.search_config.top_k,
                    self.search_config.rollout_steps,
                    self.search_config.rollouts_per_action,
                    self.search_config.army_weight,
                    self.search_config.land_weight,
                    self.search_config.prior_weight,
                    self.config.online_search_terminal_score,
                    self.config.online_search_min_score_gap,
                    -1.0,
                    self._adaptive_eval.ONLINE_SEARCH_GATE_FEATURE_DIM,
                )
            else:
                opponent_first_action = self._opponent_action(
                    self._opponent_id,
                    opponent_key,
                    opponent_obs,
                    self._random_action,
                )
                action = self._adaptive_eval.online_search_action_heuristic_opponent(
                    self.network,
                    self.policy_adapter_network,
                    None,
                    self._opponent_id,
                    state,
                    effective_size,
                    search_key,
                    fallback_action,
                    opponent_first_action,
                    player,
                    self.policy_mode_id,
                    self.config.pad_to,
                    self.config.online_search_max_steps,
                    self.config.global_context or self.config.scoreboard_history,
                    self.config.scoreboard_history,
                    self.previous_scoreboard,
                    self.config.fog_memory,
                    current_memory,
                    self.config.policy_adapter_scale,
                    self.policy_adapter_mode_id,
                    self.config.policy_adapter_min_grid_size,
                    self.config.policy_adapter_max_grid_size,
                    self.search_config.top_k,
                    self.search_config.rollout_steps,
                    self.search_config.rollouts_per_action,
                    self.search_config.army_weight,
                    self.search_config.land_weight,
                    self.search_config.prior_weight,
                    self.config.online_search_terminal_score,
                    self.config.online_search_min_score_gap,
                    -1.0,
                    self._adaptive_eval.ONLINE_SEARCH_GATE_FEATURE_DIM,
                )
        self.previous_scoreboard = current_scoreboard
        self.fog_memory = current_memory
        return action


def make_simple_general_grid(key: jnp.ndarray, grid_size: int, pad_to: int | None = None) -> jnp.ndarray:
    """Create an empty square grid with two random generals."""
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    grid = grid.at[pos_a].set(1).at[pos_b].set(2)
    target_size = grid_size if pad_to is None else pad_to
    if target_size <= grid_size:
        return grid
    return jnp.pad(
        grid,
        ((0, target_size - grid_size), (0, target_size - grid_size)),
        mode="constant",
        constant_values=-2,
    )


def make_grid(args, key: jnp.ndarray) -> jnp.ndarray:
    """Create one playable map from CLI or web options."""
    map_pad_to = getattr(args, "map_pad_to", args.grid_size)
    if args.map_generator == "simple":
        return make_simple_general_grid(key, args.grid_size, pad_to=map_pad_to)

    return generate_grid(
        key,
        grid_dims=(args.grid_size, args.grid_size),
        pad_to=map_pad_to,
        mountain_density_range=(args.mountain_density_min, args.mountain_density_max),
        num_cities_range=(args.num_cities_min, args.num_cities_max),
        min_generals_distance=args.effective_min_generals_distance,
        max_generals_distance=args.max_generals_distance,
        castle_val_range=(args.city_army_min, args.city_army_max),
    )


def resolve_alias(parser, primary_name: str, primary, alias_name: str, alias, default):
    """Resolve two CLI aliases while rejecting conflicting explicit values."""
    if primary is not None and alias is not None and primary != alias:
        parser.error(f"pass either {primary_name} or {alias_name}, not both")
    if primary is not None:
        return primary
    if alias is not None:
        return alias
    return default


def resolve_input_channels(policy_input: str, input_channels: int | None) -> int | None:
    """Resolve explicit input channels, leaving auto-detected inputs unset."""
    if input_channels is not None:
        return input_channels
    if policy_input == "auto":
        return None
    return policy_input_default_channels(policy_input)


def make_player_names(human_player: int, machine_vs_machine: bool = False) -> list[str]:
    if machine_vs_machine:
        return ["PPO 0", "PPO 1"]
    names = ["PPO Model", "PPO Model"]
    names[human_player] = "Human"
    names[1 - human_player] = "PPO Model"
    return names


def make_search_config(args) -> SearchConfig:
    """Build rollout-search settings from parsed arguments."""
    return SearchConfig(
        rollout_policy_mode=args.search_rollout_policy_mode,
        top_k=args.search_top_k,
        rollout_steps=args.search_rollout_steps,
        rollouts_per_action=args.search_rollouts_per_action,
        army_weight=args.search_army_weight,
        land_weight=args.search_land_weight,
        prior_weight=args.search_prior_weight,
    )


def make_policy_agent(
    model_path: str,
    grid_size: int,
    policy_mode: str,
    agent_id: str,
    policy_input: PolicyInputOption,
    input_channels: int | None,
    use_search_policy: bool,
    search_config: SearchConfig,
    adaptive_config: AdaptiveRuntimeConfig | None = None,
) -> PlayAgent:
    """Create either a plain PPO agent or a rollout-search wrapper."""
    if adaptive_config is not None:
        return AdaptiveWebPolicyAgent(
            model_path=model_path,
            grid_size=grid_size,
            agent_id=agent_id,
            policy_mode=policy_mode,
            search_config=search_config,
            config=adaptive_config,
        )
    if use_search_policy:
        return RolloutSearchPolicyAgent(
            model_path,
            grid_size,
            agent_id=agent_id,
            policy_input=policy_input,
            input_channels=input_channels,
            top_k=search_config.top_k,
            rollout_steps=search_config.rollout_steps,
            rollouts_per_action=search_config.rollouts_per_action,
            rollout_policy_mode=search_config.rollout_policy_mode,
            army_weight=search_config.army_weight,
            land_weight=search_config.land_weight,
            prior_weight=search_config.prior_weight,
        )
    return PPOPolicyAgent(
        model_path,
        grid_size,
        policy_mode,
        agent_id=agent_id,
        policy_input=policy_input,
        input_channels=input_channels,
    )
