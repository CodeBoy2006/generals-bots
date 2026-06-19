"""Shared PPO runtime helpers for interactive frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from examples._experimental.ppo.search_policy import rollout_search_action, rollout_search_candidates
from generals.agents.ppo_policy_agent import (
    POLICY_INPUT_CHOICES,
    PPOPolicyAgent,
    PolicyInputOption,
    PolicyPreview,
    action_tuple_to_candidate,
    load_policy_network,
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


def make_simple_general_grid(key: jnp.ndarray, grid_size: int) -> jnp.ndarray:
    """Create an empty square grid with two random generals."""
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    return grid.at[pos_a].set(1).at[pos_b].set(2)


def make_grid(args, key: jnp.ndarray) -> jnp.ndarray:
    """Create one playable map from CLI or web options."""
    if args.map_generator == "simple":
        return make_simple_general_grid(key, args.grid_size)

    return generate_grid(
        key,
        grid_dims=(args.grid_size, args.grid_size),
        pad_to=args.grid_size,
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
) -> PlayAgent:
    """Create either a plain PPO agent or a rollout-search wrapper."""
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
