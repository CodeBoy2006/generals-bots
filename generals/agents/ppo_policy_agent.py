"""PPO policy agent backed by Equinox checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core import game
from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation

from .agent import Agent

PolicyMode = Literal["greedy", "sample"]
PolicyInput = Literal["observation", "full-state", "augmented-full-state"]
PolicyInputOption = PolicyInput | Literal["auto"]
PolicyChannels = tuple[int, int, int, int]
DEFAULT_POLICY_CHANNELS: PolicyChannels = (32, 32, 32, 16)
POLICY_INPUT_NAMES: tuple[PolicyInput, ...] = ("observation", "full-state", "augmented-full-state")
POLICY_INPUT_CHOICES: tuple[PolicyInputOption, ...] = ("auto",) + POLICY_INPUT_NAMES
POLICY_INPUT_NAME_TO_ID = {name: idx for idx, name in enumerate(POLICY_INPUT_NAMES)}

_DIRECTION_TARGETS = {
    0: (-1, 0, "Up"),
    1: (1, 0, "Down"),
    2: (0, -1, "Left"),
    3: (0, 1, "Right"),
}


@dataclass(frozen=True)
class PolicyActionCandidate:
    """One semantic PPO action candidate for policy explanation."""

    action: tuple[int, int, int, int, int]
    probability: float
    source: tuple[int, int] | None
    target: tuple[int, int] | None
    direction: int | None
    direction_label: str
    is_split: bool
    is_pass: bool


@dataclass(frozen=True)
class PolicyPreview:
    """Top policy candidates plus the value estimate for one observation."""

    candidates: tuple[PolicyActionCandidate, ...]
    value: float
    policy_mode: PolicyMode


class PolicyValueNetwork(eqx.Module):
    """Convolutional policy-value network used by the experimental PPO trainer."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    policy_conv: eqx.nn.Conv2d
    value_conv: eqx.nn.Conv2d
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear

    def __init__(
        self,
        key: jnp.ndarray,
        grid_size: int = 4,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = 9,
    ):
        keys = jrandom.split(key, 8)

        self.conv1 = eqx.nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, key=keys[2])
        self.conv4 = eqx.nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, key=keys[3])
        self.policy_conv = eqx.nn.Conv2d(channels[3], 9, kernel_size=1, key=keys[4])
        self.value_conv = eqx.nn.Conv2d(channels[3], 4, kernel_size=1, key=keys[5])
        self.value_linear1 = eqx.nn.Linear(grid_size * grid_size * 4, 64, key=keys[6])
        self.value_linear2 = eqx.nn.Linear(64, 1, key=keys[7])

    def logits_value(self, obs: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute masked action logits and value for one observation."""
        grid_size = obs.shape[-1]

        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.conv4(x))

        v = jax.nn.relu(self.value_conv(x))
        value_hidden = jax.nn.relu(self.value_linear1(v.reshape(-1)))
        value = self.value_linear2(value_hidden)[0]

        logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        mask_penalty = (1 - mask_t) * -1e9
        combined_mask = jnp.concatenate(
            [
                mask_penalty,
                mask_penalty,
                jnp.zeros((1, grid_size, grid_size)),
            ],
            axis=0,
        )
        return (logits + combined_mask).reshape(-1), value

    def __call__(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        key: jnp.ndarray | None,
        action: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample or evaluate one action."""
        grid_size = obs.shape[-1]
        grid_cells = grid_size * grid_size
        logits, value = self.logits_value(obs, mask)

        if action is None:
            if key is None:
                raise ValueError("key is required when sampling an action")
            idx = jrandom.categorical(key, logits)
            direction, position = idx // grid_cells, idx % grid_cells
            row, col = position // grid_size, position % grid_size
            is_pass = direction == 8
            is_half = (direction >= 4) & (direction < 8)
            actual_dir = jnp.where(is_pass, 0, jnp.where(is_half, direction - 4, direction))
            action = jnp.array([is_pass, row, col, actual_dir, is_half], dtype=jnp.int32)
        else:
            is_pass, row, col, direction, is_half = action
            encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
            idx = encoded_dir * grid_cells + row * grid_size + col

        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[idx]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)
        return action, value, logprob, entropy


def obs_to_array(obs: Observation) -> jnp.ndarray:
    """Convert an Observation to the 9-channel tensor used by PPO checkpoints."""
    return jnp.stack(
        [
            obs.armies,
            obs.generals,
            obs.cities,
            obs.mountains,
            obs.neutral_cells,
            obs.owned_cells,
            obs.opponent_cells,
            obs.fog_cells,
            obs.structures_in_fog,
        ],
        axis=0,
    ).astype(jnp.float32)


def policy_input_default_channels(policy_input: PolicyInput | str) -> int:
    """Return the network input channel count produced by one policy input mode."""
    if policy_input not in POLICY_INPUT_NAMES:
        raise ValueError(f"policy_input must be one of {POLICY_INPUT_NAMES}")
    return 18 if policy_input == "augmented-full-state" else 9


def full_state_to_array(state: game.GameState, player: int) -> jnp.ndarray:
    """Encode privileged full-state features using the policy network's 9-channel layout."""
    opponent = 1 - player
    return jnp.stack(
        [
            state.armies,
            state.generals,
            state.cities,
            state.mountains,
            state.ownership_neutral,
            state.ownership[player],
            state.ownership[opponent],
            jnp.zeros_like(state.armies, dtype=bool),
            state.mountains | state.cities,
        ],
        axis=0,
    ).astype(jnp.float32)


def augmented_full_state_to_array(state: game.GameState, obs: Observation, player: int) -> jnp.ndarray:
    """Append privileged full-state channels after the standard fogged observation channels."""
    return jnp.concatenate([obs_to_array(obs), full_state_to_array(state, player)], axis=0)


def policy_input_array_and_mask(
    state: game.GameState,
    player: int,
    policy_input: PolicyInput,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the network input tensor and valid-action mask for a policy input mode."""
    obs = game.get_observation(state, player)
    if policy_input == "observation":
        mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
        return obs_to_array(obs), mask

    if policy_input == "full-state":
        mask = compute_valid_move_mask(state.armies, state.ownership[player], state.mountains)
        return full_state_to_array(state, player), mask

    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    return augmented_full_state_to_array(state, obs, player), mask


def normalize_action(action: jnp.ndarray) -> jnp.ndarray:
    """Keep pass actions at a canonical in-bounds source cell."""
    pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    return jnp.where(action[0] > 0, pass_action, action).astype(jnp.int32)


def index_to_action(index: jnp.ndarray, grid_size: int) -> jnp.ndarray:
    """Decode the flattened PPO policy index into the public action format."""
    grid_cells = grid_size * grid_size
    direction = index // grid_cells
    position = index % grid_cells
    row = position // grid_size
    col = position % grid_size
    is_pass = direction == 8
    is_half = (direction >= 4) & (direction < 8)
    actual_dir = jnp.where(is_pass, 0, jnp.where(is_half, direction - 4, direction))
    return normalize_action(jnp.array([is_pass, row, col, actual_dir, is_half], dtype=jnp.int32))


def action_tuple_to_candidate(action: tuple[int, int, int, int, int], probability: float) -> PolicyActionCandidate:
    """Build a display-friendly candidate from the public action tuple."""
    is_pass, row, col, direction, is_split = action
    if is_pass:
        return PolicyActionCandidate(
            action=action,
            probability=probability,
            source=None,
            target=None,
            direction=None,
            direction_label="Pass",
            is_split=False,
            is_pass=True,
        )

    row_delta, col_delta, label = _DIRECTION_TARGETS[direction]
    return PolicyActionCandidate(
        action=action,
        probability=probability,
        source=(row, col),
        target=(row + row_delta, col + col_delta),
        direction=direction,
        direction_label=label,
        is_split=bool(is_split),
        is_pass=False,
    )


def _top_policy_preview_from_logits(
    logits: jnp.ndarray,
    value: jnp.ndarray,
    grid_size: int,
    top_k: int,
    policy_mode: PolicyMode,
) -> PolicyPreview:
    """Return top semantic policy candidates with pass actions merged from logits."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    grid_cells = grid_size * grid_size
    probabilities = np.asarray(jax.nn.softmax(logits))
    semantic_probs: dict[tuple[int, int, int, int, int], float] = {}

    for index, probability in enumerate(probabilities.tolist()):
        if probability <= 0.0:
            continue
        direction = index // grid_cells
        position = index % grid_cells
        row = position // grid_size
        col = position % grid_size
        is_pass = int(direction == 8)
        is_split = int(4 <= direction < 8)
        actual_dir = 0 if is_pass else int(direction - 4 if is_split else direction)
        action = (is_pass, int(row), int(col), actual_dir, is_split)
        if is_pass:
            action = (1, 0, 0, 0, 0)
        semantic_probs[action] = semantic_probs.get(action, 0.0) + float(probability)

    candidates = [
        action_tuple_to_candidate(action, probability)
        for action, probability in sorted(semantic_probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]
    return PolicyPreview(
        candidates=tuple(candidates),
        value=float(np.asarray(value)),
        policy_mode=policy_mode,
    )


def top_policy_preview(
    network: PolicyValueNetwork,
    obs: Observation,
    top_k: int,
    policy_mode: PolicyMode,
) -> PolicyPreview:
    """Return top semantic policy candidates with pass actions merged."""
    obs_arr = obs_to_array(obs)
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    logits, value = network.logits_value(obs_arr, mask)
    return _top_policy_preview_from_logits(logits, value, obs.armies.shape[-1], top_k, policy_mode)


def top_policy_state_preview(
    network: PolicyValueNetwork,
    state: game.GameState,
    player: int,
    top_k: int,
    policy_mode: PolicyMode,
    policy_input: PolicyInput,
) -> PolicyPreview:
    """Return top policy candidates for a state-aware policy input mode."""
    obs_arr, mask = policy_input_array_and_mask(state, player, policy_input)
    logits, value = network.logits_value(obs_arr, mask)
    return _top_policy_preview_from_logits(logits, value, state.armies.shape[-1], top_k, policy_mode)


def greedy_policy_action(network: PolicyValueNetwork, obs: Observation) -> jnp.ndarray:
    """Select the maximum-logit valid action from a policy network."""
    obs_arr = obs_to_array(obs)
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    logits, _ = network.logits_value(obs_arr, mask)
    return index_to_action(jnp.argmax(logits), obs.armies.shape[-1])


def sampled_policy_action(network: PolicyValueNetwork, obs: Observation, key: jnp.ndarray) -> jnp.ndarray:
    """Sample a valid action from a policy network."""
    obs_arr = obs_to_array(obs)
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    action, _, _, _ = network(obs_arr, mask, key, None)
    return normalize_action(action)


def policy_state_action(
    network: PolicyValueNetwork,
    key: jnp.ndarray,
    state: game.GameState,
    player: int,
    policy_mode: PolicyMode,
    policy_input: PolicyInput,
) -> jnp.ndarray:
    """Select one action from either fogged observation or a state-aware policy input."""
    obs_arr, mask = policy_input_array_and_mask(state, player, policy_input)
    logits, _ = network.logits_value(obs_arr, mask)
    index = jnp.argmax(logits) if policy_mode == "greedy" else jrandom.categorical(key, logits)
    return index_to_action(index, state.armies.shape[-1])


def parse_policy_channels(channels: str | tuple[int, int, int, int] | list[int] | None) -> PolicyChannels:
    """Parse the four convolution channel sizes used by PolicyValueNetwork."""
    if channels is None:
        return DEFAULT_POLICY_CHANNELS
    if isinstance(channels, str):
        parts = tuple(int(part.strip()) for part in channels.split(",") if part.strip())
    else:
        parts = tuple(int(part) for part in channels)
    if len(parts) != 4:
        raise ValueError("policy channels must contain exactly four integers")
    if any(part <= 0 for part in parts):
        raise ValueError("policy channels must be positive")
    return parts


def load_policy_network(
    model_path: str | Path,
    grid_size: int,
    key: jnp.ndarray | None = None,
    channels: str | PolicyChannels | list[int] | None = None,
    input_channels: int = 9,
) -> PolicyValueNetwork:
    """Load a PPO PolicyValueNetwork checkpoint from an Equinox .eqx file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")

    init_key = jrandom.PRNGKey(0) if key is None else key
    parsed_channels = parse_policy_channels(channels)
    network = PolicyValueNetwork(init_key, grid_size=grid_size, channels=parsed_channels, input_channels=input_channels)
    try:
        return eqx.tree_deserialise_leaves(path, network)
    except Exception as exc:
        raise ValueError(
            f"Failed to load PPO checkpoint for grid_size={grid_size}, "
            f"channels={parsed_channels}, input_channels={input_channels}: {path}"
        ) from exc


def load_policy_network_auto(
    model_path: str | Path,
    grid_size: int,
    key: jnp.ndarray | None = None,
    channels: str | PolicyChannels | list[int] | None = None,
) -> tuple[PolicyValueNetwork, PolicyInput, int]:
    """Load a checkpoint by trying supported input-channel layouts."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")

    parsed_channels = parse_policy_channels(channels)
    failures: list[Exception] = []
    for policy_input, input_channels in (("observation", 9), ("augmented-full-state", 18)):
        try:
            network = load_policy_network(
                path,
                grid_size,
                key=key,
                channels=parsed_channels,
                input_channels=input_channels,
            )
            return network, policy_input, input_channels
        except ValueError as exc:
            failures.append(exc)

    raise ValueError(f"Failed to auto-detect PPO checkpoint input layout: {model_path}") from failures[0]


class PPOPolicyAgent(Agent):
    """Agent wrapper for trained PPO .eqx checkpoints."""

    def __init__(
        self,
        model_path: str | Path,
        grid_size: int,
        policy_mode: PolicyMode = "greedy",
        agent_id: str = "PPO",
        channels: str | PolicyChannels | list[int] | None = None,
        policy_input: PolicyInputOption = "auto",
        input_channels: int | None = None,
        **kwargs: str,
    ):
        if "id" in kwargs:
            if agent_id != "PPO":
                raise TypeError("Pass only one of 'agent_id' or 'id'")
            agent_id = kwargs.pop("id")
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"Unexpected keyword argument: {unexpected}")

        super().__init__(agent_id)
        if policy_mode not in ("greedy", "sample"):
            raise ValueError("policy_mode must be 'greedy' or 'sample'")
        self.grid_size = grid_size
        self.policy_mode: PolicyMode = policy_mode
        self.channels = parse_policy_channels(channels)
        if policy_input == "auto":
            if input_channels is None:
                self.network, self.policy_input, self.input_channels = load_policy_network_auto(
                    model_path,
                    grid_size,
                    channels=self.channels,
                )
            else:
                if input_channels == 9:
                    policy_input = "observation"
                elif input_channels == 18:
                    policy_input = "augmented-full-state"
                else:
                    raise ValueError("policy_input='auto' only supports 9 or 18 input channels")
                self.policy_input = policy_input
                self.input_channels = input_channels
                self.network = load_policy_network(
                    model_path,
                    grid_size,
                    channels=self.channels,
                    input_channels=self.input_channels,
                )
        else:
            if policy_input not in POLICY_INPUT_NAMES:
                raise ValueError(f"policy_input must be one of {POLICY_INPUT_CHOICES}")
            expected_input_channels = policy_input_default_channels(policy_input)
            if input_channels is None:
                input_channels = expected_input_channels
            if input_channels <= 0:
                raise ValueError("input_channels must be positive")
            if input_channels != expected_input_channels:
                raise ValueError(
                    f"policy_input={policy_input!r} produces {expected_input_channels} input channels, "
                    f"got input_channels={input_channels}"
                )
            self.policy_input: PolicyInput = policy_input
            self.input_channels = input_channels
            self.network = load_policy_network(
                model_path,
                grid_size,
                channels=self.channels,
                input_channels=self.input_channels,
            )

    def _validate_grid_shape(self, shape: tuple[int, ...]) -> None:
        if shape != (self.grid_size, self.grid_size):
            raise ValueError(f"PPO checkpoint expects {self.grid_size}x{self.grid_size} observations, got {shape}")

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        if self.policy_input != "observation":
            raise ValueError(f"policy_input={self.policy_input!r} requires act_for_state instead of act")
        self._validate_grid_shape(observation.armies.shape)
        if self.policy_mode == "greedy":
            return greedy_policy_action(self.network, observation)
        return sampled_policy_action(self.network, observation, key)

    def act_for_state(self, state: game.GameState, player: int, key: jnp.ndarray) -> jnp.ndarray:
        """Return one action using the configured policy input for a full game state."""
        self._validate_grid_shape(state.armies.shape)
        return policy_state_action(self.network, key, state, player, self.policy_mode, self.policy_input)

    def explain(self, observation: Observation, top_k: int = 3) -> PolicyPreview:
        """Explain the current policy by returning the top semantic action candidates."""
        if self.policy_input != "observation":
            raise ValueError(f"policy_input={self.policy_input!r} requires explain_for_state instead of explain")
        self._validate_grid_shape(observation.armies.shape)
        return top_policy_preview(self.network, observation, top_k=top_k, policy_mode=self.policy_mode)

    def explain_for_state(self, state: game.GameState, player: int, top_k: int = 3) -> PolicyPreview:
        """Explain the configured policy input for a full game state."""
        self._validate_grid_shape(state.armies.shape)
        return top_policy_state_preview(
            self.network,
            state,
            player=player,
            top_k=top_k,
            policy_mode=self.policy_mode,
            policy_input=self.policy_input,
        )
