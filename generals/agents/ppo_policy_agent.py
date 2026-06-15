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

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation

from .agent import Agent

PolicyMode = Literal["greedy", "sample"]

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

    def __init__(self, key: jnp.ndarray, grid_size: int = 4, channels: tuple[int, int, int, int] = (32, 32, 32, 16)):
        keys = jrandom.split(key, 8)

        self.conv1 = eqx.nn.Conv2d(9, channels[0], kernel_size=3, padding=1, key=keys[0])
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


def top_policy_preview(
    network: PolicyValueNetwork,
    obs: Observation,
    top_k: int,
    policy_mode: PolicyMode,
) -> PolicyPreview:
    """Return top semantic policy candidates with pass actions merged."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    obs_arr = obs_to_array(obs)
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    logits, value = network.logits_value(obs_arr, mask)

    grid_size = obs.armies.shape[-1]
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


def load_policy_network(model_path: str | Path, grid_size: int, key: jnp.ndarray | None = None) -> PolicyValueNetwork:
    """Load a PPO PolicyValueNetwork checkpoint from an Equinox .eqx file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")

    init_key = jrandom.PRNGKey(0) if key is None else key
    network = PolicyValueNetwork(init_key, grid_size=grid_size)
    try:
        return eqx.tree_deserialise_leaves(path, network)
    except Exception as exc:
        raise ValueError(f"Failed to load PPO checkpoint for grid_size={grid_size}: {path}") from exc


class PPOPolicyAgent(Agent):
    """Agent wrapper for trained PPO .eqx checkpoints."""

    def __init__(
        self,
        model_path: str | Path,
        grid_size: int,
        policy_mode: PolicyMode = "greedy",
        agent_id: str = "PPO",
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
        self.network = load_policy_network(model_path, grid_size)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        if observation.armies.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"PPO checkpoint expects {self.grid_size}x{self.grid_size} observations, "
                f"got {observation.armies.shape}"
            )
        if self.policy_mode == "greedy":
            return greedy_policy_action(self.network, observation)
        return sampled_policy_action(self.network, observation, key)

    def explain(self, observation: Observation, top_k: int = 3) -> PolicyPreview:
        """Explain the current policy by returning the top semantic action candidates."""
        if observation.armies.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"PPO checkpoint expects {self.grid_size}x{self.grid_size} observations, "
                f"got {observation.armies.shape}"
            )
        return top_policy_preview(self.network, observation, top_k=top_k, policy_mode=self.policy_mode)
