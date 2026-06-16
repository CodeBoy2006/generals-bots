"""Adaptive multisize policy-value network for experimental PPO."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import DEFAULT_POLICY_CHANNELS, PolicyChannels, parse_policy_channels

try:
    from .adaptive_common import (
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_action_to_index,
        adaptive_index_to_action,
    )
except ImportError:  # pragma: no cover - used when running this directory as scripts.
    from adaptive_common import (
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_action_to_index,
        adaptive_index_to_action,
    )


class AdaptivePolicyValueNetwork(eqx.Module):
    """Convolutional policy-value network with fixed canvas and active-cell pooling."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    policy_conv: eqx.nn.Conv2d
    pass_linear: eqx.nn.Linear
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear
    pad_size: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        pad_size: int = 16,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    ):
        keys = jrandom.split(key, 8)
        self.pad_size = pad_size
        self.conv1 = eqx.nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, key=keys[2])
        self.conv4 = eqx.nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, key=keys[3])
        self.policy_conv = eqx.nn.Conv2d(channels[3], 8, kernel_size=1, key=keys[4])
        self.pass_linear = eqx.nn.Linear(channels[3] * 2, 1, key=keys[5])
        self.value_linear1 = eqx.nn.Linear(channels[3] * 2, 64, key=keys[6])
        self.value_linear2 = eqx.nn.Linear(64, 1, key=keys[7])

    def _features(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        return jax.nn.relu(self.conv4(x))

    def _masked_pool(self, x: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        active_f = active.astype(jnp.float32)[None, :, :]
        denom = jnp.maximum(jnp.sum(active_f), 1.0)
        mean = jnp.sum(x * active_f, axis=(1, 2)) / denom
        masked = jnp.where(active_f > 0, x, -jnp.inf)
        max_pool = jnp.max(masked, axis=(1, 2))
        max_pool = jnp.where(jnp.isfinite(max_pool), max_pool, 0.0)
        return jnp.concatenate([mean, max_pool], axis=0)

    def logits_value(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute movement logits, one global pass logit, and scalar value."""
        x = self._features(obs)
        pooled = self._masked_pool(x, active)

        move_logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        move_mask = jnp.concatenate([mask_t, mask_t], axis=0)
        move_logits = move_logits + (1 - move_mask.astype(jnp.float32)) * -1e9

        pass_logit = self.pass_linear(pooled)
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        value = self.value_linear2(value_hidden)[0]
        logits = jnp.concatenate([move_logits.reshape(-1), pass_logit], axis=0)
        return logits, value

    def __call__(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        active: jnp.ndarray,
        key: jnp.ndarray | None,
        action: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample or score one action."""
        logits, value = self.logits_value(obs, mask, active)
        if action is None:
            if key is None:
                raise ValueError("key is required when sampling an action")
            index = jrandom.categorical(key, logits)
            action = adaptive_index_to_action(index, self.pad_size)
        else:
            index = adaptive_action_to_index(action, self.pad_size)

        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[index]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)
        return action, value, logprob, entropy


def load_or_create_adaptive_network(
    key: jnp.ndarray,
    pad_size: int,
    init_model_path: str | Path | None = None,
    channels: str | PolicyChannels | list[int] | None = None,
    input_channels: int = ADAPTIVE_INPUT_CHANNELS,
) -> AdaptivePolicyValueNetwork:
    """Create an adaptive network and optionally restore it from an Equinox checkpoint."""
    parsed_channels = parse_policy_channels(channels)
    network = AdaptivePolicyValueNetwork(
        key,
        pad_size=pad_size,
        channels=parsed_channels,
        input_channels=input_channels,
    )
    if init_model_path is None:
        return network
    path = Path(init_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    return eqx.tree_deserialise_leaves(path, network)
