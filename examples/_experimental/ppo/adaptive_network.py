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
    size_value_linear1: tuple[eqx.nn.Linear, ...]
    size_value_linear2: tuple[eqx.nn.Linear, ...]
    pad_size: int = eqx.field(static=True)
    value_head_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        pad_size: int = 16,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = ADAPTIVE_INPUT_CHANNELS,
        value_head_sizes: tuple[int, ...] | None = None,
    ):
        parsed_value_head_sizes = tuple(value_head_sizes or ())
        keys = jrandom.split(key, 8 + 2 * len(parsed_value_head_sizes))
        self.pad_size = pad_size
        self.value_head_sizes = parsed_value_head_sizes
        self.conv1 = eqx.nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, key=keys[2])
        self.conv4 = eqx.nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, key=keys[3])
        self.policy_conv = eqx.nn.Conv2d(channels[3], 8, kernel_size=1, key=keys[4])
        self.pass_linear = eqx.nn.Linear(channels[3] * 2, 1, key=keys[5])
        self.value_linear1 = eqx.nn.Linear(channels[3] * 2, 64, key=keys[6])
        self.value_linear2 = eqx.nn.Linear(64, 1, key=keys[7])
        self.size_value_linear1 = tuple(
            eqx.nn.Linear(channels[3] * 2, 64, key=keys[8 + 2 * index])
            for index, _ in enumerate(parsed_value_head_sizes)
        )
        self.size_value_linear2 = tuple(
            eqx.nn.Linear(64, 1, key=keys[9 + 2 * index]) for index, _ in enumerate(parsed_value_head_sizes)
        )

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

    def _shared_value(self, pooled: jnp.ndarray) -> jnp.ndarray:
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        return self.value_linear2(value_hidden)[0]

    def _size_value(self, pooled: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        if not self.value_head_sizes:
            return self._shared_value(pooled)

        active_cells = jnp.sum(active.astype(jnp.int32))
        head_cells = jnp.array([size * size for size in self.value_head_sizes], dtype=jnp.int32)
        head_index = jnp.argmin(jnp.abs(head_cells - active_cells))
        branches = tuple(
            (lambda linear1, linear2: lambda _: linear2(jax.nn.relu(linear1(pooled)))[0])(linear1, linear2)
            for linear1, linear2 in zip(self.size_value_linear1, self.size_value_linear2, strict=True)
        )
        return jax.lax.switch(head_index, branches, None)

    def logits_value(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute movement logits, one global pass logit, and scalar value."""
        x = self._features(obs)
        pooled = self._masked_pool(x, active)

        move_logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        move_mask = jnp.concatenate([mask_t, mask_t], axis=0)
        move_logits = move_logits + (1 - move_mask.astype(jnp.float32)) * -1e9

        pass_logit = self.pass_linear(pooled)
        value = self._size_value(pooled, active)
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
    init_channels: str | PolicyChannels | list[int] | None = None,
    input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    value_head_sizes: tuple[int, ...] | None = None,
    init_value_head_sizes: tuple[int, ...] | None = None,
) -> AdaptivePolicyValueNetwork:
    """Create an adaptive network and optionally restore it from an Equinox checkpoint."""
    parsed_channels = parse_policy_channels(channels)
    parsed_value_head_sizes = _normalize_value_head_sizes(value_head_sizes)
    network = AdaptivePolicyValueNetwork(
        key,
        pad_size=pad_size,
        channels=parsed_channels,
        input_channels=input_channels,
        value_head_sizes=parsed_value_head_sizes,
    )
    if init_model_path is None:
        return network
    path = Path(init_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    parsed_init_channels = parse_policy_channels(init_channels) if init_channels is not None else parsed_channels
    parsed_init_value_head_sizes = (
        parsed_value_head_sizes
        if init_value_head_sizes is None
        else _normalize_value_head_sizes(init_value_head_sizes)
    )
    if parsed_init_channels != parsed_channels or parsed_init_value_head_sizes != parsed_value_head_sizes:
        source_network = AdaptivePolicyValueNetwork(
            key,
            pad_size=pad_size,
            channels=parsed_init_channels,
            input_channels=input_channels,
            value_head_sizes=parsed_init_value_head_sizes,
        )
        source_network = eqx.tree_deserialise_leaves(path, source_network)
        return expand_adaptive_network_channels(network, source_network)
    return eqx.tree_deserialise_leaves(path, network)


def _normalize_value_head_sizes(value_head_sizes: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    """Validate optional independent value-head sizes."""
    if value_head_sizes is None:
        return ()
    sizes = tuple(int(size) for size in value_head_sizes)
    if len(set(sizes)) != len(sizes):
        raise ValueError("value head sizes must be unique")
    if any(size <= 0 for size in sizes):
        raise ValueError("value head sizes must be positive")
    return sizes


def _copy_conv_prefix(target: eqx.nn.Conv2d, source: eqx.nn.Conv2d) -> eqx.nn.Conv2d:
    """Copy source outputs while keeping extra target output channels trainable."""
    out_channels = min(target.weight.shape[0], source.weight.shape[0])
    in_channels = min(target.weight.shape[1], source.weight.shape[1])
    weight = target.weight.at[:out_channels].set(0.0)
    weight = weight.at[:out_channels, :in_channels].set(source.weight[:out_channels, :in_channels])
    bias = target.bias
    if target.bias is not None and source.bias is not None:
        bias = bias.at[:out_channels].set(source.bias[:out_channels])
    return eqx.tree_at(lambda layer: (layer.weight, layer.bias), target, (weight, bias))


def _copy_linear_prefix(target: eqx.nn.Linear, source: eqx.nn.Linear) -> eqx.nn.Linear:
    """Copy a smaller linear layer into a larger layer prefix and zero unused inputs."""
    out_features = min(target.weight.shape[0], source.weight.shape[0])
    in_features = min(target.weight.shape[1], source.weight.shape[1])
    weight = jnp.zeros_like(target.weight)
    weight = weight.at[:out_features, :in_features].set(source.weight[:out_features, :in_features])
    bias = None
    if target.bias is not None and source.bias is not None:
        bias = jnp.zeros_like(target.bias)
        bias = bias.at[:out_features].set(source.bias[:out_features])
    return eqx.tree_at(lambda layer: (layer.weight, layer.bias), target, (weight, bias))


def _copy_pooled_linear_prefix(
    target: eqx.nn.Linear,
    source: eqx.nn.Linear,
    target_channels: int,
    source_channels: int,
) -> eqx.nn.Linear:
    """Copy pooled [mean, max] weights while preserving the max-channel offset."""
    out_features = min(target.weight.shape[0], source.weight.shape[0])
    copied_channels = min(target_channels, source_channels)
    weight = jnp.zeros_like(target.weight)
    weight = weight.at[:out_features, :copied_channels].set(source.weight[:out_features, :copied_channels])
    weight = weight.at[:out_features, target_channels : target_channels + copied_channels].set(
        source.weight[:out_features, source_channels : source_channels + copied_channels]
    )
    bias = None
    if target.bias is not None and source.bias is not None:
        bias = jnp.zeros_like(target.bias)
        bias = bias.at[:out_features].set(source.bias[:out_features])
    return eqx.tree_at(lambda layer: (layer.weight, layer.bias), target, (weight, bias))


def expand_adaptive_network_channels(
    target: AdaptivePolicyValueNetwork,
    source: AdaptivePolicyValueNetwork,
) -> AdaptivePolicyValueNetwork:
    """Initialize a wider adaptive network so it initially matches a narrower source."""
    target_channels = target.conv4.weight.shape[0]
    source_channels = source.conv4.weight.shape[0]
    target = eqx.tree_at(lambda net: net.conv1, target, _copy_conv_prefix(target.conv1, source.conv1))
    target = eqx.tree_at(lambda net: net.conv2, target, _copy_conv_prefix(target.conv2, source.conv2))
    target = eqx.tree_at(lambda net: net.conv3, target, _copy_conv_prefix(target.conv3, source.conv3))
    target = eqx.tree_at(lambda net: net.conv4, target, _copy_conv_prefix(target.conv4, source.conv4))
    target = eqx.tree_at(
        lambda net: net.policy_conv,
        target,
        _copy_conv_prefix(target.policy_conv, source.policy_conv),
    )
    target = eqx.tree_at(
        lambda net: net.pass_linear,
        target,
        _copy_pooled_linear_prefix(
            target.pass_linear,
            source.pass_linear,
            target_channels,
            source_channels,
        ),
    )
    target = eqx.tree_at(
        lambda net: net.value_linear1,
        target,
        _copy_pooled_linear_prefix(
            target.value_linear1,
            source.value_linear1,
            target_channels,
            source_channels,
        ),
    )
    target = eqx.tree_at(
        lambda net: net.value_linear2,
        target,
        _copy_linear_prefix(target.value_linear2, source.value_linear2),
    )
    if target.value_head_sizes:
        source_size_heads = {
            size: (linear1, linear2)
            for size, linear1, linear2 in zip(
                source.value_head_sizes,
                source.size_value_linear1,
                source.size_value_linear2,
                strict=True,
            )
        }
        target_size_linear1 = []
        target_size_linear2 = []
        for size, target_linear1, target_linear2 in zip(
            target.value_head_sizes,
            target.size_value_linear1,
            target.size_value_linear2,
            strict=True,
        ):
            source_linear1, source_linear2 = source_size_heads.get(size, (source.value_linear1, source.value_linear2))
            target_size_linear1.append(
                _copy_pooled_linear_prefix(target_linear1, source_linear1, target_channels, source_channels)
            )
            target_size_linear2.append(_copy_linear_prefix(target_linear2, source_linear2))
        target = eqx.tree_at(lambda net: net.size_value_linear1, target, tuple(target_size_linear1))
        target = eqx.tree_at(lambda net: net.size_value_linear2, target, tuple(target_size_linear2))
    return target
