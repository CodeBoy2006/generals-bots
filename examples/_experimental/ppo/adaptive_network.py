"""Adaptive multisize policy-value network for experimental PPO."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import DEFAULT_POLICY_CHANNELS, PolicyChannels, parse_policy_channels

try:
    from .adaptive_common import (
        ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        ADAPTIVE_HISTORY_INPUT_CHANNELS,
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_action_to_index,
        adaptive_index_to_action,
    )
except ImportError:  # pragma: no cover - used when running this directory as scripts.
    from adaptive_common import (
        ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        ADAPTIVE_HISTORY_INPUT_CHANNELS,
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_action_to_index,
        adaptive_index_to_action,
    )


class StrategyAuxOutputs(NamedTuple):
    """Strategic auxiliary predictions for intent, finish, belief, and action Q."""

    intent_logits: jnp.ndarray
    finish_logits: jnp.ndarray
    enemy_general_logits: jnp.ndarray
    action_q_values: jnp.ndarray
    source_logits: jnp.ndarray | None
    target_logits: jnp.ndarray | None


def value_bin_centers(value_bins: int, value_min: float = -1.0, value_max: float = 1.0) -> jnp.ndarray:
    """Return evenly spaced categorical value support points."""
    if value_bins <= 1:
        raise ValueError("value_bins must be greater than 1")
    if value_min >= value_max:
        raise ValueError("value_min must be less than value_max")
    return jnp.linspace(value_min, value_max, value_bins, dtype=jnp.float32)


def hl_gauss_target(
    target: jnp.ndarray,
    value_bins: int,
    value_min: float = -1.0,
    value_max: float = 1.0,
    value_sigma: float = 0.04,
) -> jnp.ndarray:
    """Project scalar value targets onto an HL-Gauss categorical support."""
    if value_sigma <= 0.0:
        raise ValueError("value_sigma must be positive")
    centers = value_bin_centers(value_bins, value_min, value_max)
    clipped = jnp.clip(target, value_min, value_max)
    logits = -0.5 * ((centers - clipped[..., None]) / value_sigma) ** 2
    return jax.nn.softmax(logits, axis=-1)


def categorical_value_expectation(
    value_logits: jnp.ndarray,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> jnp.ndarray:
    """Convert categorical value logits back to a scalar expectation."""
    centers = value_bin_centers(value_logits.shape[-1], value_min, value_max)
    return jnp.sum(jax.nn.softmax(value_logits, axis=-1) * centers, axis=-1)


def hl_gauss_value_loss(
    value_logits: jnp.ndarray,
    target: jnp.ndarray,
    value_bins: int,
    value_min: float = -1.0,
    value_max: float = 1.0,
    value_sigma: float = 0.04,
) -> jnp.ndarray:
    """Cross-entropy between predicted value logits and HL-Gauss target distribution."""
    target_probs = hl_gauss_target(target, value_bins, value_min, value_max, value_sigma)
    return -jnp.sum(target_probs * jax.nn.log_softmax(value_logits, axis=-1), axis=-1)


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
    categorical_value_linear2: eqx.nn.Linear | None
    outcome_linear2: eqx.nn.Linear | None
    strategy_intent_linear2: eqx.nn.Linear | None
    strategy_finish_linear2: eqx.nn.Linear | None
    strategy_q_conv: eqx.nn.Conv2d | None
    strategy_q_pass_linear: eqx.nn.Linear | None
    strategy_enemy_general_conv: eqx.nn.Conv2d | None
    strategy_source_conv: eqx.nn.Conv2d | None
    strategy_target_conv: eqx.nn.Conv2d | None
    context_conv1: eqx.nn.Conv2d | None
    context_conv2: eqx.nn.Conv2d | None
    pyramid_down1: eqx.nn.Conv2d | None
    pyramid_down2: eqx.nn.Conv2d | None
    pyramid_up1: eqx.nn.Conv2d | None
    pyramid_up2: eqx.nn.Conv2d | None
    global_linear1: eqx.nn.Linear | None
    global_linear2: eqx.nn.Linear | None
    size_value_linear1: tuple[eqx.nn.Linear, ...]
    size_value_linear2: tuple[eqx.nn.Linear, ...]
    size_categorical_value_linear2: tuple[eqx.nn.Linear, ...]
    pad_size: int = eqx.field(static=True)
    value_head_sizes: tuple[int, ...] = eqx.field(static=True)
    value_bins: int = eqx.field(static=True)
    value_min: float = eqx.field(static=True)
    value_max: float = eqx.field(static=True)
    value_sigma: float = eqx.field(static=True)
    outcome_head: bool = eqx.field(static=True)
    strategy_aux: bool = eqx.field(static=True)
    strategy_spatial_aux: bool = eqx.field(static=True)
    global_context: bool = eqx.field(static=True)
    context_residual: bool = eqx.field(static=True)
    pyramid_context: bool = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        pad_size: int = 16,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = ADAPTIVE_INPUT_CHANNELS,
        value_head_sizes: tuple[int, ...] | None = None,
        value_bins: int = 0,
        value_min: float = -1.0,
        value_max: float = 1.0,
        value_sigma: float = 0.04,
        outcome_head: bool = False,
        strategy_aux: bool = False,
        strategy_spatial_aux: bool = False,
        global_context: bool = False,
        context_residual: bool = False,
        pyramid_context: bool = False,
    ):
        if strategy_spatial_aux and not strategy_aux:
            raise ValueError("strategy_spatial_aux requires strategy_aux")
        if global_context and input_channels == ADAPTIVE_INPUT_CHANNELS:
            input_channels = ADAPTIVE_GLOBAL_INPUT_CHANNELS
        parsed_value_head_sizes = tuple(value_head_sizes or ())
        parsed_value_bins = _normalize_value_bins(value_bins, value_min, value_max, value_sigma)
        categorical_keys = 1 + len(parsed_value_head_sizes) if parsed_value_bins > 0 else 0
        outcome_keys = 1 if outcome_head else 0
        strategy_keys = 5 if strategy_aux else 0
        strategy_spatial_keys = 2 if strategy_spatial_aux else 0
        global_keys = 2 if global_context else 0
        context_keys = 2 if context_residual else 0
        pyramid_keys = 4 if pyramid_context else 0
        keys = jrandom.split(
            key,
            8
            + 2 * len(parsed_value_head_sizes)
            + categorical_keys
            + outcome_keys
            + global_keys
            + strategy_keys
            + strategy_spatial_keys
            + context_keys
            + pyramid_keys,
        )
        self.pad_size = pad_size
        self.value_head_sizes = parsed_value_head_sizes
        self.value_bins = parsed_value_bins
        self.value_min = value_min
        self.value_max = value_max
        self.value_sigma = value_sigma
        self.outcome_head = bool(outcome_head)
        self.strategy_aux = bool(strategy_aux)
        self.strategy_spatial_aux = bool(strategy_spatial_aux)
        self.global_context = bool(global_context)
        self.context_residual = bool(context_residual)
        self.pyramid_context = bool(pyramid_context)
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
        categorical_offset = 8 + 2 * len(parsed_value_head_sizes)
        self.categorical_value_linear2 = (
            eqx.nn.Linear(64, parsed_value_bins, key=keys[categorical_offset]) if parsed_value_bins > 0 else None
        )
        self.size_categorical_value_linear2 = tuple(
            eqx.nn.Linear(64, parsed_value_bins, key=keys[categorical_offset + 1 + index])
            for index, _ in enumerate(parsed_value_head_sizes)
            if parsed_value_bins > 0
        )
        outcome_offset = categorical_offset + categorical_keys
        self.outcome_linear2 = eqx.nn.Linear(64, 3, key=keys[outcome_offset]) if outcome_head else None
        global_offset = outcome_offset + outcome_keys
        if global_context:
            global_context_features = input_channels - 13
            if global_context_features <= 0:
                raise ValueError("global_context requires at least 14 input channels")
            self.global_linear1 = eqx.nn.Linear(global_context_features, 64, key=keys[global_offset])
            global_linear2 = eqx.nn.Linear(64, channels[3], key=keys[global_offset + 1])
            self.global_linear2 = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                global_linear2,
                (jnp.zeros_like(global_linear2.weight), jnp.zeros_like(global_linear2.bias)),
            )
        else:
            self.global_linear1 = None
            self.global_linear2 = None
        strategy_offset = global_offset + global_keys
        if strategy_aux:
            self.strategy_intent_linear2 = eqx.nn.Linear(64, 8, key=keys[strategy_offset])
            self.strategy_finish_linear2 = eqx.nn.Linear(64, 2, key=keys[strategy_offset + 1])
            self.strategy_q_conv = eqx.nn.Conv2d(channels[3], 8, kernel_size=1, key=keys[strategy_offset + 2])
            self.strategy_q_pass_linear = eqx.nn.Linear(channels[3] * 2, 1, key=keys[strategy_offset + 3])
            self.strategy_enemy_general_conv = eqx.nn.Conv2d(
                channels[3],
                1,
                kernel_size=1,
                key=keys[strategy_offset + 4],
            )
        else:
            self.strategy_intent_linear2 = None
            self.strategy_finish_linear2 = None
            self.strategy_q_conv = None
            self.strategy_q_pass_linear = None
            self.strategy_enemy_general_conv = None
        strategy_spatial_offset = strategy_offset + strategy_keys
        if strategy_spatial_aux:
            self.strategy_source_conv = eqx.nn.Conv2d(
                channels[3],
                1,
                kernel_size=1,
                key=keys[strategy_spatial_offset],
            )
            self.strategy_target_conv = eqx.nn.Conv2d(
                channels[3],
                1,
                kernel_size=1,
                key=keys[strategy_spatial_offset + 1],
            )
        else:
            self.strategy_source_conv = None
            self.strategy_target_conv = None
        context_offset = strategy_spatial_offset + strategy_spatial_keys
        if context_residual:
            self.context_conv1 = eqx.nn.Conv2d(channels[3], channels[3], kernel_size=5, padding=2, key=keys[context_offset])
            context_conv2 = eqx.nn.Conv2d(
                channels[3],
                channels[3],
                kernel_size=5,
                padding=2,
                key=keys[context_offset + 1],
            )
            self.context_conv2 = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                context_conv2,
                (jnp.zeros_like(context_conv2.weight), jnp.zeros_like(context_conv2.bias)),
            )
        else:
            self.context_conv1 = None
            self.context_conv2 = None
        pyramid_offset = context_offset + context_keys
        if pyramid_context:
            self.pyramid_down1 = eqx.nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1, key=keys[pyramid_offset])
            self.pyramid_down2 = eqx.nn.Conv2d(
                channels[3],
                channels[3],
                kernel_size=3,
                padding=1,
                key=keys[pyramid_offset + 1],
            )
            self.pyramid_up1 = eqx.nn.Conv2d(
                channels[3],
                channels[3],
                kernel_size=3,
                padding=1,
                key=keys[pyramid_offset + 2],
            )
            pyramid_up2 = eqx.nn.Conv2d(
                channels[3],
                channels[3],
                kernel_size=3,
                padding=1,
                key=keys[pyramid_offset + 3],
            )
            self.pyramid_up2 = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                pyramid_up2,
                (jnp.zeros_like(pyramid_up2.weight), jnp.zeros_like(pyramid_up2.bias)),
            )
        else:
            self.pyramid_down1 = None
            self.pyramid_down2 = None
            self.pyramid_up1 = None
            self.pyramid_up2 = None

    def _features(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.conv4(x))
        if self.context_conv1 is not None and self.context_conv2 is not None:
            x = x + self.context_conv2(jax.nn.relu(self.context_conv1(x)))
        if (
            self.pyramid_down1 is not None
            and self.pyramid_down2 is not None
            and self.pyramid_up1 is not None
            and self.pyramid_up2 is not None
        ):
            x = x + self._pyramid_context_features(x)
        if self.global_context:
            x = x + self._global_context_vector(obs)[:, None, None]
        return x

    def _avg_pool2x(self, x: jnp.ndarray) -> jnp.ndarray:
        pooled = jax.lax.reduce_window(
            x,
            0.0,
            jax.lax.add,
            window_dimensions=(1, 2, 2),
            window_strides=(1, 2, 2),
            padding="VALID",
        )
        return pooled * 0.25

    def _resize_nearest(self, x: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        src_height = x.shape[1]
        src_width = x.shape[2]
        row_idx = jnp.floor(jnp.arange(height, dtype=jnp.float32) * (src_height / height)).astype(jnp.int32)
        col_idx = jnp.floor(jnp.arange(width, dtype=jnp.float32) * (src_width / width)).astype(jnp.int32)
        return x[:, row_idx, :][:, :, col_idx]

    def _pyramid_context_features(self, x: jnp.ndarray) -> jnp.ndarray:
        p1 = jax.nn.relu(self.pyramid_down1(self._avg_pool2x(x)))
        p2 = jax.nn.relu(self.pyramid_down2(self._avg_pool2x(p1)))
        p1 = p1 + self._resize_nearest(p2, p1.shape[1], p1.shape[2])
        up = jax.nn.relu(self.pyramid_up1(self._resize_nearest(p1, x.shape[1], x.shape[2])))
        return self.pyramid_up2(up)

    def _global_context_vector(self, obs: jnp.ndarray) -> jnp.ndarray:
        if self.global_linear1 is None or self.global_linear2 is None:
            return jnp.zeros((self.conv4.weight.shape[0],), dtype=obs.dtype)
        active = obs[9] > 0.5
        active_f = active.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(active_f), 1.0)
        global_features = jnp.sum(obs[13:] * active_f[None, :, :], axis=(1, 2)) / denom
        hidden = jax.nn.relu(self.global_linear1(global_features))
        return self.global_linear2(hidden)

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

    def _shared_value_logits(self, pooled: jnp.ndarray) -> jnp.ndarray:
        if self.categorical_value_linear2 is None:
            raise ValueError("categorical value head is not configured")
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        return self.categorical_value_linear2(value_hidden)

    def _outcome_logits(self, pooled: jnp.ndarray) -> jnp.ndarray | None:
        if self.outcome_linear2 is None:
            return None
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        return self.outcome_linear2(value_hidden)

    def strategy_auxiliary(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> StrategyAuxOutputs:
        """Compute optional strategic auxiliary predictions."""
        if (
            self.strategy_intent_linear2 is None
            or self.strategy_finish_linear2 is None
            or self.strategy_q_conv is None
            or self.strategy_q_pass_linear is None
            or self.strategy_enemy_general_conv is None
        ):
            raise ValueError("strategy auxiliary heads are not configured")
        x = self._features(obs)
        pooled = self._masked_pool(x, active)
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        move_q = self.strategy_q_conv(x)
        pass_q = self.strategy_q_pass_linear(pooled)
        action_q_values = jnp.concatenate([move_q.reshape(-1), pass_q], axis=0)
        enemy_general_logits = self.strategy_enemy_general_conv(x)[0]
        enemy_general_logits = jnp.where(active, enemy_general_logits, -10.0)
        source_logits = None
        target_logits = None
        if self.strategy_source_conv is not None and self.strategy_target_conv is not None:
            source_logits = self.strategy_source_conv(x)[0]
            target_logits = self.strategy_target_conv(x)[0]
            source_logits = jnp.where(active, source_logits, -10.0)
            target_logits = jnp.where(active, target_logits, -10.0)
        return StrategyAuxOutputs(
            intent_logits=self.strategy_intent_linear2(value_hidden),
            finish_logits=self.strategy_finish_linear2(value_hidden),
            enemy_general_logits=enemy_general_logits,
            action_q_values=action_q_values,
            source_logits=source_logits,
            target_logits=target_logits,
        )

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

    def _size_value_logits(self, pooled: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        if self.categorical_value_linear2 is None:
            raise ValueError("categorical value head is not configured")
        if not self.value_head_sizes:
            return self._shared_value_logits(pooled)

        active_cells = jnp.sum(active.astype(jnp.int32))
        head_cells = jnp.array([size * size for size in self.value_head_sizes], dtype=jnp.int32)
        head_index = jnp.argmin(jnp.abs(head_cells - active_cells))
        branches = tuple(
            (lambda linear1, linear2: lambda _: linear2(jax.nn.relu(linear1(pooled))))(linear1, linear2)
            for linear1, linear2 in zip(
                self.size_value_linear1,
                self.size_categorical_value_linear2,
                strict=True,
            )
        )
        return jax.lax.switch(head_index, branches, None)

    def _value_distribution(self, pooled: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        if self.value_bins <= 0:
            return self._size_value(pooled, active), None
        value_logits = self._size_value_logits(pooled, active)
        value = categorical_value_expectation(value_logits, self.value_min, self.value_max)
        return value, value_logits

    def logits_value(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute movement logits, one global pass logit, and scalar value."""
        logits, value, _ = self.logits_value_distribution(obs, mask, active)
        return logits, value

    def logits_value_distribution(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        active: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Compute policy logits, scalar value, and optional categorical value logits."""
        logits, value, value_logits, _ = self.logits_value_auxiliary(obs, mask, active)
        return logits, value, value_logits

    def logits_value_auxiliary(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        active: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Compute policy/value outputs plus optional auxiliary outcome logits."""
        x = self._features(obs)
        pooled = self._masked_pool(x, active)

        move_logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        move_mask = jnp.concatenate([mask_t, mask_t], axis=0)
        move_logits = move_logits + (1 - move_mask.astype(jnp.float32)) * -1e9

        pass_logit = self.pass_linear(pooled)
        value, value_logits = self._value_distribution(pooled, active)
        outcome_logits = self._outcome_logits(pooled)
        logits = jnp.concatenate([move_logits.reshape(-1), pass_logit], axis=0)
        return logits, value, value_logits, outcome_logits

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


class AdaptiveUNetPolicyValueNetwork(eqx.Module):
    """Small U-Net policy-value network that makes multiscale features the main trunk."""

    enc1: eqx.nn.Conv2d
    enc2: eqx.nn.Conv2d
    bottleneck: eqx.nn.Conv2d
    dec1: eqx.nn.Conv2d
    dec2: eqx.nn.Conv2d
    policy_conv: eqx.nn.Conv2d
    pass_linear: eqx.nn.Linear
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear
    categorical_value_linear2: eqx.nn.Linear | None
    outcome_linear2: eqx.nn.Linear | None
    strategy_intent_linear2: eqx.nn.Linear | None
    strategy_finish_linear2: eqx.nn.Linear | None
    strategy_q_conv: eqx.nn.Conv2d | None
    strategy_q_pass_linear: eqx.nn.Linear | None
    strategy_enemy_general_conv: eqx.nn.Conv2d | None
    strategy_source_conv: eqx.nn.Conv2d | None
    strategy_target_conv: eqx.nn.Conv2d | None
    size_value_linear1: tuple[eqx.nn.Linear, ...]
    size_value_linear2: tuple[eqx.nn.Linear, ...]
    size_categorical_value_linear2: tuple[eqx.nn.Linear, ...]
    pad_size: int = eqx.field(static=True)
    value_head_sizes: tuple[int, ...] = eqx.field(static=True)
    value_bins: int = eqx.field(static=True)
    value_min: float = eqx.field(static=True)
    value_max: float = eqx.field(static=True)
    value_sigma: float = eqx.field(static=True)
    outcome_head: bool = eqx.field(static=True)
    strategy_aux: bool = eqx.field(static=True)
    strategy_spatial_aux: bool = eqx.field(static=True)
    global_context: bool = eqx.field(static=True)
    context_residual: bool = eqx.field(static=True, default=False)
    pyramid_context: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        key: jnp.ndarray,
        pad_size: int = 16,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = ADAPTIVE_INPUT_CHANNELS,
        value_head_sizes: tuple[int, ...] | None = None,
        value_bins: int = 0,
        value_min: float = -1.0,
        value_max: float = 1.0,
        value_sigma: float = 0.04,
        outcome_head: bool = False,
        strategy_aux: bool = False,
        strategy_spatial_aux: bool = False,
        global_context: bool = False,
        context_residual: bool = False,
        pyramid_context: bool = False,
    ):
        if context_residual or pyramid_context:
            raise ValueError("AdaptiveUNetPolicyValueNetwork replaces context/pyramid add-on branches")
        if strategy_spatial_aux and not strategy_aux:
            raise ValueError("strategy_spatial_aux requires strategy_aux")
        if global_context and input_channels == ADAPTIVE_INPUT_CHANNELS:
            input_channels = ADAPTIVE_GLOBAL_INPUT_CHANNELS
        parsed_value_head_sizes = tuple(value_head_sizes or ())
        parsed_value_bins = _normalize_value_bins(value_bins, value_min, value_max, value_sigma)
        categorical_keys = 1 + len(parsed_value_head_sizes) if parsed_value_bins > 0 else 0
        outcome_keys = 1 if outcome_head else 0
        strategy_keys = 5 if strategy_aux else 0
        strategy_spatial_keys = 2 if strategy_spatial_aux else 0
        keys = jrandom.split(
            key,
            9
            + 2 * len(parsed_value_head_sizes)
            + categorical_keys
            + outcome_keys
            + strategy_keys
            + strategy_spatial_keys,
        )
        self.pad_size = pad_size
        self.value_head_sizes = parsed_value_head_sizes
        self.value_bins = parsed_value_bins
        self.value_min = value_min
        self.value_max = value_max
        self.value_sigma = value_sigma
        self.outcome_head = bool(outcome_head)
        self.strategy_aux = bool(strategy_aux)
        self.strategy_spatial_aux = bool(strategy_spatial_aux)
        self.global_context = bool(global_context)
        self.context_residual = False
        self.pyramid_context = False
        c1, c2, c3, c4 = channels
        self.enc1 = eqx.nn.Conv2d(input_channels, c1, kernel_size=3, padding=1, key=keys[0])
        self.enc2 = eqx.nn.Conv2d(c1, c2, kernel_size=3, padding=1, key=keys[1])
        self.bottleneck = eqx.nn.Conv2d(c2, c3, kernel_size=3, padding=1, key=keys[2])
        self.dec1 = eqx.nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1, key=keys[3])
        self.dec2 = eqx.nn.Conv2d(c2 + c1, c4, kernel_size=3, padding=1, key=keys[4])
        self.policy_conv = eqx.nn.Conv2d(c4, 8, kernel_size=1, key=keys[5])
        self.pass_linear = eqx.nn.Linear(c4 * 2, 1, key=keys[6])
        self.value_linear1 = eqx.nn.Linear(c4 * 2, 64, key=keys[7])
        value_key_offset = 8
        self.value_linear2 = eqx.nn.Linear(64, 1, key=keys[value_key_offset])
        self.size_value_linear1 = tuple(
            eqx.nn.Linear(c4 * 2, 64, key=keys[value_key_offset + 1 + 2 * index])
            for index, _ in enumerate(parsed_value_head_sizes)
        )
        self.size_value_linear2 = tuple(
            eqx.nn.Linear(64, 1, key=keys[value_key_offset + 2 + 2 * index])
            for index, _ in enumerate(parsed_value_head_sizes)
        )
        categorical_offset = value_key_offset + 1 + 2 * len(parsed_value_head_sizes)
        self.categorical_value_linear2 = (
            eqx.nn.Linear(64, parsed_value_bins, key=keys[categorical_offset]) if parsed_value_bins > 0 else None
        )
        self.size_categorical_value_linear2 = tuple(
            eqx.nn.Linear(64, parsed_value_bins, key=keys[categorical_offset + 1 + index])
            for index, _ in enumerate(parsed_value_head_sizes)
            if parsed_value_bins > 0
        )
        outcome_offset = categorical_offset + categorical_keys
        self.outcome_linear2 = eqx.nn.Linear(64, 3, key=keys[outcome_offset]) if outcome_head else None
        strategy_offset = outcome_offset + outcome_keys
        if strategy_aux:
            self.strategy_intent_linear2 = eqx.nn.Linear(64, 8, key=keys[strategy_offset])
            self.strategy_finish_linear2 = eqx.nn.Linear(64, 2, key=keys[strategy_offset + 1])
            self.strategy_q_conv = eqx.nn.Conv2d(c4, 8, kernel_size=1, key=keys[strategy_offset + 2])
            self.strategy_q_pass_linear = eqx.nn.Linear(c4 * 2, 1, key=keys[strategy_offset + 3])
            self.strategy_enemy_general_conv = eqx.nn.Conv2d(c4, 1, kernel_size=1, key=keys[strategy_offset + 4])
        else:
            self.strategy_intent_linear2 = None
            self.strategy_finish_linear2 = None
            self.strategy_q_conv = None
            self.strategy_q_pass_linear = None
            self.strategy_enemy_general_conv = None
        strategy_spatial_offset = strategy_offset + strategy_keys
        if strategy_spatial_aux:
            self.strategy_source_conv = eqx.nn.Conv2d(c4, 1, kernel_size=1, key=keys[strategy_spatial_offset])
            self.strategy_target_conv = eqx.nn.Conv2d(c4, 1, kernel_size=1, key=keys[strategy_spatial_offset + 1])
        else:
            self.strategy_source_conv = None
            self.strategy_target_conv = None

    def _avg_pool2x(self, x: jnp.ndarray) -> jnp.ndarray:
        pooled = jax.lax.reduce_window(
            x,
            0.0,
            jax.lax.add,
            window_dimensions=(1, 2, 2),
            window_strides=(1, 2, 2),
            padding="VALID",
        )
        return pooled * 0.25

    def _resize_nearest(self, x: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        src_height = x.shape[1]
        src_width = x.shape[2]
        row_idx = jnp.floor(jnp.arange(height, dtype=jnp.float32) * (src_height / height)).astype(jnp.int32)
        col_idx = jnp.floor(jnp.arange(width, dtype=jnp.float32) * (src_width / width)).astype(jnp.int32)
        return x[:, row_idx, :][:, :, col_idx]

    def _features(self, obs: jnp.ndarray) -> jnp.ndarray:
        skip1 = jax.nn.relu(self.enc1(obs))
        skip2 = jax.nn.relu(self.enc2(self._avg_pool2x(skip1)))
        bridge = jax.nn.relu(self.bottleneck(self._avg_pool2x(skip2)))
        up1 = self._resize_nearest(bridge, skip2.shape[1], skip2.shape[2])
        dec1 = jax.nn.relu(self.dec1(jnp.concatenate([up1, skip2], axis=0)))
        up2 = self._resize_nearest(dec1, skip1.shape[1], skip1.shape[2])
        return jax.nn.relu(self.dec2(jnp.concatenate([up2, skip1], axis=0)))

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

    def _shared_value_logits(self, pooled: jnp.ndarray) -> jnp.ndarray:
        if self.categorical_value_linear2 is None:
            raise ValueError("categorical value head is not configured")
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        return self.categorical_value_linear2(value_hidden)

    def _outcome_logits(self, pooled: jnp.ndarray) -> jnp.ndarray | None:
        if self.outcome_linear2 is None:
            return None
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        return self.outcome_linear2(value_hidden)

    def strategy_auxiliary(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> StrategyAuxOutputs:
        """Compute optional strategic auxiliary predictions."""
        if (
            self.strategy_intent_linear2 is None
            or self.strategy_finish_linear2 is None
            or self.strategy_q_conv is None
            or self.strategy_q_pass_linear is None
            or self.strategy_enemy_general_conv is None
        ):
            raise ValueError("strategy auxiliary heads are not configured")
        x = self._features(obs)
        pooled = self._masked_pool(x, active)
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        move_q = self.strategy_q_conv(x)
        pass_q = self.strategy_q_pass_linear(pooled)
        action_q_values = jnp.concatenate([move_q.reshape(-1), pass_q], axis=0)
        enemy_general_logits = self.strategy_enemy_general_conv(x)[0]
        enemy_general_logits = jnp.where(active, enemy_general_logits, -10.0)
        source_logits = None
        target_logits = None
        if self.strategy_source_conv is not None and self.strategy_target_conv is not None:
            source_logits = self.strategy_source_conv(x)[0]
            target_logits = self.strategy_target_conv(x)[0]
            source_logits = jnp.where(active, source_logits, -10.0)
            target_logits = jnp.where(active, target_logits, -10.0)
        return StrategyAuxOutputs(
            intent_logits=self.strategy_intent_linear2(value_hidden),
            finish_logits=self.strategy_finish_linear2(value_hidden),
            enemy_general_logits=enemy_general_logits,
            action_q_values=action_q_values,
            source_logits=source_logits,
            target_logits=target_logits,
        )

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

    def _size_value_logits(self, pooled: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        if self.categorical_value_linear2 is None:
            raise ValueError("categorical value head is not configured")
        if not self.value_head_sizes:
            return self._shared_value_logits(pooled)
        active_cells = jnp.sum(active.astype(jnp.int32))
        head_cells = jnp.array([size * size for size in self.value_head_sizes], dtype=jnp.int32)
        head_index = jnp.argmin(jnp.abs(head_cells - active_cells))
        branches = tuple(
            (lambda linear1, linear2: lambda _: linear2(jax.nn.relu(linear1(pooled))))(linear1, linear2)
            for linear1, linear2 in zip(
                self.size_value_linear1,
                self.size_categorical_value_linear2,
                strict=True,
            )
        )
        return jax.lax.switch(head_index, branches, None)

    def _value_distribution(self, pooled: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        if self.value_bins <= 0:
            return self._size_value(pooled, active), None
        value_logits = self._size_value_logits(pooled, active)
        value = categorical_value_expectation(value_logits, self.value_min, self.value_max)
        return value, value_logits

    def logits_value(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute movement logits, one global pass logit, and scalar value."""
        logits, value, _ = self.logits_value_distribution(obs, mask, active)
        return logits, value

    def logits_value_distribution(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        active: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Compute policy logits, scalar value, and optional categorical value logits."""
        logits, value, value_logits, _ = self.logits_value_auxiliary(obs, mask, active)
        return logits, value, value_logits

    def logits_value_auxiliary(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        active: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Compute policy/value outputs plus optional auxiliary outcome logits."""
        x = self._features(obs)
        pooled = self._masked_pool(x, active)
        move_logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        move_mask = jnp.concatenate([mask_t, mask_t], axis=0)
        move_logits = move_logits + (1 - move_mask.astype(jnp.float32)) * -1e9
        pass_logit = self.pass_linear(pooled)
        value, value_logits = self._value_distribution(pooled, active)
        outcome_logits = self._outcome_logits(pooled)
        logits = jnp.concatenate([move_logits.reshape(-1), pass_logit], axis=0)
        return logits, value, value_logits, outcome_logits

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


def _create_adaptive_network(
    key: jnp.ndarray,
    network_arch: str,
    pad_size: int,
    channels: PolicyChannels,
    input_channels: int,
    value_head_sizes: tuple[int, ...],
    value_bins: int,
    value_min: float,
    value_max: float,
    value_sigma: float,
    outcome_head: bool,
    strategy_aux: bool,
    strategy_spatial_aux: bool,
    global_context: bool,
    context_residual: bool,
    pyramid_context: bool,
):
    if network_arch == "cnn":
        return AdaptivePolicyValueNetwork(
            key,
            pad_size=pad_size,
            channels=channels,
            input_channels=input_channels,
            value_head_sizes=value_head_sizes,
            value_bins=value_bins,
            value_min=value_min,
            value_max=value_max,
            value_sigma=value_sigma,
            outcome_head=outcome_head,
            strategy_aux=strategy_aux,
            strategy_spatial_aux=strategy_spatial_aux,
            global_context=global_context,
            context_residual=context_residual,
            pyramid_context=pyramid_context,
        )
    if network_arch == "unet":
        return AdaptiveUNetPolicyValueNetwork(
            key,
            pad_size=pad_size,
            channels=channels,
            input_channels=input_channels,
            value_head_sizes=value_head_sizes,
            value_bins=value_bins,
            value_min=value_min,
            value_max=value_max,
            value_sigma=value_sigma,
            outcome_head=outcome_head,
            strategy_aux=strategy_aux,
            strategy_spatial_aux=strategy_spatial_aux,
            global_context=global_context,
            context_residual=context_residual,
            pyramid_context=pyramid_context,
        )
    raise ValueError(f"unknown adaptive network architecture: {network_arch}")


def adaptive_network_input_channels(network) -> int:
    """Return the spatial input channel count for an adaptive policy network."""
    if hasattr(network, "conv1"):
        return int(network.conv1.weight.shape[1])
    if hasattr(network, "enc1"):
        return int(network.enc1.weight.shape[1])
    raise TypeError(f"unsupported adaptive network type: {type(network)!r}")


def load_or_create_adaptive_network(
    key: jnp.ndarray,
    pad_size: int,
    init_model_path: str | Path | None = None,
    channels: str | PolicyChannels | list[int] | None = None,
    init_channels: str | PolicyChannels | list[int] | None = None,
    input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    init_input_channels: int | None = None,
    value_head_sizes: tuple[int, ...] | None = None,
    init_value_head_sizes: tuple[int, ...] | None = None,
    value_bins: int = 0,
    init_value_bins: int | None = None,
    value_min: float = -1.0,
    value_max: float = 1.0,
    value_sigma: float = 0.04,
    init_value_min: float | None = None,
    init_value_max: float | None = None,
    init_value_sigma: float | None = None,
    outcome_head: bool = False,
    init_outcome_head: bool | None = None,
    strategy_aux: bool = False,
    init_strategy_aux: bool | None = None,
    strategy_spatial_aux: bool = False,
    init_strategy_spatial_aux: bool | None = None,
    global_context: bool = False,
    init_global_context: bool | None = None,
    context_residual: bool = False,
    init_context_residual: bool | None = None,
    pyramid_context: bool = False,
    init_pyramid_context: bool | None = None,
    network_arch: str = "cnn",
    init_network_arch: str | None = None,
):
    """Create an adaptive network and optionally restore it from an Equinox checkpoint."""
    if network_arch not in ("cnn", "unet"):
        raise ValueError("--network-arch must be 'cnn' or 'unet'")
    parsed_init_network_arch = network_arch if init_network_arch is None else init_network_arch
    if parsed_init_network_arch not in ("cnn", "unet"):
        raise ValueError("--init-network-arch must be 'cnn' or 'unet'")
    parsed_channels = parse_policy_channels(channels)
    parsed_value_head_sizes = _normalize_value_head_sizes(value_head_sizes)
    parsed_value_bins = _normalize_value_bins(value_bins, value_min, value_max, value_sigma)
    network = _create_adaptive_network(
        key,
        network_arch,
        pad_size=pad_size,
        channels=parsed_channels,
        input_channels=input_channels,
        value_head_sizes=parsed_value_head_sizes,
        value_bins=parsed_value_bins,
        value_min=value_min,
        value_max=value_max,
        value_sigma=value_sigma,
        outcome_head=outcome_head,
        strategy_aux=strategy_aux,
        strategy_spatial_aux=strategy_spatial_aux,
        global_context=global_context,
        context_residual=context_residual,
        pyramid_context=pyramid_context,
    )
    if init_model_path is None:
        return network
    path = Path(init_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    parsed_init_channels = parse_policy_channels(init_channels) if init_channels is not None else parsed_channels
    parsed_init_input_channels = input_channels if init_input_channels is None else int(init_input_channels)
    parsed_init_value_head_sizes = (
        parsed_value_head_sizes
        if init_value_head_sizes is None
        else _normalize_value_head_sizes(init_value_head_sizes)
    )
    parsed_init_value_bins = parsed_value_bins if init_value_bins is None else int(init_value_bins)
    parsed_init_value_min = value_min if init_value_min is None else init_value_min
    parsed_init_value_max = value_max if init_value_max is None else init_value_max
    parsed_init_value_sigma = value_sigma if init_value_sigma is None else init_value_sigma
    parsed_init_value_bins = _normalize_value_bins(
        parsed_init_value_bins,
        parsed_init_value_min,
        parsed_init_value_max,
        parsed_init_value_sigma,
    )
    parsed_init_outcome_head = outcome_head if init_outcome_head is None else bool(init_outcome_head)
    parsed_init_strategy_aux = strategy_aux if init_strategy_aux is None else bool(init_strategy_aux)
    parsed_init_strategy_spatial_aux = (
        strategy_spatial_aux if init_strategy_spatial_aux is None else bool(init_strategy_spatial_aux)
    )
    parsed_init_global_context = global_context if init_global_context is None else bool(init_global_context)
    parsed_init_context_residual = (
        context_residual if init_context_residual is None else bool(init_context_residual)
    )
    parsed_init_pyramid_context = pyramid_context if init_pyramid_context is None else bool(init_pyramid_context)
    if parsed_init_network_arch != network_arch:
        raise ValueError(
            "Cannot warm-start across adaptive network architectures. "
            "Use --teacher-model-path/--teacher-kl-weight to anchor a new trunk to an old checkpoint."
        )
    needs_expansion = (
        parsed_init_channels != parsed_channels
        or parsed_init_input_channels != input_channels
        or parsed_init_value_head_sizes != parsed_value_head_sizes
        or parsed_init_value_bins != parsed_value_bins
        or parsed_init_value_min != value_min
        or parsed_init_value_max != value_max
        or parsed_init_value_sigma != value_sigma
        or parsed_init_outcome_head != outcome_head
        or parsed_init_strategy_aux != strategy_aux
        or parsed_init_strategy_spatial_aux != strategy_spatial_aux
        or parsed_init_global_context != global_context
        or parsed_init_context_residual != context_residual
        or parsed_init_pyramid_context != pyramid_context
    )
    if needs_expansion:
        source_network = _create_adaptive_network(
            key,
            parsed_init_network_arch,
            pad_size=pad_size,
            channels=parsed_init_channels,
            input_channels=parsed_init_input_channels,
            value_head_sizes=parsed_init_value_head_sizes,
            value_bins=parsed_init_value_bins,
            value_min=parsed_init_value_min,
            value_max=parsed_init_value_max,
            value_sigma=parsed_init_value_sigma,
            outcome_head=parsed_init_outcome_head,
            strategy_aux=parsed_init_strategy_aux,
            strategy_spatial_aux=parsed_init_strategy_spatial_aux,
            global_context=parsed_init_global_context,
            context_residual=parsed_init_context_residual,
            pyramid_context=parsed_init_pyramid_context,
        )
        source_network = eqx.tree_deserialise_leaves(path, source_network)
        if network_arch == "cnn":
            return expand_adaptive_network_channels(network, source_network)
        if network_arch == "unet":
            return expand_adaptive_unet_network(network, source_network)
        raise ValueError(f"unknown adaptive network architecture: {network_arch}")
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


def _normalize_value_bins(value_bins: int, value_min: float, value_max: float, value_sigma: float) -> int:
    """Validate optional categorical value support settings."""
    bins = int(value_bins)
    if bins < 0:
        raise ValueError("value_bins cannot be negative")
    if bins == 1:
        raise ValueError("value_bins must be 0 or greater than 1")
    if bins > 0:
        if value_min >= value_max:
            raise ValueError("value_min must be less than value_max")
        if value_sigma <= 0.0:
            raise ValueError("value_sigma must be positive")
    return bins


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
    if target.categorical_value_linear2 is not None and source.categorical_value_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.categorical_value_linear2,
            target,
            _copy_linear_prefix(target.categorical_value_linear2, source.categorical_value_linear2),
        )
    if target.value_head_sizes and target.categorical_value_linear2 is not None:
        source_categorical_heads = {}
        if source.categorical_value_linear2 is not None:
            source_categorical_heads = {
                size: linear2
                for size, linear2 in zip(
                    source.value_head_sizes,
                    source.size_categorical_value_linear2,
                    strict=True,
                )
            }
        target_size_categorical_linear2 = []
        for size, target_linear2 in zip(
            target.value_head_sizes,
            target.size_categorical_value_linear2,
            strict=True,
        ):
            source_linear2 = source_categorical_heads.get(size, source.categorical_value_linear2)
            if source_linear2 is None:
                target_size_categorical_linear2.append(target_linear2)
            else:
                target_size_categorical_linear2.append(_copy_linear_prefix(target_linear2, source_linear2))
        target = eqx.tree_at(
            lambda net: net.size_categorical_value_linear2,
            target,
            tuple(target_size_categorical_linear2),
        )
    if target.outcome_linear2 is not None and source.outcome_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.outcome_linear2,
            target,
            _copy_linear_prefix(target.outcome_linear2, source.outcome_linear2),
        )
    if target.strategy_intent_linear2 is not None and source.strategy_intent_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_intent_linear2,
            target,
            _copy_linear_prefix(target.strategy_intent_linear2, source.strategy_intent_linear2),
        )
    if target.strategy_finish_linear2 is not None and source.strategy_finish_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_finish_linear2,
            target,
            _copy_linear_prefix(target.strategy_finish_linear2, source.strategy_finish_linear2),
        )
    if target.strategy_q_conv is not None and source.strategy_q_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_q_conv,
            target,
            _copy_conv_prefix(target.strategy_q_conv, source.strategy_q_conv),
        )
    if target.strategy_q_pass_linear is not None and source.strategy_q_pass_linear is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_q_pass_linear,
            target,
            _copy_pooled_linear_prefix(
                target.strategy_q_pass_linear,
                source.strategy_q_pass_linear,
                target_channels,
                source_channels,
            ),
        )
    if target.strategy_enemy_general_conv is not None and source.strategy_enemy_general_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_enemy_general_conv,
            target,
            _copy_conv_prefix(target.strategy_enemy_general_conv, source.strategy_enemy_general_conv),
        )
    if target.strategy_source_conv is not None and source.strategy_source_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_source_conv,
            target,
            _copy_conv_prefix(target.strategy_source_conv, source.strategy_source_conv),
        )
    if target.strategy_target_conv is not None and source.strategy_target_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_target_conv,
            target,
            _copy_conv_prefix(target.strategy_target_conv, source.strategy_target_conv),
        )
    if target.context_conv1 is not None and source.context_conv1 is not None:
        target = eqx.tree_at(
            lambda net: net.context_conv1,
            target,
            _copy_conv_prefix(target.context_conv1, source.context_conv1),
        )
    if target.context_conv2 is not None and source.context_conv2 is not None:
        target = eqx.tree_at(
            lambda net: net.context_conv2,
            target,
            _copy_conv_prefix(target.context_conv2, source.context_conv2),
        )
    if target.pyramid_down1 is not None and source.pyramid_down1 is not None:
        target = eqx.tree_at(
            lambda net: net.pyramid_down1,
            target,
            _copy_conv_prefix(target.pyramid_down1, source.pyramid_down1),
        )
    if target.pyramid_down2 is not None and source.pyramid_down2 is not None:
        target = eqx.tree_at(
            lambda net: net.pyramid_down2,
            target,
            _copy_conv_prefix(target.pyramid_down2, source.pyramid_down2),
        )
    if target.pyramid_up1 is not None and source.pyramid_up1 is not None:
        target = eqx.tree_at(
            lambda net: net.pyramid_up1,
            target,
            _copy_conv_prefix(target.pyramid_up1, source.pyramid_up1),
        )
    if target.pyramid_up2 is not None and source.pyramid_up2 is not None:
        target = eqx.tree_at(
            lambda net: net.pyramid_up2,
            target,
            _copy_conv_prefix(target.pyramid_up2, source.pyramid_up2),
        )
    if (
        target.global_linear1 is not None
        and target.global_linear2 is not None
        and source.global_linear1 is not None
        and source.global_linear2 is not None
    ):
        target = eqx.tree_at(
            lambda net: net.global_linear1,
            target,
            _copy_linear_prefix(target.global_linear1, source.global_linear1),
        )
        target = eqx.tree_at(
            lambda net: net.global_linear2,
            target,
            _copy_linear_prefix(target.global_linear2, source.global_linear2),
        )
    return target


def expand_adaptive_unet_network(
    target: AdaptiveUNetPolicyValueNetwork,
    source: AdaptiveUNetPolicyValueNetwork,
) -> AdaptiveUNetPolicyValueNetwork:
    """Initialize a U-Net adaptive network from a structurally smaller U-Net checkpoint."""
    target_channels = target.dec2.weight.shape[0]
    source_channels = source.dec2.weight.shape[0]
    target = eqx.tree_at(lambda net: net.enc1, target, _copy_conv_prefix(target.enc1, source.enc1))
    target = eqx.tree_at(lambda net: net.enc2, target, _copy_conv_prefix(target.enc2, source.enc2))
    target = eqx.tree_at(lambda net: net.bottleneck, target, _copy_conv_prefix(target.bottleneck, source.bottleneck))
    target = eqx.tree_at(lambda net: net.dec1, target, _copy_conv_prefix(target.dec1, source.dec1))
    target = eqx.tree_at(lambda net: net.dec2, target, _copy_conv_prefix(target.dec2, source.dec2))
    target = eqx.tree_at(lambda net: net.policy_conv, target, _copy_conv_prefix(target.policy_conv, source.policy_conv))
    target = eqx.tree_at(
        lambda net: net.pass_linear,
        target,
        _copy_pooled_linear_prefix(target.pass_linear, source.pass_linear, target_channels, source_channels),
    )
    target = eqx.tree_at(
        lambda net: net.value_linear1,
        target,
        _copy_pooled_linear_prefix(target.value_linear1, source.value_linear1, target_channels, source_channels),
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
    if target.categorical_value_linear2 is not None and source.categorical_value_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.categorical_value_linear2,
            target,
            _copy_linear_prefix(target.categorical_value_linear2, source.categorical_value_linear2),
        )
    if target.value_head_sizes and target.categorical_value_linear2 is not None:
        source_categorical_heads = {}
        if source.categorical_value_linear2 is not None:
            source_categorical_heads = {
                size: linear2
                for size, linear2 in zip(
                    source.value_head_sizes,
                    source.size_categorical_value_linear2,
                    strict=True,
                )
            }
        target_size_categorical_linear2 = []
        for size, target_linear2 in zip(
            target.value_head_sizes,
            target.size_categorical_value_linear2,
            strict=True,
        ):
            source_linear2 = source_categorical_heads.get(size, source.categorical_value_linear2)
            if source_linear2 is None:
                target_size_categorical_linear2.append(target_linear2)
            else:
                target_size_categorical_linear2.append(_copy_linear_prefix(target_linear2, source_linear2))
        target = eqx.tree_at(
            lambda net: net.size_categorical_value_linear2,
            target,
            tuple(target_size_categorical_linear2),
        )
    if target.outcome_linear2 is not None and source.outcome_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.outcome_linear2,
            target,
            _copy_linear_prefix(target.outcome_linear2, source.outcome_linear2),
        )
    if target.strategy_intent_linear2 is not None and source.strategy_intent_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_intent_linear2,
            target,
            _copy_linear_prefix(target.strategy_intent_linear2, source.strategy_intent_linear2),
        )
    if target.strategy_finish_linear2 is not None and source.strategy_finish_linear2 is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_finish_linear2,
            target,
            _copy_linear_prefix(target.strategy_finish_linear2, source.strategy_finish_linear2),
        )
    if target.strategy_q_conv is not None and source.strategy_q_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_q_conv,
            target,
            _copy_conv_prefix(target.strategy_q_conv, source.strategy_q_conv),
        )
    if target.strategy_q_pass_linear is not None and source.strategy_q_pass_linear is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_q_pass_linear,
            target,
            _copy_pooled_linear_prefix(
                target.strategy_q_pass_linear,
                source.strategy_q_pass_linear,
                target_channels,
                source_channels,
            ),
        )
    if target.strategy_enemy_general_conv is not None and source.strategy_enemy_general_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_enemy_general_conv,
            target,
            _copy_conv_prefix(target.strategy_enemy_general_conv, source.strategy_enemy_general_conv),
        )
    if target.strategy_source_conv is not None and source.strategy_source_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_source_conv,
            target,
            _copy_conv_prefix(target.strategy_source_conv, source.strategy_source_conv),
        )
    if target.strategy_target_conv is not None and source.strategy_target_conv is not None:
        target = eqx.tree_at(
            lambda net: net.strategy_target_conv,
            target,
            _copy_conv_prefix(target.strategy_target_conv, source.strategy_target_conv),
        )
    return target
