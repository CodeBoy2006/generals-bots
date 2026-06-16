"""Residual recurrent policy-value network for experimental PPO."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import DEFAULT_POLICY_CHANNELS, PolicyChannels, PolicyValueNetwork, index_to_action


class RecurrentPolicyValueNetwork(eqx.Module):
    """Policy network with a GRU residual adapter on top of a CNN base."""

    base: PolicyValueNetwork
    memory_encoder: eqx.nn.Linear
    gru: eqx.nn.GRUCell
    policy_delta: eqx.nn.Linear
    value_delta: eqx.nn.Linear
    grid_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        grid_size: int = 4,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = 9,
        hidden_size: int = 64,
        base_network: PolicyValueNetwork | None = None,
    ):
        keys = jrandom.split(key, 5)
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.base = (
            PolicyValueNetwork(keys[0], grid_size=grid_size, channels=channels, input_channels=input_channels)
            if base_network is None
            else base_network
        )
        self.memory_encoder = eqx.nn.Linear(input_channels * grid_size * grid_size, hidden_size, key=keys[1])
        self.gru = eqx.nn.GRUCell(hidden_size, hidden_size, key=keys[2])
        policy_delta = eqx.nn.Linear(hidden_size, 9 * grid_size * grid_size, key=keys[3])
        value_delta = eqx.nn.Linear(hidden_size, 1, key=keys[4])
        self.policy_delta = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            policy_delta,
            (jnp.zeros_like(policy_delta.weight), jnp.zeros_like(policy_delta.bias)),
        )
        self.value_delta = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            value_delta,
            (jnp.zeros_like(value_delta.weight), jnp.zeros_like(value_delta.bias)),
        )

    def initial_hidden(self) -> jnp.ndarray:
        """Return one zeroed recurrent hidden state."""
        return jnp.zeros((self.hidden_size,), dtype=jnp.float32)

    def logits_value_hidden(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        hidden: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute residual logits/value and the next hidden state for one step."""
        base_logits, base_value = self.base.logits_value(obs, mask)
        memory_input = jax.nn.relu(self.memory_encoder(obs.reshape(-1)))
        next_hidden = self.gru(memory_input, hidden)
        logits = base_logits + self.policy_delta(next_hidden)
        value = base_value + self.value_delta(next_hidden)[0]
        return logits, value, next_hidden

    def __call__(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        hidden: jnp.ndarray,
        key: jnp.ndarray | None,
        action: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample or evaluate one recurrent policy action."""
        grid_cells = self.grid_size * self.grid_size
        logits, value, next_hidden = self.logits_value_hidden(obs, mask, hidden)

        if action is None:
            if key is None:
                raise ValueError("key is required when sampling an action")
            index = jrandom.categorical(key, logits)
            action = index_to_action(index, self.grid_size)
        else:
            is_pass, row, col, direction, is_half = action
            encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
            index = encoded_dir * grid_cells + row * self.grid_size + col

        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[index]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)
        return action, value, logprob, entropy, next_hidden
