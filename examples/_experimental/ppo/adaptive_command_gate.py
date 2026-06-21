"""Lightweight command-acceptance gate for adaptive source-target plans."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

COMMAND_GATE_FEATURE_NAMES = (
    "policy_logit_delta",
    "action_q_delta",
    "source_logit",
    "target_logit",
    "finish_probability",
    "source_army_log1p",
    "route_distance_norm",
    "candidate_policy_logit",
    "current_policy_logit",
    "candidate_q",
    "current_q",
    "seat",
    "active_area_fraction",
)
COMMAND_GATE_FEATURE_DIM = len(COMMAND_GATE_FEATURE_NAMES)


class CommandGateNetwork(eqx.Module):
    """Small normalized MLP that predicts whether to accept a command action."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int = COMMAND_GATE_FEATURE_DIM,
        hidden_dim: int = 32,
        feature_mean: jnp.ndarray | None = None,
        feature_std: jnp.ndarray | None = None,
    ):
        key1, key2 = jrandom.split(key)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.linear1 = eqx.nn.Linear(self.input_dim, self.hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(self.hidden_dim, 1, key=key2)
        self.feature_mean = (
            jnp.zeros((self.input_dim,), dtype=jnp.float32)
            if feature_mean is None
            else jnp.asarray(feature_mean, dtype=jnp.float32)
        )
        self.feature_std = (
            jnp.ones((self.input_dim,), dtype=jnp.float32)
            if feature_std is None
            else jnp.asarray(feature_std, dtype=jnp.float32)
        )

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        x = (features - self.feature_mean) / jnp.maximum(self.feature_std, 1.0e-6)
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)[0]
