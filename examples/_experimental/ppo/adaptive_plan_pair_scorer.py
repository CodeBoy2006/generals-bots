"""Explicit source-target plan pair scorer used for Plan-Q ranking probes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

PLAN_PAIR_BASE_FEATURE_NAMES = (
    "policy_logit_delta",
    "action_q_delta",
    "source_logit",
    "target_logit",
    "finish_probability",
    "source_army_log1p",
    "target_army_log1p",
    "route_distance_norm",
    "candidate_policy_logit",
    "current_policy_logit",
    "candidate_q",
    "current_q",
    "source_row_norm",
    "source_col_norm",
    "target_row_norm",
    "target_col_norm",
    "row_delta_norm",
    "col_delta_norm",
    "grid_size_norm",
    "turn_norm",
    "seat",
)


def plan_pair_feature_names(input_channels: int) -> tuple[str, ...]:
    """Return feature names for one scorer input vector."""
    source_names = tuple(f"source_obs_ch{idx}" for idx in range(input_channels))
    target_names = tuple(f"target_obs_ch{idx}" for idx in range(input_channels))
    return PLAN_PAIR_BASE_FEATURE_NAMES + source_names + target_names


class PlanPairScorerNetwork(eqx.Module):
    """Small normalized MLP that scores one source-target plan candidate."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int,
        hidden_dim: int = 128,
        feature_mean: jnp.ndarray | None = None,
        feature_std: jnp.ndarray | None = None,
    ):
        key1, key2, key3 = jrandom.split(key, 3)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.linear1 = eqx.nn.Linear(self.input_dim, self.hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=key2)
        self.linear3 = eqx.nn.Linear(self.hidden_dim, 1, key=key3)
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
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)[0]
