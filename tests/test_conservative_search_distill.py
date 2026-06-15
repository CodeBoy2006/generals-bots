import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.conservative_search_distill import (
    compute_conservative_loss,
    select_search_improvements,
)
from generals.agents.ppo_policy_agent import PolicyValueNetwork


def test_select_search_improvements_requires_switch_and_margin():
    candidate_indices = jnp.array(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=jnp.int32,
    )
    search_scores = jnp.array(
        [
            [5.0, 8.5, 7.0],
            [9.0, 8.0, 7.0],
            [1.0, 2.2, 2.0],
        ],
        dtype=jnp.float32,
    )

    targets, weights, margins = select_search_improvements(
        candidate_indices,
        search_scores,
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
    )

    assert jnp.array_equal(targets, jnp.array([11, 20, 31], dtype=jnp.int32))
    assert jnp.allclose(margins, jnp.array([3.5, 0.0, 1.2], dtype=jnp.float32))
    assert jnp.allclose(weights, jnp.array([0.375, 0.0, 0.0], dtype=jnp.float32))


def test_compute_conservative_loss_is_zero_for_matching_networks_without_improvements():
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    obs = jnp.zeros((2, 9, 4, 4), dtype=jnp.float32)
    masks = jnp.ones((2, 4, 4, 4), dtype=bool)
    targets = jnp.array([0, 1], dtype=jnp.int32)
    improve_weights = jnp.zeros((2,), dtype=jnp.float32)
    kl_weights = jnp.ones((2,), dtype=jnp.float32)

    loss, metrics = compute_conservative_loss(
        network,
        network,
        obs,
        masks,
        targets,
        improve_weights,
        kl_weights,
        kl_weight=1.0,
        improve_weight=0.1,
        temperature=1.0,
    )

    assert abs(float(loss)) < 1e-6
    assert abs(float(metrics["kl_loss"])) < 1e-6
    assert float(metrics["improve_loss"]) == 0.0
