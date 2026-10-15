"""Weighted binary gate updates shared by command and adapter training."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax


@eqx.filter_jit
def train_step(gate, opt_state, batch, optimizer):
    features, labels, weights = batch

    def loss_fn(model):
        logits = jax.vmap(model)(features)
        losses = optax.sigmoid_binary_cross_entropy(logits, labels)
        normalizer = jnp.maximum(jnp.sum(weights), 1.0)
        loss = jnp.sum(losses * weights) / normalizer
        probs = jax.nn.sigmoid(logits)
        preds = probs >= 0.5
        accuracy = jnp.sum((preds == (labels >= 0.5)).astype(jnp.float32) * weights) / normalizer
        positive_prob = jnp.sum(probs * labels * weights) / jnp.maximum(jnp.sum(labels * weights), 1.0)
        negative_prob = jnp.sum(probs * (1.0 - labels) * weights) / jnp.maximum(
            jnp.sum((1.0 - labels) * weights),
            1.0,
        )
        return loss, {
            "accuracy": accuracy,
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "mean_prob": jnp.mean(probs),
        }

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(gate)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(gate, eqx.is_inexact_array))
    return eqx.apply_updates(gate, updates), opt_state, loss, metrics


def train_epoch(gate, opt_state, dataset, optimizer, key, minibatch_size: int):
    num_examples = dataset["features"].shape[0]
    permutation = jrandom.permutation(key, num_examples)
    num_batches = max(num_examples // minibatch_size, 1)
    metrics_sum = None
    loss_sum = 0.0
    for batch_index in range(num_batches):
        start = batch_index * minibatch_size
        end = min(start + minibatch_size, num_examples)
        idx = permutation[start:end]
        batch = (dataset["features"][idx], dataset["labels"][idx], dataset["weights"][idx])
        gate, opt_state, loss, metrics = train_step(gate, opt_state, batch, optimizer)
        loss_sum += loss
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
    return gate, opt_state, loss_sum / num_batches, jax.tree.map(lambda value: value / num_batches, metrics_sum)
