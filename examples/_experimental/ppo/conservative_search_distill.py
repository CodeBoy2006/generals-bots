"""Conservative rollout-search distillation for PPO checkpoints.

This trainer treats rollout search as a noisy policy-improvement oracle. It
keeps the student close to a fixed base checkpoint with KL regularization and
only applies action supervision when search clearly improves on the base
checkpoint's top-prior action.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from generals.agents.ppo_policy_agent import parse_policy_channels
from generals.core import game
from generals.core.action import compute_valid_move_mask

from common import (
    POLICY_INPUT_NAME_TO_ID,
    POLICY_INPUT_NAMES,
    POLICY_MODE_NAME_TO_ID,
    POLICY_MODE_NAMES,
    make_grids,
    policy_input_array_and_mask,
    policy_input_default_channels,
    policy_network_action,
    policy_state_action,
)
from network import obs_to_array
from search_policy import rollout_search_candidates
from train import load_or_create_network, stack_learner_actions

TARGET_MODE_NAMES = ("hard", "soft")


def select_search_improvements(
    candidate_indices,
    search_scores,
    min_margin: float,
    margin_scale: float,
    max_weight: float,
):
    """Select high-confidence search improvements over the base top-prior action."""
    best_positions = jnp.argmax(search_scores, axis=-1)
    target_indices = jnp.take_along_axis(candidate_indices, best_positions[..., None], axis=-1)[..., 0]
    best_scores = jnp.take_along_axis(search_scores, best_positions[..., None], axis=-1)[..., 0]
    base_indices = candidate_indices[..., 0]
    base_scores = search_scores[..., 0]
    margins = best_scores - base_scores

    switched = target_indices != base_indices
    scaled_weights = (margins - min_margin) / margin_scale
    clipped_weights = jnp.clip(scaled_weights, 0.0, max_weight)
    weights = jnp.where(switched & (margins >= min_margin), clipped_weights, 0.0)
    return target_indices.astype(jnp.int32), weights.astype(jnp.float32), margins.astype(jnp.float32)


def search_score_target_probs(search_scores, temperature: float):
    """Convert top-k rollout-search scores into a stable soft target distribution."""
    centered_scores = search_scores - jnp.max(search_scores, axis=-1, keepdims=True)
    return jax.nn.softmax(centered_scores / temperature, axis=-1)


def weighted_topk_cross_entropy(log_probs, candidate_indices, target_probs, weights):
    """Return weighted CE over sparse top-k candidate targets."""
    candidate_log_probs = jnp.take_along_axis(log_probs, candidate_indices, axis=1)
    losses = -jnp.sum(target_probs * candidate_log_probs, axis=-1)
    normalizer = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(losses * weights) / normalizer


def compute_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    base_obs,
    base_masks,
    target_indices,
    improve_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return KL-to-base plus weighted high-confidence improvement loss."""

    def logits_for_sample(network, sample_obs, sample_mask):
        logits, _ = network.logits_value(sample_obs, sample_mask)
        return logits

    student_logits = jax.vmap(lambda sample_obs, sample_mask: logits_for_sample(student_network, sample_obs, sample_mask))(
        obs,
        masks,
    )
    base_logits = jax.lax.stop_gradient(
        jax.vmap(lambda sample_obs, sample_mask: logits_for_sample(base_network, sample_obs, sample_mask))(
            base_obs,
            base_masks,
        )
    )

    student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    base_log_probs = jax.nn.log_softmax(base_logits / temperature, axis=-1)
    base_probs = jax.nn.softmax(base_logits / temperature, axis=-1)
    kl_per_sample = jnp.sum(base_probs * (base_log_probs - student_log_probs_for_kl), axis=-1)
    kl_normalizer = jnp.maximum(jnp.sum(kl_weights), 1.0)
    kl_loss = jnp.sum(kl_per_sample * kl_weights) / kl_normalizer

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    action_losses = -jnp.take_along_axis(student_log_probs, target_indices[:, None], axis=1)[:, 0]
    improve_normalizer = jnp.maximum(jnp.sum(improve_weights), 1.0)
    improve_loss = jnp.sum(action_losses * improve_weights) / improve_normalizer

    loss = kl_weight * kl_loss + improve_weight * improve_loss
    selected = improve_weights > 0.0
    selected_count = jnp.sum(selected.astype(jnp.float32))
    active_count = jnp.maximum(jnp.sum(kl_weights), 1.0)
    accuracy = jnp.sum((jnp.argmax(student_logits, axis=-1) == target_indices) * improve_weights) / improve_normalizer
    metrics = {
        "kl_loss": kl_loss,
        "improve_loss": jnp.where(selected_count > 0.0, improve_loss, 0.0),
        "selected_fraction": selected_count / active_count,
        "accuracy": jnp.where(selected_count > 0.0, accuracy, 0.0),
    }
    return loss, metrics


def compute_soft_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    base_obs,
    base_masks,
    candidate_indices,
    target_probs,
    search_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return KL-to-base plus weighted soft top-k search-score loss."""

    def logits_for_sample(network, sample_obs, sample_mask):
        logits, _ = network.logits_value(sample_obs, sample_mask)
        return logits

    student_logits = jax.vmap(lambda sample_obs, sample_mask: logits_for_sample(student_network, sample_obs, sample_mask))(
        obs,
        masks,
    )
    base_logits = jax.lax.stop_gradient(
        jax.vmap(lambda sample_obs, sample_mask: logits_for_sample(base_network, sample_obs, sample_mask))(
            base_obs,
            base_masks,
        )
    )

    student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    base_log_probs = jax.nn.log_softmax(base_logits / temperature, axis=-1)
    base_probs = jax.nn.softmax(base_logits / temperature, axis=-1)
    kl_per_sample = jnp.sum(base_probs * (base_log_probs - student_log_probs_for_kl), axis=-1)
    kl_normalizer = jnp.maximum(jnp.sum(kl_weights), 1.0)
    kl_loss = jnp.sum(kl_per_sample * kl_weights) / kl_normalizer

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    search_loss = weighted_topk_cross_entropy(student_log_probs, candidate_indices, target_probs, search_weights)
    loss = kl_weight * kl_loss + improve_weight * search_loss

    best_targets = jnp.take_along_axis(candidate_indices, jnp.argmax(target_probs, axis=-1)[:, None], axis=1)[:, 0]
    search_normalizer = jnp.maximum(jnp.sum(search_weights), 1.0)
    accuracy = jnp.sum((jnp.argmax(student_logits, axis=-1) == best_targets) * search_weights) / search_normalizer
    target_entropy = -jnp.sum(target_probs * jnp.log(jnp.clip(target_probs, 1e-8, 1.0)), axis=-1)
    metrics = {
        "kl_loss": kl_loss,
        "improve_loss": search_loss,
        "selected_fraction": jnp.sum(search_weights) / kl_normalizer,
        "accuracy": accuracy,
        "target_entropy": jnp.sum(target_entropy * search_weights) / search_normalizer,
    }
    return loss, metrics


@eqx.filter_jit
def train_conservative_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
):
    """Train one conservative search-distillation minibatch."""
    obs, masks, base_obs, base_masks, target_indices, improve_weights, kl_weights = minibatch

    def loss_fn(net):
        return compute_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            base_obs,
            base_masks,
            target_indices,
            improve_weights,
            kl_weights,
            kl_weight,
            improve_weight,
            temperature,
        )

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(student_network)
    params = eqx.filter(student_network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    student_network = eqx.apply_updates(student_network, updates)
    return student_network, opt_state, loss, metrics


@eqx.filter_jit
def train_soft_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
):
    """Train one soft search-score distillation minibatch."""
    obs, masks, base_obs, base_masks, candidate_indices, target_probs, search_weights, kl_weights = minibatch

    def loss_fn(net):
        return compute_soft_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            base_obs,
            base_masks,
            candidate_indices,
            target_probs,
            search_weights,
            kl_weights,
            kl_weight,
            improve_weight,
            temperature,
        )

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(student_network)
    params = eqx.filter(student_network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    student_network = eqx.apply_updates(student_network, updates)
    return student_network, opt_state, loss, metrics


def flatten_conservative_batch(obs, masks, base_obs, base_masks, target_indices, improve_weights, kl_weights, margins):
    """Flatten time/environment axes for conservative distillation updates."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        target_indices.reshape(batch_size),
        improve_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
        margins.reshape(batch_size),
    )


def flatten_soft_batch(obs, masks, base_obs, base_masks, candidate_indices, target_probs, search_weights, kl_weights):
    """Flatten time/environment axes for soft search-score distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        candidate_indices.reshape(batch_size, *candidate_indices.shape[2:]),
        target_probs.reshape(batch_size, *target_probs.shape[2:]),
        search_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
    )


def train_conservative_epoch(
    student_network,
    base_network,
    opt_state,
    flat_batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    improve_weight,
    temperature,
):
    """Run conservative distillation over shuffled minibatches."""
    obs, masks, base_obs, base_masks, target_indices, improve_weights, kl_weights, margins = flat_batch
    batch_size = obs.shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = (
            obs[permutation],
            masks[permutation],
            base_obs[permutation],
            base_masks[permutation],
            target_indices[permutation],
            improve_weights[permutation],
            kl_weights[permutation],
        )
        epoch_loss = 0.0
        epoch_metrics = None

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            student_network, opt_state, loss, metrics = train_conservative_minibatch(
                student_network,
                base_network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                improve_weight,
                temperature,
            )
            epoch_loss += loss
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, metrics)

        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda x: x / num_complete_batches, epoch_metrics)

    selected_margins = jnp.where(improve_weights > 0.0, margins, 0.0)
    selected_count = jnp.maximum(jnp.sum((improve_weights > 0.0).astype(jnp.float32)), 1.0)
    avg_metrics = dict(avg_metrics)
    avg_metrics["mean_selected_margin"] = jnp.sum(selected_margins) / selected_count
    avg_metrics["selected_samples"] = jnp.sum((improve_weights > 0.0).astype(jnp.float32))
    return student_network, opt_state, avg_loss, avg_metrics, key


def train_soft_epoch(
    student_network,
    base_network,
    opt_state,
    flat_batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    improve_weight,
    temperature,
):
    """Run soft search-score distillation over shuffled minibatches."""
    obs, masks, base_obs, base_masks, candidate_indices, target_probs, search_weights, kl_weights = flat_batch
    batch_size = obs.shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = (
            obs[permutation],
            masks[permutation],
            base_obs[permutation],
            base_masks[permutation],
            candidate_indices[permutation],
            target_probs[permutation],
            search_weights[permutation],
            kl_weights[permutation],
        )
        epoch_loss = 0.0
        epoch_metrics = None

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            student_network, opt_state, loss, metrics = train_soft_minibatch(
                student_network,
                base_network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                improve_weight,
                temperature,
            )
            epoch_loss += loss
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, metrics)

        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda x: x / num_complete_batches, epoch_metrics)

    avg_metrics = dict(avg_metrics)
    avg_metrics["selected_samples"] = jnp.sum((search_weights > 0.0).astype(jnp.float32))
    avg_metrics["mean_selected_margin"] = 0.0
    return student_network, opt_state, avg_loss, avg_metrics, key


@eqx.filter_jit
def collect_conservative_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    policy_input,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    min_margin,
    margin_scale,
    max_weight,
):
    """Collect student-distribution states labeled by conservative search improvements."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        learner_obs_arr, masks = jax.vmap(
            lambda state, obs: policy_input_array_and_mask(state, obs, learner_player, policy_input)
        )(states, learner_obs)
        base_obs_arr = jax.vmap(obs_to_array)(learner_obs)
        base_masks = jax.vmap(lambda obs: compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains))(
            learner_obs
        )

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores = jax.vmap(
            lambda state, sample_key: rollout_search_candidates(
                base_network,
                state,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
            )
        )(states, search_keys)
        target_indices, improve_weights, margins = select_search_improvements(
            candidate_indices,
            search_scores,
            min_margin,
            margin_scale,
            max_weight,
        )
        active_weights = active.astype(jnp.float32)
        improve_weights = improve_weights * active_weights

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(
            lambda state, sample_key, obs: policy_state_action(
                student_network,
                sample_key,
                state,
                obs,
                learner_player,
                policy_mode,
                policy_input,
            )
        )(
            states,
            learner_keys,
            learner_obs,
        )
        opponent_actions = jax.vmap(
            lambda sample_key, obs: policy_network_action(opponent_network, sample_key, obs, opponent_policy_mode)
        )(opponent_keys, opponent_obs)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        return (final_states, key), (
            learner_obs_arr,
            masks,
            base_obs_arr,
            base_masks,
            target_indices,
            improve_weights,
            active_weights,
            margins,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=num_steps)
    return states, batch, key


@eqx.filter_jit
def collect_soft_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    policy_input,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    score_temperature,
):
    """Collect student-distribution states labeled by soft search-score targets."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        learner_obs_arr, masks = jax.vmap(
            lambda state, obs: policy_input_array_and_mask(state, obs, learner_player, policy_input)
        )(states, learner_obs)
        base_obs_arr = jax.vmap(obs_to_array)(learner_obs)
        base_masks = jax.vmap(lambda obs: compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains))(
            learner_obs
        )

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores = jax.vmap(
            lambda state, sample_key: rollout_search_candidates(
                base_network,
                state,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
            )
        )(states, search_keys)
        target_probs = search_score_target_probs(search_scores, score_temperature)
        active_weights = active.astype(jnp.float32)

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(
            lambda state, sample_key, obs: policy_state_action(
                student_network,
                sample_key,
                state,
                obs,
                learner_player,
                policy_mode,
                policy_input,
            )
        )(
            states,
            learner_keys,
            learner_obs,
        )
        opponent_actions = jax.vmap(
            lambda sample_key, obs: policy_network_action(opponent_network, sample_key, obs, opponent_policy_mode)
        )(opponent_keys, opponent_obs)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        return (final_states, key), (
            learner_obs_arr,
            masks,
            base_obs_arr,
            base_masks,
            candidate_indices,
            target_probs,
            active_weights,
            active_weights,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=num_steps)
    return states, batch, key


def parse_args():
    parser = argparse.ArgumentParser(description="Conservatively distill rollout-search improvements into a checkpoint.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--base-model-path", required=True, help="Fixed base checkpoint used for search labels and KL.")
    parser.add_argument("--init-model-path", default=None, help="Student warm-start checkpoint. Defaults to base model.")
    parser.add_argument("--opponent-policy-path", default=None, help="Frozen opponent checkpoint. Defaults to base model.")
    parser.add_argument("--model-path", default="/tmp/generals-conservative-search-distill.eqx")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--learner-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--policy-input", choices=POLICY_INPUT_NAMES, default="observation")
    parser.add_argument("--target-mode", choices=TARGET_MODE_NAMES, default="hard")
    parser.add_argument("--channels", default=None, help="Student channels, for example 64,64,64,32.")
    parser.add_argument("--base-channels", default=None, help="Base/search checkpoint channels.")
    parser.add_argument("--opponent-channels", default=None, help="Opponent checkpoint channels. Defaults to base channels.")
    parser.add_argument("--input-channels", type=int, default=None, help="Student network input channels.")
    parser.add_argument(
        "--init-input-channels",
        type=int,
        default=None,
        help="Input channels of the warm-start checkpoint before optional expansion.",
    )
    parser.add_argument("--base-input-channels", type=int, default=9)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--improve-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=25.0)
    parser.add_argument("--margin-scale", type=float, default=100.0)
    parser.add_argument("--max-improve-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--rollouts-per-action", type=int, default=4)
    parser.add_argument("--army-weight", type=float, default=12.0)
    parser.add_argument("--land-weight", type=float, default=8.0)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.num_envs <= 0:
        parser.error("num_envs must be positive")
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if args.num_iterations <= 0:
        parser.error("--num-iterations must be positive")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.base_input_channels <= 0 or args.opponent_input_channels <= 0:
        parser.error("--base-input-channels and --opponent-input-channels must be positive")
    if args.kl_weight < 0.0 or args.improve_weight < 0.0:
        parser.error("--kl-weight and --improve-weight must be non-negative")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.score_temperature <= 0.0:
        parser.error("--score-temperature must be positive")
    if args.margin_scale <= 0.0:
        parser.error("--margin-scale must be positive")
    if args.max_improve_weight <= 0.0:
        parser.error("--max-improve-weight must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.rollouts_per_action <= 0:
        parser.error("--rollouts-per-action must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    try:
        args.channels = parse_policy_channels(args.channels)
        args.base_channels = parse_policy_channels(args.base_channels)
        args.opponent_channels = parse_policy_channels(args.opponent_channels or args.base_channels)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)

    init_model_path = args.init_model_path or args.base_model_path
    opponent_policy_path = args.opponent_policy_path or args.base_model_path
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    input_channels = args.input_channels or policy_input_default_channels(args.policy_input)
    init_input_channels = args.init_input_channels
    if init_input_channels is None and init_model_path == args.base_model_path and input_channels != args.base_input_channels:
        init_input_channels = args.base_input_channels

    print("Conservative rollout-search distillation")
    print(f"Device:        {jax.devices()[0]}")
    print(
        f"Student:       {init_model_path} channels={args.channels} "
        f"input={args.policy_input} input_channels={input_channels}"
    )
    print(f"Base/Search:   {args.base_model_path} channels={args.base_channels} input_channels={args.base_input_channels}")
    print(
        f"Opponent:      {opponent_policy_path} channels={args.opponent_channels} "
        f"input_channels={args.opponent_input_channels} ({args.opponent_policy_mode})"
    )
    print(f"Grid:          {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Rollout:       {args.num_iterations} x {args.num_steps} steps, envs={args.num_envs}")
    print(
        "Search:        "
        f"top_k={args.top_k}, rollout_steps={args.rollout_steps}, rollouts/action={args.rollouts_per_action}"
    )
    print(
        "Objective:     "
        f"kl={args.kl_weight:g}, improve={args.improve_weight:g}, "
        f"mode={args.target_mode}, min_margin={args.min_margin:g}, "
        f"margin_scale={args.margin_scale:g}, score_temp={args.score_temperature:g}"
    )
    print()

    key = jrandom.PRNGKey(args.seed)
    key, student_key, base_key, opponent_key = jrandom.split(key, 4)
    student_network = load_or_create_network(
        student_key,
        args.grid_size,
        init_model_path=init_model_path,
        channels=args.channels,
        input_channels=input_channels,
        init_input_channels=init_input_channels,
    )
    base_network = load_or_create_network(
        base_key,
        args.grid_size,
        init_model_path=args.base_model_path,
        channels=args.base_channels,
        input_channels=args.base_input_channels,
    )
    opponent_network = load_or_create_network(
        opponent_key,
        args.grid_size,
        init_model_path=opponent_policy_path,
        channels=args.opponent_channels,
        input_channels=args.opponent_input_channels,
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(student_network, eqx.is_inexact_array))

    for iteration in range(args.num_iterations):
        t0 = time.time()
        key, map_key, rollout_key, update_key = jrandom.split(key, 4)
        grids = make_grids(
            map_key,
            args.num_envs,
            args.grid_size,
            args.map_generator,
            (args.mountain_density_min, args.mountain_density_max),
            (args.num_cities_min, args.num_cities_max),
            min_generals_distance,
            args.max_generals_distance,
            (args.city_army_min, args.city_army_max),
        )
        states = jax.vmap(game.create_initial_state)(grids)
        if args.target_mode == "hard":
            _, batch, rollout_key = collect_conservative_batch(
                student_network,
                base_network,
                opponent_network,
                states,
                rollout_key,
                args.num_steps,
                policy_mode,
                opponent_policy_mode,
                args.learner_player,
                policy_input,
                args.top_k,
                args.rollout_steps,
                args.rollouts_per_action,
                args.army_weight,
                args.land_weight,
                args.prior_weight,
                args.min_margin,
                args.margin_scale,
                args.max_improve_weight,
            )
            flat_batch = flatten_conservative_batch(*batch)
            student_network, opt_state, loss, metrics, update_key = train_conservative_epoch(
                student_network,
                base_network,
                opt_state,
                flat_batch,
                optimizer,
                update_key,
                args.num_epochs,
                args.minibatch_size,
                args.kl_weight,
                args.improve_weight,
                args.temperature,
            )
        else:
            _, batch, rollout_key = collect_soft_batch(
                student_network,
                base_network,
                opponent_network,
                states,
                rollout_key,
                args.num_steps,
                policy_mode,
                opponent_policy_mode,
                args.learner_player,
                policy_input,
                args.top_k,
                args.rollout_steps,
                args.rollouts_per_action,
                args.army_weight,
                args.land_weight,
                args.prior_weight,
                args.score_temperature,
            )
            flat_batch = flatten_soft_batch(*batch)
            student_network, opt_state, loss, metrics, update_key = train_soft_epoch(
                student_network,
                base_network,
                opt_state,
                flat_batch,
                optimizer,
                update_key,
                args.num_epochs,
                args.minibatch_size,
                args.kl_weight,
                args.improve_weight,
                args.temperature,
            )
        jax.block_until_ready(student_network)

        if iteration % 5 == 0 or iteration == args.num_iterations - 1:
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.5f} | "
                f"KL: {float(metrics['kl_loss']):.5f} | "
                f"Improve: {float(metrics['improve_loss']):.4f} | "
                f"Selected: {int(metrics['selected_samples']):5d}/{samples} "
                f"({float(metrics['selected_fraction']) * 100:4.1f}%) | "
                f"Margin: {float(metrics['mean_selected_margin']):6.1f} | "
                f"SPS: {samples / elapsed:7.0f} | Time: {elapsed:.2f}s"
            )

    eqx.tree_serialise_leaves(args.model_path, student_network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
