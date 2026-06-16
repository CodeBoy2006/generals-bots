"""Adaptive conservative rollout-search distillation for multisize checkpoints."""

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
from generals.core.game import GameInfo
from generals.core.observation import Observation

from adaptive_common import (
    ADAPTIVE_INPUT_CHANNELS,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import POLICY_MODE_NAME_TO_ID, POLICY_MODE_NAMES
from conservative_search_distill import (
    search_score_target_probs,
    select_search_improvements,
    weighted_topk_cross_entropy,
)
from train import checkpoint_path_for_iteration, prune_old_checkpoints, stack_learner_actions

TARGET_MODE_NAMES = ("hard", "soft")


def compute_adaptive_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    target_indices,
    improve_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return adaptive KL-to-base plus weighted hard search-target loss."""

    def logits_for_sample(network, sample_obs, sample_mask, sample_active):
        logits, _ = network.logits_value(sample_obs, sample_mask, sample_active)
        return logits

    student_logits = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: logits_for_sample(
            student_network,
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)
    base_logits = jax.lax.stop_gradient(
        jax.vmap(
            lambda sample_obs, sample_mask, sample_active: logits_for_sample(
                base_network,
                sample_obs,
                sample_mask,
                sample_active,
            )
        )(base_obs, base_masks, base_active)
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


def compute_adaptive_soft_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    candidate_indices,
    target_probs,
    search_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return adaptive KL-to-base plus weighted soft top-k search-target loss."""

    def logits_for_sample(network, sample_obs, sample_mask, sample_active):
        logits, _ = network.logits_value(sample_obs, sample_mask, sample_active)
        return logits

    student_logits = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: logits_for_sample(
            student_network,
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)
    base_logits = jax.lax.stop_gradient(
        jax.vmap(
            lambda sample_obs, sample_mask, sample_active: logits_for_sample(
                base_network,
                sample_obs,
                sample_mask,
                sample_active,
            )
        )(base_obs, base_masks, base_active)
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


def adaptive_score_observation(
    info: GameInfo,
    obs: Observation,
    player: int,
    army_weight: float = 12.0,
    land_weight: float = 8.0,
    terminal_score: float = 1000.0,
):
    """Score a final adaptive rollout observation from one player's perspective."""
    army_balance = (obs.owned_army_count.astype(jnp.float32) - obs.opponent_army_count.astype(jnp.float32)) / jnp.maximum(
        obs.owned_army_count + obs.opponent_army_count,
        1,
    )
    land_balance = (obs.owned_land_count.astype(jnp.float32) - obs.opponent_land_count.astype(jnp.float32)) / obs.armies.size
    terminal = jnp.where(
        info.winner == player,
        terminal_score,
        jnp.where(info.winner == 1 - player, -terminal_score, 0.0),
    )
    return terminal + army_weight * army_balance + land_weight * land_balance


def adaptive_policy_action(network, obs, effective_size, key, policy_mode, pad_size: int):
    """Dispatch an adaptive checkpoint action using greedy or sampled execution."""
    obs_arr, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    mask = compute_adaptive_valid_move_mask(
        obs.armies,
        obs.owned_cells,
        obs.mountains,
        effective_size,
        pad_size,
    )
    logits, _ = network.logits_value(obs_arr, mask, active)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, pad_size)


@eqx.filter_jit
def adaptive_rollout_search_candidates(
    network,
    state,
    effective_size,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    pad_size,
):
    """Return adaptive top-k prior candidates and short rollout-search scores."""
    obs = game.get_observation(state, player)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits, _ = network.logits_value(obs_arr, mask, active)
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda idx: adaptive_index_to_action(idx, pad_size))(candidate_indices)

    opponent_player = 1 - player
    opponent_obs = game.get_observation(state, opponent_player)
    key, opponent_key = jrandom.split(key)
    opponent_first_action = adaptive_policy_action(
        network,
        opponent_obs,
        effective_size,
        opponent_key,
        policy_mode,
        pad_size,
    )

    def rollout_score(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, step_key = carry
            step_key, k0, k1 = jrandom.split(step_key, 3)
            obs_p0 = game.get_observation(rollout_state, 0)
            obs_p1 = game.get_observation(rollout_state, 1)
            action_p0 = adaptive_policy_action(network, obs_p0, effective_size, k0, policy_mode, pad_size)
            action_p1 = adaptive_policy_action(network, obs_p1, effective_size, k1, policy_mode, pad_size)
            next_state, _ = game.step(rollout_state, jnp.stack([action_p0, action_p1]))
            already_done = game.get_info(rollout_state).is_done
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            return (final_state, step_key), None

        (final_state, _), _ = jax.lax.scan(body, (initial_state, rollout_key), None, length=rollout_steps)
        final_info = game.get_info(final_state)
        final_obs = game.get_observation(final_state, player)
        return adaptive_score_observation(final_info, final_obs, player, army_weight, land_weight, terminal_score)

    def candidate_score(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        scores = jax.vmap(lambda rollout_key: rollout_score(next_state, rollout_key))(rollout_keys)
        first_terminal = jnp.where(
            first_info.winner == player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        return first_terminal + jnp.mean(scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(candidate_score)(candidate_actions, prior_scores, candidate_keys)
    return candidate_actions, candidate_indices, prior_scores, scores


@eqx.filter_jit
def adaptive_rollout_search_action(
    network,
    state,
    effective_size,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    pad_size,
):
    """Choose one adaptive action by scoring top-k prior actions with short rollouts."""
    candidate_actions, _, _, scores = adaptive_rollout_search_candidates(
        network,
        state,
        effective_size,
        key,
        player,
        top_k,
        rollout_steps,
        rollouts_per_action,
        policy_mode,
        army_weight,
        land_weight,
        prior_weight,
        terminal_score,
        pad_size,
    )
    return candidate_actions[jnp.argmax(scores)]


@eqx.filter_jit
def train_adaptive_conservative_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
):
    """Train one adaptive hard-target distillation minibatch."""
    obs, masks, active, base_obs, base_masks, base_active, target_indices, improve_weights, kl_weights = minibatch

    def loss_fn(net):
        return compute_adaptive_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            active,
            base_obs,
            base_masks,
            base_active,
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
def train_adaptive_soft_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
):
    """Train one adaptive soft-target distillation minibatch."""
    obs, masks, active, base_obs, base_masks, base_active, candidate_indices, target_probs, search_weights, kl_weights = (
        minibatch
    )

    def loss_fn(net):
        return compute_adaptive_soft_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            active,
            base_obs,
            base_masks,
            base_active,
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


def flatten_adaptive_conservative_batch(
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    target_indices,
    improve_weights,
    kl_weights,
    margins,
):
    """Flatten time/environment axes for adaptive hard-target distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        base_active.reshape(batch_size, *base_active.shape[2:]),
        target_indices.reshape(batch_size),
        improve_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
        margins.reshape(batch_size),
    )


def flatten_adaptive_soft_batch(
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    candidate_indices,
    target_probs,
    search_weights,
    kl_weights,
):
    """Flatten time/environment axes for adaptive soft-target distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        base_active.reshape(batch_size, *base_active.shape[2:]),
        candidate_indices.reshape(batch_size, *candidate_indices.shape[2:]),
        target_probs.reshape(batch_size, *target_probs.shape[2:]),
        search_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
    )


def train_adaptive_conservative_epoch(
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
    """Run adaptive hard-target distillation over shuffled minibatches."""
    obs, masks, active, base_obs, base_masks, base_active, target_indices, improve_weights, kl_weights, margins = (
        flat_batch
    )
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
            active[permutation],
            base_obs[permutation],
            base_masks[permutation],
            base_active[permutation],
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
            student_network, opt_state, loss, metrics = train_adaptive_conservative_minibatch(
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


def train_adaptive_soft_epoch(
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
    """Run adaptive soft-target distillation over shuffled minibatches."""
    obs, masks, active, base_obs, base_masks, base_active, candidate_indices, target_probs, search_weights, kl_weights = (
        flat_batch
    )
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
            active[permutation],
            base_obs[permutation],
            base_masks[permutation],
            base_active[permutation],
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
            student_network, opt_state, loss, metrics = train_adaptive_soft_minibatch(
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
def collect_adaptive_conservative_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    effective_sizes,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    min_margin,
    margin_scale,
    max_weight,
    pad_size,
):
    """Collect adaptive learner states labeled by hard search improvements."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        is_active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        learner_obs_arr, active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
            learner_obs,
            effective_sizes,
        )
        masks = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(learner_obs, effective_sizes)
        base_obs_arr = learner_obs_arr
        base_masks = masks
        base_active = active

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores = jax.vmap(
            lambda state, size, sample_key: adaptive_rollout_search_candidates(
                base_network,
                state,
                size,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
                terminal_score,
                pad_size,
            )
        )(states, effective_sizes, search_keys)
        target_indices, improve_weights, margins = select_search_improvements(
            candidate_indices,
            search_scores,
            min_margin,
            margin_scale,
            max_weight,
        )
        active_weights = is_active.astype(jnp.float32)
        improve_weights = improve_weights * active_weights

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                student_network,
                obs,
                size,
                sample_key,
                policy_mode,
                pad_size,
            )
        )(learner_obs, effective_sizes, learner_keys)
        opponent_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                opponent_network,
                obs,
                size,
                sample_key,
                opponent_policy_mode,
                pad_size,
            )
        )(opponent_obs, effective_sizes, opponent_keys)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(is_active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        return (final_states, key), (
            learner_obs_arr,
            masks,
            active,
            base_obs_arr,
            base_masks,
            base_active,
            target_indices,
            improve_weights,
            active_weights,
            margins,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=num_steps)
    return states, batch, key


@eqx.filter_jit
def collect_adaptive_soft_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    effective_sizes,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    score_temperature,
    pad_size,
):
    """Collect adaptive learner states labeled by soft search-score targets."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        is_active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        learner_obs_arr, active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
            learner_obs,
            effective_sizes,
        )
        masks = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(learner_obs, effective_sizes)
        base_obs_arr = learner_obs_arr
        base_masks = masks
        base_active = active

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores = jax.vmap(
            lambda state, size, sample_key: adaptive_rollout_search_candidates(
                base_network,
                state,
                size,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
                terminal_score,
                pad_size,
            )
        )(states, effective_sizes, search_keys)
        target_probs = search_score_target_probs(search_scores, score_temperature)
        active_weights = is_active.astype(jnp.float32)

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                student_network,
                obs,
                size,
                sample_key,
                policy_mode,
                pad_size,
            )
        )(learner_obs, effective_sizes, learner_keys)
        opponent_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                opponent_network,
                obs,
                size,
                sample_key,
                opponent_policy_mode,
                pad_size,
            )
        )(opponent_obs, effective_sizes, opponent_keys)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(is_active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        return (final_states, key), (
            learner_obs_arr,
            masks,
            active,
            base_obs_arr,
            base_masks,
            base_active,
            candidate_indices,
            target_probs,
            active_weights,
            active_weights,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=num_steps)
    return states, batch, key
