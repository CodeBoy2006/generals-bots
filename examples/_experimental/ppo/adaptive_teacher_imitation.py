"""Teacher-imitation bootstrap for adaptive U-Net policy trunks."""

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

from adaptive_common import (
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_action_to_index,
    adaptive_input_channel_count,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    empty_adaptive_fog_memory,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
    reset_adaptive_fog_memory,
    reset_adaptive_scoreboard_history,
    update_adaptive_fog_memory,
)
from adaptive_network import adaptive_network_input_channels, load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, opponent_action, policy_network_action
from generals.agents.ppo_policy_agent import PolicyValueNetwork, obs_to_array, parse_policy_channels
from generals.core import game
from generals.core.action import compute_valid_move_mask
from train import checkpoint_path_for_iteration, prune_old_checkpoints, random_action, stack_learner_actions
from train_adaptive import split_mixed_env_counts, teacher_obs_from_student_obs

POLICY_MODE_NAMES = ("sample", "greedy")
POLICY_MODE_NAME_TO_ID = {name: index for index, name in enumerate(POLICY_MODE_NAMES)}


def empty_scoreboard_history(num_envs: int) -> jnp.ndarray:
    """Return empty previous-scoreboard features for vectorized imitation rollouts."""
    return jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)


def policy_action_from_logits(logits: jnp.ndarray, key: jnp.ndarray, policy_mode_id: int, pad_size: int) -> jnp.ndarray:
    """Sample or greedily select one adaptive action from logits."""
    index = jax.lax.cond(
        policy_mode_id == POLICY_MODE_NAME_TO_ID["greedy"],
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, pad_size)


def crop_observation(obs, size: int):
    """Crop padded adaptive observations before feeding a fixed-size policy."""
    return obs._replace(
        armies=obs.armies[:size, :size],
        generals=obs.generals[:size, :size],
        cities=obs.cities[:size, :size],
        mountains=obs.mountains[:size, :size],
        neutral_cells=obs.neutral_cells[:size, :size],
        owned_cells=obs.owned_cells[:size, :size],
        opponent_cells=obs.opponent_cells[:size, :size],
        fog_cells=obs.fog_cells[:size, :size],
        structures_in_fog=obs.structures_in_fog[:size, :size],
    )


def fixed_policy_logits_to_adaptive_logits(logits: jnp.ndarray, grid_size: int, pad_size: int) -> jnp.ndarray:
    """Place fixed-size policy logits onto the padded adaptive action lattice."""
    planes = logits.reshape(9, grid_size, grid_size)
    padded_moves = jnp.full((8, pad_size, pad_size), -1.0e9, dtype=logits.dtype)
    padded_moves = padded_moves.at[:, :grid_size, :grid_size].set(planes[:8])
    pass_logit = jax.nn.logsumexp(planes[8].reshape(-1))
    return jnp.concatenate([padded_moves.reshape(-1), pass_logit[None]], axis=0)


def fixed_policy_teacher_logits(network, obs, grid_size: int, pad_size: int) -> jnp.ndarray:
    """Return fixed-policy logits in the adaptive padded action space."""
    cropped = crop_observation(obs, grid_size)
    mask = compute_valid_move_mask(cropped.armies, cropped.owned_cells, cropped.mountains)
    logits, _ = network.logits_value(obs_to_array(cropped), mask)
    return fixed_policy_logits_to_adaptive_logits(logits, grid_size, pad_size)


@eqx.filter_jit
def collect_teacher_imitation_step(
    states,
    effective_sizes,
    pool,
    teacher_network,
    key,
    truncation,
    opponent_id,
    learner_player,
    teacher_policy_mode_id,
    pad_size,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    fixed_teacher_network=None,
    fixed_teacher_grid_size: int = 0,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
):
    """Collect one vectorized teacher-driven imitation step."""
    num_envs = states.armies.shape[0]
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p0_prior, lambda _: obs_p1_prior, None)
    opponent_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p1_prior, lambda _: obs_p0_prior, None)

    if scoreboard_history is None:
        scoreboard_history = empty_scoreboard_history(num_envs)
    if fog_memory is None:
        fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)
    if fog_memory_enabled:
        current_fog_memory = jax.vmap(update_adaptive_fog_memory)(fog_memory, learner_obs_prior)
    else:
        current_fog_memory = fog_memory

    if scoreboard_history_enabled:
        current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
            learner_obs_prior,
            effective_sizes,
        )
        history_context = adaptive_scoreboard_history_context(scoreboard_history, current_scoreboard)
        if fog_memory_enabled:
            obs_arr, active = jax.vmap(
                lambda obs, size, history, memory: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                    fog_memory=memory,
                )
            )(learner_obs_prior, effective_sizes, history_context, current_fog_memory)
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size, history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                )
            )(learner_obs_prior, effective_sizes, history_context)
    else:
        current_scoreboard = scoreboard_history
        if fog_memory_enabled:
            obs_arr, active = jax.vmap(
                lambda obs, size, memory: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=global_context,
                    fog_memory=memory,
                )
            )(learner_obs_prior, effective_sizes, current_fog_memory)
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
            )(learner_obs_prior, effective_sizes)

    masks = jax.vmap(
        lambda obs, size: compute_adaptive_valid_move_mask(
            obs.armies,
            obs.owned_cells,
            obs.mountains,
            size,
            pad_size,
        )
    )(learner_obs_prior, effective_sizes)

    if fixed_teacher_network is not None:
        teacher_logits = jax.vmap(
            lambda obs: fixed_policy_teacher_logits(fixed_teacher_network, obs, fixed_teacher_grid_size, pad_size)
        )(learner_obs_prior)
    else:
        teacher_obs_arr = jax.vmap(lambda obs: teacher_obs_from_student_obs(obs, teacher_input_channels))(obs_arr)
        teacher_logits = jax.vmap(lambda obs, mask, active: teacher_network.logits_value(obs, mask, active)[0])(
            teacher_obs_arr,
            masks,
            active,
        )
    key, teacher_key, opponent_key = jrandom.split(key, 3)
    teacher_keys = jrandom.split(teacher_key, num_envs)
    learner_actions = jax.vmap(lambda logits, sample_key: policy_action_from_logits(logits, sample_key, teacher_policy_mode_id, pad_size))(
        teacher_logits,
        teacher_keys,
    )
    teacher_indices = jax.vmap(lambda action: adaptive_action_to_index(action, pad_size))(learner_actions)

    opponent_keys = jrandom.split(opponent_key, num_envs)
    if opponent_policy_network is not None:
        opponent_actions = jax.vmap(
            lambda k, obs: policy_network_action(
                opponent_policy_network,
                k,
                crop_observation(obs, opponent_policy_grid_size),
                opponent_policy_mode,
            )
        )(opponent_keys, opponent_obs_prior)
    else:
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent_id, k, obs, random_action))(
            opponent_keys,
            opponent_obs_prior,
        )
    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)

    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated

    pool_size = pool.states.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda x: x[reset_indices], pool.states)
    reset_sizes = pool.effective_sizes[reset_indices]
    next_pool_idx = jnp.where(dones, new_states.pool_idx + num_envs, new_states.pool_idx)
    reset_states = reset_states._replace(pool_idx=next_pool_idx)
    current_states = new_states._replace(pool_idx=next_pool_idx)
    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        current_states,
    )
    final_sizes = jnp.where(dones, reset_sizes, effective_sizes)
    final_scoreboard_history = reset_adaptive_scoreboard_history(current_scoreboard, dones)
    final_fog_memory = reset_adaptive_fog_memory(current_fog_memory, dones)
    return (
        final_states,
        final_sizes,
        final_scoreboard_history,
        final_fog_memory,
        (obs_arr, masks, active, learner_actions, teacher_indices, teacher_logits, dones, infos),
        key,
    )


def collect_teacher_imitation_rollout(
    states,
    effective_sizes,
    pool,
    teacher_network,
    key,
    num_steps,
    truncation,
    opponent_id,
    learner_player,
    teacher_policy_mode_id,
    pad_size,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    fixed_teacher_network=None,
    fixed_teacher_grid_size: int = 0,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
):
    """Collect a teacher-driven imitation rollout using a Python loop."""
    step_data = []
    for _ in range(num_steps):
        states, effective_sizes, scoreboard_history, fog_memory, data, key = collect_teacher_imitation_step(
            states,
            effective_sizes,
            pool,
            teacher_network,
            key,
            truncation,
            opponent_id,
            learner_player,
            teacher_policy_mode_id,
            pad_size,
            global_context,
            scoreboard_history,
            scoreboard_history_enabled,
            fog_memory,
            fog_memory_enabled,
            teacher_input_channels,
            fixed_teacher_network,
            fixed_teacher_grid_size,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
        )
        step_data.append(data)
    return states, effective_sizes, scoreboard_history, fog_memory, jax.tree.map(lambda *xs: jnp.stack(xs), *step_data), key


def collect_mixed_teacher_imitation_rollout(
    states_p0,
    effective_sizes_p0,
    states_p1,
    effective_sizes_p1,
    pool,
    teacher_network,
    key,
    num_steps,
    truncation,
    opponent_id,
    teacher_policy_mode_id,
    pad_size,
    global_context=False,
    scoreboard_history_p0=None,
    scoreboard_history_p1=None,
    scoreboard_history_enabled=False,
    fog_memory_p0=None,
    fog_memory_p1=None,
    fog_memory_enabled=False,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    fixed_teacher_network=None,
    fixed_teacher_grid_size: int = 0,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
):
    """Collect and concatenate both learner seats for one imitation update."""
    key, p0_key, p1_key = jrandom.split(key, 3)
    states_p0, effective_sizes_p0, scoreboard_history_p0, fog_memory_p0, rollout_p0, _ = (
        collect_teacher_imitation_rollout(
            states_p0,
            effective_sizes_p0,
            pool,
            teacher_network,
            p0_key,
            num_steps,
            truncation,
            opponent_id,
            0,
            teacher_policy_mode_id,
            pad_size,
            global_context,
            scoreboard_history_p0,
            scoreboard_history_enabled,
            fog_memory_p0,
            fog_memory_enabled,
            teacher_input_channels,
            fixed_teacher_network,
            fixed_teacher_grid_size,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
        )
    )
    states_p1, effective_sizes_p1, scoreboard_history_p1, fog_memory_p1, rollout_p1, _ = (
        collect_teacher_imitation_rollout(
            states_p1,
            effective_sizes_p1,
            pool,
            teacher_network,
            p1_key,
            num_steps,
            truncation,
            opponent_id,
            1,
            teacher_policy_mode_id,
            pad_size,
            global_context,
            scoreboard_history_p1,
            scoreboard_history_enabled,
            fog_memory_p1,
            fog_memory_enabled,
            teacher_input_channels,
            fixed_teacher_network,
            fixed_teacher_grid_size,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
        )
    )
    rollout_data = jax.tree.map(lambda left, right: jnp.concatenate([left, right], axis=1), rollout_p0, rollout_p1)
    return (
        states_p0,
        effective_sizes_p0,
        scoreboard_history_p0,
        fog_memory_p0,
        states_p1,
        effective_sizes_p1,
        scoreboard_history_p1,
        fog_memory_p1,
        rollout_data,
        key,
    )


def flatten_imitation_batch(batch):
    """Flatten time/environment axes for imitation training."""
    obs, masks, active, actions, teacher_indices, teacher_logits, dones, infos = batch
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        actions.reshape(batch_size, -1),
        teacher_indices.reshape(batch_size),
        teacher_logits.reshape(batch_size, teacher_logits.shape[-1]),
    )


@eqx.filter_jit
def train_imitation_minibatch(
    network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight: float,
    action_ce_weight: float,
    entropy_weight: float,
    temperature: float,
):
    """Train one shuffled imitation minibatch."""
    obs, masks, active, actions, teacher_indices, teacher_logits = minibatch

    def loss_fn(net):
        student_logits = jax.vmap(lambda o, m, ac: net.logits_value(o, m, ac)[0])(obs, masks, active)
        teacher_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(teacher_logits / temperature, axis=-1))
        teacher_probs = jax.lax.stop_gradient(jax.nn.softmax(teacher_logits / temperature, axis=-1))
        student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
        kl_losses = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs_for_kl), axis=-1)

        student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
        action_indices = jax.vmap(lambda action: adaptive_action_to_index(action, net.pad_size))(actions)
        action_ce_losses = -student_log_probs[jnp.arange(student_log_probs.shape[0]), action_indices]
        probs = jax.nn.softmax(student_logits, axis=-1)
        entropy = -jnp.sum(probs * student_log_probs, axis=-1)
        loss = (
            kl_weight * jnp.mean(kl_losses) * (temperature**2)
            + action_ce_weight * jnp.mean(action_ce_losses)
            - entropy_weight * jnp.mean(entropy)
        )
        accuracy = jnp.mean(jnp.argmax(student_logits, axis=-1) == teacher_indices)
        return loss, (jnp.mean(kl_losses), jnp.mean(action_ce_losses), accuracy, jnp.mean(entropy))

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss, metrics


def train_imitation_epoch(
    network,
    opt_state,
    batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    action_ce_weight,
    entropy_weight,
    temperature,
):
    """Run multiple shuffled imitation epochs over one collected rollout."""
    flat_batch = flatten_imitation_batch(batch)
    batch_size = flat_batch[0].shape[0]
    actual_minibatch_size = batch_size if minibatch_size is None else min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None
    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = tuple(array[permutation] for array in flat_batch)
        epoch_loss = 0.0
        epoch_metrics = []
        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(array[start:end] for array in shuffled)
            network, opt_state, loss, metrics = train_imitation_minibatch(
                network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                action_ce_weight,
                entropy_weight,
                temperature,
            )
            epoch_loss += loss
            epoch_metrics.append(metrics)
        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda *xs: sum(xs) / len(xs), *epoch_metrics)
    return network, opt_state, avg_loss, avg_metrics, key


def parse_args():
    parser = argparse.ArgumentParser(description="Train an adaptive student by imitating a policy checkpoint teacher.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--truncation", type=int, default=750)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--teacher-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="unet")
    parser.add_argument("--init-network-arch", choices=("cnn", "unet"), default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--fog-memory", action="store_true")
    parser.add_argument("--init-global-context", action="store_true")
    parser.add_argument("--init-input-channels", type=int, default=None)
    parser.add_argument("--value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--init-value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--value-head-sizes", default=None)
    parser.add_argument("--init-value-head-sizes", default=None)
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--init-value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--init-value-bins", type=int, default=None)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--value-sigma", type=float, default=0.04)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--init-outcome-head", action="store_true")
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--teacher-network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--teacher-channels", default=None)
    parser.add_argument("--teacher-input-channels", type=int, default=None)
    parser.add_argument("--teacher-global-context", action="store_true")
    parser.add_argument("--teacher-scoreboard-history", action="store_true")
    parser.add_argument("--fixed-teacher-model-path", default=None)
    parser.add_argument("--fixed-teacher-channels", default=None)
    parser.add_argument("--fixed-teacher-input-channels", type=int, default=9)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--action-ce-weight", type=float, default=3.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/generals-adaptive-teacher-imitation.eqx")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--keep-checkpoints", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.grid_size_weights = parse_grid_size_weights(args.grid_size_weights, args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.value_head_sizes = (
            parse_grid_sizes(args.value_head_sizes) if args.value_head_sizes is not None else args.grid_sizes
        )
        args.init_value_head_sizes = (
            parse_grid_sizes(args.init_value_head_sizes) if args.init_value_head_sizes is not None else args.grid_sizes
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.fixed_teacher_channels = parse_policy_channels(args.fixed_teacher_channels)
        args.opponent_channels = parse_policy_channels(args.opponent_channels)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_envs < 2:
        parser.error("num_envs must be at least 2 for mixed-seat imitation")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_steps <= 0 or args.num_iterations <= 0 or args.num_epochs <= 0:
        parser.error("--num-steps, --num-iterations, and --num-epochs must be positive")
    if args.minibatch_size is not None and args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.teacher_input_channels is not None and args.teacher_input_channels <= 0:
        parser.error("--teacher-input-channels must be positive")
    if args.fixed_teacher_input_channels <= 0:
        parser.error("--fixed-teacher-input-channels must be positive")
    if args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    if (args.teacher_model_path is None) == (args.fixed_teacher_model_path is None):
        parser.error("provide exactly one of --teacher-model-path or --fixed-teacher-model-path")
    if args.fixed_teacher_model_path is not None and len(args.grid_sizes) != 1:
        parser.error("--fixed-teacher-model-path requires exactly one --grid-sizes value")
    if args.opponent_policy_path is not None and len(args.grid_sizes) != 1:
        parser.error("--opponent-policy-path requires exactly one --grid-sizes value")
    if args.kl_weight < 0.0 or args.action_ce_weight < 0.0 or args.entropy_weight < 0.0:
        parser.error("loss weights must be non-negative")
    if args.kl_weight == 0.0 and args.action_ce_weight == 0.0:
        parser.error("at least one of --kl-weight or --action-ce-weight must be positive")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.value_loss == "hl-gauss":
        if args.value_bins <= 1:
            parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
        if args.value_min >= args.value_max:
            parser.error("--value-min must be less than --value-max")
        if args.value_sigma <= 0.0:
            parser.error("--value-sigma must be positive")
    if args.init_value_loss == "hl-gauss":
        init_bins = args.value_bins if args.init_value_bins is None else args.init_value_bins
        if init_bins <= 1:
            parser.error("--init-value-bins must be greater than 1 for --init-value-loss hl-gauss")
    elif args.init_value_bins is not None:
        parser.error("--init-value-bins requires --init-value-loss hl-gauss")
    if args.checkpoint_every < 0 or args.keep_checkpoints < 0:
        parser.error("checkpoint settings cannot be negative")
    return args


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)
    key, student_key, teacher_key, pool_key = jrandom.split(key, 4)
    student_global_context = args.global_context or args.scoreboard_history
    teacher_global_context = args.teacher_global_context or args.teacher_scoreboard_history
    input_channels = adaptive_input_channel_count(student_global_context, args.scoreboard_history, args.fog_memory)
    teacher_input_channels = ADAPTIVE_INPUT_CHANNELS
    if args.teacher_model_path is not None:
        teacher_input_channels = (
            args.teacher_input_channels
            if args.teacher_input_channels is not None
            else adaptive_input_channel_count(teacher_global_context, args.teacher_scoreboard_history, False)
        )
    init_input_channels = args.init_input_channels
    if init_input_channels is None and args.init_model_path is not None and student_global_context and not args.init_global_context:
        init_input_channels = ADAPTIVE_INPUT_CHANNELS
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )

    print("Adaptive teacher imitation")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Environments:  {args.num_envs} mixed seats")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Student:       arch={args.network_arch} channels={args.channels or 'default'} inputs={input_channels}")
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
    if args.teacher_model_path is not None:
        print(f"Teacher:       {args.teacher_model_path} arch={args.teacher_network_arch} inputs={teacher_input_channels}")
    else:
        print(f"Teacher:       fixed policy {args.fixed_teacher_model_path}")
        print(f"Teacher arch:  PolicyValueNetwork channels={args.fixed_teacher_channels} inputs={args.fixed_teacher_input_channels}")
    if args.opponent_policy_path is None:
        print(f"Opponent:      {args.opponent}")
    else:
        print(f"Opponent:      fixed policy {args.opponent_policy_path}")
        print(f"Opponent mode: {args.opponent_policy_mode}")
    print(f"Teacher mode:  {args.teacher_policy_mode}")
    print(f"Iterations:    {args.num_iterations} x {args.num_steps} steps x {args.num_epochs} epochs")
    print(f"Minibatch:     {args.minibatch_size or args.num_envs * args.num_steps}")
    print(
        "Loss:          "
        f"kl={args.kl_weight:g}, ce={args.action_ce_weight:g}, "
        f"entropy={args.entropy_weight:g}, temp={args.temperature:g}"
    )
    if args.scoreboard_history:
        print("Score history: enabled")
    elif student_global_context:
        print("Global ctx:    enabled")
    if args.fog_memory:
        print("Fog memory:    enabled")
    if args.checkpoint_dir is not None and args.checkpoint_every > 0:
        print(f"Checkpoints:   every {args.checkpoint_every} iterations in {args.checkpoint_dir}")
    print()

    student_network = load_or_create_adaptive_network(
        student_key,
        pad_size=args.pad_to,
        init_model_path=args.init_model_path,
        channels=args.channels,
        init_channels=args.init_channels,
        input_channels=input_channels,
        init_input_channels=init_input_channels,
        value_head_sizes=args.value_head_sizes if args.value_heads == "per-size" else (),
        init_value_head_sizes=args.init_value_head_sizes if args.init_value_heads == "per-size" else (),
        value_bins=value_bins,
        init_value_bins=init_value_bins,
        value_min=args.value_min,
        value_max=args.value_max,
        value_sigma=args.value_sigma,
        outcome_head=args.outcome_head,
        init_outcome_head=args.init_outcome_head,
        global_context=student_global_context,
        init_global_context=args.init_global_context,
        network_arch=args.network_arch,
        init_network_arch=args.init_network_arch,
    )
    teacher_network = None
    fixed_teacher_network = None
    fixed_teacher_grid_size = args.grid_sizes[0]
    if args.teacher_model_path is not None:
        teacher_network = load_or_create_adaptive_network(
            teacher_key,
            pad_size=args.pad_to,
            init_model_path=args.teacher_model_path,
            channels=args.teacher_channels,
            input_channels=teacher_input_channels,
            init_input_channels=teacher_input_channels,
            global_context=teacher_global_context,
            init_global_context=teacher_global_context,
            network_arch=args.teacher_network_arch,
            init_network_arch=args.teacher_network_arch,
        )
        teacher_input_channels = adaptive_network_input_channels(teacher_network)
    else:
        fixed_teacher_network = PolicyValueNetwork(
            teacher_key,
            grid_size=fixed_teacher_grid_size,
            channels=args.fixed_teacher_channels,
            input_channels=args.fixed_teacher_input_channels,
        )
        fixed_teacher_network = eqx.tree_deserialise_leaves(args.fixed_teacher_model_path, fixed_teacher_network)
    opponent_policy_network = None
    opponent_policy_mode = 0 if args.opponent_policy_mode == "greedy" else 1
    opponent_policy_grid_size = args.grid_sizes[0]
    if args.opponent_policy_path is not None:
        opponent_policy_network = PolicyValueNetwork(
            teacher_key,
            grid_size=opponent_policy_grid_size,
            channels=args.opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_policy_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_policy_network)

    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(student_network, eqx.is_inexact_array))
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    teacher_policy_mode_id = POLICY_MODE_NAME_TO_ID[args.teacher_policy_mode]
    pool = make_adaptive_state_pool(
        pool_key,
        args.pool_size,
        args.grid_sizes,
        args.pad_to,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
        args.grid_size_weights,
    )
    jax.block_until_ready(pool.states.armies)

    p0_envs, p1_envs = split_mixed_env_counts(args.num_envs)
    states, effective_sizes = make_adaptive_initial_states(pool, args.num_envs)
    states_p0 = jax.tree.map(lambda value: value[:p0_envs], states)
    effective_sizes_p0 = effective_sizes[:p0_envs]
    states_p1 = jax.tree.map(lambda value: value[p0_envs:], states)
    effective_sizes_p1 = effective_sizes[p0_envs:]
    scoreboard_history_p0 = empty_scoreboard_history(p0_envs)
    scoreboard_history_p1 = empty_scoreboard_history(p1_envs)
    fog_memory_p0 = empty_adaptive_fog_memory(p0_envs, args.pad_to)
    fog_memory_p1 = empty_adaptive_fog_memory(p1_envs, args.pad_to)

    checkpoint_paths = []
    model_stem = Path(args.model_path).stem
    for iteration in range(1, args.num_iterations + 1):
        t0 = time.time()
        key, rollout_key, train_key = jrandom.split(key, 3)
        (
            states_p0,
            effective_sizes_p0,
            scoreboard_history_p0,
            fog_memory_p0,
            states_p1,
            effective_sizes_p1,
            scoreboard_history_p1,
            fog_memory_p1,
            batch,
            _,
        ) = collect_mixed_teacher_imitation_rollout(
            states_p0,
            effective_sizes_p0,
            states_p1,
            effective_sizes_p1,
            pool,
            teacher_network,
            rollout_key,
            args.num_steps,
            args.truncation,
            opponent_id,
            teacher_policy_mode_id,
            args.pad_to,
            student_global_context,
            scoreboard_history_p0,
            scoreboard_history_p1,
            args.scoreboard_history,
            fog_memory_p0,
            fog_memory_p1,
            args.fog_memory,
            teacher_input_channels,
            fixed_teacher_network,
            fixed_teacher_grid_size,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
        )
        jax.block_until_ready(states_p0)
        jax.block_until_ready(states_p1)
        student_network, opt_state, loss, metrics, key = train_imitation_epoch(
            student_network,
            opt_state,
            batch,
            optimizer,
            train_key,
            args.num_epochs,
            args.minibatch_size,
            args.kl_weight,
            args.action_ce_weight,
            args.entropy_weight,
            args.temperature,
        )
        jax.block_until_ready(student_network)

        if args.checkpoint_dir is not None and args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, model_stem, iteration)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(checkpoint_path, student_network)
            checkpoint_paths.append(checkpoint_path)
            prune_old_checkpoints(checkpoint_paths, args.keep_checkpoints)

        if iteration == 1 or iteration % 10 == 0 or iteration == args.num_iterations:
            _, _, _, _, teacher_indices, _, dones, infos = batch
            episodes = int(jnp.sum(dones))
            wins = int(jnp.sum(dones & (infos.winner >= 0)))
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            kl, ce, accuracy, entropy = metrics
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"KL: {float(kl):.4f} | CE: {float(ce):.4f} | "
                f"Acc: {float(accuracy) * 100:5.1f}% | Ent: {float(entropy):.2f} | "
                f"Episodes: {episodes:4d} | Decisive: {wins:4d} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s"
            )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, student_network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
