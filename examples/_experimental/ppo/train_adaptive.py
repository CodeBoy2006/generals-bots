"""Raw-game PPO trainer for adaptive multisize policy checkpoints."""

from __future__ import annotations

import argparse
import math
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
    ADAPTIVE_GLOBAL_INPUT_CHANNELS,
    ADAPTIVE_HISTORY_INPUT_CHANNELS,
    ADAPTIVE_FOG_MEMORY_CHANNELS,
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_action_to_index,
    adaptive_input_channel_count,
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
from adaptive_network import adaptive_network_input_channels, hl_gauss_value_loss, load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, POLICY_MODE_NAMES, opponent_action, policy_network_action
from generals.core import game
from generals.core.rewards import composite_reward_fn
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from train import (
    apply_terminal_reward,
    checkpoint_path_for_iteration,
    compute_gae,
    prune_old_checkpoints,
    random_action,
    stack_learner_actions,
)

REWARD_MODE_NAMES = ("composite", "terminal")
REWARD_MODE_NAME_TO_ID = {name: idx for idx, name in enumerate(REWARD_MODE_NAMES)}
OUTCOME_LOSS = 0
OUTCOME_DRAW = 1
OUTCOME_WIN = 2


def apply_truncation_reward(rewards, truncated, scale):
    """Penalize non-terminal timeout rows without changing decisive games."""
    return rewards - jnp.where(truncated, scale, 0.0)


def apply_reward_mode(composite_rewards, reward_mode_id):
    """Keep dense rewards or drop them for sparse terminal-only training."""
    return jnp.where(
        reward_mode_id == REWARD_MODE_NAME_TO_ID["terminal"],
        jnp.zeros_like(composite_rewards),
        composite_rewards,
    )


def top_advantage_weights(advantages: jnp.ndarray, fraction: float) -> jnp.ndarray:
    """Select the highest-advantage samples for policy-gradient updates."""
    if fraction >= 1.0:
        return jnp.ones_like(advantages, dtype=jnp.float32)
    flat = advantages.reshape(-1)
    count = max(1, math.ceil(flat.shape[0] * fraction))
    _, indices = jax.lax.top_k(flat, count)
    weights = jnp.zeros_like(flat, dtype=jnp.float32).at[indices].set(1.0)
    return weights.reshape(advantages.shape)


def masked_top_advantage_weights(advantages: jnp.ndarray, mask: jnp.ndarray, fraction: float) -> jnp.ndarray:
    """Select top-advantage samples inside one static task slice."""
    flat_mask = mask.reshape(-1)
    count_available = int(jnp.sum(flat_mask))
    if count_available <= 0:
        return jnp.zeros_like(advantages, dtype=jnp.float32)
    count = max(1, math.ceil(count_available * fraction))
    flat_advantages = advantages.reshape(-1)
    candidates = jnp.where(flat_mask, flat_advantages, -jnp.inf)
    _, indices = jax.lax.top_k(candidates, count)
    weights = jnp.zeros_like(flat_advantages, dtype=jnp.float32).at[indices].set(1.0)
    return weights.reshape(advantages.shape)


def stratified_top_advantage_weights(
    advantages: jnp.ndarray,
    active: jnp.ndarray,
    learner_players: jnp.ndarray,
    grid_sizes: tuple[int, ...],
    fraction: float,
) -> jnp.ndarray:
    """Apply top-advantage filtering independently per size and learner seat."""
    if fraction >= 1.0:
        return jnp.ones_like(advantages, dtype=jnp.float32)
    active_cells = jnp.sum(active.astype(jnp.int32), axis=(-2, -1))
    players = learner_players[None, :]
    weights = jnp.zeros_like(advantages, dtype=jnp.float32)
    for size in grid_sizes:
        size_mask = active_cells == size * size
        for player in (0, 1):
            task_mask = size_mask & (players == player)
            weights = weights + masked_top_advantage_weights(advantages, task_mask, fraction)
    return jnp.minimum(weights, 1.0)


def outcome_class_from_winner(winner: jnp.ndarray, learner_player: jnp.ndarray) -> jnp.ndarray:
    """Map a winner id to loss/draw/win from each learner perspective."""
    return jnp.where(
        winner < 0,
        OUTCOME_DRAW,
        jnp.where(winner == learner_player, OUTCOME_WIN, OUTCOME_LOSS),
    ).astype(jnp.int32)


def rollout_outcome_targets(
    winners: jnp.ndarray,
    dones: jnp.ndarray,
    learner_players: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Assign each rollout sample the next known episode outcome within the rollout."""
    learner_players = jnp.broadcast_to(learner_players, winners.shape[1:])
    default_targets = jnp.full(winners.shape[1:], OUTCOME_LOSS, dtype=jnp.int32)
    default_weights = jnp.zeros(winners.shape[1:], dtype=jnp.float32)

    def scan_step(carry, inputs):
        next_targets, next_weights = carry
        done, winner = inputs
        current_targets = outcome_class_from_winner(winner, learner_players)
        targets = jnp.where(done, current_targets, next_targets)
        weights = jnp.where(done, 1.0, next_weights)
        return (targets, weights), (targets, weights)

    _, (targets_rev, weights_rev) = jax.lax.scan(
        scan_step,
        (default_targets, default_weights),
        (dones[::-1], winners[::-1]),
    )
    return targets_rev[::-1], weights_rev[::-1]


def update_ema_network(ema_network, network, decay: float):
    """Return an exponential moving average over trainable floating-point leaves."""

    def update_leaf(ema_leaf, current_leaf):
        if eqx.is_inexact_array(ema_leaf) and eqx.is_inexact_array(current_leaf):
            return decay * ema_leaf + (1.0 - decay) * current_leaf
        return current_leaf

    return jax.tree.map(update_leaf, ema_network, network)


def zero_inexact_tree(tree):
    """Return a tree with trainable floating-point leaves zeroed."""
    return jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else x, tree)


def context_only_grad_tree(grads):
    """Keep gradients only for the optional residual context branch."""
    zeroed = zero_inexact_tree(grads)
    zeroed = eqx.tree_at(
        lambda net: (net.context_conv1, net.context_conv2),
        zeroed,
        (grads.context_conv1, grads.context_conv2),
        is_leaf=lambda value: value is None,
    )
    return eqx.tree_at(
        lambda net: (net.pyramid_down1, net.pyramid_down2, net.pyramid_up1, net.pyramid_up2),
        zeroed,
        (grads.pyramid_down1, grads.pyramid_down2, grads.pyramid_up1, grads.pyramid_up2),
        is_leaf=lambda value: value is None,
    )


def resolve_learner_player(value: str, iteration: int) -> int:
    """Resolve fixed or alternating learner seat for one training iteration."""
    if value == "alternate":
        return (iteration - 1) % 2
    if value == "mixed":
        raise ValueError("mixed learner mode is resolved by collect_mixed_rollout")
    return int(value)


def split_mixed_env_counts(num_envs: int) -> tuple[int, int]:
    """Split total vectorized environments across learner seats for mixed PPO."""
    if num_envs < 2:
        raise ValueError("mixed learner mode requires at least two environments")
    p0_envs = num_envs // 2
    return p0_envs, num_envs - p0_envs


def empty_scoreboard_history(num_envs: int) -> jnp.ndarray:
    """Return empty previous-scoreboard features for vectorized rollouts."""
    return jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)


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


def policy_sized_observation(obs, effective_size: jnp.ndarray, policy_grid_size: int):
    """Return a legal fixed-size observation only for rows matching the policy size.

    JAX evaluates both branches of a later action mix, so non-matching 12/16 rows
    still need a benign 8x8 observation before querying an 8x8 frozen opponent.
    """
    cropped = crop_observation(obs, policy_grid_size)
    shape = cropped.armies.shape
    valid = effective_size == policy_grid_size

    dummy_armies = jnp.zeros(shape, dtype=cropped.armies.dtype).at[0, 0].set(2)
    dummy_bool = jnp.zeros(shape, dtype=bool)
    dummy_owned = dummy_bool.at[0, 0].set(True)
    dummy_generals = dummy_bool.at[0, 0].set(True)
    dummy_neutral = ~dummy_owned

    return cropped._replace(
        armies=jnp.where(valid, cropped.armies, dummy_armies),
        generals=jnp.where(valid, cropped.generals, dummy_generals),
        cities=jnp.where(valid, cropped.cities, dummy_bool),
        mountains=jnp.where(valid, cropped.mountains, dummy_bool),
        neutral_cells=jnp.where(valid, cropped.neutral_cells, dummy_neutral),
        owned_cells=jnp.where(valid, cropped.owned_cells, dummy_owned),
        opponent_cells=jnp.where(valid, cropped.opponent_cells, dummy_bool),
        fog_cells=jnp.where(valid, cropped.fog_cells, dummy_bool),
        structures_in_fog=jnp.where(valid, cropped.structures_in_fog, dummy_bool),
        owned_land_count=jnp.where(valid, cropped.owned_land_count, jnp.asarray(1, cropped.owned_land_count.dtype)),
        owned_army_count=jnp.where(valid, cropped.owned_army_count, jnp.asarray(2, cropped.owned_army_count.dtype)),
        opponent_land_count=jnp.where(
            valid,
            cropped.opponent_land_count,
            jnp.asarray(0, cropped.opponent_land_count.dtype),
        ),
        opponent_army_count=jnp.where(
            valid,
            cropped.opponent_army_count,
            jnp.asarray(0, cropped.opponent_army_count.dtype),
        ),
    )


@eqx.filter_jit
def rollout_step(
    states,
    effective_sizes,
    pool,
    network,
    key,
    truncation,
    opponent_id,
    learner_player,
    reward_mode_id,
    terminal_reward_scale,
    truncation_reward_scale,
    pad_size,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    teacher_network=None,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    teacher_rollout_actions: bool = False,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
    opponent_policy_mix_prob: float = 1.0,
):
    """Collect one vectorized adaptive PPO rollout step."""
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
            )(
                learner_obs_prior,
                effective_sizes,
                history_context,
                current_fog_memory,
            )
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size, history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=history,
                )
            )(
                learner_obs_prior,
                effective_sizes,
                history_context,
            )
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
            )(
                learner_obs_prior,
                effective_sizes,
                current_fog_memory,
            )
        else:
            obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
            )(
                learner_obs_prior,
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
    )(learner_obs_prior, effective_sizes)

    key, learner_key = jrandom.split(key)
    learner_keys = jrandom.split(learner_key, num_envs)
    learner_actions, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, 0, None))(
        obs_arr,
        masks,
        active,
        learner_keys,
        None,
    )
    if teacher_network is not None and teacher_rollout_actions:
        key, teacher_key = jrandom.split(key)
        teacher_obs_arr = jax.vmap(lambda obs: teacher_obs_from_student_obs(obs, teacher_input_channels))(obs_arr)
        teacher_keys = jrandom.split(teacher_key, num_envs)
        teacher_actions, _, _, _ = jax.vmap(teacher_network, in_axes=(0, 0, 0, 0, None))(
            teacher_obs_arr,
            masks,
            active,
            teacher_keys,
            None,
        )
        learner_actions, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, 0, 0))(
            obs_arr,
            masks,
            active,
            learner_keys,
            teacher_actions,
        )

    key, opponent_key = jrandom.split(key)
    opponent_keys = jrandom.split(opponent_key, num_envs)
    heuristic_actions = jax.vmap(lambda k, obs: opponent_action(opponent_id, k, obs, random_action))(
        opponent_keys,
        opponent_obs_prior,
    )
    if opponent_policy_network is not None:
        policy_actions = jax.vmap(
            lambda k, obs, size: policy_network_action(
                opponent_policy_network,
                k,
                policy_sized_observation(obs, size, opponent_policy_grid_size),
                opponent_policy_mode,
            )
        )(opponent_keys, opponent_obs_prior, effective_sizes)
        key, opponent_mix_key = jrandom.split(key)
        use_policy = (effective_sizes == opponent_policy_grid_size) & (
            jrandom.uniform(opponent_mix_key, (num_envs,)) < opponent_policy_mix_prob
        )
        opponent_actions = jnp.where(use_policy[:, None], policy_actions, heuristic_actions)
    else:
        opponent_actions = heuristic_actions

    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)

    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    obs_p1_new = jax.vmap(lambda s: game.get_observation(s, 1))(new_states)
    learner_obs_new = jax.lax.cond(learner_player == 0, lambda _: obs_p0_new, lambda _: obs_p1_new, None)
    composite_rewards = jax.vmap(composite_reward_fn)(learner_obs_prior, learner_actions, learner_obs_new)
    rewards = apply_reward_mode(composite_rewards, reward_mode_id)

    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated
    rewards = apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale)
    rewards = apply_truncation_reward(rewards, truncated, truncation_reward_scale)

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
        (obs_arr, masks, active, learner_actions, logprobs, values, rewards, dones, infos),
        key,
    )


def collect_rollout(
    states,
    effective_sizes,
    pool,
    network,
    key,
    num_steps,
    truncation,
    opponent_id,
    learner_player,
    reward_mode_id,
    terminal_reward_scale,
    truncation_reward_scale,
    pad_size,
    global_context=False,
    scoreboard_history=None,
    scoreboard_history_enabled=False,
    fog_memory=None,
    fog_memory_enabled=False,
    teacher_network=None,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    teacher_rollout_actions: bool = False,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
    opponent_policy_mix_prob: float = 1.0,
):
    """Collect a Python-loop rollout, stacking step data on axis 0."""
    step_data = []
    for _ in range(num_steps):
        states, effective_sizes, scoreboard_history, fog_memory, data, key = rollout_step(
            states,
            effective_sizes,
            pool,
            network,
            key,
            truncation,
            opponent_id,
            learner_player,
            reward_mode_id,
            terminal_reward_scale,
            truncation_reward_scale,
            pad_size,
            global_context,
            scoreboard_history,
            scoreboard_history_enabled,
            fog_memory,
            fog_memory_enabled,
            teacher_network,
            teacher_input_channels,
            teacher_rollout_actions,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
            opponent_policy_mix_prob,
        )
        step_data.append(data)
    rollout_data = jax.tree.map(lambda *xs: jnp.stack(xs), *step_data)
    return states, effective_sizes, scoreboard_history, fog_memory, rollout_data, key


def collect_mixed_rollout(
    states_p0,
    effective_sizes_p0,
    states_p1,
    effective_sizes_p1,
    pool,
    network,
    key,
    num_steps,
    truncation,
    opponent_id,
    reward_mode_id,
    terminal_reward_scale,
    truncation_reward_scale,
    pad_size,
    global_context=False,
    scoreboard_history_p0=None,
    scoreboard_history_p1=None,
    scoreboard_history_enabled=False,
    fog_memory_p0=None,
    fog_memory_p1=None,
    fog_memory_enabled=False,
    teacher_network=None,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    teacher_rollout_actions: bool = False,
    opponent_policy_network=None,
    opponent_policy_mode: int = 1,
    opponent_policy_grid_size: int = 0,
    opponent_policy_mix_prob: float = 1.0,
):
    """Collect P0 and P1 learner trajectories, then combine them for one PPO update."""
    key, p0_key, p1_key = jrandom.split(key, 3)
    states_p0, effective_sizes_p0, scoreboard_history_p0, fog_memory_p0, rollout_p0, _ = collect_rollout(
        states_p0,
        effective_sizes_p0,
        pool,
        network,
        p0_key,
        num_steps,
        truncation,
        opponent_id,
        0,
        reward_mode_id,
        terminal_reward_scale,
        truncation_reward_scale,
        pad_size,
        global_context,
        scoreboard_history_p0,
        scoreboard_history_enabled,
        fog_memory_p0,
        fog_memory_enabled,
        teacher_network,
        teacher_input_channels,
        teacher_rollout_actions,
        opponent_policy_network,
        opponent_policy_mode,
        opponent_policy_grid_size,
        opponent_policy_mix_prob,
    )
    states_p1, effective_sizes_p1, scoreboard_history_p1, fog_memory_p1, rollout_p1, _ = collect_rollout(
        states_p1,
        effective_sizes_p1,
        pool,
        network,
        p1_key,
        num_steps,
        truncation,
        opponent_id,
        1,
        reward_mode_id,
        terminal_reward_scale,
        truncation_reward_scale,
        pad_size,
        global_context,
        scoreboard_history_p1,
        scoreboard_history_enabled,
        fog_memory_p1,
        fog_memory_enabled,
        teacher_network,
        teacher_input_channels,
        teacher_rollout_actions,
        opponent_policy_network,
        opponent_policy_mode,
        opponent_policy_grid_size,
        opponent_policy_mix_prob,
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


@jax.jit
def ppo_loss_terms(network, obs, mask, active, action, old_logprob, advantage, ret, clip=0.2):
    """Return per-sample PPO policy, value, and entropy terms."""
    logits, value, value_logits = network.logits_value_distribution(obs, mask, active)
    action_index = adaptive_action_to_index(action, network.pad_size)
    log_probs = jax.nn.log_softmax(logits)
    logprob = log_probs[action_index]
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs)
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    if network.value_bins > 0:
        value_loss = hl_gauss_value_loss(
            value_logits,
            ret,
            network.value_bins,
            network.value_min,
            network.value_max,
            network.value_sigma,
        )
    else:
        value_loss = 0.5 * (value - ret) ** 2
    return policy_loss, value_loss, entropy


@jax.jit
def ppo_loss_terms_with_outcome(
    network,
    obs,
    mask,
    active,
    action,
    old_logprob,
    advantage,
    ret,
    outcome_target,
    outcome_weight,
    clip=0.2,
):
    """Return PPO terms plus masked outcome auxiliary cross-entropy."""
    logits, value, value_logits, outcome_logits = network.logits_value_auxiliary(obs, mask, active)
    action_index = adaptive_action_to_index(action, network.pad_size)
    log_probs = jax.nn.log_softmax(logits)
    logprob = log_probs[action_index]
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs)
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    if network.value_bins > 0:
        value_loss = hl_gauss_value_loss(
            value_logits,
            ret,
            network.value_bins,
            network.value_min,
            network.value_max,
            network.value_sigma,
        )
    else:
        value_loss = 0.5 * (value - ret) ** 2
    if network.outcome_head:
        outcome_log_probs = jax.nn.log_softmax(outcome_logits)
        outcome_loss = -outcome_log_probs[outcome_target] * outcome_weight
    else:
        outcome_loss = jnp.asarray(0.0, dtype=jnp.float32)
    return policy_loss, value_loss, entropy, outcome_loss


@jax.jit
def ppo_loss(network, obs, mask, active, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for one adaptive sample."""
    policy_loss, value_loss, entropy = ppo_loss_terms(network, obs, mask, active, action, old_logprob, advantage, ret, clip)
    return policy_loss + value_loss - 0.01 * entropy


def teacher_obs_from_student_obs(obs: jnp.ndarray, teacher_input_channels: int) -> jnp.ndarray:
    """Drop student-only memory/history planes when querying a legacy teacher."""
    if obs.shape[0] == teacher_input_channels:
        return obs
    if teacher_input_channels == ADAPTIVE_INPUT_CHANNELS:
        return obs[:ADAPTIVE_INPUT_CHANNELS]
    fog_end = ADAPTIVE_INPUT_CHANNELS + ADAPTIVE_FOG_MEMORY_CHANNELS
    if obs.shape[0] == ADAPTIVE_GLOBAL_INPUT_CHANNELS + ADAPTIVE_FOG_MEMORY_CHANNELS:
        if teacher_input_channels == ADAPTIVE_GLOBAL_INPUT_CHANNELS:
            return jnp.concatenate([obs[:ADAPTIVE_INPUT_CHANNELS], obs[fog_end : fog_end + 5]], axis=0)
    if obs.shape[0] == ADAPTIVE_HISTORY_INPUT_CHANNELS + ADAPTIVE_FOG_MEMORY_CHANNELS:
        if teacher_input_channels == ADAPTIVE_GLOBAL_INPUT_CHANNELS:
            return jnp.concatenate([obs[:ADAPTIVE_INPUT_CHANNELS], obs[fog_end : fog_end + 5]], axis=0)
        if teacher_input_channels == ADAPTIVE_HISTORY_INPUT_CHANNELS:
            return jnp.concatenate([obs[:ADAPTIVE_INPUT_CHANNELS], obs[fog_end : fog_end + 15]], axis=0)
    return obs[:teacher_input_channels]


def teacher_policy_kl(network, teacher_network, obs, mask, active, teacher_input_channels: int) -> jnp.ndarray:
    """KL(teacher || student) policy anchor for trunk replacement experiments."""
    student_logits, _ = network.logits_value(obs, mask, active)
    teacher_obs = teacher_obs_from_student_obs(obs, teacher_input_channels)
    teacher_logits, _ = teacher_network.logits_value(teacher_obs, mask, active)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits)
    teacher_probs = jax.lax.stop_gradient(jax.nn.softmax(teacher_logits))
    teacher_log_probs = jax.lax.stop_gradient(teacher_log_probs)
    return jnp.sum(teacher_probs * (teacher_log_probs - jax.nn.log_softmax(student_logits)))


def policy_action_cross_entropy(network, obs, mask, active, action) -> jnp.ndarray:
    """Cross-entropy for an externally supplied adaptive action label."""
    logits, _ = network.logits_value(obs, mask, active)
    action_index = adaptive_action_to_index(action, network.pad_size)
    return -jax.nn.log_softmax(logits)[action_index]


@eqx.filter_jit
def train_minibatch_step(
    network,
    opt_state,
    minibatch,
    optimizer,
    outcome_aux_weight: float = 0.0,
    context_only_update: bool = False,
    teacher_network=None,
    teacher_kl_weight: float = 0.0,
    teacher_input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    teacher_action_ce_weight: float = 0.0,
):
    """Run one PPO update on a flattened adaptive minibatch."""
    (
        obs,
        masks,
        active,
        actions,
        old_logprobs,
        advantages,
        returns,
        policy_weights,
        outcome_targets,
        outcome_weights,
    ) = minibatch

    def loss_fn(net):
        policy_losses, value_losses, entropies, outcome_losses = jax.vmap(
            lambda o, m, ac, a, olp, adv, r, ot, ow: ppo_loss_terms_with_outcome(
                net,
                o,
                m,
                ac,
                a,
                olp,
                adv,
                r,
                ot,
                ow,
            )
        )(
            obs,
            masks,
            active,
            actions,
            old_logprobs,
            advantages,
            returns,
            outcome_targets,
            outcome_weights,
        )
        policy_normalizer = jnp.maximum(jnp.sum(policy_weights), 1.0)
        policy_loss = jnp.sum(policy_losses * policy_weights) / policy_normalizer
        entropy_loss = -0.01 * jnp.sum(entropies * policy_weights) / policy_normalizer
        value_loss = jnp.mean(value_losses)
        outcome_normalizer = jnp.maximum(jnp.sum(outcome_weights), 1.0)
        outcome_loss = jnp.sum(outcome_losses) / outcome_normalizer
        teacher_loss = jnp.asarray(0.0, dtype=jnp.float32)
        if teacher_network is not None and teacher_kl_weight > 0.0:
            teacher_losses = jax.vmap(
                lambda o, m, ac: teacher_policy_kl(net, teacher_network, o, m, ac, teacher_input_channels)
            )(obs, masks, active)
            teacher_loss = jnp.mean(teacher_losses)
        teacher_action_ce_loss = jnp.asarray(0.0, dtype=jnp.float32)
        if teacher_action_ce_weight > 0.0:
            teacher_action_ce_losses = jax.vmap(
                lambda o, m, ac, a: policy_action_cross_entropy(net, o, m, ac, a)
            )(obs, masks, active, actions)
            teacher_action_ce_loss = jnp.mean(teacher_action_ce_losses)
        return (
            policy_loss
            + value_loss
            + entropy_loss
            + outcome_aux_weight * outcome_loss
            + teacher_kl_weight * teacher_loss
            + teacher_action_ce_weight * teacher_action_ce_loss
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    if context_only_update:
        grads = context_only_grad_tree(grads)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss


def flatten_training_batch(batch):
    """Flatten rollout time/environment axes into one sample axis."""
    if len(batch) == 8:
        obs, masks, active, actions, old_logprobs, advantages, returns, policy_weights = batch
        outcome_targets = jnp.zeros_like(returns, dtype=jnp.int32)
        outcome_weights = jnp.zeros_like(returns, dtype=jnp.float32)
    else:
        (
            obs,
            masks,
            active,
            actions,
            old_logprobs,
            advantages,
            returns,
            policy_weights,
            outcome_targets,
            outcome_weights,
        ) = batch
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        actions.reshape(batch_size, -1),
        old_logprobs.reshape(batch_size),
        advantages.reshape(batch_size),
        returns.reshape(batch_size),
        policy_weights.reshape(batch_size),
        outcome_targets.reshape(batch_size),
        outcome_weights.reshape(batch_size),
    )


def train_epoch(
    network,
    opt_state,
    batch,
    optimizer,
    key,
    num_epochs=1,
    minibatch_size=None,
    outcome_aux_weight=0.0,
    context_only_update=False,
    teacher_network=None,
    teacher_kl_weight=0.0,
    teacher_input_channels=ADAPTIVE_INPUT_CHANNELS,
    teacher_action_ce_weight=0.0,
):
    """Run adaptive PPO epochs with optional minibatching."""
    flat_batch = flatten_training_batch(batch)
    batch_size = flat_batch[0].shape[0]
    actual_minibatch_size = batch_size if minibatch_size is None else min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = tuple(x[permutation] for x in flat_batch)
        epoch_loss = 0.0
        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            network, opt_state, loss = train_minibatch_step(
                network,
                opt_state,
                minibatch,
                optimizer,
                outcome_aux_weight,
                context_only_update,
                teacher_network,
                teacher_kl_weight,
                teacher_input_channels,
                teacher_action_ce_weight,
            )
            epoch_loss += loss
        avg_loss = epoch_loss / num_complete_batches

    return network, opt_state, avg_loss, key


def parse_args():
    parser = argparse.ArgumentParser(description="Train an adaptive multisize PPO policy.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--top-advantage-fraction", type=float, default=1.0)
    parser.add_argument("--top-advantage-mode", choices=("global", "stratified"), default="global")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--eval-ema", action="store_true")
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--truncation", type=int, default=750)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-grid-size", type=int, default=None)
    parser.add_argument("--opponent-policy-mix-prob", type=float, default=1.0)
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--learner-player", choices=("0", "1", "alternate", "mixed"), default="0")
    parser.add_argument("--reward-mode", choices=REWARD_MODE_NAMES, default="composite")
    parser.add_argument("--terminal-reward-scale", type=float, default=0.0)
    parser.add_argument("--truncation-reward-scale", type=float, default=0.0)
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--init-network-arch", choices=("cnn", "unet"), default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--fog-memory", action="store_true")
    parser.add_argument("--init-global-context", action="store_true")
    parser.add_argument("--context-residual", action="store_true")
    parser.add_argument("--init-context-residual", action="store_true")
    parser.add_argument("--pyramid-context", action="store_true")
    parser.add_argument("--init-pyramid-context", action="store_true")
    parser.add_argument("--context-only-update", action="store_true")
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
    parser.add_argument("--outcome-aux-weight", type=float, default=0.0)
    parser.add_argument("--init-outcome-head", action="store_true")
    parser.add_argument("--strategy-aux", action="store_true")
    parser.add_argument("--strategy-spatial-aux", action="store_true")
    parser.add_argument("--strategy-finish-outputs", type=int, default=2)
    parser.add_argument("--init-strategy-aux", action="store_true")
    parser.add_argument("--init-strategy-spatial-aux", action="store_true")
    parser.add_argument("--init-strategy-finish-outputs", type=int, default=2)
    parser.add_argument(
        "--drop-mismatched-init-leaves",
        action="store_true",
        help="Load matching checkpoint leaves and reinitialize shape-mismatched legacy leaves.",
    )
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--teacher-channels", default=None)
    parser.add_argument("--teacher-input-channels", type=int, default=None)
    parser.add_argument("--teacher-global-context", action="store_true")
    parser.add_argument("--teacher-scoreboard-history", action="store_true")
    parser.add_argument("--teacher-value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--teacher-value-head-sizes", default=None)
    parser.add_argument("--teacher-value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--teacher-value-bins", type=int, default=128)
    parser.add_argument("--teacher-outcome-head", action="store_true")
    parser.add_argument("--teacher-strategy-aux", action="store_true")
    parser.add_argument("--teacher-strategy-finish-outputs", type=int, default=2)
    parser.add_argument(
        "--teacher-drop-mismatched-init-leaves",
        action="store_true",
        help="Load matching teacher checkpoint leaves and reinitialize shape-mismatched legacy leaves.",
    )
    parser.add_argument("--teacher-rollout-actions", action="store_true")
    parser.add_argument("--teacher-action-ce-weight", type=float, default=0.0)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/generals-adaptive-ppo.eqx")
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
        args.opponent_channels = parse_policy_channels(args.opponent_channels)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.teacher_value_head_sizes = (
            parse_grid_sizes(args.teacher_value_head_sizes)
            if args.teacher_value_head_sizes is not None
            else args.grid_sizes
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_envs <= 0:
        parser.error("num_envs must be positive")
    if args.learner_player == "mixed" and args.num_envs < 2:
        parser.error("--learner-player mixed requires num_envs >= 2")
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if args.num_iterations <= 0:
        parser.error("--num-iterations must be positive")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.minibatch_size is not None and args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if not (0.0 <= args.gamma <= 1.0):
        parser.error("--gamma must be between 0 and 1")
    if not (0.0 <= args.gae_lambda <= 1.0):
        parser.error("--gae-lambda must be between 0 and 1")
    if not (0.0 < args.top_advantage_fraction <= 1.0):
        parser.error("--top-advantage-fraction must be in (0, 1]")
    if not (0.0 <= args.ema_decay < 1.0):
        parser.error("--ema-decay must be in [0, 1)")
    if args.eval_ema and args.ema_decay <= 0.0:
        parser.error("--eval-ema requires --ema-decay > 0")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    if not (0.0 <= args.opponent_policy_mix_prob <= 1.0):
        parser.error("--opponent-policy-mix-prob must be in [0, 1]")
    if args.opponent_policy_grid_size is not None and args.opponent_policy_grid_size <= 0:
        parser.error("--opponent-policy-grid-size must be positive")
    if args.opponent_policy_path is not None:
        if len(args.grid_sizes) > 1 and args.opponent_policy_grid_size is None:
            parser.error("--opponent-policy-grid-size is required with multi-size --opponent-policy-path")
        if args.opponent_policy_grid_size is None:
            args.opponent_policy_grid_size = args.grid_sizes[0]
        if args.opponent_policy_grid_size not in args.grid_sizes:
            parser.error("--opponent-policy-grid-size must be one of --grid-sizes")
        if args.opponent_policy_grid_size > args.pad_to:
            parser.error("--opponent-policy-grid-size must be <= --pad-to")
    elif args.opponent_policy_grid_size is not None:
        parser.error("--opponent-policy-grid-size requires --opponent-policy-path")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
    if args.terminal_reward_scale < 0.0:
        parser.error("--terminal-reward-scale must be non-negative")
    if args.truncation_reward_scale < 0.0:
        parser.error("--truncation-reward-scale must be non-negative")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
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
    if args.outcome_aux_weight < 0.0:
        parser.error("--outcome-aux-weight must be non-negative")
    if args.strategy_spatial_aux and not args.strategy_aux:
        parser.error("--strategy-spatial-aux requires --strategy-aux")
    if args.init_strategy_spatial_aux and not args.init_strategy_aux:
        parser.error("--init-strategy-spatial-aux requires --init-strategy-aux")
    if args.strategy_finish_outputs <= 0:
        parser.error("--strategy-finish-outputs must be positive")
    if args.init_strategy_finish_outputs <= 0:
        parser.error("--init-strategy-finish-outputs must be positive")
    if args.context_only_update and not (args.context_residual or args.pyramid_context):
        parser.error("--context-only-update requires --context-residual or --pyramid-context")
    if args.network_arch == "unet" and (args.context_residual or args.pyramid_context or args.context_only_update):
        parser.error("--network-arch unet replaces context/pyramid add-on branches")
    if args.teacher_kl_weight < 0.0:
        parser.error("--teacher-kl-weight must be non-negative")
    if args.teacher_kl_weight > 0.0 and args.teacher_model_path is None:
        parser.error("--teacher-kl-weight requires --teacher-model-path")
    if args.teacher_rollout_actions and args.teacher_model_path is None:
        parser.error("--teacher-rollout-actions requires --teacher-model-path")
    if args.teacher_action_ce_weight < 0.0:
        parser.error("--teacher-action-ce-weight must be non-negative")
    if args.teacher_action_ce_weight > 0.0 and not args.teacher_rollout_actions:
        parser.error("--teacher-action-ce-weight requires --teacher-rollout-actions")
    if args.teacher_input_channels is not None and args.teacher_input_channels <= 0:
        parser.error("--teacher-input-channels must be positive")
    if args.teacher_value_loss == "hl-gauss" and args.teacher_value_bins <= 1:
        parser.error("--teacher-value-bins must be greater than 1 for --teacher-value-loss hl-gauss")
    if args.teacher_strategy_finish_outputs <= 0:
        parser.error("--teacher-strategy-finish-outputs must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative")
    if args.keep_checkpoints < 0:
        parser.error("--keep-checkpoints cannot be negative")
    return args


def main():
    args = parse_args()

    print("Adaptive JAX PPO")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Environments:  {args.num_envs}")
    if args.learner_player == "alternate":
        learner_label = "alternate players 0/1"
    elif args.learner_player == "mixed":
        mixed_p0_envs, mixed_p1_envs = split_mixed_env_counts(args.num_envs)
        learner_label = f"mixed players 0/1 ({mixed_p0_envs}+{mixed_p1_envs} envs)"
    else:
        learner_label = f"player {args.learner_player}"
    print(f"Learner:       {learner_label}")
    if args.opponent_policy_path is None:
        print(f"Opponent:      {args.opponent}")
    else:
        print("Opponent:      policy checkpoint")
        print(f"Opp model:     {args.opponent_policy_path}")
        print(f"Opp mode:      {args.opponent_policy_mode}")
        print(f"Opp size:      {args.opponent_policy_grid_size}")
        if args.opponent_policy_mix_prob < 1.0 or len(args.grid_sizes) > 1:
            print(f"Opp mix:       policy rows p={args.opponent_policy_mix_prob:g}, fallback={args.opponent}")
        print(f"Opp channels:  {args.opponent_channels}")
        print(f"Opp inputs:    {args.opponent_input_channels}")
    print(f"Reward mode:   {args.reward_mode}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Network arch:  {args.network_arch}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Iterations:    {args.num_iterations} x {args.num_steps} steps")
    print(f"PPO updates:   epochs={args.num_epochs}, minibatch={args.minibatch_size or args.num_envs * args.num_steps}")
    print(f"GAE:           gamma={args.gamma:g}, lambda={args.gae_lambda:g}")
    if args.top_advantage_fraction < 1.0:
        print(f"Top advantage: {args.top_advantage_fraction:g} ({args.top_advantage_mode})")
    if args.ema_decay > 0.0:
        ema_target = "EMA" if args.eval_ema else "last iterate"
        print(f"EMA:           decay={args.ema_decay:g}, saving={ema_target}")
    if args.truncation_reward_scale > 0.0:
        print(f"Timeout reward: -{args.truncation_reward_scale:g}")
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
        if args.init_channels is not None:
            print(f"Warm channels: {args.init_channels}")
        if args.init_input_channels is not None:
            print(f"Warm inputs:   {args.init_input_channels} channels")
        if args.init_global_context:
            print("Warm global:   enabled")
        if args.init_context_residual:
            print("Warm context:  enabled")
        if args.init_pyramid_context:
            print("Warm pyramid:  enabled")
    if args.init_strategy_aux:
        print("Warm strategy: enabled")
    if args.strategy_aux:
        print(f"Strategy aux:  finish_outputs={args.strategy_finish_outputs}")
    if args.strategy_spatial_aux:
        print("Strategy map:  source/target heads enabled")
    if args.context_residual:
        print("Context res:   5x5 zero-init residual branch")
    if args.pyramid_context:
        print("Pyramid ctx:   16->8->4 zero-init U-Net branch")
    if args.context_only_update:
        print("Update scope:  context/pyramid branch only")
    network_global_context = args.global_context or args.scoreboard_history
    if network_global_context:
        print(f"Global ctx:    scoreboard channels ({ADAPTIVE_GLOBAL_INPUT_CHANNELS})")
    if args.scoreboard_history:
        print(f"Score history: previous+delta channels ({ADAPTIVE_HISTORY_INPUT_CHANNELS})")
    if args.fog_memory:
        print("Fog memory:    explored/enemy/city/general planes")
    if args.value_heads != "shared":
        print(f"Value heads:   {args.value_heads}")
        print(f"Value sizes:   {','.join(str(size) for size in args.value_head_sizes)}")
        if args.init_model_path is not None:
            print(f"Init values:   {args.init_value_heads}")
            print(f"Init sizes:    {','.join(str(size) for size in args.init_value_head_sizes)}")
    if args.value_loss == "hl-gauss":
        print(
            "Value loss:    "
            f"hl-gauss bins={args.value_bins} range=[{args.value_min:g},{args.value_max:g}] "
            f"sigma={args.value_sigma:g}"
        )
    if args.outcome_aux_weight > 0.0:
        print(f"Outcome aux:   weight={args.outcome_aux_weight:g}")
    if args.teacher_model_path is not None:
        print(f"Teacher:       {args.teacher_model_path} arch={args.teacher_network_arch}")
        if args.teacher_value_heads != "shared":
            print(
                "Teacher value: "
                f"{args.teacher_value_heads} {','.join(str(size) for size in args.teacher_value_head_sizes)}"
            )
        if args.teacher_value_loss == "hl-gauss":
            print(f"Teacher value: hl-gauss bins={args.teacher_value_bins}")
        if args.teacher_outcome_head:
            print("Teacher aux:   outcome head")
        if args.teacher_strategy_aux:
            print(f"Teacher strat: finish_outputs={args.teacher_strategy_finish_outputs}")
        if args.teacher_kl_weight > 0.0:
            print(f"Teacher KL:    weight={args.teacher_kl_weight:g}")
        if args.teacher_rollout_actions:
            print("Teacher acts:  rollout bootstrap enabled")
        if args.teacher_action_ce_weight > 0.0:
            print(f"Teacher CE:    weight={args.teacher_action_ce_weight:g}")
    if args.checkpoint_dir is not None and args.checkpoint_every > 0:
        print(f"Checkpoints:   every {args.checkpoint_every} iterations in {args.checkpoint_dir}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, net_key, pool_key = jrandom.split(key, 3)
    value_bins = args.value_bins if args.value_loss == "hl-gauss" else 0
    init_value_bins = (
        (args.value_bins if args.init_value_bins is None else args.init_value_bins)
        if args.init_value_loss == "hl-gauss"
        else 0
    )
    network_global_context = args.global_context or args.scoreboard_history
    input_channels = adaptive_input_channel_count(network_global_context, args.scoreboard_history, args.fog_memory)
    init_input_channels = args.init_input_channels
    if (
        init_input_channels is None
        and args.init_model_path is not None
        and network_global_context
        and not args.init_global_context
    ):
        init_input_channels = ADAPTIVE_INPUT_CHANNELS
    network = load_or_create_adaptive_network(
        net_key,
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
        outcome_head=args.outcome_aux_weight > 0.0,
        init_outcome_head=args.init_outcome_head,
        strategy_aux=args.strategy_aux,
        init_strategy_aux=args.init_strategy_aux,
        strategy_spatial_aux=args.strategy_spatial_aux,
        init_strategy_spatial_aux=args.init_strategy_spatial_aux,
        strategy_finish_outputs=args.strategy_finish_outputs,
        init_strategy_finish_outputs=args.init_strategy_finish_outputs,
        global_context=network_global_context,
        init_global_context=args.init_global_context,
        context_residual=args.context_residual,
        init_context_residual=args.init_context_residual,
        pyramid_context=args.pyramid_context,
        init_pyramid_context=args.init_pyramid_context,
        network_arch=args.network_arch,
        init_network_arch=args.init_network_arch,
        drop_mismatched_init_leaves=args.drop_mismatched_init_leaves,
    )
    teacher_network = None
    teacher_input_channels = ADAPTIVE_INPUT_CHANNELS
    if args.teacher_model_path is not None:
        teacher_global_context = args.teacher_global_context or args.teacher_scoreboard_history
        teacher_input_channels = (
            args.teacher_input_channels
            if args.teacher_input_channels is not None
            else adaptive_input_channel_count(teacher_global_context, args.teacher_scoreboard_history, False)
        )
        teacher_value_bins = args.teacher_value_bins if args.teacher_value_loss == "hl-gauss" else 0
        teacher_network = load_or_create_adaptive_network(
            net_key,
            pad_size=args.pad_to,
            init_model_path=args.teacher_model_path,
            channels=args.teacher_channels,
            input_channels=teacher_input_channels,
            init_input_channels=teacher_input_channels,
            value_head_sizes=args.teacher_value_head_sizes if args.teacher_value_heads == "per-size" else (),
            init_value_head_sizes=args.teacher_value_head_sizes if args.teacher_value_heads == "per-size" else (),
            value_bins=teacher_value_bins,
            init_value_bins=teacher_value_bins,
            outcome_head=args.teacher_outcome_head,
            init_outcome_head=args.teacher_outcome_head,
            strategy_aux=args.teacher_strategy_aux,
            init_strategy_aux=args.teacher_strategy_aux,
            strategy_finish_outputs=args.teacher_strategy_finish_outputs,
            init_strategy_finish_outputs=args.teacher_strategy_finish_outputs,
            global_context=teacher_global_context,
            init_global_context=teacher_global_context,
            network_arch=args.teacher_network_arch,
            init_network_arch=args.teacher_network_arch,
            drop_mismatched_init_leaves=args.teacher_drop_mismatched_init_leaves,
        )
        teacher_input_channels = adaptive_network_input_channels(teacher_network)
    opponent_policy_network = None
    opponent_policy_mode = 0 if args.opponent_policy_mode == "greedy" else 1
    opponent_policy_grid_size = args.opponent_policy_grid_size or args.grid_sizes[0]
    if args.opponent_policy_path is not None:
        opponent_policy_network = PolicyValueNetwork(
            net_key,
            grid_size=opponent_policy_grid_size,
            channels=args.opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_policy_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_policy_network)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    ema_network = network if args.ema_decay > 0.0 else None
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    reward_mode_id = REWARD_MODE_NAME_TO_ID[args.reward_mode]

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
    mixed_learner = args.learner_player == "mixed"
    if mixed_learner:
        mixed_p0_envs, mixed_p1_envs = split_mixed_env_counts(args.num_envs)
        mixed_states, mixed_effective_sizes = make_adaptive_initial_states(pool, args.num_envs)
        states_p0 = jax.tree.map(lambda x: x[:mixed_p0_envs], mixed_states)
        effective_sizes_p0 = mixed_effective_sizes[:mixed_p0_envs]
        states_p1 = jax.tree.map(lambda x: x[mixed_p0_envs:], mixed_states)
        effective_sizes_p1 = mixed_effective_sizes[mixed_p0_envs:]
        scoreboard_history_p0 = empty_scoreboard_history(mixed_p0_envs)
        scoreboard_history_p1 = empty_scoreboard_history(mixed_p1_envs)
        fog_memory_p0 = empty_adaptive_fog_memory(mixed_p0_envs, args.pad_to)
        fog_memory_p1 = empty_adaptive_fog_memory(mixed_p1_envs, args.pad_to)
        learner_players_for_log = jnp.concatenate(
            [
                jnp.zeros((mixed_p0_envs,), dtype=jnp.int32),
                jnp.ones((mixed_p1_envs,), dtype=jnp.int32),
            ]
        )
    else:
        states, effective_sizes = make_adaptive_initial_states(pool, args.num_envs)
        scoreboard_history = empty_scoreboard_history(args.num_envs)
        fog_memory = empty_adaptive_fog_memory(args.num_envs, args.pad_to)

    print("Warming up...")
    if mixed_learner:
        key, warmup_p0_key, warmup_p1_key = jrandom.split(key, 3)
        states_p0, effective_sizes_p0, scoreboard_history_p0, fog_memory_p0, _, _ = rollout_step(
            states_p0,
            effective_sizes_p0,
            pool,
            network,
            warmup_p0_key,
            args.truncation,
            opponent_id,
            0,
            reward_mode_id,
            args.terminal_reward_scale,
            args.truncation_reward_scale,
            args.pad_to,
            network_global_context,
            scoreboard_history_p0,
            args.scoreboard_history,
            fog_memory_p0,
            args.fog_memory,
            teacher_network,
            teacher_input_channels,
            args.teacher_rollout_actions,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
            args.opponent_policy_mix_prob,
        )
        states_p1, effective_sizes_p1, scoreboard_history_p1, fog_memory_p1, _, _ = rollout_step(
            states_p1,
            effective_sizes_p1,
            pool,
            network,
            warmup_p1_key,
            args.truncation,
            opponent_id,
            1,
            reward_mode_id,
            args.terminal_reward_scale,
            args.truncation_reward_scale,
            args.pad_to,
            network_global_context,
            scoreboard_history_p1,
            args.scoreboard_history,
            fog_memory_p1,
            args.fog_memory,
            teacher_network,
            teacher_input_channels,
            args.teacher_rollout_actions,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
            args.opponent_policy_mix_prob,
        )
        jax.block_until_ready(states_p0)
        jax.block_until_ready(states_p1)
    else:
        warmup_learner_player = resolve_learner_player(args.learner_player, 1)
        states, effective_sizes, scoreboard_history, fog_memory, _, key = rollout_step(
            states,
            effective_sizes,
            pool,
            network,
            key,
            args.truncation,
            opponent_id,
            warmup_learner_player,
            reward_mode_id,
            args.terminal_reward_scale,
            args.truncation_reward_scale,
            args.pad_to,
            network_global_context,
            scoreboard_history,
            args.scoreboard_history,
            fog_memory,
            args.fog_memory,
            teacher_network,
            teacher_input_channels,
            args.teacher_rollout_actions,
            opponent_policy_network,
            opponent_policy_mode,
            opponent_policy_grid_size,
            args.opponent_policy_mix_prob,
        )
        jax.block_until_ready(states)

    checkpoint_paths = []
    model_stem = Path(args.model_path).stem
    for iteration in range(1, args.num_iterations + 1):
        t0 = time.time()
        if mixed_learner:
            (
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
            ) = collect_mixed_rollout(
                states_p0,
                effective_sizes_p0,
                states_p1,
                effective_sizes_p1,
                pool,
                network,
                key,
                args.num_steps,
                args.truncation,
                opponent_id,
                reward_mode_id,
                args.terminal_reward_scale,
                args.truncation_reward_scale,
                args.pad_to,
                network_global_context,
                scoreboard_history_p0,
                scoreboard_history_p1,
                args.scoreboard_history,
                fog_memory_p0,
                fog_memory_p1,
                args.fog_memory,
                teacher_network,
                teacher_input_channels,
                args.teacher_rollout_actions,
                opponent_policy_network,
                opponent_policy_mode,
                opponent_policy_grid_size,
                args.opponent_policy_mix_prob,
            )
            jax.block_until_ready(states_p0)
            jax.block_until_ready(states_p1)
        else:
            iteration_learner_player = resolve_learner_player(args.learner_player, iteration)
            states, effective_sizes, scoreboard_history, fog_memory, rollout_data, key = collect_rollout(
                states,
                effective_sizes,
                pool,
                network,
                key,
                args.num_steps,
                args.truncation,
                opponent_id,
                iteration_learner_player,
                reward_mode_id,
                args.terminal_reward_scale,
                args.truncation_reward_scale,
                args.pad_to,
                network_global_context,
                scoreboard_history,
                args.scoreboard_history,
                fog_memory,
                args.fog_memory,
                teacher_network,
                teacher_input_channels,
                args.teacher_rollout_actions,
                opponent_policy_network,
                opponent_policy_mode,
                opponent_policy_grid_size,
                args.opponent_policy_mix_prob,
            )
            jax.block_until_ready(states)
        obs, masks, active, actions, logprobs, values, rewards, dones, infos = rollout_data
        advantages, returns = compute_gae(rewards, values, dones, args.gamma, args.gae_lambda)
        policy_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if mixed_learner:
            learner_players_for_batch = learner_players_for_log
        else:
            learner_players_for_batch = jnp.full((args.num_envs,), iteration_learner_player, dtype=jnp.int32)
        if args.top_advantage_mode == "stratified":
            policy_weights = stratified_top_advantage_weights(
                policy_advantages,
                active,
                learner_players_for_batch,
                args.grid_sizes,
                args.top_advantage_fraction,
            )
        else:
            policy_weights = top_advantage_weights(policy_advantages, args.top_advantage_fraction)
        outcome_targets, outcome_weights = rollout_outcome_targets(infos.winner, dones, learner_players_for_batch)
        batch = (
            obs,
            masks,
            active,
            actions,
            logprobs,
            policy_advantages,
            returns,
            policy_weights,
            outcome_targets,
            outcome_weights,
        )
        key, train_key = jrandom.split(key)
        network, opt_state, loss, key = train_epoch(
            network,
            opt_state,
            batch,
            optimizer,
            train_key,
            args.num_epochs,
            args.minibatch_size,
            args.outcome_aux_weight,
            args.context_only_update,
            teacher_network,
            args.teacher_kl_weight,
            teacher_input_channels,
            args.teacher_action_ce_weight,
        )
        jax.block_until_ready(network)
        if ema_network is not None:
            ema_network = update_ema_network(ema_network, network, args.ema_decay)
            jax.block_until_ready(ema_network)

        if args.checkpoint_dir is not None and args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, model_stem, iteration)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_network = ema_network if args.eval_ema and ema_network is not None else network
            eqx.tree_serialise_leaves(checkpoint_path, checkpoint_network)
            checkpoint_paths.append(checkpoint_path)
            prune_old_checkpoints(checkpoint_paths, args.keep_checkpoints)

        if iteration % 10 == 0 or iteration == 1 or iteration == args.num_iterations:
            elapsed = time.time() - t0
            episodes = int(jnp.sum(dones))
            if mixed_learner:
                wins = int(jnp.sum(dones & (infos.winner == learner_players_for_log[None, :])))
            else:
                wins = int(jnp.sum(dones & (infos.winner == iteration_learner_player)))
            draws = int(jnp.sum(dones & (infos.winner < 0)))
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Episodes: {episodes:4d} | Wins: {wins:4d} | Draws: {draws:4d} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s"
            )

    final_network = ema_network if args.eval_ema and ema_network is not None else network
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, final_network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
