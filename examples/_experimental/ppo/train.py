"""Clean JAX PPO using the raw game API for maximum performance."""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core.action import compute_valid_move_mask
from generals.core import game
from generals.core.grid import generate_grid
from generals.core.rewards import (
    composite_reward_fn,
    general_target_reward_fn,
    path_assignment_reward_fn,
)

from common import (
    OPPONENT_NAME_TO_ID,
    OPPONENT_NAMES,
    POLICY_INPUT_NAME_TO_ID,
    POLICY_INPUT_NAMES,
    POLICY_MODE_NAME_TO_ID,
    POLICY_MODE_NAMES,
    opponent_action,
    policy_input_array_and_mask,
    policy_input_default_channels,
    policy_state_action,
)
from network import PolicyValueNetwork
from generals.agents.ppo_policy_agent import parse_policy_channels


def random_action(key, obs):
    """Random valid action."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = jnp.argwhere(mask, size=mask.size, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid >= 0, axis=-1))

    k1, k2 = jrandom.split(key)

    idx = jrandom.randint(k1, (), 0, jnp.maximum(num_valid, 1))
    move = jnp.where(
        num_valid > 0,
        valid[idx],
        jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    should_pass = num_valid == 0
    is_half = jrandom.randint(k2, (), 0, 2)

    return jnp.array([should_pass, move[0], move[1], move[2], is_half], dtype=jnp.int32)


def make_simple_general_grid(key, grid_size):
    """Create an empty square grid with two random generals."""
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    return grid.at[pos_a].set(1).at[pos_b].set(2)


def make_state_pool(
    key,
    pool_size,
    grid_size,
    map_generator,
    mountain_density_range,
    num_cities_range,
    min_generals_distance,
    max_generals_distance,
    castle_val_range,
):
    """Generate a reusable pool of initial states for auto-reset."""
    keys = jrandom.split(key, pool_size)

    if map_generator == "simple":
        grids = jax.vmap(lambda k: make_simple_general_grid(k, grid_size))(keys)
    else:
        grids = jax.vmap(
            lambda k: generate_grid(
                k,
                grid_dims=(grid_size, grid_size),
                pad_to=grid_size,
                mountain_density_range=mountain_density_range,
                num_cities_range=num_cities_range,
                min_generals_distance=min_generals_distance,
                max_generals_distance=max_generals_distance,
                castle_val_range=castle_val_range,
            )
        )(keys)

    return jax.vmap(game.create_initial_state)(grids)


def make_initial_states(pool, num_envs):
    """Take initial states from the pool and spread future reset indices."""
    states = jax.tree.map(lambda x: x[:num_envs], pool)
    pool_size = pool.armies.shape[0]
    pool_idx = (jnp.arange(num_envs, dtype=jnp.int32) + num_envs) % pool_size
    return states._replace(pool_idx=pool_idx)


def parse_opponent_policy_pool(pool_value, modes_value):
    """Parse comma-separated frozen opponent checkpoint paths and execution modes."""
    if not pool_value:
        return []
    paths = [item.strip() for item in pool_value.split(",") if item.strip()]
    if modes_value:
        modes = [item.strip() for item in modes_value.split(",") if item.strip()]
        if len(modes) != len(paths):
            raise ValueError("--opponent-policy-pool-modes must match --opponent-policy-pool length")
    else:
        modes = ["sample"] * len(paths)
    for mode in modes:
        if mode not in POLICY_MODE_NAMES:
            raise ValueError(f"Unsupported opponent policy mode: {mode}")
    return list(zip(paths, modes))


def load_opponent_policy_pool(key, pool_specs, grid_size, channels, input_channels):
    """Load a same-architecture pool of ordinary PolicyValueNetwork opponents."""
    networks = []
    modes = []
    for index, (path_value, mode_name) in enumerate(pool_specs):
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Opponent pool checkpoint not found: {path}")
        network = PolicyValueNetwork(
            jrandom.fold_in(key, index),
            grid_size=grid_size,
            channels=channels,
            input_channels=input_channels,
        )
        networks.append(eqx.tree_deserialise_leaves(path, network))
        modes.append(POLICY_MODE_NAME_TO_ID[mode_name])
    return tuple(networks), jnp.array(modes, dtype=jnp.int32)


def policy_pool_state_action(opponent_networks, opponent_modes, opponent_index, key, state, obs, player, policy_input):
    """Dispatch one action from a tuple of frozen ordinary policy opponents."""
    branches = tuple(
        (
            lambda network, mode: (
                lambda _: policy_state_action(
                    network,
                    key,
                    state,
                    obs,
                    player,
                    mode,
                    policy_input,
                )
            )
        )(network, opponent_modes[index])
        for index, network in enumerate(opponent_networks)
    )
    return jax.lax.switch(opponent_index, branches, None)


def resolve_opponent_source(opponent_policy_path, self_play_opponent, opponent_policy_pool=None):
    """Select which opponent source the PPO rollout loop should use."""
    has_pool = bool(opponent_policy_pool)
    enabled_sources = sum(bool(value) for value in (opponent_policy_path, self_play_opponent, has_pool))
    if enabled_sources > 1:
        raise ValueError(
            "--opponent-policy-path, --self-play-opponent, and --opponent-policy-pool are mutually exclusive"
        )
    if self_play_opponent:
        return "current"
    if has_pool:
        return "checkpoint_pool"
    if opponent_policy_path is not None:
        return "checkpoint"
    return "heuristic"


def load_or_create_network(key, grid_size, init_model_path=None, channels=None, input_channels=9, init_input_channels=None):
    """Create a policy network and optionally restore its leaves from a checkpoint."""
    parsed_channels = parse_policy_channels(channels)
    network = PolicyValueNetwork(
        key,
        grid_size=grid_size,
        channels=parsed_channels,
        input_channels=input_channels,
    )
    if init_model_path is None:
        return network

    path = Path(init_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    if init_input_channels is not None and init_input_channels != input_channels:
        if init_input_channels > input_channels:
            raise ValueError("init_input_channels cannot exceed input_channels")
        source_network = PolicyValueNetwork(
            key,
            grid_size=grid_size,
            channels=parsed_channels,
            input_channels=init_input_channels,
        )
        source_network = eqx.tree_deserialise_leaves(path, source_network)
        conv1_weight = jnp.zeros_like(network.conv1.weight)
        conv1_weight = conv1_weight.at[:, :init_input_channels, :, :].set(source_network.conv1.weight)
        network = eqx.tree_at(lambda net: net.conv1.weight, network, conv1_weight)
        network = eqx.tree_at(
            lambda net: (
                net.conv1.bias,
                net.conv2,
                net.conv3,
                net.conv4,
                net.policy_conv,
                net.value_conv,
                net.value_linear1,
                net.value_linear2,
            ),
            network,
            (
                source_network.conv1.bias,
                source_network.conv2,
                source_network.conv3,
                source_network.conv4,
                source_network.policy_conv,
                source_network.value_conv,
                source_network.value_linear1,
                source_network.value_linear2,
            ),
        )
        return network
    return eqx.tree_deserialise_leaves(path, network)


def checkpoint_path_for_iteration(checkpoint_dir, model_stem, iteration):
    """Return the periodic checkpoint path for one training iteration."""
    return Path(checkpoint_dir) / f"{model_stem}-iter-{iteration:06d}.eqx"


def prune_old_checkpoints(paths, keep):
    """Delete older periodic checkpoints when a positive keep limit is configured."""
    if keep is None or keep <= 0:
        return
    for path in list(paths)[:-keep]:
        Path(path).unlink(missing_ok=True)


def stack_learner_actions(learner_actions, opponent_actions, learner_player):
    """Place learner/opponent actions into the environment's player slots."""
    return jax.lax.cond(
        learner_player == 0,
        lambda _: jnp.stack([learner_actions, opponent_actions], axis=1),
        lambda _: jnp.stack([opponent_actions, learner_actions], axis=1),
        None,
    )


def select_learner_obs(obs_p0, obs_p1, learner_player):
    """Select observations from the learner player's perspective."""
    return jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)


def select_opponent_obs(obs_p0, obs_p1, learner_player):
    """Select observations from the opponent player's perspective."""
    return jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)


def apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale):
    """Add an optional zero-sum win/loss reward on decisive terminal transitions."""
    opponent_player = 1 - learner_player
    terminal_bonus = jnp.where(
        infos.winner == learner_player,
        terminal_reward_scale,
        jnp.where(infos.winner == opponent_player, -terminal_reward_scale, 0.0),
    )
    terminal_bonus = jnp.where(infos.is_done & (infos.winner >= 0), terminal_bonus, 0.0)
    return rewards + terminal_bonus


def apply_general_target_rewards(
    rewards,
    prior_states,
    states,
    learner_player,
    general_target_reward_scale,
    general_target_max_distance,
    general_target_min_army,
):
    """Add state-aware pressure shaping toward the opponent general."""
    shaping = jax.vmap(
        lambda prior_state, state: general_target_reward_fn(
            prior_state,
            state,
            learner_player,
            general_target_reward_scale,
            general_target_max_distance,
            general_target_min_army,
        )
    )(prior_states, states)
    return rewards + shaping


def apply_path_assignment_rewards(
    rewards,
    prior_states,
    states,
    learner_player,
    path_assignment_reward_scale,
    path_assignment_max_distance,
    path_assignment_min_army,
    path_assignment_general_weight,
    path_assignment_city_weight,
    path_assignment_frontier_weight,
):
    """Add shortest-path target-assignment shaping for transport progress."""
    if path_assignment_reward_scale <= 0.0:
        return rewards

    shaping = jax.vmap(
        lambda prior_state, state: path_assignment_reward_fn(
            prior_state,
            state,
            learner_player,
            path_assignment_reward_scale,
            path_assignment_max_distance,
            path_assignment_min_army,
            path_assignment_general_weight,
            path_assignment_city_weight,
            path_assignment_frontier_weight,
        )
    )(prior_states, states)
    return rewards + shaping


@eqx.filter_jit
def rollout_step(
    states,
    pool,
    network,
    key,
    truncation,
    opponent_id,
    learner_player,
    terminal_reward_scale,
    policy_input=0,
    general_target_reward_scale=0.0,
    general_target_max_distance=16,
    general_target_min_army=2,
    path_assignment_reward_scale=0.0,
    path_assignment_max_distance=64,
    path_assignment_min_army=2,
    path_assignment_general_weight=1.0,
    path_assignment_city_weight=0.8,
    path_assignment_frontier_weight=0.25,
):
    """Vectorized rollout step for all environments."""
    num_envs = states.armies.shape[0]
    
    # Observations (BEFORE step for reward calculation)
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = select_learner_obs(obs_p0_prior, obs_p1_prior, learner_player)
    opponent_obs_prior = select_opponent_obs(obs_p0_prior, obs_p1_prior, learner_player)
    
    # Actions from network
    obs_arr, masks = jax.vmap(
        lambda state, obs: policy_input_array_and_mask(state, obs, learner_player, policy_input)
    )(states, learner_obs_prior)
    
    key, *keys = jrandom.split(key, num_envs + 1)
    learner_actions, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )
    
    # Opponent actions for the non-learner player.
    key, *keys = jrandom.split(key, num_envs + 1)
    opponent_actions = jax.vmap(lambda k, o: opponent_action(opponent_id, k, o, random_action))(
        jnp.stack(keys), opponent_obs_prior
    )
    
    # Step game
    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)
    
    # Get new observations (AFTER step)
    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    obs_p1_new = jax.vmap(lambda s: game.get_observation(s, 1))(new_states)
    learner_obs_new = select_learner_obs(obs_p0_new, obs_p1_new, learner_player)
    
    # Compute rewards using composite reward function
    rewards = jax.vmap(composite_reward_fn)(
        learner_obs_prior, learner_actions, learner_obs_new
    )
    rewards = apply_general_target_rewards(
        rewards,
        states,
        new_states,
        learner_player,
        general_target_reward_scale,
        general_target_max_distance,
        general_target_min_army,
    )
    rewards = apply_path_assignment_rewards(
        rewards,
        states,
        new_states,
        learner_player,
        path_assignment_reward_scale,
        path_assignment_max_distance,
        path_assignment_min_army,
        path_assignment_general_weight,
        path_assignment_city_weight,
        path_assignment_frontier_weight,
    )
    rewards = apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale)
    
    # Terminated/truncated
    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated

    # Auto-reset from a pre-generated pool. This keeps rollout_step fast and
    # lets the raw trainer use complex generated maps without regenerating them
    # inside every environment step.
    pool_size = pool.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda x: x[reset_indices], pool)
    next_pool_idx = jnp.where(dones, new_states.pool_idx + num_envs, new_states.pool_idx)
    reset_states = reset_states._replace(pool_idx=next_pool_idx)
    current_states = new_states._replace(pool_idx=next_pool_idx)

    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        current_states,
    )
    
    return final_states, (obs_arr, masks, learner_actions, logprobs, values, rewards, dones, infos), key


@eqx.filter_jit
def rollout_step_policy_opponent(
    states,
    pool,
    network,
    opponent_network,
    key,
    truncation,
    opponent_policy_mode,
    learner_player,
    terminal_reward_scale,
    policy_input=0,
    opponent_policy_input=0,
    general_target_reward_scale=0.0,
    general_target_max_distance=16,
    general_target_min_army=2,
    path_assignment_reward_scale=0.0,
    path_assignment_max_distance=64,
    path_assignment_min_army=2,
    path_assignment_general_weight=1.0,
    path_assignment_city_weight=0.8,
    path_assignment_frontier_weight=0.25,
):
    """Vectorized rollout step against a frozen policy checkpoint opponent."""
    num_envs = states.armies.shape[0]
    opponent_player = 1 - learner_player

    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = select_learner_obs(obs_p0_prior, obs_p1_prior, learner_player)
    opponent_obs_prior = select_opponent_obs(obs_p0_prior, obs_p1_prior, learner_player)

    obs_arr, masks = jax.vmap(
        lambda state, obs: policy_input_array_and_mask(state, obs, learner_player, policy_input)
    )(states, learner_obs_prior)

    key, *keys = jrandom.split(key, num_envs + 1)
    learner_actions, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )

    key, *keys = jrandom.split(key, num_envs + 1)
    opponent_actions = jax.vmap(
        lambda state, k, obs: policy_state_action(
            opponent_network,
            k,
            state,
            obs,
            opponent_player,
            opponent_policy_mode,
            opponent_policy_input,
        )
    )(
        states,
        jnp.stack(keys),
        opponent_obs_prior,
    )

    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)

    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    obs_p1_new = jax.vmap(lambda s: game.get_observation(s, 1))(new_states)
    learner_obs_new = select_learner_obs(obs_p0_new, obs_p1_new, learner_player)
    rewards = jax.vmap(composite_reward_fn)(learner_obs_prior, learner_actions, learner_obs_new)
    rewards = apply_general_target_rewards(
        rewards,
        states,
        new_states,
        learner_player,
        general_target_reward_scale,
        general_target_max_distance,
        general_target_min_army,
    )
    rewards = apply_path_assignment_rewards(
        rewards,
        states,
        new_states,
        learner_player,
        path_assignment_reward_scale,
        path_assignment_max_distance,
        path_assignment_min_army,
        path_assignment_general_weight,
        path_assignment_city_weight,
        path_assignment_frontier_weight,
    )
    rewards = apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale)

    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated

    pool_size = pool.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda x: x[reset_indices], pool)
    next_pool_idx = jnp.where(dones, new_states.pool_idx + num_envs, new_states.pool_idx)
    reset_states = reset_states._replace(pool_idx=next_pool_idx)
    current_states = new_states._replace(pool_idx=next_pool_idx)

    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        current_states,
    )

    return final_states, (obs_arr, masks, learner_actions, logprobs, values, rewards, dones, infos), key


@eqx.filter_jit
def rollout_step_policy_pool_opponent(
    states,
    pool,
    network,
    opponent_networks,
    opponent_modes,
    opponent_indices,
    key,
    truncation,
    learner_player,
    terminal_reward_scale,
    policy_input=0,
    opponent_policy_input=0,
    general_target_reward_scale=0.0,
    general_target_max_distance=16,
    general_target_min_army=2,
    path_assignment_reward_scale=0.0,
    path_assignment_max_distance=64,
    path_assignment_min_army=2,
    path_assignment_general_weight=1.0,
    path_assignment_city_weight=0.8,
    path_assignment_frontier_weight=0.25,
):
    """Vectorized rollout step against a pool of frozen policy checkpoint opponents."""
    num_envs = states.armies.shape[0]
    opponent_player = 1 - learner_player

    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = select_learner_obs(obs_p0_prior, obs_p1_prior, learner_player)
    opponent_obs_prior = select_opponent_obs(obs_p0_prior, obs_p1_prior, learner_player)

    obs_arr, masks = jax.vmap(
        lambda state, obs: policy_input_array_and_mask(state, obs, learner_player, policy_input)
    )(states, learner_obs_prior)

    key, *keys = jrandom.split(key, num_envs + 1)
    learner_actions, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )

    key, *keys = jrandom.split(key, num_envs + 1)
    opponent_actions = jax.vmap(
        lambda opponent_index, state, k, obs: policy_pool_state_action(
            opponent_networks,
            opponent_modes,
            opponent_index,
            k,
            state,
            obs,
            opponent_player,
            opponent_policy_input,
        )
    )(
        opponent_indices,
        states,
        jnp.stack(keys),
        opponent_obs_prior,
    )

    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)

    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    obs_p1_new = jax.vmap(lambda s: game.get_observation(s, 1))(new_states)
    learner_obs_new = select_learner_obs(obs_p0_new, obs_p1_new, learner_player)
    rewards = jax.vmap(composite_reward_fn)(learner_obs_prior, learner_actions, learner_obs_new)
    rewards = apply_general_target_rewards(
        rewards,
        states,
        new_states,
        learner_player,
        general_target_reward_scale,
        general_target_max_distance,
        general_target_min_army,
    )
    rewards = apply_path_assignment_rewards(
        rewards,
        states,
        new_states,
        learner_player,
        path_assignment_reward_scale,
        path_assignment_max_distance,
        path_assignment_min_army,
        path_assignment_general_weight,
        path_assignment_city_weight,
        path_assignment_frontier_weight,
    )
    rewards = apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale)

    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated

    pool_size = pool.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda x: x[reset_indices], pool)
    next_pool_idx = jnp.where(dones, new_states.pool_idx + num_envs, new_states.pool_idx)
    reset_states = reset_states._replace(pool_idx=next_pool_idx)
    current_states = new_states._replace(pool_idx=next_pool_idx)

    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        current_states,
    )

    return final_states, (obs_arr, masks, learner_actions, logprobs, values, rewards, dones, infos), key


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and value returns."""
    num_steps, num_envs = rewards.shape
    values_with_bootstrap = jnp.concatenate([values, jnp.zeros((1, num_envs))], axis=0)

    def gae_step(carry, inputs):
        last_adv = carry
        reward, value, next_value, done = inputs
        nonterminal = 1.0 - done
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * lam * nonterminal * last_adv
        return advantage, advantage

    inputs = (
        rewards[::-1],
        values[::-1],
        values_with_bootstrap[1:][::-1],
        dones[::-1],
    )
    _, advantages_rev = jax.lax.scan(gae_step, jnp.zeros(num_envs), inputs)
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


@jax.jit
def ppo_loss(network, obs, mask, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for single sample."""
    _, value, logprob, entropy = network(obs, mask, None, action)
    
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    
    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy
    
    return policy_loss + value_loss + entropy_loss


@eqx.filter_jit
def train_step(network, opt_state, batch, optimizer):
    """Single training step."""
    obs, masks, actions, old_logprobs, advantages, returns = batch
    
    # Flatten batch
    bs = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(bs, *obs.shape[2:])
    masks_flat = masks.reshape(bs, *masks.shape[2:])
    actions_flat = actions.reshape(bs, -1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)
    
    def loss_fn(net):
        losses = jax.vmap(lambda o, m, a, olp, adv, r: ppo_loss(net, o, m, a, olp, adv, r))(
            obs_flat, masks_flat, actions_flat, old_logprobs_flat, advantages_flat, returns_flat
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)

    return network, opt_state, loss


@eqx.filter_jit
def train_minibatch_step(network, opt_state, minibatch, optimizer):
    """Single PPO update for an already-flattened minibatch."""
    obs, masks, actions, old_logprobs, advantages, returns = minibatch

    def loss_fn(net):
        losses = jax.vmap(lambda o, m, a, olp, adv, r: ppo_loss(net, o, m, a, olp, adv, r))(
            obs,
            masks,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)

    return network, opt_state, loss


def flatten_training_batch(batch):
    """Flatten rollout time/environment axes into a single PPO sample axis."""
    obs, masks, actions, old_logprobs, advantages, returns = batch
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        actions.reshape(batch_size, -1),
        old_logprobs.reshape(batch_size),
        advantages.reshape(batch_size),
        returns.reshape(batch_size),
    )


def train_epoch(network, opt_state, batch, optimizer, key, num_epochs=1, minibatch_size=None):
    """Run one or more PPO epochs over a rollout batch with optional minibatching."""
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
            network, opt_state, loss = train_minibatch_step(network, opt_state, minibatch, optimizer)
            epoch_loss += loss

        avg_loss = epoch_loss / num_complete_batches

    return network, opt_state, avg_loss, key


def main():
    parser = argparse.ArgumentParser(description="Train the experimental raw-game JAX PPO agent.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout steps per PPO iteration.")
    parser.add_argument("--num-iterations", type=int, default=50, help="Number of PPO iterations.")
    parser.add_argument("--num-epochs", type=int, default=1, help="PPO epochs per rollout batch.")
    parser.add_argument("--minibatch-size", type=int, default=None, help="Minibatch size for PPO updates.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--grid-size", type=int, default=4, help="Square map size used by the policy network.")
    parser.add_argument("--truncation", type=int, default=500, help="Maximum game steps before an auto-reset.")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random", help="Player-1 training opponent.")
    parser.add_argument(
        "--learner-player",
        type=int,
        choices=(0, 1),
        default=0,
        help="Environment player slot controlled by the learner.",
    )
    parser.add_argument(
        "--terminal-reward-scale",
        type=float,
        default=0.0,
        help="Optional win/loss reward added on decisive terminal transitions.",
    )
    parser.add_argument(
        "--general-target-reward-scale",
        type=float,
        default=0.0,
        help="Optional shaping reward for reducing distance from strong owned cells to the opponent general.",
    )
    parser.add_argument(
        "--general-target-max-distance",
        type=int,
        default=None,
        help="Distance horizon for --general-target-reward-scale. Defaults to the map Manhattan diameter.",
    )
    parser.add_argument(
        "--general-target-min-army",
        type=int,
        default=2,
        help="Minimum army count for owned cells that count as pressure toward the opponent general.",
    )
    parser.add_argument(
        "--path-assignment-reward-scale",
        type=float,
        default=0.0,
        help="Optional shortest-path target-assignment shaping reward for transport progress.",
    )
    parser.add_argument(
        "--path-assignment-max-distance",
        type=int,
        default=None,
        help="Distance horizon for --path-assignment-reward-scale. Defaults to grid_size squared.",
    )
    parser.add_argument(
        "--path-assignment-min-army",
        type=int,
        default=2,
        help="Minimum army count for owned cells that receive path-assignment transport shaping.",
    )
    parser.add_argument(
        "--path-assignment-general-weight",
        type=float,
        default=1.0,
        help="Target-assignment weight for the enemy general shortest-path field.",
    )
    parser.add_argument(
        "--path-assignment-city-weight",
        type=float,
        default=0.8,
        help="Target-assignment weight for non-owned city shortest-path fields.",
    )
    parser.add_argument(
        "--path-assignment-frontier-weight",
        type=float,
        default=0.25,
        help="Target-assignment weight for nearest non-owned passable frontier cells.",
    )
    parser.add_argument(
        "--policy-input",
        choices=POLICY_INPUT_NAMES,
        default="observation",
        help="Input encoding used by the learner policy.",
    )
    parser.add_argument("--input-channels", type=int, default=None, help="Learner network input channels.")
    parser.add_argument(
        "--init-input-channels",
        type=int,
        default=None,
        help="Warm-start checkpoint input channels before optional learner input expansion.",
    )
    parser.add_argument(
        "--opponent-policy-path",
        default=None,
        help="Optional frozen PPO checkpoint to use as the player-1 opponent instead of --opponent.",
    )
    parser.add_argument(
        "--opponent-policy-pool",
        default=None,
        help="Comma-separated frozen PPO checkpoints to sample as ordinary policy opponents.",
    )
    parser.add_argument(
        "--opponent-policy-pool-modes",
        default=None,
        help="Comma-separated execution modes for --opponent-policy-pool. Defaults to sample for every checkpoint.",
    )
    parser.add_argument(
        "--self-play-opponent",
        action="store_true",
        help="Use the current learner policy as the non-learner opponent on each rollout.",
    )
    parser.add_argument(
        "--opponent-policy-mode",
        choices=POLICY_MODE_NAMES,
        default="sample",
        help="Execution mode for --opponent-policy-path.",
    )
    parser.add_argument(
        "--opponent-policy-input",
        choices=POLICY_INPUT_NAMES,
        default=None,
        help="Input encoding used by a policy opponent. Defaults to observation for frozen checkpoints and learner input for current self-play.",
    )
    parser.add_argument("--opponent-input-channels", type=int, default=None, help="Frozen opponent network input channels.")
    parser.add_argument("--pool-size", type=int, default=2048, help="Number of pre-generated reset states.")
    parser.add_argument(
        "--map-generator",
        choices=("simple", "generated"),
        default="simple",
        help="Use simple empty maps or generated maps with mountains/cities.",
    )
    parser.add_argument("--mountain-density-min", type=float, default=0.18, help="Generated-map minimum mountain density.")
    parser.add_argument("--mountain-density-max", type=float, default=0.24, help="Generated-map maximum mountain density.")
    parser.add_argument("--num-cities-min", type=int, default=9, help="Generated-map minimum number of cities.")
    parser.add_argument("--num-cities-max", type=int, default=11, help="Generated-map maximum number of cities.")
    parser.add_argument(
        "--min-generals-distance",
        type=int,
        default=None,
        help="Generated-map minimum Manhattan distance between generals.",
    )
    parser.add_argument(
        "--max-generals-distance",
        type=int,
        default=None,
        help="Generated-map maximum Manhattan distance between generals.",
    )
    parser.add_argument("--city-army-min", type=int, default=40, help="Generated city minimum starting army.")
    parser.add_argument("--city-army-max", type=int, default=51, help="Generated city maximum starting army.")
    parser.add_argument(
        "--channels",
        default=None,
        help="Policy network channels as four comma-separated integers, for example 64,64,64,32.",
    )
    parser.add_argument(
        "--opponent-channels",
        default=None,
        help="Frozen checkpoint opponent channels. Defaults to --channels when omitted.",
    )
    parser.add_argument("--init-model-path", default=None, help="Optional checkpoint to warm-start PPO from.")
    parser.add_argument("--model-path", default="jax_ppo_model.eqx", help="Path where the trained model is saved.")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional directory for periodic training checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save a periodic checkpoint every N iterations.")
    parser.add_argument("--keep-checkpoints", type=int, default=0, help="Keep only the newest N periodic checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Training PRNG seed.")
    args = parser.parse_args()

    num_envs = args.num_envs
    num_steps = args.num_steps
    num_iterations = args.num_iterations
    lr = args.lr
    grid_size = args.grid_size
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, grid_size // 2)

    if grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.pool_size < num_envs:
        parser.error("--pool-size must be at least num_envs")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.minibatch_size is not None and args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive when provided")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative")
    if args.keep_checkpoints < 0:
        parser.error("--keep-checkpoints cannot be negative")
    if args.terminal_reward_scale < 0.0:
        parser.error("--terminal-reward-scale must be non-negative")
    if args.general_target_reward_scale < 0.0:
        parser.error("--general-target-reward-scale must be non-negative")
    if args.general_target_max_distance is not None and args.general_target_max_distance <= 0:
        parser.error("--general-target-max-distance must be positive when provided")
    if args.general_target_min_army <= 0:
        parser.error("--general-target-min-army must be positive")
    general_target_max_distance = args.general_target_max_distance or max(1, 2 * (grid_size - 1))
    if args.path_assignment_reward_scale < 0.0:
        parser.error("--path-assignment-reward-scale must be non-negative")
    if args.path_assignment_max_distance is not None and args.path_assignment_max_distance <= 0:
        parser.error("--path-assignment-max-distance must be positive when provided")
    if args.path_assignment_min_army <= 0:
        parser.error("--path-assignment-min-army must be positive")
    if args.path_assignment_general_weight < 0.0:
        parser.error("--path-assignment-general-weight must be non-negative")
    if args.path_assignment_city_weight < 0.0:
        parser.error("--path-assignment-city-weight must be non-negative")
    if args.path_assignment_frontier_weight < 0.0:
        parser.error("--path-assignment-frontier-weight must be non-negative")
    if (
        args.path_assignment_reward_scale > 0.0
        and args.path_assignment_general_weight + args.path_assignment_city_weight + args.path_assignment_frontier_weight <= 0.0
    ):
        parser.error("at least one path-assignment target weight must be positive when reward scale is enabled")
    path_assignment_max_distance = args.path_assignment_max_distance or max(1, grid_size * grid_size)
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if args.opponent_input_channels is not None and args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    try:
        opponent_policy_pool = parse_opponent_policy_pool(args.opponent_policy_pool, args.opponent_policy_pool_modes)
        opponent_source = resolve_opponent_source(args.opponent_policy_path, args.self_play_opponent, opponent_policy_pool)
    except ValueError as exc:
        parser.error(str(exc))
    input_channels = args.input_channels or policy_input_default_channels(args.policy_input)
    init_input_channels = args.init_input_channels
    if init_input_channels is None and args.init_model_path is not None and input_channels != 9:
        init_input_channels = 9
    opponent_policy_input_name = args.opponent_policy_input
    if opponent_source == "current":
        if opponent_policy_input_name is not None and opponent_policy_input_name != args.policy_input:
            parser.error("--opponent-policy-input must match --policy-input when using --self-play-opponent")
        if args.opponent_input_channels is not None and args.opponent_input_channels != input_channels:
            parser.error("--opponent-input-channels must match --input-channels when using --self-play-opponent")
        opponent_policy_input_name = args.policy_input
        opponent_input_channels = input_channels
    else:
        opponent_policy_input_name = opponent_policy_input_name or "observation"
        opponent_input_channels = args.opponent_input_channels or policy_input_default_channels(opponent_policy_input_name)
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    opponent_policy_input = POLICY_INPUT_NAME_TO_ID[opponent_policy_input_name]
    try:
        channels = parse_policy_channels(args.channels)
        opponent_channels = parse_policy_channels(args.opponent_channels if args.opponent_channels is not None else args.channels)
    except ValueError as exc:
        parser.error(str(exc))
    
    print("JAX PPO (Raw Game API - Max Performance)")
    print(f"Environments:  {num_envs}")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Learner:       player {args.learner_player}")
    print(f"Policy input:  {args.policy_input} ({input_channels} channels)")
    if opponent_source == "heuristic":
        print(f"Opponent:      {args.opponent}")
    elif opponent_source == "checkpoint":
        print(f"Opponent:      policy checkpoint ({args.opponent_policy_mode})")
        print(f"Opponent path: {args.opponent_policy_path}")
        print(f"Opponent ch:   {opponent_channels}")
        print(f"Opponent in:   {opponent_policy_input_name} ({opponent_input_channels} channels)")
    elif opponent_source == "checkpoint_pool":
        print(f"Opponent:      policy checkpoint pool ({len(opponent_policy_pool)} checkpoints)")
        for pool_index, (pool_path, pool_mode) in enumerate(opponent_policy_pool):
            print(f"  pool[{pool_index}]: {pool_mode} {pool_path}")
        print(f"Opponent ch:   {opponent_channels}")
        print(f"Opponent in:   {opponent_policy_input_name} ({opponent_input_channels} channels)")
    else:
        print(f"Opponent:      current policy self-play ({args.opponent_policy_mode})")
    print(f"Grid:          {grid_size}x{grid_size} ({args.map_generator}, truncation={args.truncation})")
    print(f"Channels:      {channels}")
    if args.map_generator == "generated":
        print(f"Mountains:     {args.mountain_density_min:.2f}-{args.mountain_density_max:.2f}")
        print(f"Cities:        {args.num_cities_min}-{args.num_cities_max}")
        print(f"General dist:  min={min_generals_distance}, max={args.max_generals_distance}")
    print(f"Reset pool:    {args.pool_size}")
    print(f"PPO updates:   epochs={args.num_epochs}, minibatch={args.minibatch_size or num_envs * num_steps}")
    if args.terminal_reward_scale > 0.0:
        print(f"Terminal rw:   +/-{args.terminal_reward_scale:g}")
    if args.general_target_reward_scale > 0.0:
        print(
            "General target:"
            f" scale={args.general_target_reward_scale:g}, "
            f"max_dist={general_target_max_distance}, min_army={args.general_target_min_army}"
        )
    if args.path_assignment_reward_scale > 0.0:
        print(
            "Path assign:  "
            f"scale={args.path_assignment_reward_scale:g}, max_dist={path_assignment_max_distance}, "
            f"min_army={args.path_assignment_min_army}, "
            f"weights=general:{args.path_assignment_general_weight:g}/"
            f"city:{args.path_assignment_city_weight:g}/frontier:{args.path_assignment_frontier_weight:g}"
        )
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
    if args.checkpoint_dir is not None and args.checkpoint_every > 0:
        print(f"Checkpoints:   every {args.checkpoint_every} iterations in {args.checkpoint_dir}")
    print()
    
    # Initialize
    key = jrandom.PRNGKey(args.seed)
    key, net_key, opponent_net_key = jrandom.split(key, 3)
    network = load_or_create_network(
        net_key,
        grid_size=grid_size,
        init_model_path=args.init_model_path,
        channels=channels,
        input_channels=input_channels,
        init_input_channels=init_input_channels,
    )
    opponent_network = None
    opponent_networks = ()
    opponent_pool_modes = jnp.array([], dtype=jnp.int32)
    if opponent_source == "checkpoint":
        opponent_network = load_or_create_network(
            opponent_net_key,
            grid_size=grid_size,
            init_model_path=args.opponent_policy_path,
            channels=opponent_channels,
            input_channels=opponent_input_channels,
        )
    elif opponent_source == "checkpoint_pool":
        opponent_networks, opponent_pool_modes = load_opponent_policy_pool(
            opponent_net_key,
            opponent_policy_pool,
            grid_size=grid_size,
            channels=opponent_channels,
            input_channels=opponent_input_channels,
        )
    optimizer = optax.adam(lr)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]

    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    key, pool_key = jrandom.split(key)
    pool = make_state_pool(
        pool_key,
        args.pool_size,
        grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    jax.block_until_ready(pool.armies)
    states = make_initial_states(pool, num_envs)
    
    print("\nWarming up...")
    for _ in range(3):
        if opponent_source == "heuristic":
            states, _, key = rollout_step(
                states,
                pool,
                network,
                key,
                args.truncation,
                opponent_id,
                args.learner_player,
                args.terminal_reward_scale,
                policy_input,
                args.general_target_reward_scale,
                general_target_max_distance,
                args.general_target_min_army,
                args.path_assignment_reward_scale,
                path_assignment_max_distance,
                args.path_assignment_min_army,
                args.path_assignment_general_weight,
                args.path_assignment_city_weight,
                args.path_assignment_frontier_weight,
            )
        elif opponent_source in ("checkpoint", "current"):
            active_opponent_network = network if opponent_source == "current" else opponent_network
            states, _, key = rollout_step_policy_opponent(
                states,
                pool,
                network,
                active_opponent_network,
                key,
                args.truncation,
                opponent_policy_mode,
                args.learner_player,
                args.terminal_reward_scale,
                policy_input,
                opponent_policy_input,
                args.general_target_reward_scale,
                general_target_max_distance,
                args.general_target_min_army,
                args.path_assignment_reward_scale,
                path_assignment_max_distance,
                args.path_assignment_min_army,
                args.path_assignment_general_weight,
                args.path_assignment_city_weight,
                args.path_assignment_frontier_weight,
            )
        else:
            key, opponent_index_key = jrandom.split(key)
            opponent_indices = jrandom.randint(
                opponent_index_key,
                (num_envs,),
                0,
                len(opponent_networks),
                dtype=jnp.int32,
            )
            states, _, key = rollout_step_policy_pool_opponent(
                states,
                pool,
                network,
                opponent_networks,
                opponent_pool_modes,
                opponent_indices,
                key,
                args.truncation,
                args.learner_player,
                args.terminal_reward_scale,
                policy_input,
                opponent_policy_input,
                args.general_target_reward_scale,
                general_target_max_distance,
                args.general_target_min_army,
                args.path_assignment_reward_scale,
                path_assignment_max_distance,
                args.path_assignment_min_army,
                args.path_assignment_general_weight,
                args.path_assignment_city_weight,
                args.path_assignment_frontier_weight,
            )
    jax.block_until_ready(states)
    
    print("Training...\n")
    saved_checkpoints = []
    checkpoint_stem = Path(args.model_path).stem
    if args.checkpoint_dir is not None:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    for iteration in range(num_iterations):
        t0 = time.time()
        opponent_indices = None
        if opponent_source == "checkpoint_pool":
            key, opponent_index_key = jrandom.split(key)
            opponent_indices = jrandom.randint(
                opponent_index_key,
                (num_envs,),
                0,
                len(opponent_networks),
                dtype=jnp.int32,
            )
        
        # Collect rollout
        rollout_data = []
        for _ in range(num_steps):
            if opponent_source == "heuristic":
                states, data, key = rollout_step(
                    states,
                    pool,
                    network,
                    key,
                    args.truncation,
                    opponent_id,
                    args.learner_player,
                    args.terminal_reward_scale,
                    policy_input,
                    args.general_target_reward_scale,
                    general_target_max_distance,
                    args.general_target_min_army,
                    args.path_assignment_reward_scale,
                    path_assignment_max_distance,
                    args.path_assignment_min_army,
                    args.path_assignment_general_weight,
                    args.path_assignment_city_weight,
                    args.path_assignment_frontier_weight,
                )
            elif opponent_source in ("checkpoint", "current"):
                active_opponent_network = network if opponent_source == "current" else opponent_network
                states, data, key = rollout_step_policy_opponent(
                    states,
                    pool,
                    network,
                    active_opponent_network,
                    key,
                    args.truncation,
                    opponent_policy_mode,
                    args.learner_player,
                    args.terminal_reward_scale,
                    policy_input,
                    opponent_policy_input,
                    args.general_target_reward_scale,
                    general_target_max_distance,
                    args.general_target_min_army,
                    args.path_assignment_reward_scale,
                    path_assignment_max_distance,
                    args.path_assignment_min_army,
                    args.path_assignment_general_weight,
                    args.path_assignment_city_weight,
                    args.path_assignment_frontier_weight,
                )
            else:
                states, data, key = rollout_step_policy_pool_opponent(
                    states,
                    pool,
                    network,
                    opponent_networks,
                    opponent_pool_modes,
                    opponent_indices,
                    key,
                    args.truncation,
                    args.learner_player,
                    args.terminal_reward_scale,
                    policy_input,
                    opponent_policy_input,
                    args.general_target_reward_scale,
                    general_target_max_distance,
                    args.general_target_min_army,
                    args.path_assignment_reward_scale,
                    path_assignment_max_distance,
                    args.path_assignment_min_army,
                    args.path_assignment_general_weight,
                    args.path_assignment_city_weight,
                    args.path_assignment_frontier_weight,
                )
            rollout_data.append(data)
        jax.block_until_ready(states)
        
        # Stack data
        obs = jnp.stack([d[0] for d in rollout_data])
        masks = jnp.stack([d[1] for d in rollout_data])
        actions = jnp.stack([d[2] for d in rollout_data])
        logprobs = jnp.stack([d[3] for d in rollout_data])
        values = jnp.stack([d[4] for d in rollout_data])
        rewards = jnp.stack([d[5] for d in rollout_data])
        dones = jnp.stack([d[6] for d in rollout_data])
        infos_list = [d[7] for d in rollout_data]
        infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos_list)
        
        # Compute advantages
        advantages, returns = compute_gae(rewards, values, dones)
        policy_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train
        batch = (obs, masks, actions, logprobs, policy_advantages, returns)
        key, update_key = jrandom.split(key)
        network, opt_state, loss, key = train_epoch(
            network,
            opt_state,
            batch,
            optimizer,
            update_key,
            args.num_epochs,
            args.minibatch_size,
        )
        jax.block_until_ready(network)
        
        elapsed = time.time() - t0
        
        if iteration % 10 == 0:
            avg_reward = rewards.mean()
            num_episodes = int(dones.sum())
            wins = int(jnp.sum((dones) & (infos.winner == args.learner_player)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (num_envs * num_steps) / elapsed
            print(f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                  f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                  f"Wins: {wins:2d}/{num_episodes} ({win_rate:.0f}%) | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s")

        completed_iteration = iteration + 1
        if (
            args.checkpoint_dir is not None
            and args.checkpoint_every > 0
            and completed_iteration % args.checkpoint_every == 0
        ):
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, checkpoint_stem, completed_iteration)
            eqx.tree_serialise_leaves(checkpoint_path, network)
            saved_checkpoints.append(checkpoint_path)
            prune_old_checkpoints(saved_checkpoints, args.keep_checkpoints)
            saved_checkpoints = [path for path in saved_checkpoints if path.exists()]
    
    print("\nTraining complete!")
    
    # Save model
    model_path = args.model_path
    eqx.tree_serialise_leaves(model_path, network)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
