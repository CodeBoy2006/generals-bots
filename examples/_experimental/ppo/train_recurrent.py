"""Train an experimental residual-GRU PPO policy."""

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
from recurrent_network import RecurrentPolicyValueNetwork
from train import (
    apply_general_target_rewards,
    apply_path_assignment_rewards,
    apply_terminal_reward,
    compute_gae,
    load_or_create_network,
    make_initial_states,
    make_state_pool,
    random_action,
    select_learner_obs,
    select_opponent_obs,
    stack_learner_actions,
)
from generals.agents.ppo_policy_agent import PolicyValueNetwork, parse_policy_channels
from generals.core import game
from generals.core.rewards import composite_reward_fn


def load_or_create_recurrent_network(
    key,
    grid_size,
    init_model_path=None,
    init_recurrent_model_path=None,
    channels=None,
    input_channels=9,
    init_input_channels=None,
    hidden_size=64,
):
    """Create a recurrent network, optionally warm-starting the CNN base from v5."""
    parsed_channels = parse_policy_channels(channels)
    base_key, recurrent_key = jrandom.split(key)
    base_network = load_or_create_network(
        base_key,
        grid_size=grid_size,
        init_model_path=init_model_path,
        channels=parsed_channels,
        input_channels=input_channels,
        init_input_channels=init_input_channels,
    )
    network = RecurrentPolicyValueNetwork(
        recurrent_key,
        grid_size=grid_size,
        channels=parsed_channels,
        input_channels=input_channels,
        hidden_size=hidden_size,
        base_network=base_network,
    )
    if init_recurrent_model_path is None:
        return network

    path = Path(init_recurrent_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Recurrent checkpoint not found: {path}")
    return eqx.tree_deserialise_leaves(path, network)


@eqx.filter_jit
def rollout_step_recurrent_heuristic_opponent(
    states,
    pool,
    hidden,
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
    """Vectorized recurrent rollout step against random or heuristic opponents."""
    num_envs = states.armies.shape[0]

    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = select_learner_obs(obs_p0_prior, obs_p1_prior, learner_player)
    opponent_obs_prior = select_opponent_obs(obs_p0_prior, obs_p1_prior, learner_player)

    obs_arr, masks = jax.vmap(
        lambda state, obs: policy_input_array_and_mask(state, obs, learner_player, policy_input)
    )(states, learner_obs_prior)

    key, *keys = jrandom.split(key, num_envs + 1)
    learner_actions, values, logprobs, entropies, next_hidden = jax.vmap(
        network,
        in_axes=(0, 0, 0, 0, None),
    )(obs_arr, masks, hidden, jnp.stack(keys), None)

    key, *keys = jrandom.split(key, num_envs + 1)
    opponent_actions = jax.vmap(lambda action_key, obs: opponent_action(opponent_id, action_key, obs, random_action))(
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
    final_hidden = jnp.where(dones[:, None], jnp.zeros_like(next_hidden), next_hidden)

    return (
        final_states,
        final_hidden,
        (obs_arr, masks, hidden, learner_actions, logprobs, values, rewards, dones, infos),
        key,
    )


@eqx.filter_jit
def rollout_step_recurrent_policy_opponent(
    states,
    pool,
    hidden,
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
    """Vectorized recurrent rollout step against a frozen policy checkpoint."""
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
    learner_actions, values, logprobs, entropies, next_hidden = jax.vmap(
        network,
        in_axes=(0, 0, 0, 0, None),
    )(obs_arr, masks, hidden, jnp.stack(keys), None)

    key, *keys = jrandom.split(key, num_envs + 1)
    opponent_actions = jax.vmap(
        lambda state, action_key, obs: policy_state_action(
            opponent_network,
            action_key,
            state,
            obs,
            opponent_player,
            opponent_policy_mode,
            opponent_policy_input,
        )
    )(states, jnp.stack(keys), opponent_obs_prior)

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
    final_hidden = jnp.where(dones[:, None], jnp.zeros_like(next_hidden), next_hidden)

    return (
        final_states,
        final_hidden,
        (obs_arr, masks, hidden, learner_actions, logprobs, values, rewards, dones, infos),
        key,
    )


@jax.jit
def recurrent_ppo_loss(network, obs, mask, hidden, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for one recurrent sample with rollout hidden treated as input."""
    _, value, logprob, entropy, _ = network(obs, mask, hidden, None, action)
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy
    return policy_loss + value_loss + entropy_loss


def flatten_recurrent_training_batch(batch):
    """Flatten rollout time/environment axes for recurrent PPO updates."""
    obs, masks, hiddens, actions, old_logprobs, advantages, returns = batch
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        hiddens.reshape(batch_size, hiddens.shape[-1]),
        actions.reshape(batch_size, -1),
        old_logprobs.reshape(batch_size),
        advantages.reshape(batch_size),
        returns.reshape(batch_size),
    )


@eqx.filter_jit
def train_recurrent_minibatch_step(network, opt_state, minibatch, optimizer, freeze_base=False):
    """Single PPO update for an already-flattened recurrent minibatch."""
    obs, masks, hiddens, actions, old_logprobs, advantages, returns = minibatch

    def loss_fn(net):
        losses = jax.vmap(lambda o, m, h, a, olp, adv, r: recurrent_ppo_loss(net, o, m, h, a, olp, adv, r))(
            obs,
            masks,
            hiddens,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    if freeze_base:
        zero_base_grads = jax.tree.map(
            lambda grad: None if grad is None else jnp.zeros_like(grad),
            grads.base,
        )
        grads = eqx.tree_at(lambda tree: tree.base, grads, zero_base_grads)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, loss


def train_recurrent_epoch(network, opt_state, batch, optimizer, key, num_epochs=1, minibatch_size=None, freeze_base=False):
    """Run PPO epochs over recurrent rollout samples."""
    flat_batch = flatten_recurrent_training_batch(batch)
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
            network, opt_state, loss = train_recurrent_minibatch_step(
                network,
                opt_state,
                minibatch,
                optimizer,
                freeze_base,
            )
            epoch_loss += loss
        avg_loss = epoch_loss / num_complete_batches

    return network, opt_state, avg_loss, key


def main():
    parser = argparse.ArgumentParser(description="Train an experimental residual-GRU PPO agent.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--truncation", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze the CNN base and train only recurrent memory/delta heads.",
    )
    parser.add_argument("--learner-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--terminal-reward-scale", type=float, default=0.0)
    parser.add_argument("--general-target-reward-scale", type=float, default=0.0)
    parser.add_argument("--general-target-max-distance", type=int, default=None)
    parser.add_argument("--general-target-min-army", type=int, default=2)
    parser.add_argument("--path-assignment-reward-scale", type=float, default=0.0)
    parser.add_argument("--path-assignment-max-distance", type=int, default=None)
    parser.add_argument("--path-assignment-min-army", type=int, default=2)
    parser.add_argument("--path-assignment-general-weight", type=float, default=1.0)
    parser.add_argument("--path-assignment-city-weight", type=float, default=0.8)
    parser.add_argument("--path-assignment-frontier-weight", type=float, default=0.25)
    parser.add_argument("--policy-input", choices=POLICY_INPUT_NAMES, default="observation")
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--init-input-channels", type=int, default=None)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-input", choices=POLICY_INPUT_NAMES, default="observation")
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--init-recurrent-model-path", default=None)
    parser.add_argument("--model-path", default="jax_recurrent_ppo_model.eqx")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.minibatch_size is not None and args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive when provided")
    if args.terminal_reward_scale < 0.0:
        parser.error("--terminal-reward-scale must be non-negative")
    if args.general_target_reward_scale < 0.0:
        parser.error("--general-target-reward-scale must be non-negative")
    if args.general_target_min_army <= 0:
        parser.error("--general-target-min-army must be positive")
    if args.path_assignment_reward_scale < 0.0:
        parser.error("--path-assignment-reward-scale must be non-negative")
    if args.path_assignment_min_army <= 0:
        parser.error("--path-assignment-min-army must be positive")
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")

    grid_size = args.grid_size
    min_generals_distance = args.min_generals_distance or max(3, grid_size // 2)
    general_target_max_distance = args.general_target_max_distance or max(1, 2 * (grid_size - 1))
    path_assignment_max_distance = args.path_assignment_max_distance or max(1, grid_size * grid_size)
    input_channels = args.input_channels or policy_input_default_channels(args.policy_input)
    init_input_channels = args.init_input_channels
    if init_input_channels is None and args.init_model_path is not None and input_channels != 9:
        init_input_channels = 9
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    opponent_policy_input = POLICY_INPUT_NAME_TO_ID[args.opponent_policy_input]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    try:
        channels = parse_policy_channels(args.channels)
        opponent_channels = parse_policy_channels(args.opponent_channels if args.opponent_channels is not None else args.channels)
    except ValueError as exc:
        parser.error(str(exc))

    print("JAX Recurrent PPO (Residual GRU)")
    print(f"Environments:  {args.num_envs}")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Learner:       player {args.learner_player}")
    print(f"Policy input:  {args.policy_input} ({input_channels} channels)")
    print(f"Hidden size:   {args.hidden_size}")
    if args.freeze_base:
        print("Base:          frozen")
    if args.opponent_policy_path is None:
        print(f"Opponent:      {args.opponent}")
    else:
        print(f"Opponent path: {args.opponent_policy_path}")
    print(f"Grid:          {grid_size}x{grid_size} ({args.map_generator}, truncation={args.truncation})")
    print(f"Reset pool:    {args.pool_size}")
    print(f"PPO updates:   epochs={args.num_epochs}, minibatch={args.minibatch_size or args.num_envs * args.num_steps}")
    if args.init_model_path is not None:
        print(f"Base warm:     {args.init_model_path}")
    if args.init_recurrent_model_path is not None:
        print(f"RNN warm:      {args.init_recurrent_model_path}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, net_key, opponent_net_key, pool_key = jrandom.split(key, 4)
    network = load_or_create_recurrent_network(
        net_key,
        grid_size=grid_size,
        init_model_path=args.init_model_path,
        init_recurrent_model_path=args.init_recurrent_model_path,
        channels=channels,
        input_channels=input_channels,
        init_input_channels=init_input_channels,
        hidden_size=args.hidden_size,
    )
    opponent_network = None
    if args.opponent_policy_path is not None:
        opponent_network = PolicyValueNetwork(
            opponent_net_key,
            grid_size=grid_size,
            channels=opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_network)
    optimizer = optax.adam(args.lr)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")

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
    states = make_initial_states(pool, args.num_envs)
    hidden = jnp.zeros((args.num_envs, args.hidden_size), dtype=jnp.float32)

    print("\nWarming up...")
    for _ in range(3):
        if opponent_network is None:
            states, hidden, _, key = rollout_step_recurrent_heuristic_opponent(
                states,
                pool,
                hidden,
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
        else:
            states, hidden, _, key = rollout_step_recurrent_policy_opponent(
                states,
                pool,
                hidden,
                network,
                opponent_network,
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
    jax.block_until_ready(states)

    print("Training...\n")
    for iteration in range(args.num_iterations):
        t0 = time.time()
        rollout_data = []
        for _ in range(args.num_steps):
            if opponent_network is None:
                states, hidden, data, key = rollout_step_recurrent_heuristic_opponent(
                    states,
                    pool,
                    hidden,
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
            else:
                states, hidden, data, key = rollout_step_recurrent_policy_opponent(
                    states,
                    pool,
                    hidden,
                    network,
                    opponent_network,
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
            rollout_data.append(data)
        jax.block_until_ready(states)

        obs = jnp.stack([d[0] for d in rollout_data])
        masks = jnp.stack([d[1] for d in rollout_data])
        hiddens = jnp.stack([d[2] for d in rollout_data])
        actions = jnp.stack([d[3] for d in rollout_data])
        logprobs = jnp.stack([d[4] for d in rollout_data])
        values = jnp.stack([d[5] for d in rollout_data])
        rewards = jnp.stack([d[6] for d in rollout_data])
        dones = jnp.stack([d[7] for d in rollout_data])
        infos_list = [d[8] for d in rollout_data]
        infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos_list)

        advantages, returns = compute_gae(rewards, values, dones)
        policy_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch = (obs, masks, hiddens, actions, logprobs, policy_advantages, returns)
        key, update_key = jrandom.split(key)
        network, opt_state, loss, key = train_recurrent_epoch(
            network,
            opt_state,
            batch,
            optimizer,
            update_key,
            args.num_epochs,
            args.minibatch_size,
            args.freeze_base,
        )
        jax.block_until_ready(network)

        if iteration % 10 == 0:
            elapsed = time.time() - t0
            num_episodes = int(dones.sum())
            wins = int(jnp.sum(dones & (infos.winner == args.learner_player)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (args.num_envs * args.num_steps) / elapsed
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Reward: {float(rewards.mean()):+.4f} | Episodes: {num_episodes:3d} | "
                f"Wins: {wins:2d}/{num_episodes} ({win_rate:.0f}%) | "
                f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s"
            )

    print("\nTraining complete!")
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
