"""Raw-game PPO trainer for adaptive multisize policy checkpoints."""

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
    adaptive_obs_to_array,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, opponent_action
from generals.core import game
from generals.core.rewards import composite_reward_fn
from train import (
    apply_terminal_reward,
    checkpoint_path_for_iteration,
    compute_gae,
    prune_old_checkpoints,
    random_action,
    stack_learner_actions,
)


def apply_truncation_reward(rewards, truncated, scale):
    """Penalize non-terminal timeout rows without changing decisive games."""
    return rewards - jnp.where(truncated, scale, 0.0)


def resolve_learner_player(value: str, iteration: int) -> int:
    """Resolve fixed or alternating learner seat for one training iteration."""
    if value == "alternate":
        return (iteration - 1) % 2
    return int(value)


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
    terminal_reward_scale,
    truncation_reward_scale,
    pad_size,
):
    """Collect one vectorized adaptive PPO rollout step."""
    num_envs = states.armies.shape[0]
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p0_prior, lambda _: obs_p1_prior, None)
    opponent_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p1_prior, lambda _: obs_p0_prior, None)

    obs_arr, active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
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

    key, opponent_key = jrandom.split(key)
    opponent_keys = jrandom.split(opponent_key, num_envs)
    opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent_id, k, obs, random_action))(
        opponent_keys,
        opponent_obs_prior,
    )

    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)

    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    obs_p1_new = jax.vmap(lambda s: game.get_observation(s, 1))(new_states)
    learner_obs_new = jax.lax.cond(learner_player == 0, lambda _: obs_p0_new, lambda _: obs_p1_new, None)
    rewards = jax.vmap(composite_reward_fn)(learner_obs_prior, learner_actions, learner_obs_new)

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
    return final_states, final_sizes, (obs_arr, masks, active, learner_actions, logprobs, values, rewards, dones, infos), key


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
    terminal_reward_scale,
    truncation_reward_scale,
    pad_size,
):
    """Collect a Python-loop rollout, stacking step data on axis 0."""
    step_data = []
    for _ in range(num_steps):
        states, effective_sizes, data, key = rollout_step(
            states,
            effective_sizes,
            pool,
            network,
            key,
            truncation,
            opponent_id,
            learner_player,
            terminal_reward_scale,
            truncation_reward_scale,
            pad_size,
        )
        step_data.append(data)
    rollout_data = jax.tree.map(lambda *xs: jnp.stack(xs), *step_data)
    return states, effective_sizes, rollout_data, key


@jax.jit
def ppo_loss(network, obs, mask, active, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for one adaptive sample."""
    _, value, logprob, entropy = network(obs, mask, active, None, action)
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy
    return policy_loss + value_loss + entropy_loss


@eqx.filter_jit
def train_minibatch_step(network, opt_state, minibatch, optimizer):
    """Run one PPO update on a flattened adaptive minibatch."""
    obs, masks, active, actions, old_logprobs, advantages, returns = minibatch

    def loss_fn(net):
        losses = jax.vmap(lambda o, m, ac, a, olp, adv, r: ppo_loss(net, o, m, ac, a, olp, adv, r))(
            obs,
            masks,
            active,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss


def flatten_training_batch(batch):
    """Flatten rollout time/environment axes into one sample axis."""
    obs, masks, active, actions, old_logprobs, advantages, returns = batch
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        actions.reshape(batch_size, -1),
        old_logprobs.reshape(batch_size),
        advantages.reshape(batch_size),
        returns.reshape(batch_size),
    )


def train_epoch(network, opt_state, batch, optimizer, key, num_epochs=1, minibatch_size=None):
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
            network, opt_state, loss = train_minibatch_step(network, opt_state, minibatch, optimizer)
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
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--truncation", type=int, default=750)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--learner-player", choices=("0", "1", "alternate"), default="0")
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
    parser.add_argument("--channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="/tmp/generals-adaptive-ppo.eqx")
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
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_envs <= 0:
        parser.error("num_envs must be positive")
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
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
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
    learner_label = "alternate players 0/1" if args.learner_player == "alternate" else f"player {args.learner_player}"
    print(f"Learner:       {learner_label}")
    print(f"Opponent:      {args.opponent}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Iterations:    {args.num_iterations} x {args.num_steps} steps")
    print(f"PPO updates:   epochs={args.num_epochs}, minibatch={args.minibatch_size or args.num_envs * args.num_steps}")
    if args.truncation_reward_scale > 0.0:
        print(f"Timeout reward: -{args.truncation_reward_scale:g}")
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
        if args.init_channels is not None:
            print(f"Warm channels: {args.init_channels}")
    if args.checkpoint_dir is not None and args.checkpoint_every > 0:
        print(f"Checkpoints:   every {args.checkpoint_every} iterations in {args.checkpoint_dir}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, net_key, pool_key = jrandom.split(key, 3)
    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.init_model_path,
        channels=args.channels,
        init_channels=args.init_channels,
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]

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
    states, effective_sizes = make_adaptive_initial_states(pool, args.num_envs)

    print("Warming up...")
    warmup_learner_player = resolve_learner_player(args.learner_player, 1)
    states, effective_sizes, _, key = rollout_step(
        states,
        effective_sizes,
        pool,
        network,
        key,
        args.truncation,
        opponent_id,
        warmup_learner_player,
        args.terminal_reward_scale,
        args.truncation_reward_scale,
        args.pad_to,
    )
    jax.block_until_ready(states)

    checkpoint_paths = []
    model_stem = Path(args.model_path).stem
    for iteration in range(1, args.num_iterations + 1):
        t0 = time.time()
        iteration_learner_player = resolve_learner_player(args.learner_player, iteration)
        states, effective_sizes, rollout_data, key = collect_rollout(
            states,
            effective_sizes,
            pool,
            network,
            key,
            args.num_steps,
            args.truncation,
            opponent_id,
            iteration_learner_player,
            args.terminal_reward_scale,
            args.truncation_reward_scale,
            args.pad_to,
        )
        jax.block_until_ready(states)
        obs, masks, active, actions, logprobs, values, rewards, dones, infos = rollout_data
        advantages, returns = compute_gae(rewards, values, dones)
        policy_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch = (obs, masks, active, actions, logprobs, policy_advantages, returns)
        key, train_key = jrandom.split(key)
        network, opt_state, loss, key = train_epoch(
            network,
            opt_state,
            batch,
            optimizer,
            train_key,
            args.num_epochs,
            args.minibatch_size,
        )
        jax.block_until_ready(network)

        if args.checkpoint_dir is not None and args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, model_stem, iteration)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(checkpoint_path, network)
            checkpoint_paths.append(checkpoint_path)
            prune_old_checkpoints(checkpoint_paths, args.keep_checkpoints)

        if iteration % 10 == 0 or iteration == 1 or iteration == args.num_iterations:
            elapsed = time.time() - t0
            episodes = int(jnp.sum(dones))
            wins = int(jnp.sum(dones & (infos.winner == iteration_learner_player)))
            draws = int(jnp.sum(dones & (infos.winner < 0)))
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Episodes: {episodes:4d} | Wins: {wins:4d} | Draws: {draws:4d} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s"
            )

    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
