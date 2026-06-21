"""Behavior cloning warm-start for adaptive multisize PPO policies."""

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
    adaptive_action_to_index,
    adaptive_action_to_target_probs,
    adaptive_expander_target_probs,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import TEACHER_NAME_TO_ID, TEACHER_NAMES, heuristic_action
from generals.core import game
from train import checkpoint_path_for_iteration, prune_old_checkpoints, random_action


@eqx.filter_jit
def collect_teacher_batch(states, effective_sizes, pool, key, steps, truncation, teacher_id, pad_size):
    """Roll out teacher-vs-random games and collect adaptive BC labels."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, effective_sizes, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)

        key, teacher_key, random_key = jrandom.split(key, 3)
        teacher_keys = jrandom.split(teacher_key, num_envs)
        random_keys = jrandom.split(random_key, num_envs)

        if teacher_id == 0:
            targets = jax.vmap(lambda o, s: adaptive_expander_target_probs(o, s, pad_size))(obs_p0, effective_sizes)
            teacher_indices = jax.vmap(lambda k, p: jrandom.categorical(k, jnp.log(p + 1e-8)))(
                teacher_keys,
                targets,
            )
            actions_p0 = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(teacher_indices)
        else:
            actions_p0 = jax.vmap(lambda k, o: heuristic_action(teacher_id - 1, k, o))(teacher_keys, obs_p0)
            teacher_indices = jax.vmap(lambda action: adaptive_action_to_index(action, pad_size))(actions_p0)
            targets = jax.vmap(lambda action: adaptive_action_to_target_probs(action, pad_size))(actions_p0)

        actions_p1 = jax.vmap(random_action)(random_keys, obs_p1)
        new_states, infos = jax.vmap(game.step)(states, jnp.stack([actions_p0, actions_p1], axis=1))

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

        obs_arr, active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
            obs_p0,
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
        )(obs_p0, effective_sizes)
        return (final_states, final_sizes, key), (obs_arr, masks, active, targets, teacher_indices, dones, infos.winner)

    (states, effective_sizes, key), batch = jax.lax.scan(body, (states, effective_sizes, key), None, length=steps)
    return states, effective_sizes, batch, key


@eqx.filter_jit
def train_bc_step(network, opt_state, obs, masks, active, targets, teacher_indices, optimizer):
    """Train one adaptive behavior-cloning batch."""
    batch_size = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(batch_size, *obs.shape[2:])
    masks_flat = masks.reshape(batch_size, *masks.shape[2:])
    active_flat = active.reshape(batch_size, *active.shape[2:])
    targets_flat = targets.reshape(batch_size, targets.shape[-1])
    teacher_indices_flat = teacher_indices.reshape(batch_size)

    def loss_fn(net):
        logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs_flat, masks_flat, active_flat)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.sum(targets_flat * log_probs, axis=-1)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == teacher_indices_flat)
        return jnp.mean(losses), accuracy

    (loss, accuracy), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-clone an adaptive multisize PPO policy.")
    parser.add_argument("num_envs", nargs="?", type=int, default=512)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--teacher", choices=TEACHER_NAMES, default="expander-soft")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--truncation", type=int, default=500)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/generals-adaptive-bc-8-12-16.eqx")
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
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_envs <= 0:
        parser.error("num_envs must be positive")
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if args.num_iterations <= 0:
        parser.error("--num-iterations must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
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

    print("Adaptive behavior cloning from heuristic teacher")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Teacher:       {args.teacher}")
    print(f"Environments:  {args.num_envs}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Iterations:    {args.num_iterations} x {args.num_steps} steps")
    print(f"Reset pool:    {args.pool_size}")
    if args.channels is not None:
        print(f"Channels:      {args.channels}")
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
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
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))

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

    checkpoint_paths = []
    model_stem = Path(args.model_path).stem
    for iteration in range(args.num_iterations):
        t0 = time.time()
        states, effective_sizes, batch, key = collect_teacher_batch(
            states,
            effective_sizes,
            pool,
            key,
            args.num_steps,
            args.truncation,
            TEACHER_NAME_TO_ID[args.teacher],
            args.pad_to,
        )
        obs, masks, active, targets, teacher_indices, dones, winners = batch
        network, opt_state, loss, accuracy = train_bc_step(
            network,
            opt_state,
            obs,
            masks,
            active,
            targets,
            teacher_indices,
            optimizer,
        )
        jax.block_until_ready(network)

        iteration_number = iteration + 1
        if (
            args.checkpoint_dir is not None
            and args.checkpoint_every > 0
            and iteration_number % args.checkpoint_every == 0
        ):
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, model_stem, iteration_number)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(checkpoint_path, network)
            checkpoint_paths.append(checkpoint_path)
            prune_old_checkpoints(checkpoint_paths, args.keep_checkpoints)

        if iteration % 10 == 0 or iteration == args.num_iterations - 1:
            episodes = int(dones.sum())
            wins = int(jnp.sum(dones & (winners == 0)))
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Acc: {float(accuracy) * 100:5.1f}% | "
                f"Episodes: {episodes:4d} | Teacher wins: {wins:4d} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s"
            )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
