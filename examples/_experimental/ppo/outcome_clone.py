"""Outcome-conditioned cloning for PPO policy checkpoints.

This auxiliary trainer rolls out full games against a frozen checkpoint and
clones the actions taken from the final winner's perspective. It is intended as
an expert-iteration style complement to PPO when sparse win/loss credit is too
delayed for short rollout windows.
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

from generals.core import game
from generals.core.action import compute_valid_move_mask

from common import POLICY_MODE_NAME_TO_ID, POLICY_MODE_NAMES, action_to_index, make_grids, policy_network_action
from network import PolicyValueNetwork, obs_to_array
from train import load_or_create_network, stack_learner_actions

SAMPLE_SOURCE_NAMES = ("both", "learner", "opponent")
SAMPLE_SOURCE_NAME_TO_ID = {name: idx for idx, name in enumerate(SAMPLE_SOURCE_NAMES)}


def _broadcast_winner_mask(mask, target_ndim):
    """Broadcast a per-game mask across time and feature dimensions."""
    return mask.reshape((1, mask.shape[0]) + (1,) * (target_ndim - 2))


def _sample_source_mask(winners, learner_player, sample_source):
    """Return which decisive games are eligible for supervised outcome samples."""
    return jax.lax.switch(
        sample_source,
        (
            lambda _: winners >= 0,
            lambda _: winners == learner_player,
            lambda _: winners == 1 - learner_player,
        ),
        None,
    )


def choose_winner_trajectories(
    obs_p0,
    obs_p1,
    masks_p0,
    masks_p1,
    actions_p0,
    actions_p1,
    active,
    winners,
    learner_player=0,
    sample_source=0,
):
    """Select the final winner's perspective for each game and mask draws."""
    winner_is_p1 = winners == 1
    decisive = _sample_source_mask(winners, learner_player, sample_source)

    obs = jnp.where(_broadcast_winner_mask(winner_is_p1, obs_p0.ndim), obs_p1, obs_p0)
    masks = jnp.where(_broadcast_winner_mask(winner_is_p1, masks_p0.ndim), masks_p1, masks_p0)
    actions = jnp.where(_broadcast_winner_mask(winner_is_p1, actions_p0.ndim), actions_p1, actions_p0)
    weights = active.astype(jnp.float32) * decisive.reshape((1, decisive.shape[0])).astype(jnp.float32)
    return obs, masks, actions, weights


def choose_loser_trajectories(
    obs_p0,
    obs_p1,
    masks_p0,
    masks_p1,
    actions_p0,
    actions_p1,
    active,
    winners,
    learner_player=0,
    sample_source=0,
):
    """Select the final loser's perspective for each decisive game."""
    loser_is_p1 = winners == 0
    decisive = _sample_source_mask(winners, learner_player, sample_source)

    obs = jnp.where(_broadcast_winner_mask(loser_is_p1, obs_p0.ndim), obs_p1, obs_p0)
    masks = jnp.where(_broadcast_winner_mask(loser_is_p1, masks_p0.ndim), masks_p1, masks_p0)
    actions = jnp.where(_broadcast_winner_mask(loser_is_p1, actions_p0.ndim), actions_p1, actions_p0)
    weights = active.astype(jnp.float32) * decisive.reshape((1, decisive.shape[0])).astype(jnp.float32)
    return obs, masks, actions, weights


@eqx.filter_jit
def collect_outcome_batch(
    network,
    opponent_network,
    states,
    key,
    max_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    sample_source,
):
    """Roll out full games and return winner-perspective supervised samples."""
    num_envs = states.armies.shape[0]
    grid_size = states.armies.shape[-1]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        masks_p0 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
        masks_p1 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p1)

        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)

        key, learner_key, opponent_key = jrandom.split(key, 3)
        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(lambda k, o: policy_network_action(network, k, o, policy_mode))(
            learner_keys, learner_obs
        )
        opponent_actions = jax.vmap(lambda k, o: policy_network_action(opponent_network, k, o, opponent_policy_mode))(
            opponent_keys, opponent_obs
        )
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        new_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            new_states,
        )

        return (final_states, key), (
            jax.vmap(obs_to_array)(obs_p0),
            masks_p0,
            actions[:, 0],
            jax.vmap(obs_to_array)(obs_p1),
            masks_p1,
            actions[:, 1],
            active,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=max_steps)
    obs_p0, masks_p0, actions_p0, obs_p1, masks_p1, actions_p1, active = batch
    info = jax.vmap(game.get_info)(states)
    obs, masks, actions, weights = choose_winner_trajectories(
        obs_p0,
        obs_p1,
        masks_p0,
        masks_p1,
        actions_p0,
        actions_p1,
        active,
        info.winner,
        learner_player,
        sample_source,
    )
    loser_obs, loser_masks, loser_actions, _ = choose_loser_trajectories(
        obs_p0,
        obs_p1,
        masks_p0,
        masks_p1,
        actions_p0,
        actions_p1,
        active,
        info.winner,
        learner_player,
        sample_source,
    )
    target_indices = jax.vmap(jax.vmap(lambda action: action_to_index(action, grid_size)))(actions)
    loser_target_indices = jax.vmap(jax.vmap(lambda action: action_to_index(action, grid_size)))(loser_actions)
    return obs, masks, target_indices, loser_obs, loser_masks, loser_target_indices, weights, info, key


def flatten_outcome_batch(obs, masks, target_indices, loser_obs, loser_masks, loser_target_indices, weights):
    """Flatten time/environment axes for supervised minibatch training."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        target_indices.reshape(batch_size),
        loser_obs.reshape(batch_size, *loser_obs.shape[2:]),
        loser_masks.reshape(batch_size, *loser_masks.shape[2:]),
        loser_target_indices.reshape(batch_size),
        weights.reshape(batch_size),
    )


@eqx.filter_jit
def train_sparse_bc_minibatch(network, opt_state, minibatch, optimizer, negative_weight):
    """Train one weighted sparse cross-entropy minibatch."""
    obs, masks, target_indices, loser_obs, loser_masks, loser_target_indices, weights = minibatch

    def loss_fn(net):
        def logits_for_sample(sample_obs, sample_mask):
            logits, _ = net.logits_value(sample_obs, sample_mask)
            return logits

        logits = jax.vmap(logits_for_sample)(obs, masks)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(log_probs, target_indices[:, None], axis=1)[:, 0]
        normalizer = jnp.maximum(jnp.sum(weights), 1.0)
        positive_loss = jnp.sum(losses * weights) / normalizer

        loser_logits = jax.vmap(logits_for_sample)(loser_obs, loser_masks)
        loser_probs = jax.nn.softmax(loser_logits, axis=-1)
        loser_action_probs = jnp.take_along_axis(loser_probs, loser_target_indices[:, None], axis=1)[:, 0]
        negative_losses = -jnp.log(jnp.clip(1.0 - loser_action_probs, 1e-6, 1.0))
        negative_loss = jnp.sum(negative_losses * weights) / normalizer

        loss = positive_loss + negative_weight * negative_loss
        accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == target_indices) * weights) / normalizer
        return loss, accuracy

    (loss, accuracy), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, loss, accuracy


def train_sparse_bc_epoch(network, opt_state, flat_batch, optimizer, key, num_epochs, minibatch_size, negative_weight):
    """Run sparse behavior cloning over shuffled minibatches."""
    batch_size = flat_batch[0].shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_accuracy = 0.0

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = tuple(x[permutation] for x in flat_batch)
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            network, opt_state, loss, accuracy = train_sparse_bc_minibatch(
                network, opt_state, minibatch, optimizer, negative_weight
            )
            epoch_loss += loss
            epoch_accuracy += accuracy

        avg_loss = epoch_loss / num_complete_batches
        avg_accuracy = epoch_accuracy / num_complete_batches

    return network, opt_state, avg_loss, avg_accuracy, key


def parse_args():
    parser = argparse.ArgumentParser(description="Clone winner trajectories from policy-vs-policy outcomes.")
    parser.add_argument("num_envs", nargs="?", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=500, help="Maximum steps per full-game rollout.")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=0.0,
        help="Optional weight for lowering the final loser's taken action probability.",
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-path", required=True)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--learner-player", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--winner-source",
        choices=SAMPLE_SOURCE_NAMES,
        default="both",
        help="Which decisive games contribute winner-perspective samples.",
    )
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/generals-outcome-clone.eqx")
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
    if args.negative_weight < 0.0:
        parser.error("--negative-weight must be non-negative")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    return args


def main():
    args = parse_args()
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)

    print("Outcome-conditioned cloning")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Learner:       player {args.learner_player} ({args.policy_mode})")
    print(f"Opponent:      {args.opponent_policy_path} ({args.opponent_policy_mode})")
    print(f"Winner source: {args.winner_source}")
    print(f"Environments:  {args.num_envs}")
    print(f"Grid:          {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Rollout:       {args.num_iterations} x {args.num_steps} steps")
    print(f"Updates:       epochs={args.num_epochs}, minibatch={args.minibatch_size}, lr={args.lr:g}")
    if args.negative_weight > 0.0:
        print(f"Negative loss: weight={args.negative_weight:g}")
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, net_key, opponent_key = jrandom.split(key, 3)
    network = load_or_create_network(net_key, grid_size=args.grid_size, init_model_path=args.init_model_path)
    opponent_network = load_or_create_network(
        opponent_key,
        grid_size=args.grid_size,
        init_model_path=args.opponent_policy_path,
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    sample_source = SAMPLE_SOURCE_NAME_TO_ID[args.winner_source]

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
        (
            obs,
            masks,
            target_indices,
            loser_obs,
            loser_masks,
            loser_target_indices,
            weights,
            info,
            rollout_key,
        ) = collect_outcome_batch(
            network,
            opponent_network,
            states,
            rollout_key,
            args.num_steps,
            policy_mode,
            opponent_policy_mode,
            args.learner_player,
            sample_source,
        )
        flat_batch = flatten_outcome_batch(
            obs,
            masks,
            target_indices,
            loser_obs,
            loser_masks,
            loser_target_indices,
            weights,
        )
        network, opt_state, loss, accuracy, update_key = train_sparse_bc_epoch(
            network,
            opt_state,
            flat_batch,
            optimizer,
            update_key,
            args.num_epochs,
            args.minibatch_size,
            args.negative_weight,
        )
        jax.block_until_ready(network)

        if iteration % 5 == 0 or iteration == args.num_iterations - 1:
            elapsed = time.time() - t0
            p0_wins = int(jnp.sum(info.winner == 0))
            p1_wins = int(jnp.sum(info.winner == 1))
            draws = int(jnp.sum(info.winner < 0))
            decisive = p0_wins + p1_wins
            learner_wins = p0_wins if args.learner_player == 0 else p1_wins
            valid_samples = int(jnp.sum(weights))
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Acc: {float(accuracy) * 100:5.1f}% | "
                f"Learner wins: {learner_wins:4d}/{args.num_envs} | "
                f"Decisive: {decisive:4d} | Draws: {draws:4d} | "
                f"Samples: {valid_samples:7d} | SPS: {args.num_envs * args.num_steps / elapsed:7.0f} | "
                f"Time: {elapsed:.2f}s"
            )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
