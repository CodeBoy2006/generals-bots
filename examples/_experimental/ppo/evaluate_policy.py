"""Batch evaluation for experimental PPO policy checkpoints."""

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

from generals.core import game

from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, greedy_policy_action, make_grids, opponent_action, sampled_policy_action
from network import PolicyValueNetwork
from train import random_action


@eqx.filter_jit
def evaluate_batch(network, states, key, max_steps, opponent, policy_mode, policy_player):
    """Evaluate a network against Random or a heuristic on a batch of states."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)

        key, k0, k1 = jrandom.split(key, 3)
        policy_keys = jrandom.split(k0, num_envs)
        actions_p0 = jax.lax.cond(
            policy_mode == 0,
            lambda _: jax.vmap(lambda o: greedy_policy_action(network, o))(policy_obs),
            lambda _: jax.vmap(lambda o, k: sampled_policy_action(network, o, k))(policy_obs, policy_keys),
            None,
        )
        opponent_keys = jrandom.split(k1, num_envs)
        actions_p1 = jax.vmap(lambda k, o: opponent_action(opponent, k, o, random_action))(opponent_keys, opponent_obs)
        actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([actions_p0, actions_p1], axis=1),
            lambda _: jnp.stack([actions_p1, actions_p0], axis=1),
            None,
        )

        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new), states, new_states)
        return (final_states, key), infos

    (states, key), _ = jax.lax.scan(body, (states, key), None, length=max_steps)
    info = jax.vmap(game.get_info)(states)
    return info


def summarize_policy_results(info, policy_player, num_games):
    """Return scalar outcome metrics from the policy player's perspective."""
    opponent_player = 1 - policy_player
    wins = int(jnp.sum(info.winner == policy_player))
    losses = int(jnp.sum(info.winner == opponent_player))
    draws = int(jnp.sum(info.winner < 0))
    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "decisive_win_rate": wins / max(decisive, 1),
        "draw_rate": draws / num_games,
        "mean_time": float(jnp.mean(info.time)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an experimental PPO policy checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--policy-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def main():
    args = parse_args()
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)

    key = jrandom.PRNGKey(args.seed)
    key, net_key, map_key, eval_key = jrandom.split(key, 4)
    network = PolicyValueNetwork(net_key, grid_size=args.grid_size)
    network = eqx.tree_deserialise_leaves(args.model_path, network)

    grids = make_grids(
        map_key,
        args.num_games,
        args.grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    states = jax.vmap(game.create_initial_state)(grids)

    opponent_code = OPPONENT_NAME_TO_ID[args.opponent]
    policy_mode = 0 if args.policy_mode == "greedy" else 1
    t0 = time.time()
    info = evaluate_batch(network, states, eval_key, args.max_steps, opponent_code, policy_mode, args.policy_player)
    jax.block_until_ready(info.winner)
    elapsed = time.time() - t0

    summary = summarize_policy_results(info, args.policy_player, args.num_games)

    print("Policy evaluation")
    print(f"Model:              {args.model_path}")
    print(f"Device:             {jax.devices()[0]}")
    print(f"Grid:               {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Opponent:           {args.opponent}")
    print(f"Policy mode:        {args.policy_mode}")
    print(f"Policy player:      {args.policy_player}")
    print(f"Games:              {args.num_games}")
    print(f"Max steps:          {args.max_steps}")
    print(f"Wins/Losses/Draws:  {summary['wins']}/{summary['losses']}/{summary['draws']}")
    print(f"Win rate:           {summary['win_rate']:.4f}")
    print(f"Decisive win rate:  {summary['decisive_win_rate']:.4f}")
    print(f"Draw rate:          {summary['draw_rate']:.4f}")
    print(f"Mean final time:    {summary['mean_time']:.1f}")
    print(f"Eval seconds:       {elapsed:.2f}")


if __name__ == "__main__":
    main()
