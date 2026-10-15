"""Rollout-search policy improvement for PPO checkpoints.

This script keeps the checkpoint fixed and uses it as a policy prior plus a
short rollout evaluator. It is useful both as a stronger evaluation-time policy
and as a teacher for later distillation attempts.
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

from generals.core import game

from common import POLICY_MODE_NAMES, POLICY_MODE_NAME_TO_ID, make_grids, summarize_policy_results
from generals.agents.ppo_policy_agent import PolicyValueNetwork, policy_network_action
from generals.agents._ppo_search import rollout_search_action


@eqx.filter_jit
def evaluate_rollout_search_batch(
    network,
    states,
    key,
    max_steps,
    search_player,
    opponent_policy_mode,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
):
    """Evaluate rollout search as one player against the raw checkpoint policy."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        key, search_key, opponent_key = jrandom.split(key, 3)
        search_keys = jrandom.split(search_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        search_actions = jax.vmap(
            lambda state, action_key: rollout_search_action(
                network,
                state,
                action_key,
                search_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
            )
        )(states, search_keys)

        opponent_player = 1 - search_player
        opponent_obs = jax.vmap(lambda state: game.get_observation(state, opponent_player))(states)
        opponent_actions = jax.vmap(
            lambda action_key, obs: policy_network_action(network, action_key, obs, opponent_policy_mode)
        )(opponent_keys, opponent_obs)
        actions = jax.lax.cond(
            search_player == 0,
            lambda _: jnp.stack([search_actions, opponent_actions], axis=1),
            lambda _: jnp.stack([opponent_actions, search_actions], axis=1),
            None,
        )

        next_states, infos = jax.vmap(game.step)(states, actions)
        already_done = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(already_done.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            next_states,
        )
        return (final_states, key), infos

    (states, key), _ = jax.lax.scan(body, (states, key), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a rollout-search policy around a PPO checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--num-games", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--search-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
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
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.rollouts_per_action <= 0:
        parser.error("--rollouts-per-action must be positive")
    return args


def main():
    args = parse_args()
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)

    key = jrandom.PRNGKey(args.seed)
    key, network_key, map_key, eval_key = jrandom.split(key, 4)
    network = PolicyValueNetwork(network_key, grid_size=args.grid_size)
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

    t0 = time.time()
    info = evaluate_rollout_search_batch(
        network,
        states,
        eval_key,
        args.max_steps,
        args.search_player,
        POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode],
        args.top_k,
        args.rollout_steps,
        args.rollouts_per_action,
        args.army_weight,
        args.land_weight,
        args.prior_weight,
    )
    jax.block_until_ready(info.winner)
    elapsed = time.time() - t0
    summary = summarize_policy_results(info, args.search_player, args.num_games)

    print("Rollout-search policy evaluation")
    print(f"Model:              {args.model_path}")
    print(f"Device:             {jax.devices()[0]}")
    print(f"Grid:               {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Search player:      {args.search_player}")
    print(f"Opponent mode:      {args.opponent_policy_mode}")
    print(f"Games:              {args.num_games}")
    print(f"Max steps:          {args.max_steps}")
    print(f"Search:             top_k={args.top_k}, rollout_steps={args.rollout_steps}, rollouts/action={args.rollouts_per_action}")
    print(f"Wins/Losses/Draws:  {summary['wins']}/{summary['losses']}/{summary['draws']}")
    print(f"Win rate:           {summary['win_rate']:.4f}")
    print(f"Decisive win rate:  {summary['decisive_win_rate']:.4f}")
    print(f"Draw rate:          {summary['draw_rate']:.4f}")
    print(f"Mean final time:    {summary['mean_time']:.1f}")
    print(f"Eval seconds:       {elapsed:.2f}")


if __name__ == "__main__":
    main()
