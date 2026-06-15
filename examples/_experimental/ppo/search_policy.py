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
from generals.core.action import compute_valid_move_mask
from generals.core.game import GameInfo
from generals.core.observation import Observation

from common import POLICY_MODE_NAME_TO_ID, POLICY_MODE_NAMES, make_grids, policy_network_action
from network import PolicyValueNetwork, obs_to_array
from generals.agents.ppo_policy_agent import index_to_action


def score_observation(
    info: GameInfo,
    obs: Observation,
    player: int,
    army_weight: float = 12.0,
    land_weight: float = 8.0,
    terminal_score: float = 1000.0,
):
    """Score a final rollout observation from one player's perspective."""
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


@eqx.filter_jit
def rollout_search_candidates(
    network,
    state,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
):
    """Return top-k policy-prior candidates and their rollout-search scores."""
    obs = game.get_observation(state, player)
    obs_arr = obs_to_array(obs)
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    logits, _ = network.logits_value(obs_arr, mask)
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    grid_size = obs.armies.shape[-1]
    candidate_actions = jax.vmap(lambda idx: index_to_action(idx, grid_size))(candidate_indices)

    opponent_player = 1 - player
    opponent_obs = game.get_observation(state, opponent_player)
    key, opponent_key = jrandom.split(key)
    opponent_first_action = policy_network_action(network, opponent_key, opponent_obs, policy_mode)

    def rollout_score(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, step_key = carry
            step_key, k0, k1 = jrandom.split(step_key, 3)
            obs_p0 = game.get_observation(rollout_state, 0)
            obs_p1 = game.get_observation(rollout_state, 1)
            action_p0 = policy_network_action(network, k0, obs_p0, policy_mode)
            action_p1 = policy_network_action(network, k1, obs_p1, policy_mode)
            next_state, _ = game.step(rollout_state, jnp.stack([action_p0, action_p1]))
            already_done = game.get_info(rollout_state).is_done
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            return (final_state, step_key), None

        (final_state, _), _ = jax.lax.scan(body, (initial_state, rollout_key), None, length=rollout_steps)
        final_info = game.get_info(final_state)
        final_obs = game.get_observation(final_state, player)
        return score_observation(final_info, final_obs, player, army_weight, land_weight)

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
            1000.0,
            jnp.where(first_info.winner == opponent_player, -1000.0, 0.0),
        )
        return first_terminal + jnp.mean(scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(candidate_score)(candidate_actions, prior_scores, candidate_keys)
    return candidate_actions, candidate_indices, prior_scores, scores


@eqx.filter_jit
def rollout_search_action(
    network,
    state,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
):
    """Choose one action by scoring top-k policy-prior actions with short rollouts."""
    candidate_actions, _, _, scores = rollout_search_candidates(
        network,
        state,
        key,
        player,
        top_k,
        rollout_steps,
        rollouts_per_action,
        policy_mode,
        army_weight,
        land_weight,
        prior_weight,
    )
    return candidate_actions[jnp.argmax(scores)]


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


def summarize(info, player, num_games):
    opponent = 1 - player
    wins = int(jnp.sum(info.winner == player))
    losses = int(jnp.sum(info.winner == opponent))
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
    summary = summarize(info, args.search_player, args.num_games)

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
