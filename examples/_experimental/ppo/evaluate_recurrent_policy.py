"""Batch evaluation for experimental recurrent PPO checkpoints."""

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

from common import (
    OPPONENT_NAME_TO_ID,
    OPPONENT_NAMES,
    POLICY_INPUT_NAME_TO_ID,
    POLICY_INPUT_NAMES,
    POLICY_MODE_NAME_TO_ID,
    POLICY_MODE_NAMES,
    make_grids,
    opponent_action,
    policy_input_array_and_mask,
    policy_input_default_channels,
    policy_state_action,
)
from evaluate_policy import summarize_policy_results
from network import PolicyValueNetwork
from recurrent_network import RecurrentPolicyValueNetwork
from train import random_action, stack_learner_actions
from generals.agents.ppo_policy_agent import parse_policy_channels
from generals.core import game


@eqx.filter_jit
def evaluate_recurrent_batch(
    network,
    states,
    hidden,
    key,
    max_steps,
    opponent,
    policy_mode,
    policy_player,
    policy_input,
):
    """Evaluate a recurrent network against Random or a heuristic on a batch of states."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, hidden, key = carry
        was_done = jax.vmap(game.get_info)(states).is_done
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)

        obs_arr, masks = jax.vmap(
            lambda state, obs: policy_input_array_and_mask(state, obs, policy_player, policy_input)
        )(states, policy_obs)

        key, k0, k1 = jrandom.split(key, 3)
        policy_keys = jrandom.split(k0, num_envs)
        policy_actions, _, _, _, next_hidden = jax.vmap(
            network,
            in_axes=(0, 0, 0, 0, None),
        )(obs_arr, masks, hidden, policy_keys, None)
        opponent_keys = jrandom.split(k1, num_envs)
        opponent_actions = jax.vmap(lambda action_key, obs: opponent_action(opponent, action_key, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(was_done.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_hidden = jnp.where(was_done[:, None], hidden, next_hidden)
        return (final_states, final_hidden, key), infos

    (states, hidden, key), _ = jax.lax.scan(body, (states, hidden, key), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


@eqx.filter_jit
def evaluate_recurrent_policy_opponent_batch(
    network,
    opponent_network,
    states,
    hidden,
    key,
    max_steps,
    policy_mode,
    policy_player,
    opponent_policy_mode,
    policy_input,
    opponent_policy_input,
):
    """Evaluate a recurrent network against a frozen PPO checkpoint opponent."""
    num_envs = states.armies.shape[0]
    opponent_player = 1 - policy_player

    def body(carry, _):
        states, hidden, key = carry
        was_done = jax.vmap(game.get_info)(states).is_done
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)

        obs_arr, masks = jax.vmap(
            lambda state, obs: policy_input_array_and_mask(state, obs, policy_player, policy_input)
        )(states, policy_obs)

        key, k0, k1 = jrandom.split(key, 3)
        policy_keys = jrandom.split(k0, num_envs)
        policy_actions, _, _, _, next_hidden = jax.vmap(
            network,
            in_axes=(0, 0, 0, 0, None),
        )(obs_arr, masks, hidden, policy_keys, None)
        opponent_keys = jrandom.split(k1, num_envs)
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
        )(states, opponent_keys, opponent_obs)
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(was_done.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_hidden = jnp.where(was_done[:, None], hidden, next_hidden)
        return (final_states, final_hidden, key), infos

    (states, hidden, key), _ = jax.lax.scan(body, (states, hidden, key), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a recurrent PPO policy checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--policy-input", choices=POLICY_INPUT_NAMES, default="observation")
    parser.add_argument("--opponent-policy-path", default=None)
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-input", choices=POLICY_INPUT_NAMES, default="observation")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--opponent-channels", default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--policy-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-steps", type=int, default=500)
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
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    try:
        args.channels = parse_policy_channels(args.channels)
        args.opponent_channels = parse_policy_channels(
            args.opponent_channels if args.opponent_channels is not None else args.channels
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    min_generals_distance = args.min_generals_distance or max(3, args.grid_size // 2)
    key = jrandom.PRNGKey(args.seed)
    key, net_key, opponent_net_key, map_key, eval_key = jrandom.split(key, 5)
    input_channels = args.input_channels or policy_input_default_channels(args.policy_input)
    network = RecurrentPolicyValueNetwork(
        net_key,
        grid_size=args.grid_size,
        channels=args.channels,
        input_channels=input_channels,
        hidden_size=args.hidden_size,
    )
    network = eqx.tree_deserialise_leaves(args.model_path, network)
    opponent_network = None
    if args.opponent_policy_path is not None:
        opponent_network = PolicyValueNetwork(
            opponent_net_key,
            grid_size=args.grid_size,
            channels=args.opponent_channels,
            input_channels=args.opponent_input_channels,
        )
        opponent_network = eqx.tree_deserialise_leaves(args.opponent_policy_path, opponent_network)

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
    hidden = jnp.zeros((args.num_games, args.hidden_size), dtype=jnp.float32)
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    opponent_policy_input = POLICY_INPUT_NAME_TO_ID[args.opponent_policy_input]
    opponent_code = OPPONENT_NAME_TO_ID[args.opponent]

    t0 = time.time()
    if opponent_network is None:
        info = evaluate_recurrent_batch(
            network,
            states,
            hidden,
            eval_key,
            args.max_steps,
            opponent_code,
            policy_mode,
            args.policy_player,
            policy_input,
        )
    else:
        info = evaluate_recurrent_policy_opponent_batch(
            network,
            opponent_network,
            states,
            hidden,
            eval_key,
            args.max_steps,
            policy_mode,
            args.policy_player,
            opponent_policy_mode,
            policy_input,
            opponent_policy_input,
        )
    jax.block_until_ready(info.winner)
    elapsed = time.time() - t0
    summary = summarize_policy_results(info, args.policy_player, args.num_games)

    print("Recurrent policy evaluation")
    print(f"Model:              {args.model_path}")
    print(f"Device:             {jax.devices()[0]}")
    print(f"Grid:               {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Hidden size:        {args.hidden_size}")
    print(f"Input channels:     {input_channels}")
    if args.opponent_policy_path is None:
        print(f"Opponent:           {args.opponent}")
    else:
        print(f"Opponent model:     {args.opponent_policy_path}")
    print(f"Policy mode:        {args.policy_mode}")
    print(f"Policy input:       {args.policy_input}")
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
