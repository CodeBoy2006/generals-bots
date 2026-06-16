"""Evaluate a policy checkpoint against heuristic and checkpoint league opponents."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (
    HEURISTIC_NAMES,
    OPPONENT_NAME_TO_ID,
    POLICY_INPUT_NAME_TO_ID,
    POLICY_INPUT_NAMES,
    POLICY_MODE_NAME_TO_ID,
    POLICY_MODE_NAMES,
    make_grids,
    opponent_action,
    policy_input_default_channels,
    policy_state_action,
)
from evaluate_policy import evaluate_batch, evaluate_policy_opponent_batch, summarize_policy_results
from generals.agents.ppo_policy_agent import parse_policy_channels
from generals.core import game
from network import PolicyValueNetwork
from search_policy import rollout_search_action
from train import random_action

REQUIRED_HEURISTIC_OPPONENTS = tuple(HEURISTIC_NAMES)


@dataclass(frozen=True)
class LeagueRow:
    opponent_type: str
    opponent_name: str
    policy_player: int
    wins: int
    losses: int
    draws: int
    games: int
    mean_time: float
    required: bool

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def decisive_win_rate(self) -> float:
        decisive = self.wins + self.losses
        return self.wins / max(decisive, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games


def _default_checkpoint_name(path: str) -> str:
    return Path(path).stem


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, str, str]]:
    """Parse checkpoint opponent specs as name/path/mode tuples."""
    specs = []
    for raw in values:
        name_and_path, sep, mode = raw.rpartition(":")
        if not sep:
            name_and_path = raw
            mode = "sample"
        if mode not in POLICY_MODE_NAMES:
            raise ValueError(f"Unsupported checkpoint policy mode: {mode}")
        name, sep, path = name_and_path.partition("=")
        if not sep:
            path = name_and_path
            name = _default_checkpoint_name(path)
        specs.append((name, path, mode))
    return specs


def compute_league_summary(rows: list[LeagueRow], threshold: float) -> dict[str, object]:
    """Summarize league rows with a conservative minimum required win rate."""
    required_rows = [row for row in rows if row.required]
    passed_pairs = sum(row.win_rate >= threshold for row in required_rows)
    league_score = min((row.win_rate for row in required_rows), default=0.0)
    return {
        "required_pairs": len(required_rows),
        "passed_pairs": passed_pairs,
        "league_score": league_score,
        "threshold": threshold,
        "passes_threshold": bool(required_rows) and passed_pairs == len(required_rows),
    }


def row_to_dict(row: LeagueRow, threshold: float) -> dict[str, object]:
    """Convert one league row to JSON-ready metrics."""
    data = asdict(row)
    data["win_rate"] = row.win_rate
    data["decisive_win_rate"] = row.decisive_win_rate
    data["draw_rate"] = row.draw_rate
    data["passes_threshold"] = (not row.required) or row.win_rate >= threshold
    return data


def make_eval_states(args: argparse.Namespace, key):
    """Create the shared evaluation state batch for every league row."""
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)
    grids = make_grids(
        key,
        args.num_games,
        args.grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    return jax.vmap(game.create_initial_state)(grids)


def load_policy_network(path, key, grid_size, channels, input_channels):
    """Load one ordinary PolicyValueNetwork checkpoint."""
    network = PolicyValueNetwork(
        key,
        grid_size=grid_size,
        channels=channels,
        input_channels=input_channels,
    )
    return eqx.tree_deserialise_leaves(path, network)


def row_from_info(opponent_type, opponent_name, policy_player, info, num_games, required):
    """Create one league row from a batch evaluation result."""
    summary = summarize_policy_results(info, policy_player, num_games)
    return LeagueRow(
        opponent_type=opponent_type,
        opponent_name=opponent_name,
        policy_player=policy_player,
        wins=summary["wins"],
        losses=summary["losses"],
        draws=summary["draws"],
        games=num_games,
        mean_time=summary["mean_time"],
        required=required,
    )


def evaluate_heuristic_rows(args, network, states, key):
    """Evaluate the candidate against configured heuristic opponents."""
    rows = []
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    for opponent_name in args.heuristic:
        required = opponent_name in REQUIRED_HEURISTIC_OPPONENTS
        for policy_player in (0, 1):
            key, eval_key = jrandom.split(key)
            info = evaluate_batch(
                network,
                states,
                eval_key,
                args.max_steps,
                OPPONENT_NAME_TO_ID[opponent_name],
                policy_mode,
                policy_player,
                policy_input,
            )
            jax.block_until_ready(info.winner)
            rows.append(
                row_from_info(
                    "heuristic",
                    opponent_name,
                    policy_player,
                    info,
                    args.num_games,
                    required,
                )
            )
    return rows


def evaluate_checkpoint_rows(args, network, states, key, checkpoint_specs):
    """Evaluate the candidate against configured ordinary checkpoint opponents."""
    rows = []
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    for index, (opponent_name, opponent_path, opponent_mode_name) in enumerate(checkpoint_specs):
        opponent_network = load_policy_network(
            opponent_path,
            jrandom.fold_in(key, 1000 + index),
            args.grid_size,
            args.opponent_channels,
            args.opponent_input_channels,
        )
        opponent_mode = POLICY_MODE_NAME_TO_ID[opponent_mode_name]
        for policy_player in (0, 1):
            key, eval_key = jrandom.split(key)
            info = evaluate_policy_opponent_batch(
                network,
                opponent_network,
                states,
                eval_key,
                args.max_steps,
                policy_mode,
                policy_player,
                opponent_mode,
                policy_input,
            )
            jax.block_until_ready(info.winner)
            rows.append(
                row_from_info(
                    "checkpoint",
                    opponent_name,
                    policy_player,
                    info,
                    args.num_games,
                    True,
                )
            )
    return rows


@eqx.filter_jit
def evaluate_search_heuristic_batch(
    network,
    states,
    key,
    max_steps,
    search_player,
    opponent_id,
    opponent_policy_mode,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
):
    """Evaluate rollout-search policy against one random/heuristic opponent."""
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
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent_id, k, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )
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


@eqx.filter_jit
def evaluate_search_policy_opponent_batch(
    network,
    opponent_network,
    states,
    key,
    max_steps,
    search_player,
    opponent_policy_mode,
    opponent_policy_input,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
):
    """Evaluate rollout-search policy against one frozen ordinary policy opponent."""
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
            lambda state, action_key, obs: policy_state_action(
                opponent_network,
                action_key,
                state,
                obs,
                opponent_player,
                opponent_policy_mode,
                opponent_policy_input,
            )
        )(
            states,
            opponent_keys,
            opponent_obs,
        )
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


def evaluate_search_heuristic_rows(args, network, states, key):
    """Evaluate rollout-search policy against configured heuristic opponents."""
    rows = []
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]
    for opponent_name in args.heuristic:
        required = opponent_name in REQUIRED_HEURISTIC_OPPONENTS
        for policy_player in (0, 1):
            key, eval_key = jrandom.split(key)
            info = evaluate_search_heuristic_batch(
                network,
                states,
                eval_key,
                args.max_steps,
                policy_player,
                OPPONENT_NAME_TO_ID[opponent_name],
                opponent_policy_mode,
                args.top_k,
                args.rollout_steps,
                args.rollouts_per_action,
                args.army_weight,
                args.land_weight,
                args.prior_weight,
            )
            jax.block_until_ready(info.winner)
            rows.append(row_from_info("heuristic", opponent_name, policy_player, info, args.num_games, required))
    return rows


def evaluate_search_checkpoint_rows(args, network, states, key, checkpoint_specs):
    """Evaluate rollout-search policy against configured frozen checkpoint opponents."""
    rows = []
    opponent_policy_input = POLICY_INPUT_NAME_TO_ID["observation"]
    for index, (opponent_name, opponent_path, opponent_mode_name) in enumerate(checkpoint_specs):
        opponent_network = load_policy_network(
            opponent_path,
            jrandom.fold_in(key, 1000 + index),
            args.grid_size,
            args.opponent_channels,
            args.opponent_input_channels,
        )
        opponent_mode = POLICY_MODE_NAME_TO_ID[opponent_mode_name]
        for policy_player in (0, 1):
            key, eval_key = jrandom.split(key)
            info = evaluate_search_policy_opponent_batch(
                network,
                opponent_network,
                states,
                eval_key,
                args.max_steps,
                policy_player,
                opponent_mode,
                opponent_policy_input,
                args.top_k,
                args.rollout_steps,
                args.rollouts_per_action,
                args.army_weight,
                args.land_weight,
                args.prior_weight,
            )
            jax.block_until_ready(info.winner)
            rows.append(row_from_info("checkpoint", opponent_name, policy_player, info, args.num_games, True))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against a policy league.")
    parser.add_argument("candidate_path")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--checkpoint-opponent", action="append", default=[])
    parser.add_argument("--heuristic", action="append", default=None)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--policy-input", choices=POLICY_INPUT_NAMES, default="observation")
    parser.add_argument("--search-policy", action="store_true", help="Evaluate rollout-search actions around the checkpoint.")
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--rollouts-per-action", type=int, default=4)
    parser.add_argument("--army-weight", type=float, default=12.0)
    parser.add_argument("--land-weight", type=float, default=8.0)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument(
        "--channels",
        default=None,
        help="Policy model channels as four comma-separated integers, for example 64,64,64,32.",
    )
    parser.add_argument(
        "--opponent-channels",
        default=None,
        help="Opponent checkpoint channels. Defaults to --channels when omitted.",
    )
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--opponent-input-channels", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=5)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--require-threshold", action="store_true")
    args = parser.parse_args()

    if args.heuristic is None:
        args.heuristic = list(REQUIRED_HEURISTIC_OPPONENTS)
    if args.include_random and "random" not in args.heuristic:
        args.heuristic.append("random")
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
    try:
        args.channels = parse_policy_channels(args.channels)
        args.opponent_channels = parse_policy_channels(args.opponent_channels or args.channels)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    checkpoint_specs = parse_checkpoint_specs(args.checkpoint_opponent)
    key = jrandom.PRNGKey(args.seed)
    key, net_key, map_key = jrandom.split(key, 3)
    input_channels = args.input_channels or policy_input_default_channels(args.policy_input)
    network = load_policy_network(
        args.candidate_path,
        net_key,
        args.grid_size,
        args.channels,
        input_channels,
    )
    states = make_eval_states(args, map_key)
    t0 = time.time()
    if args.search_policy:
        rows = evaluate_search_heuristic_rows(args, network, states, key)
        rows.extend(evaluate_search_checkpoint_rows(args, network, states, key, checkpoint_specs))
        policy_kind = "rollout-search"
    else:
        rows = evaluate_heuristic_rows(args, network, states, key)
        rows.extend(evaluate_checkpoint_rows(args, network, states, key, checkpoint_specs))
        policy_kind = "checkpoint"
    summary = compute_league_summary(rows, args.threshold)
    result = {
        "candidate_path": args.candidate_path,
        "policy_kind": policy_kind,
        "threshold": args.threshold,
        "elapsed_seconds": time.time() - t0,
        "rows": [row_to_dict(row, args.threshold) for row in rows],
        "summary": summary,
    }
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.require_threshold and not summary["passes_threshold"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
