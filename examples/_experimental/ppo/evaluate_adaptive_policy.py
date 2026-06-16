"""Evaluate adaptive multisize PPO checkpoints against heuristic opponents."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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

from adaptive_common import (
    ADAPTIVE_GLOBAL_INPUT_CHANNELS,
    ADAPTIVE_HISTORY_INPUT_CHANNELS,
    ADAPTIVE_INPUT_CHANNELS,
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    make_adaptive_state_pool,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, opponent_action
from generals.core import game
from train import random_action, stack_learner_actions


@dataclass(frozen=True)
class AdaptiveEvalRow:
    grid_size: int
    policy_player: int
    wins: int
    losses: int
    draws: int
    num_games: int
    mean_time: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.num_games

    @property
    def decisive_win_rate(self) -> float:
        return self.wins / max(self.wins + self.losses, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / self.num_games

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["win_rate"] = self.win_rate
        data["decisive_win_rate"] = self.decisive_win_rate
        data["draw_rate"] = self.draw_rate
        return data


@eqx.filter_jit
def _policy_action(network, obs_arr, mask, active, key, policy_mode):
    logits, _ = network.logits_value(obs_arr, mask, active)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, network.pad_size)


def summarize_row(info, grid_size: int, policy_player: int, num_games: int) -> AdaptiveEvalRow:
    opponent_player = 1 - policy_player
    wins = jnp.sum(info.winner == policy_player)
    losses = jnp.sum(info.winner == opponent_player)
    draws = jnp.sum(info.winner < 0)
    return AdaptiveEvalRow(
        grid_size=grid_size,
        policy_player=policy_player,
        wins=wins,
        losses=losses,
        draws=draws,
        num_games=num_games,
        mean_time=jnp.mean(info.time),
    )


@eqx.filter_jit
def evaluate_batch(
    network,
    states,
    effective_size,
    key,
    max_steps,
    opponent,
    policy_mode,
    policy_player,
    pad_size,
    global_context=False,
    scoreboard_history=False,
):
    """Evaluate one adaptive checkpoint on one grid size and player seat."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)
    initial_history = jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)

    def body(carry, _):
        states, key, history = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)

        if scoreboard_history:
            current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                policy_obs,
                effective_sizes,
            )
            history_context = adaptive_scoreboard_history_context(history, current_scoreboard)
            obs_arr, active = jax.vmap(
                lambda obs, size, row_history: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=row_history,
                )
            )(
                policy_obs,
                effective_sizes,
                history_context,
            )
        else:
            current_scoreboard = history
            obs_arr, active = jax.vmap(
                lambda obs, size: adaptive_obs_to_array(obs, size, pad_size, include_global_context=global_context)
            )(
                policy_obs,
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
        )(policy_obs, effective_sizes)

        key, policy_key, opponent_key = jrandom.split(key, 3)
        policy_keys = jrandom.split(policy_key, num_envs)
        policy_actions = jax.vmap(lambda o, m, a, k: _policy_action(network, o, m, a, k, policy_mode))(
            obs_arr,
            masks,
            active,
            policy_keys,
        )
        opponent_keys = jrandom.split(opponent_key, num_envs)
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent, k, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        return (final_states, key, current_scoreboard), infos

    (states, key, _), _ = jax.lax.scan(body, (states, key, initial_history), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an adaptive multisize PPO checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--value-sigma", type=float, default=0.04)
    parser.add_argument("--outcome-head", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--require-win-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.value_loss == "hl-gauss":
        if args.value_bins <= 1:
            parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
        if args.value_min >= args.value_max:
            parser.error("--value-min must be less than --value-max")
        if args.value_sigma <= 0.0:
            parser.error("--value-sigma must be positive")
    if args.require_win_rate is not None and not (0.0 <= args.require_win_rate <= 1.0):
        parser.error("--require-win-rate must be between 0 and 1")
    return args


def _row_to_printable(row: AdaptiveEvalRow) -> str:
    return (
        f"{row.grid_size}x{row.grid_size} player {row.policy_player}: "
        f"wins/losses/draws={row.wins}/{row.losses}/{row.draws}, "
        f"win_rate={row.win_rate * 100:.2f}%, "
        f"decisive={row.decisive_win_rate * 100:.2f}%, "
        f"draw={row.draw_rate * 100:.2f}%, "
        f"mean_time={row.mean_time:.1f}"
    )


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)
    key, net_key = jrandom.split(key)
    network_global_context = args.global_context or args.scoreboard_history
    if args.scoreboard_history:
        input_channels = ADAPTIVE_HISTORY_INPUT_CHANNELS
    elif network_global_context:
        input_channels = ADAPTIVE_GLOBAL_INPUT_CHANNELS
    else:
        input_channels = ADAPTIVE_INPUT_CHANNELS
    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.model_path,
        channels=args.channels,
        input_channels=input_channels,
        init_input_channels=input_channels,
        value_head_sizes=args.grid_sizes if args.value_heads == "per-size" else (),
        value_bins=args.value_bins if args.value_loss == "hl-gauss" else 0,
        value_min=args.value_min,
        value_max=args.value_max,
        value_sigma=args.value_sigma,
        outcome_head=args.outcome_head,
        global_context=network_global_context,
        init_global_context=network_global_context,
    )
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    policy_mode = 0 if args.policy_mode == "greedy" else 1
    rows = []

    print("Adaptive policy evaluation")
    print(f"Model:       {args.model_path}")
    print(f"Device:      {jax.devices()[0]}")
    print(f"Grid sizes:  {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Opponent:    {args.opponent}")
    print(f"Mode:        {args.policy_mode}")
    if args.value_heads != "shared":
        print(f"Value heads: {args.value_heads}")
    if args.value_loss == "hl-gauss":
        print(
            "Value loss:  "
            f"hl-gauss bins={args.value_bins} range=[{args.value_min:g},{args.value_max:g}] "
            f"sigma={args.value_sigma:g}"
        )
    if args.outcome_head:
        print("Outcome:    auxiliary head loaded")
    if network_global_context:
        print(f"Global ctx: {input_channels} input channels")
    if args.scoreboard_history:
        print("Score hist: previous+delta channels")
    print()

    for grid_size in args.grid_sizes:
        for policy_player in (0, 1):
            key, pool_key, eval_key = jrandom.split(key, 3)
            pool = make_adaptive_state_pool(
                pool_key,
                args.num_games,
                (grid_size,),
                args.pad_to,
                args.map_generator,
                (args.mountain_density_min, args.mountain_density_max),
                (args.num_cities_min, args.num_cities_max),
                args.max_generals_distance,
                (args.city_army_min, args.city_army_max),
            )
            states = pool.states
            t0 = time.time()
            info = evaluate_batch(
                network,
                states,
                grid_size,
                eval_key,
                args.max_steps,
                opponent_id,
                policy_mode,
                policy_player,
                args.pad_to,
                network_global_context,
                args.scoreboard_history,
            )
            jax.block_until_ready(info.winner)
            row_jax = summarize_row(info, grid_size, policy_player, args.num_games)
            row = AdaptiveEvalRow(
                grid_size=grid_size,
                policy_player=policy_player,
                wins=int(row_jax.wins),
                losses=int(row_jax.losses),
                draws=int(row_jax.draws),
                num_games=args.num_games,
                mean_time=float(row_jax.mean_time),
            )
            rows.append(row)
            elapsed = time.time() - t0
            print(f"{_row_to_printable(row)} | elapsed={elapsed:.2f}s")

    min_win_rate = min(row.win_rate for row in rows)
    payload = {
        "model_path": args.model_path,
        "grid_sizes": list(args.grid_sizes),
        "pad_to": args.pad_to,
        "opponent": args.opponent,
        "policy_mode": args.policy_mode,
        "num_games": args.num_games,
        "max_steps": args.max_steps,
        "global_context": network_global_context,
        "scoreboard_history": args.scoreboard_history,
        "min_win_rate": min_win_rate,
        "rows": [row.to_dict() for row in rows],
    }
    if args.json_output is not None:
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Minimum win rate: {min_win_rate * 100:.2f}%")
    if args.require_win_rate is not None and min_win_rate < args.require_win_rate:
        print(f"Required win rate {args.require_win_rate * 100:.2f}% not reached")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
