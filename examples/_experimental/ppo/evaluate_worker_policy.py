"""Evaluate adaptive Worker checkpoints with an observation-only command wrapper."""

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
    ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS,
    adaptive_index_to_action,
    adaptive_input_channel_count,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    empty_adaptive_fog_memory,
    make_adaptive_state_pool,
    parse_grid_sizes,
    reset_adaptive_fog_memory,
    update_adaptive_fog_memory,
)
from adaptive_network import load_or_create_adaptive_network
from adaptive_worker_pretrain import (
    WORKER_COMMAND_NAME_TO_ID,
    WORKER_COMMAND_NAMES,
    WORKER_INPUT_CHANNELS,
    worker_command_obs_to_array,
)
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, opponent_action
from generals.core import game
from train import random_action, stack_learner_actions

WORKER_TRIGGER_NAMES = ("always", "visible-general", "contact", "turn")
WORKER_TRIGGER_NAME_TO_ID = {name: index for index, name in enumerate(WORKER_TRIGGER_NAMES)}
WORKER_HYBRID_MODE_NAMES = ("switch", "rerank")
WORKER_HYBRID_MODE_NAME_TO_ID = {name: index for index, name in enumerate(WORKER_HYBRID_MODE_NAMES)}
WORKER_HYBRID_SWITCH = WORKER_HYBRID_MODE_NAME_TO_ID["switch"]


@dataclass(frozen=True)
class WorkerEvalRow:
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
def _worker_action(network, obs_arr, mask, active, key, policy_mode):
    logits, _ = network.logits_value(obs_arr, mask, active)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, network.pad_size)


@eqx.filter_jit
def _fallback_action(network, obs_arr, mask, active, key, policy_mode):
    logits, _ = network.logits_value(obs_arr, mask, active)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, network.pad_size)


def _indices_from_logits(logits, keys, policy_mode):
    return jax.vmap(
        lambda row, key: jax.lax.cond(
            policy_mode == 0,
            lambda _: jnp.argmax(row),
            lambda _: jrandom.categorical(key, row),
            None,
        )
    )(logits, keys)


def _actions_from_indices(indices, pad_size: int) -> jnp.ndarray:
    return jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(indices)


def worker_rerank_logits(
    fallback_logits: jnp.ndarray,
    worker_logits: jnp.ndarray,
    trigger: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    """Use centered legal Worker logits as a small bias on fallback policy logits."""
    legal = worker_logits > -1.0e8
    legal_count = jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1)
    legal_mean = jnp.sum(jnp.where(legal, worker_logits, 0.0), axis=-1, keepdims=True) / legal_count
    worker_bias = jnp.where(legal, worker_logits - legal_mean, 0.0)
    reranked = fallback_logits + scale * worker_bias
    return jnp.where(trigger[:, None], reranked, fallback_logits)


def summarize_worker_row(info, grid_size: int, policy_player: int, num_games: int) -> WorkerEvalRow:
    opponent_player = 1 - policy_player
    wins = jnp.sum(info.winner == policy_player)
    losses = jnp.sum(info.winner == opponent_player)
    draws = jnp.sum(info.winner < 0)
    return WorkerEvalRow(
        grid_size=grid_size,
        policy_player=policy_player,
        wins=wins,
        losses=losses,
        draws=draws,
        num_games=num_games,
        mean_time=jnp.mean(info.time),
    )


@eqx.filter_jit
def evaluate_worker_batch(
    network,
    states,
    effective_size,
    key,
    max_steps,
    opponent,
    policy_mode,
    policy_player,
    pad_size,
    command_mode,
    min_army,
):
    """Evaluate one Worker checkpoint with an observation-only command wrapper."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        obs_arr, active = jax.vmap(
            lambda obs, size: worker_command_obs_to_array(obs, size, pad_size, command_mode, min_army)
        )(policy_obs, effective_sizes)
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
        policy_actions = jax.vmap(lambda o, m, a, k: _worker_action(network, o, m, a, k, policy_mode))(
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
        new_states, _ = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        return (final_states, key), None

    (states, _), _ = jax.lax.scan(body, (states, key), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


def worker_trigger_mask(policy_obs, trigger_mode: int, min_turn: int) -> jnp.ndarray:
    """Return rows where control should switch from fallback policy to Worker."""
    visible_general = jnp.any(policy_obs.generals & policy_obs.opponent_cells, axis=(1, 2))
    contact = jnp.any(policy_obs.opponent_cells, axis=(1, 2))
    turn = policy_obs.timestep >= min_turn
    always = jnp.ones_like(visible_general)
    options = jnp.stack([always, visible_general, contact, turn], axis=0)
    return options[trigger_mode]


@eqx.filter_jit
def evaluate_hybrid_worker_batch(
    worker_network,
    fallback_network,
    states,
    effective_size,
    key,
    max_steps,
    opponent,
    worker_policy_mode,
    fallback_policy_mode,
    policy_player,
    pad_size,
    command_mode,
    min_army,
    trigger_mode,
    trigger_min_turn,
    hybrid_mode,
    worker_logit_scale,
    fallback_global_context=False,
    fallback_scoreboard_history=False,
    fallback_fog_memory=False,
):
    """Evaluate fallback policy with conditional Worker execution or reranking."""
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)
    initial_history = jnp.zeros((num_envs, ADAPTIVE_SCOREBOARD_FEATURE_CHANNELS), dtype=jnp.float32)
    initial_fog_memory = empty_adaptive_fog_memory(num_envs, pad_size)

    def body(carry, _):
        states, key, history, memory = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        if fallback_fog_memory:
            current_memory = jax.vmap(update_adaptive_fog_memory)(memory, policy_obs)
        else:
            current_memory = memory

        worker_obs_arr, worker_active = jax.vmap(
            lambda obs, size: worker_command_obs_to_array(obs, size, pad_size, command_mode, min_army)
        )(policy_obs, effective_sizes)
        masks = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(policy_obs, effective_sizes)
        if fallback_scoreboard_history:
            current_scoreboard = jax.vmap(lambda obs, size: adaptive_scoreboard_features(obs, size))(
                policy_obs,
                effective_sizes,
            )
            history_context = adaptive_scoreboard_history_context(history, current_scoreboard)
            fallback_obs_arr, fallback_active = jax.vmap(
                lambda obs, size, row_history, row_memory: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=True,
                    scoreboard_history=row_history,
                    fog_memory=row_memory if fallback_fog_memory else None,
                )
            )(
                policy_obs,
                effective_sizes,
                history_context,
                current_memory,
            )
        else:
            current_scoreboard = history
            fallback_obs_arr, fallback_active = jax.vmap(
                lambda obs, size, row_memory: adaptive_obs_to_array(
                    obs,
                    size,
                    pad_size,
                    include_global_context=fallback_global_context,
                    fog_memory=row_memory if fallback_fog_memory else None,
                )
            )(
                policy_obs,
                effective_sizes,
                current_memory,
            )

        key, worker_key, fallback_key, opponent_key = jrandom.split(key, 4)
        worker_keys = jrandom.split(worker_key, num_envs)
        fallback_keys = jrandom.split(fallback_key, num_envs)
        worker_logits = jax.vmap(lambda o, m, a: worker_network.logits_value(o, m, a)[0])(
            worker_obs_arr,
            masks,
            worker_active,
        )
        fallback_logits = jax.vmap(lambda o, m, a: fallback_network.logits_value(o, m, a)[0])(
            fallback_obs_arr,
            masks,
            fallback_active,
        )
        trigger = worker_trigger_mask(policy_obs, trigger_mode, trigger_min_turn)

        def switch_actions(_):
            worker_indices = _indices_from_logits(worker_logits, worker_keys, worker_policy_mode)
            fallback_indices = _indices_from_logits(fallback_logits, fallback_keys, fallback_policy_mode)
            worker_actions = _actions_from_indices(worker_indices, pad_size)
            fallback_actions = _actions_from_indices(fallback_indices, pad_size)
            return jnp.where(trigger[:, None], worker_actions, fallback_actions)

        def rerank_actions(_):
            combined_logits = worker_rerank_logits(fallback_logits, worker_logits, trigger, worker_logit_scale)
            combined_indices = _indices_from_logits(combined_logits, fallback_keys, fallback_policy_mode)
            return _actions_from_indices(combined_indices, pad_size)

        policy_actions = jax.lax.cond(
            hybrid_mode == WORKER_HYBRID_SWITCH,
            switch_actions,
            rerank_actions,
            None,
        )
        opponent_keys = jrandom.split(opponent_key, num_envs)
        opponent_actions = jax.vmap(lambda k, obs: opponent_action(opponent, k, obs, random_action))(
            opponent_keys,
            opponent_obs,
        )
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, _ = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        final_info = jax.vmap(game.get_info)(final_states)
        final_memory = reset_adaptive_fog_memory(current_memory, final_info.is_done)
        return (final_states, key, current_scoreboard, final_memory), None

    (states, _, _, _), _ = jax.lax.scan(
        body,
        (states, key, initial_history, initial_fog_memory),
        None,
        length=max_steps,
    )
    return jax.vmap(game.get_info)(states)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an adaptive Worker checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--num-games", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--command-mode", choices=WORKER_COMMAND_NAMES, default="auto")
    parser.add_argument("--fallback-model-path", default=None)
    parser.add_argument("--fallback-channels", default=None)
    parser.add_argument("--fallback-policy-mode", choices=("greedy", "sample"), default=None)
    parser.add_argument("--fallback-network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--fallback-global-context", action="store_true")
    parser.add_argument("--fallback-scoreboard-history", action="store_true")
    parser.add_argument("--fallback-fog-memory", action="store_true")
    parser.add_argument("--fallback-value-heads", choices=("shared", "per-size"), default="shared")
    parser.add_argument("--fallback-value-head-sizes", default=None)
    parser.add_argument("--fallback-value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--fallback-value-bins", type=int, default=128)
    parser.add_argument("--fallback-value-min", type=float, default=-1.0)
    parser.add_argument("--fallback-value-max", type=float, default=1.0)
    parser.add_argument("--fallback-value-sigma", type=float, default=0.04)
    parser.add_argument("--hybrid-mode", choices=WORKER_HYBRID_MODE_NAMES, default="switch")
    parser.add_argument("--worker-logit-scale", type=float, default=0.25)
    parser.add_argument("--worker-trigger", choices=WORKER_TRIGGER_NAMES, default="always")
    parser.add_argument("--worker-trigger-min-turn", type=int, default=80)
    parser.add_argument("--min-army", type=int, default=2)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        args.grid_sizes = parse_grid_sizes(args.grid_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.fallback_value_head_sizes = (
            parse_grid_sizes(args.fallback_value_head_sizes)
            if args.fallback_value_head_sizes is not None
            else args.grid_sizes
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to < max(args.grid_sizes):
        parser.error("--pad-to must be at least the maximum grid size")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.min_army < 2:
        parser.error("--min-army must be at least 2")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.worker_trigger_min_turn < 0:
        parser.error("--worker-trigger-min-turn must be non-negative")
    if args.worker_logit_scale < 0.0:
        parser.error("--worker-logit-scale must be non-negative")
    if args.fallback_value_loss == "hl-gauss":
        if args.fallback_value_bins <= 1:
            parser.error("--fallback-value-bins must be greater than 1 for --fallback-value-loss hl-gauss")
        if args.fallback_value_min >= args.fallback_value_max:
            parser.error("--fallback-value-min must be less than --fallback-value-max")
        if args.fallback_value_sigma <= 0.0:
            parser.error("--fallback-value-sigma must be positive")
    args.command_mode_id = WORKER_COMMAND_NAME_TO_ID[args.command_mode]
    args.worker_trigger_id = WORKER_TRIGGER_NAME_TO_ID[args.worker_trigger]
    args.hybrid_mode_id = WORKER_HYBRID_MODE_NAME_TO_ID[args.hybrid_mode]
    return args


def _row_to_printable(row: WorkerEvalRow) -> str:
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
    network = load_or_create_adaptive_network(
        net_key,
        pad_size=args.pad_to,
        init_model_path=args.model_path,
        channels=args.channels,
        input_channels=WORKER_INPUT_CHANNELS,
        init_input_channels=WORKER_INPUT_CHANNELS,
    )
    fallback_network = None
    fallback_global_context = args.fallback_global_context or args.fallback_scoreboard_history
    if args.fallback_model_path is not None:
        fallback_input_channels = adaptive_input_channel_count(
            fallback_global_context,
            args.fallback_scoreboard_history,
            args.fallback_fog_memory,
        )
        fallback_network = load_or_create_adaptive_network(
            net_key,
            pad_size=args.pad_to,
            init_model_path=args.fallback_model_path,
            channels=args.fallback_channels,
            input_channels=fallback_input_channels,
            init_input_channels=fallback_input_channels,
            value_head_sizes=args.fallback_value_head_sizes if args.fallback_value_heads == "per-size" else (),
            value_bins=args.fallback_value_bins if args.fallback_value_loss == "hl-gauss" else 0,
            value_min=args.fallback_value_min,
            value_max=args.fallback_value_max,
            value_sigma=args.fallback_value_sigma,
            global_context=fallback_global_context,
            init_global_context=fallback_global_context,
            network_arch=args.fallback_network_arch,
            init_network_arch=args.fallback_network_arch,
        )
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]
    policy_mode = 0 if args.policy_mode == "greedy" else 1
    fallback_policy_mode = 0 if (args.fallback_policy_mode or args.policy_mode) == "greedy" else 1
    rows = []

    print("Adaptive Worker policy evaluation")
    print(f"Model:       {args.model_path}")
    print(f"Device:      {jax.devices()[0]}")
    print(f"Grid sizes:  {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    print(f"Opponent:    {args.opponent}")
    print(f"Mode:        {args.policy_mode}")
    print(f"Command:     {args.command_mode}")
    if args.fallback_model_path is not None:
        print(f"Fallback:    {args.fallback_model_path}")
        print(f"Fallback arch/input: {args.fallback_network_arch}/{fallback_input_channels}")
        print(f"Hybrid mode: {args.hybrid_mode}")
        if args.hybrid_mode == "rerank":
            print(f"Logit scale: {args.worker_logit_scale:g}")
        print(f"Trigger:     {args.worker_trigger}")
    print(f"Input chans: {WORKER_INPUT_CHANNELS}")
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
            if fallback_network is None:
                info = evaluate_worker_batch(
                    network,
                    states,
                    grid_size,
                    eval_key,
                    args.max_steps,
                    opponent_id,
                    policy_mode,
                    policy_player,
                    args.pad_to,
                    args.command_mode_id,
                    args.min_army,
                )
            else:
                info = evaluate_hybrid_worker_batch(
                    network,
                    fallback_network,
                    states,
                    grid_size,
                    eval_key,
                    args.max_steps,
                    opponent_id,
                    policy_mode,
                    fallback_policy_mode,
                    policy_player,
                    args.pad_to,
                    args.command_mode_id,
                    args.min_army,
                    args.worker_trigger_id,
                    args.worker_trigger_min_turn,
                    args.hybrid_mode_id,
                    args.worker_logit_scale,
                    fallback_global_context,
                    args.fallback_scoreboard_history,
                    args.fallback_fog_memory,
                )
            jax.block_until_ready(info.winner)
            row_jax = summarize_worker_row(info, grid_size, policy_player, args.num_games)
            row = WorkerEvalRow(
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
        "command_mode": args.command_mode,
        "fallback_model_path": args.fallback_model_path,
        "fallback_network_arch": args.fallback_network_arch,
        "fallback_global_context": args.fallback_global_context,
        "fallback_scoreboard_history": args.fallback_scoreboard_history,
        "fallback_fog_memory": args.fallback_fog_memory,
        "fallback_value_heads": args.fallback_value_heads,
        "fallback_value_loss": args.fallback_value_loss,
        "hybrid_mode": args.hybrid_mode,
        "worker_logit_scale": args.worker_logit_scale,
        "worker_trigger": args.worker_trigger,
        "num_games": args.num_games,
        "max_steps": args.max_steps,
        "min_win_rate": min_win_rate,
        "rows": [row.to_dict() for row in rows],
    }
    if args.json_output is not None:
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Minimum win rate: {min_win_rate * 100:.2f}%")


if __name__ == "__main__":
    main()
