"""BFS-supervised Worker pretraining for adaptive multisize policies."""

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
    ADAPTIVE_INPUT_CHANNELS,
    adaptive_action_space_size,
    adaptive_action_to_index,
    adaptive_expander_target_probs,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    active_cells_for_size,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from generals.core import game
from generals.core.action import DIRECTIONS
from generals.core.rewards import shortest_path_distance_map
from train import checkpoint_path_for_iteration, prune_old_checkpoints

WORKER_EXTRA_CHANNELS = 3
WORKER_INPUT_CHANNELS = ADAPTIVE_INPUT_CHANNELS + WORKER_EXTRA_CHANNELS
WORKER_TARGET_NAMES = ("general", "city", "frontier", "random")
WORKER_TARGET_NAME_TO_ID = {name: index for index, name in enumerate(WORKER_TARGET_NAMES)}
WORKER_TARGET_RANDOM = WORKER_TARGET_NAME_TO_ID["random"]
WORKER_COMMAND_NAMES = ("auto", "visible-general", "city", "frontier")
WORKER_COMMAND_NAME_TO_ID = {name: index for index, name in enumerate(WORKER_COMMAND_NAMES)}
WORKER_COMMAND_AUTO = WORKER_COMMAND_NAME_TO_ID["auto"]
WORKER_COMMAND_VISIBLE_GENERAL = WORKER_COMMAND_NAME_TO_ID["visible-general"]
WORKER_COMMAND_CITY = WORKER_COMMAND_NAME_TO_ID["city"]
WORKER_COMMAND_FRONTIER = WORKER_COMMAND_NAME_TO_ID["frontier"]


def worker_target_mask(state, player: int, effective_size: int, pad_size: int, target_family: int) -> jnp.ndarray:
    """Return an oracle target mask for the Worker command."""
    active = active_cells_for_size(effective_size, pad_size)
    opponent = 1 - player
    target = state.general_positions[opponent]
    general_targets = jnp.zeros_like(state.passable).at[target[0], target[1]].set(True) & active
    city_targets = state.cities & ~state.ownership[player] & state.passable & active
    frontier_targets = ~state.ownership[player] & state.passable & active
    masks = jnp.stack([general_targets, city_targets, frontier_targets], axis=0)
    selected = masks[target_family]
    fallback = jnp.where(jnp.any(selected), selected, frontier_targets)
    return fallback


def worker_route_distance(
    state,
    player: int,
    effective_size: int,
    pad_size: int,
    target_family: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return passable shortest-path distances for one Worker target family."""
    active = active_cells_for_size(effective_size, pad_size)
    passable = state.passable & active
    target_mask = worker_target_mask(state, player, effective_size, pad_size, target_family)
    distance = shortest_path_distance_map(passable, target_mask)
    return target_mask, distance


def worker_obs_to_array(
    state,
    obs,
    player: int,
    effective_size: int,
    pad_size: int,
    target_family: int,
    min_army: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Append Worker command channels to the standard adaptive observation."""
    base, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    target_mask, distance = worker_route_distance(state, player, effective_size, pad_size, target_family)
    active_f = active.astype(jnp.float32)
    eligible = state.ownership[player] & active & (state.armies >= min_army)
    army_scale = jnp.maximum(jnp.log1p(jnp.max(state.armies.astype(jnp.float32))), 1.0)
    source_heatmap = jnp.where(eligible, jnp.log1p(state.armies.astype(jnp.float32)) / army_scale, 0.0)
    max_distance = jnp.maximum(jnp.asarray(effective_size * effective_size, dtype=jnp.float32), 1.0)
    route_potential = 1.0 - jnp.minimum(distance.astype(jnp.float32), max_distance) / max_distance
    route_potential = jnp.where(active, route_potential, 0.0)
    extra = jnp.stack(
        [
            target_mask.astype(jnp.float32) * active_f,
            source_heatmap,
            route_potential,
        ],
        axis=0,
    )
    return jnp.concatenate([base, extra], axis=0), active


def worker_command_target_mask(obs, effective_size: int, pad_size: int, command_mode: int) -> jnp.ndarray:
    """Build a command target mask from fogged observation only."""
    active = active_cells_for_size(effective_size, pad_size)
    visible_general = obs.generals & obs.opponent_cells & active
    city_targets = ((obs.cities & ~obs.owned_cells) | obs.structures_in_fog) & active
    frontier_targets = (obs.opponent_cells | obs.neutral_cells | obs.fog_cells | obs.structures_in_fog) & active
    fallback = jnp.where(jnp.any(frontier_targets), frontier_targets, active & ~obs.owned_cells)
    general_or_frontier = jnp.where(jnp.any(visible_general), visible_general, fallback)
    city_or_frontier = jnp.where(jnp.any(city_targets), city_targets, fallback)
    auto_targets = jnp.where(jnp.any(visible_general), visible_general, city_or_frontier)
    options = jnp.stack([auto_targets, general_or_frontier, city_or_frontier, fallback], axis=0)
    return options[command_mode]


def worker_command_obs_to_array(
    obs,
    effective_size: int,
    pad_size: int,
    command_mode: int,
    min_army: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Append observation-only Worker command channels for policy execution."""
    base, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    target_mask = worker_command_target_mask(obs, effective_size, pad_size, command_mode)
    passable = active & ~obs.mountains
    distance = shortest_path_distance_map(passable, target_mask)
    eligible = obs.owned_cells & active & (obs.armies >= min_army)
    army_scale = jnp.maximum(jnp.log1p(jnp.max(obs.armies.astype(jnp.float32))), 1.0)
    source_heatmap = jnp.where(eligible, jnp.log1p(obs.armies.astype(jnp.float32)) / army_scale, 0.0)
    max_distance = jnp.maximum(jnp.asarray(effective_size * effective_size, dtype=jnp.float32), 1.0)
    route_potential = 1.0 - jnp.minimum(distance.astype(jnp.float32), max_distance) / max_distance
    route_potential = jnp.where(active, route_potential, 0.0)
    extra = jnp.stack(
        [
            target_mask.astype(jnp.float32) * active.astype(jnp.float32),
            source_heatmap,
            route_potential,
        ],
        axis=0,
    )
    return jnp.concatenate([base, extra], axis=0), active


def worker_bfs_move_scores(
    state,
    player: int,
    effective_size: int,
    pad_size: int,
    target_family: int,
    min_army: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Score all one-step moves that decrease BFS distance to the target."""
    _, distance = worker_route_distance(state, player, effective_size, pad_size, target_family)
    valid_mask = compute_adaptive_valid_move_mask(
        state.armies,
        state.ownership[player],
        state.mountains,
        effective_size,
        pad_size,
    )
    rows = jnp.arange(pad_size)[:, None, None]
    cols = jnp.arange(pad_size)[None, :, None]
    dest_i = rows + DIRECTIONS[None, None, :, 0]
    dest_j = cols + DIRECTIONS[None, None, :, 1]
    safe_i = jnp.clip(dest_i, 0, pad_size - 1)
    safe_j = jnp.clip(dest_j, 0, pad_size - 1)
    source_distance = distance[:, :, None]
    dest_distance = distance[safe_i, safe_j]
    reachable = source_distance < (pad_size * pad_size + 1)
    progresses = dest_distance < source_distance
    eligible = valid_mask & reachable & progresses & (state.armies[:, :, None] >= min_army)
    max_distance = jnp.maximum(jnp.asarray(effective_size * effective_size, dtype=jnp.float32), 1.0)
    source_closeness = 1.0 - jnp.minimum(source_distance.astype(jnp.float32), max_distance) / max_distance
    army_score = jnp.log1p(state.armies.astype(jnp.float32))[:, :, None]
    progress_score = (source_distance - dest_distance).astype(jnp.float32)
    scores = 4.0 * army_score + 2.0 * source_closeness + progress_score
    scores = jnp.where(eligible, scores, -1.0e9)
    return scores, jnp.any(eligible)


def worker_bfs_action_index(
    state,
    player: int,
    effective_size: int,
    pad_size: int,
    target_family: int,
    min_army: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Choose the highest-scoring one-step BFS move."""
    scores, has_label = worker_bfs_move_scores(state, player, effective_size, pad_size, target_family, min_army)
    flat = jnp.argmax(scores.reshape(-1))
    source_flat = flat // 4
    row = source_flat // pad_size
    col = source_flat % pad_size
    direction = flat % 4
    action = jnp.array([0, row, col, direction, 0], dtype=jnp.int32)
    pass_index = adaptive_action_space_size(pad_size) - 1
    index = jnp.where(has_label, adaptive_action_to_index(action, pad_size), pass_index)
    return index.astype(jnp.int32), has_label.astype(jnp.float32)


def worker_bfs_target_probs(
    state,
    player: int,
    effective_size: int,
    pad_size: int,
    target_family: int,
    min_army: int,
    target_temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a soft distribution over all useful BFS-progress moves."""
    scores, has_label = worker_bfs_move_scores(state, player, effective_size, pad_size, target_family, min_army)
    full_planes = jnp.transpose(scores, (2, 0, 1)).reshape(4 * pad_size * pad_size)
    half_planes = jnp.full_like(full_planes, -1.0e9)
    pass_score = jnp.asarray([-1.0e9], dtype=jnp.float32)
    logits = jnp.concatenate([full_planes, half_planes, pass_score], axis=0) / target_temperature
    pass_index = adaptive_action_space_size(pad_size) - 1
    fallback = jax.nn.one_hot(pass_index, adaptive_action_space_size(pad_size), dtype=jnp.float32)
    probs = jnp.where(has_label, jax.nn.softmax(logits, axis=0), fallback)
    label = jnp.argmax(probs).astype(jnp.int32)
    return probs, label, has_label.astype(jnp.float32)


def worker_source_direction_targets(targets: jnp.ndarray, pad_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Marginalize adaptive action targets into source-cell and direction targets."""
    leading_shape = targets.shape[:-1]
    move_planes = targets[..., : 8 * pad_size * pad_size].reshape(*leading_shape, 8, pad_size, pad_size)
    source_targets = jnp.sum(move_planes, axis=-3).reshape(*leading_shape, pad_size * pad_size)
    full_direction_targets = jnp.sum(move_planes[..., :4, :, :], axis=(-1, -2))
    half_direction_targets = jnp.sum(move_planes[..., 4:8, :, :], axis=(-1, -2))
    direction_targets = full_direction_targets + half_direction_targets
    return source_targets, direction_targets


def worker_source_direction_logits(logits: jnp.ndarray, pad_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert flat adaptive logits into source-cell and direction logits."""
    leading_shape = logits.shape[:-1]
    move_logits = logits[..., : 8 * pad_size * pad_size].reshape(*leading_shape, 8, pad_size, pad_size)
    source_logits = jax.nn.logsumexp(move_logits, axis=-3).reshape(*leading_shape, pad_size * pad_size)
    direction_logits = jax.nn.logsumexp(
        jnp.stack([move_logits[..., :4, :, :], move_logits[..., 4:8, :, :]], axis=-1),
        axis=(-1, -2, -3),
    )
    return source_logits, direction_logits


def target_families_for_mode(key, num_envs: int, target_mode: int) -> jnp.ndarray:
    """Sample or broadcast Worker target family ids."""
    if target_mode == WORKER_TARGET_RANDOM:
        return jrandom.randint(key, (num_envs,), 0, 3, dtype=jnp.int32)
    return jnp.full((num_envs,), target_mode, dtype=jnp.int32)


def expander_action(obs, effective_size: int, key, pad_size: int) -> jnp.ndarray:
    """Sample one adaptive Expander action for data generation rollouts."""
    probs = adaptive_expander_target_probs(obs, effective_size, pad_size)
    index = jrandom.categorical(key, jnp.log(probs + 1.0e-8))
    return adaptive_index_to_action(index, pad_size)


@eqx.filter_jit
def collect_worker_batch(
    states,
    effective_sizes,
    pool,
    key,
    steps: int,
    truncation: int,
    target_mode: int,
    pad_size: int,
    min_army: int,
    target_temperature: float,
):
    """Collect Worker supervised labels from Expander-vs-Expander rollouts."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, effective_sizes, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)

        key, family_key0, family_key1, action_key = jrandom.split(key, 4)
        families_p0 = target_families_for_mode(family_key0, num_envs, target_mode)
        families_p1 = target_families_for_mode(family_key1, num_envs, target_mode)

        obs_arr_p0, active_p0 = jax.vmap(
            lambda state, obs, size, family: worker_obs_to_array(state, obs, 0, size, pad_size, family, min_army)
        )(states, obs_p0, effective_sizes, families_p0)
        obs_arr_p1, active_p1 = jax.vmap(
            lambda state, obs, size, family: worker_obs_to_array(state, obs, 1, size, pad_size, family, min_army)
        )(states, obs_p1, effective_sizes, families_p1)
        masks_p0 = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(obs_p0, effective_sizes)
        masks_p1 = jax.vmap(
            lambda obs, size: compute_adaptive_valid_move_mask(
                obs.armies,
                obs.owned_cells,
                obs.mountains,
                size,
                pad_size,
            )
        )(obs_p1, effective_sizes)
        targets_p0, labels_p0, weights_p0 = jax.vmap(
            lambda state, size, family: worker_bfs_target_probs(
                state,
                0,
                size,
                pad_size,
                family,
                min_army,
                target_temperature,
            )
        )(states, effective_sizes, families_p0)
        targets_p1, labels_p1, weights_p1 = jax.vmap(
            lambda state, size, family: worker_bfs_target_probs(
                state,
                1,
                size,
                pad_size,
                family,
                min_army,
                target_temperature,
            )
        )(states, effective_sizes, families_p1)

        action_keys = jrandom.split(action_key, num_envs * 2).reshape(num_envs, 2, 2)
        actions_p0 = jax.vmap(lambda obs, size, sample_key: expander_action(obs, size, sample_key, pad_size))(
            obs_p0,
            effective_sizes,
            action_keys[:, 0],
        )
        actions_p1 = jax.vmap(lambda obs, size, sample_key: expander_action(obs, size, sample_key, pad_size))(
            obs_p1,
            effective_sizes,
            action_keys[:, 1],
        )
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

        obs_arr = jnp.concatenate([obs_arr_p0, obs_arr_p1], axis=0)
        masks = jnp.concatenate([masks_p0, masks_p1], axis=0)
        active = jnp.concatenate([active_p0, active_p1], axis=0)
        targets = jnp.concatenate([targets_p0, targets_p1], axis=0)
        labels = jnp.concatenate([labels_p0, labels_p1], axis=0)
        weights = jnp.concatenate([weights_p0, weights_p1], axis=0)
        families = jnp.concatenate([families_p0, families_p1], axis=0)
        return (final_states, final_sizes, key), (obs_arr, masks, active, targets, labels, weights, families, dones)

    (states, effective_sizes, key), batch = jax.lax.scan(
        body,
        (states, effective_sizes, key),
        None,
        length=steps,
    )
    return states, effective_sizes, batch, key


@eqx.filter_jit
def train_worker_step(
    network,
    opt_state,
    obs,
    masks,
    active,
    targets,
    labels,
    weights,
    optimizer,
    action_loss_weight: float,
    source_loss_weight: float,
    direction_loss_weight: float,
):
    """Train one Worker supervised batch."""
    batch_size = obs.shape[0] * obs.shape[1]
    pad_size = active.shape[-1]
    obs_flat = obs.reshape(batch_size, *obs.shape[2:])
    masks_flat = masks.reshape(batch_size, *masks.shape[2:])
    active_flat = active.reshape(batch_size, *active.shape[2:])
    targets_flat = targets.reshape(batch_size, targets.shape[-1])
    labels_flat = labels.reshape(batch_size)
    weights_flat = weights.reshape(batch_size)
    source_targets, direction_targets = worker_source_direction_targets(targets_flat, pad_size)

    def loss_fn(net):
        logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs_flat, masks_flat, active_flat)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        action_losses = -jnp.sum(targets_flat * log_probs, axis=-1)
        source_logits, direction_logits = worker_source_direction_logits(logits, pad_size)
        source_log_probs = jax.nn.log_softmax(source_logits, axis=-1)
        direction_log_probs = jax.nn.log_softmax(direction_logits, axis=-1)
        source_losses = -jnp.sum(source_targets * source_log_probs, axis=-1)
        direction_losses = -jnp.sum(direction_targets * direction_log_probs, axis=-1)
        losses = (
            action_loss_weight * action_losses
            + source_loss_weight * source_losses
            + direction_loss_weight * direction_losses
        )
        normalizer = jnp.maximum(jnp.sum(weights_flat), 1.0)
        loss = jnp.sum(losses * weights_flat) / normalizer
        predictions = jnp.argmax(logits, axis=-1)
        source_predictions = jnp.argmax(source_logits, axis=-1)
        direction_predictions = jnp.argmax(direction_logits, axis=-1)
        source_labels = jnp.argmax(source_targets, axis=-1)
        direction_labels = jnp.argmax(direction_targets, axis=-1)
        accuracy = jnp.sum((predictions == labels_flat) * weights_flat) / normalizer
        source_accuracy = jnp.sum((source_predictions == source_labels) * weights_flat) / normalizer
        direction_accuracy = jnp.sum((direction_predictions == direction_labels) * weights_flat) / normalizer
        predicted_mass = jnp.take_along_axis(targets_flat, predictions[:, None], axis=1)[:, 0]
        useful = predicted_mass > 0.0
        useful_accuracy = jnp.sum(useful.astype(jnp.float32) * weights_flat) / normalizer
        mean_predicted_mass = jnp.sum(predicted_mass * weights_flat) / normalizer
        valid_fraction = jnp.mean(weights_flat > 0.0)
        return loss, (
            accuracy,
            source_accuracy,
            direction_accuracy,
            useful_accuracy,
            mean_predicted_mass,
            valid_fraction,
        )

    (
        loss,
        (accuracy, source_accuracy, direction_accuracy, useful_accuracy, mean_predicted_mass, valid_fraction),
    ), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)
    return (
        network,
        opt_state,
        loss,
        accuracy,
        source_accuracy,
        direction_accuracy,
        useful_accuracy,
        mean_predicted_mass,
        valid_fraction,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain adaptive Worker actions from BFS target assignments.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--target-family", choices=WORKER_TARGET_NAMES, default="random")
    parser.add_argument("--target-temperature", type=float, default=2.0)
    parser.add_argument("--min-army", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-loss-weight", type=float, default=0.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.0)
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
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--init-input-channels", type=int, default=None)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="runs/generals-adaptive-worker-pretrain.eqx")
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
    if args.num_steps <= 0 or args.num_iterations <= 0:
        parser.error("--num-steps and --num-iterations must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.action_loss_weight < 0.0 or args.source_loss_weight < 0.0 or args.direction_loss_weight < 0.0:
        parser.error("Worker loss weights must be non-negative")
    if args.action_loss_weight + args.source_loss_weight + args.direction_loss_weight <= 0.0:
        parser.error("At least one Worker loss weight must be positive")
    if args.target_temperature <= 0.0:
        parser.error("--target-temperature must be positive")
    if args.truncation <= 0:
        parser.error("--truncation must be positive")
    if args.min_army < 2:
        parser.error("--min-army must be at least 2")
    if args.init_input_channels is not None and args.init_input_channels <= 0:
        parser.error("--init-input-channels must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.checkpoint_every < 0 or args.keep_checkpoints < 0:
        parser.error("--checkpoint-every and --keep-checkpoints cannot be negative")
    args.target_family_id = WORKER_TARGET_NAME_TO_ID[args.target_family]
    return args


def main():
    args = parse_args()

    print("Adaptive Worker BFS pretraining")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Environments:  {args.num_envs}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Target family: {args.target_family}")
    print(f"Target temp:   {args.target_temperature:g}")
    print(f"Input chans:   {WORKER_INPUT_CHANNELS}")
    print(f"Iterations:    {args.num_iterations} x {args.num_steps} steps")
    print(
        f"Loss weights:  action={args.action_loss_weight:g}, "
        f"source={args.source_loss_weight:g}, direction={args.direction_loss_weight:g}"
    )
    print(f"Reset pool:    {args.pool_size}")
    if args.channels is not None:
        print(f"Channels:      {args.channels}")
    if args.init_model_path is not None:
        print(f"Warm start:    {args.init_model_path}")
    if args.init_input_channels is not None:
        print(f"Warm inputs:   {args.init_input_channels} channels")
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
        input_channels=WORKER_INPUT_CHANNELS,
        init_input_channels=args.init_input_channels,
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
        states, effective_sizes, batch, key = collect_worker_batch(
            states,
            effective_sizes,
            pool,
            key,
            args.num_steps,
            args.truncation,
            args.target_family_id,
            args.pad_to,
            args.min_army,
            args.target_temperature,
        )
        obs, masks, active, targets, labels, weights, families, dones = batch
        (
            network,
            opt_state,
            loss,
            accuracy,
            source_accuracy,
            direction_accuracy,
            useful_accuracy,
            mean_predicted_mass,
            valid_fraction,
        ) = train_worker_step(
            network,
            opt_state,
            obs,
            masks,
            active,
            targets,
            labels,
            weights,
            optimizer,
            args.action_loss_weight,
            args.source_loss_weight,
            args.direction_loss_weight,
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
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps * 2
            labeled = int(jnp.sum(weights))
            episodes = int(dones.sum())
            family_counts = jnp.bincount(families.reshape(-1), length=3)
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Acc: {float(accuracy) * 100:5.1f}% | Src: {float(source_accuracy) * 100:5.1f}% | "
                f"Dir: {float(direction_accuracy) * 100:5.1f}% | "
                f"Useful: {float(useful_accuracy) * 100:5.1f}% | "
                f"Mass: {float(mean_predicted_mass):.3f} | Valid: {float(valid_fraction) * 100:5.1f}% | "
                f"Labels: {labeled:5d} | Episodes: {episodes:4d} | "
                f"Fam: {int(family_counts[0])}/{int(family_counts[1])}/{int(family_counts[2])} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s"
            )

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
