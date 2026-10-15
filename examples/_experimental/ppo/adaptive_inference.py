"""Adaptive policy composition and online search shared by runtime and evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.adaptive_common import (
    adaptive_action_to_index,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    adaptive_scoreboard_features,
    adaptive_scoreboard_history_context,
    compute_adaptive_valid_move_mask,
    crop_observation,
    reset_adaptive_scoreboard_history,
    update_adaptive_fog_memory,
)
from examples._experimental.ppo.common import opponent_action, random_action
from generals.agents.ppo_policy_agent import policy_network_action
from generals.core import game

POLICY_ADAPTER_MODE_NAMES = ("delta", "blend", "replace")
POLICY_ADAPTER_MODE_TO_ID = {name: index for index, name in enumerate(POLICY_ADAPTER_MODE_NAMES)}
ONLINE_SEARCH_GATE_FEATURE_DIM = 17


def policy_adapter_delta_logits(
    policy_logits: jnp.ndarray,
    adapter_logits: jnp.ndarray,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
    """Add a centered legal delta from a separately trained policy-head adapter."""
    legal = policy_logits > -1.0e8
    raw_delta = adapter_logits - policy_logits
    legal_count = jnp.maximum(jnp.sum(legal), 1)
    legal_mean = jnp.sum(jnp.where(legal, raw_delta, 0.0)) / legal_count
    centered_delta = jnp.where(legal, raw_delta - legal_mean, 0.0)
    return policy_logits + scale * centered_delta


def policy_adapter_blend_logits(
    policy_logits: jnp.ndarray,
    adapter_logits: jnp.ndarray,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
    """Interpolate legal logits between the base policy and adapter policy."""
    legal = policy_logits > -1.0e8
    weight = jnp.clip(jnp.asarray(scale, dtype=policy_logits.dtype), 0.0, 1.0)
    blended = (1.0 - weight) * policy_logits + weight * adapter_logits
    return jnp.where(legal, blended, policy_logits)


def adapter_composed_policy_logits(
    network,
    policy_adapter_network,
    obs_arr: jnp.ndarray,
    mask: jnp.ndarray,
    active: jnp.ndarray,
    effective_size: int,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
) -> jnp.ndarray:
    """Return base policy logits with the deployment policy adapter composed in."""
    logits, _ = network.logits_value(obs_arr, mask, active)
    if policy_adapter_network is None or policy_adapter_scale <= 0.0:
        return logits
    size_allowed = (policy_adapter_min_grid_size <= 0 or effective_size >= policy_adapter_min_grid_size) and (
        policy_adapter_max_grid_size <= 0 or effective_size <= policy_adapter_max_grid_size
    )
    adapter_logits, _ = policy_adapter_network.logits_value(obs_arr, mask, active)
    adapted_logits = jax.lax.switch(
        policy_adapter_mode,
        (
            lambda _: policy_adapter_delta_logits(logits, adapter_logits, policy_adapter_scale),
            lambda _: policy_adapter_blend_logits(logits, adapter_logits, policy_adapter_scale),
            lambda _: adapter_logits,
        ),
        None,
    )
    return jnp.where(size_allowed, adapted_logits, logits)


def scalar_reset_fog_memory(memory, done: jnp.ndarray):
    """Reset one scalar fog-memory state after a terminal search rollout."""
    keep = (~done).astype(jnp.float32)
    return jax.tree.map(lambda value: value * keep, memory)


def search_score_observation(info, obs, player: int, army_weight: float, land_weight: float, terminal_score: float):
    """Score a search rollout leaf from one player's perspective."""
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


def adapter_policy_action_with_memory(
    network,
    policy_adapter_network,
    obs,
    effective_size: int,
    key,
    policy_mode: int,
    pad_size: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
):
    """Dispatch the deployment policy while carrying one-player context."""
    current_memory = update_adaptive_fog_memory(fog_memory, obs) if fog_memory_enabled else fog_memory
    current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
    history_context = (
        adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
        if scoreboard_history_enabled
        else None
    )
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
        fog_memory=current_memory if fog_memory_enabled else None,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, pad_size), current_scoreboard, current_memory


def online_search_action_policy_opponent(
    network,
    policy_adapter_network,
    command_gate_network,
    opponent_network,
    state,
    effective_size: int,
    key,
    fallback_action: jnp.ndarray,
    opponent_first_action: jnp.ndarray,
    policy_player: int,
    policy_mode: int,
    opponent_policy_mode: int,
    pad_size: int,
    max_steps: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
    top_k: int,
    rollout_steps: int,
    rollouts_per_action: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
    min_score_gap: float,
    online_search_gate_threshold: float,
    online_search_gate_feature_dim: int,
) -> jnp.ndarray:
    """Choose a primitive action by online counterfactual rollout search against a fixed policy."""
    obs = game.get_observation(state, policy_player)
    current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
    history_context = (
        adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
        if scoreboard_history_enabled
        else None
    )
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
        fog_memory=fog_memory if fog_memory_enabled else None,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    opponent_player = 1 - policy_player

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, prev_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, policy_player)
            learner_action, next_scoreboard, next_memory = adapter_policy_action_with_memory(
                network,
                policy_adapter_network,
                learner_obs,
                effective_size,
                learner_key,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                prev_scoreboard,
                fog_memory_enabled,
                memory,
                policy_adapter_scale,
                policy_adapter_mode,
                policy_adapter_min_grid_size,
                policy_adapter_max_grid_size,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = policy_network_action(
                opponent_network,
                opponent_key,
                crop_observation(opponent_obs, effective_size),
                opponent_policy_mode,
            )
            actions = jax.lax.cond(
                policy_player == 0,
                lambda _: jnp.stack([learner_action, opponent_action_value]),
                lambda _: jnp.stack([opponent_action_value, learner_action]),
                None,
            )
            next_state, _ = game.step(rollout_state, actions)
            current_info = game.get_info(rollout_state)
            already_done = current_info.is_done | (rollout_state.time >= max_steps)
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            final_info = game.get_info(final_state)
            final_scoreboard = reset_adaptive_scoreboard_history(next_scoreboard, final_info.is_done)
            final_memory = scalar_reset_fog_memory(next_memory, final_info.is_done)
            return (final_state, final_scoreboard, final_memory, step_key), None

        (final_state, _, _, _), _ = jax.lax.scan(
            body,
            (initial_state, current_scoreboard, fog_memory, rollout_key),
            None,
            length=rollout_steps,
        )
        final_info = game.get_info(final_state)
        truncated = (final_state.time >= max_steps) & ~final_info.is_done
        scored_info = final_info._replace(winner=jnp.where(truncated, -1, final_info.winner))
        final_obs = game.get_observation(final_state, policy_player)
        return search_score_observation(scored_info, final_obs, policy_player, army_weight, land_weight, terminal_score)

    def score_candidate(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        rollout_scores = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        first_terminal = jnp.where(
            first_info.winner == policy_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        return first_terminal + jnp.mean(rollout_scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(score_candidate)(candidate_actions, prior_scores, candidate_keys)
    best_position = jnp.argmax(scores)
    best_action = candidate_actions[best_position]
    score_gap = jnp.asarray(jnp.inf, dtype=scores.dtype)
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(scores, 2)
        score_gap = top_scores[0] - top_scores[1]
    accept_search = score_gap >= min_score_gap
    if online_search_gate_threshold >= 0.0:
        gate_features = online_search_gate_features(
            scores,
            prior_scores,
            best_position,
            adaptive_action_to_index(best_action, pad_size),
            logits,
            fallback_action,
            active,
            obs_arr,
            state.time,
            policy_player,
            max_steps,
            pad_size,
            online_search_gate_feature_dim,
        )
        gate_probability = jax.nn.sigmoid(command_gate_network(gate_features))
        accept_search = accept_search & (gate_probability >= online_search_gate_threshold)
    return jnp.where(accept_search, best_action, fallback_action)


def online_search_action_heuristic_opponent(
    network,
    policy_adapter_network,
    command_gate_network,
    opponent,
    state,
    effective_size: int,
    key,
    fallback_action: jnp.ndarray,
    opponent_first_action: jnp.ndarray,
    policy_player: int,
    policy_mode: int,
    pad_size: int,
    max_steps: int,
    global_context: bool,
    scoreboard_history_enabled: bool,
    previous_scoreboard: jnp.ndarray,
    fog_memory_enabled: bool,
    fog_memory,
    policy_adapter_scale: float,
    policy_adapter_mode: int,
    policy_adapter_min_grid_size: int,
    policy_adapter_max_grid_size: int,
    top_k: int,
    rollout_steps: int,
    rollouts_per_action: int,
    army_weight: float,
    land_weight: float,
    prior_weight: float,
    terminal_score: float,
    min_score_gap: float,
    online_search_gate_threshold: float,
    online_search_gate_feature_dim: int,
) -> jnp.ndarray:
    """Choose a primitive action by online counterfactual rollout search against a heuristic opponent."""
    obs = game.get_observation(state, policy_player)
    current_scoreboard = adaptive_scoreboard_features(obs, effective_size)
    history_context = (
        adaptive_scoreboard_history_context(previous_scoreboard, current_scoreboard)
        if scoreboard_history_enabled
        else None
    )
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size,
        pad_size,
        include_global_context=global_context,
        scoreboard_history=history_context,
        fog_memory=fog_memory if fog_memory_enabled else None,
    )
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits = adapter_composed_policy_logits(
        network,
        policy_adapter_network,
        obs_arr,
        mask,
        active,
        effective_size,
        policy_adapter_scale,
        policy_adapter_mode,
        policy_adapter_min_grid_size,
        policy_adapter_max_grid_size,
    )
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda index: adaptive_index_to_action(index, pad_size))(candidate_indices)
    opponent_player = 1 - policy_player

    def rollout_result(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, prev_scoreboard, memory, step_key = carry
            step_key, learner_key, opponent_key = jrandom.split(step_key, 3)
            learner_obs = game.get_observation(rollout_state, policy_player)
            learner_action, next_scoreboard, next_memory = adapter_policy_action_with_memory(
                network,
                policy_adapter_network,
                learner_obs,
                effective_size,
                learner_key,
                policy_mode,
                pad_size,
                global_context,
                scoreboard_history_enabled,
                prev_scoreboard,
                fog_memory_enabled,
                memory,
                policy_adapter_scale,
                policy_adapter_mode,
                policy_adapter_min_grid_size,
                policy_adapter_max_grid_size,
            )
            opponent_obs = game.get_observation(rollout_state, opponent_player)
            opponent_action_value = opponent_action(opponent, opponent_key, opponent_obs, random_action)
            actions = jax.lax.cond(
                policy_player == 0,
                lambda _: jnp.stack([learner_action, opponent_action_value]),
                lambda _: jnp.stack([opponent_action_value, learner_action]),
                None,
            )
            next_state, _ = game.step(rollout_state, actions)
            current_info = game.get_info(rollout_state)
            already_done = current_info.is_done | (rollout_state.time >= max_steps)
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            final_info = game.get_info(final_state)
            final_scoreboard = reset_adaptive_scoreboard_history(next_scoreboard, final_info.is_done)
            final_memory = scalar_reset_fog_memory(next_memory, final_info.is_done)
            return (final_state, final_scoreboard, final_memory, step_key), None

        (final_state, _, _, _), _ = jax.lax.scan(
            body,
            (initial_state, current_scoreboard, fog_memory, rollout_key),
            None,
            length=rollout_steps,
        )
        final_info = game.get_info(final_state)
        truncated = (final_state.time >= max_steps) & ~final_info.is_done
        scored_info = final_info._replace(winner=jnp.where(truncated, -1, final_info.winner))
        final_obs = game.get_observation(final_state, policy_player)
        return search_score_observation(scored_info, final_obs, policy_player, army_weight, land_weight, terminal_score)

    def score_candidate(action, prior_score, candidate_key):
        first_actions = jax.lax.cond(
            policy_player == 0,
            lambda _: jnp.stack([action, opponent_first_action]),
            lambda _: jnp.stack([opponent_first_action, action]),
            None,
        )
        next_state, first_info = game.step(state, first_actions)
        rollout_keys = jrandom.split(candidate_key, rollouts_per_action)
        rollout_scores = jax.vmap(lambda rollout_key: rollout_result(next_state, rollout_key))(rollout_keys)
        first_terminal = jnp.where(
            first_info.winner == policy_player,
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        return first_terminal + jnp.mean(rollout_scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(score_candidate)(candidate_actions, prior_scores, candidate_keys)
    best_position = jnp.argmax(scores)
    best_action = candidate_actions[best_position]
    score_gap = jnp.asarray(jnp.inf, dtype=scores.dtype)
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(scores, 2)
        score_gap = top_scores[0] - top_scores[1]
    accept_search = score_gap >= min_score_gap
    if online_search_gate_threshold >= 0.0:
        gate_features = online_search_gate_features(
            scores,
            prior_scores,
            best_position,
            adaptive_action_to_index(best_action, pad_size),
            logits,
            fallback_action,
            active,
            obs_arr,
            state.time,
            policy_player,
            max_steps,
            pad_size,
            online_search_gate_feature_dim,
        )
        gate_probability = jax.nn.sigmoid(command_gate_network(gate_features))
        accept_search = accept_search & (gate_probability >= online_search_gate_threshold)
    return jnp.where(accept_search, best_action, fallback_action)


def online_search_gate_features(
    scores: jnp.ndarray,
    prior_scores: jnp.ndarray,
    best_position: jnp.ndarray,
    best_action_index: jnp.ndarray,
    policy_logits: jnp.ndarray,
    fallback_action: jnp.ndarray,
    active: jnp.ndarray,
    obs_arr: jnp.ndarray,
    time: jnp.ndarray,
    policy_player: int,
    max_steps: int,
    pad_size: int,
    feature_dim: int = ONLINE_SEARCH_GATE_FEATURE_DIM,
) -> jnp.ndarray:
    """Build online-search accept/reject features matching the offline trainer."""
    top_k = scores.shape[0]
    if top_k >= 2:
        top_scores, _ = jax.lax.top_k(scores, 2)
        second_score = top_scores[1]
        top_prior, _ = jax.lax.top_k(prior_scores, 2)
        second_prior = top_prior[1]
        prior_gap = top_prior[0] - top_prior[1]
    else:
        second_score = scores[best_position]
        second_prior = prior_scores[best_position]
        prior_gap = jnp.asarray(0.0, dtype=prior_scores.dtype)
    best_score = scores[best_position]
    mean_score = jnp.mean(scores)
    std_score = jnp.std(scores)
    best_prior = prior_scores[best_position]
    fallback_index = adaptive_action_to_index(fallback_action, pad_size)
    fallback_prior = policy_logits[fallback_index]
    search_action_changed = (best_action_index != fallback_index).astype(jnp.float32)
    active_float = active.astype(jnp.float32)
    active_fraction = jnp.mean(active_float)
    active_count = jnp.maximum(jnp.sum(active_float), 1.0)
    visible_enemy_density = jnp.sum(obs_arr[6] * active_float) / active_count
    contact = (visible_enemy_density > 0.0).astype(jnp.float32)
    denom = jnp.maximum(jnp.asarray(top_k - 1, dtype=jnp.float32), 1.0)
    features = jnp.stack(
        [
            best_score,
            second_score,
            best_score - second_score,
            mean_score,
            std_score,
            best_prior,
            second_prior,
            prior_gap,
            best_position.astype(jnp.float32) / denom,
            fallback_prior,
            best_prior - fallback_prior,
            search_action_changed,
            time.astype(jnp.float32) / jnp.asarray(max(max_steps, 1), dtype=jnp.float32),
            jnp.asarray(policy_player, dtype=jnp.float32),
            active_fraction,
            visible_enemy_density,
            contact,
        ]
    )
    return features[:feature_dim]
