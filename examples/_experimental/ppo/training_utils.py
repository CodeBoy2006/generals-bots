"""Shared PPO rollout rewards, observation selection, and update calculations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.rewards import general_target_reward_fn, path_assignment_reward_fn


def select_learner_obs(obs_p0, obs_p1, learner_player):
    """Select observations from the learner player's perspective."""
    return jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)


def select_opponent_obs(obs_p0, obs_p1, learner_player):
    """Select observations from the opponent player's perspective."""
    return jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)


def apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale):
    """Add an optional zero-sum win/loss reward on decisive terminal transitions."""
    opponent_player = 1 - learner_player
    terminal_bonus = jnp.where(
        infos.winner == learner_player,
        terminal_reward_scale,
        jnp.where(infos.winner == opponent_player, -terminal_reward_scale, 0.0),
    )
    terminal_bonus = jnp.where(infos.is_done & (infos.winner >= 0), terminal_bonus, 0.0)
    return rewards + terminal_bonus


def apply_general_target_rewards(
    rewards,
    prior_states,
    states,
    learner_player,
    general_target_reward_scale,
    general_target_max_distance,
    general_target_min_army,
):
    """Add state-aware pressure shaping toward the opponent general."""
    shaping = jax.vmap(
        lambda prior_state, state: general_target_reward_fn(
            prior_state,
            state,
            learner_player,
            general_target_reward_scale,
            general_target_max_distance,
            general_target_min_army,
        )
    )(prior_states, states)
    return rewards + shaping


def apply_path_assignment_rewards(
    rewards,
    prior_states,
    states,
    learner_player,
    path_assignment_reward_scale,
    path_assignment_max_distance,
    path_assignment_min_army,
    path_assignment_general_weight,
    path_assignment_city_weight,
    path_assignment_frontier_weight,
):
    """Add shortest-path target-assignment shaping for transport progress."""
    if path_assignment_reward_scale <= 0.0:
        return rewards

    shaping = jax.vmap(
        lambda prior_state, state: path_assignment_reward_fn(
            prior_state,
            state,
            learner_player,
            path_assignment_reward_scale,
            path_assignment_max_distance,
            path_assignment_min_army,
            path_assignment_general_weight,
            path_assignment_city_weight,
            path_assignment_frontier_weight,
        )
    )(prior_states, states)
    return rewards + shaping


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and value returns."""
    num_steps, num_envs = rewards.shape
    values_with_bootstrap = jnp.concatenate([values, jnp.zeros((1, num_envs))], axis=0)

    def gae_step(carry, inputs):
        last_adv = carry
        reward, value, next_value, done = inputs
        nonterminal = 1.0 - done
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * lam * nonterminal * last_adv
        return advantage, advantage

    inputs = (
        rewards[::-1],
        values[::-1],
        values_with_bootstrap[1:][::-1],
        dones[::-1],
    )
    _, advantages_rev = jax.lax.scan(gae_step, jnp.zeros(num_envs), inputs)
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


@jax.jit
def ppo_loss(network, obs, mask, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for single sample."""
    _, value, logprob, entropy = network(obs, mask, None, action)

    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)

    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy

    return policy_loss + value_loss + entropy_loss


def select_search_improvements(
    candidate_indices,
    search_scores,
    min_margin: float,
    margin_scale: float,
    max_weight: float,
):
    """Select high-confidence search improvements over the base top-prior action."""
    best_positions = jnp.argmax(search_scores, axis=-1)
    target_indices = jnp.take_along_axis(candidate_indices, best_positions[..., None], axis=-1)[..., 0]
    best_scores = jnp.take_along_axis(search_scores, best_positions[..., None], axis=-1)[..., 0]
    base_indices = candidate_indices[..., 0]
    base_scores = search_scores[..., 0]
    margins = best_scores - base_scores

    switched = target_indices != base_indices
    scaled_weights = (margins - min_margin) / margin_scale
    clipped_weights = jnp.clip(scaled_weights, 0.0, max_weight)
    weights = jnp.where(switched & (margins >= min_margin), clipped_weights, 0.0)
    return target_indices.astype(jnp.int32), weights.astype(jnp.float32), margins.astype(jnp.float32)


def search_score_target_probs(search_scores, temperature: float):
    """Convert top-k rollout-search scores into a stable soft target distribution."""
    centered_scores = search_scores - jnp.max(search_scores, axis=-1, keepdims=True)
    return jax.nn.softmax(centered_scores / temperature, axis=-1)


def weighted_topk_cross_entropy(log_probs, candidate_indices, target_probs, weights):
    """Return weighted CE over sparse top-k candidate targets."""
    candidate_log_probs = jnp.take_along_axis(log_probs, candidate_indices, axis=1)
    losses = -jnp.sum(target_probs * candidate_log_probs, axis=-1)
    normalizer = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(losses * weights) / normalizer
