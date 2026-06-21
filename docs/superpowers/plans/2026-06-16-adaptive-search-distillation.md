# Adaptive Search Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first adaptive conservative rollout-search distillation trainer that emits one checkpoint usable on 8x8, 12x12, and 16x16 padded boards.

**Architecture:** Add one new adaptive script that mirrors fixed-size `conservative_search_distill.py` while using adaptive encoders, adaptive action indices, active-cell masks, and adaptive state pools. The first slice uses the frozen adaptive base checkpoint as search prior, KL anchor, rollout policy, and opponent.

**Tech Stack:** Python, JAX, Equinox, Optax, existing Generals core game API, existing adaptive PPO helpers.

---

## File Structure

- Create: `examples/_experimental/ppo/adaptive_search_distill.py`
  - Owns adaptive rollout-search scoring, adaptive conservative losses, batch collection, train epochs, CLI parsing, checkpoint retention, and final checkpoint saving.
- Modify: `tests/test_adaptive_ppo.py`
  - Adds focused helper, loss, collection, and CLI smoke tests for adaptive search distillation.
- Modify: `statusquo.md`
  - Append one ledger entry after implementation and verification.
- Leave unchanged: `examples/_experimental/ppo/conservative_search_distill.py`
  - Fixed-size script stays stable; adaptive script imports reusable loss helpers where shapes match.
- Leave unchanged: `README.md`
  - No public setup step or stable user-facing API changes are introduced in this experimental script.

## Task 1: Adaptive Loss Surface

**Files:**
- Create: `examples/_experimental/ppo/adaptive_search_distill.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write the failing loss test**

Append this test to `tests/test_adaptive_ppo.py` after `test_adaptive_expander_target_probs_has_single_pass_slot`:

```python
def test_adaptive_soft_conservative_loss_is_finite_for_matching_networks():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import (
        compute_adaptive_soft_conservative_loss,
        search_score_target_probs,
    )

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    obs = jnp.zeros((2, ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    masks = jnp.ones((2, 6, 6, 4), dtype=bool)
    active = jnp.ones((2, 6, 6), dtype=bool)
    candidate_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    search_scores = jnp.array([[1.0, 2.0], [4.0, 4.0]], dtype=jnp.float32)
    target_probs = search_score_target_probs(search_scores, temperature=1.0)
    search_weights = jnp.ones((2,), dtype=jnp.float32)
    kl_weights = jnp.ones((2,), dtype=jnp.float32)

    loss, metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        kl_weights,
        kl_weight=1.0,
        improve_weight=0.05,
        temperature=1.0,
    )

    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["kl_loss"])
    assert jnp.isfinite(metrics["improve_loss"])
    assert jnp.allclose(jnp.sum(target_probs, axis=1), jnp.ones((2,), dtype=jnp.float32))
```

- [ ] **Step 2: Run the test and verify import failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_adaptive_soft_conservative_loss_is_finite_for_matching_networks
```

Expected: FAIL with `ModuleNotFoundError: No module named 'examples._experimental.ppo.adaptive_search_distill'`.

- [ ] **Step 3: Create the adaptive script with reusable adaptive losses**

Create `examples/_experimental/ppo/adaptive_search_distill.py` with this initial content:

```python
"""Adaptive conservative rollout-search distillation for multisize checkpoints."""

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

from generals.agents.ppo_policy_agent import parse_policy_channels
from generals.core import game
from generals.core.game import GameInfo
from generals.core.observation import Observation

from adaptive_common import (
    ADAPTIVE_INPUT_CHANNELS,
    adaptive_index_to_action,
    adaptive_obs_to_array,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_size_weights,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import POLICY_MODE_NAME_TO_ID, POLICY_MODE_NAMES
from conservative_search_distill import (
    search_score_target_probs,
    select_search_improvements,
    weighted_topk_cross_entropy,
)
from train import checkpoint_path_for_iteration, prune_old_checkpoints, stack_learner_actions

TARGET_MODE_NAMES = ("hard", "soft")


def compute_adaptive_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    target_indices,
    improve_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return adaptive KL-to-base plus weighted hard search-target loss."""

    def logits_for_sample(network, sample_obs, sample_mask, sample_active):
        logits, _ = network.logits_value(sample_obs, sample_mask, sample_active)
        return logits

    student_logits = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: logits_for_sample(
            student_network,
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)
    base_logits = jax.lax.stop_gradient(
        jax.vmap(
            lambda sample_obs, sample_mask, sample_active: logits_for_sample(
                base_network,
                sample_obs,
                sample_mask,
                sample_active,
            )
        )(base_obs, base_masks, base_active)
    )

    student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    base_log_probs = jax.nn.log_softmax(base_logits / temperature, axis=-1)
    base_probs = jax.nn.softmax(base_logits / temperature, axis=-1)
    kl_per_sample = jnp.sum(base_probs * (base_log_probs - student_log_probs_for_kl), axis=-1)
    kl_normalizer = jnp.maximum(jnp.sum(kl_weights), 1.0)
    kl_loss = jnp.sum(kl_per_sample * kl_weights) / kl_normalizer

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    action_losses = -jnp.take_along_axis(student_log_probs, target_indices[:, None], axis=1)[:, 0]
    improve_normalizer = jnp.maximum(jnp.sum(improve_weights), 1.0)
    improve_loss = jnp.sum(action_losses * improve_weights) / improve_normalizer

    loss = kl_weight * kl_loss + improve_weight * improve_loss
    selected = improve_weights > 0.0
    selected_count = jnp.sum(selected.astype(jnp.float32))
    active_count = jnp.maximum(jnp.sum(kl_weights), 1.0)
    accuracy = jnp.sum((jnp.argmax(student_logits, axis=-1) == target_indices) * improve_weights) / improve_normalizer
    metrics = {
        "kl_loss": kl_loss,
        "improve_loss": jnp.where(selected_count > 0.0, improve_loss, 0.0),
        "selected_fraction": selected_count / active_count,
        "accuracy": jnp.where(selected_count > 0.0, accuracy, 0.0),
    }
    return loss, metrics


def compute_adaptive_soft_conservative_loss(
    student_network,
    base_network,
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    candidate_indices,
    target_probs,
    search_weights,
    kl_weights,
    kl_weight: float,
    improve_weight: float,
    temperature: float,
):
    """Return adaptive KL-to-base plus weighted soft top-k search-target loss."""

    def logits_for_sample(network, sample_obs, sample_mask, sample_active):
        logits, _ = network.logits_value(sample_obs, sample_mask, sample_active)
        return logits

    student_logits = jax.vmap(
        lambda sample_obs, sample_mask, sample_active: logits_for_sample(
            student_network,
            sample_obs,
            sample_mask,
            sample_active,
        )
    )(obs, masks, active)
    base_logits = jax.lax.stop_gradient(
        jax.vmap(
            lambda sample_obs, sample_mask, sample_active: logits_for_sample(
                base_network,
                sample_obs,
                sample_mask,
                sample_active,
            )
        )(base_obs, base_masks, base_active)
    )

    student_log_probs_for_kl = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    base_log_probs = jax.nn.log_softmax(base_logits / temperature, axis=-1)
    base_probs = jax.nn.softmax(base_logits / temperature, axis=-1)
    kl_per_sample = jnp.sum(base_probs * (base_log_probs - student_log_probs_for_kl), axis=-1)
    kl_normalizer = jnp.maximum(jnp.sum(kl_weights), 1.0)
    kl_loss = jnp.sum(kl_per_sample * kl_weights) / kl_normalizer

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    search_loss = weighted_topk_cross_entropy(student_log_probs, candidate_indices, target_probs, search_weights)
    loss = kl_weight * kl_loss + improve_weight * search_loss

    best_targets = jnp.take_along_axis(candidate_indices, jnp.argmax(target_probs, axis=-1)[:, None], axis=1)[:, 0]
    search_normalizer = jnp.maximum(jnp.sum(search_weights), 1.0)
    accuracy = jnp.sum((jnp.argmax(student_logits, axis=-1) == best_targets) * search_weights) / search_normalizer
    target_entropy = -jnp.sum(target_probs * jnp.log(jnp.clip(target_probs, 1e-8, 1.0)), axis=-1)
    metrics = {
        "kl_loss": kl_loss,
        "improve_loss": search_loss,
        "selected_fraction": jnp.sum(search_weights) / kl_normalizer,
        "accuracy": accuracy,
        "target_entropy": jnp.sum(target_entropy * search_weights) / search_normalizer,
    }
    return loss, metrics
```

- [ ] **Step 4: Run the focused loss test**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_adaptive_soft_conservative_loss_is_finite_for_matching_networks
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add examples/_experimental/ppo/adaptive_search_distill.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive search distillation losses"
```

## Task 2: Adaptive Rollout Search Helpers

**Files:**
- Modify: `examples/_experimental/ppo/adaptive_search_distill.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write the failing candidate-search test**

Append this test after the Task 1 test in `tests/test_adaptive_ppo.py`:

```python
def test_adaptive_rollout_search_candidates_respects_effective_size():
    from examples._experimental.ppo.adaptive_common import adaptive_action_space_size
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import adaptive_rollout_search_candidates

    pad_size = 6
    effective_size = 4
    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=pad_size, channels=(16, 16, 16, 8))
    state = make_padded_state(size=effective_size, pad_to=pad_size)

    candidate_actions, candidate_indices, prior_scores, search_scores = adaptive_rollout_search_candidates(
        network,
        state,
        jnp.asarray(effective_size, dtype=jnp.int32),
        jrandom.PRNGKey(1),
        player=0,
        top_k=2,
        rollout_steps=1,
        rollouts_per_action=1,
        policy_mode=0,
        army_weight=12.0,
        land_weight=8.0,
        prior_weight=0.01,
        terminal_score=1000.0,
        pad_size=pad_size,
    )

    assert candidate_actions.shape == (2, 5)
    assert candidate_indices.shape == (2,)
    assert prior_scores.shape == (2,)
    assert search_scores.shape == (2,)
    assert jnp.all(candidate_indices >= 0)
    assert jnp.all(candidate_indices < adaptive_action_space_size(pad_size))
    non_pass = candidate_actions[:, 0] == 0
    assert jnp.all(candidate_actions[:, 1] < effective_size) | jnp.all(~non_pass)
    assert jnp.all(candidate_actions[:, 2] < effective_size) | jnp.all(~non_pass)
    assert jnp.all(jnp.isfinite(search_scores))
```

- [ ] **Step 2: Run the candidate-search test and verify missing symbol**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_adaptive_rollout_search_candidates_respects_effective_size
```

Expected: FAIL with `ImportError` for `adaptive_rollout_search_candidates`.

- [ ] **Step 3: Add adaptive policy and search helpers**

Append these functions to `examples/_experimental/ppo/adaptive_search_distill.py` after the loss functions:

```python
def adaptive_score_observation(
    info: GameInfo,
    obs: Observation,
    player: int,
    army_weight: float = 12.0,
    land_weight: float = 8.0,
    terminal_score: float = 1000.0,
):
    """Score a final adaptive rollout observation from one player's perspective."""
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


def adaptive_policy_action(network, obs, effective_size, key, policy_mode, pad_size: int):
    """Dispatch an adaptive checkpoint action using greedy or sampled execution."""
    obs_arr, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    mask = compute_adaptive_valid_move_mask(
        obs.armies,
        obs.owned_cells,
        obs.mountains,
        effective_size,
        pad_size,
    )
    logits, _ = network.logits_value(obs_arr, mask, active)
    index = jax.lax.cond(
        policy_mode == 0,
        lambda _: jnp.argmax(logits),
        lambda _: jrandom.categorical(key, logits),
        None,
    )
    return adaptive_index_to_action(index, pad_size)


@eqx.filter_jit
def adaptive_rollout_search_candidates(
    network,
    state,
    effective_size,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    pad_size,
):
    """Return adaptive top-k prior candidates and short rollout-search scores."""
    obs = game.get_observation(state, player)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size, pad_size)
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    logits, _ = network.logits_value(obs_arr, mask, active)
    prior_scores, candidate_indices = jax.lax.top_k(logits, top_k)
    candidate_actions = jax.vmap(lambda idx: adaptive_index_to_action(idx, pad_size))(candidate_indices)

    opponent_player = 1 - player
    opponent_obs = game.get_observation(state, opponent_player)
    key, opponent_key = jrandom.split(key)
    opponent_first_action = adaptive_policy_action(
        network,
        opponent_obs,
        effective_size,
        opponent_key,
        policy_mode,
        pad_size,
    )

    def rollout_score(initial_state, rollout_key):
        def body(carry, _):
            rollout_state, step_key = carry
            step_key, k0, k1 = jrandom.split(step_key, 3)
            obs_p0 = game.get_observation(rollout_state, 0)
            obs_p1 = game.get_observation(rollout_state, 1)
            action_p0 = adaptive_policy_action(network, obs_p0, effective_size, k0, policy_mode, pad_size)
            action_p1 = adaptive_policy_action(network, obs_p1, effective_size, k1, policy_mode, pad_size)
            next_state, _ = game.step(rollout_state, jnp.stack([action_p0, action_p1]))
            already_done = game.get_info(rollout_state).is_done
            final_state = jax.tree.map(lambda old, new: jnp.where(already_done, old, new), rollout_state, next_state)
            return (final_state, step_key), None

        (final_state, _), _ = jax.lax.scan(body, (initial_state, rollout_key), None, length=rollout_steps)
        final_info = game.get_info(final_state)
        final_obs = game.get_observation(final_state, player)
        return adaptive_score_observation(final_info, final_obs, player, army_weight, land_weight, terminal_score)

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
            terminal_score,
            jnp.where(first_info.winner == opponent_player, -terminal_score, 0.0),
        )
        return first_terminal + jnp.mean(scores) + prior_weight * prior_score

    candidate_keys = jrandom.split(key, top_k)
    scores = jax.vmap(candidate_score)(candidate_actions, prior_scores, candidate_keys)
    return candidate_actions, candidate_indices, prior_scores, scores


@eqx.filter_jit
def adaptive_rollout_search_action(
    network,
    state,
    effective_size,
    key,
    player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    policy_mode,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    pad_size,
):
    """Choose one adaptive action by scoring top-k prior actions with short rollouts."""
    candidate_actions, _, _, scores = adaptive_rollout_search_candidates(
        network,
        state,
        effective_size,
        key,
        player,
        top_k,
        rollout_steps,
        rollouts_per_action,
        policy_mode,
        army_weight,
        land_weight,
        prior_weight,
        terminal_score,
        pad_size,
    )
    return candidate_actions[jnp.argmax(scores)]
```

- [ ] **Step 4: Run the candidate-search test**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_adaptive_rollout_search_candidates_respects_effective_size
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add examples/_experimental/ppo/adaptive_search_distill.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive rollout search helpers"
```

## Task 3: Adaptive Collection And Training Epochs

**Files:**
- Modify: `examples/_experimental/ppo/adaptive_search_distill.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write the failing soft-batch collection test**

Append this test after the candidate-search test:

```python
def test_collect_adaptive_soft_batch_returns_expected_shapes():
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import collect_adaptive_soft_batch

    pad_size = 6
    num_envs = 2
    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=pad_size, channels=(16, 16, 16, 8))
    states = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        make_padded_state(size=4, pad_to=pad_size),
        make_padded_state(size=6, pad_to=pad_size),
    )
    effective_sizes = jnp.array([4, 6], dtype=jnp.int32)

    _, batch, _ = collect_adaptive_soft_batch(
        network,
        network,
        network,
        states,
        effective_sizes,
        jrandom.PRNGKey(2),
        num_steps=1,
        policy_mode=0,
        opponent_policy_mode=0,
        learner_player=0,
        top_k=2,
        rollout_steps=1,
        rollouts_per_action=1,
        army_weight=12.0,
        land_weight=8.0,
        prior_weight=0.01,
        terminal_score=1000.0,
        score_temperature=1.0,
        pad_size=pad_size,
    )

    obs, masks, active, base_obs, base_masks, base_active, candidate_indices, target_probs, search_weights, kl_weights = batch
    assert obs.shape[:2] == (1, num_envs)
    assert masks.shape == (1, num_envs, pad_size, pad_size, 4)
    assert active.shape == (1, num_envs, pad_size, pad_size)
    assert base_obs.shape == obs.shape
    assert base_masks.shape == masks.shape
    assert base_active.shape == active.shape
    assert candidate_indices.shape == (1, num_envs, 2)
    assert target_probs.shape == (1, num_envs, 2)
    assert search_weights.shape == (1, num_envs)
    assert kl_weights.shape == (1, num_envs)
    assert jnp.allclose(jnp.sum(target_probs, axis=-1), jnp.ones((1, num_envs), dtype=jnp.float32))
```

Also add `import jax` near the top of `tests/test_adaptive_ppo.py` if it is not present:

```python
import jax
```

- [ ] **Step 2: Run the collection test and verify missing symbol**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_collect_adaptive_soft_batch_returns_expected_shapes
```

Expected: FAIL with `ImportError` for `collect_adaptive_soft_batch`.

- [ ] **Step 3: Add flattening, train minibatches, epochs, and collectors**

Append these functions after `adaptive_rollout_search_action` in `examples/_experimental/ppo/adaptive_search_distill.py`:

```python
@eqx.filter_jit
def train_adaptive_conservative_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
):
    """Train one adaptive hard-target distillation minibatch."""
    obs, masks, active, base_obs, base_masks, base_active, target_indices, improve_weights, kl_weights = minibatch

    def loss_fn(net):
        return compute_adaptive_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            active,
            base_obs,
            base_masks,
            base_active,
            target_indices,
            improve_weights,
            kl_weights,
            kl_weight,
            improve_weight,
            temperature,
        )

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(student_network)
    params = eqx.filter(student_network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    student_network = eqx.apply_updates(student_network, updates)
    return student_network, opt_state, loss, metrics


@eqx.filter_jit
def train_adaptive_soft_minibatch(
    student_network,
    base_network,
    opt_state,
    minibatch,
    optimizer,
    kl_weight,
    improve_weight,
    temperature,
):
    """Train one adaptive soft-target distillation minibatch."""
    obs, masks, active, base_obs, base_masks, base_active, candidate_indices, target_probs, search_weights, kl_weights = minibatch

    def loss_fn(net):
        return compute_adaptive_soft_conservative_loss(
            net,
            base_network,
            obs,
            masks,
            active,
            base_obs,
            base_masks,
            base_active,
            candidate_indices,
            target_probs,
            search_weights,
            kl_weights,
            kl_weight,
            improve_weight,
            temperature,
        )

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(student_network)
    params = eqx.filter(student_network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    student_network = eqx.apply_updates(student_network, updates)
    return student_network, opt_state, loss, metrics


def flatten_adaptive_conservative_batch(
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    target_indices,
    improve_weights,
    kl_weights,
    margins,
):
    """Flatten time/environment axes for adaptive hard-target distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        base_active.reshape(batch_size, *base_active.shape[2:]),
        target_indices.reshape(batch_size),
        improve_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
        margins.reshape(batch_size),
    )


def flatten_adaptive_soft_batch(
    obs,
    masks,
    active,
    base_obs,
    base_masks,
    base_active,
    candidate_indices,
    target_probs,
    search_weights,
    kl_weights,
):
    """Flatten time/environment axes for adaptive soft-target distillation."""
    batch_size = obs.shape[0] * obs.shape[1]
    return (
        obs.reshape(batch_size, *obs.shape[2:]),
        masks.reshape(batch_size, *masks.shape[2:]),
        active.reshape(batch_size, *active.shape[2:]),
        base_obs.reshape(batch_size, *base_obs.shape[2:]),
        base_masks.reshape(batch_size, *base_masks.shape[2:]),
        base_active.reshape(batch_size, *base_active.shape[2:]),
        candidate_indices.reshape(batch_size, *candidate_indices.shape[2:]),
        target_probs.reshape(batch_size, *target_probs.shape[2:]),
        search_weights.reshape(batch_size),
        kl_weights.reshape(batch_size),
    )


def train_adaptive_conservative_epoch(
    student_network,
    base_network,
    opt_state,
    flat_batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    improve_weight,
    temperature,
):
    """Run adaptive hard-target distillation over shuffled minibatches."""
    obs, masks, active, base_obs, base_masks, base_active, target_indices, improve_weights, kl_weights, margins = flat_batch
    batch_size = obs.shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = (
            obs[permutation],
            masks[permutation],
            active[permutation],
            base_obs[permutation],
            base_masks[permutation],
            base_active[permutation],
            target_indices[permutation],
            improve_weights[permutation],
            kl_weights[permutation],
        )
        epoch_loss = 0.0
        epoch_metrics = None

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            student_network, opt_state, loss, metrics = train_adaptive_conservative_minibatch(
                student_network,
                base_network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                improve_weight,
                temperature,
            )
            epoch_loss += loss
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, metrics)

        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda x: x / num_complete_batches, epoch_metrics)

    selected_margins = jnp.where(improve_weights > 0.0, margins, 0.0)
    selected_count = jnp.maximum(jnp.sum((improve_weights > 0.0).astype(jnp.float32)), 1.0)
    avg_metrics = dict(avg_metrics)
    avg_metrics["mean_selected_margin"] = jnp.sum(selected_margins) / selected_count
    avg_metrics["selected_samples"] = jnp.sum((improve_weights > 0.0).astype(jnp.float32))
    return student_network, opt_state, avg_loss, avg_metrics, key


def train_adaptive_soft_epoch(
    student_network,
    base_network,
    opt_state,
    flat_batch,
    optimizer,
    key,
    num_epochs,
    minibatch_size,
    kl_weight,
    improve_weight,
    temperature,
):
    """Run adaptive soft-target distillation over shuffled minibatches."""
    obs, masks, active, base_obs, base_masks, base_active, candidate_indices, target_probs, search_weights, kl_weights = flat_batch
    batch_size = obs.shape[0]
    actual_minibatch_size = min(minibatch_size, batch_size)
    num_complete_batches = max(batch_size // actual_minibatch_size, 1)
    avg_loss = 0.0
    avg_metrics = None

    for _ in range(num_epochs):
        key, permutation_key = jrandom.split(key)
        permutation = jrandom.permutation(permutation_key, batch_size)
        shuffled = (
            obs[permutation],
            masks[permutation],
            active[permutation],
            base_obs[permutation],
            base_masks[permutation],
            base_active[permutation],
            candidate_indices[permutation],
            target_probs[permutation],
            search_weights[permutation],
            kl_weights[permutation],
        )
        epoch_loss = 0.0
        epoch_metrics = None

        for batch_idx in range(num_complete_batches):
            start = batch_idx * actual_minibatch_size
            end = start + actual_minibatch_size
            minibatch = tuple(x[start:end] for x in shuffled)
            student_network, opt_state, loss, metrics = train_adaptive_soft_minibatch(
                student_network,
                base_network,
                opt_state,
                minibatch,
                optimizer,
                kl_weight,
                improve_weight,
                temperature,
            )
            epoch_loss += loss
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, metrics)

        avg_loss = epoch_loss / num_complete_batches
        avg_metrics = jax.tree.map(lambda x: x / num_complete_batches, epoch_metrics)

    avg_metrics = dict(avg_metrics)
    avg_metrics["selected_samples"] = jnp.sum((search_weights > 0.0).astype(jnp.float32))
    avg_metrics["mean_selected_margin"] = 0.0
    return student_network, opt_state, avg_loss, avg_metrics, key


@eqx.filter_jit
def collect_adaptive_conservative_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    effective_sizes,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    min_margin,
    margin_scale,
    max_weight,
    pad_size,
):
    """Collect adaptive learner states labeled by hard search improvements."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        is_active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        learner_obs_arr, active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
            learner_obs,
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
        )(learner_obs, effective_sizes)
        base_obs_arr = learner_obs_arr
        base_masks = masks
        base_active = active

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores = jax.vmap(
            lambda state, size, sample_key: adaptive_rollout_search_candidates(
                base_network,
                state,
                size,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
                terminal_score,
                pad_size,
            )
        )(states, effective_sizes, search_keys)
        target_indices, improve_weights, margins = select_search_improvements(
            candidate_indices,
            search_scores,
            min_margin,
            margin_scale,
            max_weight,
        )
        active_weights = is_active.astype(jnp.float32)
        improve_weights = improve_weights * active_weights

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                student_network,
                obs,
                size,
                sample_key,
                policy_mode,
                pad_size,
            )
        )(learner_obs, effective_sizes, learner_keys)
        opponent_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                opponent_network,
                obs,
                size,
                sample_key,
                opponent_policy_mode,
                pad_size,
            )
        )(opponent_obs, effective_sizes, opponent_keys)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(is_active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        return (final_states, key), (
            learner_obs_arr,
            masks,
            active,
            base_obs_arr,
            base_masks,
            base_active,
            target_indices,
            improve_weights,
            active_weights,
            margins,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=num_steps)
    return states, batch, key


@eqx.filter_jit
def collect_adaptive_soft_batch(
    student_network,
    base_network,
    opponent_network,
    states,
    effective_sizes,
    key,
    num_steps,
    policy_mode,
    opponent_policy_mode,
    learner_player,
    top_k,
    rollout_steps,
    rollouts_per_action,
    army_weight,
    land_weight,
    prior_weight,
    terminal_score,
    score_temperature,
    pad_size,
):
    """Collect adaptive learner states labeled by soft search-score targets."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        prior_info = jax.vmap(game.get_info)(states)
        is_active = ~prior_info.is_done

        obs_p0 = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: game.get_observation(state, 1))(states)
        learner_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(learner_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        learner_obs_arr, active = jax.vmap(lambda obs, size: adaptive_obs_to_array(obs, size, pad_size))(
            learner_obs,
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
        )(learner_obs, effective_sizes)
        base_obs_arr = learner_obs_arr
        base_masks = masks
        base_active = active

        key, search_key, learner_key, opponent_key = jrandom.split(key, 4)
        search_keys = jrandom.split(search_key, num_envs)
        _, candidate_indices, _, search_scores = jax.vmap(
            lambda state, size, sample_key: adaptive_rollout_search_candidates(
                base_network,
                state,
                size,
                sample_key,
                learner_player,
                top_k,
                rollout_steps,
                rollouts_per_action,
                opponent_policy_mode,
                army_weight,
                land_weight,
                prior_weight,
                terminal_score,
                pad_size,
            )
        )(states, effective_sizes, search_keys)
        target_probs = search_score_target_probs(search_scores, score_temperature)
        active_weights = is_active.astype(jnp.float32)

        learner_keys = jrandom.split(learner_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        learner_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                student_network,
                obs,
                size,
                sample_key,
                policy_mode,
                pad_size,
            )
        )(learner_obs, effective_sizes, learner_keys)
        opponent_actions = jax.vmap(
            lambda obs, size, sample_key: adaptive_policy_action(
                opponent_network,
                obs,
                size,
                sample_key,
                opponent_policy_mode,
                pad_size,
            )
        )(opponent_obs, effective_sizes, opponent_keys)
        actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
        next_states, _ = jax.vmap(game.step)(states, actions)
        final_states = jax.tree.map(
            lambda old, new: jnp.where(is_active.reshape(num_envs, *([1] * (old.ndim - 1))), new, old),
            states,
            next_states,
        )
        return (final_states, key), (
            learner_obs_arr,
            masks,
            active,
            base_obs_arr,
            base_masks,
            base_active,
            candidate_indices,
            target_probs,
            active_weights,
            active_weights,
        )

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=num_steps)
    return states, batch, key
```

- [ ] **Step 4: Run the collection test and the loss test**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q \
  tests/test_adaptive_ppo.py::test_adaptive_soft_conservative_loss_is_finite_for_matching_networks \
  tests/test_adaptive_ppo.py::test_adaptive_rollout_search_candidates_respects_effective_size \
  tests/test_adaptive_ppo.py::test_collect_adaptive_soft_batch_returns_expected_shapes
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add examples/_experimental/ppo/adaptive_search_distill.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: collect adaptive search distillation batches"
```

## Task 4: CLI, Checkpoints, And Smoke Test

**Files:**
- Modify: `examples/_experimental/ppo/adaptive_search_distill.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write the failing CLI smoke and pruning test**

Append this test after `test_train_adaptive_cli_smoke`:

```python
def test_adaptive_search_distill_cli_smoke_saves_and_prunes_checkpoints(tmp_path):
    import os
    import subprocess
    import sys

    import equinox as eqx

    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    base_model_path = tmp_path / "adaptive-base.eqx"
    model_path = tmp_path / "adaptive-search-distill.eqx"
    checkpoint_dir = tmp_path / "search-ckpts"
    eqx.tree_serialise_leaves(
        base_model_path,
        AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8)),
    )

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/adaptive_search_distill.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--grid-size-weights",
        "4:1,6:2",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--base-model-path",
        str(base_model_path),
        "--model-path",
        str(model_path),
        "--target-mode",
        "soft",
        "--learner-player",
        "1",
        "--num-steps",
        "1",
        "--num-iterations",
        "3",
        "--num-epochs",
        "1",
        "--minibatch-size",
        "2",
        "--top-k",
        "2",
        "--rollout-steps",
        "1",
        "--rollouts-per-action",
        "1",
        "--channels",
        "16,16,16,8",
        "--base-channels",
        "16,16,16,8",
        "--init-channels",
        "16,16,16,8",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "1",
        "--keep-checkpoints",
        "2",
        "--seed",
        "44000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
    assert sorted(path.name for path in checkpoint_dir.glob("*.eqx")) == [
        "adaptive-search-distill-iter-000002.eqx",
        "adaptive-search-distill-iter-000003.eqx",
    ]
```

- [ ] **Step 2: Run the CLI smoke test and verify missing CLI**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_adaptive_search_distill_cli_smoke_saves_and_prunes_checkpoints
```

Expected: FAIL because `adaptive_search_distill.py` has no `main()` CLI.

- [ ] **Step 3: Add parser validation and main training loop**

Append this code to the bottom of `examples/_experimental/ppo/adaptive_search_distill.py`:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Adaptively distill rollout-search improvements into one multisize checkpoint.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128)
    parser.add_argument("--grid-sizes", default="8,12,16")
    parser.add_argument("--grid-size-weights", default=None)
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--model-path", default="/tmp/generals-adaptive-search-distill.eqx")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--base-channels", default=None)
    parser.add_argument("--init-channels", default=None)
    parser.add_argument("--target-mode", choices=TARGET_MODE_NAMES, default="soft")
    parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--opponent-policy-mode", choices=POLICY_MODE_NAMES, default="sample")
    parser.add_argument("--learner-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--improve-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=25.0)
    parser.add_argument("--margin-scale", type=float, default=100.0)
    parser.add_argument("--max-improve-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--army-weight", type=float, default=12.0)
    parser.add_argument("--land-weight", type=float, default=8.0)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument("--terminal-score", type=float, default=1000.0)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
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
    if args.num_envs <= 0:
        parser.error("num_envs must be positive")
    if args.pool_size < args.num_envs:
        parser.error("--pool-size must be at least num_envs")
    if args.num_steps <= 0 or args.num_iterations <= 0 or args.num_epochs <= 0:
        parser.error("--num-steps, --num-iterations, and --num-epochs must be positive")
    if args.minibatch_size <= 0:
        parser.error("--minibatch-size must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.kl_weight < 0.0 or args.improve_weight < 0.0:
        parser.error("--kl-weight and --improve-weight must be non-negative")
    if args.temperature <= 0.0 or args.score_temperature <= 0.0:
        parser.error("--temperature and --score-temperature must be positive")
    if args.margin_scale <= 0.0 or args.max_improve_weight <= 0.0:
        parser.error("--margin-scale and --max-improve-weight must be positive")
    if args.top_k <= 0 or args.rollout_steps <= 0 or args.rollouts_per_action <= 0:
        parser.error("--top-k, --rollout-steps, and --rollouts-per-action must be positive")
    if args.terminal_score <= 0.0:
        parser.error("--terminal-score must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    if args.checkpoint_every < 0 or args.keep_checkpoints < 0:
        parser.error("--checkpoint-every and --keep-checkpoints cannot be negative")
    try:
        args.channels = parse_policy_channels(args.channels)
        args.base_channels = parse_policy_channels(args.base_channels or args.channels)
        args.init_channels = parse_policy_channels(args.init_channels) if args.init_channels is not None else None
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    init_model_path = args.init_model_path or args.base_model_path
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    opponent_policy_mode = POLICY_MODE_NAME_TO_ID[args.opponent_policy_mode]

    print("Adaptive conservative rollout-search distillation")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Grid sizes:    {','.join(str(size) for size in args.grid_sizes)} padded to {args.pad_to}")
    if args.grid_size_weights is not None:
        weights_label = ",".join(
            f"{size}:{weight:g}" for size, weight in zip(args.grid_sizes, args.grid_size_weights, strict=True)
        )
        print(f"Size weights:  {weights_label}")
    print(f"Student:       {init_model_path} channels={args.channels}")
    print(f"Base/Search:   {args.base_model_path} channels={args.base_channels}")
    print(f"Learner:       player {args.learner_player}")
    print(f"Rollout:       {args.num_iterations} x {args.num_steps} steps, envs={args.num_envs}")
    print(
        "Search:        "
        f"top_k={args.top_k}, rollout_steps={args.rollout_steps}, rollouts/action={args.rollouts_per_action}"
    )
    print(
        "Objective:     "
        f"kl={args.kl_weight:g}, improve={args.improve_weight:g}, "
        f"mode={args.target_mode}, score_temp={args.score_temperature:g}"
    )
    if args.checkpoint_dir is not None and args.checkpoint_every > 0:
        print(f"Checkpoints:   every {args.checkpoint_every} iterations in {args.checkpoint_dir}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, student_key, base_key, pool_key = jrandom.split(key, 4)
    student_network = load_or_create_adaptive_network(
        student_key,
        pad_size=args.pad_to,
        init_model_path=init_model_path,
        channels=args.channels,
        init_channels=args.init_channels,
    )
    base_network = load_or_create_adaptive_network(
        base_key,
        pad_size=args.pad_to,
        init_model_path=args.base_model_path,
        channels=args.base_channels,
    )
    opponent_network = base_network
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(student_network, eqx.is_inexact_array))

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

    checkpoint_paths = []
    model_stem = Path(args.model_path).stem
    for iteration in range(1, args.num_iterations + 1):
        t0 = time.time()
        key, rollout_key, update_key, pool_key = jrandom.split(key, 4)
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
        states, effective_sizes = make_adaptive_initial_states(pool, args.num_envs)

        if args.target_mode == "hard":
            _, batch, rollout_key = collect_adaptive_conservative_batch(
                student_network,
                base_network,
                opponent_network,
                states,
                effective_sizes,
                rollout_key,
                args.num_steps,
                policy_mode,
                opponent_policy_mode,
                args.learner_player,
                args.top_k,
                args.rollout_steps,
                args.rollouts_per_action,
                args.army_weight,
                args.land_weight,
                args.prior_weight,
                args.terminal_score,
                args.min_margin,
                args.margin_scale,
                args.max_improve_weight,
                args.pad_to,
            )
            flat_batch = flatten_adaptive_conservative_batch(*batch)
            student_network, opt_state, loss, metrics, update_key = train_adaptive_conservative_epoch(
                student_network,
                base_network,
                opt_state,
                flat_batch,
                optimizer,
                update_key,
                args.num_epochs,
                args.minibatch_size,
                args.kl_weight,
                args.improve_weight,
                args.temperature,
            )
        else:
            _, batch, rollout_key = collect_adaptive_soft_batch(
                student_network,
                base_network,
                opponent_network,
                states,
                effective_sizes,
                rollout_key,
                args.num_steps,
                policy_mode,
                opponent_policy_mode,
                args.learner_player,
                args.top_k,
                args.rollout_steps,
                args.rollouts_per_action,
                args.army_weight,
                args.land_weight,
                args.prior_weight,
                args.terminal_score,
                args.score_temperature,
                args.pad_to,
            )
            flat_batch = flatten_adaptive_soft_batch(*batch)
            student_network, opt_state, loss, metrics, update_key = train_adaptive_soft_epoch(
                student_network,
                base_network,
                opt_state,
                flat_batch,
                optimizer,
                update_key,
                args.num_epochs,
                args.minibatch_size,
                args.kl_weight,
                args.improve_weight,
                args.temperature,
            )
        jax.block_until_ready(student_network)

        if args.checkpoint_dir is not None and args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, model_stem, iteration)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(checkpoint_path, student_network)
            checkpoint_paths.append(checkpoint_path)
            prune_old_checkpoints(checkpoint_paths, args.keep_checkpoints)

        if iteration % 5 == 0 or iteration == 1 or iteration == args.num_iterations:
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.5f} | "
                f"KL: {float(metrics['kl_loss']):.5f} | "
                f"Improve: {float(metrics['improve_loss']):.4f} | "
                f"Selected: {int(metrics['selected_samples']):5d}/{samples} "
                f"({float(metrics['selected_fraction']) * 100:4.1f}%) | "
                f"Margin: {float(metrics['mean_selected_margin']):6.1f} | "
                f"SPS: {samples / elapsed:7.0f} | Time: {elapsed:.2f}s"
            )

    eqx.tree_serialise_leaves(args.model_path, student_network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the CLI smoke test**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_adaptive_search_distill_cli_smoke_saves_and_prunes_checkpoints
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add examples/_experimental/ppo/adaptive_search_distill.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive search distillation cli"
```

## Task 5: Verification, Ledger, And Push

**Files:**
- Modify: `statusquo.md`

- [ ] **Step 1: Run focused adaptive tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py
```

Expected: PASS.

- [ ] **Step 2: Run syntax compilation**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev python -m compileall examples/_experimental/ppo tests
```

Expected: command exits 0.

- [ ] **Step 3: Run repository checks**

Run:

```bash
git diff --check
JAX_PLATFORMS=cpu uv run --extra dev pytest -q
```

Expected: `git diff --check` exits 0 and full pytest passes.

- [ ] **Step 4: Append status ledger**

Append this exact entry to `statusquo.md`, using the current local timestamp:

```markdown
## [YYYY-MM-DD HH:MM] Adaptive Search Distillation Implementation
- **Changes:** Added `examples/_experimental/ppo/adaptive_search_distill.py` with adaptive rollout-search candidates, hard/soft conservative losses, adaptive batch collection, CLI training loop, and checkpoint retention. Added focused tests in `tests/test_adaptive_ppo.py`.
- **Status:** Completed
- **Next Steps:** Run a small CUDA narrow distillation from `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`, evaluate retained checkpoints at 128 or 256 games per size-seat row, and only promote checkpoints above the current 70.31% minimum row to 512-game evaluation.
- **Context:** This implements the stronger-teacher step; it does not prove the 90% target. The best verified checkpoint remains `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` until a new evaluation beats it.
```

- [ ] **Step 5: Commit verification ledger**

```bash
git status --short
git add .
git -c commit.gpgsign=false commit -m "chore: record adaptive search distillation implementation"
```

- [ ] **Step 6: Push the branch**

```bash
git push
```

Expected: branch `adaptive-multisize-expander` pushes to origin.

## Task 6: First Experiment Gate

**Files:**
- Modify after results: `docs/expander-training-strategy.md`
- Modify after results: `statusquo.md`

- [ ] **Step 1: Run the first narrow CUDA distillation**

Run:

```bash
JAX_PLATFORMS=cuda uv run --extra dev python examples/_experimental/ppo/adaptive_search_distill.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --pool-size 4096 \
  --base-model-path /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx \
  --model-path /tmp/generals-adaptive-search-distill-p1-v1.eqx \
  --target-mode soft \
  --learner-player 1 \
  --num-steps 8 \
  --num-iterations 40 \
  --num-epochs 1 \
  --minibatch-size 2048 \
  --lr 0.000001 \
  --kl-weight 1.0 \
  --improve-weight 0.05 \
  --top-k 4 \
  --rollout-steps 8 \
  --rollouts-per-action 2 \
  --channels 32,32,32,16 \
  --base-channels 32,32,32,16 \
  --init-channels 32,32,32,16 \
  --checkpoint-dir /tmp/generals-adaptive-search-distill-p1-v1-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 4 \
  --seed 45000
```

Expected: final model plus four retained checkpoints are written.

- [ ] **Step 2: Evaluate retained checkpoints at triage size**

Run one evaluation per retained checkpoint:

```bash
JAX_PLATFORMS=cuda uv run --extra dev python examples/_experimental/ppo/evaluate_adaptive_policy.py \
  /tmp/generals-adaptive-search-distill-p1-v1.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 256 \
  --max-steps 750 \
  --opponent expander \
  --json-output /tmp/generals-adaptive-search-distill-p1-v1-256.json \
  --seed 45100
```

Expected: JSON contains six rows. Promote only if `min(row["win_rate"] for row in rows) > 0.7031`.

- [ ] **Step 3: Document the result**

Append one result row to `docs/expander-training-strategy.md` with:

```markdown
| Adaptive search distill p1 v1 | `/tmp/generals-adaptive-search-distill-p1-v1.eqx` | 256/row | X.XX% | Result summary and whether it beat 70.31% |
```

Append a `statusquo.md` entry that includes the checkpoint path, evaluation JSON path, all six size-seat win rates, and whether the result is promoted to 512 games per row.

- [ ] **Step 4: Commit and push experiment notes**

```bash
git add docs/expander-training-strategy.md statusquo.md
git -c commit.gpgsign=false commit -m "docs: record adaptive search distillation probe"
git push
```

## Self-Review

- Spec coverage: The plan covers the adaptive script, adaptive candidate scorer, soft and hard losses, adaptive batch collection, checkpoint pruning, CLI smoke test, verification, and the first experiment gate.
- Scope: The first implementation uses scalar `--learner-player 0|1` and frozen adaptive base opponent, matching the design. Alternating seats and Expander-in-the-loop collection are deliberately outside this first slice.
- Type consistency: Adaptive loss functions include `active` and `base_active` everywhere, matching `AdaptivePolicyValueNetwork.logits_value(obs, mask, active)`.
- Red-flag scan: After saving the plan, run `PATTERN='TB''D|TO''DO|FIX''ME|\\?\\?|place''holder|implement late''r|fill in detail''s|add appropriat''e|Write tests for the abov''e|Similar to Tas''k'; rg -n "$PATTERN" docs/superpowers/plans/2026-06-16-adaptive-search-distillation.md` and fix every match before committing.
