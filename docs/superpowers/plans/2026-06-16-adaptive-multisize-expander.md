# Adaptive Multisize Expander Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the adaptive PPO infrastructure needed to train one checkpoint for 8x8, 12x12, and 16x16 generated maps against Expander.

**Architecture:** Keep the existing fixed-size PPO code intact and add adaptive modules beside it. Adaptive training uses a fixed `pad_to` canvas, explicit effective-size metadata, size-aware input channels, a single global pass logit, and an active-cell pooled value head.

**Tech Stack:** Python 3.12, JAX, Equinox, Optax, pytest, existing `generals.core.game` and experimental PPO scripts.

---

## File Structure

- Create `examples/_experimental/ppo/adaptive_common.py`: adaptive grid-size parsing, reset-pool generation, active-cell masks, observation encoding, valid-move masks, action-index helpers, target distributions, and policy action dispatch.
- Create `examples/_experimental/ppo/adaptive_network.py`: `AdaptivePolicyValueNetwork`, checkpoint loading helpers, and network constants.
- Create `examples/_experimental/ppo/behavior_clone_adaptive.py`: Expander/heuristic behavior-cloning warm start for mixed effective sizes.
- Create `examples/_experimental/ppo/train_adaptive.py`: raw-game PPO trainer for the adaptive checkpoint.
- Create `examples/_experimental/ppo/evaluate_adaptive_policy.py`: size/seat matrix evaluator with JSON output and threshold failure.
- Create `tests/test_adaptive_ppo.py`: unit, smoke, CLI, and checkpoint tests for the adaptive path.
- Modify `README.md`, `docs/zh-manual.md`, `docs/expander-training-strategy.md`, and `statusquo.md` after code is verified.

## Task 1: Adaptive Common Primitives

**Files:**
- Create: `examples/_experimental/ppo/adaptive_common.py`
- Create: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write failing tests for adaptive parsing, active masks, input channels, masks, and action encoding**

Add these tests to `tests/test_adaptive_ppo.py`:

```python
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game


def make_padded_state(size=4, pad_to=6):
    grid = jnp.full((pad_to, pad_to), -2, dtype=jnp.int32)
    grid = grid.at[:size, :size].set(0)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[size - 1, size - 1].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(6))
    return state


def test_parse_grid_sizes_and_auto_distances():
    from examples._experimental.ppo.adaptive_common import parse_grid_sizes, min_distance_for_size

    assert parse_grid_sizes("8,12,16") == (8, 12, 16)
    assert min_distance_for_size(8) == 5
    assert min_distance_for_size(12) == 7
    assert min_distance_for_size(16) == 9


def test_adaptive_obs_to_array_marks_padding_separately_from_real_mountain():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS, adaptive_obs_to_array

    state = make_padded_state(size=4, pad_to=6)
    state = state._replace(mountains=state.mountains.at[1, 1].set(True), passable=state.passable.at[1, 1].set(False))
    obs = game.get_observation(state, 0)

    arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)

    assert arr.shape == (ADAPTIVE_INPUT_CHANNELS, 6, 6)
    assert active.shape == (6, 6)
    assert bool(active[3, 3])
    assert not bool(active[4, 4])
    assert arr[3, 1, 1] == 1.0
    assert arr[3, 4, 4] == 1.0
    assert arr[9, 1, 1] == 1.0
    assert arr[9, 4, 4] == 0.0
    assert arr[10, 4, 4] == 1.0


def test_compute_adaptive_valid_move_mask_blocks_padding_destinations():
    from examples._experimental.ppo.adaptive_common import compute_adaptive_valid_move_mask

    state = make_padded_state(size=4, pad_to=6)
    state = state._replace(armies=state.armies.at[3, 3].set(5), ownership=state.ownership.at[0, 3, 3].set(True))
    mask = compute_adaptive_valid_move_mask(
        state.armies,
        state.ownership[0],
        state.mountains,
        effective_size=4,
        pad_size=6,
    )

    assert mask.shape == (6, 6, 4)
    assert not bool(mask[3, 3, 1])
    assert not bool(mask[3, 3, 3])
    assert not bool(mask[4, 4, 0])


def test_adaptive_action_encoding_uses_single_pass_index():
    from examples._experimental.ppo.adaptive_common import (
        adaptive_action_to_index,
        adaptive_action_to_target_probs,
        adaptive_index_to_action,
    )

    pad_size = 6
    pass_index = 8 * pad_size * pad_size
    pass_a = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    pass_b = jnp.array([1, 5, 5, 3, 1], dtype=jnp.int32)
    move = jnp.array([0, 2, 1, 3, 1], dtype=jnp.int32)

    assert int(adaptive_action_to_index(pass_a, pad_size)) == pass_index
    assert int(adaptive_action_to_index(pass_b, pad_size)) == pass_index
    assert int(adaptive_action_to_index(move, pad_size)) == (7 * pad_size * pad_size + 2 * pad_size + 1)
    assert adaptive_index_to_action(jnp.asarray(pass_index), pad_size).tolist() == [1, 0, 0, 0, 0]
    assert adaptive_action_to_target_probs(pass_b, pad_size).shape == (8 * pad_size * pad_size + 1,)
```

- [ ] **Step 2: Run tests and verify they fail because adaptive module is missing**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'examples._experimental.ppo.adaptive_common'`.

- [ ] **Step 3: Create adaptive common helpers**

Create `examples/_experimental/ppo/adaptive_common.py` with these definitions:

```python
"""Shared helpers for adaptive multisize PPO training."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents._heuristic_logic import HEURISTIC_NAMES, heuristic_action
from generals.agents.ppo_policy_agent import obs_to_array
from generals.core import game
from generals.core.action import DIRECTIONS
from generals.core.grid import generate_grid

ADAPTIVE_INPUT_CHANNELS = 15
ADAPTIVE_MOVE_PLANES = 8


class AdaptiveStatePool(NamedTuple):
    states: game.GameState
    effective_sizes: jnp.ndarray


def parse_grid_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("at least one grid size is required")
    if any(size < 4 for size in sizes):
        raise ValueError("grid sizes must be at least 4")
    if len(set(sizes)) != len(sizes):
        raise ValueError("grid sizes must be unique")
    return sizes


def min_distance_for_size(size: int) -> int:
    return {8: 5, 12: 7, 16: 9}.get(size, max(3, size // 2))


def active_cells_for_size(effective_size: int, pad_size: int) -> jnp.ndarray:
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    return (rows < effective_size) & (cols < effective_size)


def adaptive_obs_to_array(obs, effective_size: int, pad_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    base = obs_to_array(obs)
    active = active_cells_for_size(effective_size, pad_size)
    padding = ~active
    rows = jnp.arange(pad_size, dtype=jnp.float32)[:, None]
    cols = jnp.arange(pad_size, dtype=jnp.float32)[None, :]
    denom = jnp.maximum(jnp.asarray(effective_size - 1, dtype=jnp.float32), 1.0)
    row_coord = jnp.where(active, rows / denom, 0.0)
    col_coord = jnp.where(active, cols / denom, 0.0)
    size_channel = jnp.ones((pad_size, pad_size), dtype=jnp.float32) * (effective_size / pad_size)
    area_channel = jnp.ones((pad_size, pad_size), dtype=jnp.float32) * (
        (effective_size * effective_size) / (pad_size * pad_size)
    )
    normalized_base = base.at[0].set(jnp.log1p(jnp.maximum(base[0], 0.0)))
    extra = jnp.stack(
        [
            active.astype(jnp.float32),
            padding.astype(jnp.float32),
            row_coord,
            col_coord,
            size_channel,
            area_channel,
        ],
        axis=0,
    )
    return jnp.concatenate([normalized_base, extra], axis=0), active


def compute_adaptive_valid_move_mask(
    armies: jnp.ndarray,
    owned_cells: jnp.ndarray,
    mountains: jnp.ndarray,
    effective_size: int,
    pad_size: int,
) -> jnp.ndarray:
    active = active_cells_for_size(effective_size, pad_size)
    can_move_from = active & owned_cells & (armies > 1)
    passable = active & ~mountains
    rows = jnp.arange(pad_size)[:, None]
    cols = jnp.arange(pad_size)[None, :]
    dest_i = rows[:, :, None] + DIRECTIONS[None, None, :, 0]
    dest_j = cols[:, :, None] + DIRECTIONS[None, None, :, 1]
    in_bounds = (dest_i >= 0) & (dest_i < effective_size) & (dest_j >= 0) & (dest_j < effective_size)
    safe_i = jnp.clip(dest_i, 0, pad_size - 1)
    safe_j = jnp.clip(dest_j, 0, pad_size - 1)
    dest_passable = passable[safe_i, safe_j]
    return can_move_from[:, :, None] & in_bounds & dest_passable


def adaptive_action_space_size(pad_size: int) -> int:
    return ADAPTIVE_MOVE_PLANES * pad_size * pad_size + 1


def adaptive_action_to_index(action: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    is_pass, row, col, direction, is_half = action
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    plane = direction + jnp.where(is_half > 0, 4, 0)
    move_index = plane * pad_size * pad_size + row * pad_size + col
    return jnp.where(is_pass > 0, pass_index, move_index).astype(jnp.int32)


def adaptive_index_to_action(index: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    pass_index = ADAPTIVE_MOVE_PLANES * pad_size * pad_size
    is_pass = index == pass_index
    safe_index = jnp.minimum(index, pass_index - 1)
    plane = safe_index // (pad_size * pad_size)
    position = safe_index % (pad_size * pad_size)
    row = position // pad_size
    col = position % pad_size
    direction = plane % 4
    is_half = plane >= 4
    return jnp.where(
        is_pass,
        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.array([0, row, col, direction, is_half], dtype=jnp.int32),
    )


def adaptive_action_to_target_probs(action: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    index = adaptive_action_to_index(action, pad_size)
    return jax.nn.one_hot(index, adaptive_action_space_size(pad_size), dtype=jnp.float32)
```

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
```

Expected: PASS for the four Task 1 tests.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add examples/_experimental/ppo/adaptive_common.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive PPO common helpers"
```

## Task 2: Adaptive Policy-Value Network

**Files:**
- Create: `examples/_experimental/ppo/adaptive_network.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write failing network tests**

Append to `tests/test_adaptive_ppo.py`:

```python
def test_adaptive_network_forward_uses_fixed_action_space_and_finite_value():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6)
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value = network.logits_value(obs_arr, mask, active)

    assert logits.shape == (8 * 6 * 6 + 1,)
    assert jnp.isfinite(value)
    assert jnp.isfinite(logits[-1])


def test_adaptive_network_samples_and_scores_action():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6)
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    action, value, logprob, entropy = network(obs_arr, mask, active, jrandom.PRNGKey(1), None)

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert jnp.isfinite(value)
    assert jnp.isfinite(logprob)
    assert jnp.isfinite(entropy)
```

- [ ] **Step 2: Run tests and verify they fail because adaptive network is missing**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'examples._experimental.ppo.adaptive_network'`.

- [ ] **Step 3: Create adaptive network**

Create `examples/_experimental/ppo/adaptive_network.py`:

```python
"""Adaptive multisize policy-value network for experimental PPO."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.ppo_policy_agent import DEFAULT_POLICY_CHANNELS, PolicyChannels, parse_policy_channels

from adaptive_common import (
    ADAPTIVE_INPUT_CHANNELS,
    adaptive_action_space_size,
    adaptive_action_to_index,
    adaptive_index_to_action,
)


class AdaptivePolicyValueNetwork(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    policy_conv: eqx.nn.Conv2d
    pass_linear: eqx.nn.Linear
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear
    pad_size: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        pad_size: int = 16,
        channels: PolicyChannels = DEFAULT_POLICY_CHANNELS,
        input_channels: int = ADAPTIVE_INPUT_CHANNELS,
    ):
        keys = jrandom.split(key, 8)
        self.pad_size = pad_size
        self.conv1 = eqx.nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, key=keys[2])
        self.conv4 = eqx.nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, key=keys[3])
        self.policy_conv = eqx.nn.Conv2d(channels[3], 8, kernel_size=1, key=keys[4])
        self.pass_linear = eqx.nn.Linear(channels[3] * 2, 1, key=keys[5])
        self.value_linear1 = eqx.nn.Linear(channels[3] * 2, 64, key=keys[6])
        self.value_linear2 = eqx.nn.Linear(64, 1, key=keys[7])

    def _features(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        return jax.nn.relu(self.conv4(x))

    def _masked_pool(self, x: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        active_f = active.astype(jnp.float32)[None, :, :]
        denom = jnp.maximum(jnp.sum(active_f), 1.0)
        mean = jnp.sum(x * active_f, axis=(1, 2)) / denom
        masked = jnp.where(active_f > 0, x, -jnp.inf)
        max_pool = jnp.max(masked, axis=(1, 2))
        max_pool = jnp.where(jnp.isfinite(max_pool), max_pool, 0.0)
        return jnp.concatenate([mean, max_pool], axis=0)

    def logits_value(self, obs: jnp.ndarray, mask: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self._features(obs)
        pooled = self._masked_pool(x, active)
        move_logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        move_mask = jnp.concatenate([mask_t, mask_t], axis=0)
        move_logits = move_logits + (1 - move_mask.astype(jnp.float32)) * -1e9
        pass_logit = self.pass_linear(pooled)
        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        value = self.value_linear2(value_hidden)[0]
        logits = jnp.concatenate([move_logits.reshape(-1), pass_logit], axis=0)
        return logits, value

    def __call__(
        self,
        obs: jnp.ndarray,
        mask: jnp.ndarray,
        active: jnp.ndarray,
        key: jnp.ndarray | None,
        action: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, value = self.logits_value(obs, mask, active)
        if action is None:
            if key is None:
                raise ValueError("key is required when sampling an action")
            index = jrandom.categorical(key, logits)
            action = adaptive_index_to_action(index, self.pad_size)
        else:
            index = adaptive_action_to_index(action, self.pad_size)
        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[index]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)
        return action, value, logprob, entropy


def load_or_create_adaptive_network(
    key: jnp.ndarray,
    pad_size: int,
    init_model_path: str | Path | None = None,
    channels: str | PolicyChannels | list[int] | None = None,
    input_channels: int = ADAPTIVE_INPUT_CHANNELS,
) -> AdaptivePolicyValueNetwork:
    parsed_channels = parse_policy_channels(channels)
    network = AdaptivePolicyValueNetwork(key, pad_size=pad_size, channels=parsed_channels, input_channels=input_channels)
    if init_model_path is None:
        return network
    path = Path(init_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    return eqx.tree_deserialise_leaves(path, network)
```

- [ ] **Step 4: Run network tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
```

Expected: PASS for Task 1 and Task 2 tests.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add examples/_experimental/ppo/adaptive_network.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive PPO network"
```

## Task 3: Adaptive Reset Pool and Teacher Targets

**Files:**
- Modify: `examples/_experimental/ppo/adaptive_common.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write failing tests for size-balanced pools and Expander targets**

Append:

```python
def test_make_adaptive_state_pool_balances_sizes():
    from examples._experimental.ppo.adaptive_common import make_adaptive_state_pool

    pool = make_adaptive_state_pool(
        jrandom.PRNGKey(0),
        pool_size=5,
        grid_sizes=(4, 6),
        pad_size=6,
        map_generator="simple",
        mountain_density_range=(0.0, 0.0),
        num_cities_range=(2, 2),
        max_generals_distance=None,
        castle_val_range=(10, 11),
    )

    assert pool.states.armies.shape == (5, 6, 6)
    assert sorted(pool.effective_sizes.tolist()) == [4, 4, 6, 6, 6]


def test_adaptive_expander_target_probs_has_single_pass_slot():
    from examples._experimental.ppo.adaptive_common import adaptive_expander_target_probs

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    target = adaptive_expander_target_probs(obs, effective_size=4, pad_size=6)

    assert target.shape == (8 * 6 * 6 + 1,)
    assert jnp.isclose(jnp.sum(target), 1.0)
```

- [ ] **Step 2: Run tests and verify expected missing-function failures**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
```

Expected: FAIL with import errors for `make_adaptive_state_pool` and `adaptive_expander_target_probs`.

- [ ] **Step 3: Add pool and target helpers**

Append to `examples/_experimental/ppo/adaptive_common.py`:

```python
def make_simple_general_grid(key, grid_size: int, pad_size: int) -> jnp.ndarray:
    grid = jnp.full((pad_size, pad_size), -2, dtype=jnp.int32)
    grid = grid.at[:grid_size, :grid_size].set(0)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    return grid.at[pos_a].set(1).at[pos_b].set(2)


def _pool_counts(pool_size: int, grid_sizes: tuple[int, ...]) -> tuple[int, ...]:
    base = pool_size // len(grid_sizes)
    remainder = pool_size % len(grid_sizes)
    counts = [base] * len(grid_sizes)
    for index in range(remainder):
        counts[len(counts) - 1 - index] += 1
    return tuple(counts)


def make_adaptive_state_pool(
    key,
    pool_size: int,
    grid_sizes: tuple[int, ...],
    pad_size: int,
    map_generator: str,
    mountain_density_range: tuple[float, float],
    num_cities_range: tuple[int, int],
    max_generals_distance: int | None,
    castle_val_range: tuple[int, int],
) -> AdaptiveStatePool:
    keys = jrandom.split(key, pool_size + 1)
    shuffle_key = keys[0]
    offset = 1
    pools = []
    sizes = []
    for grid_size, count in zip(grid_sizes, _pool_counts(pool_size, grid_sizes), strict=True):
        combo_keys = keys[offset : offset + count]
        offset += count
        if map_generator == "simple":
            grids = jax.vmap(lambda k, s=grid_size: make_simple_general_grid(k, s, pad_size))(combo_keys)
        else:
            grids = jax.vmap(
                lambda k, s=grid_size: generate_grid(
                    k,
                    grid_dims=(s, s),
                    pad_to=pad_size,
                    mountain_density_range=mountain_density_range,
                    num_cities_range=num_cities_range,
                    min_generals_distance=min_distance_for_size(s),
                    max_generals_distance=max_generals_distance,
                    castle_val_range=castle_val_range,
                )
            )(combo_keys)
        pools.append(jax.vmap(game.create_initial_state)(grids))
        sizes.append(jnp.full((count,), grid_size, dtype=jnp.int32))
    states = jax.tree.map(lambda *xs: jnp.concatenate(xs), *pools)
    effective_sizes = jnp.concatenate(sizes)
    permutation = jrandom.permutation(shuffle_key, pool_size)
    return AdaptiveStatePool(
        states=jax.tree.map(lambda x: x[permutation], states),
        effective_sizes=effective_sizes[permutation],
    )


def make_adaptive_initial_states(pool: AdaptiveStatePool, num_envs: int) -> tuple[game.GameState, jnp.ndarray]:
    states = jax.tree.map(lambda x: x[:num_envs], pool.states)
    sizes = pool.effective_sizes[:num_envs]
    pool_size = pool.states.armies.shape[0]
    pool_idx = (jnp.arange(num_envs, dtype=jnp.int32) + num_envs) % pool_size
    return states._replace(pool_idx=pool_idx), sizes


def adaptive_expander_target_probs(obs, effective_size: int, pad_size: int) -> jnp.ndarray:
    valid_mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, effective_size, pad_size)
    target = jnp.zeros(adaptive_action_space_size(pad_size), dtype=jnp.float32)
    armies = obs.armies
    directions = DIRECTIONS
    rows = jnp.arange(pad_size)[:, None, None]
    cols = jnp.arange(pad_size)[None, :, None]
    dest_i = jnp.clip(rows + directions[None, None, :, 0], 0, pad_size - 1)
    dest_j = jnp.clip(cols + directions[None, None, :, 1], 0, pad_size - 1)
    source_armies = armies[:, :, None]
    dest_armies = armies[dest_i, dest_j]
    is_opponent = obs.opponent_cells[dest_i, dest_j]
    is_neutral = obs.neutral_cells[dest_i, dest_j]
    is_owned = obs.owned_cells[dest_i, dest_j]
    can_capture = source_armies > dest_armies + 1
    is_expansion = ~is_owned & (is_opponent | is_neutral)
    scores = source_armies.astype(jnp.float32)
    scores = jnp.where(is_expansion & can_capture, scores * jnp.where(is_opponent, 20.0, 10.0), scores)
    scores = jnp.where(valid_mask & can_capture, scores, 0.0)
    score_sum = jnp.sum(scores)
    num_valid = jnp.sum(valid_mask)
    move_probs = jnp.where(score_sum > 0, scores / score_sum, valid_mask.astype(jnp.float32) / jnp.maximum(num_valid, 1))
    move_probs = jnp.where(num_valid > 0, move_probs, jnp.zeros_like(move_probs))
    full_planes = jnp.transpose(move_probs, (2, 0, 1)).reshape(4 * pad_size * pad_size)
    target = target.at[: 4 * pad_size * pad_size].set(full_planes)
    target = target.at[-1].set(jnp.where(num_valid == 0, 1.0, 0.0))
    return target
```

- [ ] **Step 4: Run tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
```

Expected: PASS through Task 3 tests.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add examples/_experimental/ppo/adaptive_common.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive PPO state pools"
```

## Task 4: Adaptive Behavior Cloning Script

**Files:**
- Create: `examples/_experimental/ppo/behavior_clone_adaptive.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write failing CLI smoke test**

Append:

```python
def test_behavior_clone_adaptive_cli_smoke(tmp_path):
    import os
    import subprocess
    import sys

    model_path = tmp_path / "adaptive-bc.eqx"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/behavior_clone_adaptive.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--num-steps",
        "1",
        "--num-iterations",
        "1",
        "--model-path",
        str(model_path),
        "--seed",
        "41000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
```

- [ ] **Step 2: Run test and verify missing script failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py::test_behavior_clone_adaptive_cli_smoke
```

Expected: FAIL because `examples/_experimental/ppo/behavior_clone_adaptive.py` does not exist.

- [ ] **Step 3: Create adaptive behavior cloning script**

Create `examples/_experimental/ppo/behavior_clone_adaptive.py` using this structure:

```python
"""Behavior cloning warm-start for adaptive multisize PPO."""

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
    adaptive_action_to_index,
    adaptive_action_to_target_probs,
    adaptive_expander_target_probs,
    adaptive_obs_to_array,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import TEACHER_NAME_TO_ID, TEACHER_NAMES, heuristic_action
from train import random_action
from generals.core import game


@eqx.filter_jit
def collect_teacher_batch(states, effective_sizes, pool, key, steps, truncation, teacher_id, pad_size):
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, effective_sizes, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        key, teacher_key, random_key = jrandom.split(key, 3)
        teacher_keys = jrandom.split(teacher_key, num_envs)
        random_keys = jrandom.split(random_key, num_envs)

        def soft_target(obs, size, k):
            target = adaptive_expander_target_probs(obs, size, pad_size)
            index = jrandom.categorical(k, jnp.log(target + 1e-8))
            return target, index

        targets, teacher_indices = jax.vmap(soft_target)(obs_p0, effective_sizes, teacher_keys)
        teacher_actions = jax.vmap(lambda idx: __import__("adaptive_common").adaptive_common.adaptive_index_to_action(idx, pad_size))(teacher_indices)
        hard_actions = jax.vmap(lambda k, o: heuristic_action(teacher_id - 1, k, o))(teacher_keys, obs_p0)
        hard_indices = jax.vmap(lambda a: adaptive_action_to_index(a, pad_size))(hard_actions)
        hard_targets = jax.vmap(lambda a: adaptive_action_to_target_probs(a, pad_size))(hard_actions)
        actions_p0 = jax.lax.cond(teacher_id == 0, lambda _: teacher_actions, lambda _: hard_actions, None)
        targets = jax.lax.cond(teacher_id == 0, lambda _: targets, lambda _: hard_targets, None)
        teacher_indices = jax.lax.cond(teacher_id == 0, lambda _: teacher_indices, lambda _: hard_indices, None)
        actions_p1 = jax.vmap(random_action)(random_keys, obs_p1)
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
        obs_arr, active = jax.vmap(lambda o, s: adaptive_obs_to_array(o, s, pad_size))(obs_p0, effective_sizes)
        masks = jax.vmap(lambda o, s: compute_adaptive_valid_move_mask(o.armies, o.owned_cells, o.mountains, s, pad_size))(
            obs_p0, effective_sizes
        )
        return (final_states, final_sizes, key), (obs_arr, masks, active, targets, teacher_indices, dones, infos.winner)

    (states, effective_sizes, key), batch = jax.lax.scan(body, (states, effective_sizes, key), None, length=steps)
    return states, effective_sizes, batch, key


@eqx.filter_jit
def train_bc_step(network, opt_state, obs, masks, active, targets, teacher_indices, optimizer):
    batch_size = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(batch_size, *obs.shape[2:])
    masks_flat = masks.reshape(batch_size, *masks.shape[2:])
    active_flat = active.reshape(batch_size, *active.shape[2:])
    targets_flat = targets.reshape(batch_size, targets.shape[-1])
    teacher_indices_flat = teacher_indices.reshape(batch_size)

    def loss_fn(net):
        logits = jax.vmap(lambda o, m, a: net.logits_value(o, m, a)[0])(obs_flat, masks_flat, active_flat)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.sum(targets_flat * log_probs, axis=-1)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == teacher_indices_flat)
        return jnp.mean(losses), accuracy

    (loss, accuracy), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss, accuracy
```

Add `parse_args()` with these arguments: positional `num_envs`, `--grid-sizes`, `--pad-to`, `--map-generator`, `--teacher`, `--num-steps`, `--num-iterations`, `--lr`, `--pool-size`, `--truncation`, generated-map terrain settings, `--init-model-path`, `--model-path`, and `--seed`. Add validation that `pad_to >= max(grid_sizes)`, `pool_size >= num_envs`, city ranges are valid, mountain density is ordered, and teacher is one of `TEACHER_NAMES`.

Add `main()` with this execution flow: parse args, create PRNG keys, parse grid sizes, call `load_or_create_adaptive_network`, initialize Optax Adam, call `make_adaptive_state_pool`, call `make_adaptive_initial_states`, loop over `num_iterations`, call `collect_teacher_batch`, call `train_bc_step`, print iteration loss/accuracy/episode stats, serialize the final checkpoint with `eqx.tree_serialise_leaves(args.model_path, network)`, and print the saved path.

- [ ] **Step 4: Replace dynamic import in the script with a direct import**

Add `adaptive_index_to_action` to the import list from `adaptive_common`, and replace:

```python
teacher_actions = jax.vmap(lambda idx: __import__("adaptive_common").adaptive_common.adaptive_index_to_action(idx, pad_size))(teacher_indices)
```

with:

```python
teacher_actions = jax.vmap(lambda idx: adaptive_index_to_action(idx, pad_size))(teacher_indices)
```

- [ ] **Step 5: Run CLI smoke test**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py::test_behavior_clone_adaptive_cli_smoke
```

Expected: PASS and a checkpoint file exists under pytest temp storage.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add examples/_experimental/ppo/behavior_clone_adaptive.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive behavior cloning"
```

## Task 5: Adaptive PPO Trainer

**Files:**
- Create: `examples/_experimental/ppo/train_adaptive.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write failing trainer smoke test**

Append:

```python
def test_train_adaptive_cli_smoke(tmp_path):
    import os
    import subprocess
    import sys

    model_path = tmp_path / "adaptive-ppo.eqx"
    checkpoint_dir = tmp_path / "ckpts"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/train_adaptive.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--num-steps",
        "1",
        "--num-iterations",
        "1",
        "--num-epochs",
        "1",
        "--minibatch-size",
        "2",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "1",
        "--model-path",
        str(model_path),
        "--seed",
        "42000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
    assert (checkpoint_dir / "adaptive-ppo-iter-000001.eqx").exists()
```

- [ ] **Step 2: Run test and verify missing script failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py::test_train_adaptive_cli_smoke
```

Expected: FAIL because `train_adaptive.py` does not exist.

- [ ] **Step 3: Create adaptive PPO trainer**

Create `examples/_experimental/ppo/train_adaptive.py` by reusing fixed-size PPO structure with adaptive batch items:

```python
"""Raw-game PPO trainer for adaptive multisize policy checkpoints."""

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
    adaptive_obs_to_array,
    compute_adaptive_valid_move_mask,
    make_adaptive_initial_states,
    make_adaptive_state_pool,
    parse_grid_sizes,
)
from adaptive_network import load_or_create_adaptive_network
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, opponent_action
from train import (
    apply_terminal_reward,
    checkpoint_path_for_iteration,
    compute_gae,
    prune_old_checkpoints,
    random_action,
    stack_learner_actions,
)
from generals.core import game
from generals.core.rewards import composite_reward_fn


@eqx.filter_jit
def rollout_step(states, effective_sizes, pool, network, key, truncation, opponent_id, learner_player, terminal_reward_scale, pad_size):
    num_envs = states.armies.shape[0]
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    learner_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p0_prior, lambda _: obs_p1_prior, None)
    opponent_obs_prior = jax.lax.cond(learner_player == 0, lambda _: obs_p1_prior, lambda _: obs_p0_prior, None)
    obs_arr, active = jax.vmap(lambda o, s: adaptive_obs_to_array(o, s, pad_size))(learner_obs_prior, effective_sizes)
    masks = jax.vmap(lambda o, s: compute_adaptive_valid_move_mask(o.armies, o.owned_cells, o.mountains, s, pad_size))(
        learner_obs_prior,
        effective_sizes,
    )
    key, learner_key = jrandom.split(key)
    learner_keys = jrandom.split(learner_key, num_envs)
    learner_actions, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, 0, None))(
        obs_arr,
        masks,
        active,
        learner_keys,
        None,
    )
    key, opponent_key = jrandom.split(key)
    opponent_keys = jrandom.split(opponent_key, num_envs)
    opponent_actions = jax.vmap(lambda k, o: opponent_action(opponent_id, k, o, random_action))(opponent_keys, opponent_obs_prior)
    actions = stack_learner_actions(learner_actions, opponent_actions, learner_player)
    new_states, infos = jax.vmap(game.step)(states, actions)
    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    obs_p1_new = jax.vmap(lambda s: game.get_observation(s, 1))(new_states)
    learner_obs_new = jax.lax.cond(learner_player == 0, lambda _: obs_p0_new, lambda _: obs_p1_new, None)
    rewards = jax.vmap(composite_reward_fn)(learner_obs_prior, learner_actions, learner_obs_new)
    rewards = apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale)
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
    return final_states, final_sizes, (obs_arr, masks, active, learner_actions, logprobs, values, rewards, dones, infos), key


@jax.jit
def ppo_loss(network, obs, mask, active, action, old_logprob, advantage, ret, clip=0.2):
    _, value, logprob, entropy = network(obs, mask, active, None, action)
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy
    return policy_loss + value_loss + entropy_loss


@eqx.filter_jit
def train_minibatch_step(network, opt_state, minibatch, optimizer):
    obs, masks, active, actions, old_logprobs, advantages, returns = minibatch

    def loss_fn(net):
        losses = jax.vmap(lambda o, m, ac, a, olp, adv, r: ppo_loss(net, o, m, ac, a, olp, adv, r))(
            obs,
            masks,
            active,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(network, updates), opt_state, loss
```

Add `flatten_training_batch(batch)` that flattens `(steps, envs, ...)` for `obs`, `masks`, `active`, `actions`, `old_logprobs`, `advantages`, and `returns`. Add `train_epoch(...)` that shuffles the flattened batch, slices minibatches, and calls `train_minibatch_step`.

Add `parse_args()` with positional `num_envs`, `--grid-sizes`, `--pad-to`, `--num-steps`, `--num-iterations`, `--num-epochs`, `--minibatch-size`, `--lr`, `--pool-size`, `--truncation`, `--opponent`, `--learner-player`, `--terminal-reward-scale`, generated-map terrain settings, `--channels`, `--init-model-path`, `--model-path`, `--checkpoint-dir`, `--checkpoint-every`, `--keep-checkpoints`, and `--seed`. Validate positive counts, valid terrain ranges, `pad_to >= max(grid_sizes)`, non-negative checkpoint settings, and non-negative terminal reward scale.

Add `main()` with this execution flow: parse args, create/load adaptive network, initialize optimizer, build adaptive pool and initial states, run three warmup `rollout_step` calls, collect rollouts with `jax.lax.scan` or a Python loop over `num_steps`, compute GAE, normalize advantages, train for `num_epochs`, write periodic checkpoints through `checkpoint_path_for_iteration`, prune with `prune_old_checkpoints`, print rollout win/draw statistics, save final `model_path`, and print the saved path.

- [ ] **Step 4: Run trainer smoke test**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py::test_train_adaptive_cli_smoke
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add examples/_experimental/ppo/train_adaptive.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive PPO trainer"
```

## Task 6: Adaptive Evaluator

**Files:**
- Create: `examples/_experimental/ppo/evaluate_adaptive_policy.py`
- Modify: `tests/test_adaptive_ppo.py`

- [ ] **Step 1: Write failing evaluator smoke test**

Append:

```python
def test_evaluate_adaptive_policy_cli_writes_size_rows(tmp_path):
    import json
    import os
    import subprocess
    import sys

    import equinox as eqx
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    model_path = tmp_path / "adaptive.eqx"
    output_path = tmp_path / "adaptive-eval.json"
    eqx.tree_serialise_leaves(model_path, AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6))
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/evaluate_adaptive_policy.py",
        str(model_path),
        "--grid-sizes",
        "4,6",
        "--pad-to",
        "6",
        "--num-games",
        "2",
        "--max-steps",
        "4",
        "--map-generator",
        "simple",
        "--json-output",
        str(output_path),
        "--seed",
        "43000",
    ]

    completed = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert "adaptive policy evaluation" in completed.stdout.lower()
    assert len(data["rows"]) == 4
    assert {row["grid_size"] for row in data["rows"]} == {4, 6}
    assert {row["policy_player"] for row in data["rows"]} == {0, 1}
```

- [ ] **Step 2: Run test and verify missing script failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py::test_evaluate_adaptive_policy_cli_writes_size_rows
```

Expected: FAIL because `evaluate_adaptive_policy.py` does not exist.

- [ ] **Step 3: Create adaptive evaluator**

Create `examples/_experimental/ppo/evaluate_adaptive_policy.py` with:

```python
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

from adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask, make_adaptive_state_pool, parse_grid_sizes
from adaptive_network import AdaptivePolicyValueNetwork
from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, opponent_action
from train import random_action, stack_learner_actions
from generals.core import game


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
def evaluate_batch(network, states, effective_size, key, max_steps, opponent, policy_player, pad_size):
    num_envs = states.armies.shape[0]
    effective_sizes = jnp.full((num_envs,), effective_size, dtype=jnp.int32)

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)
        policy_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p0, lambda _: obs_p1, None)
        opponent_obs = jax.lax.cond(policy_player == 0, lambda _: obs_p1, lambda _: obs_p0, None)
        obs_arr, active = jax.vmap(lambda o, s: adaptive_obs_to_array(o, s, pad_size))(policy_obs, effective_sizes)
        masks = jax.vmap(lambda o, s: compute_adaptive_valid_move_mask(o.armies, o.owned_cells, o.mountains, s, pad_size))(
            policy_obs,
            effective_sizes,
        )
        key, policy_key, opponent_key = jrandom.split(key, 3)
        policy_keys = jrandom.split(policy_key, num_envs)
        policy_actions, _, _, _ = jax.vmap(network, in_axes=(0, 0, 0, 0, None))(obs_arr, masks, active, policy_keys, None)
        opponent_keys = jrandom.split(opponent_key, num_envs)
        opponent_actions = jax.vmap(lambda k, o: opponent_action(opponent, k, o, random_action))(opponent_keys, opponent_obs)
        actions = stack_learner_actions(policy_actions, opponent_actions, policy_player)
        new_states, infos = jax.vmap(game.step)(states, actions)
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        return (final_states, key), infos

    (states, key), _ = jax.lax.scan(body, (states, key), None, length=max_steps)
    return jax.vmap(game.get_info)(states)
```

Add `summarize_row(info, grid_size, policy_player, num_games)` that returns `AdaptiveEvalRow` from the policy player's perspective. Add `parse_args()` with positional `model_path`, `--grid-sizes`, `--pad-to`, `--num-games`, `--max-steps`, `--opponent`, `--map-generator`, generated-map terrain settings, `--channels`, `--json-output`, `--require-win-rate`, and `--seed`. Validate positive counts, valid `pad_to`, and valid terrain ranges.

Add `main()` with this execution flow: parse args, create/load `AdaptivePolicyValueNetwork`, loop over each grid size and policy player, generate a single-size adaptive pool with `make_adaptive_state_pool`, evaluate with `evaluate_batch`, summarize with `summarize_row`, print row metrics, write JSON when `--json-output` is present, compute the minimum row win rate, and raise `SystemExit(1)` when `--require-win-rate` is set and any row is below threshold.

- [ ] **Step 4: Run evaluator smoke test**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py::test_evaluate_adaptive_policy_cli_writes_size_rows
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add examples/_experimental/ppo/evaluate_adaptive_policy.py tests/test_adaptive_ppo.py
git -c commit.gpgsign=false commit -m "feat: add adaptive policy evaluator"
```

## Task 7: Documentation, Status Log, and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/zh-manual.md`
- Modify: `docs/expander-training-strategy.md`
- Modify: `statusquo.md`

- [ ] **Step 1: Run targeted and full verification before docs**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
JAX_PLATFORMS=cpu uv run pytest -q
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
```

Expected: adaptive tests pass, full pytest passes, compileall succeeds, and no whitespace errors.

- [ ] **Step 2: Update README adaptive commands**

Add this section near the PPO experiment commands in `README.md`:

```markdown
### Adaptive 8/12/16 PPO checkpoint

The adaptive PPO path trains one checkpoint on a fixed `pad-to` canvas while mixing effective board sizes:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 12288 \
  --num-steps 32 \
  --num-iterations 1000 \
  --model-path /tmp/generals-adaptive-bc-8-12-16.eqx
```

Evaluate one adaptive checkpoint across every required size and seat:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_adaptive_policy.py /tmp/candidate.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 2048 \
  --opponent expander \
  --require-win-rate 0.90
```
```

- [ ] **Step 3: Update Chinese manual and training strategy**

In `docs/zh-manual.md`, add a short Chinese note that the adaptive path uses one checkpoint, `--grid-sizes 8,12,16`, and `--pad-to 16`.

In `docs/expander-training-strategy.md`, add an "Adaptive 8/12/16 checkpoint" section with:

```markdown
## Adaptive 8/12/16 checkpoint target

The adaptive target is stricter than the fixed 8x8 v5 result: one checkpoint must exceed 90% total win rate against Expander on 8x8, 12x12, and 16x16, in both seats, with sampled policy execution. Mixed aggregate win rate is not sufficient.
```

- [ ] **Step 4: Append status log**

Append:

```markdown
## [2026-06-16 HH:MM] Adaptive Multisize PPO Infrastructure
- **Changes:** Added adaptive PPO common helpers, size-invariant policy-value network, behavior cloning, PPO training, evaluation smoke paths, tests, and documentation for one checkpoint across 8x8/12x12/16x16.
- **Status:** Completed
- **Next Steps:** Run adaptive BC warm start on GPU, then PPO-vs-Expander mixed-size training and per-size evaluation gates.
- **Context:** This does not yet prove the 90% target; it creates the training and evaluation infrastructure needed to produce and verify candidate checkpoints.
```

Use the real current timestamp from `date '+%Y-%m-%d %H:%M'`.

- [ ] **Step 5: Run final verification**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
JAX_PLATFORMS=cpu uv run pytest -q
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
git status --short
```

Expected: tests and compileall pass; status shows only intended tracked file changes plus existing untracked `.superpowers/`.

- [ ] **Step 6: Commit and push infrastructure**

Run:

```bash
git add README.md docs/zh-manual.md docs/expander-training-strategy.md statusquo.md
git -c commit.gpgsign=false commit -m "docs: document adaptive multisize PPO workflow"
git push
```

## Task 8: First Training and Evaluation Runs

**Files:**
- Modify: `docs/expander-training-strategy.md`
- Modify: `statusquo.md`

- [ ] **Step 1: Run adaptive BC warm start on GPU**

Run:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --pool-size 12288 \
  --num-steps 32 \
  --num-iterations 1000 \
  --lr 0.0007 \
  --truncation 500 \
  --model-path /tmp/generals-adaptive-bc-8-12-16-v1.eqx \
  --seed 46000
```

Expected: model saved to `/tmp/generals-adaptive-bc-8-12-16-v1.eqx`.

- [ ] **Step 2: Run adaptive PPO probe**

Run:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --pool-size 12288 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.00005 \
  --truncation 750 \
  --opponent expander \
  --terminal-reward-scale 1.0 \
  --init-model-path /tmp/generals-adaptive-bc-8-12-16-v1.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-probe-checkpoints \
  --checkpoint-every 50 \
  --keep-checkpoints 6 \
  --model-path /tmp/generals-adaptive-ppo-8-12-16-probe.eqx \
  --seed 46100
```

Expected: periodic checkpoints plus final probe checkpoint.

- [ ] **Step 3: Evaluate the probe by size and seat**

Run:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_adaptive_policy.py /tmp/generals-adaptive-ppo-8-12-16-probe.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 512 \
  --opponent expander \
  --policy-mode sample \
  --json-output /tmp/generals-adaptive-ppo-8-12-16-probe-eval.json \
  --seed 46200
```

Expected: JSON file with six rows. Use the weakest row to choose the next fine-tune schedule.

- [ ] **Step 4: Record training evidence**

Append command settings, checkpoint paths, and per-row evaluation results to `docs/expander-training-strategy.md`. Append a status entry to `statusquo.md` stating whether the run is below gate, in progress, or complete.

- [ ] **Step 5: Commit training evidence**

Run:

```bash
git add docs/expander-training-strategy.md statusquo.md
git -c commit.gpgsign=false commit -m "docs: record adaptive multisize PPO probe"
git push
```

## Self-Review Notes

- Spec coverage: Tasks cover adaptive common helpers, network, pool, BC, PPO, evaluator, docs, and first training/evaluation evidence.
- Placeholder scan: This plan intentionally avoids placeholder markers and vague catch-all steps. Each task has concrete files, tests, commands, and expected outcomes.
- Type consistency: Adaptive helpers consistently use `effective_size`, `pad_size`, `AdaptiveStatePool`, `active`, and the action space `8 * pad_size * pad_size + 1`.
