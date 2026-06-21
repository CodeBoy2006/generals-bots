# Adaptive Trainer V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adaptive PPO trainer controls for weighted size sampling, timeout/draw reward shaping, and alternating learner seats so the next CUDA run can move the 8/12/16 Expander win-rate target beyond the current plateau.

**Architecture:** Keep `AdaptivePolicyValueNetwork` and checkpoint format unchanged. Extend `adaptive_common.py` for deterministic size-weight parsing and weighted pool counts, and extend `train_adaptive.py` for truncation reward shaping plus per-iteration learner-seat resolution.

**Tech Stack:** Python 3.12, JAX, Equinox, Optax, pytest, `uv run --extra dev`.

---

## File Structure

- Modify `examples/_experimental/ppo/adaptive_common.py`: parse `--grid-size-weights`, compute deterministic weighted reset-pool counts, and pass optional weights into `make_adaptive_state_pool`.
- Modify `examples/_experimental/ppo/train_adaptive.py`: add CLI flags, apply truncation reward, resolve `--learner-player alternate`, and pass pool weights into pool creation.
- Modify `tests/test_adaptive_ppo.py`: add red-green tests for parser validation, weighted pool counts, truncation reward, and trainer CLI smoke.
- Modify `README.md` and `docs/zh-manual.md`: document the new trainer controls.
- Modify `docs/expander-training-strategy.md`: record the trainer-v2 recipe for the next CUDA continuation.
- Modify `statusquo.md`: append an implementation log entry after verification.

---

### Task 1: Weighted Size Parser And Pool Counts

**Files:**
- Modify: `tests/test_adaptive_ppo.py`
- Modify: `examples/_experimental/ppo/adaptive_common.py`

- [ ] **Step 1: Write failing tests**

Add these tests near the existing grid-size and pool tests:

```python
def test_parse_grid_size_weights_requires_matching_positive_sizes():
    import pytest

    from examples._experimental.ppo.adaptive_common import parse_grid_size_weights

    assert parse_grid_size_weights("8:1,12:1.5,16:2", (8, 12, 16)) == (1.0, 1.5, 2.0)
    assert parse_grid_size_weights(None, (8, 12, 16)) is None

    for value in ("8:1,12:1", "8:1,12:1,16:0", "8:1,12:1,20:2", "8:1,8:2,16:1"):
        with pytest.raises(ValueError):
            parse_grid_size_weights(value, (8, 12, 16))


def test_make_adaptive_state_pool_uses_grid_size_weights():
    from examples._experimental.ppo.adaptive_common import make_adaptive_state_pool

    pool = make_adaptive_state_pool(
        jrandom.PRNGKey(2),
        pool_size=8,
        grid_sizes=(4, 6, 8),
        pad_size=8,
        map_generator="simple",
        mountain_density_range=(0.0, 0.0),
        num_cities_range=(2, 2),
        max_generals_distance=None,
        castle_val_range=(10, 11),
        grid_size_weights=(1.0, 1.0, 2.0),
    )

    assert sorted(pool.effective_sizes.tolist()) == [4, 4, 6, 6, 8, 8, 8, 8]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q \
  tests/test_adaptive_ppo.py::test_parse_grid_size_weights_requires_matching_positive_sizes \
  tests/test_adaptive_ppo.py::test_make_adaptive_state_pool_uses_grid_size_weights
```

Expected: FAIL because `parse_grid_size_weights` and the `grid_size_weights` parameter do not exist yet.

- [ ] **Step 3: Implement parser and weighted counts**

In `adaptive_common.py`, add:

```python
def parse_grid_size_weights(value: str | None, grid_sizes: tuple[int, ...]) -> tuple[float, ...] | None:
    """Parse size:weight pairs aligned to configured adaptive grid sizes."""
    if value is None or not value.strip():
        return None
    parsed: dict[int, float] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--grid-size-weights entries must use size:weight")
        size_text, weight_text = part.split(":", 1)
        size = int(size_text.strip())
        weight = float(weight_text.strip())
        if size in parsed:
            raise ValueError("--grid-size-weights cannot repeat a grid size")
        if weight <= 0.0:
            raise ValueError("--grid-size-weights values must be positive")
        parsed[size] = weight
    expected = set(grid_sizes)
    actual = set(parsed)
    if actual != expected:
        raise ValueError("--grid-size-weights must specify exactly the same sizes as --grid-sizes")
    return tuple(parsed[size] for size in grid_sizes)
```

Then add `_weighted_pool_counts(pool_size, grid_sizes, weights)` and update `make_adaptive_state_pool(..., grid_size_weights=None)` to call it when weights are provided.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q \
  tests/test_adaptive_ppo.py::test_parse_grid_size_weights_requires_matching_positive_sizes \
  tests/test_adaptive_ppo.py::test_make_adaptive_state_pool_uses_grid_size_weights
```

Expected: PASS.

---

### Task 2: Truncation Reward Shaping

**Files:**
- Modify: `tests/test_adaptive_ppo.py`
- Modify: `examples/_experimental/ppo/train_adaptive.py`

- [ ] **Step 1: Write failing test**

Add:

```python
def test_apply_truncation_reward_penalizes_only_truncated_rows():
    from examples._experimental.ppo.train_adaptive import apply_truncation_reward

    rewards = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float32)
    truncated = jnp.array([True, False, True])

    shaped = apply_truncation_reward(rewards, truncated, 0.5)

    assert jnp.allclose(shaped, jnp.array([0.5, 0.5, -0.75], dtype=jnp.float32))
    assert jnp.allclose(apply_truncation_reward(rewards, truncated, 0.0), rewards)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q \
  tests/test_adaptive_ppo.py::test_apply_truncation_reward_penalizes_only_truncated_rows
```

Expected: FAIL because `apply_truncation_reward` does not exist.

- [ ] **Step 3: Implement reward helper and rollout wiring**

Add the helper in `train_adaptive.py` near `rollout_step`:

```python
def apply_truncation_reward(rewards, truncated, scale):
    """Penalize non-terminal timeout rows without changing decisive games."""
    return rewards - jnp.where(truncated, scale, 0.0)
```

Add `truncation_reward_scale` to `rollout_step` and `collect_rollout` signatures. In `rollout_step`, compute:

```python
terminated = infos.is_done
truncated = (new_states.time >= truncation) & ~terminated
dones = terminated | truncated
rewards = apply_terminal_reward(rewards, infos, learner_player, terminal_reward_scale)
rewards = apply_truncation_reward(rewards, truncated, truncation_reward_scale)
```

Pass `args.truncation_reward_scale` from warmup and rollout calls.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q \
  tests/test_adaptive_ppo.py::test_apply_truncation_reward_penalizes_only_truncated_rows
```

Expected: PASS.

---

### Task 3: Trainer CLI And Alternating Learner Seat

**Files:**
- Modify: `tests/test_adaptive_ppo.py`
- Modify: `examples/_experimental/ppo/train_adaptive.py`

- [ ] **Step 1: Write failing CLI smoke changes**

In `test_train_adaptive_cli_smoke`, add:

```python
        "--grid-size-weights",
        "4:1,6:2",
        "--truncation-reward-scale",
        "0.25",
        "--learner-player",
        "alternate",
```

- [ ] **Step 2: Run CLI smoke to verify failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_train_adaptive_cli_smoke
```

Expected: FAIL because the CLI does not accept these new argument forms.

- [ ] **Step 3: Implement CLI parsing and seat resolution**

In `train_adaptive.py`:

```python
from adaptive_common import parse_grid_size_weights


def resolve_learner_player(value: str, iteration: int) -> int:
    """Resolve fixed or alternating learner seat for one training iteration."""
    if value == "alternate":
        return (iteration - 1) % 2
    return int(value)
```

Change `--learner-player` to `choices=("0", "1", "alternate")`, add `--grid-size-weights`, add `--truncation-reward-scale`, validate it is non-negative, and parse weights after `args.grid_sizes`.

In the main loop:

```python
iteration_learner_player = resolve_learner_player(args.learner_player, iteration)
```

Use `iteration_learner_player` in `collect_rollout` and win logging. Keep warmup on player 0 via `resolve_learner_player(args.learner_player, 1)`.

- [ ] **Step 4: Run CLI smoke to verify pass**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py::test_train_adaptive_cli_smoke
```

Expected: PASS.

---

### Task 4: Documentation, Verification, And Commit

**Files:**
- Modify: `README.md`
- Modify: `docs/zh-manual.md`
- Modify: `docs/expander-training-strategy.md`
- Modify: `statusquo.md`

- [ ] **Step 1: Update docs**

Add the new flags to the adaptive PPO documentation:

```text
--grid-size-weights 8:1.5,12:1,16:2
--truncation-reward-scale 0.5
--learner-player alternate
```

Explain that the first flag oversamples harder sizes, the second penalizes non-terminal truncation, and the third alternates learner seats by training iteration.

- [ ] **Step 2: Run focused verification**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py
JAX_PLATFORMS=cpu uv run --extra dev python -m compileall examples/_experimental/ppo tests
git diff --check
```

Expected: all commands pass.

- [ ] **Step 3: Run full verification**

Run:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Append status log**

Append a `statusquo.md` entry:

```markdown
## [2026-06-16 HH:MM] Adaptive Trainer V2 Controls
- **Changes:** Added weighted adaptive reset-pool sampling, truncation reward shaping, alternating learner seats, tests, and docs for the next Expander PPO continuation.
- **Status:** Completed
- **Next Steps:** Run the CUDA trainer-v2 continuation from `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`, then evaluate retained checkpoints across all 8/12/16 size-seat rows.
- **Context:** This implements training controls only. The 90% Expander target remains unproven until an independently evaluated checkpoint clears every required row.
```

- [ ] **Step 5: Commit and push**

Run:

```bash
git status
git add .
git -c commit.gpgsign=false commit -m "feat: add adaptive trainer v2 controls"
git push
```

Expected: commit and push succeed, and `git status --short --branch` shows a clean branch aligned with origin.

---

## Plan Self-Review

- Spec coverage: weighted pool sampling is Task 1; truncation reward is Task 2; alternating learner seats and CLI are Task 3; docs, status, commit, and verification are Task 4.
- Placeholder scan: the plan contains no deferred implementation markers.
- Type consistency: parser returns `tuple[float, ...] | None`; pool accepts `grid_size_weights`; trainer keeps scalar integer seats inside rollout and uses the string only at CLI/main-loop boundaries.
