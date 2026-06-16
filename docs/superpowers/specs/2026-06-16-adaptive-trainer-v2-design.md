# Adaptive Trainer V2 Design

## Goal

Improve the adaptive PPO training loop enough to restart progress toward one checkpoint that beats Expander by more than 90% total win rate on every required size-seat row:

- 8x8, player 0 and player 1.
- 12x12, player 0 and player 1.
- 16x16, player 0 and player 1.

The current best checkpoint is `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`. Its independent 512-game-per-row evaluation at 750 steps reached a minimum total win rate of 70.31%. Repeating the existing single-seat, low-learning-rate continuation recipe has plateaued below that result.

This design does not change the adaptive network architecture. It changes only the trainer controls and reward signal needed for the next round of experiments.

## Evidence From Current Runs

The strongest checkpoint is no longer failing because the model cannot run on all sizes. Infrastructure is working: one `AdaptivePolicyValueNetwork` can train, save, load, and evaluate across 8x8, 12x12, and 16x16 boards padded to 16.

The failure mode is training signal quality:

- 16x16 rows still have high draw counts at the evaluation horizon.
- Single-seat continuations improve one seat or one size, then give back strength elsewhere.
- Raising terminal reward alone did not improve the best minimum win rate.
- Equal reset-pool allocation gives the larger, slower maps no extra sampling despite their higher variance and draw risk.

The next iteration should therefore make the trainer explicitly optimize against those bottlenecks.

## Considered Approaches

### Recommended: Trainer V2 Controls

Add three narrow controls to `train_adaptive.py` and the shared adaptive pool helpers:

- Weighted reset-pool allocation by effective board size.
- A non-terminal truncation penalty for learner timeouts/draws.
- Alternating learner seats inside one training run.

This keeps the current architecture, optimizer, evaluator, checkpoint format, and scripts intact. It is the lowest-risk change that directly targets the observed plateau.

### Alternative: True Dual-Seat Batch Training

Collect player-0 and player-1 learner samples in the same rollout batch, then train PPO on their union.

This could reduce forgetting more strongly than alternating iterations, but it is more invasive. The current adaptive rollout path assumes a scalar `learner_player` when selecting observations, stacking actions, computing rewards, and reporting wins. Splitting every environment batch by seat would touch more JAX control flow and shape logic, so it should wait until the narrower alternating-seat version is tested.

### Alternative: New Network Or Curriculum Stack

Change the architecture, add recurrence, train a league opponent, or add search/distillation.

Those may eventually be needed for a 90% target, but they should not be the next step. The current plateau has a simpler explanation: timeout/draw reward and cross-seat forgetting are not represented well enough in the trainer.

## Proposed Interface

Extend `examples/_experimental/ppo/train_adaptive.py` with:

```text
--grid-size-weights 8:1,12:1,16:2
--truncation-reward-scale 0.5
--learner-player 0|1|alternate
```

`--grid-size-weights` controls how many reset-pool states are generated for each effective size. The parser should require positive weights and require the same size keys as `--grid-sizes`. If the flag is omitted, the existing equal-size allocation remains the default.

`--truncation-reward-scale` adds a negative reward to learner samples only when a rollout transition hits the configured truncation horizon without a decisive terminal result. It must not modify decisive terminal rewards. The default is `0.0`, preserving current behavior.

`--learner-player alternate` alternates the scalar learner seat by training iteration. This intentionally reuses the existing rollout path instead of introducing split-seat batching. Fixed `0` and `1` modes remain compatible with existing commands.

## Pool Weighting

Add a weighted count helper beside the existing balanced `_pool_counts` logic in `adaptive_common.py`.

Rules:

- With no weights, keep the current equal split and larger-size remainder behavior.
- With weights, compute proportional integer counts for the configured `pool_size`.
- If `pool_size >= len(grid_sizes)`, every size receives at least one state.
- Allocate remaining slots by largest fractional remainder.
- Break exact ties toward larger board sizes, because larger maps are currently harder.

Example:

```text
pool_size=8
grid_sizes=(8,12,16)
weights=(1,1,2)
counts=(2,2,4)
```

The helper should stay deterministic so tests can assert exact counts.

## Truncation Reward

Add a small helper in `train_adaptive.py`:

```text
apply_truncation_reward(rewards, truncated, scale)
```

Expected behavior:

- `scale == 0.0` returns rewards unchanged.
- `truncated == True` subtracts `scale`.
- Decisive terminal rows are not passed as truncated, so terminal reward and truncation reward do not double-apply.

In `rollout_step`, compute `terminated`, `truncated`, and `dones` before final reward shaping, then apply terminal reward and truncation reward in that order.

The first experiment should use a modest scale such as `0.25` or `0.5`. This should push the policy to finish games instead of preserving a draw-heavy position, without overwhelming the shaped territory/capture rewards.

## Alternating Seat Training

Parse `--learner-player` as a string, then resolve it per training iteration:

- `"0"` means always player 0.
- `"1"` means always player 1.
- `"alternate"` means odd iterations use player 0 and even iterations use player 1.

Warmup can use player 0. The main loop should pass the resolved integer seat into `collect_rollout`, win logging, and checkpoint behavior.

This approach keeps JIT compilation simple: each rollout still sees a scalar static learner seat. It may compile separate traces for the two seats, which is acceptable for long GPU runs and avoids a more complex mixed-seat batch implementation.

## First Experiment After Implementation

Start from the current best checkpoint:

```text
/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx
```

Run a mixed-size continuation with larger-map oversampling and alternating seats:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1.5,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 8192 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --opponent expander \
  --learner-player alternate \
  --terminal-reward-scale 1.0 \
  --truncation-reward-scale 0.5 \
  --init-model-path /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-gpu-v2-checkpoints \
  --checkpoint-every 50 \
  --keep-checkpoints 6 \
  --model-path /tmp/generals-adaptive-ppo-gpu-v2.eqx \
  --seed 62016
```

Evaluate the final checkpoint and every retained checkpoint with the existing adaptive evaluator. Use 256 games per row for triage, then 512 or more for candidates that improve the minimum row.

## Testing

Implementation must use TDD. Add failing tests before changing production code.

Required tests:

- `parse_grid_size_weights` accepts `8:1,12:1,16:2` and rejects missing, duplicate, unknown, or non-positive weights.
- Weighted pool generation produces deterministic effective-size counts.
- `apply_truncation_reward` subtracts only on non-terminal truncation rows and leaves scale-zero rewards unchanged.
- `train_adaptive.py` CLI smoke accepts `--grid-size-weights`, `--truncation-reward-scale`, and `--learner-player alternate`.
- Existing adaptive smoke tests still pass.

Verification commands:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py
JAX_PLATFORMS=cpu uv run --extra dev python -m compileall examples/_experimental/ppo tests
git diff --check
```

Full CPU pytest should be run before committing implementation changes unless the runtime cost becomes unreasonable.

## Promotion Criteria

Trainer V2 implementation is complete when the new controls are tested, documented, and committed. That does not mean the 90% training target is complete.

The full adaptive target remains open until a single checkpoint passes:

```text
8x8 p0, 8x8 p1, 12x12 p0, 12x12 p1, 16x16 p0, 16x16 p1 > 90% total win rate
```

using independent generated-map evaluation against Expander, with draws counted as non-wins.
