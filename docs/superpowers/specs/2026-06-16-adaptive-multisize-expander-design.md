# Adaptive Multisize Expander Policy Design

## Goal

Train and select one adaptive PPO checkpoint that exceeds 90% total win rate against the randomized Expander opponent on all required map sizes:

- 8x8 generated maps.
- 12x12 generated maps.
- 16x16 generated maps.

The same checkpoint must be evaluated on all three sizes. Separate per-size checkpoints do not satisfy this design goal.

Acceptance uses sampled policy execution, matching the existing 8x8 v5 success criterion. A candidate is accepted only when independent evaluations show more than 90% total win rate for both player seats on each required size. Draws are not wins.

## Current Constraints

The existing ordinary `PolicyValueNetwork` is tied to one grid size:

- The policy head is convolutional, but the value head flattens `4 * grid_size * grid_size` into a fixed `Linear` layer.
- The public `PPOPolicyAgent` rejects observations whose shape does not match the checkpoint grid size.
- Existing behavior-clone and PPO target encodings use `9 * grid_size * grid_size` logits, including one pass logit per board cell.

The environment already has a useful base for this work:

- `GeneralsEnv` can generate variable-size maps padded to a fixed `pad_to` size.
- `generate_grid` can generate 8x8, 12x12, and 16x16 generated maps with compatible terrain settings.
- The raw PPO trainer already has strong fixed-size 8x8 training/evaluation patterns and Expander benchmarks.

## Recommended Approach

Use one fixed 16x16 tensor canvas for training and inference, while preserving the effective board size through an explicit active-cell mask.

For 8x8 and 12x12 games, the generated map is padded to 16x16. Padding cells are treated as impassable for game mechanics, but the neural input must identify them separately from real mountains. The model always receives arrays with shape based on `pad_to=16`, so JAX compilation and batching remain simple.

This approach is intentionally narrower than arbitrary HxW shape polymorphism. The target sizes are known, and a fixed canvas gives the fastest route to reliable training and repeatable evaluation.

## Adaptive Network

Add a new network type rather than replacing the existing fixed-size network. Existing checkpoints and GUI workflows should continue to use `PolicyValueNetwork`.

Proposed class:

```text
AdaptivePolicyValueNetwork
```

Core properties:

- `pad_size`: static maximum canvas size, initially 16.
- `input_channels`: default adaptive input channel count.
- Four convolution blocks similar to the existing PPO network.
- A movement policy head producing 8 movement planes over the 16x16 canvas.
- A scalar pass head producing exactly one pass logit.
- A masked value head using active-cell global pooling instead of flattening the whole board.

Policy logits are ordered as:

```text
0 .. 8 * pad_size * pad_size - 1: movement logits
8 * pad_size * pad_size: pass logit
```

Movement plane encoding:

- Planes 0-3: full-army moves in directions up, down, left, right.
- Planes 4-7: half-army moves in directions up, down, left, right.

The pass action has one global logit. This avoids the current size-dependent pass prior caused by `H * W` duplicate pass logits.

The value head pools over active cells only:

```text
pooled = concat(masked_mean(features), masked_max(features))
value = MLP(pooled)
```

This keeps value parameters independent of board area and prevents padded cells from influencing value estimates.

## Adaptive Input Encoding

Create an adaptive tensor encoder that wraps the existing observation channels and adds size-aware features.

Base channels:

- Existing 9 observation channels from `obs_to_array`.

Additional channels:

- `active_cells`: true for real board cells, false for padding.
- `padding_cells`: true for padded cells.
- normalized row coordinate over the effective board.
- normalized col coordinate over the effective board.
- effective board size divided by `pad_size`.
- effective board area divided by `pad_size * pad_size`.

Default adaptive observation channel count: 15.

Army features should use `log1p` normalization in the adaptive encoder. Boolean channels remain 0/1. Scalar broadcast channels must be float32. Coordinate channels are normalized over the effective board and set to 0.0 on padding cells.

The effective size must be carried explicitly by the adaptive reset pool and rollout data. It cannot be recovered reliably from `GameState` alone because padded cells and real mountains are both represented as impassable mountain cells for game mechanics.

## Action Encoding

Add adaptive action helpers independent of the existing fixed-size helpers:

```text
adaptive_action_to_index(action, pad_size)
adaptive_index_to_action(index, pad_size)
adaptive_action_to_target_probs(action, pad_size)
```

Canonical behavior:

- Pass action always maps to index `8 * pad_size * pad_size`.
- Movement actions map to `plane * pad_size * pad_size + row * pad_size + col`.
- `row` and `col` are canvas coordinates.
- Padding-source moves and off-board destination moves are masked out before sampling or scoring.

These helpers should not change existing `action_to_index` or `index_to_action` behavior used by fixed-size checkpoints.

## Valid Masks

The adaptive policy mask has shape `(pad_size, pad_size, 4)` for movement legality plus one implicit valid pass action.

Movement is valid only when:

- The source cell is active.
- The source cell is owned by the learner.
- The source army count is greater than 1.
- The destination is inside the active effective board.
- The destination is not a mountain.
- The destination is not padding.

Padding cells should remain mountains in the game state for environment safety, but the mask must still use `active_cells` to avoid treating real mountains and padding as the same semantic feature.

## Training Entry Points

Keep the current fixed-size scripts working. Add adaptive variants instead of overloading all existing fixed-size behavior at once.

Recommended files:

- `examples/_experimental/ppo/adaptive_network.py`
- `examples/_experimental/ppo/adaptive_common.py`
- `examples/_experimental/ppo/train_adaptive.py`
- `examples/_experimental/ppo/behavior_clone_adaptive.py`
- `examples/_experimental/ppo/evaluate_adaptive_policy.py`

The adaptive trainer should accept:

```text
--grid-sizes 8,12,16
--pad-to 16
--map-generator generated
--mountain-density-min 0.12
--mountain-density-max 0.22
--num-cities-min 4
--num-cities-max 8
--min-generals-distance auto
```

For `--min-generals-distance auto`, use per-size defaults:

- 8x8: 5
- 12x12: 7
- 16x16: 9

These preserve a comparable opening separation while keeping maps playable.

## Reset Pool Sampling

The adaptive reset pool must be size-balanced. A rollout batch should not be dominated by the cheapest or most common size.

Initial rule:

- Generate one third of the pool for each required size.
- Shuffle pool entries after concatenation.
- Track the effective size for each pool entry.

If `pool_size` is not divisible by the number of sizes, allocate the remainder to larger sizes first, because larger maps have higher variance and are the harder target.

## Training Curriculum

Stage 1: adaptive smoke training.

- Run small CPU-compatible tests and tiny PPO/BC smoke runs on 4x4/6x6 padded to 8 for speed.
- Confirm the same checkpoint can act on multiple effective sizes in one batch.

Stage 2: Expander behavior cloning warm start.

- Train from Expander soft targets on 8/12/16 mixed generated maps.
- Use the adaptive single-pass action target.
- Save the warm-start checkpoint outside the repository, for example `/tmp/generals-adaptive-bc-8-12-16.eqx`.

Stage 3: PPO against Expander.

- Start from adaptive BC.
- Use size-balanced rollouts.
- Use sample mode for learner action collection.
- Keep terminal reward scale available.
- Keep periodic checkpoint saving available.

Stage 4: per-size fine-tuning without losing coverage.

- Continue mixed-size training with larger maps oversampled only if 12x12 or 16x16 lags.
- Promotion still requires all three sizes and both seats above 90%.

## Evaluation

Create an adaptive evaluator that reports a size/seat matrix.

Required rows:

```text
8x8 player 0
8x8 player 1
12x12 player 0
12x12 player 1
16x16 player 0
16x16 player 1
```

Each row reports:

- wins
- losses
- draws
- total win rate
- decisive win rate
- draw rate
- mean final step

Acceptance default:

- 2048 games per row.
- Two independent seeds per row for final promotion.
- `--policy-mode sample`.
- `--opponent expander`.
- `--max-steps 500` for 8x8, `750` for 12x12, and `1000` for 16x16 unless training evidence shows a stricter shared horizon is viable.

The promotion score is the minimum total win rate over all required size/seat/seed rows.

## Testing

Unit tests must cover:

- Adaptive encoder shape and channel semantics for 8x8 and 12x12 padded boards.
- Padding cells differ from real mountains in the adaptive input.
- Adaptive movement mask blocks padding-source and padding-destination moves.
- Adaptive action encoding maps all pass actions to one index.
- `AdaptivePolicyValueNetwork` forward pass returns `8 * pad_size * pad_size + 1` logits for different effective sizes on the same pad size.
- Value output remains finite when only part of the canvas is active.
- A tiny adaptive BC target distribution has the expected shape and pass index.
- A tiny adaptive evaluation smoke test runs at least one game for each configured size.

Verification commands for implementation work:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_adaptive_ppo.py
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
```

Full verification before promotion:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_adaptive_policy.py /tmp/candidate.eqx \
  --grid-sizes 8,12,16 \
  --num-games 2048 \
  --seeds 8501,8503 \
  --opponent expander \
  --policy-mode sample \
  --require-win-rate 0.90
```

## Documentation

Update documentation when implementation lands:

- README: adaptive training and evaluation commands.
- `docs/zh-manual.md`: Chinese usage notes.
- `docs/expander-training-strategy.md`: adaptive curriculum, command history, and evaluation evidence.
- `statusquo.md`: append-only entries for design, implementation, and training runs.

## Non-Goals

- Arbitrary unpadded HxW inference is not required for this milestone.
- Rectangular maps are not required.
- Search-assisted policy execution is not part of the 90% checkpoint target.
- Existing 8x8 fixed-size checkpoints do not need to load into the adaptive architecture without a deliberate migration utility.

## Promotion Rule

The goal is complete only when current evidence proves that one adaptive checkpoint exceeds 90% total win rate against Expander for every required size and seat:

- 8x8 player 0 and player 1.
- 12x12 player 0 and player 1.
- 16x16 player 0 and player 1.

The evidence must include exact checkpoint path, command settings, seeds, game counts, and row-level win/loss/draw counts. A mixed-size aggregate above 90% is not sufficient.
