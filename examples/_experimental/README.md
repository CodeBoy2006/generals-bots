# Experimental Examples

This folder contains experimental code and advanced examples that are not yet part of the main API.

## Contents

### `benchmark_performance.py`
Performance benchmark for measuring throughput with vectorized environments.

```bash
python benchmark_performance.py [num_envs] [num_steps] [iterations]

# Examples
python benchmark_performance.py 256 100 5      # 256 envs, 100 steps, 5 iterations
python benchmark_performance.py 1024 500 3     # 1024 envs, 500 steps, 3 iterations
python benchmark_performance.py 4096 100 2     # 4096 envs, 100 steps, 2 iterations
```

Outputs throughput statistics (steps/s) and helps identify performance bottlenecks.

### `ppo/`
Experimental PPO (Proximal Policy Optimization) training implementation. Work in progress.

The raw-game trainer is the primary path for quick experiments:

```bash
uv run python examples/_experimental/ppo/train.py 64 --num-steps 64 --num-iterations 10 --model-path runs/generals-ppo-4x4.eqx
```

Use `--grid-size` for larger square maps. The default `--map-generator simple`
keeps empty maps with two random generals, which is useful for fast regression
checks:

```bash
uv run python examples/_experimental/ppo/train.py 64 --grid-size 8 --num-steps 64 --num-iterations 10 --model-path runs/generals-ppo-8x8-simple.eqx
```

Use `--map-generator generated` to train on maps with mountains and cities:

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --num-steps 64 \
  --num-iterations 10 \
  --pool-size 512 \
  --model-path runs/generals-ppo-8x8-generated.eqx
```

The saved model must be visualized with the same `--grid-size` and compatible
map settings:

```bash
uv run python examples/_experimental/visualize_policy.py runs/generals-ppo-8x8-generated.eqx 10 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5
```

For faster warm-starts, behavior-clone the policy from the randomized Expander
teacher and then evaluate the checkpoint on independent generated maps:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --model-path runs/generals-bc-8x8-soft.eqx

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py runs/generals-bc-8x8-soft.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --max-steps 500 \
  --opponent random \
  --policy-mode sample
```

The PPO stack also exposes a built-in heuristic pool that can be used as a
teacher, opponent, or standalone evaluation target:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --teacher balanced \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --model-path runs/generals-bc-balanced.eqx

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --map-generator generated \
  --opponent mixed \
  --num-steps 64 \
  --num-iterations 10 \
  --pool-size 512 \
  --model-path runs/generals-ppo-mixed.eqx

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_heuristics.py \
  --agent general-hunter \
  --opponent expander \
  --num-games 512 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500

For larger maps, the same heuristic pool is meant to be evaluated and trained
with `--grid-size 12` or `--grid-size 16` plus longer horizon settings such as
`--max-steps 500` or higher.
```

### `visualize_policy.py`
Visualization tools for trained policies. Work in progress.
