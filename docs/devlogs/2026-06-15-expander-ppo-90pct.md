# Expander PPO 90 Percent Run

Date: 2026-06-15

## Summary

This run produced an 8x8 generated-map PPO checkpoint that exceeds 90% total win rate against the randomized Expander heuristic when executed with sampled policy actions.

Final checkpoint:

```text
/tmp/generals-ppo-8x8-expander-gpu-v5.eqx
```

The checkpoint is an experiment artifact and is intentionally kept outside the repository.

## Code Changes

### `examples/_experimental/ppo/train.py`

- Added `--init-model-path` so PPO can continue from behavior-cloning or prior PPO checkpoints.
- Added `--seed` for reproducible trainer initialization.
- Added `--num-epochs` and `--minibatch-size` for multi-epoch PPO updates over each rollout batch.
- Added reusable helpers for checkpoint loading, rollout-batch flattening, and minibatch updates.

### `examples/_experimental/ppo/behavior_clone.py`

- Added `--init-model-path` so behavior cloning can resume from an existing checkpoint.

### `examples/_experimental/ppo/evaluate_policy.py`

- Added `--policy-player` so the model can be evaluated as player 0 or player 1.
- Added policy-perspective result summarization so mirrored evaluations count wins and losses correctly.

### `README.md`

- Documented PPO continuation from checkpoints and mirrored policy evaluation.

## Training Path

The run started from a regenerated soft-Expander behavior-cloning checkpoint:

```text
/tmp/generals-bc-8x8-soft-cpu-v1.eqx
```

A short PPO probe against Expander improved the model from roughly 33% to the mid-40% range:

```text
/tmp/generals-ppo-8x8-expander-probe.eqx
```

GPU fine-tuning then used 8x8 generated maps with:

- mountain density: 0.12-0.22
- cities: 4-8
- minimum general distance: 5
- max steps: 500
- opponent: randomized Expander
- policy execution for acceptance: sample

The final successful stage was:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 700 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --opponent expander \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v4.eqx \
  --model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --seed 9104
```

## Final Validation

Independent 2048-game evaluations against Expander:

```text
seed 8501, policy_player=0, sample:
  wins/losses/draws = 1854/150/44
  win rate = 90.53%
  decisive win rate = 92.51%

seed 8501, policy_player=1, sample:
  wins/losses/draws = 1846/168/34
  win rate = 90.14%
  decisive win rate = 91.66%

seed 8503, policy_player=0, sample:
  wins/losses/draws = 1859/155/34
  win rate = 90.77%
  decisive win rate = 92.30%

seed 8503, policy_player=1, sample:
  wins/losses/draws = 1856/160/32
  win rate = 90.62%
  decisive win rate = 92.06%
```

The same checkpoint also retained strong performance against Random:

```text
seed 8504, policy_player=0, sample:
  wins/losses/draws = 2039/2/7
  win rate = 99.56%
```

Greedy execution remained below the 90% total-win target in this run, so claims about exceeding 90% should specify sampled policy execution.

## Verification Notes

CUDA was verified with:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The result was:

```text
gpu
[CudaDevice(id=0)]
```
