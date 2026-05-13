# Heuristic Teacher Pool

Date: 2026-05-13

## Summary

I added a reusable pool of JAX-compatible heuristic agents for the experimental PPO stack and used them both as teachers and as opponents. The new pool keeps the existing Expander heuristic, and adds `city-rush`, `general-hunter`, `defensive-expander`, `balanced`, and `mixed` variants.

The implementation is intentionally shared across training and evaluation so the same decision logic is exercised in:

- behavior cloning labels,
- PPO rollout opponents,
- standalone heuristic-vs-heuristic evaluation,
- and direct agent exports.

## What Changed

### Shared heuristic module

File: `generals/agents/_heuristic_logic.py`

This new module centralizes the JAX-compatible heuristics and keeps their action format aligned with the rest of the project:

- `expander_action`
- `expander_greedy_action`
- `city_rush_action`
- `general_hunter_action`
- `defensive_expander_action`
- `balanced_strategic_action`
- `mixed_heuristic_action`
- `heuristic_action(heuristic_id, key, observation)`

The heuristics are implemented as pure JAX functions so they can be `jit`-compiled and `vmap`-ed inside training/evaluation code. I used `jax.lax.switch` to keep branch selection inside the compiled graph.

The shared scoring logic follows a two-stage pattern:

1. If a move can capture something useful, prioritize capture moves.
2. Otherwise fall back to the heuristic’s positional preference, such as city pursuit, general pursuit, or defensive repositioning.

That split kept the heuristics from wasting too many turns on “pretty” movement that never converts into territory.

### BC teacher support

File: `examples/_experimental/ppo/behavior_clone.py`

Behavior cloning now accepts `--teacher` and can train from:

- `expander-soft`
- `expander`
- `city-rush`
- `general-hunter`
- `defensive-expander`
- `balanced`
- `mixed`

The soft-target Expander path remains available for backward compatibility. The new heuristics use hard one-hot labels derived from sampled teacher actions.

### PPO opponent support

File: `examples/_experimental/ppo/train.py`

The raw PPO trainer now accepts `--opponent` so player 1 can be:

- `random`
- any of the heuristic agents above

### Batch evaluation

File: `examples/_experimental/ppo/evaluate_heuristics.py`

I added a standalone heuristic evaluation script for direct agent-vs-agent testing without requiring a network checkpoint.

### Policy evaluation

File: `examples/_experimental/ppo/evaluate_policy.py`

The policy evaluator now accepts the same heuristic opponent pool.

### Exports

File: `generals/agents/__init__.py`

The heuristic names and dispatch function are now exported for convenience.

## Test Results

Validation ran on 8x8 generated maps with:

- mountain density `0.12-0.22`
- city count `4-8`
- minimum general distance `5`
- 512 games per condition
- seeds fixed at `123`

### Against Random

At 250 steps:

- `expander`: 265/0/247
- `city-rush`: 61/3/448
- `general-hunter`: 146/0/366
- `defensive-expander`: 23/0/489
- `balanced`: 84/0/428
- `mixed`: 122/3/387

At 500 steps:

- `expander`: 480/1/31
- `city-rush`: 273/6/233
- `general-hunter`: 338/0/174
- `defensive-expander`: 130/2/380
- `balanced`: 245/0/267
- `mixed`: 333/3/176

### Against Expander

At 500 steps:

- `city-rush`: 49/402/61
- `general-hunter`: 67/359/86
- `defensive-expander`: 14/365/133
- `balanced`: 68/340/104
- `mixed`: 77/362/73

## Interpretation

The new heuristics are useful as teachers and curriculum opponents, but they are not uniformly stronger than Expander.

The main finding is that Expander remains the strongest immediate baseline on this generated-map setting. The new heuristics provide diversity rather than a clean strength upgrade:

- `city-rush` is the most aggressive of the new variants, but it still loses badly to Expander.
- `general-hunter` is the most credible second-line opponent.
- `balanced` and `mixed` are the most useful for training because they are harder to exploit than a single fixed policy, even when they are not the best in pure win rate.
- `defensive-expander` is intentionally conservative and is the weakest in raw short-horizon conversion.

## Verification

Passed:

- `uv run python -m compileall generals/agents examples/_experimental/ppo tests`
- `uv run pytest`
- heuristic evaluation smoke runs on CUDA with `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`

## Follow-Up

The next improvement would be a true teacher mixer that chooses among the heuristic family per episode or per map family, plus a stronger opponent curriculum for PPO self-play.
