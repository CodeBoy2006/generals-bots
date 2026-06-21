# Adaptive Search Distillation Design

## Goal

Create an adaptive multisize distillation step that can improve the current best adaptive checkpoint before PPO fine-tuning.

The final objective remains one checkpoint above 90% total win rate against Expander on:

- 8x8 player 0 and player 1.
- 12x12 player 0 and player 1.
- 16x16 player 0 and player 1.

The current best is still `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`, with a verified 512-game-per-row minimum win rate of 70.31%. Recent PPO-only, weighted-pool, timeout-penalty, and wider-network probes did not beat it. The next step must add a stronger teacher signal, not another basic PPO continuation.

## Current Evidence

The adaptive infrastructure now supports:

- One `AdaptivePolicyValueNetwork` checkpoint across 8x8, 12x12, and 16x16 effective boards padded to 16.
- Weighted adaptive reset pools.
- Alternating learner seats.
- Adaptive BC and PPO with configurable channels.
- Output-preserving channel expansion from narrow adaptive checkpoints to wider adaptive networks.
- Full size-seat matrix evaluation.

Failed or plateaued routes:

- Trainer-v2 weighted pool plus timeout penalty: below current best.
- Weighted pool plus alternating seats without timeout penalty: below current best.
- Wide adaptive BC from scratch: weak, especially on 16x16.
- Expanded-width PPO from current best: preserves behavior but still trades strength between size-seat rows.

The consistent bottleneck is not model loading or basic PPO mechanics. It is tactical finish rate and action-quality signal, especially where 16x16 draw rate stays high.

## Considered Approaches

### Recommended: Adaptive Conservative Search Distillation

Port the fixed-size `conservative_search_distill.py` pattern to adaptive checkpoints.

For sampled states from adaptive generated maps:

1. Use a frozen base adaptive checkpoint as the search prior and KL anchor.
2. Score its top-k adaptive actions with short rollouts.
3. Train a student adaptive checkpoint with:
   - KL-to-base on all active samples.
   - Hard or soft top-k search target loss on selected samples.
4. Save a new adaptive `.eqx` checkpoint for normal `evaluate_adaptive_policy.py` and later PPO fine-tune.

This is the strongest next step because it converts search improvements into a reusable checkpoint while guarding against catastrophic drift through KL.

### Alternative: Adaptive Rollout-Search Evaluator Only

Add an adaptive version of `search_policy.py` and measure search-vs-Expander directly.

This is useful for diagnosing whether rollout search is actually stronger on 8/12/16, but it does not itself produce the trained checkpoint the target requires. It can be built after the distillation collector exists, because both need the same adaptive candidate scorer.

### Alternative: More PPO Curricula

Continue with 16-only, 8x16, seat-alternating, timeout-penalized, or wider PPO schedules.

The last several probes already tested this family. None beat the 70.31% baseline. Repeating that family is unlikely to move the target toward 90%.

## Architecture

Add a new script:

```text
examples/_experimental/ppo/adaptive_search_distill.py
```

It should mirror the fixed-size conservative distillation structure but use adaptive modules:

- `adaptive_obs_to_array`
- `compute_adaptive_valid_move_mask`
- `adaptive_index_to_action`
- `adaptive_action_to_index`
- `make_adaptive_state_pool`
- `make_adaptive_initial_states`
- `load_or_create_adaptive_network`

The script should not change fixed-size `conservative_search_distill.py`.

## Adaptive Candidate Search

Add adaptive equivalents of the fixed-size search helpers inside the new script:

```text
adaptive_score_observation(info, obs, player, army_weight, land_weight, terminal_score)
adaptive_rollout_search_candidates(network, state, effective_size, key, player, top_k, rollout_steps, rollouts_per_action, policy_mode, army_weight, land_weight, prior_weight, pad_size)
adaptive_rollout_search_action(...)
```

Candidate generation:

- Encode the learner observation with `adaptive_obs_to_array`.
- Compute the adaptive move mask with `compute_adaptive_valid_move_mask`.
- Use `network.logits_value(obs_arr, mask, active)` to get adaptive logits.
- Use `jax.lax.top_k(logits, top_k)`.
- Decode candidate indices with `adaptive_index_to_action(index, pad_size)`.

Rollout scoring:

- First apply the candidate action and an opponent action.
- For the initial version, the opponent should be the same adaptive base checkpoint in sample mode, not Expander.
- Then roll out both sides with the adaptive base checkpoint for `rollout_steps`.
- Score the final observation using terminal result, army balance, and land balance as in fixed-size search.

The candidate scorer must carry `effective_size` explicitly so padded cells remain excluded from masks and inputs.

## Distillation Objective

Reuse the fixed-size conservative objective shapes:

- `select_search_improvements`
- `search_score_target_probs`
- hard target CE with margin-derived weights
- soft top-k CE from search-score softmax
- KL-to-base over the full adaptive action space

The adaptive network forward path returns `8 * pad_to * pad_to + 1` logits, so KL and sparse CE can operate exactly like the fixed-size version after adaptive encoding.

Defaults should be conservative:

```text
--target-mode soft
--kl-weight 1.0
--improve-weight 0.05
--score-temperature 1.0
--top-k 4
--rollout-steps 16
--rollouts-per-action 2
```

Use low `rollouts-per-action` first because adaptive multisize rollout-search is expensive.

## Data Flow

Each training iteration should:

1. Generate or sample a mixed-size adaptive state pool.
2. Start `num_envs` states with matching effective sizes.
3. For `num_steps`, collect learner observations and search labels.
4. Step the states using the student policy for the learner seat and the frozen adaptive base checkpoint for the opponent seat.
5. Flatten the rollout batch.
6. Train the student for `num_epochs`.
7. Periodically save checkpoints.

Initial implementation should support one scalar `--learner-player 0|1` and the frozen adaptive base checkpoint as the opponent. Alternating seats and Expander-in-the-loop collection can be added after the basic distillation signal is proven.

## CLI

Suggested arguments:

```text
num_envs
--grid-sizes 8,12,16
--grid-size-weights 8:1,12:1,16:2
--pad-to 16
--base-model-path /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx
--init-model-path optional
--model-path /tmp/generals-adaptive-search-distill.eqx
--channels optional
--base-channels optional
--init-channels optional
--target-mode hard|soft
--learner-player 0|1
--num-steps
--num-iterations
--num-epochs
--minibatch-size
--lr
--kl-weight
--improve-weight
--top-k
--rollout-steps
--rollouts-per-action
--checkpoint-dir
--checkpoint-every
--keep-checkpoints
--seed
```

The first useful experiment should avoid widening:

```text
base channels = student channels = 32,32,32,16
base model = /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx
target mode = soft
learner player = 1 first, because current best weakest row is 16x16 p1
```

After a successful narrow run, use `--channels 64,64,64,32 --init-channels 32,32,32,16`.

## Testing

Implementation must use TDD.

Required tests:

- Adaptive search candidate helper returns top-k candidate indices, actions, prior scores, and rollout scores with expected shapes on a padded small board.
- Candidate indices decode to valid adaptive actions or the global pass action.
- Soft target probabilities over top-k scores sum to 1.
- Adaptive conservative loss returns finite loss and metrics for a tiny batch.
- CLI smoke runs one tiny distillation iteration on 4/6 boards padded to 6 and writes a model checkpoint.
- Checkpoint pruning works if checkpoint retention is implemented in the first slice.

Verification commands:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q tests/test_adaptive_ppo.py
JAX_PLATFORMS=cpu uv run --extra dev python -m compileall examples/_experimental/ppo tests
git diff --check
JAX_PLATFORMS=cpu uv run --extra dev pytest -q
```

## Promotion And Experiment Gate

The script is only useful if it produces a checkpoint that improves the size-seat matrix. After implementation:

1. Run a tiny CPU smoke only for wiring.
2. Run a small CUDA narrow distillation from the current best.
3. Evaluate final and retained checkpoints at 128 or 256 games per row.
4. Only promote candidates above the current 70.31% minimum row to 512 games per row.
5. Continue to 2048 games per row only if a candidate is clearly moving toward the 90% target.

If adaptive search distillation does not beat the 70.31% baseline, the next design should focus on target-assignment/full-state auxiliary rewards or a stronger opponent/search evaluator, not more PPO-only continuation.
