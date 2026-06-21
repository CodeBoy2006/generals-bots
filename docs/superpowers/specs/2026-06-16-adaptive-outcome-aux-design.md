# Adaptive Outcome Auxiliary Design

## Problem

Adaptive PPO is stuck around 70-71% minimum win rate across the 8x8/12x12/16x16 size-seat matrix. The latest v3-noarch and HL-Gauss runs show the same pattern: direct sparse PPO updates can reduce value loss without improving the weak 8x8 and 16x16 rows, and 16x16 still carries high draw rates.

## Chosen Approach

Add an optional outcome auxiliary head to `AdaptivePolicyValueNetwork` and train it from rollout-local, known episode outcomes.

The auxiliary target has three classes from the learner perspective:

- `loss`
- `draw`
- `win`

Labels are assigned by scanning each rollout backward. A sample is supervised only if the same rollout contains a later terminal or truncated transition for that episode segment. Samples whose episode has not finished inside the rollout are masked out. This avoids pretending that unfinished rollout tails have a known final result.

## Non-Goals

- Do not change the PPO reward or terminal reward in this step.
- Do not add privileged full-state inputs.
- Do not train on guessed future outcomes beyond the collected rollout window.
- Do not replace HL-Gauss value support; this auxiliary can compose with MSE or HL-Gauss.

## Implementation Shape

- Network:
  - Add optional `outcome_head=True` construction.
  - Reuse the active-cell pooled value hidden layer.
  - Emit 3 logits through an outcome linear head.
  - Preserve existing `logits_value` and `logits_value_distribution` APIs.
  - Add a richer auxiliary forward method for trainer use.

- Trainer:
  - Add `rollout_outcome_targets(winners, dones, learner_players)`.
  - Add `--outcome-aux-weight`.
  - If weight is positive, create/save a checkpoint with the outcome head.
  - Mix `outcome_aux_weight * masked_cross_entropy` into the PPO minibatch loss.

- Evaluation:
  - Add `--outcome-head` so evaluator can load checkpoints that include the auxiliary head, even though policy action selection does not use it.

## Test Plan

- Unit-test rollout-local outcome target assignment across wins, losses, draws, and unfinished rollout tails.
- Unit-test network auxiliary logits shape and backward-compatible legacy forward APIs.
- Unit-test scalar checkpoint warm-start into an outcome-head target network.
- Unit-test finite outcome auxiliary loss.
- Extend adaptive trainer/evaluator CLI smoke tests to cover saving and loading outcome-head checkpoints.
