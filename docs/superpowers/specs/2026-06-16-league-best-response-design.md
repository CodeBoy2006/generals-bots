# League Best-Response Optimization Design

## Goal

Train and select a pure checkpoint policy that reaches at least 80% total win rate against every current heuristic opponent and against the v5 checkpoint. The gate is intentionally strict:

- Map setting: 8x8 generated, mountain density 0.12-0.22, 4-8 cities, minimum general distance 5.
- Horizon: 500 steps.
- Policy execution: sample mode unless a candidate is explicitly designed for another mode.
- Seats: player 0 and player 1 are evaluated separately.
- Heuristic gate: every heuristic in `HEURISTIC_NAMES` must be above 80% win rate for both seats.
- Checkpoint gate: v5 must be above 80% win rate for both seats.
- Random is kept as a sanity opponent, but it is not a meaningful blocker once heuristic and v5 gates pass.

The current v5 baseline already exceeds Expander, so the main research target is no longer single-opponent fine-tuning. The optimizer must avoid overfitting one frozen opponent and must promote checkpoints only when they improve the whole league.

## Recommended Approach

Build a small checkpoint-league workflow around the existing PPO stack:

1. Add league evaluation that evaluates one candidate against a matrix of opponents.
2. Add periodic checkpoint saving to training so mid-run candidates are not lost.
3. Add multi-opponent frozen checkpoint training so each episode can sample from a pool instead of always fighting one v5 copy.
4. Promote a model only if it improves the league score and does not regress below required heuristic gates.

This should be implemented before more long training runs. Previous experiments show that single-opponent PPO, direct reward shaping, and simple recurrent memory produce only small seat-dependent changes. A league workflow gives stronger evidence and makes every training run reusable.

## Components

### League Evaluation

Create `examples/_experimental/ppo/evaluate_league.py`.

Inputs:

- `candidate_path`
- heuristic opponent list, defaulting to all `HEURISTIC_NAMES`
- checkpoint opponent list, defaulting to `generals-ppo-8x8-expander-gpu-v5.eqx` when present
- optional recurrent candidate metadata
- common map/evaluation settings
- output JSON/Markdown path

Behavior:

- For each opponent and seat, run a fixed-size evaluation.
- Report wins/losses/draws, total win rate, decisive win rate, draw rate, and mean final time.
- Compute a league score as the minimum win rate over all required opponent-seat pairs. This is conservative and aligns with the 80% target.
- Mark each pair as pass/fail against the configured threshold.

The evaluator should reuse existing JAX batch evaluation functions where possible. The first implementation can shell out through shared Python functions rather than trying to unify ordinary and recurrent models behind one large abstraction.

### Periodic Checkpoint Saving

Extend `train.py` and `train_recurrent.py` with:

- `--checkpoint-dir`
- `--checkpoint-every`
- `--keep-checkpoints`

Behavior:

- Save numbered checkpoints at fixed iteration intervals.
- Always save the final `--model-path`.
- Keep the newest N periodic checkpoints when `--keep-checkpoints` is set.

This directly addresses the earlier high-margin/search and PPO experiments where useful intermediate policies could be lost when a run was interrupted or collapsed late.

### Checkpoint Opponent Pool

Extend ordinary PPO training first. Recurrent pool training can follow after the ordinary path is stable.

Inputs:

- `--opponent-policy-pool path1,path2,...`
- `--opponent-policy-pool-modes sample,sample,...` with `sample` default
- optional future weights, not needed in the first version

Behavior:

- Load a small list of frozen checkpoint opponents with matching architecture/input settings.
- During rollout, assign each environment an opponent index sampled from the pool.
- The opponent action dispatcher chooses the appropriate frozen network for that environment.

Initial constraint:

- All opponents in one pool must have the same ordinary `PolicyValueNetwork` architecture and input channel count.
- Recurrent checkpoints are evaluated in the league, but not used as training opponents until recurrent pooling is designed separately.

This keeps the first implementation tractable and avoids mixing ordinary and recurrent hidden-state handling inside the PPO rollout.

## Training Policy

Start with ordinary v5 warm-start because it is the strongest stable pure checkpoint. Use the restored root checkpoint:

```text
generals-ppo-8x8-expander-gpu-v5.eqx
```

Initial opponent pool:

```text
generals-ppo-8x8-expander-gpu-v2.eqx
generals-ppo-8x8-expander-gpu-v3.eqx
generals-ppo-8x8-expander-gpu-v4.eqx
generals-ppo-8x8-expander-gpu-v5.eqx
```

Hold out recurrent and augmented checkpoints for evaluation until the ordinary pool runner is verified.

Recommended first run:

- 512 envs
- 64 steps
- 300-500 iterations
- 4 epochs
- minibatch 4096
- learning rate 1e-6 to 2e-6
- terminal reward scale 1.0 to 2.0
- learner player 0 and player 1 as separate runs
- checkpoint every 50 iterations

After each run, evaluate every periodic checkpoint against:

- all heuristics
- v5
- current recurrent p0 candidate
- the best promoted candidate so far, when one exists

Only promote if the minimum required gate score improves.

## Error Handling

- Missing checkpoint paths fail before training starts.
- Mixed pool architectures fail with a clear message rather than a shape error inside JAX.
- `--opponent-policy-path` and `--opponent-policy-pool` are mutually exclusive.
- League evaluation exits non-zero if `--require-threshold` is set and any required pair fails.
- Evaluation reports should include exact seeds and command settings so results can be reproduced.

## Testing

Unit and smoke coverage:

- argument validation for opponent pool and checkpoint saving
- periodic checkpoint pruning behavior
- rollout smoke test with a two-checkpoint pool
- league evaluator summary math
- league evaluator failure behavior under `--require-threshold`

Verification commands:

```bash
JAX_PLATFORMS=cpu uv run pytest -q
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
```

Training and evaluation evidence must be recorded in `docs/expander-training-strategy.md` after each substantive run.

## Promotion Rule

The goal is achieved only when a checkpoint has authoritative current evidence showing:

- more than 80% total win rate versus every heuristic in both seats
- more than 80% total win rate versus v5 in both seats
- no missing or indirectly inferred opponent-seat result

Until then, the best checkpoint should be described as an interim candidate, not as success.
