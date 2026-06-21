## [2026-05-13 18:10] PPO Training Repair
- **Changes:** Repaired experimental PPO training scripts, declared Equinox/Optax dependencies, fixed policy visualization imports, corrected core army-growth timing, and added an English devlog under `docs/devlogs/`.
- **Status:** Completed
- **Next Steps:** Add a dedicated PPO smoke test and evaluate longer training runs before making policy-quality claims.
- **Context:** Devlog generation must remain outside runtime training code; the raw PPO trainer is the primary supported experimental path.

## [2026-05-13 18:28] CUDA 13 JAX Environment Repair
- **Changes:** Added a reproducible CUDA 13 optional dependency extra, raised the JAX/JAXLIB floor to the installed 0.10 line, and documented GPU setup verification.
- **Status:** Completed
- **Next Steps:** Investigate the remaining non-fatal NVIDIA driver-version parse log if clean stderr is required; PPO training now runs with `JAX_PLATFORMS=cuda`.
- **Context:** The machine has an RTX 5070 Ti with driver 595.79/CUDA 13.2; the failure mode was a stale `jax-cuda13-plugin==0.9.2` beside `jax/jaxlib==0.10.0`.

## [2026-05-13 19:13] Larger PPO Map Support
- **Changes:** Added configurable PPO grid size, generated maps with mountains/cities, reset-pool based raw-trainer auto-reset, larger-model visualization arguments, generator capacity fixes, README examples, and an English devlog.
- **Status:** In Progress
- **Next Steps:** Run compile checks, generated-map training smoke tests, grid tests, and commit if verification passes.
- **Context:** 4x4 remains the default smoke target; meaningful experiments should use larger generated maps such as 8x8 or above with explicit terrain settings.

## [2026-05-13 19:23] Larger PPO Map Support Verification
- **Changes:** Verified the larger-map PPO changes and updated the English devlog with concrete GPU smoke-test results.
- **Status:** Completed
- **Next Steps:** Run longer 8x8+ evaluation jobs before making policy-quality claims; 4x4 should remain only a smoke target.
- **Context:** Verified compile checks, raw PPO 4x4/simple GPU smoke, raw PPO 8x8/generated GPU smoke, GeneralsEnv PPO 8x8/generated GPU smoke, 8x8 model load/forward, targeted grid tests, full pytest, and `git diff --check`.

## [2026-05-13 19:27] Devlog Modification Inventory
- **Changes:** Expanded `docs/devlogs/2026-05-13-larger-map-ppo-support.md` with a file-by-file inventory of the larger-map PPO support changes from commit `1dfe86d`.
- **Status:** Completed
- **Next Steps:** None for this documentation-only update.
- **Context:** This update only records prior code changes in the devlog; it does not change runtime behavior.

## [2026-05-13 19:57] Large-Scale Policy Training
- **Changes:** Added behavior-cloning and batch-evaluation tools, exposed policy logits for supervised training, recorded large-scale GPU training results, and produced `/tmp/generals-bc-8x8-soft-v3.eqx`.
- **Status:** Completed
- **Next Steps:** Use the v3 checkpoint as a warm start for PPO or self-play if stronger-than-Expander performance is required.
- **Context:** Final 8x8 generated-map sampled policy reached 90.8%, 92.1%, and 93.0% win rate over 2048-game/500-step independent evaluations against Random; it is not stronger than Expander.

## [2026-05-13 20:15] Chinese Project Manual
- **Changes:** Added `docs/zh-manual.md` with a detailed Chinese project guide, setup instructions, core API explanation, experiment commands, PPO workflow, evaluation guidance, and current benchmark-script caveats.
- **Status:** Completed
- **Next Steps:** Keep the manual in sync if the README examples or `GeneralsEnv` interface are updated.
- **Context:** The manual intentionally documents the current `reset(key) -> (pool, state)` and `step(state, actions, pool)` interface and warns that older benchmark scripts still need interface cleanup before use.

## [2026-05-13 20:17] README Chinese Summary
- **Changes:** Replaced the original README with a concise Chinese summary derived from `docs/zh-manual.md`, covering setup, project layout, current environment API, common experiments, validation, and caveats.
- **Status:** Completed
- **Next Steps:** None for this documentation-only update.
- **Context:** The README now points readers to `docs/zh-manual.md` for the detailed manual and no longer contains the original English README content.

## [2026-05-13 21:11] Heuristic Teacher Pool
- **Changes:** Added `generals/agents/_heuristic_logic.py` with Expander, CityRush, GeneralHunter, DefensiveExpander, Balanced, and Mixed heuristics; exposed them through `generals/agents/__init__.py`; extended PPO behavior cloning, PPO rollout opponents, and policy evaluation to accept heuristic teachers/opponents; added `examples/_experimental/ppo/evaluate_heuristics.py`; updated `examples/_experimental/README.md` and wrote `docs/devlogs/2026-05-13-heuristic-teacher-pool.md`; added `tests/test_heuristic_agents.py`.
- **Status:** Completed
- **Next Steps:** Consider a per-episode heuristic teacher mixer or a stronger self-play curriculum if higher policy quality is needed.
- **Context:** Heuristic evaluation on 8x8 generated maps showed Expander remains the strongest short-horizon baseline; the new heuristics mainly add diversity and curriculum value rather than beating Expander outright.

## [2026-05-13 21:51] Large Map Heuristic Optimization
- **Changes:** Reworked `generals/agents/_heuristic_logic.py` to use map-scale-aware distance maps, frontier scoring, and large-map weighting; added a 12x12 heuristic smoke test; documented larger-map usage in `examples/_experimental/README.md`; and added `docs/devlogs/2026-05-13-large-map-heuristic-optimization.md`.
- **Status:** Completed
- **Next Steps:** Use 16x16 evaluations with 1000+ step horizons when judging policy quality; consider self-play or search-style opponent logic if the goal is to beat Expander.
- **Context:** Compile checks and pytest passed. Large-map runs showed tuned Mixed improved over its first large-map version, but Expander remains the strongest decisive baseline; 16x16/500 is too short for reliable strength conclusions.

## [2026-06-08 20:55] Parallel Training GIF
- **Changes:** Added `examples/generate_parallel_training_gif.py` to render a tiled batch-rollout training animation and generated `generals/assets/gifs/parallel_training_process.gif`.
- **Status:** Completed
- **Next Steps:** Use `uv run --with pillow python examples/generate_parallel_training_gif.py` to regenerate the asset after visual or simulation tweaks.
- **Context:** Pillow is only a generation-time dependency for the presentation asset; the project runtime dependencies were not changed.

## [2026-06-08 21:08] Parallel Training Video
- **Changes:** Updated the parallel training animation to a 60 second 0.8x loop, added optional MP4 export support to `examples/generate_parallel_training_gif.py`, regenerated `generals/assets/gifs/parallel_training_process.gif`, and added `generals/assets/videos/parallel_training_process.mp4`.
- **Status:** Completed
- **Next Steps:** Use `uv run --with pillow --with imageio --with imageio-ffmpeg python examples/generate_parallel_training_gif.py --video-output generals/assets/videos/parallel_training_process.mp4` to regenerate both assets.
- **Context:** Video generation uses temporary generation-time dependencies only; MP4 metadata verified as H.264, 1120x790, about 60 seconds.

## [2026-06-08 21:21] Dense Square Training Image
- **Changes:** Added `examples/generate_parallel_training_square_image.py` for a 20x20 tiled rollout contact sheet and generated `generals/assets/images/parallel_training_square.png`.
- **Status:** Completed
- **Next Steps:** Regenerate with `uv run --with pillow python examples/generate_parallel_training_square_image.py` if the tile density, seed, or palette should change.
- **Context:** The image intentionally avoids titles, metrics, labels, and explanatory text; it is a 2048x2048 RGB PNG with 400 parallel rollout boards.

## [2026-06-08 21:23] Smaller Square Training GIF
- **Changes:** Added `examples/generate_parallel_training_square_gif.py` and generated `generals/assets/gifs/parallel_training_square_tiled.gif` as an 8x8 tiled square rollout animation.
- **Status:** Completed
- **Next Steps:** Regenerate with `uv run --with pillow python examples/generate_parallel_training_square_gif.py` if the animation length, seed, or tile count should change.
- **Context:** The GIF intentionally avoids titles, metrics, labels, and explanatory text; it is 1280x1280 with 64 parallel boards.

## [2026-06-15 18:44] Expander PPO 90 Percent Run
- **Changes:** Added PPO checkpoint warm-starting, multi-epoch/minibatch PPO updates, behavior-cloning resume support, mirrored policy evaluation, README usage notes, tests for checkpoint loading/result summaries, and a devlog for the 90%+ Expander run.
- **Status:** Completed
- **Next Steps:** Keep `/tmp/generals-ppo-8x8-expander-gpu-v5.eqx` outside git as the current sampled-policy checkpoint; future work can target 90%+ greedy execution or larger maps.
- **Context:** Sampled policy validation against Expander exceeded 90% total win rate across two independent seeds and both player seats; greedy execution stayed below the 90% total-win target.

## [2026-06-15 18:47] Expander Training Strategy Docs
- **Changes:** Added `docs/expander-training-strategy.md` with the reusable Expander training process, strategy notes, commands, evaluation gates, final metrics, and continuation guidance; linked it from README and the Chinese manual.
- **Status:** Completed
- **Next Steps:** Keep the strategy guide updated if future runs change the accepted checkpoint, map distribution, policy mode, or evaluation threshold.
- **Context:** The guide distinguishes sampled-policy 90%+ results from greedy execution, which remains below the same total-win threshold.

## [2026-06-15 19:01] Frozen Policy Self-Play
- **Changes:** Added frozen PPO checkpoint opponents for `train.py`, policy-vs-policy evaluation support, a shared policy action dispatcher, self-play smoke coverage, README/manual usage notes, and self-play guidance in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Use frozen self-play before attempting a full checkpoint league; after each candidate run, re-evaluate against Expander, heuristic opponents, historical checkpoints, and mirrored player seats.
- **Context:** This is frozen checkpoint self-play. The training loop still updates only the learner as player 0; current-vs-current simultaneous self-play remains future work.

## [2026-06-15 19:29] Self-Play Auxiliary Training
- **Changes:** Added learner seat selection and optional terminal win/loss rewards to PPO training; added `examples/_experimental/ppo/outcome_clone.py` for outcome-conditioned winner trajectory cloning with optional loser-action contrastive loss; documented the new commands and v5 self-play experiment results.
- **Status:** Completed
- **Next Steps:** Treat `/tmp/generals-ppo-8x8-expander-gpu-v5.eqx` as the frozen baseline; future attempts to exceed it by 80% likely need a checkpoint league, explicit opponent modeling, search-generated teachers, or larger policy capacity.
- **Context:** Multiple v5-vs-v5 auxiliary runs improved at most to roughly 55% decisive win rate as player 0 and did not approach the 80% target; `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-15 19:48] Rollout Search Teacher
- **Changes:** Added `examples/_experimental/ppo/search_policy.py` for top-k rollout-search policy improvement around a PPO checkpoint; added score tests; documented the search-assisted commands and v5-vs-v5 results in README and the Expander training strategy.
- **Status:** Completed
- **Next Steps:** Use rollout-search as the strong teacher for a more capable distillation path; a pure `.eqx` student still needs architecture/capacity or data-mixing changes before claiming the 80% checkpoint target.
- **Context:** Search-assisted v5 reached 454/46/12 as player 0 and 449/47/16 as player 1 over 512 games against v5 sample. The distilled `/tmp/generals-ppo-8x8-rollout-search-distill-v1.eqx` stayed near 50% decisive, so the active goal is not complete for a pure checkpoint.

## [2026-06-15 19:57] Custom Policy Capacity
- **Changes:** Added custom `PolicyValueNetwork` channel parsing/loading support, exposed learner/opponent channels in PPO training and checkpoint evaluation, and documented expanded-capacity distillation results.
- **Status:** Completed
- **Next Steps:** Use the new `--channels`/`--opponent-channels` paths for larger-capacity experiments, but avoid plain search-label cross-entropy without stronger v5-preservation or confidence filtering.
- **Context:** A v5-expanded `(64,64,64,32)` checkpoint preserved the v5 baseline, but DAgger-style rollout-search distillation regressed to 371/1512/165 against v5 sample; the pure-checkpoint 80% target remains unmet.

## [2026-06-15 20:12] Conservative Search Distillation
- **Changes:** Added reusable rollout-search candidate scoring, introduced `examples/_experimental/ppo/conservative_search_distill.py`, covered the search scoring and conservative loss helpers with tests, and documented the new training route plus same-seed evaluation results.
- **Status:** Completed
- **Next Steps:** Treat conservative search distillation as an experimental tool; pure checkpoint results remain near the v5 baseline, so future attempts should consider explicit Q/value-improvement heads or keeping search as an inference-time planner.
- **Context:** Default and expanded-64 conservative distillation preserved v5 behavior better than plain DAgger but did not approach the 80% pure-checkpoint target. `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-15 20:41] Soft Search and Full-State Input Probe
- **Changes:** Added soft top-k rollout-search targets, separated student and base/KL inputs in conservative distillation, added shared observation/full-state policy input helpers, exposed `evaluate_policy.py --policy-input`, and documented soft/full-state probe results.
- **Status:** Completed
- **Next Steps:** Do not replace the original 9 observation channels with full-state semantics. If continuing the privileged route, extend the network input channels and copy v5 conv1 weights for the original observation channels while zero-initializing extra full-state/search channels.
- **Context:** Soft observation distillation stayed near v5, while replacement full-state inputs regressed to roughly 38-40% total win rate against v5 sample. The active >80% pure-checkpoint target remains unmet.

## [2026-06-15 21:07] Augmented Inputs and Current Self-Play
- **Changes:** Added 18-channel `augmented-full-state` policy inputs, input-channel checkpoint expansion, current-policy self-play via `train.py --self-play-opponent`, evaluation support for non-default input channels, tests for the new loaders/input modes, and updated README/manual/training-strategy docs.
- **Status:** Completed
- **Next Steps:** Treat current-policy self-play and augmented privileged inputs as reusable research tools, but keep rollout-search or a stronger value/improvement head as the main path toward an 80%+ pure checkpoint.
- **Context:** Augmented soft distillation preserved v5 but did not improve; augmented hard v1 showed only small noisy gains, v2 regressed, and current-policy self-play v1 underperformed same-seed v5 baselines. The pure-checkpoint >80% target remains unmet.

## [2026-06-15 21:32] Augmented PPO Best-Response Runs
- **Changes:** Extended `train.py` PPO rollouts to support `--policy-input`, learner/opponent input channels, and 18-channel augmented self-play training; added rollout coverage; documented the full augmented PPO, expanded-64, terminal-reward, and high-margin search runs.
- **Status:** Completed
- **Next Steps:** Add periodic checkpointing before any longer high-margin search-distillation run; direct PPO best-response with current rewards should not be treated as the main route to 80%.
- **Context:** Augmented PPO produced only small seat-dependent movement, expanded-64 did not help, terminal reward scale 20 collapsed to 21.68% over a 512-game check, and the high-margin search run was interrupted around iter 470 without saving a checkpoint. The pure-checkpoint >80% target remains unmet.

## [2026-06-15 21:41] V5 Human Match Launcher
- **Changes:** Added `play-v5.command` as a one-click macOS/terminal launcher for the v5 PPO human match, documented it in README, and added launch-script coverage.
- **Status:** Completed
- **Next Steps:** Use `./play-v5.command` or double-click it in Finder to start the 8x8 generated human-vs-PPO match.
- **Context:** The launcher pins `uv run --python 3.12` to avoid pygame source builds under Python 3.14 and expects the v5 `.eqx` checkpoint in the repository root unless `MODEL_PATH` is set.

## [2026-06-15 21:44] Playable Opening Warmup
- **Changes:** Added playable-mode opening auto-pass logic so human-vs-PPO games start on a frame where the human has at least one legal move; added regression coverage and documented the behavior.
- **Status:** Completed
- **Next Steps:** Restart the match with `./play-v5.command` so the first visible board is already clickable.
- **Context:** Initial Generals states place each general at 1 army, while GUI source selection correctly requires `armies > 1`; without two opening pass turns, every first-frame click is rejected as an invalid source.

## [2026-06-15 21:55] Automatic Playable Ticks
- **Changes:** Added `--auto-tick` and `--tick-rate` to playable PPO matches, made `play-v5.command` auto-advance at 2 turns/sec, and covered tick timing/action selection with tests.
- **Status:** Completed
- **Next Steps:** Launch with `./play-v5.command`; idle human turns now pass automatically while selected-source targeting pauses auto-pass.
- **Context:** Automatic ticks keep the board moving without requiring a human action every turn, while still letting manual clicks override the idle pass action.

## [2026-06-15 21:58] Sample Auto-Tick Defaults
- **Changes:** Switched playable PPO defaults to sample policy mode with auto tick enabled by default, added `--no-auto-tick` as the opt-out, and updated the v5 launcher/docs.
- **Status:** Completed
- **Next Steps:** Use `./play-v5.command` for the default sample + automatic tick human match; add `--policy-mode greedy` or `--no-auto-tick` only for manual overrides.
- **Context:** The v5 checkpoint's documented strongest evaluation mode is sampled execution, so the human-facing launcher should match that default.

## [2026-06-15 22:01] PPO Machine Match Viewer
- **Changes:** Added `--machine-vs-machine`, optional opponent checkpoint/policy mode, shared machine action selection, `watch-v5.command`, tests, and docs for watching PPO-vs-PPO games.
- **Status:** Completed
- **Next Steps:** Use `./watch-v5.command` to watch v5 sample self-play, or pass `--opponent-model-path` to compare two PPO checkpoints.
- **Context:** Machine mode reuses the pygame board and automatic tick loop; no human input is required, but the existing preview panel still shows PPO 0's next candidates.

## [2026-06-15 22:10] Explicit Machine Model Selection
- **Changes:** Added `--model-0-path` and `--model-1-path` for symmetric PPO machine-match model selection, made the primary positional model optional when `--model-0-path` is supplied, and updated `watch-v5.command` to accept `MODEL_0_PATH`/`MODEL_1_PATH`.
- **Status:** Completed
- **Next Steps:** Compare checkpoints with `./watch-v5.command` plus `MODEL_0_PATH=... MODEL_1_PATH=...`, or pass the two explicit CLI flags directly.
- **Context:** Existing positional `model_path` and `--opponent-model-path` remain supported for backward compatibility; the new names are clearer for two-AI matches.

## [2026-06-15 22:21] Augmented Match Viewer Inputs
- **Changes:** Added state-aware PPO policy inputs to `PPOPolicyAgent`, exposed per-model `--model-0-policy-input`/`--model-1-policy-input` and input-channel resolution in `play_against_model.py`, made `play-v5.command`/`watch-v5.command` infer `augmented-full-state` from checkpoint filenames, and documented the overrides.
- **Status:** Completed
- **Next Steps:** Re-run `./watch-v5.command` with the mixed augmented/v5 checkpoint command; use `MODEL_0_POLICY_INPUT` or `MODEL_1_POLICY_INPUT` only if filename inference is wrong.
- **Context:** The augmented PPO checkpoint has 18 conv1 input channels, while ordinary v5 observation checkpoints have 9. Loading it without `augmented-full-state` caused the Equinox shape mismatch at `conv1.weight`.

## [2026-06-15 22:28] General Target Reward
- **Changes:** Added state-aware general-target shaping reward, wired it into PPO rollout CLI, added reward/rollout tests, and documented v5 warm-start aggressive training results.
- **Status:** Completed
- **Next Steps:** Treat `--general-target-reward-scale` as an aggression knob, not a strength improvement yet; tune lower scales or combine with checkpoint saving/search teacher before longer runs.
- **Context:** `/tmp/generals-ppo-8x8-general-target-p0-v1.eqx` lowered player-0 draw rate and mean final time but did not beat the same-seed v5 baseline; the pure-checkpoint >80% target remains unmet.

## [2026-06-15 22:49] Path Assignment Reward
- **Changes:** Added shortest-path target-assignment shaping with reward-local distance caches, wired it into PPO rollout CLI, added reward/rollout tests, and documented v5 warm-start training results.
- **Status:** Completed
- **Next Steps:** Treat `--path-assignment-reward-scale` as an experimental transport-shaping knob; avoid frontier weight by default until target quality is improved.
- **Context:** `/tmp/generals-ppo-8x8-path-assignment-p0-v2.eqx` slightly improved player-1 total win rate against same-seed v5 but did not improve player 0 or decisive win rate; the pure-checkpoint >80% target remains unmet.

## [2026-06-15 22:58] Auto-Detected Match Viewer Inputs
- **Changes:** Replaced filename-based launcher inference with `policy_input=auto`, added agent-side checkpoint input layout auto-detection for 9-channel observation and 18-channel augmented-full-state checkpoints, and updated launch tests/docs.
- **Status:** Completed
- **Next Steps:** Launch mixed checkpoints without policy-input overrides; use explicit `MODEL_0_POLICY_INPUT`/`MODEL_1_POLICY_INPUT` only for 9-channel full-state checkpoints or future unsupported layouts.
- **Context:** `generals-ppo-8x8-path-assignment-p0-v2.eqx` has 18 conv1 input channels but does not include `augmented` in its filename, so filename-based inference still loaded it as a 9-channel observation policy.

## [2026-06-16 11:09] Residual GRU PPO
- **Changes:** Added residual GRU recurrent PPO network, recurrent train/evaluate scripts, heuristic/checkpoint opponent support, freeze-base training, tests, and docs with training results.
- **Status:** Completed
- **Next Steps:** Restore `/tmp/generals-ppo-8x8-expander-gpu-v5.eqx` before testing v5 warm-start memory gains; start with `--freeze-base`, then cautiously unfreeze or lower LR.
- **Context:** Current environment lacked v5/v4 checkpoints, so fallback BC/RNN runs were evaluated against Expander. Frozen-base RNN preserved weak BC strength but did not improve it; fresh RNN PPO did not learn Expander wins.

## [2026-06-16 11:21] V5 Residual GRU Training Results
- **Changes:** Documented the restored-v5 recurrent PPO runs, including freeze-base training commands, v5-vs-v5 same-seed comparisons, Expander evaluations, and the current best recurrent checkpoint path.
- **Status:** Completed
- **Next Steps:** Treat `/tmp/generals-recurrent-ppo-8x8-v5-freeze-p0-v1.eqx` as the best recurrent candidate so far, but keep v5 as the main baseline; next experiments should add periodic checkpoint selection or cautiously unfreeze later CNN layers.
- **Context:** The p0-trained frozen-base GRU improved slightly over same-seed v5 baselines and Expander evaluations, but the gain is still small and does not approach the 80%+ pure-checkpoint best-response target. `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-16 11:28] League Best-Response Design
- **Changes:** Added `docs/superpowers/specs/2026-06-16-league-best-response-design.md` specifying the checkpoint-league workflow, 80% per-opponent/per-seat promotion gate, periodic checkpoint saving, opponent policy pools, and league evaluation requirements.
- **Status:** Completed
- **Next Steps:** Implement league evaluation and periodic checkpointing first, then add ordinary PPO opponent-policy pools before starting longer best-response runs.
- **Context:** The active target is now stricter than the earlier Expander goal: the final checkpoint must exceed 80% total win rate against every current heuristic strategy and v5 in both player seats. `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-16 11:32] League Best-Response Implementation Plan
- **Changes:** Added `docs/superpowers/plans/2026-06-16-league-best-response.md` with a task-by-task plan for league evaluation, periodic checkpoint saving, ordinary checkpoint opponent pools, verification, documentation, and the first league training run.
- **Status:** Completed
- **Next Steps:** Execute Task 1 first: implement `evaluate_league.py` and its tests, then commit before moving to checkpoint saving.
- **Context:** The plan keeps recurrent checkpoints as evaluation targets first, not training-pool opponents, to avoid mixing hidden-state handling into ordinary PPO rollouts prematurely. `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-16 11:33] Worktree Isolation Prep
- **Changes:** Added `.worktrees/` to `.gitignore` so project-local git worktrees can be created without risking accidental commits of nested checkout files.
- **Status:** Completed
- **Next Steps:** Create a `league-best-response` worktree branch and execute the implementation plan there.
- **Context:** This is preparation for isolated implementation of the league evaluator, checkpoint saving, and opponent-pool training changes. `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-16 11:39] League Evaluator
- **Changes:** Added `examples/_experimental/ppo/evaluate_league.py` for ordinary checkpoint league evaluation against heuristic and checkpoint opponents; added `tests/test_evaluate_league.py` covering checkpoint spec parsing, league summary math, required heuristic selection, and a CLI smoke evaluation.
- **Status:** Completed
- **Next Steps:** Implement periodic checkpoint saving in ordinary and recurrent PPO training before adding opponent checkpoint pools.
- **Context:** Targeted verification passed with `JAX_PLATFORMS=cpu uv run pytest -q tests/test_evaluate_league.py` in the `league-best-response` worktree.

## [2026-06-16 11:43] Periodic Training Checkpoints
- **Changes:** Added periodic checkpoint helpers and CLI flags to ordinary and recurrent PPO training; both scripts can now save numbered `*-iter-000NNN.eqx` checkpoints and prune older periodic files.
- **Status:** Completed
- **Next Steps:** Implement ordinary PPO opponent checkpoint pools so best-response training can sample multiple frozen historical policies.
- **Context:** Targeted verification passed for `tests/test_ppo_train.py` and `tests/test_recurrent_ppo.py`, including CLI smoke runs that write and prune periodic checkpoints.

## [2026-06-16 11:48] PPO Opponent Checkpoint Pool
- **Changes:** Added `--opponent-policy-pool` and `--opponent-policy-pool-modes` to ordinary PPO training, same-architecture frozen pool loading, pool action dispatch, and rollout support that samples one frozen opponent index per environment per training iteration.
- **Status:** Completed
- **Next Steps:** Run full verification, document league workflow commands, then start the first v5 warm-start league best-response training run.
- **Context:** `JAX_PLATFORMS=cpu uv run pytest -q tests/test_ppo_train.py` passed, including parser, loader, and CLI smoke coverage for checkpoint-pool training.

## [2026-06-16 11:50] League Workflow Docs
- **Changes:** Documented checkpoint-pool training, periodic checkpoint saving, and `evaluate_league.py` promotion-gate usage in README, the Chinese manual, and `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Start the first league best-response run from v5 with v2-v5 as the frozen opponent pool, then evaluate the resulting checkpoints with `evaluate_league.py`.
- **Context:** Full CPU verification passed before the docs update: `139 passed`, compileall succeeded, and `git diff --check` passed.

## [2026-06-16 12:04] Rollout Search League Evaluation
- **Changes:** Extended `evaluate_league.py` with `--search-policy` support so rollout-search actions around a checkpoint can be evaluated against the same heuristic/checkpoint league matrix; added CLI smoke coverage.
- **Status:** Completed
- **Next Steps:** Evaluate `v5 + rollout-search` against all required heuristics and v5 to determine whether the search-assisted strategy already crosses the 80% gate.
- **Context:** Checkpoint-pool PPO training produced only noise-level v5 gains, so the next high-signal path is to verify the previously strong rollout-search strategy under the stricter league gate.

## [2026-06-16 12:13] Search-Assisted League Results
- **Changes:** Documented the first checkpoint-pool PPO run and the `v5 + rollout-search` league evidence in `docs/expander-training-strategy.md`; added README guidance for `evaluate_league.py --search-policy`.
- **Status:** Completed
- **Next Steps:** If the target accepts search-assisted strategy, treat `v5 + rollout-search` as the current 80%+ solution; if the target requires a pure `.eqx` checkpoint, continue with search distillation or a value-improvement head.
- **Context:** Checkpoint-pool PPO iter 100 only reached a 44.14% league score against v5 gates. `v5 + rollout-search` reached 90.23% as player 0 and 88.67% as player 1 against v5 over 512 games, and scored at least 97.66% over the 128-game heuristic league matrix.

## [2026-06-16 12:21] Search-Assisted Heuristic 512-Gate
- **Changes:** Updated `docs/expander-training-strategy.md` with the expanded 512 games/row `v5 + rollout-search` heuristic league matrix.
- **Status:** Completed
- **Next Steps:** Run full verification, merge the `league-best-response` implementation branch, and keep pure-checkpoint distillation as future work rather than the current 80% solution.
- **Context:** `v5 + rollout-search` passed all 12 heuristic opponent-seat pairs over 512 games/row with a minimum total win rate of 96.68%; v5-vs-v5 search evidence remains 90.23% as player 0 and 88.67% as player 1 over 512 games.

## [2026-06-16 12:58] Rollout Search GUI
- **Changes:** Added rollout-search support to `examples/play_against_model.py`, `play-v5.command`, and `watch-v5.command`; documented `SEARCH_POLICY`, `MODEL_0_SEARCH_POLICY`, and `MODEL_1_SEARCH_POLICY` GUI usage in README, the Chinese manual, and the expander training strategy.
- **Status:** Completed
- **Next Steps:** Use `SEARCH_POLICY=1 ./play-v5.command` for human-vs-search or `MODEL_0_SEARCH_POLICY=1 ./watch-v5.command` to watch search-assisted v5 against ordinary v5.
- **Context:** GUI search wraps 9-channel observation checkpoints only. Verification passed with full `uv run --python 3.12 pytest -q`, `compileall`, `git diff --check`, and a real v5 lightweight rollout-search action smoke test.

## [2026-06-16 13:19] Adaptive Multisize Expander Design
- **Changes:** Added `docs/superpowers/specs/2026-06-16-adaptive-multisize-expander-design.md` specifying the single-checkpoint adaptive PPO target for 8x8, 12x12, and 16x16 Expander play, including fixed 16x16 canvas padding, active-cell features, adaptive action encoding, masked value pooling, curriculum, and per-size promotion gates.
- **Status:** Completed
- **Next Steps:** Write the implementation plan, then implement adaptive network/common helpers and smoke tests before starting BC/PPO training runs.
- **Context:** The target is stricter than current fixed-size v5 evidence: success requires one checkpoint, sampled policy execution, both seats, and more than 90% total win rate on each required size. `.superpowers/` remains an unrelated untracked local directory.

## [2026-06-16 13:27] Adaptive Multisize Implementation Plan
- **Changes:** Added `docs/superpowers/plans/2026-06-16-adaptive-multisize-expander.md` with a task-by-task plan for adaptive common helpers, adaptive network, state pools, behavior cloning, PPO training, evaluation, documentation, and first training runs.
- **Status:** Completed
- **Next Steps:** Execute Task 1 first using TDD: adaptive parsing, active-cell input encoding, valid masks, and single-pass action encoding.
- **Context:** The plan keeps fixed-size PPO paths intact and adds adaptive scripts beside them. It does not yet prove the 90% target; it defines the infrastructure path toward producing and verifying one 8/12/16 checkpoint.

## [2026-06-19 20:11] Web Renderer Design
- **Changes:** Added `docs/superpowers/specs/2026-06-19-web-renderer-design.md` specifying the browser Canvas renderer, Python/FastAPI WebSocket session layer, JSON snapshot and command protocols, migration phases, error handling, dependencies, and verification strategy for replacing pygame-only display paths.
- **Status:** Completed
- **Next Steps:** Review and approve the spec, then write an implementation plan before adding `generals/web/`, `examples/play_web.py`, and frontend assets.
- **Context:** The design keeps Python/JAX authoritative for game rules and PPO inference, retains pygame during migration, and targets remote browser usability without porting JAX or checkpoints to WebAssembly.

## [2026-06-19 20:17] Web Renderer Implementation Plan
- **Changes:** Added `docs/superpowers/plans/2026-06-19-web-renderer.md` with a TDD-oriented task plan for web snapshot serialization, `WebGameSession`, FastAPI/Uvicorn server entry point, static browser Canvas UI, documentation, verification, and push.
- **Status:** Completed
- **Next Steps:** Execute Task 1 first: write failing protocol tests for JSON-safe snapshots, then implement `generals/web/schemas.py`.
- **Context:** The plan keeps pygame compatibility as an explicit verification gate and treats Python/JAX as authoritative for game state, move validation, PPO inference, and rollout search.

## [2026-06-19 20:30] Web Renderer Server Entry
- **Changes:** Added the FastAPI/Uvicorn web server entry point, browser CLI argument parsing, expanded web session configuration, and server/CLI protocol tests.
- **Status:** Completed
- **Next Steps:** Build the static Canvas browser UI and wire it to the `/ws/game` snapshot and command protocol.
- **Context:** The server creates game sessions only after WebSocket connection, so importing the CLI or creating the app does not start pygame rendering or load checkpoints.

## [2026-06-19 20:35] Web Renderer Browser UI
- **Changes:** Added the static browser Canvas UI, WebSocket client, responsive HUD/control layout, real asset mounts for existing Generals images/fonts, and static asset coverage in tests.
- **Status:** Completed
- **Next Steps:** Update README and the Chinese manual with the new browser entry point, then run full regression and browser smoke verification.
- **Context:** The page uses the `/ws/game` snapshot protocol directly, hides fogged cells on the client, and sends semantic commands for select, move, pass, cancel, split, auto tick, and restart.

## [2026-06-19 20:37] Web Renderer Documentation
- **Changes:** Documented `generals/web/`, `examples/play_web.py`, remote browser access, WebSocket/Canvas responsibilities, trusted-network caveat, and browser control mapping in README and the Chinese manual.
- **Status:** Completed
- **Next Steps:** Run full regression, smoke the server and browser UI, then merge and push the completed implementation.
- **Context:** The browser renderer is now documented as the preferred path for remote machines without GUI display, while the pygame path remains available for local desktop use.

## [2026-06-19 20:46] Web Runtime Decoupling
- **Changes:** Moved shared PPO map/model/rollout-search factories into `generals/agents/ppo_runtime.py`, updated pygame and web entry points to use it, added a no-pygame web runtime import test, and fixed the browser board frame to stay square.
- **Status:** Completed
- **Next Steps:** Run final verification, merge the `web-renderer` worktree branch back to master, and push.
- **Context:** Browser WebSocket sessions now load PPO agents without importing `generals.gui` or emitting pygame startup logs; pygame remains available through `examples/play_against_model.py`.

## [2026-06-19 21:20] Web Move Queue
- **Changes:** Added server-authoritative browser move queues, queued pass/undo/clear commands, projected target validation for chained moves, generals.io-style keyboard bindings, queue HUD/canvas rendering, and updated web controls documentation.
- **Status:** Completed
- **Next Steps:** Use `examples/play_web.py` to continue playtesting longer games and tune any queue affordances that feel unclear.
- **Context:** Auto tick now consumes one queued human action per due tick, pauses only when the human has a selection and the queue is empty, and otherwise auto-passes as before.

## [2026-06-19 22:17] Web Unlimited Match Length
- **Changes:** Removed the browser renderer's default 500-step cap, made `--max-steps 0` equivalent to unlimited, preserved explicit positive max-step limits, and documented the new web default.
- **Status:** Completed
- **Next Steps:** Start `examples/play_web.py` normally for unlimited browser matches, or pass `--max-steps N` only for bounded smoke tests.
- **Context:** The pygame play script and training/evaluation utilities keep their existing max-step defaults; this change is scoped to the browser session path.

## [2026-06-19 22:39] Web Selected Auto Tick
- **Changes:** Let browser auto tick continue while a source cell is selected, preserving the selection only while it remains a valid movable source and clearing it after state changes make it invalid.
- **Status:** Completed
- **Next Steps:** Playtest the browser queue flow at normal tick rates to confirm the new real-time selection behavior feels right.
- **Context:** Queued moves still take priority over auto pass. The pygame play script keeps its existing selected-cell pause behavior.

## [2026-06-19 23:36] Web Dynamic Model Control
- **Changes:** Added a browser model catalog, per-player Human/Model control modes, active human selection, dynamic model switching, session-side agent caching, and a Control panel for live takeover/hosting without restarting the match.
- **Status:** Completed
- **Next Steps:** Playtest with multiple compatible 8x8 checkpoints and decide whether to add an explicit safe allowlist for external model directories.
- **Context:** The model selector scans CLI-provided checkpoints plus `.eqx` files in the repository root and `legacymodels/`; incompatible checkpoints fail gracefully instead of crashing the WebSocket.

## [2026-06-19 23:50] Web Stable Model Select
- **Changes:** Changed the browser Control panel to incrementally sync existing player rows and select options instead of rebuilding the DOM on every tick, preserving focused model/control selectors while snapshots stream in.
- **Status:** Completed
- **Next Steps:** Continue playtesting live handoff at higher tick rates and with slower model loads.
- **Context:** Added a Node-backed static app regression test proving tick renders keep the same select nodes and preserve focused values.
## [2026-06-16 13:33] Adaptive Common Helpers
- **Changes:** Added `examples/_experimental/ppo/adaptive_common.py` with adaptive grid-size parsing, active-cell/padding input channels, adaptive valid-move masks, and single-pass action encoding; added `tests/test_adaptive_ppo.py` coverage for those primitives.
- **Status:** Completed
- **Next Steps:** Implement `AdaptivePolicyValueNetwork` with fixed adaptive action space and active-cell value pooling.
- **Context:** Task 1 follows TDD: tests first failed on missing adaptive module, then passed after implementation. Padding is distinguished through explicit adaptive channels because fogged observations may report padded mountains as structures-in-fog rather than visible mountains.

## [2026-06-16 13:35] Adaptive Policy Network
- **Changes:** Added `examples/_experimental/ppo/adaptive_network.py` with `AdaptivePolicyValueNetwork`, eight movement planes, one global pass logit, active-cell masked pooling for the value/pass heads, checkpoint loading, and tests for finite forward/sample outputs.
- **Status:** Completed
- **Next Steps:** Add adaptive size-balanced reset pools and Expander target distributions for behavior cloning.
- **Context:** Task 2 followed TDD: network tests first failed on missing module, then passed after implementation. The existing fixed-size `PolicyValueNetwork` remains unchanged.

## [2026-06-16 13:37] Adaptive Pool And Targets
- **Changes:** Extended `examples/_experimental/ppo/adaptive_common.py` with padded simple-grid generation, size-balanced adaptive state pools, adaptive initial-state selection, and soft Expander target distributions using the single-pass adaptive action space.
- **Status:** Completed
- **Next Steps:** Add `behavior_clone_adaptive.py` and its CLI smoke test so the adaptive checkpoint can get an Expander warm start.
- **Context:** Task 3 followed TDD: pool/target tests first failed on missing helpers, then passed after implementation. The pool assigns remainder slots to larger sizes to bias scarce capacity toward harder maps.

## [2026-06-16 13:40] Adaptive Behavior Cloning
- **Changes:** Added `examples/_experimental/ppo/behavior_clone_adaptive.py` for mixed-size adaptive BC from Expander-soft or heuristic teachers, plus CLI smoke coverage in `tests/test_adaptive_ppo.py`.
- **Status:** Completed
- **Next Steps:** Add the adaptive PPO trainer so BC checkpoints can be fine-tuned against Expander.
- **Context:** Task 4 followed TDD: CLI smoke first failed on missing script, then passed after implementation. The smoke test trains a tiny 4/6 padded checkpoint on CPU and verifies an `.eqx` artifact is written.

## [2026-06-16 13:44] Adaptive PPO Trainer
- **Changes:** Added `examples/_experimental/ppo/train_adaptive.py` with adaptive rollout collection, PPO loss/update, effective-size reset handling, periodic checkpoint saving, and CLI smoke coverage.
- **Status:** Completed
- **Next Steps:** Add adaptive policy evaluation across configured sizes and seats.
- **Context:** Task 5 followed TDD: trainer smoke first failed on missing script, then passed after implementation. The trainer keeps the existing fixed-size PPO scripts untouched and writes periodic checkpoints using the existing checkpoint helper.

## [2026-06-16 13:49] Adaptive Policy Evaluator
- **Changes:** Added `examples/_experimental/ppo/evaluate_adaptive_policy.py` for adaptive size/seat matrix evaluation, JSON output, sample/greedy policy modes, optional win-rate threshold, and CLI smoke coverage.
- **Status:** Completed
- **Next Steps:** Update README, Chinese manual, expander training strategy, then run broader verification.
- **Context:** Task 6 followed TDD: evaluator smoke first failed on missing script, then passed after implementation. A JAX concretization error in `lax.scan(length=max_steps)` was fixed by switching evaluator JIT wrappers to `eqx.filter_jit` so scalar config stays static.

## [2026-06-16 13:55] Adaptive Multisize Documentation
- **Changes:** Documented the adaptive 8/12/16 BC, PPO, and evaluation workflow in `README.md`, `docs/zh-manual.md`, and `docs/expander-training-strategy.md`, including the six-row size/seat 90% acceptance gate.
- **Status:** Completed
- **Next Steps:** Start a full adaptive BC warm start, then run adaptive PPO with periodic checkpoints and evaluate candidate `min_win_rate` across all required sizes and seats.
- **Context:** Verification passed after the docs update: adaptive focused tests, full CPU pytest, compileall, and `git diff --check`. This proves the infrastructure and docs, not the 90% target checkpoint.

## [2026-06-16 14:03] Adaptive BC Checkpoints
- **Changes:** Added `--checkpoint-dir`, `--checkpoint-every`, and `--keep-checkpoints` to `behavior_clone_adaptive.py`, plus CLI coverage and docs updates for long adaptive BC warm-start runs.
- **Status:** Completed
- **Next Steps:** Run a medium CPU adaptive BC baseline using periodic checkpoints, then evaluate the resulting checkpoint across the 8/12/16 size-seat matrix.
- **Context:** Task followed TDD: the checkpoint-pruning CLI test first failed because the adaptive BC parser lacked the flags, then passed after wiring the existing checkpoint helpers. Full CPU pytest passed with 155 tests.

## [2026-06-16 14:08] Adaptive CPU Baseline
- **Changes:** Documented a medium CPU adaptive BC and PPO baseline in `docs/expander-training-strategy.md`, including size-seat evaluation matrices for `/tmp/generals-adaptive-bc-medium.eqx` and `/tmp/generals-adaptive-ppo-medium.eqx`.
- **Status:** Completed
- **Next Steps:** Run the full adaptive BC recipe on a CUDA JAX environment, or first design a dual-seat adaptive PPO mode so one checkpoint trains against both required evaluation seats.
- **Context:** The CPU baseline verified the training/evaluation chain and checkpoint retention but did not approach the target: both BC and short PPO had `min_win_rate = 0.00%` over 32 games/row at 300 steps.

## [2026-06-16 14:14] Adaptive Hard-Teacher Baseline
- **Changes:** Added the hard `--teacher expander` adaptive BC iter-100 comparison to `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Avoid spending more CPU time on short hard-teacher BC; use CUDA for full BC or design dual-seat adaptive PPO before longer PPO runs.
- **Context:** The hard-teacher checkpoint also had `min_win_rate = 0.00%` over 32 games/row at 300 steps, with 8x8 player 1, 12x12 player 1, and both 16x16 seats at 0% wins.

## [2026-06-16 14:17] Adaptive Player-1 PPO Baseline
- **Changes:** Documented a short `learner_player=1` adaptive PPO baseline in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Design dual-seat or alternating-seat adaptive PPO before spending longer PPO runs on a single learner seat.
- **Context:** Player-1 short PPO from the same soft BC checkpoint improved a few rows locally but still had `min_win_rate = 0.00%`, with 12x12 player 1 and both 16x16 seats at 0% wins.

## [2026-06-16 14:51] Adaptive GPU PPO V1
- **Changes:** Documented the first CUDA adaptive BC/PPO run in `docs/expander-training-strategy.md`, including `/tmp/generals-adaptive-bc-gpu-v1.eqx`, alternating PPO checkpoints, 16x16-only curriculum, and the current best `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`.
- **Status:** Completed
- **Next Steps:** Continue from the 70.31% min-win checkpoint with a targeted plan for 16x16 draw reduction and 8x8 decisive strength, likely requiring draw/timeout reward shaping, size weighting, or dual-seat batch training.
- **Context:** CUDA JAX is available through `uv run --extra dev --extra cuda13`. The best adaptive checkpoint reached `min_win_rate = 70.31%` over 512 games/row at 750 steps, not the required >90% on all 8x8/12x12/16x16 seats.

## [2026-06-16 15:06] Adaptive PPO Plateau Probes
- **Changes:** Documented additional CUDA continuation probes in `docs/expander-training-strategy.md`: 8x16 p1/p0 curricula, second 16x16-only p1 continuation, and all-size p1 with `--terminal-reward-scale 2.0`.
- **Status:** Completed
- **Next Steps:** Stop repeating single-seat low-lr continuations; implement or design a new training signal such as draw/timeout reward shaping, size weighting, or dual-seat batch training.
- **Context:** None of the probes beat the current best `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` at `min_win_rate = 70.31%` over 512 games/row. The best rejected follow-up was terminal-reward-scale 2.0 at 69.92% over 256 games/row.

## [2026-06-16 15:12] Adaptive Trainer V2 Design
- **Changes:** Added `docs/superpowers/specs/2026-06-16-adaptive-trainer-v2-design.md` describing weighted reset pools, truncation reward shaping, alternating learner seats, tests, and the first GPU continuation recipe from the current best checkpoint.
- **Status:** Completed
- **Next Steps:** Review the spec, then implement the trainer controls with TDD before launching the next CUDA PPO run.
- **Context:** This is a design-only step. The 90% adaptive Expander target remains open; the current best verified minimum row is still 70.31% over 512 games/row.

## [2026-06-16 15:14] Adaptive Trainer V2 Plan
- **Changes:** Added `docs/superpowers/plans/2026-06-16-adaptive-trainer-v2.md` with TDD tasks for weighted size sampling, truncation reward shaping, alternating learner seats, documentation, verification, and commit/push.
- **Status:** Completed
- **Next Steps:** Execute the plan inline with red-green tests, then run the trainer-v2 CUDA continuation.
- **Context:** This is still pre-training infrastructure work. The adaptive 90% Expander target remains open.

## [2026-06-16 15:24] Adaptive Trainer V2 Controls
- **Changes:** Added weighted adaptive reset-pool sampling, `--grid-size-weights` support in adaptive BC/PPO, PPO truncation reward shaping, `--learner-player alternate`, tests, and docs for the next Expander PPO continuation.
- **Status:** Completed
- **Next Steps:** Run the CUDA trainer-v2 continuation from `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`, then evaluate retained checkpoints across all 8/12/16 size-seat rows.
- **Context:** Verification passed with focused adaptive tests, compileall, `git diff --check`, and full CPU pytest. This implements training controls only; the 90% Expander target remains unproven until an independently evaluated checkpoint clears every required row.

## [2026-06-16 15:40] Adaptive Capacity Route
- **Changes:** Added `--channels` support to `behavior_clone_adaptive.py`, extended adaptive BC smoke coverage, and documented trainer-v2/no-trunc CUDA results plus the next larger-capacity adaptive route.
- **Status:** Completed
- **Next Steps:** Train a `64,64,64,32` adaptive BC warm start, continue PPO against Expander, and evaluate all retained checkpoints across 8x8/12x12/16x16 and both seats.
- **Context:** Trainer-v2 controls alone did not beat the current best 70.31% minimum row: v2 reached 67.97% over 256 games/row, iter-100 reached 67.19% over 512 games/row, and the no-truncation control reached 68.36% over 256 games/row.

## [2026-06-16 15:59] Adaptive Channel Expansion
- **Changes:** Added output-preserving adaptive channel expansion warm starts via `init_channels`, exposed `--init-channels` in `train_adaptive.py`, added regression coverage that expanded logits/value match the source checkpoint while extra conv channels remain trainable, and documented wide-from-BC negative results.
- **Status:** Completed
- **Next Steps:** Expand `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` from `32,32,32,16` to `64,64,64,32`, run PPO continuation, and evaluate retained checkpoints.
- **Context:** Wide-from-BC did not preserve current strength: `/tmp/generals-adaptive-bc-wide-v1.eqx` reached 16.41% min over 128 games/row, wide small-batch PPO reached 18.75%, and wide p0 PPO reached 31.25%. The new expansion path preserves the current best policy before fine-tuning.

## [2026-06-16 16:24] Expanded Adaptive PPO Probes
- **Changes:** Documented expanded adaptive PPO probes from the current best checkpoint: all-size expanded-v1, trainable-extra-channel expanded-v2, and 16x16-only player-1 fine-tune.
- **Status:** Completed
- **Next Steps:** Stop repeating basic PPO continuations; design an adaptive distillation or stronger-teacher step that can improve finish rate before PPO fine-tuning.
- **Context:** None of the expanded PPO probes beat the current best 70.31% minimum row. Expanded-v1 reached 68.75% over 256 games/row, its iter-100 checkpoint reached 67.58% over 512 games/row, expanded-v2 reached 68.36% over 256 games/row, and expanded 16p1 reached 68.75% over 256 games/row.

## [2026-06-16 16:26] Adaptive Search Distillation Design
- **Changes:** Added `docs/superpowers/specs/2026-06-16-adaptive-search-distillation-design.md` for an adaptive conservative rollout-search distillation step that reuses the fixed-size KL/search-target pattern with adaptive encoders, action space, and reset pools.
- **Status:** Completed
- **Next Steps:** Write the implementation plan, then implement the adaptive search-distillation script with TDD.
- **Context:** This is a design-only step. The current best remains `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` at 70.31% minimum win rate over 512 games/row; the 90% target remains open.

## [2026-06-16 16:36] Adaptive Search Distillation Plan
- **Changes:** Added `docs/superpowers/plans/2026-06-16-adaptive-search-distillation.md` with a TDD implementation plan for adaptive rollout-search distillation, CLI smoke coverage, checkpoint retention, verification, and the first experiment gate.
- **Status:** Completed
- **Next Steps:** Execute the plan task-by-task, starting with adaptive loss tests and the new `adaptive_search_distill.py` scaffold.
- **Context:** The plan intentionally scopes the first implementation to scalar learner seat selection and a frozen adaptive base opponent before adding alternating seats or Expander-in-the-loop collection.

## [2026-06-16 16:50] Adaptive Search Distillation Implementation
- **Changes:** Added `examples/_experimental/ppo/adaptive_search_distill.py` with adaptive rollout-search candidates, hard/soft conservative losses, adaptive batch collection, CLI training loop, and checkpoint retention. Added focused tests in `tests/test_adaptive_ppo.py`.
- **Status:** Completed
- **Next Steps:** Run a small CUDA narrow distillation from `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`, evaluate retained checkpoints at 128 or 256 games per size-seat row, and only promote checkpoints above the current 70.31% minimum row to 512-game evaluation.
- **Context:** This implements the stronger-teacher step; it does not prove the 90% target. The best verified checkpoint remains `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` until a new evaluation beats it.

## [2026-06-16 16:55] Adaptive Search Distillation Probe
- **Changes:** Ran p1 narrow adaptive search distillation from `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` to `/tmp/generals-adaptive-search-distill-p1-v1.eqx`, evaluated retained iter 10/20/30/40 checkpoints, and documented results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Treat `/tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx` as the next low-bar candidate, then test stronger search-budget distillation or a low-learning-rate PPO fine-tune; do not promote to 2048-row final validation until the minimum row moves much closer to 90%.
- **Context:** iter 40 reached 512 games/row win rates: 8p0 71.68%, 8p1 74.61%, 12p0 82.81%, 12p1 83.40%, 16p0 71.68%, 16p1 71.29%, min 71.29%. This is above the prior 70.31% baseline but far below the >90% target.

## [2026-06-16 17:05] Adaptive Search Distillation Follow-ups
- **Changes:** Ran and documented three follow-ups in `docs/expander-training-strategy.md`: p1-to-p0 sequential search distillation, p1 distillation with `rollout_steps=16`, and low-lr alternate PPO from the p1 r8 iter40 candidate.
- **Status:** Completed
- **Next Steps:** Stop repeating simple seat swaps, rollout-budget bumps, or standard alternate PPO follow-ups; next implementation should change the objective, such as high-confidence search-improvement filtering, finish/draw auxiliary targets, or true dual-seat same-batch KL/CE.
- **Context:** Follow-up 256-row minimums did not beat the p1 r8 iter40 71.29%/512 candidate: p1->p0 reached at best 68.36%, r16 reached at best 69.92%, and PPO follow-up reached at best 69.53%.

## [2026-06-16 17:10] Adaptive Soft Search Weight Mode
- **Changes:** Added `--soft-weight-mode active|improvement` to `examples/_experimental/ppo/adaptive_search_distill.py`; improvement mode reuses margin-based search-improvement weights for soft targets instead of weighting every active sample. Added focused tests in `tests/test_adaptive_ppo.py`.
- **Status:** Completed
- **Next Steps:** Run a high-confidence adaptive distillation probe with `--soft-weight-mode improvement`, tuned `--min-margin`, and retained checkpoints.
- **Context:** Verified with `tests/test_adaptive_ppo.py`, compileall, `git diff --check`, and full `pytest` (`164 passed`). This is intended to reduce noisy all-sample soft distillation; it has not yet been evaluated for win rate.

## [2026-06-16 17:16] Adaptive Improvement-Weighted Probe
- **Changes:** Ran and documented two `--soft-weight-mode improvement` p1 probes in `docs/expander-training-strategy.md`: `min_margin=1, margin_scale=4` and `min_margin=0.2, margin_scale=1`.
- **Status:** Completed
- **Next Steps:** Do not use pure improvement-only weighting as a replacement for active soft targets; implement a mixed objective or finish/draw/Q auxiliary target next.
- **Context:** Improvement-only probes did not beat the active-soft p1 r8 iter40 candidate at 71.29%/512. Best 256-row minimums were 71.09% for margin 1 and 69.92% for margin 0.2.

## [2026-06-16 17:24] Adaptive Mixed Soft Search Objective
- **Changes:** Added `--soft-improvement-extra-weight` to `examples/_experimental/ppo/adaptive_search_distill.py`, allowing active soft target training plus an extra high-margin improvement CE term. Updated tests and documented the parameter in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Run a mixed objective probe from `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` using `--soft-weight-mode active --soft-improvement-extra-weight`.
- **Context:** Verified with `tests/test_adaptive_ppo.py` (`22 passed`), compileall, `git diff --check`, and full pytest (`165 passed`). No win-rate experiment has been run for the mixed objective yet.

## [2026-06-16 17:31] Adaptive Mixed Objective Probes
- **Changes:** Ran and documented two mixed-objective p1 probes in `docs/expander-training-strategy.md`: active soft plus improvement extra CE at weights `0.02` and `0.005`.
- **Status:** Completed
- **Next Steps:** Stop tuning the current search-CE family; next direction should be outcome/finish/draw auxiliary learning or an adaptive rollout-search evaluator to estimate teacher upper bound across 8/12/16.
- **Context:** Mixed v1 reached 71.88% min over 256 games/row at iter 10 but fell to 68.95% over 512 games/row. Mixed v2 reached only 71.09% over 256 games/row. The best candidate remains `/tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx` at 71.29% over 512 games/row.

## [2026-06-16 21:21] Adaptive PPO V3-Noarch Controls
- **Changes:** Added adaptive PPO controls for `--reward-mode terminal`, configurable `--gamma`/`--gae-lambda`, top-advantage policy filtering, EMA parameter tracking/saving, and updated README/manual/training strategy docs. Preserved existing mixed-seat and per-size value-head work.
- **Status:** Completed
- **Next Steps:** Run the full GPU v3-noarch continuation from the 71.29% search-distill candidate for 80 iterations, then evaluate retained EMA checkpoints at 256 games/row before any 512-row promotion.
- **Context:** Verification passed with `tests/test_adaptive_ppo.py` (28 passed), `compileall`, `git diff --check`, and full CPU pytest (171 passed). CUDA smoke used `CudaDevice(id=0)` and produced `/tmp/generals-adaptive-ppo-v3-noarch-gpu-smoke.eqx`; 64 games/row eval had `min_win_rate = 67.19%`, which validates the GPU/EMA/evaluator path but is too small to judge model strength.

## [2026-06-16 21:31] Adaptive PPO V3-Noarch GPU Probes
- **Changes:** Ran and documented GPU v3-noarch continuations in `docs/expander-training-strategy.md`, including the 512-env OOM, the working 256-env terminal EMA run, a same-seed base control, a composite reward control, and a terminal last-iterate control.
- **Status:** Completed
- **Next Steps:** Stop tuning the current no-architecture PPO recipe; implement HL-Gauss/by-size value targets or finish/draw auxiliary heads next, then rerun GPU triage.
- **Context:** Local GPU is a 16GB RTX 5070 Ti, so 512 envs x 256 steps with minibatch 4096 OOMs in `train_minibatch_step`; 256 envs x 256 steps with minibatch 1024 runs around 88k SPS. Best v3 256-row result was terminal EMA at `min_win_rate = 69.53%` on seed 66030, only noise-level above the same-seed base control at 68.75% and below the prior 512-row 71.29% candidate.

## [2026-06-16 21:51] Adaptive HL-Gauss Value Heads
- **Changes:** Added HL-Gauss categorical value helpers, optional shared/per-size categorical value heads, PPO categorical value loss, train/eval CLI value-loss flags, scalar-to-categorical warm-start coverage, and README/manual/strategy docs for the v3-hlgauss route.
- **Status:** Completed
- **Next Steps:** Run the 256-env GPU v3-hlgauss triage from the 71.29% search-distill checkpoint and compare retained checkpoints with 256 games/row before any 512-row promotion.
- **Context:** Verification passed with focused adaptive tests (`32 passed`), compileall, `git diff --check`, and full CPU pytest (`175 passed`). CUDA smoke used `CudaDevice(id=0)`, trained `/tmp/generals-adaptive-ppo-v3-hlgauss-smoke.eqx`, and confirmed the categorical checkpoint loads in the evaluator with matching `--value-loss hl-gauss`; the 16 games/row smoke is a loader/runtime check, not a strength result.

## [2026-06-16 22:01] Adaptive HL-Gauss GPU Triage
- **Changes:** Added `--value-heads shared|per-size` to `evaluate_adaptive_policy.py`, extended CLI coverage for per-size categorical checkpoint loading, and documented the 256-env v3-hlgauss GPU triage in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote the v3-hlgauss checkpoints; implement a representation/auxiliary-target change next, such as memory stack/global context, finish/draw auxiliary heads, or search-to-Q/intent distillation.
- **Context:** The 40-iteration CUDA run saved `/tmp/generals-adaptive-ppo-v3-hlgauss.eqx` and iter 10/20/30/40 checkpoints. Best 256 games/row result was iter 30 at `min_win_rate = 69.92%`, only slightly above the same-seed scalar base at 69.14% and below the prior 71.29%/512-row candidate. Verification passed with focused adaptive tests (`32 passed`), compileall, `git diff --check`, and full CPU pytest (`175 passed`).

## [2026-06-16 22:13] Adaptive Outcome Auxiliary Head
- **Changes:** Added an optional loss/draw/win outcome auxiliary head to `AdaptivePolicyValueNetwork`, rollout-local outcome target assignment, masked auxiliary CE in adaptive PPO, train/eval CLI flags, tests, docs, and a design spec.
- **Status:** Completed
- **Next Steps:** Run the 256-env v3-outcome GPU triage from the 71.29% search-distill checkpoint and compare final/retained checkpoints with 256 games/row before any promotion.
- **Context:** Verification passed with focused adaptive tests (`36 passed`), compileall, `git diff --check`, full CPU pytest (`179 passed`), and a CUDA smoke that trained `/tmp/generals-adaptive-ppo-v3-outcome-smoke.eqx` then loaded it with `evaluate_adaptive_policy.py --outcome-head`. The smoke evaluation used only 16 games/row at 300 steps and is not a strength result.

## [2026-06-16 22:21] Adaptive Outcome Auxiliary GPU Triage
- **Changes:** Documented outcome auxiliary GPU probes in `docs/expander-training-strategy.md`, including weight `0.05`, lower weight `0.005`, same-seed scalar base control, and a 512 games/row promotion check.
- **Status:** Completed
- **Next Steps:** Stop sweeping this exact rollout-local outcome auxiliary loss; next implementation should change representation or teacher signal, such as memory/global context channels, scoreboard history, or search-to-Q/intent distillation.
- **Context:** `outcome_aux_weight=0.05` damaged size-seat stability with best 256-row min `67.58%`. `0.005` reached `71.88%` at iter 10 over 256 games/row but failed 512-row promotion at `68.95%`, below the existing 71.29%/512-row search-distill candidate.

## [2026-06-16 22:35] Adaptive Global Context Branch
- **Changes:** Added optional 20-channel adaptive global-context inputs, a zero-initialized scoreboard MLP branch in `AdaptivePolicyValueNetwork`, train/eval CLI flags, warm-start support from legacy 15-channel checkpoints, focused tests, and README/manual/strategy docs.
- **Status:** Completed
- **Next Steps:** Run the 256-env CUDA global-context triage from `/tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx`, then evaluate retained checkpoints at 256 games/row before any 512-row promotion.
- **Context:** Verification passed with focused adaptive tests (`39 passed`), compileall, `git diff --check`, full CPU pytest (`182 passed`), and CUDA train/eval smoke on `cuda:0`. The smoke only validates training/saving/loading with `--global-context`; it is not a strength result.

## [2026-06-16 22:40] Adaptive Global Context GPU Triage
- **Changes:** Ran the 256-env CUDA global-context triage and documented retained checkpoint evaluations in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote the global-context checkpoints; next try should add scoreboard history/memory channels, search-to-Q/intent targets, or a lower-risk distillation objective for the new global branch.
- **Context:** Best retained global checkpoint was iter 10 at `69.14%` min over 256 games/row on seed 69040. It improved same-seed source 16p1 but remained below the existing 512-row `71.29%` search-distill candidate; iter 20/30/40 continued the PPO drift pattern.

## [2026-06-16 22:52] Adaptive Scoreboard History Branch
- **Changes:** Added `--scoreboard-history` 30-channel adaptive inputs with previous scoreboard features plus one-step deltas, per-env history carry/reset in adaptive PPO, evaluator history carry, variable-width global context MLP loading, tests, and docs.
- **Status:** Completed
- **Next Steps:** Run the 256-env CUDA scoreboard-history triage from the 71.29% search-distill checkpoint and compare retained checkpoints at 256 games/row.
- **Context:** Focused adaptive tests passed (`43 passed`), compileall and `git diff --check` passed, and CUDA train/eval smoke ran on `cuda:0`. Full pytest was interrupted after 53 passing tests to prioritize fast training iteration per user direction.

## [2026-06-16 22:58] Adaptive Scoreboard History GPU Triage
- **Changes:** Ran scoreboard-history CUDA triage, documented 256-env OOM and 128-env retained checkpoint evaluations in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote history checkpoints; next fast follow-up should reduce PPO drift, for example lower LR/shorter history continuation or distillation/replay for the new history/global branch.
- **Context:** 256 envs OOMed even with minibatch 512. The 128-env run completed, and best retained checkpoint was iter 40/final at `70.31%` min over 256 games/row on seed 70040, improving same-seed source `68.75%` but below the existing 512-row `71.29%` candidate.

## [2026-06-16 23:02] Adaptive History Lower-LR Follow-up
- **Changes:** Ran and documented a lower-LR (`1e-6`) scoreboard-history CUDA follow-up at 128 envs for 20 iterations.
- **Status:** Completed
- **Next Steps:** Stop plain PPO continuation sweeps for history inputs; implement a distillation/replay path that trains the new global/history representation while anchoring the old policy.
- **Context:** Lower LR improved same-seed source min from `66.02%` to `68.75%` over 256 games/row, mainly repairing 16p0/16p1, but remained below the 71.29%/512 best and does not merit promotion.

## [2026-06-16 23:08] Adaptive History Search Distillation
- **Changes:** Added `--global-context` and `--scoreboard-history` student-input support to `examples/_experimental/ppo/adaptive_search_distill.py`, including per-env scoreboard-history carry/reset and 15-channel base/search KL anchors. Updated the focused CLI smoke test and docs.
- **Status:** Completed
- **Next Steps:** Run the short CUDA history-distill p1 probe, evaluate retained checkpoints at 256 games/row, and only promote if the six-row min clearly beats the current `71.29%`/512 best.
- **Context:** Verification was intentionally focused for speed: compileall, `git diff --check`, one CPU CLI smoke (`1 passed`), and a CUDA smoke on `cuda:0` saving `/tmp/generals-adaptive-history-distill-smoke.eqx`.

## [2026-06-16 23:14] Adaptive Mixed-Seat Search Distillation
- **Changes:** Added `adaptive_search_distill.py --learner-player mixed`, splitting envs across p0/p1, collecting both search-labeled batches, and concatenating them into one distillation update. Updated the focused smoke test and docs.
- **Status:** Completed
- **Next Steps:** Run the mixed-seat history-distill CUDA probe and compare retained checkpoints against the same-seed source before any wider promotion.
- **Context:** Verification stayed lightweight per user direction: compileall, `git diff --check`, one CPU CLI smoke (`1 passed`), and a CUDA mixed smoke on `cuda:0` saving `/tmp/generals-adaptive-history-distill-mixed-smoke.eqx`.

## [2026-06-16 23:19] Adaptive Freeze-Legacy Distillation
- **Changes:** Added `adaptive_search_distill.py --freeze-legacy-weights`, which masks gradients to the new post-15-channel conv1 inputs and global/history MLP while freezing the legacy adaptive trunk and heads. Updated the focused smoke test and docs.
- **Status:** Completed
- **Next Steps:** Run a mixed-seat history-distill probe with `--freeze-legacy-weights` and compare 256-row retained checkpoints against the active-soft mixed run.
- **Context:** Verification stayed lightweight: compileall, `git diff --check`, one CPU CLI smoke (`1 passed`), and a CUDA freeze smoke on `cuda:0` saving `/tmp/generals-adaptive-history-distill-freeze-smoke.eqx`.

## [2026-06-16 23:24] Adaptive History Distill GPU Triage
- **Changes:** Ran and documented single-seat, mixed-seat, freeze-legacy, and high-LR freeze-legacy history-distill CUDA probes in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote any history-distill checkpoint; stop this action-level search-CE/history sweep and move to value/finish/Q/intent targets from search or outcome labels.
- **Context:** On eval seed 71140, source min was `64.84%`. Best history-distill retained checkpoints only reached `67.19%` over 256 games/row (`p0-only iter10` and `freeze lr=1e-4 iter10`), while mixed-seat active-soft fell to `61.33%` by iter20. Current best remains `/tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx` at `71.29%` over 512 games/row.

## [2026-06-16 23:29] Adaptive Search Value Distillation
- **Changes:** Added `--search-value-weight` and `--search-value-scale` to `adaptive_search_distill.py`, deriving a bounded scalar value target from top-k rollout-search scores and adding a weighted MSE value loss alongside action KL/CE. Updated focused tests and docs.
- **Status:** Completed
- **Next Steps:** Run a mixed-seat scoreboard-history CUDA probe with search-value supervision and compare retained checkpoints against the same-seed source.
- **Context:** Verification stayed focused: RED test failed on missing loss API, then passed after implementation; related loss/collector tests passed, CPU CLI smoke passed, and CUDA smoke with `--search-value-weight 0.1` saved `/tmp/generals-adaptive-search-value-smoke.eqx`.

## [2026-06-16 23:35] Adaptive Search Value GPU Triage
- **Changes:** Ran and documented mixed-seat search-value CUDA probes, a p0 continuation, and a value-first lower action-CE variant in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote search-value checkpoints; implement explicit finish/draw/outcome supervision from actual rollout terminal status.
- **Context:** Best retained search-value checkpoint was `/tmp/generals-adaptive-search-value-mixed-v1-ckpts/generals-adaptive-search-value-mixed-v1-iter-000020.eqx` at `68.36%` min over 256 games/row on seed 71140. It improved 16p1 to `71.48%` but left 16p0 at `68.36%`, below the current `71.29%`/512 best.

## [2026-06-16 23:41] Adaptive Search Outcome Distillation
- **Changes:** Added `--search-outcome-weight` to `adaptive_search_distill.py`, deriving loss/draw-or-unfinished/win labels from the best rollout-search candidate and training the existing outcome head with weighted CE. Updated focused tests and docs.
- **Status:** Completed
- **Next Steps:** Run a mixed-seat scoreboard-history CUDA probe with search outcome supervision and evaluate retained checkpoints with `--outcome-head`.
- **Context:** Verification stayed focused: RED test failed on missing outcome loss API, then passed after implementation; related loss/collector/search tests and CPU CLI smoke passed; CUDA smoke with `--search-outcome-weight 0.1` saved `/tmp/generals-adaptive-search-outcome-smoke.eqx`.

## [2026-06-17 00:05] Adaptive Outcome and PPO V3 GPU Triage
- **Changes:** Documented CUDA outcome-distillation results, PPO v3-noarch reruns, OOM boundaries, same-seed source controls, and checkpoint-structure findings in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop plain noarch PPO reward/value sweeps; implement true seat x size stratified PPO batches or gradient-conflict mitigation before spending more GPU time on this CNN trainer.
- **Context:** Outcome distillation failed promotion (`70.31%` over 256 games/row dropped to `65.82%` over 512). PPO v3 reruns improved some weak seeds but moved the bottleneck to another seat/size; best new 256-row min was `69.92%`, below the current `71.29%`/512 search-distill candidate.

## [2026-06-17 10:42] Adaptive Strategy Auxiliary Heads
- **Changes:** Added `adaptive_strategy_aux.py`, optional strategy auxiliary heads on `AdaptivePolicyValueNetwork`, and Q/intent/finish/belief losses in `adaptive_search_distill.py`; added `evaluate_adaptive_policy.py --strategy-aux`; moved training defaults and examples away from `/tmp` to ignored `runs/`; documented the Phase 1 strategy-aux pivot.
- **Status:** Completed
- **Next Steps:** Move the real 71.29% adaptive candidate into `runs/`, then run mixed-seat scoreboard-history strategy-aux distillation on GPU and triage at 256 games/row before any 512-row promotion.
- **Context:** Focused adaptive tests passed (`6 passed`), compileall and `git diff --check` passed, and CUDA train/load smoke passed on `cuda:0` using `runs/strategy-aux-smoke-postpatch/`. The smoke uses a synthetic tiny checkpoint and is not a strength result; the historical best checkpoint was absent from its old `/tmp` path.

## [2026-06-17 11:11] History-Base Strategy Aux Probe
- **Changes:** Added `adaptive_search_distill.py --base-global-context` / `--base-scoreboard-history`, allowing shared-MSE 20/30-channel adaptive checkpoints to act as base/search/opponent teacher networks with per-seat scoreboard-history carry through rollout search. Added a focused collector test and documented `legacymodels/` ranking plus v1/v2 GPU probes.
- **Status:** Completed
- **Next Steps:** Do not promote the v1/v2 strategy-aux checkpoints. Next probe should disable all-active action soft labels or restrict action CE to high-margin rows while keeping Q/intent/belief representation losses; optionally add base template flags for outcome/HL-Gauss teacher checkpoints.
- **Context:** `legacymodels/` does not contain the exact 71.29% historical best, but `generals-adaptive-ppo-v3-composite-balanced-probe1.eqx` was the best shared-MSE history teacher found in 64 games/row triage (`70.31%` min on seed 74000). v2 reached `68.75%` min on seed 74000 and `64.06%` on seed 74210, so it is an engineering validation rather than a promotion candidate. Focused verification passed: compileall, `git diff --check`, and 3 targeted CPU pytest cases.

## [2026-06-17 11:29] Strategy Aux High-Margin Probes
- **Changes:** Added `adaptive_search_distill.py --freeze-strategy-aux-only`, which masks gradients to only intent/finish/Q/belief auxiliary heads for policy-preserving auxiliary pretraining. Documented v3 high-margin-only CE, v4 close-spawn curriculum, v5 aux-only pretrain plus fine-tune, and 256 games/row base/v3 comparison.
- **Status:** Completed
- **Next Steps:** Do not promote v3/v4/v5. Stop sweeping this strategy-aux action-distill variant; next concrete branch should either build an offline replay/representation trainer or start Worker pretraining with BFS/path-assignment target heatmaps to address 16x draw/finish execution.
- **Context:** Best new checkpoint was `runs/adaptive-strategy-aux-v3/ckpts/generals-adaptive-strategy-aux-v3-iter-000008.eqx`; it reached `67.19%` min over 256 games/row on seed 74000 versus `66.80%` for the shared-MSE history base, with 16p0 still bottlenecked by draws. v4 close-spawn and v5 aux-head pretrain both topped out at `67.19%` over 64 games/row and were not worth promotion.

## [2026-06-17 11:38] Adaptive Worker BFS Pretraining
- **Changes:** Added `adaptive_worker_pretrain.py`, an 18-channel Worker pretraining script that appends target heatmap, eligible source heatmap, and BFS route potential to adaptive observations, then trains soft BFS-progress action targets from full-state labels. Added a focused Worker label test and documented GPU Worker runs.
- **Status:** Completed
- **Next Steps:** Do not treat Worker checkpoints as promotion policies yet. Next step is to split Worker outputs into source and direction heads or wire a thin Commander/route wrapper that supplies target/route channels, then test whether Worker execution reduces 16x draw rate.
- **Context:** Hard one-hot Worker labels were too fragmented despite being legal; label top share was only about `2.1%`. After switching to soft BFS-progress targets, `runs/adaptive-worker-pretrain-general-v2/` improved Useful support accuracy from `7.1%` to `62.7%` over 100 GPU iterations on `cuda:0`. The checkpoint is stored under `runs/`, not a cache directory.

## [2026-06-17 11:50] Worker Command Wrapper Evaluation
- **Changes:** Added `evaluate_worker_policy.py`, which supplies Worker target/source/route channels from fogged observation and can optionally hybridize a Worker checkpoint with a fallback adaptive policy using visible-general/contact/turn triggers. Added observation-command tests and documented GPU wrapper evaluations.
- **Status:** Completed
- **Next Steps:** Stop hard-switching to the flat-action Worker. Next Worker branch should split source selection and direction prediction, or use Worker logits only as a candidate reranker under the adaptive policy rather than replacing policy actions.
- **Context:** Pure Worker `general-v2` scored `0.00%` min over 64 games/row on seed 74710. Hybrid fallback with Worker never triggered stayed in base range (`64.06%` min), validating the evaluator path, but visible-general/contact/turn Worker takeover collapsed to `0.00-15.62%` min. The negative result shows the current flat action head learned route support but cannot execute finish moves precisely.

## [2026-06-17 11:58] Worker Split-Loss Probe
- **Changes:** Added Worker source/direction target marginalization and split loss weights to `adaptive_worker_pretrain.py`, preserving the old flat action CE defaults. Added a focused marginalization test and documented the GPU split-loss probe plus hybrid visible-general evaluation.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-worker-split-general-v1` as a policy. Use its logits only as a small rerank bias/candidate proposer under the adaptive fallback policy, or move to explicit separate source/direction heads before another hard-switch attempt.
- **Context:** GPU split run in `runs/adaptive-worker-split-general-v1/` reached final `Acc 21.7%`, `Src 40.0%`, `Dir 51.8%`, `Useful 91.0%`, `Mass 0.172`, improving supervised Worker metrics over flat `general-v2`. However a 64 games/row hybrid visible-general takeover still scored only `14.06%` min win rate, so hard-switch Worker control remains rejected.

## [2026-06-17 12:04] Worker Logit Rerank Probe
- **Changes:** Added `evaluate_worker_policy.py --hybrid-mode rerank --worker-logit-scale`, which centers legal Worker logits and applies them as a bias on fallback adaptive logits instead of switching control. Added a focused rerank helper test and documented the GPU scale sweep.
- **Status:** Completed
- **Next Steps:** Do not reuse raw Worker action logits as production bias. Next Worker branch should train explicit source/direction heads or a dedicated candidate-rerank head against fallback-success/search-Q labels.
- **Context:** On seed 74820 with 64 games/row, rerank scale `0.00` matched fallback at `68.75%` min, scale `0.05` stayed `68.75%` but hurt 16x p1 and draw rate, and scale `0.10` dropped to `67.19%` min. The result rejects raw Worker-logit bias for promotion.

## [2026-06-17 12:12] Strategy-Q Rerank Probe
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-q-rerank-scale`, which centers legal strategy auxiliary Q predictions and applies them as a policy-logit bias for `--strategy-aux` checkpoints. Added a focused helper test and documented 64/256 games-per-row GPU probes.
- **Status:** Completed
- **Next Steps:** Do not promote raw strategy-Q rerank. If Q is pursued further, train/calibrate a dedicated candidate-ranking head with seat/size-balanced batches and inference-scaled targets.
- **Context:** On v3 iter8, 64 games/row seed 74900 peaked at `67.19%` min with scale `0.02`, but 256 games/row seed 74920 showed only a tiny best improvement: scale `0.01` reached `67.97%` min versus `67.58%` at scale `0.00`, still far below the `71.29%` candidate. Q bias mostly moves weakness between 16p0/16p1 instead of solving it.

## [2026-06-17 12:18] Strategy-Q Pairwise Rank Probe
- **Changes:** Added `adaptive_search_distill.py --strategy-q-rank-weight` / `--strategy-q-rank-min-margin`, a pairwise candidate ranking loss for strategy-Q heads. Added a focused rank-loss test and documented the short GPU aux-only calibration run.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-strategy-q-rank-v1`. Q-ranking needs online validation or better target normalization before it should bias policy logits.
- **Context:** The rank loss optimized on GPU (`StratRank 0.1746 -> 0.0381`, `StratQ 90.0352 -> 76.4928`) in `runs/adaptive-strategy-q-rank-v1/`, but inference got worse: 64 games/row seed 75020 scored `64.06%` min at rerank scale `0.01` and `57.81%` at `0.02`. The run shows rank supervision is wired but not yet aligned with policy replacement outcomes.

## [2026-06-17 12:27] 8x8 P1 League Best-Response Probe
- **Changes:** Ran the missing fixed 8x8 player-1 PPO league best-response probe from v5 against the v2-v5 checkpoint pool, stored model/log artifacts under `runs/8x8-league-p1-v1/`, and documented checkpoint evaluations in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop ordinary fixed 8x8 PPO continuation against the historical checkpoint pool; use rollout-search inference/replacement-outcome distillation or return to adaptive long-rollout seat/size-balanced training.
- **Context:** Intermediate checkpoints iter040/080/120/160 all stayed below frozen v5 in both seats; final 512-game evaluations scored p0 `47.85%` and p1 `43.55%` total win rate versus v5 sample. `legacymodels/` contains old adaptive/search checkpoints but no fixed 8x8 v5/Expander models.

## [2026-06-17 12:34] Strategy-Q Replacement Outcome Target
- **Changes:** Added `adaptive_search_distill.py --strategy-q-target {score,outcome,outcome-score}` and `--strategy-q-outcome-score-weight`, documented the option in README, added a focused target-conversion test, and recorded short CUDA outcome-Q probes in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote the outcome-Q checkpoints. If Q rerank is revisited, use longer rollouts or policy-action replacement episodes so terminal outcome labels are less sparse.
- **Context:** Pure outcome target produced `StratRank 0.0000` because short candidate rollouts usually tied as draw/unfinished. `outcome-score` created rank signal but rerank still hurt: 64 games/row max750 min fell from `68.75%` at scale `0.00` to `67.19%` at `0.005` and `65.62%` at `0.01`.

## [2026-06-17 12:38] Strategy-Q Outcome R16 Diagnostic
- **Changes:** Ran and documented a longer candidate-search diagnostic with `--rollout-steps 16` and pure outcome Q/rank targets under `runs/adaptive-strategy-q-outcome-r16-v1/`.
- **Status:** Completed
- **Next Steps:** Stop outcome-Q rerank calibration from short candidate rollouts; prefer online validation, full policy-action replacement episodes, or a dedicated accepted-replacement rerank dataset.
- **Context:** Outcome rank signal appeared only intermittently (`StratRank 0.6351` at iter5 but `0.0000` at iter10/final). 64 games/row max750 eval scored `62.50%` min at rerank scale `0.00` and `59.38%` at `0.005`, so longer candidate rollout did not produce a usable inference bias.

## [2026-06-17 12:46] Strategy-Q Outcome R64 Diagnostic
- **Changes:** Ran and documented `rollout_steps=64` outcome-Q calibration under `runs/adaptive-strategy-q-outcome-r64-v1/`, `v2/`, and `v3/`, including 64-row scale sweeps and a 256-row confirmation.
- **Status:** Completed
- **Next Steps:** Do not promote raw outcome-Q rerank. Use the r64 result only as evidence that long candidate rollouts can create outcome diversity; the next version should train a dedicated accepted-replacement rerank head or online gate.
- **Context:** r64 produced stable outcome rank signal and reduced StratQ from roughly `82` to `36`, but inference did not validate. Best 64-row scale was `0.001` at `70.31%` min; 256-row confirmation fell to `66.80%` with 16p1 bottlenecked.

## [2026-06-17 12:47] Accepted-Replacement Q Weighting
- **Changes:** Added `adaptive_search_distill.py --strategy-q-weight-mode accepted`, which trains Q/rank only on candidate rows where long rollout search finds an outcome-improving or high-margin same-outcome replacement for the top-prior action. Added focused tests, README docs, training-log `StratS`, and a short CUDA accepted-r64 probe.
- **Status:** Completed
- **Next Steps:** Do not continue accepted-only online batches without replay/oversampling; accepted rows are too sparse. If revisited, build an accepted-row buffer or dedicated rerank dataset.
- **Context:** `runs/adaptive-strategy-q-accepted-r64-v1/` saw only `6-22` accepted Q rows per 512 samples and unstable Q/rank losses. 64 games/row max750 min improved from `59.38%` at scale `0.000` to `62.50%` at `0.001`, still far below the current platform.

## [2026-06-17 12:50] Accepted-Replacement Low-LR Fine-Tune
- **Changes:** Ran and documented `runs/adaptive-strategy-q-accepted-r64-v2/`, a low-LR accepted-row fine-tune initialized from the stronger outcome-r64-v3 Q checkpoint.
- **Status:** Completed
- **Next Steps:** Stop accepted-only online fine-tuning; build replay/oversampling before returning to accepted replacements.
- **Context:** Accepted rows remained sparse (`5-26` per 512). 64 games/row max750 min was `68.75%` with no Q bias, then dropped to `62.50%` at scale `0.001` and `67.19%` at scale `0.002`.

## [2026-06-17 12:55] Long-Rollout Mixed PPO Recheck
- **Changes:** Ran and documented two mixed-seat long-rollout PPO probes from the shared-MSE history base: `runs/adaptive-ppo-terminal-hlgauss-mixed-v1/` and `runs/adaptive-ppo-long-mixed-v2/`.
- **Status:** Completed
- **Next Steps:** Do not continue these exact PPO recipes. Next useful branch should add replay-balanced accepted replacements or change the network/data representation instead of more small-CNN PPO continuation.
- **Context:** Terminal+HL-Gauss v1 collapsed rollout wins after iter1 and only reached `67.19%` min at iter10 EMA on a 64-row eval. Conservative composite v2 was stable but only moved same-seed min from base `64.06%` to `65.62%` at iter10, then final EMA returned to `64.06%`.

## [2026-06-17 13:06] Accepted Replay and Policy Distill
- **Changes:** Added accepted-row strategy-Q replay (`--strategy-q-replay-capacity`, `--strategy-q-replay-ratio`) and accepted policy distill weighting (`--soft-weight-mode accepted`) to `adaptive_search_distill.py`, with focused tests and README/docs updates. Ran GPU probes under `runs/adaptive-strategy-q-accepted-replay-r64-v1/` and `runs/adaptive-accepted-policy-r64-v1/`.
- **Status:** Completed
- **Next Steps:** Do not promote raw Q replay or accepted action distill. The next aligned branch should change representation/architecture or build a safer accepted-replacement objective, not directly bias policy logits from the current Q head.
- **Context:** Q replay raised effective Q samples to about `1030` per update, but 256-row scale `0.001` still scored only `68.75%` min; the scale-0 `73.05%` matched the init policy exactly and was seed variance. Accepted policy distill hurt 16p0, scoring `60.94%` min versus same-seed base `70.31%`.

## [2026-06-17 13:21] Context Residual PPO Probe
- **Changes:** Added optional zero-output 5x5 residual context branch to `AdaptivePolicyValueNetwork`, with `train_adaptive.py --context-residual`, `--context-only-update`, evaluator loading via `--context-residual`, README docs, and a focused warm-start invariance test. Ran GPU probes in `runs/adaptive-context-residual-only-v1/` and `runs/adaptive-context-residual-joint-v2/`.
- **Status:** Completed
- **Next Steps:** Do not promote these context PPO checkpoints. Next branch should pretrain representation heads with belief/finish/intent supervision or move to a real U-Net/Transformer torso rather than more small residual PPO continuation.
- **Context:** Context-only v1 had a favorable 64-row seed at `73.44%` min versus base `68.75%`, but 256-row confirmation was `70.70%` versus same-seed base `71.88%`. Low-LR joint v2 scored only `65.62%` min over 64 games/row while same-seed base was `71.88%`.

## [2026-06-17 13:30] Context Auxiliary Pretrain Probe
- **Changes:** Added `adaptive_search_distill.py --context-residual`, `--init-context-residual`, and `--freeze-context-strategy-aux`, plus `train_adaptive.py --init-strategy-aux` so PPO can warm start from context+strategy auxiliary checkpoints while dropping aux heads. Ran GPU probes in `runs/adaptive-context-aux-v1/` and `runs/adaptive-context-aux-ppo-v1/`.
- **Status:** Completed
- **Next Steps:** Do not promote these checkpoints. The next architecture branch should be an actual U-Net/Transformer torso or richer memory/belief-map input, since the small residual context branch still leaves 16x draw/finish bottlenecks.
- **Context:** Context aux v1 reduced intent loss from `10.25` to `4.98` and belief loss from `20.25` to `8.38` with KL only `0.017`, but direct 64-row eval was only `65.62%` min versus base `64.06%`. Aux->PPO improved a weak 64-row seed from base `60.94%` to `64.06%`, still far below the promotion platform.

## [2026-06-17 13:46] Pyramid Context Torso Probe
- **Changes:** Added optional zero-output U-Net-style pyramid branch (`--pyramid-context`, `--init-pyramid-context`) to adaptive training/evaluation/distillation, extended context-only update and context+strategy-aux freezing to include pyramid fields, and documented GPU probes under `runs/adaptive-pyramid-context-only-v1/`, `runs/adaptive-pyramid-context-joint-v2/`, and `runs/adaptive-pyramid-aux-v1/`.
- **Status:** Completed
- **Next Steps:** Do not promote pyramid add-on checkpoints. Move to a trunk replacement or explicit memory/belief input channels; the add-on branches repair some weak seeds but still shift bottlenecks.
- **Context:** Pyramid-only looked good at 64 games/row (`73.44%` vs same-seed base `67.19%`) but failed 256-row confirmation (`64.84%` vs base `67.97%`). Low-LR joint pyramid repaired weak seeds only (`67.19%` vs base `62.50%` on one 64-row seed), and pyramid aux direct matched base at `64.06%` min.

## [2026-06-17 14:31] Adaptive U-Net Trunk Bootstrap
- **Changes:** Added `AdaptiveUNetPolicyValueNetwork`, `train_adaptive.py --network-arch unet`, evaluator support for U-Net checkpoints, explicit adaptive fog-memory input planes, teacher KL anchoring, teacher-driven rollout actions, and teacher action CE for trunk replacement bootstrap. Updated README and strategy docs.
- **Status:** Completed
- **Next Steps:** Do not promote the current U-Net checkpoints. Build a dedicated offline teacher-imitation/search-to-strategy dataset so the U-Net can match the legacy CNN teacher at 256 games/row before sparse PPO fine-tuning.
- **Context:** GPU probes ran under `runs/adaptive-unet-v1*/`. EMA bootstrap v1 saved a near-random policy (`0.00%` min). Best checkpoint v1d reached `62.50%` min on 64-row smoke, but 256-row triage scored `66.41%` min versus same-seed base `69.14%`. U-Net large-map rows are close (`16p0=70.70%`, `16p1=71.09%`), but 8/12 rows remain under-cloned.

## [2026-06-17 14:47] Adaptive U-Net Teacher Imitation
- **Changes:** Added `adaptive_teacher_imitation.py`, a pure teacher-imitation trainer for adaptive trunks using teacher-driven mixed-seat rollouts, all-action KL, greedy/sampled teacher action CE, fog-memory inputs, and checkpoint pruning. Updated README and training strategy docs.
- **Status:** Completed
- **Next Steps:** Treat `runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx` as the current U-Net promotion candidate. Next run should fine-tune this checkpoint with sparse PPO/search-to-strategy auxiliary and then require 2048 games/row before replacing the legacy CNN base.
- **Context:** Sampled-action imitation v2 stayed weak (`59.38%` 64-row min). Greedy-action imitation v3 passed promotion-candidate gates: 256-row min `72.66%` vs same-seed base `69.92%`, and 512-row min `75.00%` vs base `70.12%`. Large-map draw rates improved versus base: `16p0/16p1` draw `15.04%/16.21%` vs `18.36%/18.55%`.

## [2026-06-17 15:01] Adaptive U-Net PPO v4
- **Changes:** Ran 2048-row final evidence for imitation v3 against the legacy adaptive CNN, then ran a low-LR sparse terminal PPO fine-tune from v3 under `runs/adaptive-unet-ppo-v4/`. Documented 256/512/2048 games-per-row results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Treat `runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx` as the current stronger U-Net base. Next iteration should target the slight 8p1 regression while preserving the 16x draw reduction.
- **Context:** v3 2048-row min was `73.68%` vs legacy base `70.65%`, confirming U-Net replacement quality. PPO v4 then beat v3 at 2048-row min `73.05%` vs `72.66%`, with 16x draw reduced from v3 `14.06%/14.60%` to v4 `12.16%/12.89%`.

## [2026-06-17 15:23] U-Net Top-Advantage Diagnostics
- **Changes:** Added `train_adaptive.py --top-advantage-mode {global,stratified}` so PPO can select top-advantage samples either globally or independently by effective board size and learner seat. Updated README and recorded v5/v6/v7 GPU runs in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Keep `runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx` as the active base. Do not promote v5/v6/v7. Next useful branch should change the learning signal, e.g. belief/finish heads or search-to-strategy supervision with row-wise trust control, rather than another plain low-LR continuation.
- **Context:** v5 global top-advantage failed 256-row triage (`67.97%` min vs v4 `73.83%`). v6 stratified looked strong at 256/512 rows but failed 2048-row confirmation (`70.80%` min vs same-seed v4 `72.17%`). v7 EMA stratified failed 512-row triage (`70.70%` min vs same-seed v4 `71.48%`). Training/eval used `cuda:0`; all checkpoints and JSON artifacts are under ignored `runs/`, not cache directories.

## [2026-06-17 15:58] Adaptive Fixed-v5 Gate
- **Changes:** Added fixed `PolicyValueNetwork` opponent support to `evaluate_adaptive_policy.py` and `train_adaptive.py`, plus fixed-policy teacher/opponent support in `adaptive_teacher_imitation.py`. Added `--value-head-sizes` / `--init-value-head-sizes` so single-size gates can load multi-size per-value-head checkpoints. Updated README and strategy docs with GPU v5-gate results.
- **Status:** Completed
- **Next Steps:** Do not promote the fixed-v5 imitation or finish checkpoints. Keep U-Net PPO v4 as the Expander base. Next branch should train explicit finish/draw-risk or search-to-outcome targets on v5-vs-v5 rollouts; plain PPO timeout fine-tuning did not solve the 250-step draw bottleneck.
- **Context:** U-Net PPO v4 scored only `9.57%` min over 512 games/seat against fixed 8x8 v5 sample at max250. Direct PPO vs v5 stayed at `7.62%` min. Fixed-v5 imitation improved the gate, with best final v3 at `13.48%` min and roughly `48-52%` decisive rate, but draw stayed near `70%`. At max750, imitation v1 reached `42.77%` min, showing the clone is slow rather than purely losing. All training/eval used `cuda:0`; model artifacts stayed under ignored `runs/`.

## [2026-06-17 16:14] Fixed-v5 Imitation Calibration
- **Changes:** Added outcome-weighted KL/CE support to `adaptive_teacher_imitation.py` via `--outcome-weight-mode terminal` and per-outcome action weights. Updated README and strategy docs with winner-biased and sample-teacher GPU runs.
- **Status:** Completed
- **Next Steps:** Do not continue pure action-distribution imitation as the main route. The next aligned branch should train finish/draw-risk or search-labeled finish actions on v5-vs-v5 states, because the best imitation checkpoint still fails the fixed-v5 gate.
- **Context:** Winner-biased v4 improved same-seed min only from v3 `12.70%` to `13.48%`. Sample-teacher/KL v5 produced the best artifact so far, `runs/adaptive-fixed-v5-imitation-v5/ckpts/generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx`, at `14.65%` min over 512 games/seat max250. KL-heavy v6 regressed to `12.70%` min. Draw remains the core bottleneck (`~66-70%` at max250).

## [2026-06-17 16:39] Fixed-v5 Finish Reweighting Probe
- **Changes:** Added U-Net structural warm-start expansion, outcome-head supervision in `adaptive_teacher_imitation.py`, terminal-window action weighting, and p0/p1 action sample weights. Updated README and strategy docs with GPU outcome/terminal-window/PPO probes.
- **Status:** Completed
- **Next Steps:** Stop simple outcome/window/seat reweighting for the fixed-v5 gate. Build a replacement-outcome/search dataset with accepted candidate actions, time-to-terminal deltas, draw-risk deltas, and seat labels, then train a dedicated rerank/finish correction head.
- **Context:** Outcome aux v1 learned (`Out 1.36 -> 0.98`) but 512-row min was only `12.30%`. Terminal-window v1 improved p0 to `17.97%` but collapsed p1 to `8.98%`; p1-weighted v2 regressed versus same-seed base (`12.89%` min vs base `19.14%`). KL-anchored PPO against fixed v5 scored only `9.77%` min. Current best fixed-v5 artifact remains imitation v5 iter030 at `14.65%` min over 512 games/seat max250.

## [2026-06-17 16:48] U-Net Imitation v3 Phase-0 Validation
- **Changes:** Ran a second CUDA 2048 games/row Expander evaluation for `runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx` and documented the two-seed matrix in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Use U-Net imitation v3 as the clean supervised-policy base/teacher for `adaptive_strategy_dataset.py`; keep U-Net PPO v4 as the stronger active Expander baseline until strategy supervision produces a better checkpoint.
- **Context:** v3 scored `73.68%` min on seed `78860` and `74.61%` min on seed `80720`, with 16x draw around `14-15%`. v4 still has better 12/16 rows and lower 16x draw on its existing 2048-row seed, so v3 is validated for supervised strategy work but not promoted over v4 as active Expander base.

## [2026-06-17 17:01] Adaptive Strategy Dataset v0
- **Changes:** Added `adaptive_strategy_dataset.py`, an offline shard collector for adaptive observations, teacher logits/actions, outcome/terminal labels, belief maps, weak intent, source/target heatmaps, and contact/density probes. Updated README and strategy docs with usage and v0 shard statistics.
- **Status:** Completed
- **Next Steps:** Add the frozen-trunk strategy-head trainer that consumes these shards and learns finish/draw-risk, enemy-general belief, hidden enemy maps, weak intent, source heatmap, and target heatmap without changing policy logits.
- **Context:** CUDA smoke passed for U-Net v3 teacher and fixed-v5 teacher. Initial shards were written under ignored `runs/adaptive-strategy-dataset-v0/`: U-Net v3 vs Expander has 2048 samples and 0.145 finish-within-250 mean; fixed-v5 max250 has 4160 samples, 0.661 draw-risk mean, 0.029 finish-within-250 mean, and balanced p0/p1 seats.

## [2026-06-17 17:11] Frozen Strategy Head Supervision v0
- **Changes:** Added `adaptive_strategy_supervised.py`, a frozen-base offline trainer for U-Net strategy auxiliary heads from strategy dataset shards. Updated README and strategy docs with commands, GPU smoke, v0 training curve, and policy-preservation check.
- **Status:** Completed
- **Next Steps:** Expand the strategy dataset with fixed-v5 max500/max750 and more max250 decisive/draw rows, then run policy-coupled strategy training with a KL anchor so finish/belief/intent features can influence action logits.
- **Context:** CUDA v0 training on 6208 samples reduced loss `2.3954 -> 0.3566`, intent accuracy reached `76.4%`, finish accuracy reached `89.0%`, and belief BCE fell `1.6549 -> 0.1375`. Outcome supervision was left off because the warm-start outcome head is badly calibrated on fixed-v5 draw-heavy states. Policy logit diff versus U-Net imitation v3 was exactly `0.0`, confirming no gameplay promotion is expected yet.

## [2026-06-17 17:35] Policy-Coupled Strategy Supervision
- **Changes:** Extended `adaptive_strategy_supervised.py` with `--update-scope all`, policy KL/action CE anchors, and `--max-samples-per-shard`; documented fixed-v5 max500/max750 and v4 Expander anchor shards plus coupled GPU probes in README and `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote coupled v1/v4 checkpoints. Next branch should make strategy predictions affect action selection more directly, such as finish/Q reranking, target-conditioned action bias, or replacement-outcome search heads, instead of full-trunk action-KL coupling.
- **Context:** Coupled v1 from v3 improved 512-row Expander min over same-seed v4 (`74.61%` vs `73.63%`) but weakened 16x rows and fixed-v5 stayed low (`10.55%` min at 256 max250). v4-coupled unbalanced collapsed 8p0 to `70.70%` at 256. Balanced v4-coupled preserved 16x but still trailed v4 on same-seed Expander (`73.44%` vs `73.83%` min) and fixed-v5 (`9.77%` vs `10.55%` min). All models stayed under ignored `runs/`; evaluation should be run serially on GPU to avoid allocation warnings.

## [2026-06-17 17:47] Strategy-Q Rerank Probe
- **Changes:** Added `--q-kl-weight` and `--q-action-ce-weight` to `adaptive_strategy_supervised.py` so frozen strategy-Q heads can learn teacher action distributions for inference-time reranking. Updated README and training strategy docs with Q-rerank commands and GPU scale sweeps.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-strategy-q-rerank-v1`. Next strategy-inference branch should use more structured target/finish gating or replacement-outcome search Q instead of global centered all-action reranking.
- **Context:** Q head training was successful as supervision (`QKL 5.4646 -> 0.5882`, Q action match `10.7% -> 58.9%`), but gameplay did not pass gate. Expander scale `0.05` improved 256-row min (`73.44% -> 75.00%`) but failed 512-row confirmation (`71.68%` vs scale-0 `71.88%`). Fixed-v5 max250 128-row did not improve (`10.94%` min at scale 0 and 0.05, `10.16%` at 0.10). Artifacts remain under ignored `runs/`.

## [2026-06-17 17:58] Target-Conditioned Rerank Probe
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-target-rerank-scale` and `--strategy-target-finish-gate`, using the enemy-general belief head as an inference-time movement target. The evaluator now creates `--json-output` parent directories automatically. Updated README and strategy docs with target-rerank commands and GPU sweeps.
- **Status:** Completed
- **Next Steps:** Do not promote target-rerank inference. The next useful branch should train explicit target/source heads or replacement-outcome correction targets instead of deriving a hand-coded Manhattan movement bias from belief logits.
- **Context:** Ungated target scale `0.50` looked best at 128-row Expander (`74.22%` min), but 256-row confirmation regressed against no target bias (`73.44%` vs `75.78%` min), mostly by hurting 8x while helping 16x. Finish gating did not stabilize the effect. Fixed-v5 max250 stayed far below gate (`6.25%` min at target scale `0.50` vs `5.47%` same-seed baseline).

## [2026-06-17 18:10] Explicit Source Target Strategy Heads
- **Changes:** Added optional `strategy_spatial_aux` source/target heads to adaptive CNN and U-Net networks with backward-compatible checkpoint expansion; extended `adaptive_strategy_supervised.py` to train `source_heatmap`/`target_heatmap` spatial CE losses; added `evaluate_adaptive_policy.py --strategy-spatial-rerank-scale`; documented commands/results.
- **Status:** Completed
- **Next Steps:** Do not promote `runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx`. Keep the spatial heads as diagnostics, then improve target labels or train replacement-outcome/search target heads before using source/target bias in policy selection.
- **Context:** CUDA training on 24576 offline samples learned source/target labels (`Src CE 6.19 -> 3.17`, `Tgt CE 5.18 -> 4.34`). Expander 128-row scale `0.05` looked good (`75.00%` min vs `70.31%`), but 256-row confirmation regressed (`73.05%` vs scale-0 `73.83%`). Fixed-v5 max250 also regressed (`9.38%` min vs `10.94%`) and draw did not fall.

## [2026-06-17 18:23] Adaptive Plan-Q Dataset v0
- **Changes:** Added `adaptive_plan_q_dataset.py`, which collects source-target candidate plans, forces each plan's first action, scores short counterfactual rollouts, and saves `plan_q`, `plan_scores`, `plan_outcomes`, `source_score_probs`, and `target_score_probs` shards. Documented commands and smoke results.
- **Status:** Completed
- **Next Steps:** Use this collector for longer fixed-v5 max250 shards, then train source_q/target_q from `source_score_probs` and `target_score_probs`. Do not return to spatial rerank scale sweeps.
- **Context:** CUDA smoke compiled and wrote ignored `runs/` shards. With `score_scale=10`, the 8-env/16-step/4x4-plan smoke produced 128 samples with mean `plan_q_gap=0.0689`, source max-prob mean `0.2700`, and target max-prob mean `0.2704`. The 8-step horizon still yielded all draw outcomes, so anti-draw supervision needs longer fixed-v5 plan rollouts.

## [2026-06-17 18:27] Plan-Q Source Target Supervision v0
- **Changes:** Added `adaptive_plan_q_supervised.py`, a frozen-head trainer that learns source/target spatial heads from Plan-Q shard marginals instead of static source/target CE labels. Updated README and strategy docs with usage and smoke curve.
- **Status:** Completed
- **Next Steps:** Collect longer fixed-v5 max250 Plan-Q shards and rerun this trainer with `--gap-weighting`; only then evaluate whether Plan-Q target maps can support a Worker/mixture policy.
- **Context:** CUDA smoke on `runs/adaptive-plan-q-v0-scale10/*.npz` trained 8 epochs from `adaptive-strategy-spatial-v1`. Source loss improved `3.0467 -> 2.8660` and source top1 `19.5% -> 28.1%`; target loss improved `4.3221 -> 4.1629` but target top1 stayed weak (`~2%`). The checkpoint only updates auxiliary heads and is not a promotion candidate.

## [2026-06-17 18:34] Fixed-v5 Warmed Plan-Q Probe
- **Changes:** Added `adaptive_plan_q_dataset.py --warmup-steps` and made counterfactual plan rollouts truncation-aware. Added candidate-local accuracy metrics to `adaptive_plan_q_supervised.py`. Documented fixed-v5 warm shard and gap-weighted training results.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-plan-q-fixed-v5-supervised-v1`; policy logits are unchanged and the shard contains no winning plans. Next route should strengthen plan execution/scoring with a Worker or rollout-search after the forced plan step.
- **Context:** Warmed fixed-v5 max250 shard (`warmup_steps=190`, `plan_rollout_steps=64`) produced 64 samples and 1024 plan scores: `186 loss`, `838 draw`, `0 win`, mean `plan_q_gap=0.2712`. Gap-weighted supervision fit the safety marginals (`Src loss 4.2827 -> 3.8626`, `Tgt loss 4.1558 -> 3.8177`; candidate acc around `25-28%`) but cannot train anti-draw finish behavior without winning-plan labels.

## [2026-06-17 18:40] Plan-Q Worker Probe
- **Changes:** Added `adaptive_plan_q_dataset.py --plan-worker-steps`, which keeps the initial forced source-target move and then optionally executes additional target-conditioned worker steps before returning to the base policy rollout. The worker reselects a movable owned source near the plan route and moves it toward the target, giving plan labels a multi-step execution path instead of a single-step bias.
- **Status:** Completed
- **Next Steps:** Scale worker-conditioned Plan-Q collection beyond the 64-sample probe, then train source/target/plan-Q heads on larger mixed fixed-v5 max250/max500 shards. The current supervised checkpoint only updates spatial auxiliary heads and is not a gameplay promotion candidate.
- **Context:** CUDA smoke passed on `cuda:0`. Fixed-v5 warm190 worker shard (`plan_rollout_steps=64`, `plan_worker_steps=16`) produced 64 samples and 1024 plan scores: `339 loss`, `683 draw`, `2 win`, mean `plan_q_gap=0.3770`, best-plan outcomes `4 loss / 58 draw / 2 win`. Gap-weighted head training saved `runs/adaptive-plan-q-fixed-v5-worker-supervised-v0/generals-adaptive-plan-q-fixed-v5-worker-supervised-v0.eqx` with loss `4.4931 -> 3.8817`; labels are now richer than the first-step version, but sample count remains too small for promotion.

## [2026-06-17 18:54] Plan-Q Action-Q Supervision
- **Changes:** Fixed Plan-Q pass action encoding by reusing the canonical adaptive action index, added `adaptive_plan_q_supervised.py` action-Q ranking/MSE losses, and documented the corrected worker shard plus Q-rerank probes. Action-Q supervision now aggregates duplicate plan slots onto their shared primitive action before full legal-action CE.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-plan-q-action-q-v1` or continue Q-rerank scale sweeps. Next useful route is a target-conditioned Worker/mixture policy or larger decisive Plan-Q data so the learned plan signal affects execution more structurally than centered logit bias.
- **Context:** Corrected worker v2 shard had no invalid action indices (`16..2048`), 512 states, 8192 plans, `3138 loss / 4917 draw / 137 win`, best-plan outcomes `36 loss / 419 draw / 57 win`, and mean `plan_q_gap=0.4299`. Action-Q v1 training improved `AQ rank/MSE/action-acc` from `12.9955 / 9.8694 / 9.7%` to `5.8049 / 5.3826 / 27.1%`. Fixed-v5 max250 Q-rerank did not confirm: 256-row scale `0.01` scored `10.55%` min vs scale `0` at `10.94%`.

## [2026-06-17 18:58] Plan-Q Policy Distillation Probe
- **Changes:** Added `adaptive_plan_q_supervised.py --plan-policy-weight`, which applies corrected worker Plan-Q action targets directly to policy logits under `--update-scope all` and a positive policy KL anchor. Documented the probe and negative result.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-plan-q-plan-policy-v0`. The next aligned implementation should separate plan choice from execution with a target-conditioned Worker or mixture policy instead of forcing plan actions into the primitive policy distribution.
- **Context:** Plan-policy v0 used `policy_kl_weight=1.0`, `plan_policy_weight=0.05`, `lr=2e-5`, 60 epochs on corrected worker v2. Training reduced plan-policy CE `8.8217 -> 4.9325` with KL around `0.035`, but fixed-v5 max250 128-row regressed to `6.25%` min (`p0 10.94%`, `p1 6.25%`). This mirrors earlier action-distill seat tradeoff failures.

## [2026-06-17 19:05] Strategy Worker Mixture Probe
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-worker-mix-prob`, `--strategy-worker-finish-gate`, and `--strategy-worker-policy-margin` for explicit one-step source/target worker execution from spatial strategy heads. Updated README and strategy docs with commands and results.
- **Status:** Completed
- **Next Steps:** Do not use worker mixture for promotion with current spatial heads. Next Worker work should train a target-conditioned action head from successful worker/Plan-Q trajectories instead of hand-coded source-to-target movement.
- **Context:** Fixed-v5 max250 128-row baseline for `adaptive-strategy-spatial-v1` was `11.72%` min. Worker mix regressed across all tested settings: mix `0.02` min `8.59%`, mix `0.05` min `9.38%`, mix `0.10` min `3.91%`, mix `0.20` min `3.12%`; finish-gating and policy-margin gates also stayed below baseline. The worker reduced draws mostly by increasing losses.

## [2026-06-17 19:19] Learned Worker Hybrid Probe
- **Changes:** Extended `evaluate_worker_policy.py` so hybrid fallback evaluation can load U-Net checkpoints with fog-memory and scoreboard-history inputs, fallback architecture selection, and fallback value-head template flags. Documented learned Worker rerank and mixed-target Worker results.
- **Status:** Completed
- **Next Steps:** Do not promote learned Worker rerank or keep sweeping `worker-logit-scale`. Next useful branch should train a dedicated accepted-replacement/policy-improvement gate, or collect stronger decisive Plan-Q/Worker trajectories before letting Worker logits affect the production policy.
- **Context:** General-target Worker rerank scale `0.10` looked positive at 128 games/row (`72.66%` min vs `71.09%` baseline), but failed 256-row confirmation (`70.31%` min vs `75.00%` baseline), mostly hurting 8p1. A mixed-target Worker continuation reached supervised `Src 71.1%`, `Dir 48.7%`, `Useful 70.6%`, but Expander 128-row hybrid still regressed to `69.53%` min while only improving 16p0.

## [2026-06-17 19:29] Strategy-Q Replacement Gate Probe
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-q-replace-threshold` and `--strategy-q-replace-policy-margin`, a conservative action-Q replacement gate that only swaps the policy action for the best legal strategy-Q action when predicted Q advantage clears a threshold. Updated README/manual and documented fixed-v5 gate results.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-plan-q-action-q-v1` or continue threshold sweeps. Build a true accepted-replacement/policy-improvement head from rows labeled by full replacement outcomes instead of using the current scalar action-Q head directly as the gate.
- **Context:** Fixed-v5 max250 128-row threshold `4` gave a weak signal (`9.38%` min vs `7.81%` off), but 256-row confirmation only improved p0 (`9.77% -> 13.28%`) while p1 remained the bottleneck (`8.20%`), so min stayed `8.20%`. Policy margins `2` and `4` both fell to `7.81%` min.

## [2026-06-17 19:39] Plan-Q Replacement Gate Probe
- **Changes:** Added `adaptive_plan_q_supervised.py` pairwise accepted-replacement training flags for Plan-Q shards, documented the CLI in README/zh manual, and recorded fixed-v5 max250 replacement-gate results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-plan-q-replacement-gate-v1`; next Plan-Q work should score generated source-target candidates directly or add legal non-candidate negatives before using a replacement gate.
- **Context:** The v1 gate trained offline from 41.6% to 52.8% replacement accuracy, but 256-row fixed-v5 max250 confirmation dropped min win rate from 8.20% baseline to 7.81% while lowering draw, indicating the gate converts some draws into losses rather than producing reliable wins.

## [2026-06-17 19:44] Candidate-Only Strategy-Q Gate
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-q-replace-worker-candidate`, which restricts strategy-Q replacement to the source/target worker candidate instead of all legal actions. Documented the option and fixed-v5 max250 results.
- **Status:** Completed
- **Next Steps:** Do not keep sweeping candidate gate thresholds; move toward a trained target-conditioned Worker action head or a larger accepted-plan dataset with an explicit candidate-scoring gate.
- **Context:** Candidate-only gate avoided the worst full-legal argmax failures but still did not confirm. At 256 fixed-v5 max250, baseline was `10.55%/9.77%` p0/p1 wins (`9.77%` min), while candidate threshold `1` with policy margin `4` scored `9.38%/11.72%` (`9.38%` min), again moving weakness between seats rather than increasing min win rate.

## [2026-06-17 19:59] Plan-Q Target-Conditioned Worker
- **Changes:** Added `adaptive_plan_worker_supervised.py` for offline Plan-Q best-plan Worker training, and extended `evaluate_adaptive_policy.py` with `--strategy-plan-worker-path` plus rerank-scale support. Documented commands and fixed-v5/Expander results.
- **Status:** Completed
- **Next Steps:** Do not promote v0 directly. Scale Plan-Q/Worker data with more fixed-v5 decisive states, then train mixed best/all accepted-plan Workers and add a confidence gate before stronger Worker influence.
- **Context:** Best-plan v0 trained on 403 non-pass Plan-Q examples (`Act 16.4% -> 41.2%`, `Src 30.8% -> 54.6%`, `Dir 45.2% -> 65.5%`). Fixed-v5 max250 256-row improved at low scale (`9.77%` min baseline to `10.55%` at scale `0.02`, `11.33%` at scale `0.05`), while Expander 256-row at scale `0.02` was roughly flat (`76.56%` min baseline to `76.17%`).

## [2026-06-17 20:03] Plan-Worker Data Scaling
- **Changes:** Collected four ignored fixed-v5 Plan-Q/Worker shards under `runs/adaptive-plan-q-fixed-v5-worker-v3/`, trained `adaptive-plan-worker-best-v1`, and documented the data/training/eval results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop scaling best-only Worker data. Train the next Worker on mixed best/all accepted plans and add an explicit confidence/gate label before stronger inference influence.
- **Context:** v1 used 1620 best non-pass plans and fit them much better (`Act 29.4% -> 73.8%`, `Src 44.5% -> 97.0%`, `Dir 56.2% -> 76.0%`), but fixed-v5 max250 128-row only improved min from `7.03%` to `7.81%`; p1 remains the bottleneck, so better command imitation alone is not enough.

## [2026-06-17 20:12] Plan-Worker Accepted Mixed Probe
- **Changes:** Added `adaptive_plan_worker_supervised.py --selection accepted|mixed` with accepted-plan score margins and sample weights; added `evaluate_adaptive_policy.py --strategy-plan-worker-min-margin` to gate Worker rerank by legal top1/top2 logit confidence; documented mixed and accepted-only training/eval results.
- **Status:** Completed
- **Next Steps:** Stop Plan-Worker rerank-scale and confidence-margin sweeps. Next Plan-Q work should train an explicit command/gate scorer on model-generated source-target candidates, or collect stronger decisive counterfactuals before Worker logits affect primitive inference.
- **Context:** Mixed v0 fit 2369 examples well (`Act 23.0% -> 77.2%`) but failed 256-row fixed-v5 confirmation (`10.55%` baseline min to `8.59%` at scale `0.01`, `8.98%` at scale `0.02` margin `2`). Accepted-only v0 fit 749 examples modestly (`Act 17.5% -> 44.4%`); its 128-row scale `0.02` margin `1` signal (`12.50% -> 14.84%` min) also failed 256-row confirmation (`8.98% -> 8.20%` min).

## [2026-06-17 20:21] Model Candidate Plan-Q Gate Probe
- **Changes:** Added `adaptive_plan_q_dataset.py --candidate-source model --candidate-target model` to score source/target candidates generated by the checkpoint's spatial heads instead of privileged full-state heuristics. Collected a 512-state fixed-v5 warm190 model-candidate shard, trained `adaptive-plan-q-model-candidate-gate-v0`, and documented results.
- **Status:** Completed
- **Next Steps:** Keep model-generated candidates for future data, but do not promote `adaptive-plan-q-model-candidate-gate-v0` or continue scalar action-Q threshold sweeps. Next route should use a direct binary command-acceptance head, per-seat gate calibration, or a larger model-candidate dataset before any inference gate.
- **Context:** Model candidates produced a stronger shard than heuristic v3 shard0 (`best_win=19.3%`, `best_q=0.1190`, `mean_gap=0.3931`). Gate v0 improved offline replacement accuracy only from `39.4%` to `44.3%`. Fixed-v5 max250 128-row regressed: baseline `10.94%` min, candidate threshold `1` with policy margin `4` fell to `8.59%`, and threshold `0` fell to `5.47%`.

## [2026-06-17 20:33] Binary Command Gate Probe
- **Changes:** Added standalone `adaptive_command_gate.py` and `adaptive_command_gate_supervised.py`; extended `evaluate_adaptive_policy.py` with `--strategy-command-gate-path` and threshold support; added `adaptive_plan_q_dataset.py --candidate-source model-worker` for evaluator-aligned source proposals; documented model/model-worker gate results.
- **Status:** Completed
- **Next Steps:** Do not promote command-gate checkpoints or continue threshold sweeps. Next work should improve source/target proposal quality directly, for example with outcome-supervised source/target maps or a low-rate Commander proposal head before Worker/gate execution.
- **Context:** Model-candidate gate fit offline (`53.4% -> 73.5%` balanced acc) but 128-row fixed-v5 regressed at thresholds `0.5/0.6/0.7`. Model-worker source candidates were more inference-aligned but weaker (`best_win=9.8%`, `best_q=-0.0432`); their gate fit offline (`41.2% -> 78.8%`) and had a 128-row threshold `0.7` signal (`3.91% -> 7.03%` min), but failed 256-row confirmation (`6.64% -> 5.47%` min).

## [2026-06-17 20:38] Multi Command Gate Probe
- **Changes:** Extended `evaluate_adaptive_policy.py` with `--strategy-command-gate-source-count` and `--strategy-command-gate-target-count`, allowing the command gate to score top-k source by top-k target proposals instead of only the single evaluator worker command. Documented fixed-v5 max250 results.
- **Status:** Completed
- **Next Steps:** Do not continue multi-command gate sweeps. Train the source/target proposal maps from outcome targets or add a low-rate Commander proposal head; post-hoc gate selection is not enough with the current spatial ranking.
- **Context:** On fixed-v5 max250 128-row seed `84420`, baseline min was `7.03%`. Model-candidate gate top-k settings all regressed: `0.7` with `2x2` fell to `4.69%`, `0.8` with `2x2` to `6.25%`, and `0.8` with `4x4` to `6.25%`. Draw reduction again mostly created losses, especially for p1.

## [2026-06-17 20:47] Source Target Outcome-Q Maps
- **Changes:** Added source/target Q-MSE and candidate-local Q-rank losses to `adaptive_plan_q_supervised.py`, with optional outcome blending from Plan-Q loss/draw/win labels. Documented the new CLI and GPU smoke results.
- **Status:** Completed
- **Next Steps:** Do not promote the Q-map checkpoints directly. Use outcome-Q target ranking for target proposals, but redesign source labels around executor-aware source selection, accepted-plan positives, or Worker-conditioned route/army features.
- **Context:** CUDA smoke on `runs/adaptive-plan-q-model-candidates-v0/plan-q-00000.npz` showed Q-MSE falls without ranking (`target best 26.7%`, corr `-0.107`). Sharp rank-only supervision improved target ranking from `25.7%` to `42.6%` and corr to `+0.276`, but source ranking stayed near random (`23.7%`). Fixed-v5 gameplay smoke was skipped because `/tmp/generals-ppo-8x8-expander-gpu-v5.eqx` is absent.

## [2026-06-17 20:53] Plan Pair Ranking Probe
- **Changes:** Added `adaptive_plan_q_supervised.py --plan-pair-rank-weight`, an additive source+target candidate-pair rank loss for Plan-Q shards. Ran worker-source and pair-rank GPU probes and documented results.
- **Status:** Completed
- **Next Steps:** Do not promote pair-rank checkpoints. Next route should use an explicit plan-pair/Commander scorer or target-conditioned source selector/Worker-conditioned source head; additive source+target decomposition is too weak.
- **Context:** Model-worker source rank reached `33.4%`/corr `+0.127`; pair-rank reached `15.3%` pair top1 (`S33.8%/T43.9%`, corr `+0.179`); combo reached `15.9%` pair top1 but did not materially improve source/target. All runs were CUDA and saved under ignored `runs/`.

## [2026-06-17 21:05] Explicit Plan Pair Scorer
- **Changes:** Added `adaptive_plan_pair_scorer.py` and `adaptive_plan_pair_supervised.py`, a validation-tracked explicit MLP ranker for source-target Plan-Q candidates. Ran CUDA v1/v2/small probes and documented results.
- **Status:** Completed
- **Next Steps:** Do not connect the scorer to gameplay inference yet. Collect a larger same-family `model-worker` Plan-Q shard or train a target-conditioned source selector before promotion-oriented evaluator integration.
- **Context:** Same-split additive baseline on the worker-source shard had validation pair top1 `4.19%`. Gap-weighted worker-source v1 reached best validation pair top1 `23.3%` (`S32.9%/T43.7%`, corr `+0.154`) at epoch 62. Mixed model+model-worker v2 fell to `12.9%`, and no-gap small worker-source fell to `13.7%`, so the signal is real but data-sensitive. All models/logs are under ignored `runs/`.

## [2026-06-17 21:13] High-Gap Plan Pair Scorer
- **Changes:** Added `--min-plan-gap` to `adaptive_plan_pair_supervised.py`, collected a 4-shard model-worker Plan-Q v1 dataset, and documented full/gap-filtered scorer probes.
- **Status:** Completed
- **Next Steps:** Use high-gap filtering or a decisive-plan curriculum for Commander scorer training; collect more high-gap model-worker rows before evaluator integration.
- **Context:** v1 dataset had 2048 rows, overall best_win `10.1%`, draw `86.4%`; `gap>=0.25` kept 779 rows with win `26.4%`, `gap>=0.5` kept 374 with win `46.3%`. Full scorer v3 best val pair top1 `11.3%` vs additive `9.34%`; `min_gap=0.25` reached `16.8%`, `min_gap=0.5` reached `22.2%`.

## [2026-06-17 21:16] High-Gap Plan-Q Collection Filter
- **Changes:** Added save-time `--min-plan-gap` and `--require-best-plan-win` filters to `adaptive_plan_q_dataset.py`, with metadata for pre-filter sample count and dropped rows. Updated README, zh manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Collect a high-gap model-worker dataset under `runs/adaptive-plan-q-model-worker-highgap-v0/`, then retrain the explicit pair scorer on filtered rows instead of unfiltered draw-heavy shards.
- **Context:** Filtering happens after scoring and before shard write, so it does not alter counterfactual plan labels. Empty filtered shards are skipped to avoid downstream validation split failures.

## [2026-06-17 21:23] High-Gap Model-Worker Plan-Q Probe
- **Changes:** Collected `runs/adaptive-plan-q-model-worker-highgap-v0/` with save-time `min_plan_gap=0.25`, trained highgap pair scorers, and documented aggregate/baseline results.
- **Status:** Completed
- **Next Steps:** Scale `min_gap=0.25` model-worker collection before evaluator integration; do not tighten to `0.5` until there are enough rows.
- **Context:** Highgap v0 kept 818/2048 rows with best-plan win `32.3%`, draw `67.7%`, mean gap `0.680`. Pair scorer highgap-v0 beat same-split additive val pair top1 `18.0%` vs `15.2%`; highgap-gap05-v0 used 410 rows and fell to `15.3%`, effectively additive baseline.

## [2026-06-17 21:35] Midgame Plan-Q Save Filter
- **Changes:** Added `--min-save-turn` and `--max-save-turn` to `adaptive_plan_q_dataset.py` and documented the highgap-v1 scaling results that motivated the turn filter.
- **Status:** Completed
- **Next Steps:** Collect mid/late high-gap model-worker data with `min_gap=0.25` and `min_save_turn` around 100 or 120 before training another pair scorer.
- **Context:** Highgap-v1 scaled to 3415 rows but shifted earlier than v0: median turn `96` vs `128`, and p1 best-plan win fell to `22.0%` vs v0 `36.5%`. Scorer still beat additive (`8.2%` vs `5.0%`; gap05 `11.3%` vs `7.1%`) but absolute pair top1 was too weak.

## [2026-06-17 21:42] Plan Pair Top-K Metrics
- **Changes:** Added pair@2 and pair@4 reporting to `adaptive_plan_pair_supervised.py` while keeping best checkpoint selection on validation pair@1. Updated README, zh manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Re-run scorer probes with top-k metrics before deciding whether Commander candidate selection is unusable or only needs a top-k rerank/Worker stage.
- **Context:** Mid100 high-gap data improved scorer correlation more than argmax pair@1, so top-k visibility is needed to judge whether the scorer can shortlist useful plans even when top1 remains weak.

## [2026-06-17 21:44] High-Gap Pair Top-K Recheck
- **Changes:** Recomputed top-k metrics for `adaptive-plan-pair-scorer-highgap-v0.best` and documented the result.
- **Status:** Completed
- **Next Steps:** Treat explicit pair scorer as a top-k shortlist candidate only; do not wire it as a single-plan evaluator selector.
- **Context:** Highgap-v0 validation metrics were pair@1 `18.0%`, pair@2 `23.8%`, pair@4 `33.5%`, source `30.7%`, target `38.3%`, corr `+0.127`. Top-k is better than argmax but still too weak for direct gameplay integration.

## [2026-06-17 21:52] Plan Pair Evaluator
- **Changes:** Added `adaptive_plan_pair_evaluate.py`, a standalone no-training evaluator for additive and explicit Plan-Q pair scorers. Updated README and zh manual with the diagnostic use case.
- **Status:** Completed
- **Next Steps:** Use this only for scorer sanity checks. The next promotion-oriented route is midgame decisive trajectory imitation, not online Plan-Q scorer integration.
- **Context:** The evaluator reports pair@1/pair@2/pair@4, source/target accuracy, correlation, and margin on the same split used by scorer training. It keeps top-k Plan-Q diagnostics reproducible without starting another training run.

## [2026-06-17 22:03] Midgame Strategy Dataset Filters
- **Changes:** Added current `time`, `visible_enemy_count`, and save-time row filters to `adaptive_strategy_dataset.py` for midgame/contact/terminal-window decisive trajectory shards. Updated README, zh manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Collect A2 U-Net active-base midgame decisive shards on GPU, then run policy-coupled `adaptive_strategy_supervised.py` with KL anchor plus finish/outcome/belief/intent losses.
- **Context:** CUDA smoke wrote an ignored shard under `runs/adaptive-strategy-filter-smoke/` using `teacher-kind expander` and `--min-save-turn 1`; models and future training artifacts remain under ignored `runs/`, not cache directories.

## [2026-06-17 22:10] Midgame Decisive Imitation Probe
- **Changes:** Collected ignored GPU trajectory shards under `runs/adaptive-strategy-midgame-decisive-v0/`, trained `adaptive-midgame-decisive-imitation-v0/v1/v2/v3/v2-cont`, and documented results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote these checkpoints. Add a real multi-horizon finish head or oversample true U-Net-vs-v5 winning trajectories before the next full-policy imitation run.
- **Context:** Best diagnostic checkpoint was v2: fixed-v5 max250 256-row min `12.11%`, above same-seed active base `9.38%`, but Expander 64-row min was only `70.31%`. v1/v3 64-row positives did not survive 256-row fixed-v5 confirmation. All model artifacts are in ignored `runs/`, not cache directories.

## [2026-06-17 22:17] Multi-Horizon Finish Head
- **Changes:** Added configurable `strategy_finish_outputs`, `adaptive_strategy_supervised.py --finish-head-mode multi-horizon`, evaluator `--strategy-finish-outputs`, and PPO warm-start `--init-strategy-finish-outputs`. Documented the new CLI and GPU probes.
- **Status:** Completed
- **Next Steps:** Do not promote v4/v5. Next data should be rollout-search winning trajectories or accepted Plan-Q oracle executions, because vanilla sampled winning windows are not enough.
- **Context:** Old 2-logit U-Net checkpoints can warm-start into a 3-logit finish head. v4 multi-horizon mixed data learned finish labels (`69.9%`) but fixed-v5 max250 256-row fell to `8.98%`; v5 true-v5-win-only learned finish labels (`79.6%`) but fixed-v5 64-row min was `7.81%`.

## [2026-06-17 22:28] Search Teacher Strategy Shards
- **Changes:** Added `adaptive_strategy_dataset.py --teacher-kind search`, search budget CLI flags, and fog-memory support in adaptive rollout-search helpers so 35-channel U-Net checkpoints can emit rollout-search teacher actions while preserving prior logits for KL.
- **Status:** Completed
- **Next Steps:** Collect midgame decisive shards with `min_save_turn>=80`, contact, win/finish, and terminal-window filters; start with U-Net v4 search trajectories against Expander and fixed v5.
- **Context:** CUDA smoke wrote `runs/adaptive-strategy-search-smoke/search-smoke-00000.npz` with `adaptive-unet-ppo-v4`, scoreboard history, fog memory, `top_k=2`, `rollout_steps=2`, and `rollouts_per_action=1`.

## [2026-06-17 22:52] Midgame Search Imitation Probes
- **Changes:** Added `adaptive_strategy_supervised.py --action-ce-weight-mode` so draw-heavy shards can keep KL/outcome/finish/belief/intent losses without contributing draw actions to CE/QCE. Collected search decisive shards and trained/evaluated v0-v3 under ignored `runs/`.
- **Status:** Completed
- **Next Steps:** Do not promote v0-v3. Keep v1 as the strongest diagnostic candidate, but next work should use search trajectories for value/finish/intent or collect more diverse winning windows rather than scaling action CE rows.
- **Context:** v1 was best: Expander 512-row min `74.61%` vs same-seed v4 base `70.51%`, but it missed the 75% line and fixed-v5 max250 stayed weak (`10.94%` min at 64 rows). v2 showed draw action CE filtering helps 64-row but failed 256-row; v3 showed adding more search-action rows regressed to `70.31%` 256-row min.

## [2026-06-17 23:10] KL-Anchored V1 PPO Probe
- **Changes:** Extended `train_adaptive.py` teacher loading flags so teacher KL can deserialize complex U-Net adaptive checkpoints with per-size HL-Gauss value heads, outcome head, and strategy aux finish logits. Ran v1 PPO probes under ignored `runs/adaptive-midgame-search-ppo-*`.
- **Status:** Completed
- **Next Steps:** Do not promote PPO probes. Use the teacher-loader support for future KL-anchored experiments, but avoid direct small PPO from v1 until rollout signal or microbatching is improved.
- **Context:** `48 env x 128 steps` with minibatch 256 ran but fell to Expander 256-row min `69.53%`; teacher-KL `0.1` to v1 loaded correctly but fell to 64-row min `67.19%`. Larger U-Net PPO configs hit GPU memory/compile pressure.

## [2026-06-17 23:33] High-Gap Search Trajectory Probe
- **Changes:** Added search top-k score/outcome fields and high-gap save filters to `adaptive_strategy_dataset.py`, added complex search-teacher aux flags, and added `adaptive_strategy_supervised.py --balance-strata size-seat`. Updated README, zh manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-midgame-highgap-imitation-v0/v1/v2-balanced/v3-mixed`. Use high-gap search rows for value/finish/outcome supervision or carefully weighted auxiliary replay, not dominant primitive action CE.
- **Context:** CUDA high-gap search collection from `adaptive-midgame-search-imitation-v1` produced 2294 decisive rows under ignored `runs/adaptive-strategy-search-highgap-v1/`. Direct main-policy imitation regressed Expander 64-row min versus same-seed v1 baseline `73.44%`: v0 `62.50%`, v1 `65.62%`, v2-balanced `64.06%`, v3-mixed `62.50%`, all weak on 8x8 p1.

## [2026-06-17 23:47] Search-Q Rank Supervision Probe
- **Changes:** Added `adaptive_strategy_supervised.py --search-q-rank-weight` and `--search-q-temperature` to train the strategy action-Q head from saved rollout-search top-k rankings. Updated README, zh manual, and Expander strategy notes.
- **Status:** Completed
- **Next Steps:** Do not promote q-rerank from this probe. Use high-gap search rankings for calibrated/gated Q, finish/value/intent heads, or broader mixed datasets before any full-policy update.
- **Context:** CUDA run `runs/adaptive-midgame-search-q-rank-v0/` reduced SQ loss `3.8067 -> 2.7584` but top1 stayed weak. Expander 256-row seed85280 scale=0 was `71.09%`; q-rerank scale `0.001` was `71.48%`, while scale `0.01` collapsed 64-row min to `54.69%` on 8p1. A low-lr full-policy decisive imitation recheck with KL/action CE/finish/outcome/belief/intent still collapsed 64-row 8p1 to `56.25%`.

## [2026-06-19 17:26] Search-Best Outcome Head Probe
- **Changes:** Added `adaptive_strategy_supervised.py --label-source search-best`, `--balance-finish-labels`, and `--balance-outcome-labels` so high-gap search shards can train frozen finish/outcome heads from local rollout-search best-action outcomes. Updated README, zh manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Do not use this checkpoint for promotion yet. Next attempt should either unfreeze a small value/strategy bottleneck under policy freeze/KL, or collect broader search-best labels with negative and draw-heavy non-winning rows before using finish gates in gameplay.
- **Context:** Default sandbox hid GPU devices, but escalated checks confirmed RTX 5070 Ti and JAX `cuda:0`; balanced GPU run `adaptive-search-best-outcome-head-gpu-v0` reached finish loss `0.7009`, outcome loss `1.1157`, outcome accuracy `38.3%`. `search_best_outcome` win labels in highgap-v1 are rare (`19%-35%` by stratum). Unbalanced v0 reached finish accuracy `72.0%`, matching draw-majority behavior; balanced CPU v1 reached similar loss and passed a 2-logit finish evaluator load smoke.

## [2026-06-19 19:02] Search-Best Value Bottleneck Probe
- **Changes:** Added `adaptive_strategy_supervised.py --update-scope strategy-value-heads`, which keeps trunk/policy/action logits frozen while updating strategy heads, optional outcome head, and `value_linear1`. Documented the GPU bottleneck probes and broader contact high-gap dataset.
- **Status:** Completed
- **Next Steps:** Treat `adaptive-search-best-bottleneck-gpu-v1` as a calibrated finish/outcome representation probe, not a promotion checkpoint. Next collect fixed-v5 max250 search-best contact rows or wire these signals into a gated Commander/finish probe.
- **Context:** GPU run `adaptive-search-best-bottleneck-gpu-v1` mixed high-gap decisive rows with 4909 broader contact rows and improved finish accuracy to `62.6%` and outcome accuracy to `59.8%` under search-best labels, versus the frozen-head GPU probe's outcome accuracy `38.3%`. Primitive policy logits remain unchanged.

## [2026-06-19 19:13] Fixed-V5 Search-Best Contact Data
- **Changes:** Collected GPU fixed-v5 max250 contact/high-gap search-best shards under `runs/adaptive-strategy-search-fixed-v5-contact-highgap-v0/`, trained `adaptive-search-best-bottleneck-fixed-v5-v0`, and documented the results.
- **Status:** Completed
- **Next Steps:** Use the fixed-v5 search-best rows for a gate/candidate model or recollect with longer search rollout horizon. Do not treat the bottleneck checkpoint as promotion because primitive policy logits are unchanged.
- **Context:** The old best fixed-v5 imitation v5 checkpoint does not deserialize under the current U-Net value-head template, so collection used loadable `adaptive-midgame-search-imitation-v1` as search prior against fixed v5. The new dataset kept 7525 rows with search-best labels `946` win / `6578` draw / `1` loss; fixed-v5 bottleneck training reached finish accuracy `57.8%`, outcome accuracy `58.8%`, and passed a GPU evaluator load smoke.
## [2026-06-19 19:26] Fixed-v5 Search-Q Value Probe
- **Changes:** Added `adaptive_strategy_supervised.py --search-q-value-weight` plus score-scale/tie-break options to regress strategy action-Q values to search candidate outcomes; added a focused unit test and updated README/manual/training log docs. Ran GPU fixed-v5 max250 probes for `adaptive-fixed-v5-search-q-value-v0`.
- **Status:** Completed
- **Next Steps:** Stop Q-rerank/Q-replace threshold scans; collect and train Midgame Decisive Trajectory Imitation windows with contact/high-gap/win-or-finish filters.
- **Context:** Q-value loss dropped strongly (`14.0065 -> 0.3222`), but q-replace underperformed baseline at 128 games/seat (`min 9.38%` off vs `5.47%` at threshold `0.25` and `7.81%` at threshold `1.0`). The issue is action-level replacement, not trainability of the head.

## [2026-06-19 19:36] Midgame Decisive Imitation Probe
- **Changes:** Collected GPU fixed-v5 max250 decisive trajectory windows and draw-heavy contrast windows under `runs/`, trained `adaptive-midgame-decisive-imitation-v0/v1/v2`, and documented the results.
- **Status:** Completed
- **Next Steps:** Do not tune v0-v2 loss weights further. Collect larger first-contact-to-terminal/search-success trajectories and restrict primitive action CE to rows where the search rollout actually wins over multiple steps.
- **Context:** Strict decisive shard kept 683 rows; draw-heavy contrast kept 3753 rows. Against fixed-v5 max250 at 128 games/seat seed `86360`, base min was `11.72%`; v0 `10.94%`, v1 `7.81%`, v2 `10.16%`. The wiring is fast on GPU, but one-step terminal-window imitation is not yet enough to encode the gather/attack/finish chain.

## [2026-06-19 19:48] Search-Best-Win Action Gate
- **Changes:** Added `adaptive_strategy_supervised.py --action-ce-weight-mode search-best-win`, which trains primitive action CE only on rows where local rollout-search best outcome is win while retaining all rows for KL and auxiliary losses. Collected `adaptive-midgame-contact-searchwin-fixed-v5-v0`, trained `adaptive-midgame-contact-searchwin-imitation-v3`, and documented GPU results.
- **Status:** Completed
- **Next Steps:** Scale the same midgame/contact/high-gap data recipe 3-5x and mix 12/16 Expander contact rows before another full-policy update; do not return to terminal-window-only imitation.
- **Context:** New dataset kept 16315 rows with trajectory loss/draw/win `5365/8347/2603` and search-best win rows `2151` (`13.18%`). Fixed-v5 max250 256-row min improved from `9.77%` to `11.33%`; Expander 8/12/16 128-row min moved from `75.00%` to `75.78%`.

## [2026-06-19 20:18] Scaled Decisive Imitation Probes
- **Changes:** Collected/derived ignored GPU datasets under `runs/adaptive-midgame-contact-searchwin-fixed-v5-v1/`, `runs/adaptive-midgame-contact-searchwin-expander-12-16-v0/`, and `runs/adaptive-midgame-contact-searchwin-decisive-filter-v0/`; trained/evaluated v4-v8 variants; documented results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop action-CE imitation sweeps on these shards. Use the decisive filtered data for a plan-conditioned Worker or gated plan executor, mixed with the base policy rather than globally biasing primitive logits.
- **Context:** v4 reached fixed-v5 max250 256-row min `11.72%` but regressed Expander min to `73.44%`; v5-lite `9.38%`, v6 `8.59%`, v7 `11.33%` with Expander `72.66%`, and v8 decisive-only `10.94%`. Current safe checkpoint remains `runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx`.

## [2026-06-19 20:27] Strategy Worker Command Probe
- **Changes:** Added `adaptive_plan_worker_supervised.py --dataset-format strategy` for decisive strategy shards, added `evaluate_adaptive_policy.py --strategy-plan-worker-command-source belief-main-stack`, updated README/manual/docs, and trained/evaluated `runs/adaptive-strategy-decisive-worker-v1/`.
- **Status:** Completed
- **Next Steps:** Do not scan worker rerank scale or margin further. Train a gate/scorer on whether the Worker action improves the base action under rollout, then invoke the Worker only when that gate is confident.
- **Context:** Corrected Worker loader expands the 4-direction legal mask to full/half 8 action planes before filtering labels. Worker v1 trained on `15941` decisive rows and reached action/source/direction accuracy `29.3%/67.7%/41.6%`, but fixed-v5 max250 128-row seed `87060` did not improve min: off `8.59%`, scale `0.02` `8.59%`, scale `0.05` `5.47%`, scale `0.02` margin `1.0` `7.81%`.

## [2026-06-19 20:48] Strategy Worker Gate Probe
- **Changes:** Added `adaptive_command_gate_supervised.py --dataset-format strategy-worker` to train a binary gate from strategy shards and a learned Plan-Worker, added `evaluate_adaptive_policy.py --strategy-plan-worker-gate-*` inference support, updated README/manual/docs, and trained/evaluated `runs/adaptive-strategy-worker-gate-v1/`.
- **Status:** Completed
- **Next Steps:** Stop proxy worker-gate and threshold scans; return to direct Midgame Decisive Trajectory Imitation so decisive search/win signals update the U-Net main policy instead of acting as inference-time overrides.
- **Context:** GPU training used safe v3 + decisive Worker on fixed-v5 v0/v1 shards: 44,789 rows, 17,311 worker-change examples, 5.52% positives, final gate P+/P- `0.561/0.439`. Fixed-v5 max250 seed `87060`: no gate min `8.59%`, gate `0.55` min `6.25%`, gate `0.65` min `7.81%`; not promotion-worthy.

## [2026-06-19 21:11] Safe-v3 Domain-Balanced Imitation
- **Changes:** Collected safe-v3 rollout-search fixed-v5 rows under `runs/adaptive-midgame-contact-searchwin-fixed-v5-safev3-v0/`, collected safe-v3 Expander protection rows under `runs/adaptive-midgame-contact-searchwin-expander-safev3-v0/`, added `adaptive_strategy_supervised.py --balance-strata size-seat-domain`, trained/evaluated v0/v1/v2, and updated README/manual/strategy docs.
- **Status:** Completed
- **Next Steps:** Do not promote v0/v1/v2 or blindly sweep CE/KL/domain ratio. Next iteration should target fixed-v5 p0 specifically while keeping domain-balanced Expander protection, or test a smaller policy-head/delta update instead of full U-Net trunk updates.
- **Context:** Fixed-v5-only v0 improved fixed-v5 128-row min `7.03% -> 9.38%` but regressed Expander min `73.44% -> 66.41%`. Mixed v1 recovered Expander to `69.53%` but fixed-v5 min was only `7.81%`. Domain-balanced v2 recovered Expander further to `71.09%` but fixed-v5 p0 fell, min `6.25%`. Current safe checkpoint remains v3.

## [2026-06-19 21:28] Policy-Head Decisive Imitation Probe
- **Changes:** Added `adaptive_strategy_supervised.py --update-scope policy-heads`, which freezes the trunk/value heads while updating `policy_conv`, `pass_linear`, strategy auxiliary heads, and optional outcome head. Documented the mode in README/manual and logged GPU policy-head runs.
- **Status:** Completed
- **Next Steps:** Keep safe-v3 as the active base. Treat `policyhead-fixed-v0` as a diagnostic only; next useful direction is a gated/adapter-style policy delta for fixed-v5-like midgame/contact states or cleaner p0 decisive trajectories, not stronger global CE/KL sweeps.
- **Context:** Mixed-domain policy-head v0 reached fixed-v5 max250 128-row min `7.81%` and Expander min `71.88%`. Fixed-v5-only policyhead v0 reached matched 256-row fixed-v5 min `12.11%` vs safe-v3 `10.55%`, with 128-row Expander min `72.66%`. Stronger lr `1e-5` policyhead-fixed-v1 regressed fixed-v5 256-row min to `8.98%`, so pushing the output head harder is not reliable.

## [2026-06-19 21:38] p0-Focused Policy-Head Mix Probe
- **Changes:** Derived ignored p0-only fixed-v5 safe-v3 shards under `runs/adaptive-midgame-contact-searchwin-fixed-v5-safev3-p0-v0/`, trained/evaluated unbalanced `policyhead-p0mix-v0` and domain-balanced `policyhead-p0domain-v0`, and documented the GPU results.
- **Status:** Completed
- **Next Steps:** Stop static p0/domain ratio sweeps. Implement a conditional gate or adapter/delta policy head that only activates in fixed-v5-like midgame/contact finish states.
- **Context:** p0-only fixed-v5 data kept `30561` rows with `4170` search-win rows. Unbalanced p0mix-v0 improved fixed-v5 max250 256-row min to `12.50%` but dropped Expander min to `70.31%`. Domain-balanced p0domain-v0 protected Expander slightly better at `71.09%` but fixed-v5 min was only `10.94%`. Active base remains safe-v3.

## [2026-06-19 21:47] Finish-Gated Policy Adapter Probe
- **Changes:** Added `evaluate_adaptive_policy.py --policy-adapter-path/--policy-adapter-scale/--policy-adapter-finish-threshold`, loading a second adaptive checkpoint as a centered legal-logit delta with optional hard finish gating. Documented README/manual usage and GPU p0mix adapter results.
- **Status:** Completed
- **Next Steps:** Train a learned adapter gate from rollout replacement outcomes using finish/outcome, policy-delta, seat, and contact features. Do not sweep static thresholds as the main path.
- **Context:** `scale=1.0` no-gate reproduced p0mix fixed-v5 behavior (`12.50%` min), validating the delta implementation. Hard finish-gated p0mix adapter reached fixed-v5 max250 256-row min `12.11%` and Expander 128-row min `72.66%`, recovering most Expander loss from full p0mix (`70.31%`) but still below promotion.

## [2026-06-19 22:03] Learned Policy Adapter Gate Probe
- **Changes:** Added `adaptive_policy_adapter_gate_supervised.py` and evaluator support for `--policy-adapter-gate-path/--policy-adapter-gate-threshold`; documented the learned-gate usage and GPU probe results.
- **Status:** Completed
- **Next Steps:** Do not sweep learned gate thresholds. Train a stronger policy-delta adapter or collect richer decisive adapter data before reusing the gate machinery.
- **Context:** v0 trained on only 175 changed-action examples with 4 positives; fixed-v5 max250 256-row held `12.11%` min but Expander 128-row fell to `69.53%`. v1 added unchanged-action negatives (`53,922` examples, still only 4 positives); threshold `0.1` fell to fixed-v5 min `9.38%`. The current p0mix adapter is too sparse a delta source for learned gating.

## [2026-06-19 22:29] Domain-Filtered Action CE Probe
- **Changes:** Added `adaptive_strategy_supervised.py --action-ce-path-contains` so primitive action CE can be restricted to selected shard path tokens while all shards still contribute KL/aux losses. Documented the flag and GPU results for domain-filtered policy-head, full-trunk, and terminal/search-win probes.
- **Status:** Completed
- **Next Steps:** Stop global primitive CE imitation on these fixed-v5/search-win shards. Reuse the new terminal/search-win data for finish/outcome calibration, plan/target Q, or a plan-conditioned Worker that is validated before it is mixed into the base policy.
- **Context:** GPU was freed by stopping a stale `play_web.py` process occupying ~12GB. Domain-filtered p0 policy-head reached fixed-v5 128-row min `6.25%` and Expander 64-row min `70.31%`; p0 full-trunk reached fixed-v5 `10.16%` but Expander `65.62%`; two-seat full-trunk reached fixed-v5 `10.16%` but Expander `64.06%`; terminal/search-win full-trunk reached fixed-v5 `8.59%` and Expander `62.50%`. The filter works, but primitive action CE remains the wrong insertion point.

## [2026-06-19 22:41] Winning-Trajectory Worker Probe
- **Changes:** Ran GPU non-primitive probes: binary search-best calibrator, aux-only KL-anchored trunk training, terminal/search-win Worker, and true search-controlled winning-trajectory Worker. Logged the results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop Worker scale scans with the current `belief-main-stack` command source. Use the true winning-trajectory data to train command/source-target selection or Plan-Q counterfactual gates, then invoke the Worker only when command confidence is high.
- **Context:** Binary calibrator was weak (`56.7%` finish acc); aux-only trunk still regressed Expander 8p1 (`64.06%` min). Terminal Worker learned offline (`38.8%` action acc) but hurt fixed-v5 p1. New true winning trajectories kept `25,385` rows; the Worker was stable offline (`37.8%` action acc) and safe but no-op in gameplay: fixed-v5 max250 min stayed `7.81%` at scale `0.02` and `0.05`.

## [2026-06-19 23:15] Winning-Trajectory Worker Gate Guard
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-plan-worker-max-grid-size` to disable Plan-Worker rerank/replacement on out-of-domain board sizes, updated README/zh manual/strategy docs, and trained/evaluated `runs/adaptive-searchwin-trajectory-worker-gate-v0/v1/`.
- **Status:** Completed
- **Next Steps:** Do not promote the gate yet. Collect 12/16-compatible winning-trajectory Worker/gate data or train explicit replacement outcome/Q labels before using the Worker beyond 8x8.
- **Context:** Sandbox JAX evaluator could not reliably see CUDA and fell back slowly; GPU evaluator runs used elevated access and reported `Device: cuda:0`. Gate v1 with non-decisive positives improved fixed-v5 max250 256-row min from same-seed base `8.59%` to `11.33%`; `--strategy-plan-worker-max-grid-size 8` protected Expander 12/16 and gave 64-row Expander min `67.19%` vs same-seed base `65.62%`.

## [2026-06-19 23:57] Grid-Range Worker Gate Filters
- **Changes:** Added decisive-row filters to `adaptive_plan_worker_supervised.py` and `adaptive_command_gate_supervised.py`, added active-area gate features with sidecar feature-dim compatibility, and added `evaluate_adaptive_policy.py --strategy-plan-worker-min-grid-size` so Plan-Worker rerank/replacement can be enabled only on a board-size range. Updated README, Chinese manual, and strategy logs with GPU results.
- **Status:** Completed
- **Next Steps:** Do not promote the protect gate as a general policy. Use grid-range guards for diagnostics, then move to high-gap midgame decisive trajectory imitation or source/target outcome-Q supervision.
- **Context:** Protect-v3 with `min_grid_size=12,max_grid_size=16` preserved 8x8 base behavior and gave small 12/16 same-seed Expander gains: 256-row seed `90000` stayed at base min `72.27%` while 12p1 improved `79.30% -> 80.08%`, 16p0 `77.34% -> 77.73%`, and 16p1 `75.00% -> 76.17%`. The six-row min is still 8p0-limited, so this is a control-surface improvement rather than a promotion.

## [2026-06-20 00:09] Training-Time Strategy Row Filters
- **Changes:** Added load-time row filters to `adaptive_strategy_supervised.py` for turn/contact/visible-enemy/outcome/search-best/finish/search-gap selection, documented the flags, and trained/evaluated `runs/adaptive-midgame-decisive-policyhead-v0/` on GPU.
- **Status:** Completed
- **Next Steps:** Do not promote `policyhead-v0` or sweep its CE/LR. Keep `adaptive-midgame-contact-searchwin-imitation-v3` as the active base and use the new filters for source/target outcome-Q or executor data selection.
- **Context:** `policyhead-v0` used broad fixed-v5/search-win, draw contrast, and Expander protection rows with `update_scope=policy-heads`, KL `3.0`, CE `0.20`, and `search-best-win` CE gating. It regressed same-seed fixed-v5 128-row min from base `10.94%` to `8.59%` and Expander 128-row min from `75.78%` to `74.22%`, so policy-head-only decisive imitation is ruled out for promotion.

## [2026-06-20 00:19] High-Gap Plan-Q Gate Probe
- **Changes:** Added `adaptive_plan_q_supervised.py --strategy-finish-outputs/--init-strategy-finish-outputs` for multi-horizon strategy checkpoints, documented the flags, and logged GPU training/eval for `adaptive-plan-q-source-target-highgap-mid100-v0` plus `adaptive-command-gate-highgap-mid100-v0`.
- **Status:** Completed
- **Next Steps:** Stop this inference-time source/target replacement route. Use midgame decisive trajectories to update the U-Net main policy/representation, or train a plan-conditioned executor only after command proposal quality is stronger.
- **Context:** Source/target Q proposal stayed weak offline (`pair top1 7.0%`, pair corr `+0.021`). The gate fit replacement labels moderately (`23,753` examples, `8.42%` positives, final weighted acc `69.2%`), but fixed-v5 max250 seed `86640` collapsed from gate-off min `10.94%` to gate-on min `0.00%`.

## [2026-06-20 00:24] Midgame Decisive Representation Probe
- **Changes:** Trained/evaluated ignored GPU run `runs/adaptive-midgame-decisive-repr-v0/` and documented the result in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not sweep LR/epochs on this mixed filtered data. Next useful step is stronger data selection: isolate base-draw/search-win or true search-controlled winning windows with clearer finish labels before another main-policy update.
- **Context:** `repr-v0` used `update_scope=all`, KL `10.0`, CE `0.05`, finish/outcome/belief/intent losses, and high-gap midgame/contact rows. It regressed fixed-v5 max250 128-row seed `86640` from base min `10.94%` to `7.81%`; Expander smoke 64-row seed `86680` min was `71.88%`.

## [2026-06-20 00:28] Draw/Search-Win Contrast Filter Probe
- **Changes:** Added `adaptive_strategy_supervised.py --require-outcome-draw/--require-outcome-nonwin`, documented the filters, extended the loader test, and trained/evaluated ignored GPU run `runs/adaptive-midgame-drawsearch-repr-v0/`.
- **Status:** Completed
- **Next Steps:** Do not sweep the small balanced contrast run. Collect more draw/search-win rows or mix the contrast rows as a weighted component with larger decisive trajectory data while protecting Expander with KL/protection rows.
- **Context:** Existing shards contain useful contrast (`2,234` draw/search-win rows after high-gap/contact filters), but `size-seat-domain` balancing reduced training to `312` samples. `drawsearch-repr-v0` barely learned the all-win search label (`outcome acc 8.2%`) and fixed-v5 max250 seed `86640` regressed from base min `10.94%` to `9.38%`.

## [2026-06-20 00:33] Draw/Search-Win Oversample Probe
- **Changes:** Added `adaptive_strategy_supervised.py --balance-strata size-seat-oversample/size-seat-domain-oversample`, documented the modes, extended the loader/balancer test, and trained/evaluated ignored GPU run `runs/adaptive-midgame-drawsearch-oversample-repr-v0/`.
- **Status:** Completed
- **Next Steps:** Use draw/search-win contrast rows as frozen-policy auxiliary/gating data or as a small weighted slice inside larger successful-trajectory updates; do not continue all-trunk policy updates on this standalone objective.
- **Context:** Oversampling retained contrast signal (`2,234` filtered rows -> `5,520` balanced samples) and improved finish/outcome training (`72.2%`/`77.7%`), but fixed-v5 max250 seed `86640` regressed to min `8.59%` vs base `10.94%`. Active base remains safe v3.

## [2026-06-20 00:45] Policy Adapter Feature-Gate Split
- **Changes:** Added optional policy-adapter feature model plumbing to gate training/evaluation, appended draw/win probabilities to adapter-gate features with sidecar-compatible slicing, documented the flags, and trained/evaluated ignored GPU run `runs/adaptive-policy-adapter-gate-feature-v0/`.
- **Status:** Completed
- **Next Steps:** Do not continue the p0mix adapter-gate branch unless a future adapter produces many more changed-action positives. The next useful route should train a better adapter/delta source or use contrast rows only for frozen auxiliary calibration.
- **Context:** Feature-finish gating with draw/search model improved p0 but hurt p1 (`14.84%/9.38%`, min `9.38%`). Learned 14-feature gate had only ~`0.01%` positives (`187` changed actions out of `51,835` rows) and evaluated at fixed-v5 max250 seed `86640` with min `10.94%`, effectively base behavior.

## [2026-06-20 01:00] Midgame Search-Win Trajectory Policy Probe
- **Changes:** Trained/evaluated ignored GPU runs `runs/adaptive-midgame-searchwin-trajectory-imitation-v0/` and `runs/adaptive-policy-adapter-gate-searchwin-traj-v0/`, then documented the results in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop direct all-trunk trajectory imitation and adapter gating from this data. Use search-win trajectories for finish/value calibration or move to controlled PPO from safe v3, where rollout outcome supplies the credit signal.
- **Context:** Direct all-trunk trajectory imitation learned aux labels but regressed fixed-v5 max250 to 128-row min `7.81%` and Expander 128-row min `68.75%`. Using that checkpoint only as an adapter produced more changed actions than p0mix (`2,033/112,492`) but only `0.13%` positives; learned gate threshold `0.5`/`0.1` both gave fixed-v5 max250 256-row seed `86640` min `10.55%`, below safe-v3 historical `10.94%-11.33%`.

## [2026-06-20 01:10] Controlled Fixed-v5 PPO Probe
- **Changes:** Added explicit `train_adaptive.py` output flags for preserving strategy aux heads through PPO (`--strategy-aux`, `--strategy-spatial-aux`, `--strategy-finish-outputs`), updated README/manual docs, and trained/evaluated ignored GPU runs `runs/adaptive-ppo-fixed-v5-controlled-v0/` and `runs/adaptive-ppo-fixed-v5-controlled-v1/`.
- **Status:** Completed
- **Next Steps:** Do not promote v1. Keep safe v3 as active base; use v0 only as a controlled-PPO diagnostic. Next PPO should preserve strategy aux heads and include Expander protection in the rollout curriculum or stop after the first safe fixed-v5 probe.
- **Context:** v0 used mixed-seat sparse terminal PPO vs fixed v5 with 256-step rollout, EMA, stratified top-advantage 0.25, and safe-v3 KL. It reached fixed-v5 max250 256-row seed `86640` min `11.33%` and Expander 128-row seed `86680` min `75.78%`, roughly safe-v3 level. v1 continued 60 iters but stayed fixed-v5 min `11.33%` and slipped Expander to min `75.00%`.

## [2026-06-20 01:48] Mixed Opponent Adaptive PPO
- **Changes:** Updated `examples/_experimental/ppo/train_adaptive.py` so multi-size adaptive PPO can mix a fixed-size policy opponent into matching-size rows while falling back to the configured heuristic opponent for other rows. Added `--opponent-policy-grid-size` and `--opponent-policy-mix-prob`; non-matching rows receive a benign fixed-size dummy observation before the frozen opponent is queried.
- **Status:** Completed
- **Next Steps:** Do not promote `runs/adaptive-ppo-mixed-opponent-smoke-v0` or `runs/adaptive-ppo-mixed-opponent-fixed8-v0`; both kept fixed-v5 max250 min at 11.33%. Use the mixed-opponent support for future richer curricula, but the next strength push should return to decisive trajectory/finish supervision rather than longer plain PPO.
- **Context:** GPU smoke passed. p=0.5 Expander 128-row min was 75.78% and fixed-v5 max250 256-row min was 11.33%; p=1.0 Expander 128-row min was 75.00% and fixed-v5 max250 256-row min was 11.33%. `controlled-v0` checkpoint schema is U-Net + per-size HL-Gauss + outcome head, no strategy heads; `safe-v3` has older strategy-head schema incompatible with the current strategy head shape.

## [2026-06-20 02:05] Offline Value Calibration Probe
- **Changes:** Added `adaptive_strategy_supervised.py --value-target-weight` to fit PPO value heads to selected trajectory/search-best loss-draw-win labels (`-1/0/+1`), including HL-Gauss value CE and frozen-base gradient masking for value heads. Updated README and `docs/expander-training-strategy.md`; trained/evaluated ignored GPU runs `adaptive-valuecal-trajectory-v0`, `adaptive-ppo-valuecal-mixed-v0`, and the corrected full-schema `adaptive-ppo-mixed-opponent-safev3-full-v0`.
- **Status:** Completed
- **Next Steps:** Do not promote the value-calibrated checkpoints. Stop longer plain PPO/value-only pretraining for this gate; next work should target execution/control directly with source-target outcome-Q, stronger plan-conditioned Worker positives, or advantage-labeled action selection from counterfactual plans.
- **Context:** Full safe-v3 schema loads when outcome and strategy heads are both declared; the earlier schema mismatch was from omitting the outcome head. Value calibration learned offline (`value loss 5.2276 -> 2.9189`, MAE `0.673 -> 0.609`) but value-calibrated mixed PPO still scored fixed-v5 max250 256-row min `11.33%` and Expander 128-row min `74.22%`. The first oversampled calibration attempt OOMed at 230,424 rows; the completed run used `45,866` rows.

## [2026-06-20 02:33] Fixed-v5 Short-Gate Follow-up
- **Changes:** Documented GPU results for direct midgame trajectory imitation, policy-head-only search-best action imitation, high-gap Plan-Q Worker rerank, 8x8 fixed-v5 best-response PPO, last-iterate PPO, and draw-penalty PPO in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Stop primitive action CE and short 8-only fixed-v5 PPO for this gate. Move next to cleaner counterfactual execution labels: advantage-labeled changed actions, command-gated executor positives, or source-target outcome-Q where positives are tied to later wins rather than single-step teacher choices.
- **Context:** Full-trunk decisive trajectory imitation hurt Expander (`65.62%` min). Policy-head variants preserved Expander but fixed-v5 gains were 128-row false positives (`v2` 256-row min `8.20%`, `v3` 256-row min `7.03%`). Current search-win shards already store search-best actions as `teacher_action_index` in ~`96.7%`-`99.98%` of valid rows. 8-only PPO with EMA, last iterate, and draw penalty all stayed below the safe-v3 fixed-v5 gate; draw penalty did not reduce draw enough.

## [2026-06-20 12:44] Fixed-v5 Search Strategy Teacher
- **Changes:** Added `adaptive_strategy_dataset.py --teacher-kind fixed-search` so an 8x8 fixed `PolicyValueNetwork` teacher can produce rollout-search top-k labels in adaptive padded action space. Updated README, Chinese manual, and the strategy log with GPU A1 fixed-v5 rollout-search collection/training results.
- **Status:** Completed
- **Next Steps:** Do not promote the A1 policy-head checkpoints or sweep action CE. Use fixed-search data for finish/value/command labels, command-gated executor positives, or advantage-labeled changed actions; revisit the legacy fixed-v5 imitation checkpoint only if a loader adapter is worth the time.
- **Context:** A1 collection wrote `2,168` high-gap/search-win rows under ignored `runs/adaptive-fixed-v5-searchwin-a1-v1/`. Best smoke was base-KL policy-head v2 at fixed-v5 max250 128-row min `12.50%`, but Expander 128-row min regressed to `70.31%`. Starting from `adaptive-fixed-v5-imitation-v5` iter 30 failed on an old value-head shape mismatch.

## [2026-06-20 13:04] Fixed-Search Mixed Label Probes
- **Changes:** Added `adaptive_strategy_supervised.py --label-source search-best-or-trajectory` with a loader test, documented the flag, and logged GPU value/PPO/Q probes from fixed-search A1 data.
- **Status:** Completed
- **Next Steps:** Do not continue valuecal->PPO or current strategy-Q replacement on the A1 fixed-search shard. Next collect changed-action counterfactual rows where the base action draws/loses and fixed-search replacement later wins, then train a gate/executor on that changed-action outcome.
- **Context:** Mixed-label value calibration learned offline only modestly (`finish 56.1%`, outcome `62.5%`, value CE `3.3793`). The 192-env PPO variant OOMed during compile; the 96-env run completed but fixed-v5 max250 stayed at `10.94%` min and Expander 128-row min was `74.22%`. The A1 Q probe kept weak search-Q accuracy (`26.5%`) and conservative Q replacement fell to fixed-v5 min `9.38%`.

## [2026-06-20 13:15] Changed-Action Plan-Q Gate Probe
- **Changes:** Added `adaptive_plan_q_dataset.py --strategy-finish-outputs` so Plan-Q collection can load multi-horizon strategy checkpoints, added a parser regression test, documented the collector flag, and logged the fixed-v5 changed-action command-gate GPU probe.
- **Status:** Completed
- **Next Steps:** Do not promote or threshold-sweep the command-gate route. Move to a plan-conditioned executor or longer executed-command trajectory collection with explicit finish pressure.
- **Context:** The safe-v3 policy checkpoint has no spatial source/target heads, so model-candidate collection used the policy-preserving spatial checkpoint `adaptive-plan-q-source-target-highgap-mid100-v0`. The resulting gate fit a low-positive dataset (`7805` rows, `4.09%` positives), but fixed-v5 max250 collapsed from gate-off min `10.94%` to gated min `0.00%` with `98.44%` draw on both seats.

## [2026-06-20 13:32] Executed-Prefix Plan Worker
- **Changes:** Added `adaptive_plan_q_dataset.py --save-worker-prefix-steps` to save best-command executed Worker prefixes, added `adaptive_plan_worker_supervised.py --dataset-format plan-q-prefix`, covered the loader with a synthetic test, and documented GPU prefix Worker probes.
- **Status:** Completed
- **Next Steps:** Keep the prefix infrastructure but stop expanding oracle-only prefix data. Next collect inference-matched model-worker/model or belief-main-stack prefixes whose executed plans later win or reduce draw, then mix a small oracle slice as regularization.
- **Context:** Small oracle-prefix v0 had only `8` winning states / `96` prefix labels but weakly improved fixed-v5 max250 seed `93900` min from `4.69%` to `9.38%` and held seed `93700` at `10.94%`; Expander smoke with Worker limited to 8x8 improved 8p0 and left larger rows unchanged. Larger oracle-prefix v1 (`186` states / `2195` labels, offline action `98.6%`) regressed fixed-v5 to `7.03%` and `9.38%` min, indicating oracle command-distribution mismatch.

## [2026-06-20 13:40] Belief-Matched Prefix Worker
- **Changes:** Added inference-matched Plan-Q candidate modes: `--candidate-target belief` uses `enemy_general_logits`, and `--candidate-source main-stack` uses army mass plus route distance with no source-head prior. Added parser coverage and documented the GPU belief-prefix probe.
- **Status:** Completed
- **Next Steps:** Stop Worker-rerank scale/data sweeps. Use prefix data for main-policy/finish gating or a command policy that learns when to enter and exit a plan.
- **Context:** Belief/main-stack collection produced `186` winning states and `2176` non-pass prefix labels with mean gap `0.732`; the Worker fit offline (`97.8%` action/useful), but fixed-v5 max250 stayed weak: seed `93900` min `7.81%` vs baseline `4.69%`, and seed `93700` regressed to `9.38%` vs baseline `10.94%`.

## [2026-06-20 13:57] Prefix Main-Policy Supervision
- **Changes:** Added per-prefix base-policy logits to `adaptive_plan_q_dataset.py --save-worker-prefix-steps`, and added `adaptive_strategy_supervised.py --dataset-format plan-q-prefix` so executed best-command prefixes can train the main policy with saved-logit KL plus small action CE. Updated README and the strategy log with the new prefix main-policy route and GPU probe results.
- **Status:** Completed
- **Next Steps:** Do not promote the single-source prefix CE checkpoints. Next mix prefix rows with Expander/adaptive preservation rows or add per-size/seat protected loss so fixed-v5 anti-draw gains do not transfer as 12/16 regressions.
- **Context:** GPU collection `adaptive-plan-q-prefix-logits-belief-win-v0` produced `194` high-gap best-plan-win states and `2328` non-pass prefix labels with saved `2049`-action logits. The CE/KL prefix probe improved fixed-v5 max250 128-row min from `8.59%` to `13.28%` and lowered draw, but Expander 8/12/16 64-row smoke fell from `57.81%` min to `51.56%`. A conservative CE run preserved p0 poorly (`8.59%` min), so the useful signal is real but needs mixed-batch preservation before larger eval.

## [2026-06-20 14:31] Size-Gated Midgame Adapter
- **Changes:** Added Plan-Q collector init-template flags for old checkpoint schemas, added pure policy-adapter training with `adaptive_strategy_supervised.py --no-strategy-aux`, added evaluator main-model init-template flags and policy-adapter min/max grid-size gates, and documented GPU results in README/strategy notes.
- **Status:** Completed
- **Next Steps:** Use `v4 base + midgame-v3 adapter max-grid-size 8` as the current diagnostic wrapper, but do not call it a fixed-v5 breakthrough. Collect more genuine midgame decisive trajectories and train per-size adapters before mixing them into the base.
- **Context:** GPU v4 heuristic Plan-Q prefix data produced `432` best-plan-win states and `5051` non-pass rows, but the pure v4 prefix adapter was negative at fixed-v5 max250 256-row (`base min 8.98%`, scale `1.0` min `7.81%`, scale `0.25` min `8.20%`). Reusing `adaptive-midgame-contact-searchwin-imitation-v3` as an 8x8-only adapter on v4 improved fixed-v5 max250 512-row from same-seed expanded-v4 `10.16%` min to `11.52%` and held Expander 512-row min at `73.05%`. GPU evals should run serially; parallel JAX evals triggered CUDA allocation warnings on the 16GB card.

## [2026-06-20 14:46] Legacy Fixed-v5 Adapter Loader
- **Changes:** Added explicit legacy checkpoint loading with `--drop-mismatched-init-leaves`, added `evaluate_adaptive_policy.py --policy-adapter-mode delta|blend|replace`, documented the flags, and logged GPU results for using `adaptive-fixed-v5-imitation-v5` iter 30 as an 8x8 adapter on v4.
- **Status:** Completed
- **Next Steps:** Treat `v4 + legacy fixed-v5 iter30 adapter max-grid-size 8` as the current Expander diagnostic wrapper. Do not keep sweeping adapter scales; the next fixed-v5 improvement still needs decisive trajectory and finishability supervision.
- **Context:** Legacy iter30 direct load validated at fixed-v5 max250 128-row min `16.41%`. Same-seed 512-row Expander gate improved from v4 base `71.88%` min to `78.91%` min by lifting 8p0/8p1 to `88.48%/89.26%` while 12/16 rows stayed unchanged. Same-seed fixed-v5 max250 improved from base `8.79%` min to `11.91%` min but remained high-draw, so this is not a fixed-v5 short-gate breakthrough.

## [2026-06-20 15:06] Legacy Plan-Q Prefix Adapter
- **Changes:** Exposed `--drop-mismatched-init-leaves` in `adaptive_strategy_supervised.py`, `adaptive_plan_q_dataset.py`, and `train_adaptive.py`, added `train_adaptive.py --teacher-drop-mismatched-init-leaves`, and documented GPU probes for legacy search-win CE, legacy fixed-v5 PPO, old-checkpoint selection, and legacy Plan-Q executed-prefix imitation.
- **Status:** Completed
- **Next Steps:** Scale `adaptive-plan-q-legacy-mainstack-fixedv5` collection with the same filters (`turn>=80`, best-plan-win, `plan_q_gap>=0.25`) and train a v1 policy-head prefix adapter. Do not continue fixed-v5 PPO or search-best action CE from legacy.
- **Context:** Search-win CE from legacy underperformed same-seed legacy (`13.48%` vs `14.84%` fixed-v5 max250 512-row min), and both sparse PPO variants collapsed to `0.00%` min. The small legacy Plan-Q prefix dataset kept `42` high-gap winning plan states / `312` non-pass prefix rows and produced `adaptive-legacy-planq-prefix-policy-v0`, which improved same-seed fixed-v5 max250 512-row min from legacy `12.70%` to `15.04%` while keeping Expander 8/12/16 512-row min at `79.69%`.

## [2026-06-20 15:28] Plan-Q Prefix Margin Filter
- **Changes:** Added `adaptive_strategy_supervised.py --min-teacher-action-logit-margin` for `--dataset-format plan-q-prefix`, documented the flag in README, and logged GPU Plan-Q prefix scaling plus midgame trajectory follow-up results.
- **Status:** Completed
- **Next Steps:** Do not promote v1/v2 prefix adapters or the A1 trajectory adapters. Move the decisive data into finish/value/command gating or a plan-conditioned executor with an explicit enter/exit decision.
- **Context:** Larger legacy prefix v1 data (`150` states / `1177` prefix steps) did not beat legacy same-seed fixed-v5 (`11.91%` vs `14.26%` min at 512 rows). Margin-filtered v2 trained much cleaner offline (`~75%` action accuracy) but still missed same-seed legacy (`13.28%` vs `13.87%` min). A1/terminal/rescue trajectory adapters showed only small fixed-v5 signal (`a1mix-v0` 256-row min `10.94%` vs v3 same-seed `7.42%`) and are not promotion candidates.

## [2026-06-20 15:41] Deployment-Shaped Adapter Gate
- **Changes:** Added base-model schema flags and legacy leaf dropping to `adaptive_policy_adapter_gate_supervised.py`, documented the new gate trainer usage, and logged GPU results for a learned gate on `a1mix-v0`.
- **Status:** Completed
- **Next Steps:** Do not threshold-sweep the a1mix gate. Use the result to justify stronger enter/exit-plan labels or better candidate behavior; current best remains the legacy Plan-Q prefix wrapper.
- **Context:** The A1+terminal gate data had `3890` rows, `179` changed actions, and `52` positives (`29.05%` of changed-action examples), avoiding the old sparse-gate failure. The learned gate improved same-seed fixed-v5 max250 256-row min from a1mix ungated `10.55%` to `12.11%`, but 512-row confirmation was only `10.94%` min and same-seed legacy Plan-Q v0 was stronger at `14.06%`.

## [2026-06-20 15:48] Policy Adapter Commit Probe
- **Changes:** Added `evaluate_adaptive_policy.py --policy-adapter-commit-steps` so gated policy adapters can be forced for a short fixed horizon after a learned gate or finish-threshold trigger. Updated README, Chinese manual, and strategy notes with the diagnostic behavior and GPU result.
- **Status:** Completed
- **Next Steps:** Do not sweep commit length or run a 512-row confirmation for this path. Continue with stronger decisive trajectory labels or a true plan-conditioned executor instead of forcing a weak adapter decision for multiple turns.
- **Context:** GPU smoke used `cuda:0`. `v4 base + a1mix adapter + a1mix learned gate threshold 0.5 + commit4` scored fixed-v5 max250 256-row seed `97120` at p0 `8.20%`, p1 `9.77%`, min `8.20%`, below the non-commit gate (`12.11%`) and current legacy Plan-Q wrapper (`14.06%`).

## [2026-06-20 16:00] Legacy Plan-Q Adapter Gate Probe
- **Changes:** Logged the GPU offline probe for a deployment-shaped gate over the current best `adaptive-legacy-planq-prefix-policy-v0` adapter, using `adaptive-midgame-contact-searchwin-imitation-v3` only as the feature model. No code changes were needed.
- **Status:** Completed
- **Next Steps:** Do not run gameplay eval or threshold sweeps for this legacy gate. Keep the ungated v4 + legacy Plan-Q prefix adapter as current best, and move fixed-v5 work toward better enter/exit labels or a real plan-conditioned executor.
- **Context:** The gate data had `3890` rows, `943` changed adapter actions, and `275` positives (`29.16%`). The old value-head schema required loading with a shared value-head template. Training failed to separate labels: `lr=1e-3` ended with P+ `0.872` / P- `0.871`, and `lr=1e-4` ended with loss `0.7035`, acc `46.4%`, P+ `0.458` / P- `0.465`.

## [2026-06-20 16:11] Wrapper-Aligned Search Data
- **Changes:** Added adapter-aware search teacher support to `adaptive_search_distill.py` and `adaptive_strategy_dataset.py`, including teacher adapter path/scale/mode/size-gate flags and mismatched legacy leaf loading. Updated README, Chinese manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Do not promote `adaptive-wrapper-searchwin-policy-v0/v1/v2`. Use the new collector for larger wrapper-aligned data, but move the target toward finish/outcome/belief or multi-step executor labels rather than tiny one-step search-best CE.
- **Context:** GPU collection `adaptive-wrapper-searchwin-v0` used `v4 + legacy Plan-Q adapter` against fixed-v5 and kept `436` turn>=80/contact/search-win/high-gap rows (`p0=250`, `p1=186`, mean gap `1038.24`). v0 from v4 init scored fixed-v5 256-row min `9.77%`. v1 from legacy init improved 256-row seed `97400` to `14.45%` vs legacy `12.11%`, but 512-row seed `97420` was `11.52%` vs legacy `11.72%`. v2 balanced was `12.11%` at 256-row.

## [2026-06-20 16:46] Counterfactual Prefix Adapter Max500
- **Changes:** Added base-continuation scoring, plan time-to-terminal, plan advantage, and prefix advantage fields to `adaptive_plan_q_dataset.py`; added Plan-Q prefix advantage weighting, step decay, and accepted-vs-base pairwise margin loss to `adaptive_strategy_supervised.py`; added adapter trigger/used/action-diff diagnostics to `evaluate_adaptive_policy.py`.
- **Status:** Completed
- **Next Steps:** Scale max500 accepted-prefix collection to hundreds of accepted states before another adapter promotion attempt; use fixed-v5 `max_steps=500` for future gates instead of max250, with 256-row triage and 512-row promotion.
- **Context:** GPU smoke passed. Strict max500 causal collection with `best_plan_win && plan_improves_base && gap>=0.25` produced high-quality labels but low volume: 11 accepted states in the first shard group, then 54 more across follow-up shards. Small adapters trained from these rows did not promote: same-seed fixed-v5 max500 256-row min was legacy v0 `42.97%`, v2-small `37.50%`, v3 `41.41%`, and v3-strong `30.47%`. The `max500` gate is materially different from max250: current legacy wrapper reached 128-row min `39.84%` and 256-row min `42.97%` with low draw, so future thresholds should be recalibrated around max500.

## [2026-06-20 17:03] Max500 Prefix Seat Balancing
- **Changes:** Added `adaptive_plan_q_dataset.py --learner-seat mixed|p0|p1` so accepted-prefix collection can force one learner seat; documented the flag in `README.md`; logged the max500 seat-balanced scaling run in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote the new v4-c6balanced adapters. Keep `v4 base + legacy Plan-Q v0 adapter` as current best, and use the max500 512-row baseline (`38.09%` min at seed `98800`) as the promotion reference. Convert the accepted-prefix data into enter/exit-plan or finish/outcome labels instead of further raw prefix-action imitation.
- **Context:** GPU collection with 6x6 main-stack/heuristic candidates produced `759` accepted max500 plan states / `8993` prefix rows after adding p0-only collection (`p0=462`, `p1=297`, mean advantage `1.026`, median best-plan terminal `24`). Direct policy-head continuation still regressed fixed-v5 max500: v4-c6balanced scored `35.16%` min at 256 games/seat, and the conservative variant scored `36.33%`, both below same-seed legacy v0 `42.97%`. A fresh 512-row current-best baseline scored p0 `38.09%`, p1 `41.41%`, min `38.09%`.

## [2026-06-20 17:20] Prefix Policy Adapter Gate
- **Changes:** Extended `adaptive_policy_adapter_gate_supervised.py` with `--dataset-format plan-q-prefix` and prefix-specific filters, and extended `evaluate_adaptive_policy.py` policy-adapter gate features with normalized time, scoreboard advantage, and contact. Updated README and strategy notes.
- **Status:** Completed
- **Next Steps:** Do not promote or threshold-sweep `adaptive-prefix-policy-adapter-gate-max500-v0/v1-phase`. Next build an executor that can actually produce accepted-prefix actions, then gate that executor with the new phase/contact features.
- **Context:** Prefix gate training on max500 accepted-prefix shards produced `3085` changed-action examples with `21.72%` positives. Without phase/contact features, the gate triggered 100% even at threshold `0.8`, reproducing ungated v0. With phase/contact features, offline separation improved to `72.3%` acc (`P+ 0.604`, `P- 0.388`) and max500 256-row trigger dropped to `80%–87%`, but fixed-v5 min was only `39.06%`, below same-seed ungated v0 `42.97%`.

## [2026-06-20 17:34] Max500 Plan-Worker Command Source
- **Changes:** Added `evaluate_adaptive_policy.py --strategy-plan-worker-command-source main-stack-heuristic`, relaxed Plan-Worker aux requirements for rerank-only heuristic commands, and documented the new command source in `README.md`, `docs/zh-manual.md`, and `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Keep fixed-v5 diagnostics on max500. Do not promote the max500 Plan-Worker rerank; current best deployment remains v4 base plus legacy Plan-Q prefix policy v0 adapter.
- **Context:** Trained `runs/adaptive-plan-worker-prefix-max500-mainstack-heuristic-v0/` on 8958 winning max500 prefix rows. Same-seed 256-row wrapper+Worker moved min 36.72% -> 39.45%, but 512-row seed98800 fell below the existing wrapper baseline: 36.52% vs 38.09% min.

## [2026-06-20 17:54] Online Counterfactual Search Wrapper
- **Changes:** Added fixed-opponent online rollout search to `evaluate_adaptive_policy.py` via `--online-search-*` flags. The evaluator now composes the base policy plus optional policy adapter, scores top-k primitive actions by short counterfactual rollouts against `--opponent-policy-path`, and executes the best scored action. Updated README, Chinese manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Run a 512-row confirmation for the exact `top_k=4, rollout_steps=16, turn>=80, contact-only` wrapper. Then collect online-search traces for DAgger-style distillation with search value/margin/enter-state labels. Add heuristic/Expander opponent support for 12/16 upper-bound diagnostics.
- **Context:** GPU smoke passed. Against fixed-v5 max500, current wrapper same-seed 256-row min was `38.28%`; online search lifted it to `50.39%` with p0 `53.52%`, p1 `50.39%`, and draw below `7%` on both seats. This is the strongest fixed-v5 signal so far, but it is a planner/wrapper result, not a pure `.eqx` checkpoint.

## [2026-06-20 18:20] Online Search Opponent First-Step Alignment
- **Changes:** Fixed `evaluate_adaptive_policy.py` online search so candidate scoring reuses the exact fixed-opponent first action that the real environment step will execute, instead of sampling a different opponent first action inside search. Updated README, Chinese manual, and strategy notes with aligned max500 evidence.
- **Status:** Completed
- **Next Steps:** Collect aligned online-search traces for distillation: chosen action, candidate prior, search score, margin, search-enter state, and final episode outcome. Then add equivalent heuristic/Expander online-search support for 12/16 planner upper-bound checks.
- **Context:** Same-seed fixed-v5 max500 512-row no-search baseline was p0 `44.14%`, p1 `41.99%`, min `41.99%`. Before alignment, online search min was only `44.73%`. After alignment, online search reached p0 `53.52%`, p1 `51.56%`, min `51.56%`, with draw `6.25%` / `4.69%`.

## [2026-06-20 18:47] Expander Online Search Upper Bound
- **Changes:** Extended `evaluate_adaptive_policy.py --online-search-*` to built-in heuristic opponents, including Expander, using the same first-step opponent-action alignment as the fixed checkpoint path. Updated README, Chinese manual, and strategy notes with 8/12/16 Expander results and the legacy adapter `--value-head-sizes 8` loading note.
- **Status:** Completed
- **Next Steps:** Build an aligned online-search trace dataset for distillation rather than sweeping search parameters. Save base action, search action, candidate scores, score margin, search-enter flag, and final outcome; train policy/action-Q/finish heads on high-margin states.
- **Context:** Expander 8/12/16 max750 same-seed 256-row baseline min was `76.95%` with 16x draw `14.45%` on both seats. Online search `top_k=4, rollout_steps=16, turn>=80, contact-only, max_grid=16` reached min `85.16%`, with all six rows above `85%` and 16x draw down to `5.47%` / `6.25%`. This is a planner/wrapper upper bound, not a pure checkpoint.

## [2026-06-20 19:11] Online Search Trace Dataset
- **Changes:** Added `examples/_experimental/ppo/adaptive_online_search_trace_dataset.py` to collect aligned deployment online-search traces into ignored `runs/` NPZ shards. The collector supports fixed checkpoint and heuristic opponents, warmup before saving, size/turn/contact gates, policy adapters, top-k candidate scores, score margins, weak belief/intent labels, and strategy-compatible core fields. Updated README, Chinese manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Add `dataset-format online-search` training support in `adaptive_strategy_supervised.py`: KL-to-base plus top-k soft CE/search-Q/margin losses, filtering on `search_used`, `search_action_changed`, `search_score_gap`, turn/contact, and balanced size/seat strata.
- **Context:** GPU smoke wrote finite NPZ trace fields. First real Expander shard saved `57/96` midgame rows across 8/12/16 with action-change rate `64.9%`, mean gap `9.19`, and `7` short-horizon win labels. First fixed-v5 max500 shard saved `88/96` 8x rows with balanced seats, action-change rate `56.8%`, and mean gap `3.55`; short rollouts still labeled all best outcomes draw, so fixed-v5 distillation should not filter only on `search_best_outcome=win`.

## [2026-06-20 19:23] Online Search Distillation Trainer
- **Changes:** Added `adaptive_strategy_supervised.py --dataset-format online-search`, row filters for `search_used` / `search_action_changed`, action CE modes `search-used` / `search-changed`, and pairwise margin support against `base_action_index`. Updated README, Chinese manual, and strategy notes so current fixed-v5 diagnostics and data collection use `max500` instead of `max250`, with `max750` reserved for longer confirmation.
- **Status:** Completed
- **Next Steps:** Collect larger fixed-v5 max500 online-search trace shards with later-turn/contact states and final long-episode outcomes, then run pure checkpoint distillation and evaluate wrapper-free against fixed-v5 max500 plus Expander 8/12/16 max750.
- **Context:** GPU trainer smoke consumed the first Expander/fixed-v5 trace shards, kept `51/145` rows after online-search filters, trained one policy-head epoch, and saved an ignored smoke checkpoint under `runs/adaptive-online-search-distill-smoke/`. Tiny fixed-v5 max500 eval was only `32` games/seat and min `15.62%`, so it validates the code path only, not promotion quality.

## [2026-06-20 19:41] Online Search Policy Rank
- **Changes:** Added `adaptive_strategy_supervised.py --search-policy-rank-weight` to train main policy logits from online-search top-k soft score rankings. Added `evaluate_adaptive_policy.py --policy-adapter-min-turn` and `--policy-adapter-require-contact` so policy adapters can be contained to midgame/contact states. Documented the new flags and results in README, Chinese manual, and the strategy log.
- **Status:** Completed
- **Next Steps:** Stop trying to promote static action-label distill checkpoints from the current trace shards. The next data upgrade should save final long-episode conversion labels after following the wrapper/search action, then train gate/adapter decisions on draw->win conversion rather than raw short-horizon search labels.
- **Context:** New fixed-v5 max500 trace v1h8 saved `569` rows with balanced seats, `62.0%` action-change rate, and `105` short-horizon win best-action labels. Expander v1h8 saved `624` rows across 8/12/16 with `190` win labels. Mixed distill v1 improved fixed-v5 128-row min from same-seed v4 `17.97%` to `21.09%`, but dropped Expander min from `76.56%` to `71.88%`. Fixed-v5-only context-gated adapter changed about `10%` of decisions but scored `23.44%` min versus same-seed v4 `25.78%`; not promoted.

## [2026-06-20 19:53] Max500 Online-Search Conversion Labels
- **Changes:** Added max500 continuation labels to `adaptive_online_search_trace_dataset.py`, including base/search continuation outcome, score, time, score delta, improvement, and draw/loss-to-win conversion flags. Added continuation row filters and wired `adaptive_strategy_supervised.py` to use `search-continuation` label sources plus conversion-aware action CE modes. Updated README, Chinese manual, and strategy log.
- **Status:** Completed
- **Next Steps:** Collect a real fixed-v5 max500 conversion shard with `--truncation 500 --conversion-rollout-steps 500`, then train an 8x policy-head adapter using `--label-source search-continuation` and conversion-positive action CE if enough positives appear.
- **Context:** CUDA smoke passed for both short conversion (`conversion_rollout_steps=32`) and full max500 conversion (`conversion_rollout_steps=500`) on 2-env shards. Models/artifacts remain under ignored `runs/`; no trained model was written to cache.

## [2026-06-20 20:10] Max500 Conversion Adapter v1
- **Changes:** Fixed `adaptive_strategy_supervised.py` to mask action CE and pairwise prefix weights when the teacher action points to an invalid `teacher_logits <= -9999` entry. Collected fixed-v5 max500 conversion shards, trained `runs/adaptive-online-search-conversion-adapter-v1/generals-adaptive-online-search-conversion-adapter-v1.eqx`, and appended detailed results to `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Promote v1 as the current static fixed-v5 max500 adapter candidate, then collect more max500 conversion rows and train a stronger v2 with more legal conversion positives or a learned gate/mixture. Keep Expander protection at 512-row minimum before any deployment claim.
- **Context:** Combined max500 conversion data had `1156` rows, `117` improves, and `90` search-converts-to-win rows (`89` legal after masking). Fixed-v5 max500 512-row same-seed baseline was `23.05%` min; v1 adapter reached `38.09%` min. Expander 8/12/16 max750 512-row with the adapter reached `78.32%` min. Artifacts remain under ignored `runs/`, not cache directories.

## [2026-06-20 20:51] V1-Policy Conversion DAgger v2
- **Changes:** Collected v1-policy fixed-v5 max500 conversion traces under `runs/adaptive-online-search-fixed-v5-max500-conversion-v1policy-v0/`, trained `runs/adaptive-online-search-conversion-adapter-v2/generals-adaptive-online-search-conversion-adapter-v2.eqx`, and logged the negative result in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote v2. Keep v1 as the current static adapter candidate. Next route should be a learned gate/mixture or preservation-heavy objective that only uses v1-policy conversion labels in high-confidence states.
- **Context:** v1-policy data still had `805` rows with `59` legal search-converts-to-win rows, so online search continues to find improvements on top of v1. Naive continuation from v1 did not help: fixed-v5 max500 256-row same seed dropped from v1 `38.67%` min to v2 `36.33%` min. This points to over-broad static replacement, not lack of conversion labels.

## [2026-06-20 21:10] Max500 Online-Search Adapter Gate
- **Changes:** Added `online-search` support to `adaptive_policy_adapter_gate_supervised.py`, including conversion/improvement positive fields and adapter-vs-search-action matching. Added independent policy-adapter feature-model schema flags in `evaluate_adaptive_policy.py` so a pure v4 base/adapter can use a strategy-aux feature model for learned gate features. Preserved counterfactual adapter continuation fields in `adaptive_online_search_trace_dataset.py`.
- **Status:** Completed
- **Next Steps:** Do not promote the learned gate yet. Keep `runs/adaptive-online-search-conversion-adapter-v1/generals-adaptive-online-search-conversion-adapter-v1.eqx` as the current max500 static adapter. The next data step should increase adapter-matching conversion positives or train a calibrated gate from direct adapter-vs-base continuation labels.
- **Context:** Conversion-only gate v0 had only `6/1162` strict positives and triggered `100%` even at thresholds `0.8` and `0.95`, making it equivalent to static v1. Broader `search_improves_continuation` gate had `9/1132` positives and triggered about `68%`, but fixed-v5 max500 128-row same seed min was `38.28%`, below static v1's `40.62%`. The max500 switch remains correct; learned gating is not yet the mechanism to improve on v1.

## [2026-06-20 21:11] Max500 Gate Documentation
- **Changes:** Updated `README.md`, `docs/zh-manual.md`, and `docs/expander-training-strategy.md` with the new online-search gate dataset format, independent feature-model schema flags, and the negative max500 learned-gate result.
- **Status:** Completed
- **Next Steps:** Future gate work should use direct adapter continuation labels (`adapter_improves_continuation` / `adapter_converts_to_win`) or a larger adapter-matching conversion set before running another 512-row gate.
- **Context:** Documentation now reflects that fixed-v5 diagnostics stay on `max500`, static conversion adapter v1 remains current best, and learned gate v0/improve-v0 are diagnostic artifacts under ignored `runs/`.

## [2026-06-20 21:16] Max500 Adapter-Causal Gate Smoke
- **Changes:** Collected a direct adapter-vs-base max500 counterfactual shard under `runs/adaptive-online-search-fixed-v5-max500-adaptercf-smoke/` and trained low/high-LR adapter-causal gates under `runs/adaptive-policy-adapter-gate-online-max500-adaptercf-v0/` and `v1/`. No model artifacts were placed in cache directories.
- **Status:** Completed
- **Next Steps:** Do not promote the adapter-causal gate. Keep static v1 as the current max500 adapter candidate, and use adapter-vs-base continuation labels for policy/adapter training or richer gate features rather than threshold sweeps.
- **Context:** The adapter counterfactual shard had strong labels (`894` rows, `adapter_improves=32.2%`, `adapter_converts=16.9%`), but the current 18 gate features separated weakly. Low-LR gate v1 at threshold `0.65` triggered only `13%`-`15%` of moves and scored fixed-v5 max500 128-row min `28.91%`, below same-seed static v1 `40.62%`. The label source is promising; the current gate feature set is not.

## [2026-06-20 21:25] Max500 Adapter-Causal Gate v2
- **Changes:** Let the larger adapter-vs-base max500 counterfactual collection finish its first shard under `runs/adaptive-online-search-fixed-v5-max500-adaptercf-v0/` and trained `runs/adaptive-policy-adapter-gate-online-max500-adaptercf-v2/generals-adaptive-policy-adapter-gate-online-max500-adaptercf-v2.eqx`.
- **Status:** Completed
- **Next Steps:** Stop learned-gate work on the current 18-feature adapter gate. Use the adapter-vs-base continuation data for policy/adapter training, or add richer state/phase/route features before revisiting gating.
- **Context:** The real shard had `3707` rows with `adapter_improves=25.1%`, `adapter_converts=19.3%`, and `adapter_action_changed=43.2%`. v2 trained to offline acc `75.4%` with P+ `0.654` and P- `0.375`, but gameplay still failed: fixed-v5 max500 128-row min was `28.12%` at threshold `0.5` and `26.56%` at threshold `0.6`, both far below same-seed static v1 `40.62%`.

## [2026-06-20 21:34] Max500 Finish Labels
- **Changes:** Added `finish_within_500` dataset output plus 500-step finish filters/positives across strategy supervision, Plan-Worker training, command gates, and policy-adapter gates; updated README, Chinese manual, and strategy log guidance.
- **Status:** Completed
- **Next Steps:** Use `--truncation 500`, `--conversion-rollout-steps 500`, and `--finish-target-horizon 500` for current fixed-v5 max500 shards; keep max750 for longer confirmation.
- **Context:** The old `finish_within_250` path remains available for historical shards and compressed ablations. Targeted py_compile and CLI help checks passed; help commands still print the expected sandbox CUDA warning before exiting successfully.

## [2026-06-20 22:01] Max500 Online Search Teacher
- **Changes:** Collected `runs/adaptive-online-search-fixed-v5-max500-v1search-v2/`, trained diagnostic `runs/adaptive-online-search-conversion-adapter-v3-v1search/generals-adaptive-online-search-conversion-adapter-v3-v1search.eqx`, and appended the fixed-v5/Expander results to `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Do not promote v3. Treat `v4 base + static conversion adapter v1 + online search top_k=4 rollout=16 contact-only turn>=80` as the current fixed-v5 max500 teacher/deployment wrapper; the next compression attempt needs richer gating or a planner-aware conditional head rather than direct policy-head CE.
- **Context:** Static v1 plus online search reached fixed-v5 max500 256-row min `51.17%` with draw below `7%`. Expander 8/12/16 64-row smoke min was `79.69%`, with 12/16 rows at `81.25%`-`90.62%`. The new v1-search shard had `3491` rows and `260` balanced search-converts-to-win positives, but v3 static compression dropped to fixed-v5 max500 128-row min `34.38%`, below static v1 same-seed `40.62%`.

## [2026-06-20 22:11] Max500 Adapter-Continuation Sharpening
- **Changes:** Added `adaptive_strategy_supervised.py --seat-loss-multipliers` for per-seat supervised action/pairwise weighting, preserved the online-search min-score-gap evaluation option, and documented the max500 adapter-continuation v3-v6 results in README, Chinese manual, and the strategy log.
- **Status:** Completed
- **Next Steps:** Do not promote v3/v4/v5/v6. Keep static conversion adapter v1 as the current pure adapter and keep the online-search wrapper as the max500 teacher. Next compression should use a planner-aware conditional head or richer enter/exit controller, not another global replace-policy CE pass.
- **Context:** Combined adapter-vs-base max500 data had `5283` rows with `521` changed&improve and `385` changed&convert rows. Best sharpened adapter v4 reached fixed-v5 max500 2048-row `1555/4096` wins versus static v1 `1498/4096`, but min only moved `35.99% -> 36.57%` and p1 regressed `37.16% -> 36.57%`. p1-weighted v5 and static-init v6 failed 256-row triage.

## [2026-06-20 22:12] Online Search Score-Gap Gate
- **Changes:** Added `evaluate_adaptive_policy.py --online-search-min-score-gap`, which falls back to the original policy action when the online-search best-vs-second rollout score gap is below a threshold; documented the fixed-v5 max500 gap-gate result.
- **Status:** Completed
- **Next Steps:** Do not use score-gap gating for promotion. Keep default `--online-search-min-score-gap 0.0`; richer gate/controller features are still needed.
- **Context:** With `v4 + static conversion adapter v1 + online search top_k=4 rollout=16`, fixed-v5 max500 128-row seed `101060` default min was `50.78%`. Gap `0.5` dropped min to `47.66%`; gap `2.0` dropped min to `46.09%`. The gate improved p1 but hurt p0, so simple score-gap confidence is not a reliable enter/exit mechanism.

## [2026-06-20 22:29] Max500 Horizon Default
- **Changes:** Changed `adaptive_strategy_supervised.py --finish-target-horizon` default from `250` to `500` and updated README, Chinese manual, and strategy notes to make fixed-v5 `max500` the primary gate. `max250` remains available only by explicit flag for historical compressed ablations.
- **Status:** Completed
- **Next Steps:** Run future fixed-v5 data collection, supervised distillation, and promotion gates at `max500`; use `max750` or longer only for confirmation/ablation after a candidate clears max500.
- **Context:** This aligns the CLI default with the current fixed-v5 evidence and avoids accidentally training new finish heads on old `finish_within_250` labels. Old shards without `finish_within_500` still fall back to `finish_within_250` for compatibility.

## [2026-06-20 22:31] Higher-Budget Online Search Teacher
- **Changes:** Evaluated `v4 base + static conversion adapter v1 + online search top_k=4 rollout_steps=16 rollouts/action=2`, saved results under `runs/adaptive-online-search-conversion-adapter-v1/`, and documented the fixed-v5/Expander smoke in README, Chinese manual, and the strategy log.
- **Status:** Completed
- **Next Steps:** Treat rollouts/action `2` as the current high-budget teacher candidate. Collect rpa2 traces/continuations next, then train a planner-aware conditional head or controller; do not resume direct full-replace action CE as the main route.
- **Context:** Fixed-v5 max500 128-row seed `101060` improved from rpa1 min `50.78%` to rpa2 min `60.16%` (`65.62%` p0, `60.16%` p1). Expander 8/12/16 32-row smoke min was `81.25%`, with 8/12 rows above `90%` and 16x p0 `96.88%`.

## [2026-06-20 22:40] Max500 Default Horizon Cleanup
- **Changes:** Changed single-size `behavior_clone.py`, `evaluate_policy.py`, and `evaluate_heuristics.py` defaults from `250` to `500`; added `evaluate_adaptive_policy.py --eval-batch-size` to split expensive max500/max750 rows into smaller JAX evaluation batches; updated README and Chinese manual.
- **Status:** Completed
- **Next Steps:** Use max500 as the default fixed-v5 diagnostic horizon and pass `--eval-batch-size 64` or similar for expensive online-search promotion checks.
- **Context:** `py_compile` passed for the touched scripts. A tiny CUDA smoke with `num_games=4`, `eval_batch_size=2`, and `max_steps=5` ran on `cuda:0` and wrote `runs/eval-batch-smoke.json` with `eval_batch_size: 2`; the first smoke attempt only failed because the legacy v4 checkpoint needed `--drop-mismatched-init-leaves`.

## [2026-06-20 22:47] Chunked rpa2 Teacher Confirmation
- **Changes:** Ran a correct-schema chunked GPU fixed-v5 max500 rpa2 teacher evaluation and logged the result in `docs/expander-training-strategy.md`.
- **Status:** Completed
- **Next Steps:** Use rpa2 as a high-budget teacher for trace/continuation collection, not as the fast inner-loop gate. Next collect rpa2 max500 conversion traces and train a conditional/planner-aware controller from those labels.
- **Context:** `adaptive-unet-ppo-v4` must load as shared HL-Gauss value (`--value-loss hl-gauss --value-bins 128`) with U-Net channels `64,96,128,64`; per-size value flags shift the Equinox tree. Correct rpa2 chunked eval at 128 games/seat, `eval_batch_size=32`, seed `104040`: p0 `72.66%`, p1 `56.25%`, min `56.25%`, draw `2.34%/6.25%`.

## [2026-06-20 22:55] rpa2 Max500 Conversion Trace v0
- **Changes:** Collected `runs/adaptive-online-search-fixed-v5-max500-rpa2-v0/` with high-budget rpa2 online search and 500-step conversion labels, then logged aggregate shard statistics.
- **Status:** Completed
- **Next Steps:** Scale this data to at least 2k-5k rows or 250+ conversion positives, then train a conditional search-entry/controller head. Avoid another full replace-policy CE run on this small shard.
- **Context:** Four GPU shards saved 471 rows with balanced seats (`223/248`), `63.69%` search action changes, `17.41%` search-improves-continuation, and `12.95%` search-converts-to-win (`61` positives). Search continuation wins were `146` vs base continuation wins `138`; the value is in conversion rows, not raw outcome count.

## [2026-06-20 23:14] RPA2 Teacher Confirmation
- **Changes:** Collected a small rpa2 fixed-v5 max500 continuation trace under `runs/adaptive-online-search-fixed-v5-max500-rpa2-conversion-smoke/`, trained a tiny pure policy-head rpa2 adapter probe under `runs/adaptive-online-search-conversion-adapter-rpa2-smoke/`, evaluated the high-budget rpa2 wrapper, and documented results in README, Chinese manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Scale rpa2 trace collection to at least `2k-5k` rows or `250+` conversion positives, then train a conditional online-search action controller/planner-aware head. Do not expand the tiny direct policy-head CE smoke.
- **Context:** The rpa2 trace smoke saved `501` rows with `48` search-converts-to-win and `63` search-improves-continuation rows; static adapter continuation converted `0` rows. The tiny rpa2 adapter regressed fixed-v5 max500 128-row min to `28.91%` versus same-seed static v1 `32.03%`. The actual rpa2 wrapper confirmed strongly: fixed-v5 max500 256-row min `58.20%`, and Expander 8/12/16 max750 64-row min `89.06%` with 12x/16x rows at `95.31%`-`98.44%`.

## [2026-06-20 23:15] rpa2 Online Search Gate Trainer
- **Changes:** Added `examples/_experimental/ppo/adaptive_online_search_gate_supervised.py`, documented rpa2 v1/max500 trace stats and the weak offline gate result in README, Chinese manual, and strategy notes.
- **Status:** Completed
- **Next Steps:** Do not use this gate for promotion. Use the rpa2 trace data for a richer controller or planner-aware conditional action head, and keep fixed-v5 diagnostics on `max500` with `max750` or longer only as confirmation.
- **Context:** Combined rpa2 v0+v1 had `2078` rows, `1317` changed actions, and `153` strict conversion rows. The conversion gate ended at precision `13.10%` / recall `94.74%`; the broader improve gate ended at precision `14.75%` / recall `56.63%`. Score/prior/phase features alone are not enough to accept search actions safely.

## [2026-06-20 23:42] Online Search Gate Hook
- **Changes:** Added and wired `adaptive_online_search_gate_supervised.py`; added `evaluate_adaptive_policy.py --online-search-gate-path/--online-search-gate-threshold/--online-search-gate-hidden-dim`; trained `runs/adaptive-online-search-gate-rpa2-improve-v0/generals-adaptive-online-search-gate-rpa2-improve-v0.eqx`; documented the result.
- **Status:** Completed
- **Next Steps:** Do not promote the first gate. Keep the hook as a controller path, but use larger rpa2 trace data and richer features before trying another online-search accept/reject gate.
- **Context:** The gate trained on `605` changed-action examples with `84` `search_improves_continuation` positives. Offline final was acc `68.9%`, precision `27.0%`, recall `64.0%`, P+ `0.538`, P- `0.415`. Fixed-v5 max500 128-row seed `104600`: gated threshold `0.5` min `57.03%`; ungated rpa2 same seed min `58.59%`.

## [2026-06-20 23:50] rpa2 Candidate Scorer Probe
- **Changes:** Added `examples/_experimental/ppo/adaptive_online_search_candidate_scorer.py`, a richer candidate-level MLP that learns online-search top-k ranking from observation/action-local features; trained all-changed, strict-convert, broader-improve, and hard-best variants under ignored `runs/`.
- **Status:** Completed
- **Next Steps:** Do not wire the scorer into gameplay yet. Use strict conversion rows with soft-rank supervision as the next controller/head direction, and collect more rpa2 conversion rows before attempting evaluator integration.
- **Context:** After removing an initial label-leak feature (`candidate_is_search_action`), the best all-changed scorer reached val top1 `38.78%`, top2 `70.0%`, pair `59.9%` versus prior top1 `0.38%`. Strict `search_converts_to_win` soft-rank rows were more promising despite only `101` rows: best val top1 `55.0%`, top2 `70.0%`, pair `60.0%`; hard-best CE overfit and fell to best top1 `30.0%`.

## [2026-06-20 23:51] Candidate Scorer Normalization Fix
- **Changes:** Stopped gradients through `feature_mean` and `feature_std` in `adaptive_online_search_candidate_scorer.py` so dataset normalization metadata stays fixed while the MLP trains.
- **Status:** Completed
- **Next Steps:** Use this fixed scorer for subsequent rpa2 strict-conversion diagnostics; older scorer checkpoints are still readable but were trained with mutable normalization leaves.
- **Context:** This is a training-stability fix before scaling independent rpa2 conversion shards. It does not change trace data or evaluator behavior.

## [2026-06-20 23:28] Frozen Candidate Scorer Rerun
- **Changes:** Reran fixed-normalization rpa2 candidate scorer probes on existing fixed-v5 max500 v0+v1 traces and documented the corrected metrics.
- **Status:** Completed
- **Next Steps:** Treat the candidate scorer as an offline signal probe only. Scale independent max500 strict-conversion rows before wiring a scorer or conditional action head into gameplay.
- **Context:** Frozen all-changed scorer best val top1/top2/pair was `34.22%/67.68%/57.54%` versus prior top1 `1.06%`. Frozen strict `search_converts_to_win` scorer best val top1/top2/pair was `40.0%/60.0%/59.66%` on `101` rows. The old mutable-normalization numbers were optimistic but the signal survives.

## [2026-06-20 23:37] rpa2 Strict Conversion Partial v2
- **Changes:** Collected partial fixed-v5 max500 rpa2 strict-conversion rows under `runs/adaptive-online-search-fixed-v5-max500-rpa2-v2-convert/`, trained a combined fixed-normalization strict-conversion candidate scorer, and documented the result.
- **Status:** Completed
- **Next Steps:** Do not integrate the scorer into gameplay. Collect substantially more independent strict-conversion rows or move to a planner-aware conditional action head.
- **Context:** The collection was interrupted after 6/8 shards and left a GPU JAX process, which was terminated after `nvidia-smi` confirmed it was still consuming GPU. The usable partial shard has `32` strict conversion rows with exact p0/p1 balance `16/16`. Combined v0+v1+partial-v2 strict scorer kept `133` rows and reached best val top1/top2/pair `37.04%/59.26%/55.90%`, not better than the `101`-row frozen v0/v1 split.

## [2026-06-20 23:39] Candidate Scorer Independent Validation
- **Changes:** Added `adaptive_online_search_candidate_scorer.py --val-dataset` / `--max-val-rows` for independent shard validation, trained `runs/adaptive-online-search-candidate-scorer-rpa2-convert-indval-v0/`, and documented the result.
- **Status:** Completed
- **Next Steps:** Use independent validation for any future scorer/controller. Do not wire the scorer into gameplay until pair/rank calibration improves on independent shards.
- **Context:** Training on rpa2 v0+v1 strict conversion rows (`101`) and validating on partial v2 strict conversion rows (`32`) gave best epoch `2`, top1/top2/pair `40.62%/53.12%/50.26%` versus prior top1 `0.00%`. Final epoch overtrained to top1 `21.88%`; scorer action top1 transfers somewhat, but ranking calibration is weak.

## [2026-06-20 23:52] rpa2 Strict Conversion v3/v4
- **Changes:** Collected two small fixed-v5 max500 rpa2 strict-conversion batches under `runs/adaptive-online-search-fixed-v5-max500-rpa2-v3-convert/` and `...v4-convert/`, then trained independent-validation scorer probes `indval-v1` and `indval-v2`.
- **Status:** Completed
- **Next Steps:** Move from standalone scorer diagnostics toward a planner-aware conditional action head with early stopping and independent validation. Do not integrate final-epoch scorer weights into gameplay.
- **Context:** v3 added `12` strict rows (`p0/p1=5/7`); v4 added `19` strict rows (`p0/p1=3/16`). Training on v0+v1+v2 and validating on v3+v4 reached best top1/top2/pair `45.16%/74.19%/58.38%`; training on v0+v1+v2+v3 and validating on p1-heavy v4 reached `63.16%/89.47%/64.91%`. Final epochs collapsed, confirming early stopping is mandatory.

## [2026-06-20 23:54] Candidate Scorer Heatmap Features
- **Changes:** Added `adaptive_online_search_candidate_scorer.py --heatmap-features`, including saved source/target/enemy-general heatmaps and target-center progress features; trained heatmap independent-validation probes.
- **Status:** Completed
- **Next Steps:** Keep heatmap features for model-aware scorer/conditional-head diagnostics, but require independent validation before integration.
- **Context:** Heatmap v0 on v3+v4 validation improved top1 from no-heatmap `45.16%` to `51.61%` with pair unchanged `58.38%`; top2 dropped from `74.19%` to `64.52%`. Heatmap v1 on p1-heavy v4 reached top1/top2/pair `57.89%/63.16%/56.14%`, below no-heatmap `63.16%/89.47%/64.91%`, so heatmaps help some splits but are not sufficient alone.

## [2026-06-21 00:04] Candidate Scorer Trunk Features
- **Changes:** Added `adaptive_online_search_candidate_scorer.py --feature-model-path` and feature-model template flags to append frozen adaptive trunk source/destination features to candidate action rows; documented usage and max500 strict-conversion probes.
- **Status:** Completed
- **Next Steps:** Do not wire this scorer directly into gameplay. Use frozen trunk features as ingredients for a planner-aware conditional action head or collect more independent strict-conversion rows before integration.
- **Context:** CUDA smoke loaded `adaptive-unet-ppo-v4` as shared HL-Gauss U-Net and extracted 64 trunk channels. On v0+v1+v2+v3 train / p1-heavy v4 validation, trunk+heatmap best top1/top2/pair was `47.37%/68.42%/53.51%`; trunk-only was `47.37%/73.68%/61.40%`. This did not beat the previous no-trunk v4-only reference `63.16%/89.47%/64.91%`, so the signal is useful but not a promotion path.

## [2026-06-21 00:06] Frozen U-Net Candidate Scorer
- **Changes:** Extended `adaptive_online_search_candidate_scorer.py` with `--feature-model-path` to append frozen adaptive trunk source/destination/delta features; ran CPU smoke checks and GPU independent-holdout diagnostics on rpa2 max500 strict-conversion rows.
- **Status:** Completed
- **Next Steps:** Stop adding standalone scorer features. Train a U-Net-internal conditional action or policy-adapter head from strict max500 conversion rows so the planner signal can enter the policy head directly.
- **Context:** The v4 trunk path loaded `runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx` as U-Net `64,96,128,64`, 35 input channels, shared HL-Gauss value. On v0+v1+v2-small -> v3-small (`217` train / `193` validation strict conversions), trunk-v0 reached only top1/top2/pair `32.12%/58.03%/51.82%`; smaller MLP `30.05%/60.10%/53.16%`; trunk with `local_channels=0` recovered `35.75%/66.32%/56.40%`; trunk+heatmap+local0 was `34.72%/62.69%/54.58%`. Frozen point features do not beat the plain/heatmap scorer enough to justify gameplay integration.

## [2026-06-21 00:18] Strategy Supervised Validation Gate
- **Changes:** Added `adaptive_strategy_supervised.py --val-dataset`, validation caps, `--selection-metric`, `.best.eqx` saving, JSON summaries, and ceil-batch epoch coverage; ran strict max500 one-step policy-head and strategy-Q probes.
- **Status:** Completed
- **Next Steps:** Stop one-step strict-conversion CE/Q distillation. Collect or train on executed-prefix / multi-step conditional action data so the model learns a short option, not just a replacement first action.
- **Context:** Policy-head CE and policy-head top-k rank both stayed at `15.79%` independent holdout top1 on v4-convert. Strategy-Q rank reached `57.89%` holdout top1, but fixed-v5 max500 64 games/seat Q replacement regressed from same-seed base min `31.25%` to `25.00%`. The validation infrastructure is useful; the one-step label family is not enough.

## [2026-06-21 00:30] v1-Init Max500 Policy-Head Fine-Tune
- **Changes:** Trained `runs/adaptive-online-search-conversion-adapter-v1-rpa2-ft-v0/` from the static conversion adapter v1 using rpa2 max500 strict-conversion rows with independent validation and evaluated fixed-v5/Expander gates.
- **Status:** Completed
- **Next Steps:** Do not promote this fine-tune. Keep static conversion adapter v1 as the cheap max500 adapter baseline; next work should use executed-prefix option data or a separate conditional action/adapter head instead of overwriting the base policy head with one-step search actions.
- **Context:** The fine-tune showed a 128 games/seat fixed-v5 max500 min of `39.06%`, but at 256 games/seat seed `120320` it scored `33.59%` min versus same-seed static v1 `36.72%`. Expander max750 128-row min was `78.12%`, so the failure is fixed-v5 conversion quality rather than catastrophic Expander forgetting.

## [2026-06-21 00:32] Conversion Policy Head Probe
- **Changes:** Added optional conversion policy heads to adaptive CNN/U-Net networks, wired `adaptive_strategy_supervised.py --update-scope conversion-policy-head`, wired evaluator `--conversion-policy-scale/--conversion-policy-mode`, and added online-search `--save-executed-prefix-steps` trace fields.
- **Status:** Completed
- **Next Steps:** Do not promote the one-step strict-conversion head. Use the new head and prefix trace fields for executed-prefix or multi-step conditional action training; keep fixed-v5 primary triage at `max500`, with `max750` or longer only as confirmation.
- **Context:** GPU training on rpa2 v0+v1+v2-small strict conversions selected epoch 34 with only `10.42%` holdout pairwise accuracy on v3-small. Replace-mode fixed-v5 max500 128 games/seat scored p0 `30.47%`, p1 `20.31%`, min `20.31%`, so the separate head reduced risk but did not solve one-step label noise.

## [2026-06-21 00:35] Max500 Executed-Prefix Trace Probe
- **Changes:** Collected `runs/adaptive-online-search-prefix-max500-v0/` with `--save-executed-prefix-steps 8`, trained `adaptive-online-search-prefix-conversion-head-v0`, `adaptive-online-search-prefix-policy-v0`, and a stronger `prefix-policy-v1-strong`, then documented the results.
- **Status:** Completed
- **Next Steps:** Do not promote these single-shard prefix checkpoints. Collect more strict max500 executed-prefix shards from the static-v1 + online-search teacher and add independent prefix validation before another training run.
- **Context:** The strict shard kept `56/1024` origin rows, p0/p1 `28/28`, with `425/448` valid prefix steps and 100% win plan outcomes. Prefix policy v0 reached fixed-v5 max500 256-row min `37.50%` versus same-seed static v1 `38.67%`; aggressive v1 dropped to 128-row min `30.47%`. The prefix data path is usable, but the first shard is too small to beat static v1.

## [2026-06-21 00:50] Multi-Shard Prefix Policy v2
- **Changes:** Collected `runs/adaptive-online-search-prefix-max500-v1/`, trained `runs/adaptive-online-search-prefix-policy-v2-multishard/` from v0 + v1-train prefix shards with independent v1 validation, and evaluated fixed-v5 max500 same-seed gates.
- **Status:** Completed
- **Next Steps:** Do not promote v2. Implement or use a true two-adapter wrapper next: v4 base, static v1 as the 8x opening/base adapter, and prefix/option adapter as a second late-game intervention. Avoid CE/pairwise weight sweeps on the current prefix rows.
- **Context:** Combined strict prefix data now has `164` origin rows and `1235` non-pass valid prefix steps. Prefix v2 best beat static v1 at 128 games/seat seed `122500` (`42.97%` vs `39.06%` min), but at 256 games/seat seed `122520` it only matched within noise (`34.38%` vs `33.98%` min) and stayed p0-limited. Static-v1 base plus late prefix v2 also failed 256-row confirmation (`32.81%` min).

## [2026-06-21 00:55] Two-Adapter Evaluator Hook
- **Changes:** Added `evaluate_adaptive_policy.py --late-policy-adapter-*` flags to compose a second policy adapter after the primary adapter, documented the CLI in README/Chinese manual, and recorded the fixed-v5 max500 wrapper result.
- **Status:** Completed
- **Next Steps:** Keep the hook. Do not promote prefix v2 as the late adapter; train a p0/p1-balanced option adapter or a learned false-positive-penalized gate before retesting the two-adapter deployment shape.
- **Context:** CPU smoke passed. True wrapper `v4 + static v1 max8 replace + prefix v2 late replace turn>=80/contact` scored fixed-v5 max500 256 games/seat seed `122520`: p0 `32.81%`, p1 `41.02%`, min `32.81%`, below same-seed static v1 min `33.98%`.

## [2026-06-21 10:20] Mixed Rollout Test Contract Repair
- **Changes:** Updated `tests/test_adaptive_ppo.py` so `test_collect_mixed_rollout_combines_both_learner_seats` unpacks the current 10-value `collect_mixed_rollout` contract and verifies carried fog-memory state shape.
- **Status:** Completed
- **Next Steps:** Continue fixed-v5 and Expander research iteration from a passing baseline.
- **Context:** Baseline `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` now passes `215 passed in 130.37s`. Sandbox CUDA probing fails with `CUDA_ERROR_NO_DEVICE`, but the same JAX probe succeeds under escalation with backend `gpu` and `CudaDevice(id=0)`.

## [2026-06-21 10:46] Focused Seat Eval and 12/16 Search Compression Probe
- **Changes:** Added `evaluate_adaptive_policy.py --policy-players` for focused seat evaluation, documented it in `README.md`, and added parser/CLI tests. Trained two ignored 12/16 Expander search-compression policy-head adapters under `runs/adaptive-expander-1216-search-policy-v0/` and `runs/adaptive-expander-1216-search-policy-v1-aggressive/`.
- **Status:** Completed
- **Next Steps:** Do not continue direct 12/16 policy-head CE/rank compression from these shards. Use focused-seat eval for weak-row checks, and treat expensive 32-step search as a teacher/diagnostic rather than an inner-loop evaluator.
- **Context:** In-session baselines: static-v1 vs fixed-v5 max500 64-row seed `200001` min `28.12%`; runtime search vs fixed-v5 max500 64-row seed `200021` min `56.25%`; runtime search vs Expander 8/12/16 64-row seed `200041` min `89.06%`. Model-only Expander seed `200061` min was `73.44%`; late 12/16 adapter v0 dropped to `71.88%`, aggressive v1 dropped to `68.75%`. Stronger 32-step teacher search on larger maps produced completed rows `12p0 90.62%`, `12p1 90.62%`, `16p0 93.75%`, then targeted `16p1 87.50%`; it is strong but too slow for routine triage.

## [2026-06-21 11:04] Candidate Scorer Deployment Hook
- **Changes:** Added `evaluate_adaptive_policy.py --candidate-scorer-*` to deploy base+local-channel online-search candidate scorers as a cheap top-k action selector, with parser validation, JSON metadata, README/Chinese manual docs, and focused tests.
- **Status:** Completed
- **Next Steps:** Keep the hook for diagnostics, but do not promote the existing rpa2 MLP scorer. The next controller should use richer model/state features or move inside the U-Net policy/conditional head rather than threshold-sweeping this scorer.
- **Context:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` passed `218 passed in 138.64s`. Feature parity against saved rpa2 shards matched offline extraction to `1.19e-07`, so gameplay failure is not a feature-order bug. GPU probe `v4 + static conversion adapter v1 + rpa2-v1 scorer top4 turn>=80/contact` vs fixed-v5 max500 64 games/seat seed `200201` scored p0 `1.56%`, p1 `7.81%`, min `1.56%`; offline scorer gap calibration also worsened with threshold (`gap>=0.5` kept `7.6%` rows at `30.4%` top1).

## [2026-06-21 11:20] RPA2 Prefix CE Probe
- **Changes:** Collected a small strict executed-prefix rpa2 teacher shard under `runs/adaptive-online-search-prefix-max500-rpa2-v2-smoke/`, trained `runs/adaptive-online-search-prefix-policy-v3-rpa2-ce/` from v0+v1+rpa2 prefix rows with independent validation, and evaluated the best checkpoint against fixed-v5 max500.
- **Status:** Completed
- **Next Steps:** Do not add more raw primitive-action CE to this adapter family. Pivot to a structural controller: learn when to enter a short search/plan option, or deploy the proven online-search wrapper while compressing only a gate/option policy.
- **Context:** The rpa2 shard kept `65/1024` strict rows, p0/p1 `36/29`, with `475` non-pass valid prefix actions and mean search continuation score delta `165.67`. Training kept `1971/2136` train prefix rows, selected epoch `4`, and reached only `33.21%` validation teacher-action accuracy. Gameplay with `v3-rpa2-ce.best.eqx` scored fixed-v5 max500 64 games/seat seed `200441`: p0 `32.81%`, p1 `39.06%`, min `32.81%`, so it is below both static-v1 and the runtime-search wrapper.

## [2026-06-21 11:24] Current Champion Wrapper Confirmation
- **Changes:** Confirmed the deployable champion as `adaptive-unet-ppo-v4 + static conversion adapter v1 on 8x8 + online search top4/r16/rpa2 after turn 80 contact`; added a focused large-map weak-seat check at `runs/goal-expander-16p1-static-v1-online-search-rpa2-128-seed200521.json`.
- **Status:** Completed
- **Next Steps:** Treat this wrapper as the current goal-satisfying baseline. Future work should improve cost/latency with a learned gate or option controller, not replace it with raw policy-head CE unless a validation gate beats this champion.
- **Context:** Existing fixed-v5 max500 8x8 128 games/seat seed `101060` scored p0 `65.62%`, p1 `60.16%`, min `60.16%`, clearly above the v5 target. Existing Expander 8/12/16 32 games/seat rpa2 check scored min `81.25%`; the new 16x16 player-1 focused confirmation at 128 games scored wins/losses/draws `121/3/4`, win_rate `94.53%`, decisive `97.58%`. Larger-map advantage is therefore very large, and the remaining research risk is compression/compute rather than playing strength.
