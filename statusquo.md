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
