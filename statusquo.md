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
