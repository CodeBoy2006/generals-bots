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
