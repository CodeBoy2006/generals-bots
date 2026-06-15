# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`generals-bots` is a JAX-based two-player Generals.io simulator and bot/RL research framework. The goal is a reproducible, massively-parallelizable game environment suitable for reinforcement learning. The detailed manual (Chinese) is in [docs/zh-manual.md](docs/zh-manual.md).

## Environment & commands

The project uses **`uv`** with `uv.lock`, Python >= 3.11. (The `Makefile` still references `poetry` and is stale — prefer the `uv` commands below, which match the README and docs.)

```bash
# Install deps (dev extras)
uv sync --extra dev

# Install GPU support (machine has CUDA 13 / RTX-class GPU)
uv sync --extra dev --extra cuda13

# Run full test suite
uv run pytest

# Run a single test file / test
uv run pytest tests/test_game_jax.py
uv run pytest tests/test_game_jax.py::test_create_initial_state

# Compile-check everything (cheap sanity gate used throughout the devlogs)
uv run python -m compileall generals examples tests

# Lint (config in pyproject.toml: ruff, line-length 120, py311)
uv run ruff check generals
```

Verify the install and JAX backend:

```bash
uv run python -c "import generals; print(generals.GeneralsEnv)"
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

GPU runs in the experimental scripts are launched with these env vars:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python ...
```

Pre-commit gate used in this repo: `uv run pytest`, then `git diff --check`.

## Architecture

The whole simulator is **pure JAX and stateless**. Game logic lives in plain functions over an immutable `GameState` NamedTuple; everything is `@jax.jit`-able and `vmap`-able. There are no mutable objects holding game state — this is the single most important fact about the codebase.

### `generals/core/` — the engine

- **`game.py`** — the heart. `GameState` (NamedTuple: armies, ownership, generals, cities, mountains, etc.) and the functions over it: `create_initial_state(grid)`, `step(state, actions) -> (state, GameInfo)`, `get_observation(state, player_idx)`, `batch_step` (vmapped step). Army growth timing lives in `global_update` (structures every 2 ticks, all cells every 50). Move-priority resolution (chasing > reinforcing > bigger army) is in `_determine_move_order`. All branching uses `lax.cond`, not Python `if`, because it must trace under JIT.
- **`env.py`** — `GeneralsEnv`, a **stateless config bag** wrapping `game`. `reset(key) -> (pool, state)` pre-generates a *pool* of initial states; `step(state, actions, pool)` takes that pool as an explicit argument so a finished game auto-resets by indexing `pool[pool_idx % pool_size]` — no recompilation when the pool changes. Supports fixed `grid_dims=(h,w)` or variable `min_grid_size/max_grid_size/pad_to` (padded with mountains). `TimeStep` carries `last_state` for bootstrap values.
- **`action.py`** — action format and validity. Actions are length-5 int arrays `[pass, row, col, direction, split]` (direction 0=UP 1=DOWN 2=LEFT 3=RIGHT; split=1 moves half, 0 moves all-but-one). `compute_valid_move_mask` returns an `(H,W,4)` mask.
- **`observation.py`**, **`grid.py`** (`generate_grid` with mountains/cities/general-distance constraints), **`rewards.py`** (`composite_reward_fn`, `win_lose_reward_fn`), **`config.py`**, **`rendering.py`**.

Numeric grid encoding (used by `create_initial_state` and generators): `-2`=mountain, `0`=empty, `1`=player-0 general, `2`=player-1 general, `40-50`=city with that army value.

### `generals/agents/`

`Agent` ABC with `act(observation, key) -> action`. Built-ins: `RandomAgent`, `ExpanderAgent`, `PPOPolicyAgent`. `_heuristic_logic.py` provides a pool of pure-function heuristics (Expander, CityRush, GeneralHunter, DefensiveExpander, Balanced, Mixed) exposed via `HEURISTIC_NAMES` / `heuristic_action(id, key, obs)`, used as PPO teachers and opponents. `ppo_policy_agent.py` defines `PolicyValueNetwork` (Equinox conv net, 9 input channels, 9 policy planes per cell) plus action-encoding helpers (`obs_to_array`, `index_to_action`, `greedy_policy_action`, `sampled_policy_action`).

### `generals/gui/` and `generals/remote/`

- `gui/` — pygame live visualization (`gui.py`, `event_handler.py`) and `replay_gui.py`.
- `remote/generalsio_client.py` — socket.io client to play on the real generals.io servers (public vs bot endpoints), driving any `Agent`.

### `examples/` and `examples/_experimental/`

Top-level examples are the supported entry points: `simple_example.py`, `vectorized_example.py`, `visualization_example.py`, `client_example.py`, `play_against_model.py`. The PPO research stack lives in `examples/_experimental/ppo/` (`train.py` raw-game trainer = primary path, `train2.py`, `behavior_clone.py`, `evaluate_policy.py`, `evaluate_heuristics.py`, `network.py`, shared `common.py`). See [examples/_experimental/README.md](examples/_experimental/README.md) for the full command catalog.

## Conventions & gotchas

- **Stay JAX-pure in `generals/core/` and agent logic.** Use `lax.cond`/`jnp.where`, not Python control flow over traced values. New game logic must remain JIT- and vmap-compatible.
- **Current env interface is `reset(key) -> (pool, state)` and `step(state, actions, pool)`.** `bench.py` and `examples/_experimental/benchmark_performance.py` still contain *older* interface assumptions and must be fixed to this signature before use — do not trust them as references.
- **`.eqx` checkpoints are experimental artifacts.** Keep them in `/tmp` or another scratch dir; never commit them. When loading a checkpoint, `--grid-size` must match the size the model was trained at.
- **`ruff` excludes `tests/` and `examples/`** (see `pyproject.toml`); lint rules apply to `generals/` only.
- **4x4 maps are smoke-test only.** Real policy-quality claims need 8x8+ generated maps with long horizons and independent-seed batch evaluation. Expander remains the strongest short-horizon baseline against which new agents are judged.
- Devlogs under `docs/devlogs/` and `statusquo.md` record the running history of the PPO/heuristic experiments — check them for context on prior training runs and decisions.
