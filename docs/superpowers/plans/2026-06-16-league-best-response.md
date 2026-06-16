# League Best-Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the league evaluation and training infrastructure needed to optimize toward 80% win rate against every current heuristic and v5 in both seats.

**Architecture:** Add a standalone league evaluator first, then add reusable checkpoint-saving helpers, then extend ordinary PPO training with a same-architecture frozen opponent pool. Training runs remain separate from promotion: every candidate must be evaluated by the league tool before it can become the current best.

**Tech Stack:** Python 3.12 via `uv run`, JAX, Equinox, Optax, pytest, existing experimental PPO scripts.

---

## File Structure

- Create `examples/_experimental/ppo/evaluate_league.py`: CLI and reusable functions for evaluating one candidate against heuristics and checkpoint opponents.
- Modify `examples/_experimental/ppo/train.py`: add periodic checkpoint saving and ordinary PPO checkpoint opponent pools.
- Modify `examples/_experimental/ppo/train_recurrent.py`: add periodic checkpoint saving only.
- Create `tests/test_evaluate_league.py`: pure summary tests and CLI smoke coverage.
- Modify `tests/test_ppo_train.py`: checkpoint-saving and opponent-pool smoke tests.
- Modify `tests/test_recurrent_ppo.py`: recurrent checkpoint-saving smoke tests.
- Modify `docs/expander-training-strategy.md`: record commands, interim results, and the current best candidate after training.
- Modify `statusquo.md`: append one entry per substantive implementation or training-result commit.

## Task 1: League Evaluator

**Files:**
- Create: `examples/_experimental/ppo/evaluate_league.py`
- Create: `tests/test_evaluate_league.py`

- [ ] **Step 1: Write failing pure summary tests**

Add `tests/test_evaluate_league.py`:

```python
from examples._experimental.ppo.evaluate_league import (
    REQUIRED_HEURISTIC_OPPONENTS,
    LeagueRow,
    compute_league_summary,
    parse_checkpoint_specs,
)


def test_parse_checkpoint_specs_accepts_name_path_mode():
    specs = parse_checkpoint_specs(["v5=generals-ppo-8x8-expander-gpu-v5.eqx:sample"])
    assert specs == [("v5", "generals-ppo-8x8-expander-gpu-v5.eqx", "sample")]


def test_parse_checkpoint_specs_defaults_name_and_mode():
    specs = parse_checkpoint_specs(["generals-ppo-8x8-expander-gpu-v5.eqx"])
    assert specs == [
        (
            "generals-ppo-8x8-expander-gpu-v5",
            "generals-ppo-8x8-expander-gpu-v5.eqx",
            "sample",
        )
    ]


def test_required_heuristics_excludes_random():
    assert "random" not in REQUIRED_HEURISTIC_OPPONENTS
    assert "expander" in REQUIRED_HEURISTIC_OPPONENTS


def test_compute_league_summary_uses_min_required_win_rate():
    rows = [
        LeagueRow("heuristic", "expander", 0, 90, 10, 0, 100, 120.0, True),
        LeagueRow("heuristic", "expander", 1, 79, 20, 1, 100, 121.0, True),
        LeagueRow("checkpoint", "v5", 0, 85, 15, 0, 100, 122.0, True),
        LeagueRow("checkpoint", "v5", 1, 82, 18, 0, 100, 123.0, True),
        LeagueRow("sanity", "random", 0, 100, 0, 0, 100, 90.0, False),
    ]
    summary = compute_league_summary(rows, threshold=0.8)
    assert summary["required_pairs"] == 4
    assert summary["passed_pairs"] == 3
    assert summary["league_score"] == 0.79
    assert summary["passes_threshold"] is False
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_evaluate_league.py
```

Expected: import failure because `examples._experimental.ppo.evaluate_league` does not exist.

- [ ] **Step 3: Implement the evaluator scaffolding and summary functions**

Create `examples/_experimental/ppo/evaluate_league.py` with:

```python
"""Evaluate a policy checkpoint against heuristic and checkpoint league opponents."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from common import HEURISTIC_NAMES, POLICY_MODE_NAMES

REQUIRED_HEURISTIC_OPPONENTS = tuple(HEURISTIC_NAMES)


@dataclass(frozen=True)
class LeagueRow:
    opponent_type: str
    opponent_name: str
    policy_player: int
    wins: int
    losses: int
    draws: int
    games: int
    mean_time: float
    required: bool

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def decisive_win_rate(self) -> float:
        decisive = self.wins + self.losses
        return self.wins / max(decisive, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games


def _default_checkpoint_name(path: str) -> str:
    return Path(path).stem


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, str, str]]:
    specs = []
    for raw in values:
        name_and_path, sep, mode = raw.rpartition(":")
        if not sep:
            name_and_path = raw
            mode = "sample"
        if mode not in POLICY_MODE_NAMES:
            raise ValueError(f"Unsupported checkpoint policy mode: {mode}")
        name, sep, path = name_and_path.partition("=")
        if not sep:
            path = name_and_path
            name = _default_checkpoint_name(path)
        specs.append((name, path, mode))
    return specs


def compute_league_summary(rows: list[LeagueRow], threshold: float) -> dict[str, object]:
    required_rows = [row for row in rows if row.required]
    passed_pairs = sum(row.win_rate >= threshold for row in required_rows)
    league_score = min((row.win_rate for row in required_rows), default=0.0)
    return {
        "required_pairs": len(required_rows),
        "passed_pairs": passed_pairs,
        "league_score": league_score,
        "threshold": threshold,
        "passes_threshold": bool(required_rows) and passed_pairs == len(required_rows),
    }


def row_to_dict(row: LeagueRow, threshold: float) -> dict[str, object]:
    data = asdict(row)
    data["win_rate"] = row.win_rate
    data["decisive_win_rate"] = row.decisive_win_rate
    data["draw_rate"] = row.draw_rate
    data["passes_threshold"] = (not row.required) or row.win_rate >= threshold
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against a policy league.")
    parser.add_argument("candidate_path")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--checkpoint-opponent", action="append", default=[])
    parser.add_argument("--heuristic", action="append", default=list(REQUIRED_HEURISTIC_OPPONENTS))
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--require-threshold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_specs = parse_checkpoint_specs(args.checkpoint_opponent)
    result = {
        "candidate_path": args.candidate_path,
        "threshold": args.threshold,
        "heuristics": args.heuristic,
        "checkpoint_opponents": checkpoint_specs,
        "rows": [],
        "summary": compute_league_summary([], args.threshold),
    }
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.require_threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run pure tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_evaluate_league.py
```

Expected: tests pass.

- [ ] **Step 5: Add actual ordinary-policy evaluation paths**

Extend `evaluate_league.py` to import existing evaluation helpers and add:

```python
import time

import equinox as eqx
import jax
import jax.random as jrandom

from evaluate_policy import (
    evaluate_batch,
    evaluate_policy_opponent_batch,
    summarize_policy_results,
)
from network import PolicyValueNetwork
from common import (
    OPPONENT_NAME_TO_ID,
    POLICY_INPUT_NAME_TO_ID,
    POLICY_MODE_NAME_TO_ID,
    make_grids,
    policy_input_default_channels,
)
from generals.core import game
from generals.agents.ppo_policy_agent import parse_policy_channels


def make_eval_states(args, key):
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)
    grids = make_grids(
        key,
        args.num_games,
        args.grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    return jax.vmap(game.create_initial_state)(grids)


def load_policy_network(path, key, grid_size, channels, input_channels):
    network = PolicyValueNetwork(
        key,
        grid_size=grid_size,
        channels=channels,
        input_channels=input_channels,
    )
    return eqx.tree_deserialise_leaves(path, network)


def row_from_info(opponent_type, opponent_name, policy_player, info, num_games, required):
    summary = summarize_policy_results(info, policy_player, num_games)
    return LeagueRow(
        opponent_type=opponent_type,
        opponent_name=opponent_name,
        policy_player=policy_player,
        wins=summary["wins"],
        losses=summary["losses"],
        draws=summary["draws"],
        games=num_games,
        mean_time=summary["mean_time"],
        required=required,
    )


def evaluate_heuristic_rows(args, network, states, key):
    rows = []
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    for opponent_name in args.heuristic:
        for policy_player in (0, 1):
            key, eval_key = jrandom.split(key)
            info = evaluate_batch(
                network,
                states,
                eval_key,
                args.max_steps,
                OPPONENT_NAME_TO_ID[opponent_name],
                policy_mode,
                policy_player,
                policy_input,
            )
            jax.block_until_ready(info.winner)
            rows.append(row_from_info("heuristic", opponent_name, policy_player, info, args.num_games, True))
    return rows


def evaluate_checkpoint_rows(args, network, states, key, checkpoint_specs):
    rows = []
    policy_mode = POLICY_MODE_NAME_TO_ID[args.policy_mode]
    policy_input = POLICY_INPUT_NAME_TO_ID[args.policy_input]
    for index, (opponent_name, opponent_path, opponent_mode_name) in enumerate(checkpoint_specs):
        opponent_network = load_policy_network(
            opponent_path,
            jrandom.fold_in(key, 1000 + index),
            args.grid_size,
            args.opponent_channels,
            args.opponent_input_channels,
        )
        opponent_mode = POLICY_MODE_NAME_TO_ID[opponent_mode_name]
        for policy_player in (0, 1):
            key, eval_key = jrandom.split(key)
            info = evaluate_policy_opponent_batch(
                network,
                opponent_network,
                states,
                eval_key,
                args.max_steps,
                policy_mode,
                policy_player,
                opponent_mode,
                policy_input,
            )
            jax.block_until_ready(info.winner)
            rows.append(row_from_info("checkpoint", opponent_name, policy_player, info, args.num_games, True))
    return rows
```

Also extend `parse_args()` with these existing evaluation options so `make_eval_states()` and loaders have concrete values:

```python
parser.add_argument("--grid-size", type=int, default=8)
parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
parser.add_argument("--policy-mode", choices=POLICY_MODE_NAMES, default="sample")
parser.add_argument("--policy-input", choices=POLICY_INPUT_NAMES, default="observation")
parser.add_argument("--channels", default=None)
parser.add_argument("--opponent-channels", default=None)
parser.add_argument("--input-channels", type=int, default=None)
parser.add_argument("--opponent-input-channels", type=int, default=9)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--mountain-density-min", type=float, default=0.12)
parser.add_argument("--mountain-density-max", type=float, default=0.22)
parser.add_argument("--num-cities-min", type=int, default=4)
parser.add_argument("--num-cities-max", type=int, default=8)
parser.add_argument("--min-generals-distance", type=int, default=5)
parser.add_argument("--max-generals-distance", type=int, default=None)
parser.add_argument("--city-army-min", type=int, default=40)
parser.add_argument("--city-army-max", type=int, default=51)
```

Parse channels in `parse_args()`:

```python
args.channels = parse_policy_channels(args.channels)
args.opponent_channels = parse_policy_channels(args.opponent_channels or args.channels)
```

Then replace the initial empty result in `main()` with actual evaluation:

```python
key = jrandom.PRNGKey(args.seed)
key, net_key, map_key = jrandom.split(key, 3)
input_channels = args.input_channels or policy_input_default_channels(args.policy_input)
network = load_policy_network(args.candidate_path, net_key, args.grid_size, args.channels, input_channels)
states = make_eval_states(args, map_key)
t0 = time.time()
rows = evaluate_heuristic_rows(args, network, states, key)
rows.extend(evaluate_checkpoint_rows(args, network, states, key, checkpoint_specs))
summary = compute_league_summary(rows, args.threshold)
result = {
    "candidate_path": args.candidate_path,
    "threshold": args.threshold,
    "elapsed_seconds": time.time() - t0,
    "rows": [row_to_dict(row, args.threshold) for row in rows],
    "summary": summary,
}
```

Keep recurrent checkpoint evaluation out of Task 1.

- [ ] **Step 6: Add a CLI smoke test with tiny games**

Add to `tests/test_evaluate_league.py`:

```python
import subprocess
from pathlib import Path


def test_evaluate_league_cli_writes_json(tmp_path):
    output = tmp_path / "league.json"
    cmd = [
        "uv",
        "run",
        "python",
        "examples/_experimental/ppo/evaluate_league.py",
        "generals-ppo-8x8-expander-gpu-v5.eqx",
        "--heuristic",
        "expander",
        "--num-games",
        "2",
        "--seed",
        "30100",
        "--json-output",
        str(output),
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    assert "league_score" in completed.stdout
    assert output.exists()
```

- [ ] **Step 7: Run evaluator tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_evaluate_league.py
```

Expected: all evaluator tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add examples/_experimental/ppo/evaluate_league.py tests/test_evaluate_league.py
git commit -m "feat: add league evaluator"
```

## Task 2: Periodic Checkpoint Saving

**Files:**
- Modify: `examples/_experimental/ppo/train.py`
- Modify: `examples/_experimental/ppo/train_recurrent.py`
- Modify: `tests/test_ppo_train.py`
- Modify: `tests/test_recurrent_ppo.py`

- [ ] **Step 1: Write checkpoint helper tests**

Add to `tests/test_ppo_train.py`:

```python
from pathlib import Path

from examples._experimental.ppo.train import (
    checkpoint_path_for_iteration,
    prune_old_checkpoints,
)


def test_checkpoint_path_for_iteration_uses_zero_padded_iteration(tmp_path):
    path = checkpoint_path_for_iteration(tmp_path, "candidate", 50)
    assert path == tmp_path / "candidate-iter-000050.eqx"


def test_prune_old_checkpoints_keeps_newest(tmp_path):
    paths = [tmp_path / f"candidate-iter-{idx:06d}.eqx" for idx in (50, 100, 150)]
    for path in paths:
        path.write_text("x", encoding="utf-8")
    prune_old_checkpoints(paths, keep=2)
    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_ppo_train.py::test_checkpoint_path_for_iteration_uses_zero_padded_iteration tests/test_ppo_train.py::test_prune_old_checkpoints_keeps_newest
```

Expected: import failure for missing helper functions.

- [ ] **Step 3: Implement checkpoint helpers in `train.py`**

Add near the existing loading helpers:

```python
def checkpoint_path_for_iteration(checkpoint_dir, model_stem, iteration):
    """Return the periodic checkpoint path for one training iteration."""
    return Path(checkpoint_dir) / f"{model_stem}-iter-{iteration:06d}.eqx"


def prune_old_checkpoints(paths, keep):
    """Delete older periodic checkpoints when a positive keep limit is configured."""
    if keep is None or keep <= 0:
        return
    for path in list(paths)[:-keep]:
        Path(path).unlink(missing_ok=True)
```

- [ ] **Step 4: Add CLI arguments and save inside the training loop**

Add arguments:

```python
parser.add_argument("--checkpoint-dir", default=None)
parser.add_argument("--checkpoint-every", type=int, default=0)
parser.add_argument("--keep-checkpoints", type=int, default=0)
```

Validate:

```python
if args.checkpoint_every < 0:
    parser.error("--checkpoint-every cannot be negative")
if args.keep_checkpoints < 0:
    parser.error("--keep-checkpoints cannot be negative")
```

Inside `main()`, before training:

```python
saved_checkpoints = []
checkpoint_stem = Path(args.model_path).stem
if args.checkpoint_dir is not None:
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
```

After each printed iteration or at the end of each iteration block:

```python
if args.checkpoint_dir is not None and args.checkpoint_every > 0 and (iteration + 1) % args.checkpoint_every == 0:
    checkpoint_path = checkpoint_path_for_iteration(args.checkpoint_dir, checkpoint_stem, iteration + 1)
    eqx.tree_serialise_leaves(checkpoint_path, network)
    saved_checkpoints.append(checkpoint_path)
    prune_old_checkpoints(saved_checkpoints, args.keep_checkpoints)
    saved_checkpoints = [path for path in saved_checkpoints if path.exists()]
```

- [ ] **Step 5: Mirror checkpoint helpers in `train_recurrent.py`**

Import or duplicate the same small helper names. Use the recurrent network variable when saving.

- [ ] **Step 6: Add recurrent helper test**

Add to `tests/test_recurrent_ppo.py`:

```python
from examples._experimental.ppo.train_recurrent import checkpoint_path_for_iteration


def test_recurrent_checkpoint_path_for_iteration(tmp_path):
    assert checkpoint_path_for_iteration(tmp_path, "rnn", 7) == tmp_path / "rnn-iter-000007.eqx"
```

- [ ] **Step 7: Run checkpoint tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_ppo_train.py::test_checkpoint_path_for_iteration_uses_zero_padded_iteration tests/test_ppo_train.py::test_prune_old_checkpoints_keeps_newest tests/test_recurrent_ppo.py::test_recurrent_checkpoint_path_for_iteration
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add examples/_experimental/ppo/train.py examples/_experimental/ppo/train_recurrent.py tests/test_ppo_train.py tests/test_recurrent_ppo.py
git commit -m "feat: add periodic training checkpoints"
```

## Task 3: Ordinary PPO Opponent Policy Pool

**Files:**
- Modify: `examples/_experimental/ppo/train.py`
- Modify: `tests/test_ppo_train.py`

- [ ] **Step 1: Write parser tests**

Add to `tests/test_ppo_train.py`:

```python
from examples._experimental.ppo.train import parse_opponent_policy_pool, resolve_opponent_source


def test_parse_opponent_policy_pool_splits_paths_and_modes():
    pool = parse_opponent_policy_pool("a.eqx,b.eqx", "sample,greedy")
    assert pool == [("a.eqx", "sample"), ("b.eqx", "greedy")]


def test_parse_opponent_policy_pool_defaults_modes():
    pool = parse_opponent_policy_pool("a.eqx,b.eqx", None)
    assert pool == [("a.eqx", "sample"), ("b.eqx", "sample")]


def test_resolve_opponent_source_rejects_single_and_pool():
    try:
        resolve_opponent_source("a.eqx", False, opponent_policy_pool=["b.eqx"])
    except ValueError as exc:
        assert "--opponent-policy-path" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run parser tests to confirm failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_ppo_train.py::test_parse_opponent_policy_pool_splits_paths_and_modes tests/test_ppo_train.py::test_parse_opponent_policy_pool_defaults_modes tests/test_ppo_train.py::test_resolve_opponent_source_rejects_single_and_pool
```

Expected: missing functions or signature mismatch.

- [ ] **Step 3: Implement pool parsing and source resolution**

In `train.py`:

```python
def parse_opponent_policy_pool(pool_value, modes_value):
    """Parse comma-separated frozen opponent checkpoint paths and modes."""
    if not pool_value:
        return []
    paths = [item.strip() for item in pool_value.split(",") if item.strip()]
    if modes_value:
        modes = [item.strip() for item in modes_value.split(",") if item.strip()]
        if len(modes) != len(paths):
            raise ValueError("--opponent-policy-pool-modes must match --opponent-policy-pool length")
    else:
        modes = ["sample"] * len(paths)
    for mode in modes:
        if mode not in POLICY_MODE_NAMES:
            raise ValueError(f"Unsupported opponent policy mode: {mode}")
    return list(zip(paths, modes))
```

Update `resolve_opponent_source`:

```python
def resolve_opponent_source(opponent_policy_path, self_play_opponent, opponent_policy_pool=None):
    """Select which opponent source the PPO rollout loop should use."""
    has_pool = bool(opponent_policy_pool)
    if sum(bool(value) for value in (opponent_policy_path, self_play_opponent, has_pool)) > 1:
        raise ValueError("--opponent-policy-path, --self-play-opponent, and --opponent-policy-pool are mutually exclusive")
    if self_play_opponent:
        return "current"
    if has_pool:
        return "checkpoint_pool"
    if opponent_policy_path is not None:
        return "checkpoint"
    return "heuristic"
```

- [ ] **Step 4: Add CLI arguments**

Add:

```python
parser.add_argument("--opponent-policy-pool", default=None)
parser.add_argument("--opponent-policy-pool-modes", default=None)
```

In `main()`:

```python
try:
    opponent_policy_pool = parse_opponent_policy_pool(args.opponent_policy_pool, args.opponent_policy_pool_modes)
    opponent_source = resolve_opponent_source(args.opponent_policy_path, args.self_play_opponent, opponent_policy_pool)
except ValueError as exc:
    parser.error(str(exc))
```

- [ ] **Step 5: Implement same-architecture pool loading**

Add:

```python
def load_opponent_policy_pool(key, pool_specs, grid_size, channels, input_channels):
    """Load a same-architecture pool of ordinary PolicyValueNetwork opponents."""
    networks = []
    modes = []
    for index, (path_value, mode_name) in enumerate(pool_specs):
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Opponent pool checkpoint not found: {path}")
        network = PolicyValueNetwork(
            jrandom.fold_in(key, index),
            grid_size=grid_size,
            channels=channels,
            input_channels=input_channels,
        )
        networks.append(eqx.tree_deserialise_leaves(path, network))
        modes.append(POLICY_MODE_NAME_TO_ID[mode_name])
    return tuple(networks), jnp.array(modes, dtype=jnp.int32)
```

- [ ] **Step 6: Implement pool action dispatch in rollout**

Add a JAX helper:

```python
def policy_pool_action(opponent_networks, opponent_modes, opponent_index, key, obs):
    """Dispatch one action from a tuple of frozen ordinary policy opponents."""
    branches = tuple(
        (lambda network: (lambda _: policy_network_action(network, key, obs, opponent_modes[opponent_index])))(network)
        for network in opponent_networks
    )
    return jax.lax.switch(opponent_index, branches, None)
```

In rollout, sample one `opponent_indices` vector per environment reset batch:

```python
pool_size = len(opponent_networks)
opponent_indices = jrandom.randint(pool_key, (num_envs,), 0, pool_size)
```

Use `jax.vmap` over `(opponent_indices, opponent_keys, opponent_obs)` when `opponent_source == "checkpoint_pool"`.

- [ ] **Step 7: Add smoke coverage**

Add a tiny test that creates two temporary checkpoints from the same initialized network, calls the pool loader, and runs a one-update training smoke command with `--opponent-policy-pool`.

Command in test:

```python
cmd = [
    "uv",
    "run",
    "python",
    "examples/_experimental/ppo/train.py",
    "4",
    "--grid-size",
    "8",
    "--map-generator",
    "generated",
    "--pool-size",
    "8",
    "--num-steps",
    "2",
    "--num-iterations",
    "1",
    "--num-epochs",
    "1",
    "--minibatch-size",
    "8",
    "--opponent-policy-pool",
    f"{path_a},{path_b}",
    "--model-path",
    str(tmp_path / "out.eqx"),
]
```

- [ ] **Step 8: Run PPO train tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_ppo_train.py
```

Expected: all PPO train tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add examples/_experimental/ppo/train.py tests/test_ppo_train.py
git commit -m "feat: add ppo opponent checkpoint pools"
```

## Task 4: Full Verification and Docs

**Files:**
- Modify: `docs/expander-training-strategy.md`
- Modify: `statusquo.md`

- [ ] **Step 1: Run full verification**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
```

Expected: all commands pass.

- [ ] **Step 2: Document usage**

Add to `docs/expander-training-strategy.md` a `checkpoint league` section with:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --opponent-policy-pool generals-ppo-8x8-expander-gpu-v2.eqx,generals-ppo-8x8-expander-gpu-v3.eqx,generals-ppo-8x8-expander-gpu-v4.eqx,generals-ppo-8x8-expander-gpu-v5.eqx \
  --init-model-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --checkpoint-dir /tmp/generals-league-p0 \
  --checkpoint-every 50 \
  --keep-checkpoints 6 \
  --model-path /tmp/generals-ppo-8x8-league-p0.eqx
```

Add evaluation command:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_league.py /tmp/generals-ppo-8x8-league-p0.eqx \
  --checkpoint-opponent v5=generals-ppo-8x8-expander-gpu-v5.eqx:sample \
  --num-games 1024 \
  --json-output /tmp/generals-ppo-8x8-league-p0-league.json
```

- [ ] **Step 3: Append status log**

Run `git status --short`. If tracked files changed, append an entry to `statusquo.md` with the implemented files, verification result, and next training step.

- [ ] **Step 4: Commit Task 4**

```bash
git add docs/expander-training-strategy.md statusquo.md
git commit -m "docs: add league training workflow"
```

## Task 5: First League Training Run

**Files:**
- Modify: `docs/expander-training-strategy.md`
- Modify: `statusquo.md`

- [ ] **Step 1: Train player 0 candidate**

Run:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000001 \
  --truncation 500 \
  --init-model-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-pool generals-ppo-8x8-expander-gpu-v2.eqx,generals-ppo-8x8-expander-gpu-v3.eqx,generals-ppo-8x8-expander-gpu-v4.eqx,generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-pool-modes sample,sample,sample,sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --checkpoint-dir /tmp/generals-league-p0 \
  --checkpoint-every 50 \
  --keep-checkpoints 8 \
  --model-path /tmp/generals-ppo-8x8-league-p0-v1.eqx \
  --seed 30200
```

- [ ] **Step 2: Evaluate player 0 candidate**

Run:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_league.py /tmp/generals-ppo-8x8-league-p0-v1.eqx \
  --checkpoint-opponent v5=generals-ppo-8x8-expander-gpu-v5.eqx:sample \
  --num-games 1024 \
  --seed 30300 \
  --json-output /tmp/generals-ppo-8x8-league-p0-v1-league.json
```

- [ ] **Step 3: Train player 1 candidate if player 0 result improves league score**

Use the same command as Step 1 with:

```text
--learner-player 1
--checkpoint-dir /tmp/generals-league-p1
--model-path /tmp/generals-ppo-8x8-league-p1-v1.eqx
--seed 30250
```

- [ ] **Step 4: Document results**

Add the training commands, league score, per-opponent minimum, and best checkpoint path to `docs/expander-training-strategy.md`.

- [ ] **Step 5: Append status and commit docs**

```bash
git add docs/expander-training-strategy.md statusquo.md
git commit -m "docs: record first league training results"
git push
```

## Self-Review

- Spec coverage: league evaluation, checkpoint saving, ordinary opponent pools, promotion gates, docs, and first training run are covered.
- Placeholder scan: the plan contains no TBD markers or open placeholder instructions.
- Type consistency: checkpoint spec tuples use `(name, path, mode)`, pool specs use `(path, mode)`, and summary rows use `LeagueRow`.
