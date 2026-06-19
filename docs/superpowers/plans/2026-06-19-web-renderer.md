# Web Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-rendered Generals UI that keeps Python/JAX authoritative for game rules and PPO inference while retaining the existing pygame GUI.

**Architecture:** Add a `generals.web` package containing JSON-safe schemas, snapshot serialization, an authoritative `WebGameSession`, and a FastAPI WebSocket/static-file server. Add a small static browser app that draws the board with Canvas, sends semantic commands, and renders the existing HUD/AI-preview concepts.

**Tech Stack:** Python 3.11+, JAX, existing PPO helpers, FastAPI, Uvicorn, browser Canvas, plain JavaScript, pytest, Playwright/manual browser smoke for final UI behavior.

---

## File Structure

- Create `generals/web/__init__.py`: exports web session and schema helpers.
- Create `generals/web/schemas.py`: dataclasses and helpers for JSON-safe snapshots and client commands.
- Create `generals/web/session.py`: authoritative game session state, command validation, tick handling, and snapshot production.
- Create `generals/web/server.py`: FastAPI app factory, WebSocket loop, static file serving, and CLI server runner.
- Create `generals/web/static/index.html`: browser shell with board, HUD, controls, and status regions.
- Create `generals/web/static/styles.css`: dense game UI layout and responsive board/HUD styling.
- Create `generals/web/static/app.js`: WebSocket client, Canvas renderer, controls, pointer mapping, and command sender.
- Create `examples/play_web.py`: command-line entry point that mirrors current GUI launch options and starts the web server.
- Create `tests/test_web_protocol.py`: snapshot serialization and DTO tests.
- Create `tests/test_web_session.py`: session command, tick, and action tests.
- Modify `pyproject.toml`: add `fastapi` and `uvicorn` runtime dependencies.
- Modify `requirements.txt`: mirror runtime dependency additions for pip users.
- Modify `README.md`: document browser UI launch commands.
- Modify `docs/zh-manual.md`: document browser UI launch commands in Chinese.
- Modify `statusquo.md`: append implementation ledger entries after substantive work.

## Task 1: Protocol And Snapshot Serialization

**Files:**
- Create: `tests/test_web_protocol.py`
- Create: `generals/web/__init__.py`
- Create: `generals/web/schemas.py`

- [ ] **Step 1: Write failing protocol tests**

Add tests that construct a tiny 4x4 `GameState`, serialize it, and assert JSON-safe ownership, visibility, players, selected cells, valid targets, and policy preview candidates.

```python
def test_snapshot_serializes_grid_players_and_human_visibility():
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)._replace(armies=game.create_initial_state(grid).armies.at[0, 0].set(5))
    info = game.get_info(state)

    snapshot = build_snapshot(
        state=state,
        info=info,
        names=["Human", "PPO Model"],
        colors=["#dc3737", "#285adc"],
        mode="human-vs-model",
        visibility_player=0,
        step_count=0,
        selected_cell=(0, 0),
        split_enabled=False,
        last_message="Selected: (0, 0)",
        auto_tick_enabled=True,
        tick_rate=2.0,
        policy_preview=None,
        valid_targets=[(0, 1), (1, 0)],
        reached_limit=False,
    )

    assert snapshot["grid"]["ownership"][0][0] == 0
    assert snapshot["grid"]["ownership"][3][3] == 1
    assert snapshot["grid"]["ownership"][1][1] == -1
    assert snapshot["grid"]["visible"][3][3] is False
    assert snapshot["players"][0]["name"] == "Human"
    assert snapshot["selected_cell"] == [0, 0]
    assert snapshot["valid_targets"] == [[0, 1], [1, 0]]
```

- [ ] **Step 2: Run protocol tests to verify RED**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_web_protocol.py
```

Expected: fail with `ModuleNotFoundError: No module named 'generals.web'`.

- [ ] **Step 3: Implement minimal schema helpers**

Implement `build_snapshot`, `serialize_policy_preview`, `encode_ownership`, `visibility_for_mode`, and scalar/list conversion helpers in `generals/web/schemas.py`.

- [ ] **Step 4: Run protocol tests to verify GREEN**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_web_protocol.py
```

Expected: all tests in `tests/test_web_protocol.py` pass.

- [ ] **Step 5: Commit protocol layer**

```bash
git add generals/web/__init__.py generals/web/schemas.py tests/test_web_protocol.py
git commit -m "feat: add web snapshot protocol"
```

## Task 2: Web Game Session

**Files:**
- Create: `tests/test_web_session.py`
- Create: `generals/web/session.py`
- Modify: `generals/web/__init__.py`

- [ ] **Step 1: Write failing session tests**

Add tests with lightweight fixed agents so checkpoint loading is not required.

```python
class FixedAgent:
    def __init__(self, action):
        self.action = jnp.array(action, dtype=jnp.int32)

    def act(self, observation, key):
        return self.action


def test_select_and_move_command_updates_state_time_and_message():
    session = WebGameSession.for_testing(
        grid=_basic_grid(),
        names=["Human", "PPO Model"],
        agents=(FixedAgent([1, 0, 0, 0, 0]),),
        human_player=0,
    )
    session.state = session.state._replace(armies=session.state.armies.at[0, 0].set(5))

    select_snapshot = session.submit_client_command({"type": "select", "row": 0, "col": 0})
    assert select_snapshot["selected_cell"] == [0, 0]

    move_snapshot = session.submit_client_command({"type": "move", "source": [0, 0], "target": [0, 1], "split": False})
    assert move_snapshot["time"] == 3
    assert move_snapshot["selected_cell"] is None
    assert move_snapshot["last_message"] == "Move queued"
```

Also test invalid source, invalid target, split, pass, restart, auto tick pause while selected, and machine-vs-machine ticking.

- [ ] **Step 2: Run session tests to verify RED**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_web_session.py
```

Expected: fail because `generals.web.session.WebGameSession` does not exist.

- [ ] **Step 3: Implement `WebGameSession`**

Implement:

- `WebSessionConfig`
- `WebGameSession.__init__`
- `WebGameSession.for_testing`
- `WebGameSession.new_game`
- `WebGameSession.snapshot`
- `WebGameSession.submit_client_command`
- `WebGameSession.tick`
- source/target validation helpers
- action creation from semantic commands

Use existing helpers from `examples.play_against_model` for `auto_tick_due`, `choose_agent_action`, `choose_human_action`, `choose_machine_actions`, `advance_until_human_can_move`, `explain_agent`, `make_grid`, `make_gui_agent`, `make_player_names`, and `make_search_config` where practical.

- [ ] **Step 4: Run session tests to verify GREEN**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_web_session.py
```

Expected: all session tests pass.

- [ ] **Step 5: Commit session layer**

```bash
git add generals/web/session.py generals/web/__init__.py tests/test_web_session.py
git commit -m "feat: add web game session"
```

## Task 3: Web Server And CLI

**Files:**
- Create: `generals/web/server.py`
- Create: `examples/play_web.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `tests/test_play_web_cli.py`

- [ ] **Step 1: Write failing server and CLI tests**

Test that the app factory exposes health and default config without loading a checkpoint, and that CLI argument parsing mirrors key current options.

```python
def test_create_app_health_endpoint():
    app = create_app(default_config=WebSessionConfig(model_path="model.eqx"))
    routes = {getattr(route, "path", None) for route in app.routes}
    assert "/healthz" in routes
    assert "/ws/game" in routes


def test_parse_web_args_accepts_machine_models(monkeypatch):
    monkeypatch.setattr("sys.argv", ["play_web.py", "--machine-vs-machine", "--model-0-path", "p0.eqx", "--model-1-path", "p1.eqx"])
    args = parse_args()
    assert args.machine_vs_machine is True
    assert args.model_path == "p0.eqx"
    assert args.opponent_model_path == "p1.eqx"
```

- [ ] **Step 2: Run server tests to verify RED**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_play_web_cli.py
```

Expected: fail because `generals.web.server` and `examples.play_web` do not exist.

- [ ] **Step 3: Add dependencies and implement server**

Add to `pyproject.toml` and `requirements.txt`:

```text
fastapi>=0.115.0
uvicorn>=0.30.0
```

Implement `create_app`, `websocket_game`, static mounts, and `run_server`. Keep app creation lazy so tests can inspect routes without loading a checkpoint.

- [ ] **Step 4: Implement `examples/play_web.py`**

Expose `parse_args`, `args_to_config`, and `main`. Reuse `examples.play_against_model.parse_args` behavior where possible, but avoid importing pygame in server startup paths.

- [ ] **Step 5: Run server tests to verify GREEN**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_play_web_cli.py tests/test_web_session.py tests/test_web_protocol.py
```

Expected: tests pass.

- [ ] **Step 6: Commit server and CLI**

```bash
git add pyproject.toml requirements.txt generals/web/server.py examples/play_web.py tests/test_play_web_cli.py
git commit -m "feat: add web server entry point"
```

## Task 4: Browser Static UI

**Files:**
- Create: `generals/web/static/index.html`
- Create: `generals/web/static/styles.css`
- Create: `generals/web/static/app.js`
- Modify: `generals/web/server.py`

- [ ] **Step 1: Write static asset tests**

Add tests to `tests/test_play_web_cli.py` that assert the static files exist and that the server has a root route.

```python
def test_static_web_assets_exist():
    root = Path("generals/web/static")
    assert (root / "index.html").is_file()
    assert (root / "styles.css").is_file()
    assert (root / "app.js").is_file()
```

- [ ] **Step 2: Run static asset test to verify RED**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_play_web_cli.py::test_static_web_assets_exist
```

Expected: fail because static files do not exist.

- [ ] **Step 3: Implement browser UI**

Create a dense single-screen app:

- Canvas board with square cells.
- Right HUD for players, time, selected cell, split state, messages, controls, and AI preview.
- Buttons for pass and restart.
- Checkbox for split.
- Checkbox and numeric input for auto tick.
- WebSocket connection to `/ws/game`.
- Pointer-to-cell mapping and semantic `select` / `move` commands.
- Canvas drawing for terrain, fog, army text, selection, valid targets, and policy arrows.

- [ ] **Step 4: Run static asset tests to verify GREEN**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_play_web_cli.py
```

Expected: tests pass.

- [ ] **Step 5: Commit static UI**

```bash
git add generals/web/static/index.html generals/web/static/styles.css generals/web/static/app.js generals/web/server.py tests/test_play_web_cli.py
git commit -m "feat: add browser game UI"
```

## Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/zh-manual.md`
- Modify: `statusquo.md`

- [ ] **Step 1: Write launch documentation**

Add a `Browser Web UI` section to README and Chinese manual:

```bash
uv run python examples/play_web.py generals-ppo-8x8-expander-gpu-v5.eqx
uv run python examples/play_web.py --machine-vs-machine --model-0-path generals-ppo-8x8-expander-gpu-v5.eqx --model-1-path generals-ppo-8x8-expander-gpu-v5.eqx
```

Document that the browser UI does not need X11/pygame display, but still runs Python/JAX on the server.

- [ ] **Step 2: Append status ledger**

Append an entry to `statusquo.md` summarizing the implementation and remaining verification status.

- [ ] **Step 3: Run documentation checks**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/zh-manual.md statusquo.md
git commit -m "docs: document web renderer"
```

## Task 6: Full Verification And Push

**Files:**
- No new files unless verification exposes defects.

- [ ] **Step 1: Run targeted backend tests**

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_web_protocol.py tests/test_web_session.py tests/test_play_web_cli.py
```

Expected: all targeted web tests pass.

- [ ] **Step 2: Run existing compatibility tests**

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_play_against_model_cli.py tests/test_playable_game_input.py tests/test_playable_rendering.py tests/test_launch_script.py
```

Expected: existing pygame/playable CLI tests still pass.

- [ ] **Step 3: Run compile and whitespace checks**

```bash
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
```

Expected: compileall succeeds and diff check emits no errors.

- [ ] **Step 4: Run web server smoke check**

Start a server with a short timeout and confirm startup imports do not fail:

```bash
timeout 8 uv run python examples/play_web.py generals-ppo-8x8-expander-gpu-v5.eqx --host 127.0.0.1 --port 8765
```

Expected: process starts and prints the local URL before timeout stops it. If `timeout` returns 124 after successful startup, treat that as expected for this smoke command.

- [ ] **Step 5: Check final git state**

```bash
git status
git log --oneline -5
```

Expected: all intended commits are present and the worktree is clean.

- [ ] **Step 6: Push**

```bash
git push
```

Expected: branch pushes to `origin/master`.
