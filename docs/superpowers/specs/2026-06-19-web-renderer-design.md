# Web Renderer Design

## Goal

Replace the current pygame-only visual interface with a browser-rendered game UI while keeping the existing Python/JAX game and policy stack authoritative.

The web renderer must support the two current GUI workflows:

- Human vs PPO checkpoint, including source selection, target selection, split moves, pass, restart, auto tick, and AI preview.
- PPO vs PPO watch mode, including independent model paths, policy input modes, rollout-search options, auto tick, restart, and AI preview.

The browser UI should initially match the current pygame board and right-side HUD semantics instead of redesigning the game. The goal is remote usability and maintainable rendering, not a new game rule implementation.

## Current Constraints

The current playable GUI is not only a renderer:

- `examples/play_against_model.py` owns game setup, checkpoint agent creation, auto tick, action composition, AI preview, restart, and the main loop.
- `generals/gui/rendering.py` owns pygame drawing for the board, HUD, selection highlights, legal target highlights, policy arrows, and policy preview text.
- `generals/gui/event_handler.py` owns pygame keyboard and mouse translation into semantic game commands.
- `generals/core/rendering.py` already adapts `GameState` and `GameInfo` into renderer-friendly numpy arrays.
- `generals/agents/ppo_policy_agent.py` already exposes `PolicyPreview` and `PolicyActionCandidate`, which are suitable for JSON serialization.

The action format must remain:

```text
[pass, row, col, direction, split]
```

where `direction` is `0=up`, `1=down`, `2=left`, and `3=right`.

The browser must not become a second game engine. Python remains authoritative for move validation, simultaneous action ordering, fog-of-war, model inference, rollout search, terminal detection, and restart behavior.

## Recommended Approach

Add a Python web session layer and a browser Canvas renderer.

Use a single-process FastAPI application for the first implementation:

- `FastAPI` hosts the API and static frontend.
- `WebSocket` endpoints accept JSON client commands and send JSON snapshots.
- `StaticFiles` serves the compiled or static web assets.

DeepWiki verification for `fastapi/fastapi` confirms the relevant primitives:

- `@app.websocket(...)` defines WebSocket endpoints.
- `await websocket.accept()` accepts a connection.
- `await websocket.receive_json()` receives parsed client commands.
- `await websocket.send_json(data)` sends server snapshots.
- `WebSocketDisconnect` is the normal disconnect cleanup signal.
- `StaticFiles(directory=...)` can be mounted to serve browser assets.

This is intentionally narrower than a distributed multiplayer service. The first web UI should be a local or LAN-accessible single-session tool that removes the need for X11, VNC, or a local pygame window.

## Non-Goals

This design does not attempt to:

- Port JAX, Equinox checkpoints, or rollout search into WebAssembly.
- Replace the existing pygame GUI immediately.
- Support multiple concurrent public games with durable server-side persistence.
- Connect browser clients directly to generals.io.
- Implement authentication, user accounts, or match history storage.
- Redesign the visual identity beyond a faithful browser version of the current board and HUD.

## Backend Components

### `generals/web/session.py`

Add a `WebGameSession` class that owns one running match.

Responsibilities:

- Parse and store launch configuration equivalent to the current playable CLI settings.
- Create the initial map and `GameState`.
- Build PPO or rollout-search agents with existing `make_gui_agent` helpers where possible.
- Maintain the PRNG key, `step_count`, `last_tick`, terminal reporting state, selected cell, split flag, and last message.
- Expose `restart()`, `submit_client_command(command)`, `tick(now)`, and `snapshot()` methods.
- Serialize all state mutations through one session lock so WebSocket receive and auto tick cannot update the same state concurrently.

The session layer should be extracted from `examples/play_against_model.py` without breaking that script. The CLI can keep using the old pygame loop during the first phase, while the web server imports the same shared helpers.

### `generals/web/schemas.py`

Define stable JSON-facing data structures. Dataclasses are sufficient for the first pass; Pydantic can be added if strict validation becomes useful.

Core schema groups:

- `GameSnapshot`
- `GridSnapshot`
- `PlayerSnapshot`
- `PolicyPreviewSnapshot`
- `PolicyCandidateSnapshot`
- `ClientCommand`
- `SessionConfig`

All numpy and JAX arrays must be converted to ordinary Python lists and scalars before sending.

### `generals/web/server.py`

Add a small FastAPI app.

Routes:

- `GET /` returns the web UI entry point.
- `GET /healthz` returns a basic health response.
- `GET /api/default-config` returns default launch settings for the frontend.
- `WebSocket /ws/game` starts or attaches to one session and streams snapshots.

The first implementation can run one session per WebSocket connection. Shared watch sessions can come later if needed.

### `examples/play_web.py`

Add a CLI entry point that mirrors the most important `play_against_model.py` options and starts the web server.

Initial command shape:

```bash
uv run python examples/play_web.py generals-ppo-8x8-expander-gpu-v5.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --human-player 0 \
  --auto-tick \
  --tick-rate 2 \
  --preview-top-k 3
```

Machine watch mode should follow the existing script:

```bash
uv run python examples/play_web.py \
  --machine-vs-machine \
  --model-0-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --model-1-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --tick-rate 4
```

The script prints the local URL after startup.

## Snapshot Protocol

The server sends a complete snapshot after connection, restart, every accepted command, and every auto tick step.

Example:

```json
{
  "type": "snapshot",
  "mode": "human-vs-model",
  "grid": {
    "height": 8,
    "width": 8,
    "armies": [[1, 0], [0, 1]],
    "ownership": [[0, -1], [-1, 1]],
    "mountains": [[false, false], [false, false]],
    "cities": [[false, false], [false, false]],
    "generals": [[true, false], [false, true]],
    "visible": [[true, true], [true, false]]
  },
  "players": [
    {"index": 0, "name": "Human", "army": 5, "land": 1, "color": "#dc3737"},
    {"index": 1, "name": "PPO Model", "army": 3, "land": 1, "color": "#285adc"}
  ],
  "time": 2,
  "step_count": 1,
  "winner": null,
  "game_done": false,
  "selected_cell": [0, 0],
  "valid_targets": [[0, 1], [1, 0]],
  "split_enabled": false,
  "last_message": "Selected: (0, 0)",
  "auto_tick": {"enabled": true, "tick_rate": 2.0},
  "policy_preview": null
}
```

Ownership is encoded as:

```text
-1: neutral or unowned
 0: player 0
 1: player 1
```

This avoids shipping a `(2, H, W)` boolean ownership tensor to the frontend.

Fog handling:

- Human-vs-model mode defaults to the human player's visibility.
- Machine-vs-machine mode defaults to full visibility for inspection.
- A later option can allow player-0, player-1, or full-map visibility in watch mode.

## Client Command Protocol

The frontend sends semantic commands, not raw JAX arrays.

Commands:

```json
{"type": "select", "row": 2, "col": 3}
{"type": "move", "source": [2, 3], "target": [2, 4], "split": false}
{"type": "pass"}
{"type": "cancel"}
{"type": "set_split", "enabled": true}
{"type": "restart"}
{"type": "set_auto_tick", "enabled": true, "tick_rate": 2.0}
{"type": "set_visibility", "mode": "human"}
```

The backend validates all commands against the current `GameState`.

For a move command, the backend computes the direction from source and target, rejects non-adjacent targets, rejects mountain targets, rejects invalid sources, and then constructs the existing public action. The frontend may highlight likely valid targets, but server validation remains authoritative.

## Frontend Components

Use a lightweight browser app with Canvas rendering. The first version can use vanilla TypeScript or plain modern JavaScript. React is not required for the board itself because Canvas drawing is imperative and compact. If a bundler is added, Vite plus TypeScript is the preferred small frontend stack.

Components:

- `GameClient`: WebSocket lifecycle, reconnect display, command send queue, snapshot store.
- `BoardCanvas`: draws the grid, fog, ownership colors, terrain, army numbers, selection, valid targets, and AI preview arrows.
- `HudPanel`: draws or renders DOM HUD for players, army, land, time, speed, selected cell, split state, terminal winner, and policy candidates.
- `Controls`: pass, split toggle, restart, auto tick toggle, tick-rate control, visibility selector for watch mode.
- `CoordinateMapper`: converts pointer coordinates into board cells and keeps resize behavior stable.

The board should use the existing asset semantics:

- General: crown image where available, otherwise a clear icon asset.
- City: current city image where available.
- Mountain: current mountain image where available.
- Font: load bundled Quicksand where feasible, with browser fallbacks.

Assets should be copied or served from `generals/assets` through the static web route. Do not hand-draw replacement assets when real project assets already exist.

## Visual Behavior

The web UI should preserve the current gameplay affordances:

- Selected source cell uses a yellow highlight.
- Legal adjacent targets use a green highlight.
- AI preview primary candidate uses a stronger blue arrow and border.
- Secondary preview candidates use lighter blue arrows.
- Queued human moves use green arrows with sequence badges.
- Right-side HUD lists queued moves and exposes undo/clear controls.
- Right-side HUD lists player stats and top policy candidates.
- Sample mode explicitly indicates that the displayed distribution is not necessarily the sampled action.
- Keyboard input follows generals.io core bindings: WASD/arrows move, Space deselects, Z toggles split, E undoes the last queued move, and Q clears a non-empty queue or queues pass when empty.

The layout should be functional and dense:

- Board on the left.
- HUD and controls in a fixed-width right panel on desktop.
- On narrow screens, board remains first and controls move below it.
- Board cells keep a square aspect ratio.
- Text must not resize the board or shift cell geometry.

## Auto Tick And Concurrency

The backend session owns time.

For each session:

- A background async task can check `tick_rate` and call `session.tick(...)`.
- Client commands and auto tick must acquire the same session lock.
- If a human source cell is selected, idle auto-pass pauses, matching the current pygame behavior.
- Machine-vs-machine mode ticks only when auto tick is enabled and the tick interval is due.
- After every state change, the server sends one snapshot.

Long model calls, especially rollout search, must not allow overlapping state updates. A simple per-session `asyncio.Lock` is enough for the first local server.

## Error Handling

Backend errors:

- Missing model path fails at server startup with a clear CLI message.
- Bad checkpoint shape fails during session creation with the existing policy-input error text where possible.
- Invalid client commands send an error snapshot field and preserve the current state.
- WebSocket disconnect cancels that session's tick task.
- A model action exception sends an error message and pauses auto tick.

Frontend errors:

- Disconnected state shows an unobtrusive overlay and disables controls.
- Invalid command responses update `last_message`.
- Slow initial model load shows a loading state instead of a blank board.

## Migration Plan

### Phase 1: Extract Session Logic

Move the non-pygame portions of `examples/play_against_model.py` into reusable helpers:

- New game creation.
- Agent construction.
- Human action selection.
- Machine action selection.
- AI preview generation.
- Restart and terminal handling.

Existing pygame scripts must keep working.

### Phase 2: Snapshot Serialization

Add snapshot generation from `GameState`, `GameInfo`, UI selection state, and optional policy preview.

Unit tests should assert exact small-board snapshots for ownership, visibility, selected cells, valid targets, and policy preview candidates.

### Phase 3: WebSocket Watch MVP

Implement `examples/play_web.py` and `generals/web/server.py`.

First acceptance:

- Launch server.
- Open browser.
- Watch PPO 0 vs PPO 1 auto-tick on an 8x8 generated map.
- Restart after terminal state or after an explicitly configured max-step limit.

### Phase 4: Human Interaction

Add browser commands for select, move, pass, split, cancel, restart, and auto tick settings.

Behavior should match the existing playable input tests:

- Source click selects only owned cells with more than one army.
- Adjacent target click appends the correct direction to the move queue.
- Queued moves can chain from projected targets before earlier moves execute.
- Split toggle applies to the next move.
- Pass does not require a selection and is queued like a move.
- Undo and clear operate on the pending queue without advancing game time.
- Invalid target keeps the source selected.

### Phase 5: AI Preview Overlay

Serialize `PolicyPreview` and draw:

- Top-K candidate list.
- Move arrows and source/target borders.
- Pass candidates in the HUD only.
- Value estimate.
- Sample mode note.

### Phase 6: Documentation And Launch Scripts

Update README and `docs/zh-manual.md` with web launch commands. Add optional shell wrappers only after the Python entry point is stable.

## Testing

Backend tests:

- Snapshot serializer returns JSON-safe Python values.
- Ownership encoding is `-1`, `0`, `1`.
- Visibility differs between human mode and full watch mode.
- `select` accepts valid sources and rejects invalid sources.
- `move` maps source and target to a queued public action without advancing time until tick.
- `set_split`, `pass`, `cancel`, and `restart` update session state correctly.
- Auto tick consumes one queued human action when available, pauses while a source is selected and the queue is empty, and otherwise auto-passes.
- Machine-vs-machine tick calls both agents.
- Policy preview DTO preserves pass, move, probability, direction label, split flag, and value.

Frontend tests:

- Pointer coordinates map to the correct row and column.
- Board draw code receives stable dimensions for 8x8 and 10x10 maps.
- Valid target overlays and policy arrows are generated from snapshot data.
- Controls send the expected JSON commands.

End-to-end smoke tests:

- Start the web server on a test port.
- Open the page with Playwright.
- Confirm the canvas is nonblank.
- In human mode, click source and target and observe `time` advance.
- In watch mode, observe auto tick advancing without user input.

Verification commands for implementation work:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_web_session.py tests/test_web_protocol.py
JAX_PLATFORMS=cpu uv run python -m compileall generals examples tests
git diff --check
```

When frontend tooling is added, include the matching package-manager checks, for example:

```bash
npm run build
npm run test
```

## Dependencies

Add web dependencies only when implementation begins:

- `fastapi`
- `uvicorn`

If the frontend uses a build step, add the smallest practical Node stack and keep generated build artifacts out of git unless the project explicitly decides to ship static assets from source control.

## Acceptance Criteria

The web renderer is accepted when:

- `examples/play_web.py` starts a local browser-accessible server.
- PPO vs PPO watch mode renders a live game without any local GUI display.
- Human vs PPO mode supports click-to-move, split, pass, restart, and auto tick.
- AI preview candidates render in the HUD and on the board.
- The existing pygame scripts still work.
- Backend session and protocol tests pass.
- Browser smoke verification confirms a nonblank, interactive board.

## Implementation Notes

Keep pygame and web UI paths side by side until the browser path reaches feature parity. The best long-term boundary is:

```text
Game/PPO logic -> WebGameSession -> JSON snapshot -> Browser renderer
```

Pygame should remain a consumer of the same game logic, not the owner of it. This prevents future rendering work from changing game rules or policy behavior.
