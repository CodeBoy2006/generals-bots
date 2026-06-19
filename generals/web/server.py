"""FastAPI server for the browser renderer."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .session import WebGameSession, WebSessionConfig

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(default_config: WebSessionConfig) -> FastAPI:
    """Create the web renderer app without starting a game session."""
    app = FastAPI(title="Generals Bots Web Renderer")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/default-config")
    def default_config_route() -> dict:
        return asdict(default_config)

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<!doctype html><title>Generals Web</title><main id=\"app\">Generals web renderer</main>"
        )

    @app.websocket("/ws/game")
    async def websocket_game(websocket: WebSocket) -> None:
        await websocket.accept()
        session = WebGameSession.from_config(default_config)
        await websocket.send_json(session.snapshot())
        try:
            while True:
                timeout = max(0.05, 1.0 / session.tick_rate) if session.auto_tick_enabled else None
                if timeout is None:
                    command = await websocket.receive_json()
                    snapshot = session.submit_client_command(command)
                else:
                    try:
                        command = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
                    except TimeoutError:
                        snapshot = session.tick(time.monotonic())
                    else:
                        snapshot = session.submit_client_command(command)
                await websocket.send_json(snapshot)
        except WebSocketDisconnect:
            return

    return app


def run_server(default_config: WebSessionConfig, host: str, port: int) -> None:
    """Run the web renderer with uvicorn."""
    import uvicorn

    app = create_app(default_config)
    print(f"Generals web renderer: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
