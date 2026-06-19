import sys
import importlib
from pathlib import Path

from examples.play_web import args_to_config, parse_args
from generals.web.server import create_app
from generals.web.session import WebSessionConfig


def test_create_app_exposes_health_config_root_and_websocket_routes():
    app = create_app(default_config=WebSessionConfig(model_path="model.eqx"))

    paths = {getattr(route, "path", None) for route in app.routes}

    assert "/" in paths
    assert "/healthz" in paths
    assert "/api/default-config" in paths
    assert "/ws/game" in paths


def test_parse_web_args_accepts_machine_model_paths(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "play_web.py",
            "--machine-vs-machine",
            "--model-0-path",
            "p0.eqx",
            "--model-1-path",
            "p1.eqx",
            "--tick-rate",
            "4",
        ],
    )

    args = parse_args()
    config = args_to_config(args)

    assert config.machine_vs_machine is True
    assert config.model_path == "p0.eqx"
    assert config.model_0_path == "p0.eqx"
    assert config.model_1_path == "p1.eqx"
    assert config.tick_rate == 4.0


def test_parse_web_args_accepts_human_model_path_and_server_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "play_web.py",
            "policy.eqx",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--human-player",
            "1",
            "--no-auto-tick",
            "--preview-top-k",
            "5",
        ],
    )

    args = parse_args()
    config = args_to_config(args)

    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert config.model_path == "policy.eqx"
    assert config.machine_vs_machine is False
    assert config.human_player == 1
    assert config.auto_tick is False
    assert config.preview_top_k == 5


def test_static_browser_assets_and_real_asset_mounts_exist():
    app = create_app(default_config=WebSessionConfig(model_path="model.eqx"))
    paths = {getattr(route, "path", None) for route in app.routes}
    static_root = Path("generals/web/static")

    assert (static_root / "index.html").is_file()
    assert (static_root / "styles.css").is_file()
    assert (static_root / "app.js").is_file()
    assert "/static" in paths
    assert "/assets/images" in paths
    assert "/assets/fonts" in paths


def test_web_runtime_factory_import_does_not_import_pygame():
    sys.modules.pop("pygame", None)

    runtime = importlib.import_module("generals.agents.ppo_runtime")
    importlib.reload(runtime)

    assert "pygame" not in sys.modules
