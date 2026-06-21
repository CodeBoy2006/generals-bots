import sys
import importlib
from pathlib import Path

from examples.play_web import args_to_config, build_model_catalog, parse_args
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
    assert config.max_steps is None
    assert config.preview_top_k == 5


def test_parse_web_args_accepts_optional_max_steps_limit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["play_web.py", "policy.eqx", "--max-steps", "42"])

    args = parse_args()
    config = args_to_config(args)

    assert config.max_steps == 42


def test_parse_web_args_accepts_adaptive_champion_runtime_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "play_web.py",
            "runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx",
            "--adaptive-policy",
            "--network-arch",
            "unet",
            "--channels",
            "64,96,128,64",
            "--pad-to",
            "16",
            "--global-context",
            "--scoreboard-history",
            "--fog-memory",
            "--value-loss",
            "hl-gauss",
            "--policy-adapter-path",
            "runs/adaptive-online-search-conversion-adapter-v1/generals-adaptive-online-search-conversion-adapter-v1.eqx",
            "--policy-adapter-scale",
            "1",
            "--policy-adapter-mode",
            "replace",
            "--policy-adapter-max-grid-size",
            "8",
            "--adaptive-online-search",
            "--search-top-k",
            "4",
            "--search-rollout-steps",
            "16",
            "--search-rollouts-per-action",
            "2",
            "--online-search-min-turn",
            "80",
            "--online-search-require-contact",
            "--adaptive-online-search-opponent-path",
            "generals-ppo-8x8-expander-gpu-v5.eqx",
            "--adaptive-online-search-opponent-channels",
            "32,32,32,16",
            "--adaptive-online-search-opponent-input-channels",
            "9",
        ],
    )

    args = parse_args()
    config = args_to_config(args)

    assert config.adaptive_policy is True
    assert config.network_arch == "unet"
    assert config.channels == "64,96,128,64"
    assert config.pad_to == 16
    assert config.global_context is True
    assert config.scoreboard_history is True
    assert config.fog_memory is True
    assert config.value_loss == "hl-gauss"
    assert config.policy_adapter_path.endswith("generals-adaptive-online-search-conversion-adapter-v1.eqx")
    assert config.policy_adapter_scale == 1.0
    assert config.policy_adapter_mode == "replace"
    assert config.policy_adapter_max_grid_size == 8
    assert config.adaptive_online_search is True
    assert config.search_top_k == 4
    assert config.search_rollouts_per_action == 2
    assert config.online_search_min_turn == 80
    assert config.online_search_require_contact is True
    assert config.adaptive_online_search_opponent_path == "generals-ppo-8x8-expander-gpu-v5.eqx"
    assert config.adaptive_online_search_opponent_channels == "32,32,32,16"
    assert config.adaptive_online_search_opponent_input_channels == 9


def test_parse_web_args_treats_zero_max_steps_as_unlimited(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["play_web.py", "policy.eqx", "--max-steps", "0"])

    args = parse_args()
    config = args_to_config(args)

    assert config.max_steps is None


def test_build_model_catalog_includes_explicit_and_discovered_checkpoints(tmp_path):
    explicit = tmp_path / "explicit.eqx"
    explicit.write_text("model", encoding="utf-8")
    root_model = tmp_path / "root.eqx"
    root_model.write_text("model", encoding="utf-8")
    legacy_dir = tmp_path / "legacymodels" / "nested"
    legacy_dir.mkdir(parents=True)
    legacy_model = legacy_dir / "legacy.eqx"
    legacy_model.write_text("model", encoding="utf-8")

    catalog = build_model_catalog([str(explicit)], repo_root=tmp_path)

    assert catalog == [
        {"id": str(explicit), "label": "explicit.eqx", "path": str(explicit)},
        {"id": str(root_model), "label": "root.eqx", "path": str(root_model)},
        {"id": str(legacy_model), "label": "legacy.eqx", "path": str(legacy_model)},
    ]


def test_static_browser_assets_and_real_asset_mounts_exist():
    app = create_app(default_config=WebSessionConfig(model_path="model.eqx"))
    paths = {getattr(route, "path", None) for route in app.routes}
    static_root = Path("generals/web/static")

    assert (static_root / "index.html").is_file()
    assert (static_root / "styles.css").is_file()
    assert (static_root / "keyboard.js").is_file()
    assert (static_root / "app.js").is_file()
    assert "/static" in paths
    assert "/assets/images" in paths
    assert "/assets/fonts" in paths


def test_web_runtime_factory_import_does_not_import_pygame():
    sys.modules.pop("pygame", None)

    runtime = importlib.import_module("generals.agents.ppo_runtime")
    importlib.reload(runtime)

    assert "pygame" not in sys.modules
