"""Run the browser-based Generals UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generals.agents.ppo_policy_agent import POLICY_INPUT_CHOICES, policy_input_default_channels
from generals.web.server import run_server
from generals.web.session import WebSessionConfig


def build_model_catalog(explicit_paths: list[str | None], repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    """Return selectable PPO checkpoints from explicit CLI paths and local model folders."""
    seen: set[str] = set()
    catalog: list[dict[str, str]] = []

    def add_model(path: Path) -> None:
        model_id = str(path)
        if model_id in seen:
            return
        seen.add(model_id)
        catalog.append({"id": model_id, "label": path.name, "path": model_id})

    for explicit_path in explicit_paths:
        if explicit_path:
            add_model(Path(explicit_path))

    for path in sorted(repo_root.glob("*.eqx")):
        add_model(path)
    legacy_root = repo_root / "legacymodels"
    if legacy_root.exists():
        for path in sorted(legacy_root.rglob("*.eqx")):
            add_model(path)

    return catalog


def _resolve_alias(parser: argparse.ArgumentParser, primary_name: str, primary, alias_name: str, alias, default):
    if primary is not None and alias is not None and primary != alias:
        parser.error(f"pass either {primary_name} or {alias_name}, not both")
    if primary is not None:
        return primary
    if alias is not None:
        return alias
    return default


def _resolve_input_channels(policy_input: str, input_channels: int | None) -> int | None:
    if input_channels is not None:
        return input_channels
    if policy_input == "auto":
        return None
    return policy_input_default_channels(policy_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a browser-based Generals game UI.")
    parser.add_argument("model_path", nargs="?", help="Primary saved Equinox .eqx PPO checkpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-0-path", default=None)
    parser.add_argument("--model-1-path", default=None)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--opponent-policy-mode", choices=("greedy", "sample"), default=None)
    parser.add_argument("--machine-vs-machine", action="store_true")
    parser.add_argument("--human-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--auto-tick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tick-rate", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps before ending the browser match. Use 0 or omit for no limit.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--preview-top-k", type=int, default=3)
    parser.add_argument("--no-ai-preview", dest="ai_preview", action="store_false")
    parser.add_argument("--search-policy", action="store_true")
    parser.add_argument("--opponent-search-policy", action="store_true")
    parser.add_argument("--search-rollout-policy-mode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--search-top-k", type=int, default=4)
    parser.add_argument("--search-rollout-steps", type=int, default=16)
    parser.add_argument("--search-rollouts-per-action", type=int, default=4)
    parser.add_argument("--search-army-weight", type=float, default=12.0)
    parser.add_argument("--search-land-weight", type=float, default=8.0)
    parser.add_argument("--search-prior-weight", type=float, default=0.01)
    parser.add_argument("--adaptive-policy", action="store_true", help="Load player 0 as an adaptive checkpoint.")
    parser.add_argument(
        "--opponent-adaptive-policy",
        action="store_true",
        help="Load player 1 as an adaptive checkpoint.",
    )
    parser.add_argument("--pad-to", type=int, default=16)
    parser.add_argument("--network-arch", choices=("cnn", "unet"), default="cnn")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--scoreboard-history", action="store_true")
    parser.add_argument("--fog-memory", action="store_true")
    parser.add_argument("--value-loss", choices=("mse", "hl-gauss"), default="mse")
    parser.add_argument("--value-bins", type=int, default=128)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--value-sigma", type=float, default=0.04)
    parser.add_argument("--policy-adapter-path", default=None)
    parser.add_argument("--policy-adapter-scale", type=float, default=0.0)
    parser.add_argument("--policy-adapter-mode", choices=("delta", "blend", "replace"), default="delta")
    parser.add_argument("--policy-adapter-min-grid-size", type=int, default=0)
    parser.add_argument("--policy-adapter-max-grid-size", type=int, default=0)
    parser.add_argument(
        "--adaptive-online-search",
        action="store_true",
        help="Run adaptive top-k rollout search after the adaptive policy chooses a fallback action.",
    )
    parser.add_argument("--online-search-min-turn", type=int, default=0)
    parser.add_argument("--online-search-require-contact", action="store_true")
    parser.add_argument("--online-search-min-grid-size", type=int, default=0)
    parser.add_argument("--online-search-max-grid-size", type=int, default=0)
    parser.add_argument("--online-search-terminal-score", type=float, default=100.0)
    parser.add_argument("--online-search-min-score-gap", type=float, default=0.0)
    parser.add_argument("--online-search-max-steps", type=int, default=750)
    parser.add_argument("--adaptive-online-search-opponent-path", default=None)
    parser.add_argument(
        "--adaptive-online-search-opponent-policy-mode",
        choices=("greedy", "sample"),
        default="sample",
    )
    parser.add_argument("--adaptive-online-search-opponent-channels", default=None)
    parser.add_argument("--adaptive-online-search-opponent-input-channels", type=int, default=9)
    parser.add_argument("--policy-input", choices=POLICY_INPUT_CHOICES, default=None)
    parser.add_argument("--model-0-policy-input", choices=POLICY_INPUT_CHOICES, default=None)
    parser.add_argument("--model-1-policy-input", choices=POLICY_INPUT_CHOICES, default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--model-0-input-channels", type=int, default=None)
    parser.add_argument("--model-1-input-channels", type=int, default=None)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    args = parser.parse_args()

    if args.model_path is None and args.model_0_path is None:
        parser.error("model_path or --model-0-path is required")
    if args.model_path is not None and args.model_0_path is not None and args.model_path != args.model_0_path:
        parser.error("pass either positional model_path or --model-0-path for player 0, not both")
    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.port <= 0:
        parser.error("--port must be positive")
    if args.tick_rate <= 0:
        parser.error("--tick-rate must be positive")
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if not (1 <= args.preview_top_k <= 5):
        parser.error("--preview-top-k must be between 1 and 5")
    if args.search_top_k <= 0:
        parser.error("--search-top-k must be positive")
    if args.search_rollout_steps <= 0:
        parser.error("--search-rollout-steps must be positive")
    if args.search_rollouts_per_action <= 0:
        parser.error("--search-rollouts-per-action must be positive")
    if args.adaptive_policy and args.search_policy:
        parser.error("Use --adaptive-online-search with --adaptive-policy instead of --search-policy")
    if args.opponent_adaptive_policy and args.opponent_search_policy:
        parser.error("Use --adaptive-online-search with --opponent-adaptive-policy instead of --opponent-search-policy")
    if args.pad_to < args.grid_size:
        parser.error("--pad-to must be at least --grid-size")
    if args.value_loss == "hl-gauss":
        if args.value_bins <= 1:
            parser.error("--value-bins must be greater than 1 for --value-loss hl-gauss")
        if args.value_min >= args.value_max:
            parser.error("--value-min must be less than --value-max")
        if args.value_sigma <= 0.0:
            parser.error("--value-sigma must be positive")
    if args.policy_adapter_scale < 0.0:
        parser.error("--policy-adapter-scale must be non-negative")
    if args.policy_adapter_scale > 0.0 and args.policy_adapter_path is None:
        parser.error("--policy-adapter-scale requires --policy-adapter-path")
    if args.policy_adapter_min_grid_size < 0 or args.policy_adapter_max_grid_size < 0:
        parser.error("--policy-adapter-min/max-grid-size must be non-negative")
    if (
        args.policy_adapter_min_grid_size > 0
        and args.policy_adapter_max_grid_size > 0
        and args.policy_adapter_min_grid_size > args.policy_adapter_max_grid_size
    ):
        parser.error("--policy-adapter-min-grid-size must be <= --policy-adapter-max-grid-size")
    if args.adaptive_online_search and not (args.adaptive_policy or args.opponent_adaptive_policy):
        parser.error("--adaptive-online-search requires --adaptive-policy or --opponent-adaptive-policy")
    if args.online_search_min_turn < 0:
        parser.error("--online-search-min-turn must be non-negative")
    if args.online_search_min_grid_size < 0 or args.online_search_max_grid_size < 0:
        parser.error("--online-search-min/max-grid-size must be non-negative")
    if (
        args.online_search_min_grid_size > 0
        and args.online_search_max_grid_size > 0
        and args.online_search_min_grid_size > args.online_search_max_grid_size
    ):
        parser.error("--online-search-min-grid-size must be <= --online-search-max-grid-size")
    if args.online_search_min_score_gap < 0.0:
        parser.error("--online-search-min-score-gap must be non-negative")
    if args.online_search_max_steps <= 0:
        parser.error("--online-search-max-steps must be positive")
    if args.adaptive_online_search_opponent_input_channels <= 0:
        parser.error("--adaptive-online-search-opponent-input-channels must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")

    args.model_path = args.model_0_path or args.model_path
    if args.max_steps == 0:
        args.max_steps = None
    args.model_0_policy_input = _resolve_alias(
        parser,
        "--model-0-policy-input",
        args.model_0_policy_input,
        "--policy-input",
        args.policy_input,
        "auto",
    )
    args.model_1_policy_input = args.model_1_policy_input or "auto"
    args.model_0_input_channels = _resolve_input_channels(
        args.model_0_policy_input,
        args.model_0_input_channels or args.input_channels,
    )
    args.model_1_input_channels = _resolve_input_channels(args.model_1_policy_input, args.model_1_input_channels)
    return args


def args_to_config(args: argparse.Namespace) -> WebSessionConfig:
    return WebSessionConfig(
        model_path=args.model_path,
        model_0_path=args.model_path,
        model_1_path=args.model_1_path,
        grid_size=args.grid_size,
        map_generator=args.map_generator,
        policy_mode=args.policy_mode,
        opponent_policy_mode=args.opponent_policy_mode,
        machine_vs_machine=args.machine_vs_machine,
        human_player=args.human_player,
        auto_tick=args.auto_tick,
        tick_rate=args.tick_rate,
        max_steps=args.max_steps,
        seed=args.seed,
        preview_top_k=args.preview_top_k,
        ai_preview=args.ai_preview,
        model_0_policy_input=args.model_0_policy_input,
        model_1_policy_input=args.model_1_policy_input,
        model_0_input_channels=args.model_0_input_channels,
        model_1_input_channels=args.model_1_input_channels,
        search_policy=args.search_policy,
        opponent_search_policy=args.opponent_search_policy,
        search_rollout_policy_mode=args.search_rollout_policy_mode,
        search_top_k=args.search_top_k,
        search_rollout_steps=args.search_rollout_steps,
        search_rollouts_per_action=args.search_rollouts_per_action,
        search_army_weight=args.search_army_weight,
        search_land_weight=args.search_land_weight,
        search_prior_weight=args.search_prior_weight,
        adaptive_policy=args.adaptive_policy,
        opponent_adaptive_policy=args.opponent_adaptive_policy,
        pad_to=args.pad_to,
        network_arch=args.network_arch,
        channels=args.channels,
        global_context=args.global_context,
        scoreboard_history=args.scoreboard_history,
        fog_memory=args.fog_memory,
        value_loss=args.value_loss,
        value_bins=args.value_bins,
        value_min=args.value_min,
        value_max=args.value_max,
        value_sigma=args.value_sigma,
        policy_adapter_path=args.policy_adapter_path,
        policy_adapter_scale=args.policy_adapter_scale,
        policy_adapter_mode=args.policy_adapter_mode,
        policy_adapter_min_grid_size=args.policy_adapter_min_grid_size,
        policy_adapter_max_grid_size=args.policy_adapter_max_grid_size,
        adaptive_online_search=args.adaptive_online_search,
        online_search_min_turn=args.online_search_min_turn,
        online_search_require_contact=args.online_search_require_contact,
        online_search_min_grid_size=args.online_search_min_grid_size,
        online_search_max_grid_size=args.online_search_max_grid_size,
        online_search_terminal_score=args.online_search_terminal_score,
        online_search_min_score_gap=args.online_search_min_score_gap,
        online_search_max_steps=args.online_search_max_steps,
        adaptive_online_search_opponent_path=args.adaptive_online_search_opponent_path,
        adaptive_online_search_opponent_policy_mode=args.adaptive_online_search_opponent_policy_mode,
        adaptive_online_search_opponent_channels=args.adaptive_online_search_opponent_channels,
        adaptive_online_search_opponent_input_channels=args.adaptive_online_search_opponent_input_channels,
        mountain_density_min=args.mountain_density_min,
        mountain_density_max=args.mountain_density_max,
        num_cities_min=args.num_cities_min,
        num_cities_max=args.num_cities_max,
        min_generals_distance=args.min_generals_distance,
        max_generals_distance=args.max_generals_distance,
        city_army_min=args.city_army_min,
        city_army_max=args.city_army_max,
        model_catalog=build_model_catalog([args.model_path, args.model_1_path]),
    )


def main() -> None:
    args = parse_args()
    run_server(args_to_config(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
