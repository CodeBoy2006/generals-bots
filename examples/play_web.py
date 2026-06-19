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
    parser.add_argument("--max-steps", type=int, default=500)
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
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not (1 <= args.preview_top_k <= 5):
        parser.error("--preview-top-k must be between 1 and 5")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")

    args.model_path = args.model_0_path or args.model_path
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
        mountain_density_min=args.mountain_density_min,
        mountain_density_max=args.mountain_density_max,
        num_cities_min=args.num_cities_min,
        num_cities_max=args.num_cities_max,
        min_generals_distance=args.min_generals_distance,
        max_generals_distance=args.max_generals_distance,
        city_army_min=args.city_army_min,
        city_army_max=args.city_army_max,
    )


def main() -> None:
    args = parse_args()
    run_server(args_to_config(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
