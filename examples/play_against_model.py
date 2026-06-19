"""Play Generals against a trained PPO .eqx checkpoint."""

import argparse
import sys
from pathlib import Path
import time

import jax.numpy as jnp
import jax.random as jrandom

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generals.agents.ppo_runtime import (
    PlayAgent,
    make_grid,
    make_player_names,
    make_policy_agent as make_gui_agent,
    make_search_config,
    resolve_alias,
    resolve_input_channels,
)
from generals.agents.ppo_policy_agent import (
    POLICY_INPUT_CHOICES,
    PolicyPreview,
)
from generals.core import game
from generals.core.action import compute_valid_move_mask, create_action
from generals.core.game import create_initial_state
from generals.core.rendering import JaxGameAdapter
from generals.gui import GUI
from generals.gui.event_handler import GameCommand
from generals.gui.properties import GuiMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a local Generals game against a trained PPO checkpoint.")
    parser.add_argument("model_path", nargs="?", help="Primary saved Equinox .eqx PPO checkpoint.")
    parser.add_argument("--model-0-path", default=None, help="PPO checkpoint for player 0.")
    parser.add_argument("--model-1-path", default=None, help="PPO checkpoint for player 1 in machine-vs-machine mode.")
    parser.add_argument("--grid-size", type=int, default=8, help="Square map size used by the saved model.")
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--search-policy", action="store_true", help="Wrap the primary PPO checkpoint with rollout search.")
    parser.add_argument(
        "--opponent-search-policy",
        action="store_true",
        help="Wrap the second PPO checkpoint with rollout search in machine-vs-machine mode.",
    )
    parser.add_argument(
        "--search-rollout-policy-mode",
        choices=("greedy", "sample"),
        default="sample",
        help="Policy mode used inside rollout-search simulations.",
    )
    parser.add_argument("--search-top-k", type=int, default=4, help="Policy-prior candidates scored by rollout search.")
    parser.add_argument("--search-rollout-steps", type=int, default=16, help="Simulated steps per rollout-search candidate.")
    parser.add_argument(
        "--search-rollouts-per-action",
        type=int,
        default=4,
        help="Rollout samples per candidate action.",
    )
    parser.add_argument("--search-army-weight", type=float, default=12.0)
    parser.add_argument("--search-land-weight", type=float, default=8.0)
    parser.add_argument("--search-prior-weight", type=float, default=0.01)
    parser.add_argument(
        "--policy-input",
        choices=POLICY_INPUT_CHOICES,
        default=None,
        help="Input encoding for the primary PPO checkpoint.",
    )
    parser.add_argument(
        "--model-0-policy-input",
        choices=POLICY_INPUT_CHOICES,
        default=None,
        help="Input encoding for player 0 in machine-vs-machine mode.",
    )
    parser.add_argument(
        "--model-1-policy-input",
        choices=POLICY_INPUT_CHOICES,
        default=None,
        help="Input encoding for player 1 in machine-vs-machine mode.",
    )
    parser.add_argument("--input-channels", type=int, default=None, help="Input channels for the primary PPO checkpoint.")
    parser.add_argument("--model-0-input-channels", type=int, default=None, help="Input channels for player 0.")
    parser.add_argument("--model-1-input-channels", type=int, default=None, help="Input channels for player 1.")
    parser.add_argument("--machine-vs-machine", action="store_true", help="Watch two PPO agents play each other.")
    parser.add_argument(
        "--opponent-model-path",
        default=None,
        help="Second PPO checkpoint for machine-vs-machine mode.",
    )
    parser.add_argument("--opponent-policy-mode", choices=("greedy", "sample"), default=None)
    parser.add_argument(
        "--opponent-policy-input",
        choices=POLICY_INPUT_CHOICES,
        default=None,
        help="Input encoding for the second PPO checkpoint.",
    )
    parser.add_argument("--opponent-input-channels", type=int, default=None, help="Input channels for the second PPO checkpoint.")
    parser.add_argument("--human-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--auto-tick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Advance turns automatically when no human action is queued.",
    )
    parser.add_argument(
        "--tick-rate",
        type=float,
        default=2.0,
        help="Automatic game turns per second when auto tick is enabled.",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--show-tile-types", action="store_true")
    parser.add_argument("--preview-top-k", type=int, default=3, help="Number of PPO action candidates to preview.")
    parser.add_argument("--no-ai-preview", dest="ai_preview", action="store_false", help="Disable PPO action preview.")
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.tick_rate <= 0:
        parser.error("--tick-rate must be positive")
    if args.search_top_k <= 0:
        parser.error("--search-top-k must be positive")
    if args.search_rollout_steps <= 0:
        parser.error("--search-rollout-steps must be positive")
    if args.search_rollouts_per_action <= 0:
        parser.error("--search-rollouts-per-action must be positive")
    if not (1 <= args.preview_top_k <= 5):
        parser.error("--preview-top-k must be between 1 and 5")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")
    if args.min_generals_distance is not None and args.min_generals_distance < 1:
        parser.error("--min-generals-distance must be >= 1")
    if args.max_generals_distance is not None and args.max_generals_distance < 1:
        parser.error("--max-generals-distance must be >= 1")
    if args.model_path is None and args.model_0_path is None:
        parser.error("model_path or --model-0-path is required")
    if args.model_path is not None and args.model_0_path is not None and args.model_path != args.model_0_path:
        parser.error("pass either positional model_path or --model-0-path for player 0, not both")
    if (
        args.opponent_model_path is not None
        and args.model_1_path is not None
        and args.opponent_model_path != args.model_1_path
    ):
        parser.error("pass either --opponent-model-path or --model-1-path for player 1, not both")

    args.model_path = args.model_0_path or args.model_path
    args.opponent_model_path = args.model_1_path or args.opponent_model_path
    if args.input_channels is not None and args.input_channels <= 0:
        parser.error("--input-channels must be positive")
    if args.opponent_input_channels is not None and args.opponent_input_channels <= 0:
        parser.error("--opponent-input-channels must be positive")
    if args.model_0_input_channels is not None and args.model_0_input_channels <= 0:
        parser.error("--model-0-input-channels must be positive")
    if args.model_1_input_channels is not None and args.model_1_input_channels <= 0:
        parser.error("--model-1-input-channels must be positive")

    args.model_0_policy_input = resolve_alias(
        parser,
        "--model-0-policy-input",
        args.model_0_policy_input,
        "--policy-input",
        args.policy_input,
        "auto",
    )
    explicit_model_1_policy_input = resolve_alias(
        parser,
        "--model-1-policy-input",
        args.model_1_policy_input,
        "--opponent-policy-input",
        args.opponent_policy_input,
        None,
    )
    args.model_1_policy_input = explicit_model_1_policy_input
    if args.model_1_policy_input is None:
        args.model_1_policy_input = "auto" if args.opponent_model_path is not None else args.model_0_policy_input

    args.model_0_input_channels = resolve_alias(
        parser,
        "--model-0-input-channels",
        args.model_0_input_channels,
        "--input-channels",
        args.input_channels,
        None,
    )
    args.model_0_input_channels = resolve_input_channels(args.model_0_policy_input, args.model_0_input_channels)

    explicit_model_1_input_channels = resolve_alias(
        parser,
        "--model-1-input-channels",
        args.model_1_input_channels,
        "--opponent-input-channels",
        args.opponent_input_channels,
        None,
    )
    args.model_1_input_channels = explicit_model_1_input_channels
    if args.model_1_input_channels is None:
        if args.opponent_model_path is None and explicit_model_1_policy_input is None:
            args.model_1_input_channels = args.model_0_input_channels
        else:
            args.model_1_input_channels = resolve_input_channels(args.model_1_policy_input, None)
    if args.search_policy:
        if args.model_0_policy_input not in ("auto", "observation"):
            parser.error("--search-policy only supports --model-0-policy-input observation or auto")
        if args.model_0_input_channels not in (None, 9):
            parser.error("--search-policy requires --model-0-input-channels 9")
    if args.opponent_search_policy:
        if args.model_1_policy_input not in ("auto", "observation"):
            parser.error("--opponent-search-policy only supports --model-1-policy-input observation or auto")
        if args.model_1_input_channels not in (None, 9):
            parser.error("--opponent-search-policy requires --model-1-input-channels 9")

    args.effective_min_generals_distance = args.min_generals_distance
    if args.effective_min_generals_distance is None:
        args.effective_min_generals_distance = max(3, args.grid_size // 2)
    if (
        args.max_generals_distance is not None
        and args.effective_min_generals_distance > args.max_generals_distance
    ):
        parser.error("generals distance must satisfy min <= max")
    return args


def human_can_move(state: game.GameState, human_player: int) -> bool:
    """Return whether the human player has at least one legal move."""
    obs = game.get_observation(state, human_player)
    valid_moves = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    return bool(jnp.any(valid_moves))


def advance_until_human_can_move(
    state: game.GameState,
    human_player: int,
    max_auto_passes: int = 10,
) -> tuple[game.GameState, game.GameInfo, int]:
    """Auto-pass the unplayable opening turns so the first frame accepts clicks."""
    info = game.get_info(state)
    pass_actions = jnp.stack([create_action(to_pass=True), create_action(to_pass=True)])
    auto_passes = 0
    while not bool(info.is_done) and not human_can_move(state, human_player) and auto_passes < max_auto_passes:
        state, info = game.step(state, pass_actions)
        auto_passes += 1
    return state, info, auto_passes


def auto_tick_due(
    auto_tick: bool,
    selected_cell: tuple[int, int] | None,
    now: float,
    last_tick: float,
    tick_rate: float,
) -> bool:
    """Return whether an idle human turn should be auto-passed now."""
    if not auto_tick or selected_cell is not None:
        return False
    return now - last_tick >= 1.0 / tick_rate


def choose_human_action(command_action: jnp.ndarray | None, auto_tick_ready: bool) -> jnp.ndarray | None:
    """Use a queued human action, or pass when an automatic tick is due."""
    if command_action is not None:
        return command_action
    if auto_tick_ready:
        return create_action(to_pass=True)
    return None


def choose_agent_action(agent: PlayAgent, state: game.GameState, player: int, key: jnp.ndarray) -> jnp.ndarray:
    """Choose an action from a state-aware agent, with legacy observation-agent fallback."""
    if hasattr(agent, "act_for_state"):
        return agent.act_for_state(state, player, key)
    return agent.act(game.get_observation(state, player), key)


def choose_machine_actions(state: game.GameState, agents: tuple[PlayAgent, PlayAgent], key: jnp.ndarray) -> jnp.ndarray:
    """Choose simultaneous actions for both machine players."""
    key_0, key_1 = jrandom.split(key)
    return jnp.stack(
        [
            choose_agent_action(agents[0], state, 0, key_0),
            choose_agent_action(agents[1], state, 1, key_1),
        ]
    )


def explain_agent(agent: PlayAgent, state: game.GameState, player: int, top_k: int) -> PolicyPreview:
    """Return a policy preview from a state-aware agent, with observation fallback."""
    if hasattr(agent, "explain_for_state"):
        return agent.explain_for_state(state, player, top_k=top_k)
    return agent.explain(game.get_observation(state, player), top_k=top_k)


def print_game_result(info: game.GameInfo, names: list[str], step_count: int, reached_limit: bool = False) -> None:
    if reached_limit and int(info.winner) < 0:
        print(f"Reached max steps ({step_count}) without a winner. Press R to restart or Q to quit.")
        return
    winner = names[int(info.winner)] if int(info.winner) >= 0 else "None"
    print(f"Game over after {step_count} steps. Winner: {winner}. Press R to restart or Q to quit.")


def main() -> None:
    args = parse_args()
    model_player = 1 - args.human_player
    key = jrandom.PRNGKey(args.seed)
    names = make_player_names(args.human_player, machine_vs_machine=args.machine_vs_machine)
    search_config = make_search_config(args)
    if args.machine_vs_machine:
        opponent_model_path = args.opponent_model_path or args.model_path
        opponent_policy_mode = args.opponent_policy_mode or args.policy_mode
        machine_agents = (
            make_gui_agent(
                args.model_path,
                args.grid_size,
                args.policy_mode,
                names[0],
                args.model_0_policy_input,
                args.model_0_input_channels,
                args.search_policy,
                search_config,
            ),
            make_gui_agent(
                opponent_model_path,
                args.grid_size,
                opponent_policy_mode,
                names[1],
                args.model_1_policy_input,
                args.model_1_input_channels,
                args.opponent_search_policy,
                search_config,
            ),
        )
        policy_agent = machine_agents[0]
        preview_player = 0
    else:
        machine_agents = None
        policy_agent = make_gui_agent(
            args.model_path,
            args.grid_size,
            args.policy_mode,
            "PPO Model",
            args.model_0_policy_input,
            args.model_0_input_channels,
            args.search_policy,
            search_config,
        )
        preview_player = model_player

    agent_data = (
        {
            "PPO 0": {"color": (220, 55, 55)},
            "PPO 1": {"color": (40, 90, 220)},
        }
        if args.machine_vs_machine
        else {
            "Human": {"color": (220, 55, 55)},
            "PPO Model": {"color": (40, 90, 220)},
        }
    )

    def new_game():
        nonlocal key
        key, map_key = jrandom.split(key)
        state = create_initial_state(make_grid(args, map_key))
        if args.machine_vs_machine:
            info = game.get_info(state)
        else:
            state, info, auto_passes = advance_until_human_can_move(state, args.human_player)
            if auto_passes:
                print(f"Auto-passed {auto_passes} opening turns so your first move is available.")
        return state, info

    state, info = new_game()
    game_adapter = JaxGameAdapter(state, names, info)
    gui = GUI(
        game_adapter,
        agent_data,
        mode=GuiMode.GAME,
        show_tile_types=args.show_tile_types,
        human_player=args.human_player,
    )

    if args.machine_vs_machine:
        print("Watching PPO 0 vs PPO 1. Press R after game over to restart, Q to quit.")
    else:
        print("Controls: left-click source, left-click adjacent target, S split, P pass, Esc/right-click cancel.")
        print("Press R after game over to restart, Q to quit.")
        print(f"Playing as player {args.human_player} on {args.grid_size}x{args.grid_size}.")
    if args.ai_preview:
        preview_kind = "rollout-search" if args.search_policy else "PPO"
        print(f"AI preview: showing top {args.preview_top_k} {preview_kind} candidate actions in the right panel.")
    if args.search_policy or (args.machine_vs_machine and args.opponent_search_policy):
        print(
            "Rollout search: "
            f"top_k={args.search_top_k}, rollout_steps={args.search_rollout_steps}, "
            f"rollouts/action={args.search_rollouts_per_action}, rollout_policy={args.search_rollout_policy_mode}."
        )
    if args.auto_tick:
        print(f"Auto tick: {args.tick_rate:g} turns/sec. Idle human turns pass automatically.")

    step_count = 0
    terminal_reported = False
    last_tick = time.monotonic()

    try:
        while True:
            reached_limit = step_count >= args.max_steps and int(info.winner) < 0
            game_done = bool(info.is_done) or reached_limit
            if args.ai_preview and not game_done:
                gui.set_policy_preview(explain_agent(policy_agent, state, preview_player, top_k=args.preview_top_k))
            else:
                gui.clear_policy_preview()

            command = gui.tick(fps=args.fps)
            if command.quit:
                break

            if isinstance(command, GameCommand) and command.restart and game_done:
                state, info = new_game()
                game_adapter.update_from_state(state, info)
                step_count = 0
                terminal_reported = False
                last_tick = time.monotonic()
                print("Starting new game.")
                continue

            if game_done:
                if not terminal_reported:
                    print_game_result(info, names, step_count, reached_limit)
                    terminal_reported = True
                time.sleep(0.02)
                continue

            if not isinstance(command, GameCommand):
                continue

            now = time.monotonic()
            if args.machine_vs_machine:
                if not auto_tick_due(args.auto_tick, None, now, last_tick, args.tick_rate):
                    continue
                key, action_key = jrandom.split(key)
                assert machine_agents is not None
                actions = choose_machine_actions(state, machine_agents, action_key)
                state, info = game.step(state, actions)
                game_adapter.update_from_state(state, info)
                step_count += 1
                last_tick = now
                continue

            auto_ready = auto_tick_due(
                args.auto_tick,
                command.selected_cell,
                now,
                last_tick,
                args.tick_rate,
            )
            human_action = choose_human_action(command.action, auto_ready)
            if human_action is None:
                continue

            key, action_key = jrandom.split(key)
            model_action = choose_agent_action(policy_agent, state, model_player, action_key)

            actions = (
                jnp.stack((human_action, model_action))
                if args.human_player == 0
                else jnp.stack((model_action, human_action))
            )
            state, info = game.step(state, actions)
            game_adapter.update_from_state(state, info)
            step_count += 1
            last_tick = now
    finally:
        gui.close()


if __name__ == "__main__":
    main()
