"""Play Generals against a trained PPO .eqx checkpoint."""

import argparse
import time

import jax.numpy as jnp
import jax.random as jrandom

from generals.agents import PPOPolicyAgent
from generals.core import game
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid
from generals.core.rendering import JaxGameAdapter
from generals.gui import GUI
from generals.gui.event_handler import GameCommand
from generals.gui.properties import GuiMode


def make_simple_general_grid(key: jnp.ndarray, grid_size: int) -> jnp.ndarray:
    """Create an empty square grid with two random generals."""
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    return grid.at[pos_a].set(1).at[pos_b].set(2)


def make_grid(args: argparse.Namespace, key: jnp.ndarray) -> jnp.ndarray:
    """Create one playable map from CLI options."""
    if args.map_generator == "simple":
        return make_simple_general_grid(key, args.grid_size)

    return generate_grid(
        key,
        grid_dims=(args.grid_size, args.grid_size),
        pad_to=args.grid_size,
        mountain_density_range=(args.mountain_density_min, args.mountain_density_max),
        num_cities_range=(args.num_cities_min, args.num_cities_max),
        min_generals_distance=args.effective_min_generals_distance,
        max_generals_distance=args.max_generals_distance,
        castle_val_range=(args.city_army_min, args.city_army_max),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a local Generals game against a trained PPO checkpoint.")
    parser.add_argument("model_path", help="Path to a saved Equinox .eqx PPO checkpoint.")
    parser.add_argument("--grid-size", type=int, default=8, help="Square map size used by the saved model.")
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--human-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--show-tile-types", action="store_true")
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
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")

    args.effective_min_generals_distance = args.min_generals_distance
    if args.effective_min_generals_distance is None:
        args.effective_min_generals_distance = max(3, args.grid_size // 2)
    return args


def make_player_names(human_player: int) -> list[str]:
    names = ["PPO Model", "PPO Model"]
    names[human_player] = "Human"
    names[1 - human_player] = "PPO Model"
    return names


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
    policy_agent = PPOPolicyAgent(args.model_path, args.grid_size, args.policy_mode, id="PPO Model")
    names = make_player_names(args.human_player)
    agent_data = {
        "Human": {"color": (220, 55, 55)},
        "PPO Model": {"color": (40, 90, 220)},
    }

    def new_game():
        nonlocal key
        key, map_key = jrandom.split(key)
        state = create_initial_state(make_grid(args, map_key))
        info = game.get_info(state)
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

    print("Controls: left-click source, left-click adjacent target, S split, P pass, Esc/right-click cancel, R restart.")
    print(f"Playing as player {args.human_player} on {args.grid_size}x{args.grid_size}.")

    step_count = 0
    terminal_reported = False

    try:
        while True:
            command = gui.tick(fps=args.fps)
            if command.quit:
                break

            reached_limit = step_count >= args.max_steps and int(info.winner) < 0
            game_done = bool(info.is_done) or reached_limit
            if isinstance(command, GameCommand) and command.restart and game_done:
                state, info = new_game()
                game_adapter.update_from_state(state, info)
                step_count = 0
                terminal_reported = False
                print("Starting new game.")
                continue

            if game_done:
                if not terminal_reported:
                    print_game_result(info, names, step_count, reached_limit)
                    terminal_reported = True
                time.sleep(0.02)
                continue

            if not isinstance(command, GameCommand) or command.action is None:
                continue

            human_action = command.action
            key, action_key = jrandom.split(key)
            model_obs = game.get_observation(state, model_player)
            model_action = policy_agent.act(model_obs, action_key)

            actions = (
                jnp.stack((human_action, model_action))
                if args.human_player == 0
                else jnp.stack((model_action, human_action))
            )
            state, info = game.step(state, actions)
            game_adapter.update_from_state(state, info)
            step_count += 1
    finally:
        gui.close()


if __name__ == "__main__":
    main()
