"""Visualize a trained PPO policy playing against a random opponent."""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PPO_DIR = SCRIPT_DIR / "ppo"
for path in (REPO_ROOT, PPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import pygame

from generals.core.action import compute_valid_move_mask
from generals.core import game
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid
from generals.gui import GUI
from generals.gui.properties import GuiMode
from generals.core.rendering import JaxGameAdapter

from common import make_simple_general_grid, random_action
from generals.agents.ppo_policy_agent import PolicyValueNetwork, obs_to_array


def load_model(model_path: str, grid_size: int = 4):
    """Load a trained PPO model from file."""
    import os

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create a dummy network with the same structure
    key = jrandom.PRNGKey(42)
    network = PolicyValueNetwork(key, grid_size=grid_size)

    # Load the saved weights
    network = eqx.tree_deserialise_leaves(model_path, network)
    print(f"Loaded model from {model_path}")
    return network


def make_grid(
    key,
    grid_size,
    map_generator,
    mountain_density_range,
    num_cities_range,
    min_generals_distance,
    max_generals_distance,
    castle_val_range,
):
    """Create a visualization map matching the trainer's map options."""
    if map_generator == "simple":
        return make_simple_general_grid(key, grid_size)

    return generate_grid(
        key,
        grid_dims=(grid_size, grid_size),
        pad_to=grid_size,
        mountain_density_range=mountain_density_range,
        num_cities_range=num_cities_range,
        min_generals_distance=min_generals_distance,
        max_generals_distance=max_generals_distance,
        castle_val_range=castle_val_range,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize an experimental PPO policy.")
    parser.add_argument("model_path", nargs="?", default="jax_ppo_model.eqx", help="Saved Equinox model path.")
    parser.add_argument("fps", nargs="?", type=int, default=10, help="Rendering frames per second.")
    parser.add_argument("--grid-size", type=int, default=4, help="Square map size used by the saved model.")
    parser.add_argument(
        "--map-generator",
        choices=("simple", "generated"),
        default="simple",
        help="Use simple empty maps or generated maps with mountains/cities.",
    )
    parser.add_argument("--mountain-density-min", type=float, default=0.18, help="Generated-map minimum mountain density.")
    parser.add_argument("--mountain-density-max", type=float, default=0.24, help="Generated-map maximum mountain density.")
    parser.add_argument("--num-cities-min", type=int, default=9, help="Generated-map minimum number of cities.")
    parser.add_argument("--num-cities-max", type=int, default=11, help="Generated-map maximum number of cities.")
    parser.add_argument("--min-generals-distance", type=int, default=None, help="Minimum distance between generals.")
    parser.add_argument("--max-generals-distance", type=int, default=None, help="Maximum distance between generals.")
    parser.add_argument("--city-army-min", type=int, default=40, help="Generated city minimum starting army.")
    parser.add_argument("--city-army-max", type=int, default=51, help="Generated city maximum starting army.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum rendered steps before exit.")
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.fps <= 0:
        parser.error("fps must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")

    return args


def main():
    args = parse_args()
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, args.grid_size // 2)

    print(f"Loading model from: {args.model_path}")
    print(f"Rendering at {args.fps} FPS")
    print(f"Grid: {args.grid_size}x{args.grid_size} ({args.map_generator})")
    if args.map_generator == "generated":
        print(f"Mountains: {args.mountain_density_min:.2f}-{args.mountain_density_max:.2f}")
        print(f"Cities: {args.num_cities_min}-{args.num_cities_max}")
        print(f"General distance: min={min_generals_distance}, max={args.max_generals_distance}")
    print()

    # Load the trained model
    network = load_model(args.model_path, grid_size=args.grid_size)

    # Initialize game state
    key = jrandom.PRNGKey(43)
    grid = make_grid(
        key,
        args.grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    state = create_initial_state(grid)

    # Agent names
    agents = ["PPO Agent", "Random"]

    # Create game adapter for rendering
    info = game.get_info(state)
    game_adapter = JaxGameAdapter(state, agents, info)

    # Create agent data for GUI
    agent_data = {
        "PPO Agent": {"color": (255, 0, 0)},  # Red
        "Random": {"color": (0, 0, 255)},      # Blue
    }

    # Initialize GUI
    gui = GUI(game_adapter, agent_data, mode=GuiMode.TRAIN, speed_multiplier=1.0)

    print("Starting game visualization...")
    print("Controls:")
    print("  - Close window to exit")
    print()

    # Game loop
    step_count = 0
    clock = pygame.time.Clock()

    while step_count < args.max_steps:
        # Handle pygame events
        command = gui.tick(args.fps)
        if command.quit:
            break

        # Check if game is done
        info = game.get_info(state)
        if info.is_done:
            winner_idx = int(info.winner)
            winner_name = agents[winner_idx] if winner_idx >= 0 else "Draw"
            print(f"\nGame over! Winner: {winner_name}")
            print(f"Total steps: {step_count}")

            # Wait a bit before resetting
            time.sleep(2)

            # Reset game
            key, reset_key = jrandom.split(key)
            grid = make_grid(
                reset_key,
                args.grid_size,
                args.map_generator,
                (args.mountain_density_min, args.mountain_density_max),
                (args.num_cities_min, args.num_cities_max),
                min_generals_distance,
                args.max_generals_distance,
                (args.city_army_min, args.city_army_max),
            )
            state = create_initial_state(grid)
            info = game.get_info(state)
            game_adapter.update_from_state(state, info)
            step_count = 0
            print("Starting new game...")
            continue

        # Get observations
        obs_p0 = game.get_observation(state, 0)
        obs_p1 = game.get_observation(state, 1)

        # PPO agent action (player 0)
        obs_arr = obs_to_array(obs_p0)
        mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
        key, action_key = jrandom.split(key)
        action_p0, value, logprob, entropy = network(obs_arr, mask, action_key, None)

        # Random agent action (player 1)
        key, action_key = jrandom.split(key)
        action_p1 = random_action(action_key, obs_p1)

        # Step the game
        actions = jnp.stack([action_p0, action_p1], axis=0)
        new_state, new_info = game.step(state, actions)

        # Update state
        state = new_state
        info = new_info

        # Update GUI adapter
        game_adapter.update_from_state(state, info)

        step_count += 1

        # Control frame rate
        clock.tick(args.fps)

    gui.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
