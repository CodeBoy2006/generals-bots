"""Generate a dense square GIF of smaller-scale parallel training rollouts.

Usage:
    uv run --with pillow python examples/generate_parallel_training_square_gif.py

The GIF intentionally has no labels, titles, or metric text. It is a square
tiled field of parallel games.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image, ImageDraw

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent, RandomAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "generals" / "assets" / "gifs" / "parallel_training_square_tiled.gif"


CANVAS_SIZE = 1280
GRID_DIMS = (8, 8)
TILE_ROWS = 8
TILE_COLS = 8
NUM_ENVS = TILE_ROWS * TILE_COLS
MARGIN = 12
GAP = 6
CELL_SIZE = 18
BOARD_SIZE = GRID_DIMS[0] * CELL_SIZE
FRAMES = 108
STEPS_PER_FRAME = 2
FRAME_DURATION_MS = 110

BG = (24, 22, 18)
GRID_LINE = (45, 40, 35)
NEUTRAL = (226, 220, 204)
MOUNTAIN = (78, 76, 70)
MOUNTAIN_LIGHT = (118, 114, 103)
CITY = (222, 166, 70)
CITY_DARK = (123, 81, 32)
P0 = (41, 184, 170)
P0_DARK = (14, 106, 99)
P1 = (231, 92, 89)
P1_DARK = (143, 48, 48)
GOLD = (244, 189, 76)
WHITE = (255, 251, 237)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Square animated GIF destination.")
    parser.add_argument("--seed", type=int, default=31, help="JAX PRNG seed.")
    return parser.parse_args()


def mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def action_endpoint(row: int, col: int, direction: int) -> tuple[int, int]:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dr, dc = offsets[int(direction) % 4]
    return row + dr, col + dc


def board_origin(index: int) -> tuple[int, int]:
    row, col = divmod(index, TILE_COLS)
    slot = (CANVAS_SIZE - 2 * MARGIN - (TILE_COLS - 1) * GAP) // TILE_COLS
    x = MARGIN + col * (slot + GAP) + (slot - BOARD_SIZE) // 2
    y = MARGIN + row * (slot + GAP) + (slot - BOARD_SIZE) // 2
    return x, y


def owned_color(base: tuple[int, int, int], army: int) -> tuple[int, int, int]:
    pressure = min(0.42, np.log1p(max(army, 0)) / 13)
    return mix(base, WHITE, 0.24 - pressure * 0.48)


def draw_mountain(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    points = [(x + 2, y + CELL_SIZE - 3), (x + CELL_SIZE // 2, y + 3), (x + CELL_SIZE - 3, y + CELL_SIZE - 3)]
    draw.polygon(points, fill=MOUNTAIN, outline=GRID_LINE)
    draw.line((x + CELL_SIZE // 2, y + 4, x + CELL_SIZE - 4, y + CELL_SIZE - 3), fill=MOUNTAIN_LIGHT)


def draw_city(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rectangle((x + 4, y + 6, x + CELL_SIZE - 5, y + CELL_SIZE - 3), fill=CITY, outline=CITY_DARK)
    draw.rectangle((x + 5, y + 3, x + CELL_SIZE - 6, y + 7), fill=mix(CITY, WHITE, 0.12))


def draw_general(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    cx = x + CELL_SIZE // 2
    points = [
        (cx - 7, y + 13),
        (cx - 6, y + 6),
        (cx - 2, y + 8),
        (cx, y + 3),
        (cx + 2, y + 8),
        (cx + 6, y + 6),
        (cx + 7, y + 13),
    ]
    draw.polygon(points, fill=color, outline=GRID_LINE)
    draw.rectangle((cx - 7, y + 12, cx + 7, y + 15), fill=color, outline=GRID_LINE)


def draw_board(draw: ImageDraw.ImageDraw, state: dict[str, np.ndarray], actions: np.ndarray, env_idx: int) -> None:
    x0, y0 = board_origin(env_idx)
    armies = state["armies"][env_idx]
    ownership = state["ownership"][env_idx]
    mountains = state["mountains"][env_idx]
    cities = state["cities"][env_idx]
    generals = state["generals"][env_idx]

    draw.rectangle((x0 - 2, y0 - 2, x0 + BOARD_SIZE + 1, y0 + BOARD_SIZE + 1), fill=(15, 13, 11))
    for row in range(GRID_DIMS[0]):
        for col in range(GRID_DIMS[1]):
            x = x0 + col * CELL_SIZE
            y = y0 + row * CELL_SIZE
            army = int(armies[row, col])
            if bool(mountains[row, col]):
                fill = MOUNTAIN
            elif bool(ownership[0, row, col]):
                fill = owned_color(P0, army)
            elif bool(ownership[1, row, col]):
                fill = owned_color(P1, army)
            elif bool(cities[row, col]):
                fill = mix(CITY, NEUTRAL, 0.16)
            else:
                fill = NEUTRAL

            draw.rectangle((x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1), fill=fill, outline=GRID_LINE)
            if bool(mountains[row, col]):
                draw_mountain(draw, x, y)
            elif bool(cities[row, col]):
                draw_city(draw, x, y)
            if bool(generals[row, col]):
                color = GOLD if bool(ownership[0, row, col]) else mix(P1, GOLD, 0.45)
                draw_general(draw, x, y, color)
            elif army >= 18 and bool(ownership[:, row, col].any()):
                color = P0_DARK if bool(ownership[0, row, col]) else P1_DARK
                draw.rectangle((x + 7, y + 7, x + 11, y + 11), fill=color)

    for player, color in [(0, P0_DARK), (1, P1_DARK)]:
        action = actions[env_idx, player]
        if int(action[0]) != 0:
            continue
        row, col, direction = int(action[1]), int(action[2]), int(action[3])
        target_row, target_col = action_endpoint(row, col, direction)
        if not (0 <= row < GRID_DIMS[0] and 0 <= col < GRID_DIMS[1]):
            continue
        if not (0 <= target_row < GRID_DIMS[0] and 0 <= target_col < GRID_DIMS[1]):
            continue
        start = (x0 + col * CELL_SIZE + CELL_SIZE // 2, y0 + row * CELL_SIZE + CELL_SIZE // 2)
        end = (x0 + target_col * CELL_SIZE + CELL_SIZE // 2, y0 + target_row * CELL_SIZE + CELL_SIZE // 2)
        draw.line((start, end), fill=color, width=2)
        draw.rectangle((end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2), fill=color)


def snapshot_state(states, actions: jnp.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    state_np = {
        "armies": np.array(states.armies),
        "ownership": np.array(states.ownership),
        "mountains": np.array(states.mountains),
        "cities": np.array(states.cities),
        "generals": np.array(states.generals),
    }
    return state_np, np.array(actions)


def render_frame(states, actions: jnp.ndarray) -> Image.Image:
    state_np, actions_np = snapshot_state(states, actions)
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG)
    draw = ImageDraw.Draw(image)
    for env_idx in range(NUM_ENVS):
        draw_board(draw, state_np, actions_np, env_idx)
    return image


def build_frames(seed: int) -> list[Image.Image]:
    env = GeneralsEnv(
        grid_dims=GRID_DIMS,
        truncation=170,
        mountain_density_range=(0.10, 0.23),
        num_cities_range=(3, 7),
        min_generals_distance=4,
        max_generals_distance=7,
        pool_size=512,
    )
    learner_expert = ExpanderAgent(id="Policy")
    learner_noise = RandomAgent(id="Explorer", split_prob=0.36, idle_prob=0.08)
    opponent = RandomAgent(id="Opponent", split_prob=0.28, idle_prob=0.06)

    key = jrandom.PRNGKey(seed)
    key, pool_key, init_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)
    states = jax.vmap(env.init_state)(jrandom.split(init_key, NUM_ENVS))

    step_vmap = jax.vmap(lambda state, action: env.step(state, action, pool))
    get_obs_p0 = jax.vmap(lambda state: get_observation(state, 0))
    get_obs_p1 = jax.vmap(lambda state: get_observation(state, 1))
    act_expert = jax.vmap(learner_expert.act)
    act_noise = jax.vmap(learner_noise.act)
    act_opponent = jax.vmap(opponent.act)

    frames: list[Image.Image] = []
    actions = jnp.zeros((NUM_ENVS, 2, 5), dtype=jnp.int32)
    for frame_idx in range(FRAMES):
        policy_strength = 0.12 + 0.82 * (frame_idx / max(1, FRAMES - 1)) ** 0.78
        for _ in range(STEPS_PER_FRAME):
            obs_p0 = get_obs_p0(states)
            obs_p1 = get_obs_p1(states)
            key, k_expert, k_noise, k_opp, k_mix = jrandom.split(key, 5)
            expert_actions = act_expert(obs_p0, jrandom.split(k_expert, NUM_ENVS))
            noise_actions = act_noise(obs_p0, jrandom.split(k_noise, NUM_ENVS))
            use_expert = jrandom.uniform(k_mix, (NUM_ENVS,)) < policy_strength
            actions_p0 = jnp.where(use_expert[:, None], expert_actions, noise_actions)
            actions_p1 = act_opponent(obs_p1, jrandom.split(k_opp, NUM_ENVS))
            actions = jnp.stack([actions_p0, actions_p1], axis=1)
            _, states = step_vmap(states, actions)
        frames.append(render_frame(states, actions))
    return frames


def main() -> None:
    args = parse_args()
    frames = build_frames(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(f"Wrote {args.output}")
    print(f"Frames: {len(frames)}, size: {CANVAS_SIZE}x{CANVAS_SIZE}, boards: {NUM_ENVS}")


if __name__ == "__main__":
    main()
