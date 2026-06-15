import jax.numpy as jnp

from generals.core import game
from generals.core.rewards import general_target_reward_fn, path_assignment_reward_fn


def make_general_target_state(player_cell):
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    row, col = player_cell
    state = state._replace(
        armies=state.armies.at[0, 0].set(1).at[row, col].set(5),
        ownership=state.ownership.at[0, row, col].set(True),
        ownership_neutral=state.ownership_neutral.at[row, col].set(False),
    )
    return state


def test_general_target_reward_is_positive_when_strong_cell_gets_closer_to_enemy_general():
    prior = make_general_target_state((0, 0))
    current = make_general_target_state((1, 0))

    reward = general_target_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=8,
        min_army=2,
    )

    assert reward > 0.0


def test_general_target_reward_is_negative_when_strong_cell_gets_farther_from_enemy_general():
    prior = make_general_target_state((1, 0))
    current = make_general_target_state((0, 0))

    reward = general_target_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=8,
        min_army=2,
    )

    assert reward < 0.0


def test_general_target_reward_scale_zero_disables_shaping():
    prior = make_general_target_state((0, 0))
    current = make_general_target_state((1, 0))

    reward = general_target_reward_fn(
        prior,
        current,
        player=0,
        scale=0.0,
        max_distance=8,
        min_army=2,
    )

    assert reward == 0.0


def make_path_assignment_state(player_cell):
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[2, 4].set(2)
    grid = grid.at[0, 2].set(-2).at[1, 2].set(-2).at[2, 2].set(-2).at[3, 2].set(-2)
    state = game.create_initial_state(grid)
    row, col = player_cell
    return state._replace(
        armies=state.armies.at[0, 0].set(1).at[row, col].set(6),
        ownership=state.ownership.at[0, row, col].set(True),
        ownership_neutral=state.ownership_neutral.at[row, col].set(False),
    )


def test_path_assignment_reward_follows_shortest_path_around_mountains():
    prior = make_path_assignment_state((2, 1))
    current = make_path_assignment_state((3, 1))

    reward = path_assignment_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=25,
        min_army=2,
        general_weight=1.0,
        city_weight=0.0,
        frontier_weight=0.0,
    )

    assert reward > 0.0


def test_path_assignment_reward_penalizes_backtracking_on_shortest_path():
    prior = make_path_assignment_state((3, 1))
    current = make_path_assignment_state((2, 1))

    reward = path_assignment_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=25,
        min_army=2,
        general_weight=1.0,
        city_weight=0.0,
        frontier_weight=0.0,
    )

    assert reward < 0.0


def make_city_assignment_state(player_cell):
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[0, 4].set(2).at[4, 0].set(40)
    state = game.create_initial_state(grid)
    row, col = player_cell
    return state._replace(
        armies=state.armies.at[0, 0].set(1).at[row, col].set(6),
        ownership=state.ownership.at[0, row, col].set(True),
        ownership_neutral=state.ownership_neutral.at[row, col].set(False),
    )


def test_path_assignment_reward_can_assign_transport_to_non_owned_city():
    prior = make_city_assignment_state((1, 0))
    current = make_city_assignment_state((2, 0))

    reward = path_assignment_reward_fn(
        prior,
        current,
        player=0,
        scale=1.0,
        max_distance=25,
        min_army=2,
        general_weight=0.0,
        city_weight=1.0,
        frontier_weight=0.0,
    )

    assert reward > 0.0


def test_path_assignment_reward_scale_zero_disables_shaping():
    prior = make_path_assignment_state((2, 1))
    current = make_path_assignment_state((3, 1))

    reward = path_assignment_reward_fn(
        prior,
        current,
        player=0,
        scale=0.0,
        max_distance=25,
        min_army=2,
        general_weight=1.0,
        city_weight=1.0,
        frontier_weight=1.0,
    )

    assert reward == 0.0
