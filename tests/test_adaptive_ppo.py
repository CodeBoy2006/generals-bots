import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game


def make_padded_state(size=4, pad_to=6):
    grid = jnp.full((pad_to, pad_to), -2, dtype=jnp.int32)
    grid = grid.at[:size, :size].set(0)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[size - 1, size - 1].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(6))
    return state


def test_parse_grid_sizes_and_auto_distances():
    from examples._experimental.ppo.adaptive_common import parse_grid_sizes, min_distance_for_size

    assert parse_grid_sizes("8,12,16") == (8, 12, 16)
    assert min_distance_for_size(8) == 5
    assert min_distance_for_size(12) == 7
    assert min_distance_for_size(16) == 9


def test_adaptive_obs_to_array_marks_padding_separately_from_real_mountain():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS, adaptive_obs_to_array

    state = make_padded_state(size=4, pad_to=6)
    state = state._replace(mountains=state.mountains.at[1, 1].set(True), passable=state.passable.at[1, 1].set(False))
    obs = game.get_observation(state, 0)

    arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)

    assert arr.shape == (ADAPTIVE_INPUT_CHANNELS, 6, 6)
    assert active.shape == (6, 6)
    assert bool(active[3, 3])
    assert not bool(active[4, 4])
    assert arr[3, 1, 1] == 1.0
    assert arr[3, 4, 4] == 0.0
    assert arr[8, 4, 4] == 1.0
    assert arr[9, 1, 1] == 1.0
    assert arr[9, 4, 4] == 0.0
    assert arr[10, 4, 4] == 1.0


def test_compute_adaptive_valid_move_mask_blocks_padding_destinations():
    from examples._experimental.ppo.adaptive_common import compute_adaptive_valid_move_mask

    state = make_padded_state(size=4, pad_to=6)
    state = state._replace(armies=state.armies.at[3, 3].set(5), ownership=state.ownership.at[0, 3, 3].set(True))
    mask = compute_adaptive_valid_move_mask(
        state.armies,
        state.ownership[0],
        state.mountains,
        effective_size=4,
        pad_size=6,
    )

    assert mask.shape == (6, 6, 4)
    assert not bool(mask[3, 3, 1])
    assert not bool(mask[3, 3, 3])
    assert not bool(mask[4, 4, 0])


def test_adaptive_action_encoding_uses_single_pass_index():
    from examples._experimental.ppo.adaptive_common import (
        adaptive_action_to_index,
        adaptive_action_to_target_probs,
        adaptive_index_to_action,
    )

    pad_size = 6
    pass_index = 8 * pad_size * pad_size
    pass_a = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    pass_b = jnp.array([1, 5, 5, 3, 1], dtype=jnp.int32)
    move = jnp.array([0, 2, 1, 3, 1], dtype=jnp.int32)

    assert int(adaptive_action_to_index(pass_a, pad_size)) == pass_index
    assert int(adaptive_action_to_index(pass_b, pad_size)) == pass_index
    assert int(adaptive_action_to_index(move, pad_size)) == (7 * pad_size * pad_size + 2 * pad_size + 1)
    assert adaptive_index_to_action(jnp.asarray(pass_index), pad_size).tolist() == [1, 0, 0, 0, 0]
    assert adaptive_action_to_target_probs(pass_b, pad_size).shape == (8 * pad_size * pad_size + 1,)
