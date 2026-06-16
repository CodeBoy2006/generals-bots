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


def test_adaptive_network_forward_uses_fixed_action_space_and_finite_value():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6)
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value = network.logits_value(obs_arr, mask, active)

    assert logits.shape == (8 * 6 * 6 + 1,)
    assert jnp.isfinite(value)
    assert jnp.isfinite(logits[-1])


def test_adaptive_network_samples_and_scores_action():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6)
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    action, value, logprob, entropy = network(obs_arr, mask, active, jrandom.PRNGKey(1), None)

    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert jnp.isfinite(value)
    assert jnp.isfinite(logprob)
    assert jnp.isfinite(entropy)


def test_make_adaptive_state_pool_balances_sizes():
    from examples._experimental.ppo.adaptive_common import make_adaptive_state_pool

    pool = make_adaptive_state_pool(
        jrandom.PRNGKey(0),
        pool_size=5,
        grid_sizes=(4, 6),
        pad_size=6,
        map_generator="simple",
        mountain_density_range=(0.0, 0.0),
        num_cities_range=(2, 2),
        max_generals_distance=None,
        castle_val_range=(10, 11),
    )

    assert pool.states.armies.shape == (5, 6, 6)
    assert sorted(pool.effective_sizes.tolist()) == [4, 4, 6, 6, 6]


def test_adaptive_expander_target_probs_has_single_pass_slot():
    from examples._experimental.ppo.adaptive_common import adaptive_expander_target_probs

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    target = adaptive_expander_target_probs(obs, effective_size=4, pad_size=6)

    assert target.shape == (8 * 6 * 6 + 1,)
    assert jnp.isclose(jnp.sum(target), 1.0)


def test_behavior_clone_adaptive_cli_smoke(tmp_path):
    import os
    import subprocess
    import sys

    model_path = tmp_path / "adaptive-bc.eqx"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/behavior_clone_adaptive.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--num-steps",
        "1",
        "--num-iterations",
        "1",
        "--model-path",
        str(model_path),
        "--seed",
        "41000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()


def test_train_adaptive_cli_smoke(tmp_path):
    import os
    import subprocess
    import sys

    model_path = tmp_path / "adaptive-ppo.eqx"
    checkpoint_dir = tmp_path / "ckpts"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/train_adaptive.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--num-steps",
        "1",
        "--num-iterations",
        "1",
        "--num-epochs",
        "1",
        "--minibatch-size",
        "2",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "1",
        "--model-path",
        str(model_path),
        "--seed",
        "42000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
    assert (checkpoint_dir / "adaptive-ppo-iter-000001.eqx").exists()
