import jax
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


def test_parse_grid_size_weights_requires_matching_positive_sizes():
    import pytest

    from examples._experimental.ppo.adaptive_common import parse_grid_size_weights

    assert parse_grid_size_weights("8:1,12:1.5,16:2", (8, 12, 16)) == (1.0, 1.5, 2.0)
    assert parse_grid_size_weights(None, (8, 12, 16)) is None

    for value in ("8:1,12:1", "8:1,12:1,16:0", "8:1,12:1,20:2", "8:1,8:2,16:1"):
        with pytest.raises(ValueError):
            parse_grid_size_weights(value, (8, 12, 16))


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


def test_adaptive_obs_to_array_can_append_global_scoreboard_channels():
    from examples._experimental.ppo.adaptive_common import (
        ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_obs_to_array,
    )

    state = make_padded_state(size=4, pad_to=6)
    state = state._replace(
        time=jnp.asarray(25, dtype=jnp.int32),
        armies=state.armies.at[0, 0].set(12).at[3, 3].set(5),
        ownership=state.ownership.at[0, 0, 1].set(True).at[1, 3, 2].set(True),
    )
    obs = game.get_observation(state, 0)

    base_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    arr, global_active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6, include_global_context=True)

    assert base_arr.shape == (ADAPTIVE_INPUT_CHANNELS, 6, 6)
    assert arr.shape == (ADAPTIVE_GLOBAL_INPUT_CHANNELS, 6, 6)
    assert jnp.array_equal(active, global_active)
    assert jnp.allclose(arr[:ADAPTIVE_INPUT_CHANNELS], base_arr)
    assert jnp.allclose(arr[15, :4, :4], 2.0 / 16.0)
    assert jnp.allclose(arr[17, :4, :4], 2.0 / 16.0)
    assert jnp.allclose(arr[19, :4, :4], 25.0 / 750.0)
    assert jnp.all(arr[15:, 4:, :] == 0.0)
    assert jnp.all(arr[15:, :, 4:] == 0.0)


def test_adaptive_obs_to_array_can_append_scoreboard_history_channels():
    from examples._experimental.ppo.adaptive_common import (
        ADAPTIVE_HISTORY_INPUT_CHANNELS,
        adaptive_obs_to_array,
    )

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    scoreboard_history = jnp.linspace(0.1, 1.0, 10, dtype=jnp.float32)

    arr, active = adaptive_obs_to_array(
        obs,
        effective_size=4,
        pad_size=6,
        include_global_context=True,
        scoreboard_history=scoreboard_history,
    )

    assert arr.shape == (ADAPTIVE_HISTORY_INPUT_CHANNELS, 6, 6)
    assert active.shape == (6, 6)
    assert jnp.allclose(arr[20:, :4, :4], scoreboard_history[:, None, None])
    assert jnp.all(arr[20:, 4:, :] == 0.0)
    assert jnp.all(arr[20:, :, 4:] == 0.0)


def test_adaptive_obs_to_array_can_append_fog_memory_channels():
    from examples._experimental.ppo.adaptive_common import (
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_input_channel_count,
        adaptive_obs_to_array,
        empty_adaptive_fog_memory,
        update_adaptive_fog_memory,
    )

    state = make_padded_state(size=4, pad_to=6)
    state = state._replace(
        armies=state.armies.at[0, 1].set(3),
        ownership=state.ownership.at[1, 0, 1].set(True),
    )
    obs = game.get_observation(state, 0)
    empty_memory = empty_adaptive_fog_memory(1, 6)
    row_memory = jax.tree.map(lambda value: value[0], empty_memory)
    memory = update_adaptive_fog_memory(row_memory, obs)

    arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6, fog_memory=memory)

    assert arr.shape == (adaptive_input_channel_count(fog_memory=True), 6, 6)
    assert active.shape == (6, 6)
    assert jnp.allclose(arr[:ADAPTIVE_INPUT_CHANNELS], adaptive_obs_to_array(obs, 4, 6)[0])
    assert arr[ADAPTIVE_INPUT_CHANNELS, 0, 0] == 1.0


def test_adaptive_scoreboard_history_context_and_reset():
    from examples._experimental.ppo.adaptive_common import (
        adaptive_scoreboard_history_context,
        reset_adaptive_scoreboard_history,
    )

    previous = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]], dtype=jnp.float32)
    current = jnp.array([[0.2, 0.1, 0.5, 0.3, 0.6], [0.6, 0.6, 0.2, 0.2, 0.2]], dtype=jnp.float32)
    dones = jnp.array([False, True])

    context = adaptive_scoreboard_history_context(previous, current)
    reset = reset_adaptive_scoreboard_history(current, dones)

    assert context.shape == (2, 10)
    assert jnp.allclose(context[:, :5], previous)
    assert jnp.allclose(context[:, 5:], current - previous)
    assert jnp.allclose(reset[0], current[0])
    assert jnp.allclose(reset[1], jnp.zeros((5,), dtype=jnp.float32))


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


def test_adaptive_unet_network_forward_uses_fixed_action_space_and_finite_value():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptiveUNetPolicyValueNetwork

    network = AdaptiveUNetPolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value = network.logits_value(obs_arr, mask, active)

    assert logits.shape == (8 * 6 * 6 + 1,)
    assert jnp.isfinite(value)
    assert jnp.isfinite(logits[-1])


def test_hl_gauss_target_is_normalized_and_clipped():
    from examples._experimental.ppo.adaptive_network import categorical_value_expectation, hl_gauss_target

    target = hl_gauss_target(jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32), 9, -1.0, 1.0, 0.15)

    assert target.shape == (3, 9)
    assert jnp.allclose(jnp.sum(target, axis=-1), jnp.ones((3,), dtype=jnp.float32), atol=1e-5)
    assert int(jnp.argmax(target[0])) == 0
    assert int(jnp.argmax(target[1])) == 4
    assert int(jnp.argmax(target[2])) == 8

    value = categorical_value_expectation(jnp.log(target[1]), -1.0, 1.0)
    assert abs(float(value)) < 0.05


def test_adaptive_network_can_expose_hl_gauss_value_logits():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        value_head_sizes=(4, 6),
        value_bins=8,
        value_min=-1.0,
        value_max=1.0,
        value_sigma=0.2,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value, value_logits = network.logits_value_distribution(obs_arr, mask, active)
    legacy_logits, legacy_value = network.logits_value(obs_arr, mask, active)

    assert logits.shape == (8 * 6 * 6 + 1,)
    assert value_logits.shape == (8,)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(value_logits))
    assert jnp.allclose(legacy_logits, logits)
    assert jnp.allclose(legacy_value, value)


def test_adaptive_network_can_expose_outcome_auxiliary_logits():
    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        outcome_head=True,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value, value_logits, outcome_logits = network.logits_value_auxiliary(obs_arr, mask, active)
    legacy_logits, legacy_value = network.logits_value(obs_arr, mask, active)

    assert logits.shape == (8 * 6 * 6 + 1,)
    assert value_logits is None
    assert outcome_logits.shape == (3,)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(outcome_logits))
    assert jnp.allclose(legacy_logits, logits)
    assert jnp.allclose(legacy_value, value)


def test_adaptive_network_global_context_accepts_global_input():
    from examples._experimental.ppo.adaptive_common import (
        adaptive_obs_to_array,
        compute_adaptive_valid_move_mask,
    )
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        global_context=True,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6, include_global_context=True)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value = network.logits_value(obs_arr, mask, active)

    assert logits.shape == (8 * 6 * 6 + 1,)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(logits))


def test_adaptive_network_global_context_accepts_history_input_width():
    from examples._experimental.ppo.adaptive_common import (
        ADAPTIVE_HISTORY_INPUT_CHANNELS,
        adaptive_obs_to_array,
        compute_adaptive_valid_move_mask,
    )
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        input_channels=ADAPTIVE_HISTORY_INPUT_CHANNELS,
        global_context=True,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(
        obs,
        effective_size=4,
        pad_size=6,
        include_global_context=True,
        scoreboard_history=jnp.zeros((10,), dtype=jnp.float32),
    )
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    logits, value = network.logits_value(obs_arr, mask, active)

    assert network.global_linear1.weight.shape[1] == ADAPTIVE_HISTORY_INPUT_CHANNELS - 13
    assert logits.shape == (8 * 6 * 6 + 1,)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(logits))


def test_load_or_create_adaptive_network_can_warm_start_hl_gauss_from_scalar_checkpoint(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    model_path = tmp_path / "adaptive-scalar.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=(16, 16, 16, 8),
        init_channels=(16, 16, 16, 8),
        value_head_sizes=(4, 6),
        init_value_head_sizes=(),
        value_bins=8,
        init_value_bins=0,
        value_sigma=0.2,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, _ = source.logits_value(obs_arr, mask, active)
    actual_logits, value, value_logits = loaded.logits_value_distribution(obs_arr, mask, active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert value_logits.shape == (8,)
    assert jnp.isfinite(value)


def test_load_or_create_adaptive_network_can_warm_start_outcome_head_from_scalar_checkpoint(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    model_path = tmp_path / "adaptive-scalar.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=(16, 16, 16, 8),
        init_channels=(16, 16, 16, 8),
        outcome_head=True,
        init_outcome_head=False,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, expected_value = source.logits_value(obs_arr, mask, active)
    actual_logits, actual_value, _, outcome_logits = loaded.logits_value_auxiliary(obs_arr, mask, active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert jnp.allclose(actual_value, expected_value, atol=1e-5)
    assert outcome_logits.shape == (3,)
    assert jnp.all(jnp.isfinite(outcome_logits))


def test_ppo_loss_terms_uses_hl_gauss_value_loss_when_available():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.train_adaptive import ppo_loss_terms

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        value_bins=8,
        value_sigma=0.2,
    )
    obs = jnp.zeros((ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    mask = jnp.ones((6, 6, 4), dtype=bool)
    active = jnp.ones((6, 6), dtype=bool)
    action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

    policy_loss, value_loss, entropy = ppo_loss_terms(
        network,
        obs,
        mask,
        active,
        action,
        old_logprob=jnp.asarray(0.0, dtype=jnp.float32),
        advantage=jnp.asarray(1.0, dtype=jnp.float32),
        ret=jnp.asarray(0.75, dtype=jnp.float32),
    )

    assert jnp.isfinite(policy_loss)
    assert jnp.isfinite(value_loss)
    assert jnp.isfinite(entropy)
    assert float(value_loss) > 0.0


def test_ppo_loss_terms_with_outcome_auxiliary_is_finite():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.train_adaptive import ppo_loss_terms_with_outcome

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        outcome_head=True,
    )
    obs = jnp.zeros((ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    mask = jnp.ones((6, 6, 4), dtype=bool)
    active = jnp.ones((6, 6), dtype=bool)
    action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

    policy_loss, value_loss, entropy, outcome_loss = ppo_loss_terms_with_outcome(
        network,
        obs,
        mask,
        active,
        action,
        old_logprob=jnp.asarray(0.0, dtype=jnp.float32),
        advantage=jnp.asarray(1.0, dtype=jnp.float32),
        ret=jnp.asarray(0.75, dtype=jnp.float32),
        outcome_target=jnp.asarray(2, dtype=jnp.int32),
        outcome_weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert jnp.isfinite(policy_loss)
    assert jnp.isfinite(value_loss)
    assert jnp.isfinite(entropy)
    assert jnp.isfinite(outcome_loss)
    assert float(outcome_loss) > 0.0


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


def test_load_or_create_adaptive_network_expands_channels_without_changing_outputs(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    source_channels = (16, 16, 16, 8)
    target_channels = (24, 24, 24, 12)
    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=source_channels)
    model_path = tmp_path / "adaptive-source.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, expected_value = source.logits_value(obs_arr, mask, active)
    expanded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=target_channels,
        init_channels=source_channels,
    )
    actual_logits, actual_value = expanded.logits_value(obs_arr, mask, active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert jnp.allclose(actual_value, expected_value, atol=1e-5)
    assert jnp.any(jnp.abs(expanded.conv1.weight[source_channels[0] :]) > 0.0)
    assert jnp.any(jnp.abs(expanded.conv2.weight[source_channels[1] :]) > 0.0)
    assert jnp.allclose(expanded.policy_conv.weight[:, source_channels[3] :], 0.0)


def test_load_or_create_adaptive_network_adds_zero_context_residual_without_changing_outputs(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    channels = (16, 16, 16, 8)
    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=channels)
    model_path = tmp_path / "adaptive-source.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, expected_value = source.logits_value(obs_arr, mask, active)
    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=channels,
        init_channels=channels,
        context_residual=True,
        init_context_residual=False,
    )
    actual_logits, actual_value = loaded.logits_value(obs_arr, mask, active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert jnp.allclose(actual_value, expected_value, atol=1e-5)
    assert loaded.context_residual
    assert jnp.allclose(loaded.context_conv2.weight, 0.0)
    assert jnp.allclose(loaded.context_conv2.bias, 0.0)


def test_load_or_create_adaptive_network_adds_zero_pyramid_context_without_changing_outputs(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    channels = (16, 16, 16, 8)
    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=channels)
    model_path = tmp_path / "adaptive-source.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, expected_value = source.logits_value(obs_arr, mask, active)
    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=channels,
        init_channels=channels,
        pyramid_context=True,
        init_pyramid_context=False,
    )
    actual_logits, actual_value = loaded.logits_value(obs_arr, mask, active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert jnp.allclose(actual_value, expected_value, atol=1e-5)
    assert loaded.pyramid_context
    assert jnp.allclose(loaded.pyramid_up2.weight, 0.0)
    assert jnp.allclose(loaded.pyramid_up2.bias, 0.0)


def test_load_or_create_adaptive_network_warm_starts_global_context_from_legacy_input(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import (
        ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        ADAPTIVE_INPUT_CHANNELS,
        adaptive_obs_to_array,
        compute_adaptive_valid_move_mask,
    )
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    model_path = tmp_path / "adaptive-legacy.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=(16, 16, 16, 8),
        init_channels=(16, 16, 16, 8),
        input_channels=ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        init_input_channels=ADAPTIVE_INPUT_CHANNELS,
        global_context=True,
        init_global_context=False,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    legacy_obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6)
    global_obs_arr, global_active = adaptive_obs_to_array(
        obs,
        effective_size=4,
        pad_size=6,
        include_global_context=True,
    )
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, expected_value = source.logits_value(legacy_obs_arr, mask, active)
    actual_logits, actual_value = loaded.logits_value(global_obs_arr, mask, global_active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert jnp.allclose(actual_value, expected_value, atol=1e-5)
    assert jnp.allclose(loaded.conv1.weight[:, ADAPTIVE_INPUT_CHANNELS:], 0.0)
    assert loaded.global_context


def test_load_or_create_adaptive_network_warm_starts_history_from_global_context(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import (
        ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        ADAPTIVE_HISTORY_INPUT_CHANNELS,
        adaptive_obs_to_array,
        compute_adaptive_valid_move_mask,
    )
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    source = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        input_channels=ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        global_context=True,
    )
    model_path = tmp_path / "adaptive-global.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=(16, 16, 16, 8),
        init_channels=(16, 16, 16, 8),
        input_channels=ADAPTIVE_HISTORY_INPUT_CHANNELS,
        init_input_channels=ADAPTIVE_GLOBAL_INPUT_CHANNELS,
        global_context=True,
        init_global_context=True,
    )
    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    global_obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=6, include_global_context=True)
    history_obs_arr, history_active = adaptive_obs_to_array(
        obs,
        effective_size=4,
        pad_size=6,
        include_global_context=True,
        scoreboard_history=jnp.zeros((10,), dtype=jnp.float32),
    )
    mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, effective_size=4, pad_size=6)

    expected_logits, expected_value = source.logits_value(global_obs_arr, mask, active)
    actual_logits, actual_value = loaded.logits_value(history_obs_arr, mask, history_active)

    assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
    assert jnp.allclose(actual_value, expected_value, atol=1e-5)
    assert jnp.allclose(loaded.conv1.weight[:, ADAPTIVE_GLOBAL_INPUT_CHANNELS:], 0.0)
    assert loaded.global_linear1.weight.shape[1] == ADAPTIVE_HISTORY_INPUT_CHANNELS - 13


def test_load_or_create_adaptive_network_copies_shared_value_into_per_size_heads(tmp_path):
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import adaptive_obs_to_array, compute_adaptive_valid_move_mask
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork, load_or_create_adaptive_network

    source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    model_path = tmp_path / "adaptive-shared-value.eqx"
    eqx.tree_serialise_leaves(model_path, source)

    loaded = load_or_create_adaptive_network(
        jrandom.PRNGKey(1),
        pad_size=6,
        init_model_path=model_path,
        channels=(16, 16, 16, 8),
        init_channels=(16, 16, 16, 8),
        value_head_sizes=(4, 6),
        init_value_head_sizes=(),
    )

    for size in (4, 6):
        state = make_padded_state(size=size, pad_to=6)
        obs = game.get_observation(state, 0)
        obs_arr, active = adaptive_obs_to_array(obs, effective_size=size, pad_size=6)
        mask = compute_adaptive_valid_move_mask(state.armies, obs.owned_cells, obs.mountains, size, pad_size=6)
        expected_logits, expected_value = source.logits_value(obs_arr, mask, active)
        actual_logits, actual_value = loaded.logits_value(obs_arr, mask, active)

        assert jnp.allclose(actual_logits, expected_logits, atol=1e-5)
        assert jnp.allclose(actual_value, expected_value, atol=1e-5)


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


def test_make_adaptive_state_pool_uses_grid_size_weights():
    from examples._experimental.ppo.adaptive_common import make_adaptive_state_pool

    pool = make_adaptive_state_pool(
        jrandom.PRNGKey(2),
        pool_size=8,
        grid_sizes=(4, 6, 8),
        pad_size=8,
        map_generator="simple",
        mountain_density_range=(0.0, 0.0),
        num_cities_range=(2, 2),
        max_generals_distance=None,
        castle_val_range=(10, 11),
        grid_size_weights=(1.0, 1.0, 2.0),
    )

    assert sorted(pool.effective_sizes.tolist()) == [4, 4, 6, 6, 8, 8, 8, 8]


def test_apply_truncation_reward_penalizes_only_truncated_rows():
    from examples._experimental.ppo.train_adaptive import apply_truncation_reward

    rewards = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float32)
    truncated = jnp.array([True, False, True])

    shaped = apply_truncation_reward(rewards, truncated, 0.5)

    assert jnp.allclose(shaped, jnp.array([0.5, 0.5, -0.75], dtype=jnp.float32))
    assert jnp.allclose(apply_truncation_reward(rewards, truncated, 0.0), rewards)


def test_apply_reward_mode_can_disable_dense_composite_rewards():
    from examples._experimental.ppo.train_adaptive import REWARD_MODE_NAME_TO_ID, apply_reward_mode

    dense = jnp.array([0.25, -0.5, 1.0], dtype=jnp.float32)

    assert jnp.allclose(apply_reward_mode(dense, REWARD_MODE_NAME_TO_ID["composite"]), dense)
    assert jnp.allclose(apply_reward_mode(dense, REWARD_MODE_NAME_TO_ID["terminal"]), jnp.zeros_like(dense))


def test_top_advantage_weights_selects_highest_fraction():
    from examples._experimental.ppo.train_adaptive import top_advantage_weights

    advantages = jnp.array([-2.0, 0.5, 3.0, 1.0, -0.25, 2.0], dtype=jnp.float32)

    weights = top_advantage_weights(advantages, 0.25)

    assert weights.dtype == jnp.float32
    assert jnp.allclose(weights, jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=jnp.float32))
    assert jnp.allclose(top_advantage_weights(advantages, 1.0), jnp.ones_like(advantages))


def test_teacher_obs_from_student_obs_removes_fog_memory_before_history():
    from examples._experimental.ppo.train_adaptive import teacher_obs_from_student_obs

    obs = jnp.arange(35 * 2 * 2, dtype=jnp.float32).reshape(35, 2, 2)
    teacher_obs = teacher_obs_from_student_obs(obs, 30)

    assert teacher_obs.shape == (30, 2, 2)
    assert jnp.array_equal(teacher_obs[:15], obs[:15])
    assert jnp.array_equal(teacher_obs[15:], obs[20:])


def test_update_ema_network_averages_trainable_arrays():
    import jax.tree_util as jtu

    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.train_adaptive import update_ema_network

    ema_source = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    current = AdaptivePolicyValueNetwork(jrandom.PRNGKey(1), pad_size=6, channels=(16, 16, 16, 8))

    updated = update_ema_network(ema_source, current, 0.25)

    source_arrays = jtu.tree_leaves(jax.tree.map(lambda x: x, ema_source, is_leaf=lambda x: isinstance(x, jnp.ndarray)))
    current_arrays = jtu.tree_leaves(jax.tree.map(lambda x: x, current, is_leaf=lambda x: isinstance(x, jnp.ndarray)))
    updated_arrays = jtu.tree_leaves(jax.tree.map(lambda x: x, updated, is_leaf=lambda x: isinstance(x, jnp.ndarray)))

    for before, now, actual in zip(source_arrays, current_arrays, updated_arrays, strict=True):
        if isinstance(before, jnp.ndarray) and jnp.issubdtype(before.dtype, jnp.inexact):
            assert jnp.allclose(actual, before * 0.25 + now * 0.75)


def test_split_mixed_env_counts_preserves_total_and_balances_seats():
    import pytest

    from examples._experimental.ppo.train_adaptive import split_mixed_env_counts

    assert split_mixed_env_counts(2) == (1, 1)
    assert split_mixed_env_counts(5) == (2, 3)

    with pytest.raises(ValueError):
        split_mixed_env_counts(1)


def test_rollout_outcome_targets_use_next_known_episode_result():
    from examples._experimental.ppo.train_adaptive import (
        OUTCOME_DRAW,
        OUTCOME_LOSS,
        OUTCOME_WIN,
        rollout_outcome_targets,
    )

    winners = jnp.array(
        [
            [-1, -1, -1],
            [0, -1, -1],
            [-1, -1, 0],
            [-1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    dones = jnp.array(
        [
            [False, False, False],
            [True, False, False],
            [False, True, True],
            [False, False, False],
        ]
    )
    learner_players = jnp.array([0, 1, 1], dtype=jnp.int32)

    targets, weights = rollout_outcome_targets(winners, dones, learner_players)

    assert targets.shape == winners.shape
    assert weights.shape == winners.shape
    assert targets[:, 0].tolist() == [OUTCOME_WIN, OUTCOME_WIN, OUTCOME_LOSS, OUTCOME_LOSS]
    assert weights[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert targets[:, 1].tolist() == [OUTCOME_DRAW, OUTCOME_DRAW, OUTCOME_DRAW, OUTCOME_LOSS]
    assert weights[:, 1].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert targets[:, 2].tolist() == [OUTCOME_LOSS, OUTCOME_LOSS, OUTCOME_LOSS, OUTCOME_LOSS]
    assert weights[:, 2].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_collect_mixed_rollout_combines_both_learner_seats():
    from examples._experimental.ppo.adaptive_common import make_adaptive_initial_states, make_adaptive_state_pool
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.common import OPPONENT_NAME_TO_ID
    from examples._experimental.ppo.train_adaptive import REWARD_MODE_NAME_TO_ID, collect_mixed_rollout

    pad_size = 6
    pool = make_adaptive_state_pool(
        jrandom.PRNGKey(0),
        pool_size=4,
        grid_sizes=(4, 6),
        pad_size=pad_size,
        map_generator="simple",
        mountain_density_range=(0.0, 0.0),
        num_cities_range=(2, 2),
        max_generals_distance=None,
        castle_val_range=(10, 11),
    )
    states_p0, sizes_p0 = make_adaptive_initial_states(pool, 1)
    states_p1, sizes_p1 = make_adaptive_initial_states(pool, 1)
    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(1), pad_size=pad_size, channels=(16, 16, 16, 8))

    _, _, history_p0, _, _, history_p1, batch, _ = collect_mixed_rollout(
        states_p0,
        sizes_p0,
        states_p1,
        sizes_p1,
        pool,
        network,
        jrandom.PRNGKey(2),
        num_steps=1,
        truncation=20,
        opponent_id=OPPONENT_NAME_TO_ID["random"],
        reward_mode_id=REWARD_MODE_NAME_TO_ID["composite"],
        terminal_reward_scale=0.0,
        truncation_reward_scale=0.0,
        pad_size=pad_size,
    )

    obs, masks, active, actions, logprobs, values, rewards, dones, infos = batch
    assert obs.shape[:2] == (1, 2)
    assert masks.shape == (1, 2, pad_size, pad_size, 4)
    assert active.shape == (1, 2, pad_size, pad_size)
    assert actions.shape == (1, 2, 5)
    assert logprobs.shape == (1, 2)
    assert values.shape == (1, 2)
    assert rewards.shape == (1, 2)
    assert dones.shape == (1, 2)
    assert infos.winner.shape == (1, 2)
    assert history_p0.shape == (1, 5)
    assert history_p1.shape == (1, 5)


def test_adaptive_expander_target_probs_has_single_pass_slot():
    from examples._experimental.ppo.adaptive_common import adaptive_expander_target_probs

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    target = adaptive_expander_target_probs(obs, effective_size=4, pad_size=6)

    assert target.shape == (8 * 6 * 6 + 1,)
    assert jnp.isclose(jnp.sum(target), 1.0)


def test_adaptive_soft_conservative_loss_is_finite_for_matching_networks():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import (
        compute_adaptive_soft_conservative_loss,
        search_score_target_probs,
    )

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    obs = jnp.zeros((2, ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    masks = jnp.ones((2, 6, 6, 4), dtype=bool)
    active = jnp.ones((2, 6, 6), dtype=bool)
    candidate_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    search_scores = jnp.array([[1.0, 2.0], [4.0, 4.0]], dtype=jnp.float32)
    target_probs = search_score_target_probs(search_scores, temperature=1.0)
    search_weights = jnp.ones((2,), dtype=jnp.float32)
    improvement_extra_weights = jnp.zeros((2,), dtype=jnp.float32)
    search_value_targets = jnp.zeros((2,), dtype=jnp.float32)
    search_value_weights = jnp.ones((2,), dtype=jnp.float32)
    search_outcome_targets = jnp.ones((2,), dtype=jnp.int32)
    search_outcome_weights = jnp.ones((2,), dtype=jnp.float32)
    kl_weights = jnp.ones((2,), dtype=jnp.float32)

    loss, metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=1.0,
        improve_weight=0.05,
        improvement_extra_weight=0.0,
        search_value_weight=0.0,
        search_outcome_weight=0.0,
        temperature=1.0,
    )

    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["kl_loss"])
    assert jnp.isfinite(metrics["improve_loss"])
    assert jnp.allclose(jnp.sum(target_probs, axis=1), jnp.ones((2,), dtype=jnp.float32))


def test_strategy_candidate_q_target_values_can_use_outcomes():
    from examples._experimental.ppo.adaptive_search_distill import (
        STRATEGY_Q_TARGET_NAME_TO_ID,
        strategy_candidate_q_target_values,
    )

    search_scores = jnp.array([[100.0, 0.0, -100.0]], dtype=jnp.float32)
    candidate_outcomes = jnp.array([[2, 1, 0]], dtype=jnp.int32)

    score_targets = strategy_candidate_q_target_values(
        search_scores,
        candidate_outcomes,
        score_scale=100.0,
        target_mode=STRATEGY_Q_TARGET_NAME_TO_ID["score"],
    )
    outcome_targets = strategy_candidate_q_target_values(
        search_scores,
        candidate_outcomes,
        score_scale=100.0,
        target_mode=STRATEGY_Q_TARGET_NAME_TO_ID["outcome"],
    )
    hybrid_targets = strategy_candidate_q_target_values(
        search_scores,
        candidate_outcomes,
        score_scale=100.0,
        target_mode=STRATEGY_Q_TARGET_NAME_TO_ID["outcome-score"],
        outcome_score_weight=0.1,
    )

    assert jnp.allclose(score_targets, jnp.tanh(search_scores / 100.0))
    assert jnp.allclose(outcome_targets, jnp.array([[1.0, 0.0, -1.0]], dtype=jnp.float32))
    assert jnp.allclose(hybrid_targets, outcome_targets + 0.1 * score_targets)


def test_search_q_value_metrics_uses_candidate_outcomes():
    from examples._experimental.ppo.adaptive_strategy_supervised import search_q_value_metrics

    candidate_indices = jnp.array([[1, 2, 3, 4]])
    prior_scores = jnp.array([[0.0, 0.0, 0.0, 0.0]])
    search_scores = jnp.array([[10.0, 20.0, 1000.0, -5.0]])
    search_outcomes = jnp.array([[1, 2, 0, -1]])
    score_gaps = jnp.array([1.0])
    correct_q = jnp.array([[0.0, 0.0, 1.0, -1.0, 0.5]])
    reversed_q = jnp.array([[0.0, -1.0, 0.0, 1.0, 0.5]])

    correct_loss, correct_acc, correct_weight = search_q_value_metrics(
        correct_q,
        candidate_indices,
        prior_scores,
        search_scores,
        search_outcomes,
        score_gaps,
        1000.0,
        0.0,
    )
    reversed_loss, reversed_acc, reversed_weight = search_q_value_metrics(
        reversed_q,
        candidate_indices,
        prior_scores,
        search_scores,
        search_outcomes,
        score_gaps,
        1000.0,
        0.0,
    )

    assert correct_weight == 1.0
    assert reversed_weight == 1.0
    assert correct_loss < reversed_loss
    assert correct_acc == 1.0
    assert reversed_acc == 0.0


def test_strategy_dataset_action_weights_can_use_search_best_wins(tmp_path):
    import numpy as np

    from examples._experimental.ppo.adaptive_strategy_supervised import OUTCOME_DRAW, OUTCOME_WIN, load_strategy_dataset

    path = tmp_path / "strategy.npz"
    samples = 4
    pad_size = 2
    action_count = 8 * pad_size * pad_size + 1
    np.savez_compressed(
        path,
        obs=np.zeros((samples, 1, pad_size, pad_size), dtype=np.float16),
        legal_mask=np.ones((samples, pad_size, pad_size, 4), dtype=np.bool_),
        active=np.ones((samples, pad_size, pad_size), dtype=np.bool_),
        intent=np.zeros((samples,), dtype=np.int8),
        outcome=np.array([OUTCOME_WIN, OUTCOME_DRAW, OUTCOME_DRAW, OUTCOME_WIN], dtype=np.int8),
        outcome_known=np.ones((samples,), dtype=np.float16),
        finish_within_250=np.array([1, 0, 0, 1], dtype=np.float16),
        enemy_general_heatmap=np.zeros((samples, pad_size, pad_size), dtype=np.float16),
        source_heatmap=np.zeros((samples, pad_size, pad_size), dtype=np.float16),
        target_heatmap=np.zeros((samples, pad_size, pad_size), dtype=np.float16),
        teacher_logits=np.zeros((samples, action_count), dtype=np.float16),
        teacher_action_index=np.arange(samples, dtype=np.int32),
        grid_size=np.full((samples,), pad_size, dtype=np.int16),
        seat=np.array([0, 0, 1, 1], dtype=np.int8),
        search_candidate_indices=np.zeros((samples, 2), dtype=np.int32),
        search_prior_scores=np.zeros((samples, 2), dtype=np.float16),
        search_scores=np.zeros((samples, 2), dtype=np.float16),
        search_outcomes=np.zeros((samples, 2), dtype=np.int8),
        search_score_gap=np.ones((samples,), dtype=np.float16),
        search_best_outcome=np.array([OUTCOME_DRAW, OUTCOME_WIN, -1, OUTCOME_WIN], dtype=np.int8),
    )

    dataset = load_strategy_dataset([path], action_ce_weight_mode="search-best-win")

    assert jnp.allclose(dataset["action_weight"], jnp.array([0.0, 1.0, 0.0, 1.0]))

    draw_search_win = load_strategy_dataset(
        [path],
        require_outcome_draw=True,
        require_search_best_win=True,
    )
    assert draw_search_win["obs"].shape[0] == 1
    assert int(draw_search_win["teacher_action"][0]) == 1

    nonwin_search_win = load_strategy_dataset(
        [path],
        require_outcome_nonwin=True,
        require_search_best_win=True,
    )
    assert nonwin_search_win["obs"].shape[0] == 1
    assert int(nonwin_search_win["teacher_action"][0]) == 1


def test_accepted_replacement_weights_prefer_outcome_then_score():
    from examples._experimental.ppo.adaptive_search_distill import accepted_replacement_weights

    candidate_indices = jnp.array(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=jnp.int32,
    )
    search_scores = jnp.array(
        [
            [0.0, -20.0, -50.0],
            [10.0, 30.0, 12.0],
            [50.0, 40.0, 35.0],
        ],
        dtype=jnp.float32,
    )
    candidate_outcomes = jnp.array(
        [
            [1, 2, 1],
            [1, 1, 1],
            [2, 2, 2],
        ],
        dtype=jnp.int32,
    )
    weights = accepted_replacement_weights(
        candidate_indices,
        search_scores,
        candidate_outcomes,
        active_weights=jnp.ones((3,), dtype=jnp.float32),
        min_margin=10.0,
        margin_scale=20.0,
        max_weight=1.0,
    )

    assert jnp.allclose(weights, jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32))


def test_strategy_q_replay_keeps_q_weights_only():
    from examples._experimental.ppo.adaptive_search_distill import (
        SOFT_KL_WEIGHT_INDEX,
        SOFT_SEARCH_WEIGHT_INDEX,
        SOFT_STRATEGY_BELIEF_WEIGHT_INDEX,
        SOFT_STRATEGY_Q_WEIGHT_INDEX,
        augment_with_strategy_q_replay,
        update_strategy_q_replay,
    )

    batch_size = 4
    flat_batch = tuple(jnp.arange(batch_size, dtype=jnp.float32) + index for index in range(23))
    flat_batch = list(flat_batch)
    flat_batch[SOFT_STRATEGY_Q_WEIGHT_INDEX] = jnp.array([0.0, 1.0, 0.0, 2.0], dtype=jnp.float32)
    flat_batch[SOFT_SEARCH_WEIGHT_INDEX] = jnp.ones((batch_size,), dtype=jnp.float32)
    flat_batch[SOFT_KL_WEIGHT_INDEX] = jnp.ones((batch_size,), dtype=jnp.float32)
    flat_batch[SOFT_STRATEGY_BELIEF_WEIGHT_INDEX] = jnp.ones((batch_size,), dtype=jnp.float32)
    flat_batch = tuple(flat_batch)

    replay, new_rows = update_strategy_q_replay(None, flat_batch, capacity=8)
    augmented, sampled_rows = augment_with_strategy_q_replay(
        flat_batch,
        replay,
        jrandom.PRNGKey(0),
        replay_ratio=0.5,
    )

    assert new_rows == 2
    assert sampled_rows == 2
    assert replay[0].shape[0] == 2
    assert augmented[0].shape[0] == 6
    assert jnp.all(augmented[SOFT_STRATEGY_Q_WEIGHT_INDEX][-2:] > 0.0)
    assert jnp.allclose(augmented[SOFT_SEARCH_WEIGHT_INDEX][-2:], 0.0)
    assert jnp.allclose(augmented[SOFT_KL_WEIGHT_INDEX][-2:], 0.0)
    assert jnp.allclose(augmented[SOFT_STRATEGY_BELIEF_WEIGHT_INDEX][-2:], 0.0)


def test_context_strategy_aux_grad_mask_preserves_context_and_aux_only():
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import mask_context_strategy_aux_grads

    grads = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        strategy_aux=True,
        context_residual=True,
    )

    masked = mask_context_strategy_aux_grads(grads)

    assert jnp.allclose(masked.conv4.weight, 0.0)
    assert jnp.allclose(masked.policy_conv.weight, 0.0)
    assert jnp.allclose(masked.context_conv1.weight, grads.context_conv1.weight)
    assert jnp.allclose(masked.strategy_intent_linear2.weight, grads.strategy_intent_linear2.weight)
    assert jnp.allclose(masked.strategy_enemy_general_conv.weight, grads.strategy_enemy_general_conv.weight)


def test_train_adaptive_context_only_grad_mask_supports_pyramid_without_residual():
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.train_adaptive import context_only_grad_tree

    grads = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        pyramid_context=True,
    )

    masked = context_only_grad_tree(grads)

    assert masked.context_conv1 is None
    assert jnp.allclose(masked.conv4.weight, 0.0)
    assert jnp.allclose(masked.policy_conv.weight, 0.0)
    assert jnp.allclose(masked.pyramid_down1.weight, grads.pyramid_down1.weight)
    assert jnp.allclose(masked.pyramid_up2.weight, grads.pyramid_up2.weight)


def test_context_strategy_aux_grad_mask_preserves_pyramid_and_aux_only():
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import mask_context_strategy_aux_grads

    grads = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        strategy_aux=True,
        pyramid_context=True,
    )

    masked = mask_context_strategy_aux_grads(grads)

    assert masked.context_conv1 is None
    assert jnp.allclose(masked.conv4.weight, 0.0)
    assert jnp.allclose(masked.policy_conv.weight, 0.0)
    assert jnp.allclose(masked.pyramid_down1.weight, grads.pyramid_down1.weight)
    assert jnp.allclose(masked.strategy_finish_linear2.weight, grads.strategy_finish_linear2.weight)


def test_adaptive_soft_loss_can_add_extra_improvement_term():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import (
        compute_adaptive_soft_conservative_loss,
        search_score_target_probs,
    )

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    obs = jnp.zeros((2, ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    masks = jnp.ones((2, 6, 6, 4), dtype=bool)
    active = jnp.ones((2, 6, 6), dtype=bool)
    candidate_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    target_probs = search_score_target_probs(jnp.array([[1.0, 3.0], [4.0, 4.0]], dtype=jnp.float32), temperature=1.0)
    search_weights = jnp.ones((2,), dtype=jnp.float32)
    improvement_extra_weights = jnp.array([1.0, 0.0], dtype=jnp.float32)
    search_value_targets = jnp.zeros((2,), dtype=jnp.float32)
    search_value_weights = jnp.ones((2,), dtype=jnp.float32)
    search_outcome_targets = jnp.ones((2,), dtype=jnp.int32)
    search_outcome_weights = jnp.ones((2,), dtype=jnp.float32)
    kl_weights = jnp.ones((2,), dtype=jnp.float32)

    base_loss, base_metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.05,
        improvement_extra_weight=0.0,
        search_value_weight=0.0,
        search_outcome_weight=0.0,
        temperature=1.0,
    )
    mixed_loss, mixed_metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.05,
        improvement_extra_weight=0.1,
        search_value_weight=0.0,
        search_outcome_weight=0.0,
        temperature=1.0,
    )

    assert float(mixed_loss) > float(base_loss)
    assert float(base_metrics["improvement_extra_loss"]) == 0.0
    assert float(mixed_metrics["improvement_extra_loss"]) > 0.0


def test_adaptive_soft_loss_can_add_search_value_target():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import (
        compute_adaptive_soft_conservative_loss,
        search_score_target_probs,
    )

    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8))
    obs = jnp.zeros((2, ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    masks = jnp.ones((2, 6, 6, 4), dtype=bool)
    active = jnp.ones((2, 6, 6), dtype=bool)
    candidate_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    target_probs = search_score_target_probs(jnp.array([[1.0, 3.0], [4.0, 4.0]], dtype=jnp.float32), temperature=1.0)
    search_weights = jnp.ones((2,), dtype=jnp.float32)
    improvement_extra_weights = jnp.zeros((2,), dtype=jnp.float32)
    kl_weights = jnp.ones((2,), dtype=jnp.float32)
    search_value_targets = jnp.array([0.75, -0.5], dtype=jnp.float32)
    search_value_weights = jnp.ones((2,), dtype=jnp.float32)
    search_outcome_targets = jnp.ones((2,), dtype=jnp.int32)
    search_outcome_weights = jnp.ones((2,), dtype=jnp.float32)

    base_loss, base_metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.0,
        improvement_extra_weight=0.0,
        search_value_weight=0.0,
        search_outcome_weight=0.0,
        temperature=1.0,
    )
    value_loss, value_metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.0,
        improvement_extra_weight=0.0,
        search_value_weight=0.5,
        search_outcome_weight=0.0,
        temperature=1.0,
    )

    assert float(base_loss) == 0.0
    assert float(base_metrics["search_value_loss"]) == 0.0
    assert float(value_loss) > 0.0
    assert float(value_metrics["search_value_loss"]) > 0.0
    assert jnp.isfinite(value_metrics["mean_search_value_target"])


def test_adaptive_soft_loss_can_add_search_outcome_target():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import (
        compute_adaptive_soft_conservative_loss,
        search_score_target_probs,
    )

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        outcome_head=True,
    )
    obs = jnp.zeros((2, ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    masks = jnp.ones((2, 6, 6, 4), dtype=bool)
    active = jnp.ones((2, 6, 6), dtype=bool)
    candidate_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    target_probs = search_score_target_probs(jnp.array([[1.0, 3.0], [4.0, 4.0]], dtype=jnp.float32), temperature=1.0)
    search_weights = jnp.ones((2,), dtype=jnp.float32)
    improvement_extra_weights = jnp.zeros((2,), dtype=jnp.float32)
    search_value_targets = jnp.zeros((2,), dtype=jnp.float32)
    search_value_weights = jnp.ones((2,), dtype=jnp.float32)
    search_outcome_targets = jnp.array([2, 0], dtype=jnp.int32)
    search_outcome_weights = jnp.ones((2,), dtype=jnp.float32)
    kl_weights = jnp.ones((2,), dtype=jnp.float32)

    base_loss, base_metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.0,
        improvement_extra_weight=0.0,
        search_value_weight=0.0,
        search_outcome_weight=0.0,
        temperature=1.0,
    )
    outcome_loss, outcome_metrics = compute_adaptive_soft_conservative_loss(
        network,
        network,
        obs,
        masks,
        active,
        obs,
        masks,
        active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.0,
        improvement_extra_weight=0.0,
        search_value_weight=0.0,
        search_outcome_weight=0.5,
        temperature=1.0,
    )

    assert float(base_loss) == 0.0
    assert float(base_metrics["search_outcome_loss"]) == 0.0
    assert float(outcome_loss) > 0.0
    assert float(outcome_metrics["search_outcome_loss"]) > 0.0
    assert jnp.isfinite(outcome_metrics["search_outcome_accuracy"])


def test_strategy_aux_targets_include_finish_intent_and_enemy_general_belief():
    from examples._experimental.ppo.adaptive_search_distill import OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_WIN
    from examples._experimental.ppo.adaptive_strategy_aux import (
        STRATEGY_INTENT_EXPAND,
        STRATEGY_INTENT_FINISH,
        strategy_aux_targets,
    )

    state = make_padded_state(size=4, pad_to=6)
    obs = game.get_observation(state, 0)
    search_scores = jnp.array([1.0, 5.0, 2.0], dtype=jnp.float32)
    search_outcomes = jnp.array([OUTCOME_DRAW, OUTCOME_WIN, OUTCOME_LOSS], dtype=jnp.int32)

    targets = strategy_aux_targets(
        state,
        obs,
        learner_player=0,
        effective_size=4,
        pad_size=6,
        search_scores=search_scores,
        search_outcomes=search_outcomes,
    )

    assert targets.intent == STRATEGY_INTENT_FINISH
    assert targets.finish == 1
    assert targets.enemy_general_heatmap.shape == (6, 6)
    assert targets.enemy_general_heatmap[3, 3] == 1.0
    assert jnp.sum(targets.enemy_general_heatmap) == 1.0

    no_finish_targets = strategy_aux_targets(
        state,
        obs,
        learner_player=0,
        effective_size=4,
        pad_size=6,
        search_scores=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        search_outcomes=jnp.array([OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_DRAW], dtype=jnp.int32),
    )

    assert no_finish_targets.intent == STRATEGY_INTENT_EXPAND
    assert no_finish_targets.finish == 0


def test_adaptive_network_strategy_aux_outputs_expected_shapes():
    from examples._experimental.ppo.adaptive_common import (
        adaptive_obs_to_array,
        compute_adaptive_valid_move_mask,
    )
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    pad_size = 6
    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=pad_size,
        channels=(16, 16, 16, 8),
        strategy_aux=True,
    )
    state = make_padded_state(size=4, pad_to=pad_size)
    obs = game.get_observation(state, 0)
    obs_arr, active = adaptive_obs_to_array(obs, effective_size=4, pad_size=pad_size)
    mask = compute_adaptive_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains, 4, pad_size)

    outputs = network.strategy_auxiliary(obs_arr, mask, active)

    assert outputs.intent_logits.shape == (8,)
    assert outputs.finish_logits.shape == (2,)
    assert outputs.enemy_general_logits.shape == (pad_size, pad_size)
    assert outputs.action_q_values.shape == (8 * pad_size * pad_size + 1,)


def test_strategy_aux_loss_uses_q_intent_finish_and_belief_targets():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import compute_strategy_aux_loss

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        channels=(16, 16, 16, 8),
        strategy_aux=True,
    )
    obs = jnp.zeros((2, ADAPTIVE_INPUT_CHANNELS, 6, 6), dtype=jnp.float32)
    masks = jnp.ones((2, 6, 6, 4), dtype=bool)
    active = jnp.ones((2, 6, 6), dtype=bool)
    candidate_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    candidate_q_targets = jnp.array([[0.5, -0.25], [0.1, 0.2]], dtype=jnp.float32)
    sample_weights = jnp.ones((2,), dtype=jnp.float32)
    intent_targets = jnp.array([0, 6], dtype=jnp.int32)
    finish_targets = jnp.array([0, 1], dtype=jnp.int32)
    enemy_general_targets = jnp.zeros((2, 6, 6), dtype=jnp.float32).at[:, 1, 2].set(1.0)

    loss, metrics = compute_strategy_aux_loss(
        network,
        obs,
        masks,
        active,
        candidate_indices,
        candidate_q_targets,
        sample_weights,
        intent_targets,
        sample_weights,
        finish_targets,
        sample_weights,
        enemy_general_targets,
        sample_weights,
        q_weight=0.5,
        intent_weight=0.25,
        finish_weight=0.25,
        belief_weight=0.1,
    )

    assert float(loss) > 0.0
    assert float(metrics["strategy_q_loss"]) > 0.0
    assert float(metrics["strategy_intent_loss"]) > 0.0
    assert float(metrics["strategy_finish_loss"]) > 0.0
    assert float(metrics["strategy_belief_loss"]) > 0.0


def test_strategy_q_pairwise_rank_loss_rewards_correct_candidate_order():
    from examples._experimental.ppo.adaptive_search_distill import strategy_q_pairwise_rank_loss

    target_q = jnp.array([[1.0, -1.0, 0.9]], dtype=jnp.float32)
    correct_pred = jnp.array([[2.0, -2.0, 0.0]], dtype=jnp.float32)
    reversed_pred = jnp.array([[-2.0, 2.0, 0.0]], dtype=jnp.float32)
    weights = jnp.ones((1,), dtype=jnp.float32)

    correct_loss = strategy_q_pairwise_rank_loss(correct_pred, target_q, weights, min_margin=0.25)
    reversed_loss = strategy_q_pairwise_rank_loss(reversed_pred, target_q, weights, min_margin=0.25)
    filtered_loss = strategy_q_pairwise_rank_loss(correct_pred, target_q, weights, min_margin=5.0)

    assert float(correct_loss) < float(reversed_loss)
    assert float(filtered_loss) == 0.0


def test_soft_search_weights_can_select_only_search_improvements():
    from examples._experimental.ppo.adaptive_search_distill import (
        SOFT_WEIGHT_MODE_NAME_TO_ID,
        soft_search_weights,
    )

    candidate_indices = jnp.array(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=jnp.int32,
    )
    search_scores = jnp.array(
        [
            [5.0, 8.5, 7.0],
            [9.0, 8.0, 7.0],
            [1.0, 2.2, 2.0],
        ],
        dtype=jnp.float32,
    )
    candidate_outcomes = jnp.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [1, 2, 1],
        ],
        dtype=jnp.int32,
    )
    active_weights = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)

    active = soft_search_weights(
        candidate_indices,
        search_scores,
        candidate_outcomes,
        active_weights,
        SOFT_WEIGHT_MODE_NAME_TO_ID["active"],
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
    )
    improvement = soft_search_weights(
        candidate_indices,
        search_scores,
        candidate_outcomes,
        active_weights,
        SOFT_WEIGHT_MODE_NAME_TO_ID["improvement"],
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
    )
    accepted = soft_search_weights(
        candidate_indices,
        search_scores,
        candidate_outcomes,
        active_weights,
        SOFT_WEIGHT_MODE_NAME_TO_ID["accepted"],
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
    )

    assert jnp.allclose(active, active_weights)
    assert jnp.allclose(improvement, jnp.array([0.375, 0.0, 0.0], dtype=jnp.float32))
    assert jnp.allclose(accepted, jnp.array([0.375, 1.0, 0.0], dtype=jnp.float32))


def test_adaptive_rollout_search_candidates_respects_effective_size():
    from examples._experimental.ppo.adaptive_common import adaptive_action_space_size
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import adaptive_rollout_search_candidates

    pad_size = 6
    effective_size = 4
    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=pad_size, channels=(16, 16, 16, 8))
    state = make_padded_state(size=effective_size, pad_to=pad_size)

    candidate_actions, candidate_indices, prior_scores, search_scores, search_outcomes = adaptive_rollout_search_candidates(
        network,
        state,
        jnp.asarray(effective_size, dtype=jnp.int32),
        jrandom.PRNGKey(1),
        player=0,
        top_k=2,
        rollout_steps=1,
        rollouts_per_action=1,
        policy_mode=0,
        army_weight=12.0,
        land_weight=8.0,
        prior_weight=0.01,
        terminal_score=1000.0,
        pad_size=pad_size,
    )

    assert candidate_actions.shape == (2, 5)
    assert candidate_indices.shape == (2,)
    assert prior_scores.shape == (2,)
    assert search_scores.shape == (2,)
    assert search_outcomes.shape == (2,)
    assert jnp.all(candidate_indices >= 0)
    assert jnp.all(candidate_indices < adaptive_action_space_size(pad_size))
    assert jnp.all((search_outcomes >= 0) & (search_outcomes <= 2))
    non_pass = candidate_actions[:, 0] == 0
    assert jnp.all((candidate_actions[:, 1] < effective_size) | ~non_pass)
    assert jnp.all((candidate_actions[:, 2] < effective_size) | ~non_pass)
    assert jnp.all(jnp.isfinite(search_scores))


def test_collect_adaptive_soft_batch_returns_expected_shapes():
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import collect_adaptive_soft_batch

    pad_size = 6
    num_envs = 2
    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=pad_size, channels=(16, 16, 16, 8))
    states = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        make_padded_state(size=4, pad_to=pad_size),
        make_padded_state(size=6, pad_to=pad_size),
    )
    effective_sizes = jnp.array([4, 6], dtype=jnp.int32)

    _, batch, _ = collect_adaptive_soft_batch(
        network,
        network,
        network,
        states,
        effective_sizes,
        jrandom.PRNGKey(2),
        num_steps=1,
        policy_mode=0,
        opponent_policy_mode=0,
        learner_player=0,
        top_k=2,
        rollout_steps=1,
        rollouts_per_action=1,
        army_weight=12.0,
        land_weight=8.0,
        prior_weight=0.01,
        terminal_score=1000.0,
        soft_weight_mode=0,
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
        score_temperature=1.0,
        search_value_scale=100.0,
        pad_size=pad_size,
    )

    (
        obs,
        masks,
        active,
        base_obs,
        base_masks,
        base_active,
        candidate_indices,
        target_probs,
        search_weights,
        improvement_extra_weights,
        search_value_targets,
        search_value_weights,
        search_outcome_targets,
        search_outcome_weights,
        kl_weights,
        strategy_candidate_q_targets,
        strategy_q_weights,
        strategy_intent_targets,
        strategy_intent_weights,
        strategy_finish_targets,
        strategy_finish_weights,
        strategy_enemy_general_targets,
        strategy_belief_weights,
    ) = batch
    assert obs.shape[:2] == (1, num_envs)
    assert masks.shape == (1, num_envs, pad_size, pad_size, 4)
    assert active.shape == (1, num_envs, pad_size, pad_size)
    assert base_obs.shape == obs.shape
    assert base_masks.shape == masks.shape
    assert base_active.shape == active.shape
    assert candidate_indices.shape == (1, num_envs, 2)
    assert target_probs.shape == (1, num_envs, 2)
    assert search_weights.shape == (1, num_envs)
    assert improvement_extra_weights.shape == (1, num_envs)
    assert search_value_targets.shape == (1, num_envs)
    assert search_value_weights.shape == (1, num_envs)
    assert search_outcome_targets.shape == (1, num_envs)
    assert search_outcome_weights.shape == (1, num_envs)
    assert kl_weights.shape == (1, num_envs)
    assert strategy_candidate_q_targets.shape == (1, num_envs, 2)
    assert strategy_q_weights.shape == (1, num_envs)
    assert strategy_intent_targets.shape == (1, num_envs)
    assert strategy_intent_weights.shape == (1, num_envs)
    assert strategy_finish_targets.shape == (1, num_envs)
    assert strategy_finish_weights.shape == (1, num_envs)
    assert strategy_enemy_general_targets.shape == (1, num_envs, pad_size, pad_size)
    assert strategy_belief_weights.shape == (1, num_envs)
    assert jnp.allclose(jnp.sum(target_probs, axis=-1), jnp.ones((1, num_envs), dtype=jnp.float32))


def test_collect_adaptive_soft_batch_supports_history_base_network():
    from examples._experimental.ppo.adaptive_common import ADAPTIVE_HISTORY_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import collect_adaptive_soft_batch

    pad_size = 6
    num_envs = 2
    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=pad_size,
        input_channels=ADAPTIVE_HISTORY_INPUT_CHANNELS,
        channels=(16, 16, 16, 8),
        global_context=True,
    )
    states = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        make_padded_state(size=4, pad_to=pad_size),
        make_padded_state(size=6, pad_to=pad_size),
    )
    effective_sizes = jnp.array([4, 6], dtype=jnp.int32)

    _, batch, _ = collect_adaptive_soft_batch(
        network,
        network,
        network,
        states,
        effective_sizes,
        jrandom.PRNGKey(3),
        num_steps=1,
        policy_mode=0,
        opponent_policy_mode=0,
        learner_player=0,
        top_k=2,
        rollout_steps=1,
        rollouts_per_action=1,
        army_weight=12.0,
        land_weight=8.0,
        prior_weight=0.01,
        terminal_score=1000.0,
        soft_weight_mode=0,
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
        score_temperature=1.0,
        search_value_scale=100.0,
        pad_size=pad_size,
        global_context=True,
        scoreboard_history_enabled=True,
        base_global_context=True,
        base_scoreboard_history_enabled=True,
    )

    obs, _, _, base_obs, *_ = batch
    assert obs.shape == (1, num_envs, ADAPTIVE_HISTORY_INPUT_CHANNELS, pad_size, pad_size)
    assert base_obs.shape == obs.shape


def test_mask_strategy_aux_grads_keeps_only_strategy_heads():
    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import ADAPTIVE_HISTORY_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import mask_strategy_aux_grads

    network = AdaptivePolicyValueNetwork(
        jrandom.PRNGKey(0),
        pad_size=6,
        input_channels=ADAPTIVE_HISTORY_INPUT_CHANNELS,
        channels=(16, 16, 16, 8),
        strategy_aux=True,
        global_context=True,
    )
    grads = jax.tree.map(lambda leaf: jnp.ones_like(leaf) if eqx.is_inexact_array(leaf) else leaf, network)
    masked = mask_strategy_aux_grads(grads)

    assert jnp.all(masked.conv1.weight == 0)
    assert jnp.all(masked.policy_conv.weight == 0)
    assert jnp.all(masked.value_linear1.weight == 0)
    assert jnp.all(masked.global_linear1.weight == 0)
    assert jnp.any(masked.strategy_intent_linear2.weight != 0)
    assert jnp.any(masked.strategy_finish_linear2.weight != 0)
    assert jnp.any(masked.strategy_q_conv.weight != 0)
    assert jnp.any(masked.strategy_q_pass_linear.weight != 0)
    assert jnp.any(masked.strategy_enemy_general_conv.weight != 0)


def test_worker_bfs_label_moves_toward_general_target():
    from examples._experimental.ppo.adaptive_common import adaptive_index_to_action
    from examples._experimental.ppo.adaptive_worker_pretrain import (
        WORKER_INPUT_CHANNELS,
        worker_bfs_action_index,
        worker_obs_to_array,
    )

    pad_size = 6
    state = make_padded_state(size=4, pad_to=pad_size)
    state = state._replace(armies=state.armies.at[0, 0].set(8))
    obs = game.get_observation(state, 0)

    obs_arr, active = worker_obs_to_array(
        state,
        obs,
        player=0,
        effective_size=4,
        pad_size=pad_size,
        target_family=0,
        min_army=2,
    )
    index, weight = worker_bfs_action_index(
        state,
        player=0,
        effective_size=4,
        pad_size=pad_size,
        target_family=0,
        min_army=2,
    )
    action = adaptive_index_to_action(index, pad_size)

    assert obs_arr.shape == (WORKER_INPUT_CHANNELS, pad_size, pad_size)
    assert active.shape == (pad_size, pad_size)
    assert weight == 1.0
    assert action.tolist() == [0, 0, 0, 1, 0]


def test_worker_command_obs_to_array_uses_observation_command_channels():
    from examples._experimental.ppo.adaptive_worker_pretrain import (
        WORKER_COMMAND_NAME_TO_ID,
        WORKER_INPUT_CHANNELS,
        worker_command_obs_to_array,
    )

    pad_size = 6
    state = make_padded_state(size=4, pad_to=pad_size)
    obs = game.get_observation(state, 0)

    obs_arr, active = worker_command_obs_to_array(
        obs,
        effective_size=4,
        pad_size=pad_size,
        command_mode=WORKER_COMMAND_NAME_TO_ID["frontier"],
        min_army=2,
    )

    assert obs_arr.shape == (WORKER_INPUT_CHANNELS, pad_size, pad_size)
    assert active.shape == (pad_size, pad_size)
    assert jnp.any(obs_arr[-3] > 0)
    assert jnp.all(obs_arr[-3, 4:, :] == 0)
    assert jnp.all(obs_arr[-3, :, 4:] == 0)


def test_worker_source_direction_targets_marginalize_full_move_planes():
    from examples._experimental.ppo.adaptive_common import adaptive_action_space_size
    from examples._experimental.ppo.adaptive_worker_pretrain import worker_source_direction_targets

    pad_size = 3
    targets = jnp.zeros((1, adaptive_action_space_size(pad_size)), dtype=jnp.float32)
    right_from_origin = 0 * pad_size * pad_size + 0 * pad_size + 0
    down_from_center = 2 * pad_size * pad_size + 1 * pad_size + 1
    targets = targets.at[0, right_from_origin].set(0.25)
    targets = targets.at[0, down_from_center].set(0.75)

    source_targets, direction_targets = worker_source_direction_targets(targets, pad_size)

    assert source_targets.shape == (1, pad_size * pad_size)
    assert direction_targets.shape == (1, 4)
    assert source_targets[0, 0] == 0.25
    assert source_targets[0, 4] == 0.75
    assert direction_targets[0, 0] == 0.25
    assert direction_targets[0, 2] == 0.75
    assert jnp.sum(source_targets) == 1.0
    assert jnp.sum(direction_targets) == 1.0


def test_worker_rerank_logits_centers_legal_worker_bias_only_when_triggered():
    from examples._experimental.ppo.evaluate_worker_policy import worker_rerank_logits

    fallback_logits = jnp.array([[1.0, 2.0, -1.0e9]], dtype=jnp.float32)
    worker_logits = jnp.array([[4.0, 8.0, -1.0e9]], dtype=jnp.float32)

    no_trigger = worker_rerank_logits(fallback_logits, worker_logits, jnp.array([False]), scale=0.5)
    zero_scale = worker_rerank_logits(fallback_logits, worker_logits, jnp.array([True]), scale=0.0)
    reranked = worker_rerank_logits(fallback_logits, worker_logits, jnp.array([True]), scale=0.5)

    assert jnp.allclose(no_trigger, fallback_logits)
    assert jnp.allclose(zero_scale, fallback_logits)
    assert jnp.allclose(reranked[0, :2], jnp.array([0.0, 3.0], dtype=jnp.float32))
    assert reranked[0, 2] < -1.0e8


def test_strategy_q_rerank_logits_centers_legal_q_bias():
    from examples._experimental.ppo.evaluate_adaptive_policy import strategy_q_rerank_logits

    policy_logits = jnp.array([[1.0, 2.0, -1.0e9]], dtype=jnp.float32)
    action_q_values = jnp.array([[0.0, 4.0, 100.0]], dtype=jnp.float32)

    zero_scale = strategy_q_rerank_logits(policy_logits, action_q_values, scale=0.0)
    reranked = strategy_q_rerank_logits(policy_logits, action_q_values, scale=0.5)

    assert jnp.allclose(zero_scale, policy_logits)
    assert jnp.allclose(reranked[0, :2], jnp.array([0.0, 3.0], dtype=jnp.float32))
    assert reranked[0, 2] < -1.0e8


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
        "--grid-size-weights",
        "4:1,6:2",
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
        "--channels",
        "16,16,16,8",
        "--model-path",
        str(model_path),
        "--seed",
        "41000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()


def test_behavior_clone_adaptive_saves_and_prunes_checkpoints(tmp_path):
    import os
    import subprocess
    import sys

    model_path = tmp_path / "adaptive-bc.eqx"
    checkpoint_dir = tmp_path / "bc-ckpts"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/behavior_clone_adaptive.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--grid-size-weights",
        "4:1,6:2",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--num-steps",
        "1",
        "--num-iterations",
        "3",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "1",
        "--keep-checkpoints",
        "2",
        "--model-path",
        str(model_path),
        "--seed",
        "41500",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
    assert sorted(path.name for path in checkpoint_dir.glob("*.eqx")) == [
        "adaptive-bc-iter-000002.eqx",
        "adaptive-bc-iter-000003.eqx",
    ]


def test_train_adaptive_cli_smoke(tmp_path):
    import os
    import subprocess
    import sys

    import equinox as eqx

    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    init_model_path = tmp_path / "adaptive-init.eqx"
    model_path = tmp_path / "adaptive-ppo.eqx"
    checkpoint_dir = tmp_path / "ckpts"
    eqx.tree_serialise_leaves(
        init_model_path,
        AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8)),
    )
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
        "--truncation-reward-scale",
        "0.25",
        "--reward-mode",
        "terminal",
        "--gamma",
        "1.0",
        "--gae-lambda",
        "0.9",
        "--top-advantage-fraction",
        "0.5",
        "--ema-decay",
        "0.5",
        "--eval-ema",
        "--learner-player",
        "mixed",
        "--channels",
        "16,16,16,8",
        "--init-channels",
        "16,16,16,8",
        "--init-model-path",
        str(init_model_path),
        "--global-context",
        "--scoreboard-history",
        "--init-input-channels",
        "15",
        "--value-heads",
        "per-size",
        "--init-value-heads",
        "shared",
        "--value-loss",
        "hl-gauss",
        "--init-value-loss",
        "mse",
        "--value-bins",
        "8",
        "--value-sigma",
        "0.2",
        "--outcome-aux-weight",
        "0.1",
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


def test_adaptive_search_distill_cli_smoke_saves_and_prunes_checkpoints(tmp_path):
    import os
    import subprocess
    import sys

    import equinox as eqx

    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    base_model_path = tmp_path / "adaptive-base.eqx"
    model_path = tmp_path / "adaptive-search-distill.eqx"
    checkpoint_dir = tmp_path / "search-ckpts"
    eqx.tree_serialise_leaves(
        base_model_path,
        AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=6, channels=(16, 16, 16, 8)),
    )

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/adaptive_search_distill.py",
        "2",
        "--grid-sizes",
        "4,6",
        "--grid-size-weights",
        "4:1,6:2",
        "--pad-to",
        "6",
        "--map-generator",
        "simple",
        "--pool-size",
        "4",
        "--base-model-path",
        str(base_model_path),
        "--model-path",
        str(model_path),
        "--target-mode",
        "soft",
        "--soft-weight-mode",
        "improvement",
        "--soft-improvement-extra-weight",
        "0.1",
        "--search-outcome-weight",
        "0.1",
        "--strategy-q-weight",
        "0.1",
        "--strategy-intent-weight",
        "0.05",
        "--strategy-finish-weight",
        "0.05",
        "--strategy-belief-weight",
        "0.02",
        "--learner-player",
        "mixed",
        "--num-steps",
        "1",
        "--num-iterations",
        "3",
        "--num-epochs",
        "1",
        "--minibatch-size",
        "2",
        "--top-k",
        "2",
        "--rollout-steps",
        "1",
        "--rollouts-per-action",
        "1",
        "--channels",
        "16,16,16,8",
        "--base-channels",
        "16,16,16,8",
        "--init-channels",
        "16,16,16,8",
        "--scoreboard-history",
        "--init-input-channels",
        "15",
        "--freeze-legacy-weights",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "1",
        "--keep-checkpoints",
        "2",
        "--seed",
        "44000",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
    assert sorted(path.name for path in checkpoint_dir.glob("*.eqx")) == [
        "adaptive-search-distill-iter-000002.eqx",
        "adaptive-search-distill-iter-000003.eqx",
    ]


def test_evaluate_adaptive_policy_cli_writes_size_rows(tmp_path):
    import json
    import os
    import subprocess
    import sys

    import equinox as eqx

    from examples._experimental.ppo.adaptive_common import ADAPTIVE_HISTORY_INPUT_CHANNELS
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    model_path = tmp_path / "adaptive.eqx"
    output_path = tmp_path / "adaptive-eval.json"
    eqx.tree_serialise_leaves(
        model_path,
        AdaptivePolicyValueNetwork(
            jrandom.PRNGKey(0),
            pad_size=6,
            input_channels=ADAPTIVE_HISTORY_INPUT_CHANNELS,
            global_context=True,
            value_head_sizes=(4, 6),
            value_bins=8,
            value_sigma=0.2,
            outcome_head=True,
        ),
    )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/evaluate_adaptive_policy.py",
        str(model_path),
        "--grid-sizes",
        "4,6",
        "--pad-to",
        "6",
        "--num-games",
        "2",
        "--max-steps",
        "4",
        "--map-generator",
        "simple",
        "--global-context",
        "--scoreboard-history",
        "--value-heads",
        "per-size",
        "--value-loss",
        "hl-gauss",
        "--value-bins",
        "8",
        "--value-sigma",
        "0.2",
        "--outcome-head",
        "--json-output",
        str(output_path),
        "--seed",
        "43000",
    ]

    completed = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert "adaptive policy evaluation" in completed.stdout.lower()
    assert len(data["rows"]) == 4
    assert {row["grid_size"] for row in data["rows"]} == {4, 6}
    assert {row["policy_player"] for row in data["rows"]} == {0, 1}
