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

    _, _, _, _, batch, _ = collect_mixed_rollout(
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
        kl_weights,
        kl_weight=1.0,
        improve_weight=0.05,
        improvement_extra_weight=0.0,
        temperature=1.0,
    )

    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["kl_loss"])
    assert jnp.isfinite(metrics["improve_loss"])
    assert jnp.allclose(jnp.sum(target_probs, axis=1), jnp.ones((2,), dtype=jnp.float32))


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
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.05,
        improvement_extra_weight=0.0,
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
        kl_weights,
        kl_weight=0.0,
        improve_weight=0.05,
        improvement_extra_weight=0.1,
        temperature=1.0,
    )

    assert float(mixed_loss) > float(base_loss)
    assert float(base_metrics["improvement_extra_loss"]) == 0.0
    assert float(mixed_metrics["improvement_extra_loss"]) > 0.0


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
    active_weights = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)

    active = soft_search_weights(
        candidate_indices,
        search_scores,
        active_weights,
        SOFT_WEIGHT_MODE_NAME_TO_ID["active"],
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
    )
    improvement = soft_search_weights(
        candidate_indices,
        search_scores,
        active_weights,
        SOFT_WEIGHT_MODE_NAME_TO_ID["improvement"],
        min_margin=2.0,
        margin_scale=4.0,
        max_weight=1.0,
    )

    assert jnp.allclose(active, active_weights)
    assert jnp.allclose(improvement, jnp.array([0.375, 0.0, 0.0], dtype=jnp.float32))


def test_adaptive_rollout_search_candidates_respects_effective_size():
    from examples._experimental.ppo.adaptive_common import adaptive_action_space_size
    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork
    from examples._experimental.ppo.adaptive_search_distill import adaptive_rollout_search_candidates

    pad_size = 6
    effective_size = 4
    network = AdaptivePolicyValueNetwork(jrandom.PRNGKey(0), pad_size=pad_size, channels=(16, 16, 16, 8))
    state = make_padded_state(size=effective_size, pad_to=pad_size)

    candidate_actions, candidate_indices, prior_scores, search_scores = adaptive_rollout_search_candidates(
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
    assert jnp.all(candidate_indices >= 0)
    assert jnp.all(candidate_indices < adaptive_action_space_size(pad_size))
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
        kl_weights,
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
    assert kl_weights.shape == (1, num_envs)
    assert jnp.allclose(jnp.sum(target_probs, axis=-1), jnp.ones((1, num_envs), dtype=jnp.float32))


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
        "--learner-player",
        "1",
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

    from examples._experimental.ppo.adaptive_network import AdaptivePolicyValueNetwork

    model_path = tmp_path / "adaptive.eqx"
    output_path = tmp_path / "adaptive-eval.json"
    eqx.tree_serialise_leaves(
        model_path,
        AdaptivePolicyValueNetwork(
            jrandom.PRNGKey(0),
            pad_size=6,
            value_head_sizes=(4, 6),
            value_bins=8,
            value_sigma=0.2,
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
        "--value-heads",
        "per-size",
        "--value-loss",
        "hl-gauss",
        "--value-bins",
        "8",
        "--value-sigma",
        "0.2",
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
