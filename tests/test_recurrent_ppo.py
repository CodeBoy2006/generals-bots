import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from examples._experimental.ppo.common import POLICY_INPUT_NAME_TO_ID
from examples._experimental.ppo.train_recurrent import (
    checkpoint_path_for_iteration,
    load_or_create_recurrent_network,
    rollout_step_recurrent_heuristic_opponent,
    rollout_step_recurrent_policy_opponent,
    train_recurrent_minibatch_step,
)
from examples._experimental.ppo.evaluate_recurrent_policy import evaluate_recurrent_batch
from examples._experimental.ppo.recurrent_network import RecurrentPolicyValueNetwork
from generals.agents.ppo_policy_agent import PolicyValueNetwork
from generals.core import game


def test_recurrent_checkpoint_path_for_iteration(tmp_path):
    assert checkpoint_path_for_iteration(tmp_path, "rnn", 7) == tmp_path / "rnn-iter-000007.eqx"


def test_train_recurrent_cli_writes_periodic_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    model_path = tmp_path / "rnn-final.eqx"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "examples/_experimental/ppo/train_recurrent.py",
        "4",
        "--grid-size",
        "4",
        "--pool-size",
        "8",
        "--num-steps",
        "2",
        "--num-iterations",
        "2",
        "--num-epochs",
        "1",
        "--minibatch-size",
        "8",
        "--hidden-size",
        "8",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "1",
        "--keep-checkpoints",
        "1",
        "--model-path",
        str(model_path),
        "--seed",
        "30410",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

    assert model_path.exists()
    assert not (checkpoint_dir / "rnn-final-iter-000001.eqx").exists()
    assert (checkpoint_dir / "rnn-final-iter-000002.eqx").exists()


def test_recurrent_network_zero_delta_matches_base_policy_outputs():
    base = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    network = RecurrentPolicyValueNetwork(
        jrandom.PRNGKey(1),
        grid_size=4,
        base_network=base,
        hidden_size=16,
    )
    obs = jnp.zeros((9, 4, 4), dtype=jnp.float32).at[0, 0, 0].set(3.0)
    mask = jnp.ones((4, 4, 4), dtype=bool)
    hidden = network.initial_hidden()

    recurrent_logits, recurrent_value, next_hidden = network.logits_value_hidden(obs, mask, hidden)
    base_logits, base_value = base.logits_value(obs, mask)

    assert next_hidden.shape == (16,)
    assert jnp.allclose(recurrent_logits, base_logits)
    assert jnp.allclose(recurrent_value, base_value)


def test_load_or_create_recurrent_network_warm_starts_from_base_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "base.eqx"
    base = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    eqx.tree_serialise_leaves(checkpoint_path, base)

    network = load_or_create_recurrent_network(
        jrandom.PRNGKey(1),
        grid_size=4,
        init_model_path=checkpoint_path,
        hidden_size=16,
    )
    obs = jnp.zeros((9, 4, 4), dtype=jnp.float32).at[0, 0, 0].set(3.0)
    mask = jnp.ones((4, 4, 4), dtype=bool)
    hidden = network.initial_hidden()

    recurrent_logits, recurrent_value, _ = network.logits_value_hidden(obs, mask, hidden)
    base_logits, base_value = base.logits_value(obs, mask)

    assert jnp.allclose(recurrent_logits, base_logits)
    assert jnp.allclose(recurrent_value, base_value)


def test_rollout_step_recurrent_policy_opponent_tracks_hidden_state():
    learner = RecurrentPolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4, input_channels=18, hidden_size=16)
    opponent = PolicyValueNetwork(jrandom.PRNGKey(1), grid_size=4)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    pool = jax.tree.map(lambda x: jnp.stack([x, x, x, x]), state)
    states = pool._replace(pool_idx=jnp.array([2, 3, 0, 1], dtype=jnp.int32))
    hidden = jnp.zeros((4, learner.hidden_size), dtype=jnp.float32)

    _, next_hidden, batch, _ = rollout_step_recurrent_policy_opponent(
        states,
        pool,
        hidden,
        learner,
        opponent,
        jrandom.PRNGKey(2),
        truncation=20,
        opponent_policy_mode=1,
        learner_player=0,
        terminal_reward_scale=0.0,
        policy_input=POLICY_INPUT_NAME_TO_ID["augmented-full-state"],
        opponent_policy_input=POLICY_INPUT_NAME_TO_ID["observation"],
    )

    obs_arr, masks, hidden_batch = batch[:3]

    assert obs_arr.shape == (4, 18, 4, 4)
    assert masks.shape == (4, 4, 4, 4)
    assert hidden_batch.shape == (4, 16)
    assert next_hidden.shape == (4, 16)


def test_rollout_step_recurrent_heuristic_opponent_tracks_hidden_state():
    learner = RecurrentPolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4, hidden_size=16)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    pool = jax.tree.map(lambda x: jnp.stack([x, x, x, x]), state)
    states = pool._replace(pool_idx=jnp.array([2, 3, 0, 1], dtype=jnp.int32))
    hidden = jnp.zeros((4, learner.hidden_size), dtype=jnp.float32)

    _, next_hidden, batch, _ = rollout_step_recurrent_heuristic_opponent(
        states,
        pool,
        hidden,
        learner,
        jrandom.PRNGKey(2),
        truncation=20,
        opponent_id=0,
        learner_player=0,
        terminal_reward_scale=0.0,
        policy_input=POLICY_INPUT_NAME_TO_ID["observation"],
    )

    obs_arr, masks, hidden_batch = batch[:3]

    assert obs_arr.shape == (4, 9, 4, 4)
    assert masks.shape == (4, 4, 4, 4)
    assert hidden_batch.shape == (4, 16)
    assert next_hidden.shape == (4, 16)


def test_evaluate_recurrent_batch_supports_heuristic_opponent():
    network = RecurrentPolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4, hidden_size=16)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    states = jax.tree.map(lambda x: jnp.stack([x, x]), game.create_initial_state(grid))
    hidden = jnp.zeros((2, network.hidden_size), dtype=jnp.float32)

    info = evaluate_recurrent_batch(
        network,
        states,
        hidden,
        jrandom.PRNGKey(1),
        max_steps=1,
        opponent=0,
        policy_mode=1,
        policy_player=0,
        policy_input=POLICY_INPUT_NAME_TO_ID["observation"],
    )

    assert info.winner.shape == (2,)


def test_train_recurrent_minibatch_step_can_freeze_base_network():
    network = RecurrentPolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4, hidden_size=16)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    obs = jnp.zeros((2, 9, 4, 4), dtype=jnp.float32)
    masks = jnp.ones((2, 4, 4, 4), dtype=bool)
    hiddens = jnp.zeros((2, 16), dtype=jnp.float32)
    actions = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    old_logprobs = jnp.zeros((2,), dtype=jnp.float32)
    advantages = jnp.ones((2,), dtype=jnp.float32)
    returns = jnp.ones((2,), dtype=jnp.float32)

    updated, _, _ = train_recurrent_minibatch_step(
        network,
        opt_state,
        (obs, masks, hiddens, actions, old_logprobs, advantages, returns),
        optimizer,
        freeze_base=True,
    )

    assert jnp.allclose(updated.base.conv1.weight, network.base.conv1.weight)
