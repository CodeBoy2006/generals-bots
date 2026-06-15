import jax.numpy as jnp
import jax.random as jrandom

from examples._experimental.ppo.search_policy import rollout_search_action, rollout_search_candidates, score_observation
from generals.agents.ppo_policy_agent import PolicyValueNetwork
from generals.core import game
from generals.core.game import GameInfo
from generals.core.observation import Observation


def _observation(owned_army, opponent_army, owned_land, opponent_land):
    board = jnp.zeros((4, 4), dtype=jnp.int32)
    mask = jnp.zeros((4, 4), dtype=bool)
    return Observation(
        armies=board,
        generals=mask,
        cities=mask,
        mountains=mask,
        neutral_cells=mask,
        owned_cells=mask,
        opponent_cells=mask,
        fog_cells=mask,
        structures_in_fog=mask,
        owned_land_count=jnp.int32(owned_land),
        owned_army_count=jnp.int32(owned_army),
        opponent_land_count=jnp.int32(opponent_land),
        opponent_army_count=jnp.int32(opponent_army),
        timestep=jnp.int32(0),
    )


def _info(winner):
    return GameInfo(
        army=jnp.zeros((2,), dtype=jnp.int32),
        land=jnp.zeros((2,), dtype=jnp.int32),
        is_done=winner >= 0,
        winner=jnp.int32(winner),
        time=jnp.int32(10),
    )


def test_score_observation_prefers_wins_and_material_advantage():
    neutral = score_observation(_info(-1), _observation(20, 20, 8, 8), player=0)
    material = score_observation(_info(-1), _observation(40, 20, 12, 6), player=0)
    win = score_observation(_info(0), _observation(5, 40, 4, 20), player=0)
    loss = score_observation(_info(1), _observation(40, 5, 20, 4), player=0)

    assert material > neutral
    assert win > material
    assert loss < neutral


def test_rollout_search_candidates_return_scored_policy_prior_actions():
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(6))

    actions, indices, prior_scores, search_scores = rollout_search_candidates(
        network,
        state,
        jrandom.PRNGKey(1),
        0,
        2,
        1,
        1,
        1,
        12.0,
        8.0,
        0.01,
    )
    chosen = rollout_search_action(
        network,
        state,
        jrandom.PRNGKey(1),
        0,
        2,
        1,
        1,
        1,
        12.0,
        8.0,
        0.01,
    )

    assert actions.shape == (2, 5)
    assert indices.shape == (2,)
    assert prior_scores.shape == (2,)
    assert search_scores.shape == (2,)
    assert jnp.array_equal(chosen, actions[jnp.argmax(search_scores)])
