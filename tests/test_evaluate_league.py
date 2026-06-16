import json
import os
import subprocess
import sys

import equinox as eqx
import jax.random as jrandom

from examples._experimental.ppo.evaluate_league import (
    REQUIRED_HEURISTIC_OPPONENTS,
    LeagueRow,
    compute_league_summary,
    parse_checkpoint_specs,
)
from examples._experimental.ppo.network import PolicyValueNetwork


def test_parse_checkpoint_specs_accepts_name_path_mode():
    specs = parse_checkpoint_specs(["v5=generals-ppo-8x8-expander-gpu-v5.eqx:sample"])
    assert specs == [("v5", "generals-ppo-8x8-expander-gpu-v5.eqx", "sample")]


def test_parse_checkpoint_specs_defaults_name_and_mode():
    specs = parse_checkpoint_specs(["generals-ppo-8x8-expander-gpu-v5.eqx"])
    assert specs == [
        (
            "generals-ppo-8x8-expander-gpu-v5",
            "generals-ppo-8x8-expander-gpu-v5.eqx",
            "sample",
        )
    ]


def test_required_heuristics_excludes_random():
    assert "random" not in REQUIRED_HEURISTIC_OPPONENTS
    assert "expander" in REQUIRED_HEURISTIC_OPPONENTS


def test_compute_league_summary_uses_min_required_win_rate():
    rows = [
        LeagueRow("heuristic", "expander", 0, 90, 10, 0, 100, 120.0, True),
        LeagueRow("heuristic", "expander", 1, 79, 20, 1, 100, 121.0, True),
        LeagueRow("checkpoint", "v5", 0, 85, 15, 0, 100, 122.0, True),
        LeagueRow("checkpoint", "v5", 1, 82, 18, 0, 100, 123.0, True),
        LeagueRow("sanity", "random", 0, 100, 0, 0, 100, 90.0, False),
    ]
    summary = compute_league_summary(rows, threshold=0.8)
    assert summary["required_pairs"] == 4
    assert summary["passed_pairs"] == 3
    assert summary["league_score"] == 0.79
    assert summary["passes_threshold"] is False


def test_evaluate_league_cli_writes_heuristic_rows(tmp_path):
    model_path = tmp_path / "policy.eqx"
    output_path = tmp_path / "league.json"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=8)
    eqx.tree_serialise_leaves(model_path, network)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"

    cmd = [
        sys.executable,
        "examples/_experimental/ppo/evaluate_league.py",
        str(model_path),
        "--heuristic",
        "expander",
        "--num-games",
        "2",
        "--max-steps",
        "4",
        "--grid-size",
        "8",
        "--map-generator",
        "simple",
        "--json-output",
        str(output_path),
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert "league_score" in completed.stdout
    assert data["summary"]["required_pairs"] == 2
    assert len(data["rows"]) == 2
    assert {row["policy_player"] for row in data["rows"]} == {0, 1}


def test_evaluate_league_cli_writes_search_heuristic_rows(tmp_path):
    model_path = tmp_path / "policy.eqx"
    output_path = tmp_path / "search-league.json"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=8)
    eqx.tree_serialise_leaves(model_path, network)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"

    cmd = [
        sys.executable,
        "examples/_experimental/ppo/evaluate_league.py",
        str(model_path),
        "--search-policy",
        "--heuristic",
        "expander",
        "--num-games",
        "2",
        "--max-steps",
        "4",
        "--grid-size",
        "8",
        "--map-generator",
        "simple",
        "--top-k",
        "2",
        "--rollout-steps",
        "2",
        "--rollouts-per-action",
        "1",
        "--json-output",
        str(output_path),
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert "rollout-search" in completed.stdout
    assert data["policy_kind"] == "rollout-search"
    assert data["summary"]["required_pairs"] == 2
    assert len(data["rows"]) == 2
