from pathlib import Path


def test_v5_launch_script_is_executable_and_points_to_checkpoint():
    script = Path("play-v5.command")

    assert script.exists()
    assert script.stat().st_mode & 0o111

    text = script.read_text()
    assert "generals-ppo-8x8-expander-gpu-v5.eqx" in text
    assert "uv run --python 3.12 python examples/play_against_model.py" in text
    assert "POLICY_INPUT" in text
    assert "--policy-input" in text
    assert "auto" in text
    assert "--grid-size 8" in text
    assert "--map-generator generated" in text
    assert "--policy-mode sample" in text
    assert "--auto-tick" in text
    assert "--tick-rate 2" in text


def test_v5_watch_script_is_executable_and_starts_machine_match():
    script = Path("watch-v5.command")

    assert script.exists()
    assert script.stat().st_mode & 0o111

    text = script.read_text()
    assert "generals-ppo-8x8-expander-gpu-v5.eqx" in text
    assert "uv run --python 3.12 python examples/play_against_model.py" in text
    assert "--machine-vs-machine" in text
    assert "--model-0-path" in text
    assert "--model-1-path" in text
    assert "MODEL_0_POLICY_INPUT" in text
    assert "MODEL_1_POLICY_INPUT" in text
    assert "--model-0-policy-input" in text
    assert "--model-1-policy-input" in text
    assert "auto" in text
    assert "--policy-mode sample" in text
    assert "--opponent-policy-mode sample" in text
    assert "--auto-tick" in text
    assert "--tick-rate 4" in text
