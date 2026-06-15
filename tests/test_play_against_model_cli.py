import pytest

from examples.play_against_model import parse_args


def parse_with_args(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["play_against_model.py", "policy.eqx", *args])
    return parse_args()


def test_parse_args_rejects_nonpositive_min_generals_distance(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--min-generals-distance", "0")


def test_parse_args_rejects_nonpositive_max_generals_distance(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--max-generals-distance", "0")


def test_parse_args_rejects_min_generals_distance_above_max(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--min-generals-distance", "5", "--max-generals-distance", "4")


def test_parse_args_rejects_default_min_generals_distance_above_max(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--grid-size", "8", "--max-generals-distance", "3")


def test_parse_args_accepts_valid_generals_distance(monkeypatch):
    args = parse_with_args(monkeypatch, "--min-generals-distance", "3", "--max-generals-distance", "5")

    assert args.effective_min_generals_distance == 3
    assert args.max_generals_distance == 5


def test_parse_args_accepts_preview_options(monkeypatch):
    args = parse_with_args(monkeypatch, "--preview-top-k", "5", "--no-ai-preview")

    assert args.preview_top_k == 5
    assert args.ai_preview is False


def test_parse_args_defaults_to_ai_preview(monkeypatch):
    args = parse_with_args(monkeypatch)

    assert args.preview_top_k == 3
    assert args.ai_preview is True


def test_parse_args_rejects_preview_top_k_below_range(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--preview-top-k", "0")


def test_parse_args_rejects_preview_top_k_above_range(monkeypatch):
    with pytest.raises(SystemExit):
        parse_with_args(monkeypatch, "--preview-top-k", "6")
