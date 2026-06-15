import numpy as np

from generals.agents.ppo_policy_agent import PolicyActionCandidate, PolicyPreview
from generals.gui.rendering import get_valid_target_cells
from generals.gui.rendering import format_policy_preview_lines, get_policy_candidate_arrow


def test_valid_target_cells_are_adjacent_in_bounds_and_passable():
    mountains = np.zeros((4, 4), dtype=bool)
    mountains[1, 0] = True

    targets = get_valid_target_cells((0, 0), mountains)

    assert targets == [(0, 1)]


def test_valid_target_cells_include_all_passable_cardinal_neighbors():
    mountains = np.zeros((4, 4), dtype=bool)

    targets = get_valid_target_cells((2, 2), mountains)

    assert len(targets) == 4
    assert set(targets) == {(1, 2), (3, 2), (2, 1), (2, 3)}


def test_valid_target_cells_without_selection_is_empty():
    mountains = np.zeros((4, 4), dtype=bool)

    assert get_valid_target_cells(None, mountains) == []


def test_policy_candidate_arrow_uses_source_and_target_for_moves():
    candidate = PolicyActionCandidate(
        action=(0, 1, 1, 3, 0),
        probability=0.49,
        source=(1, 1),
        target=(1, 2),
        direction=3,
        direction_label="Right",
        is_split=False,
        is_pass=False,
    )

    assert get_policy_candidate_arrow(candidate) == ((1, 1), (1, 2))


def test_policy_candidate_arrow_skips_pass_candidates():
    candidate = PolicyActionCandidate(
        action=(1, 0, 0, 0, 0),
        probability=0.09,
        source=None,
        target=None,
        direction=None,
        direction_label="Pass",
        is_split=False,
        is_pass=True,
    )

    assert get_policy_candidate_arrow(candidate) is None


def test_policy_preview_lines_include_move_pass_value_and_sample_note():
    move = PolicyActionCandidate(
        action=(0, 1, 1, 3, 1),
        probability=0.49,
        source=(1, 1),
        target=(1, 2),
        direction=3,
        direction_label="Right",
        is_split=True,
        is_pass=False,
    )
    pass_action = PolicyActionCandidate(
        action=(1, 0, 0, 0, 0),
        probability=0.09,
        source=None,
        target=None,
        direction=None,
        direction_label="Pass",
        is_split=False,
        is_pass=True,
    )
    preview = PolicyPreview(candidates=(move, pass_action), value=0.18, policy_mode="sample")

    lines = format_policy_preview_lines(preview)

    assert "AI Preview" in lines
    assert any("(1, 1)->(1, 2) Right split 49%" in line for line in lines)
    assert any("Pass 9%" in line for line in lines)
    assert "Value: +0.18" in lines
    assert "Sample mode: action is sampled" in lines
