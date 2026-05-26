import numpy as np

from generals.gui.rendering import get_valid_target_cells


def test_valid_target_cells_are_adjacent_in_bounds_and_passable():
    mountains = np.zeros((4, 4), dtype=bool)
    mountains[1, 0] = True

    targets = get_valid_target_cells((0, 0), mountains)

    assert targets == [(0, 1)]


def test_valid_target_cells_include_all_passable_cardinal_neighbors():
    mountains = np.zeros((4, 4), dtype=bool)

    targets = get_valid_target_cells((2, 2), mountains)

    assert targets == [(1, 2), (3, 2), (2, 1), (2, 3)]


def test_valid_target_cells_without_selection_is_empty():
    mountains = np.zeros((4, 4), dtype=bool)

    assert get_valid_target_cells(None, mountains) == []
