"""JSON-safe snapshot helpers for the browser renderer."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from generals.core import game

VisibilityMode = Literal["human-vs-model", "machine-vs-machine"]


def _bool_grid(array: Any) -> list[list[bool]]:
    return np.asarray(array, dtype=bool).tolist()


def _int_grid(array: Any) -> list[list[int]]:
    return np.asarray(array, dtype=int).tolist()


def _cell_or_none(cell: tuple[int, int] | None) -> list[int] | None:
    if cell is None:
        return None
    row, col = cell
    return [int(row), int(col)]


def _cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[list[int]]:
    return [[int(row), int(col)] for row, col in cells]


def _queued_moves(moves: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
    serialized = []
    for move in moves or []:
        serialized.append(
            {
                "source": move.get("source"),
                "target": move.get("target"),
                "split": bool(move.get("split", False)),
                "is_pass": bool(move.get("is_pass", False)),
            }
        )
    return serialized


def _model_catalog(models: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, str]]:
    return [
        {"id": str(model["id"]), "label": str(model["label"]), "path": str(model["path"])}
        for model in models or []
    ]


def encode_ownership(state: game.GameState) -> list[list[int]]:
    """Return one ownership grid using -1 for neutral/unowned cells."""
    ownership = np.full(np.asarray(state.armies).shape, -1, dtype=int)
    p0 = np.asarray(state.ownership[0], dtype=bool)
    p1 = np.asarray(state.ownership[1], dtype=bool)
    ownership[p0] = 0
    ownership[p1] = 1
    return ownership.tolist()


def visibility_for_mode(state: game.GameState, visibility_player: int | None) -> list[list[bool]]:
    """Return full visibility for watch mode or fogged visibility for one player."""
    if visibility_player is None:
        return np.ones(np.asarray(state.armies).shape, dtype=bool).tolist()
    visibility = game.get_visibility(state.ownership[int(visibility_player)])
    return _bool_grid(visibility)


def serialize_policy_preview(preview: Any | None) -> dict[str, Any] | None:
    """Convert a PolicyPreview-like object into JSON-safe values."""
    if preview is None:
        return None

    candidates = []
    for candidate in getattr(preview, "candidates", ()):
        candidates.append(
            {
                "action": [int(value) for value in getattr(candidate, "action")],
                "probability": float(getattr(candidate, "probability")),
                "source": _cell_or_none(getattr(candidate, "source", None)),
                "target": _cell_or_none(getattr(candidate, "target", None)),
                "direction": None
                if getattr(candidate, "direction", None) is None
                else int(getattr(candidate, "direction")),
                "direction_label": str(getattr(candidate, "direction_label")),
                "is_split": bool(getattr(candidate, "is_split")),
                "is_pass": bool(getattr(candidate, "is_pass")),
            }
        )

    return {
        "candidates": candidates,
        "value": float(getattr(preview, "value")),
        "policy_mode": str(getattr(preview, "policy_mode")),
    }


def build_snapshot(
    *,
    state: game.GameState,
    info: game.GameInfo,
    names: list[str],
    colors: list[str],
    mode: VisibilityMode,
    visibility_player: int | None,
    step_count: int,
    selected_cell: tuple[int, int] | None,
    split_enabled: bool,
    last_message: str,
    auto_tick_enabled: bool,
    tick_rate: float,
    policy_preview: Any | None,
    valid_targets: list[tuple[int, int]],
    reached_limit: bool,
    queued_moves: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    player_controls: list[str] | tuple[str, ...] | None = None,
    player_model_ids: list[str | None] | tuple[str | None, ...] | None = None,
    model_catalog: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    active_human_player: int | None = None,
) -> dict[str, Any]:
    """Build one complete browser snapshot from authoritative game state."""
    armies = np.asarray(state.armies)
    winner = int(np.asarray(info.winner))
    game_done = bool(np.asarray(info.is_done)) or bool(reached_limit)
    army = np.asarray(info.army)
    land = np.asarray(info.land)

    return {
        "type": "snapshot",
        "mode": mode,
        "grid": {
            "height": int(armies.shape[0]),
            "width": int(armies.shape[1]),
            "armies": _int_grid(state.armies),
            "ownership": encode_ownership(state),
            "mountains": _bool_grid(state.mountains),
            "cities": _bool_grid(state.cities),
            "generals": _bool_grid(state.generals),
            "visible": visibility_for_mode(state, visibility_player),
        },
        "players": [
            {
                "index": int(index),
                "name": str(name),
                "army": int(army[index]),
                "land": int(land[index]),
                "color": str(colors[index]),
                "control": str((player_controls or ("model", "model"))[index]),
                "model_id": None
                if (player_model_ids or (None, None))[index] is None
                else str((player_model_ids or (None, None))[index]),
            }
            for index, name in enumerate(names)
        ],
        "time": int(np.asarray(info.time)),
        "step_count": int(step_count),
        "winner": None if winner < 0 else winner,
        "game_done": game_done,
        "selected_cell": _cell_or_none(selected_cell),
        "valid_targets": _cells(valid_targets),
        "queued_moves": _queued_moves(queued_moves),
        "active_human_player": None if active_human_player is None else int(active_human_player),
        "model_catalog": _model_catalog(model_catalog),
        "split_enabled": bool(split_enabled),
        "last_message": str(last_message),
        "auto_tick": {"enabled": bool(auto_tick_enabled), "tick_rate": float(tick_rate)},
        "policy_preview": serialize_policy_preview(policy_preview),
    }
