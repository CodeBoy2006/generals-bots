"use strict";

(function initKeyboardModule(root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
    return;
  }
  root.GeneralsKeyboard = factory();
})(typeof globalThis !== "undefined" ? globalThis : window, function buildKeyboardModule() {
  const DIRECTIONS = {
    w: [-1, 0],
    arrowup: [-1, 0],
    s: [1, 0],
    arrowdown: [1, 0],
    a: [0, -1],
    arrowleft: [0, -1],
    d: [0, 1],
    arrowright: [0, 1],
  };

  const EDITING_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA"]);

  function resolveKeyboardCommand(event, snapshot, options = {}) {
    if (isEditingTarget(event && event.target)) {
      return null;
    }
    if (!snapshot || snapshot.game_done || snapshot.mode === "machine-vs-machine") {
      return null;
    }

    const key = normalizeKey(event);
    if (DIRECTIONS[key]) {
      return resolveMoveKey(DIRECTIONS[key], snapshot, Boolean(options.splitEnabled));
    }
    if (key === " " || key === "space" || key === "spacebar") {
      return commandResult({ type: "cancel" });
    }
    if (key === "z") {
      return commandResult({ type: "set_split", enabled: !Boolean(options.splitEnabled) });
    }
    if (key === "e") {
      return commandResult({ type: "undo_queue" });
    }
    if (key === "q") {
      const hasQueue = Array.isArray(snapshot.queued_moves) && snapshot.queued_moves.length > 0;
      return commandResult({ type: hasQueue ? "clear_queue" : "pass" });
    }

    return null;
  }

  function resolveMoveKey(delta, snapshot, splitEnabled) {
    const selected = snapshot.selected_cell;
    if (!Array.isArray(selected) || selected.length !== 2) {
      return commandResult(null, "Select a movable tile first");
    }

    const target = [selected[0] + delta[0], selected[1] + delta[1]];
    if (!isValidTarget(target, snapshot.valid_targets)) {
      return commandResult(null, "Invalid direction");
    }

    return commandResult({
      type: "move",
      source: selected,
      target,
      split: splitEnabled,
    });
  }

  function isValidTarget(target, validTargets) {
    if (!Array.isArray(validTargets)) {
      return false;
    }
    return validTargets.some((cell) => Array.isArray(cell) && cell[0] === target[0] && cell[1] === target[1]);
  }

  function commandResult(command, message = null) {
    return {
      command,
      message,
      preventDefault: true,
    };
  }

  function normalizeKey(event) {
    if (!event || typeof event.key !== "string") {
      return "";
    }
    return event.key.length === 1 ? event.key.toLowerCase() : event.key.toLowerCase();
  }

  function isEditingTarget(target) {
    if (!target) {
      return false;
    }
    if (target.isContentEditable) {
      return true;
    }
    return EDITING_TAGS.has(String(target.tagName || "").toUpperCase());
  }

  return {
    resolveKeyboardCommand,
  };
});
