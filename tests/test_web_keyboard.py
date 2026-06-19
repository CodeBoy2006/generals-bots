import subprocess


def test_keyboard_resolver_matches_generals_io_core_bindings():
    script = r"""
const assert = require("assert");
const keyboard = require("./generals/web/static/keyboard.js");

const baseSnapshot = {
  mode: "human-vs-model",
  game_done: false,
  selected_cell: [2, 2],
  valid_targets: [[1, 2], [3, 2], [2, 1], [2, 3]],
  queued_moves: [],
};

function resolve(key, snapshot = baseSnapshot, options = {}) {
  return keyboard.resolveKeyboardCommand(
    { key, target: { tagName: "BODY" } },
    snapshot,
    { splitEnabled: false, ...options }
  );
}

assert.deepStrictEqual(resolve("w").command, { type: "move", source: [2, 2], target: [1, 2], split: false });
assert.deepStrictEqual(resolve("ArrowRight", baseSnapshot, { splitEnabled: true }).command, {
  type: "move",
  source: [2, 2],
  target: [2, 3],
  split: true,
});
assert.deepStrictEqual(resolve(" ").command, { type: "cancel" });
assert.deepStrictEqual(resolve("z").command, { type: "set_split", enabled: true });
assert.deepStrictEqual(resolve("e").command, { type: "undo_queue" });
assert.deepStrictEqual(resolve("q", { ...baseSnapshot, queued_moves: [{ source: [2, 2], target: [2, 3] }] }).command, {
  type: "clear_queue",
});
assert.deepStrictEqual(resolve("q").command, { type: "pass" });

assert.strictEqual(
  keyboard.resolveKeyboardCommand({ key: "w", target: { tagName: "INPUT" } }, baseSnapshot, { splitEnabled: false }),
  null
);
assert.strictEqual(resolve("w", { ...baseSnapshot, mode: "machine-vs-machine" }), null);
assert.strictEqual(resolve("w", { ...baseSnapshot, selected_cell: null }).command, null);
"""

    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
