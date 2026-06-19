import subprocess


def test_control_panel_render_preserves_live_select_elements_during_ticks():
    script = r"""
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

class FakeClassList {
  toggle() {}
}

class FakeElement {
  constructor(tagName = "div", id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.children = [];
    this.className = "";
    this.textContent = "";
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this.selected = false;
    this.style = {};
    this.classList = new FakeClassList();
    this.listeners = {};
    this.clientWidth = 900;
    this.clientHeight = 900;
  }

  get options() {
    return this.children;
  }

  append(...nodes) {
    for (const node of nodes) {
      this.appendChild(node);
    }
  }

  appendChild(node) {
    this.children.push(node);
    if (this.tagName === "SELECT" && node.selected) {
      this.value = node.value;
    }
    return node;
  }

  replaceChildren(...nodes) {
    this.children = [];
    this.value = "";
    for (const node of nodes) {
      this.appendChild(node);
    }
  }

  addEventListener(type, callback) {
    this.listeners[type] = callback;
  }

  getBoundingClientRect() {
    return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight };
  }

  getContext() {
    return {
      setTransform() {},
      clearRect() {},
      fillRect() {},
      strokeRect() {},
      beginPath() {},
      arc() {},
      fill() {},
      stroke() {},
      save() {},
      restore() {},
      drawImage() {},
      strokeText() {},
      fillText() {},
      moveTo() {},
      lineTo() {},
      closePath() {},
    };
  }
}

const elements = new Map();
const ids = [
  "game-board",
  "connection-dot",
  "mode-label",
  "reconnect-button",
  "restart-button",
  "pass-button",
  "cancel-button",
  "undo-button",
  "clear-queue-button",
  "split-toggle",
  "auto-tick-toggle",
  "tick-rate-input",
  "active-human-select",
  "player-control-list",
  "status-message",
  "turn-value",
  "step-value",
  "winner-value",
  "players-list",
  "queue-count",
  "queue-list",
  "preview-value",
  "preview-list",
];
for (const id of ids) {
  const tag = id === "game-board" ? "canvas" : id.endsWith("select") ? "select" : "div";
  elements.set(id, new FakeElement(tag, id));
}

const fakeDocument = {
  activeElement: null,
  getElementById(id) {
    return elements.get(id);
  },
  createElement(tagName) {
    return new FakeElement(tagName);
  },
};

function FakeImage() {
  this.complete = false;
  this.naturalWidth = 0;
  this.addEventListener = () => {};
}

function FakeWebSocket() {
  this.readyState = FakeWebSocket.OPEN;
  this.addEventListener = () => {};
  this.close = () => {};
  this.send = () => {};
}
FakeWebSocket.OPEN = 1;

const context = {
  console,
  document: fakeDocument,
  window: {
    location: { protocol: "http:", host: "127.0.0.1" },
    clearTimeout() {},
    setTimeout() { return 1; },
    addEventListener() {},
    devicePixelRatio: 1,
    GeneralsKeyboard: null,
  },
  Image: FakeImage,
  WebSocket: FakeWebSocket,
  ResizeObserver: class {
    observe() {}
  },
};
context.globalThis = context;
vm.createContext(context);

const code = fs.readFileSync("generals/web/static/app.js", "utf8");
vm.runInContext(`${code}\nthis.__renderControlPanel = renderControlPanel;`, context);

function snapshot(overrides = {}) {
  return {
    game_done: false,
    active_human_player: 0,
    model_catalog: [
      { id: "right", label: "right.eqx" },
      { id: "down", label: "down.eqx" },
      { id: "pass", label: "pass.eqx" },
    ],
    players: [
      { index: 0, name: "Human", control: "human", model_id: "right" },
      { index: 1, name: "PPO Model", control: "model", model_id: "pass" },
    ],
    ...overrides,
  };
}

context.__renderControlPanel(snapshot());
const list = elements.get("player-control-list");
const firstRow = list.children[0];
const firstControlSelect = firstRow.children[1];
const secondModelSelect = list.children[1].children[2];

fakeDocument.activeElement = secondModelSelect;
secondModelSelect.value = "down";

context.__renderControlPanel(snapshot({ step_count: 1 }));

assert.strictEqual(list.children[0], firstRow);
assert.strictEqual(list.children[0].children[1], firstControlSelect);
assert.strictEqual(list.children[1].children[2], secondModelSelect);
assert.strictEqual(secondModelSelect.value, "down");
"""

    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
