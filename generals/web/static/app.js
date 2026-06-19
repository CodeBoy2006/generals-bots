"use strict";

const ui = {
  canvas: document.getElementById("game-board"),
  connectionDot: document.getElementById("connection-dot"),
  modeLabel: document.getElementById("mode-label"),
  reconnectButton: document.getElementById("reconnect-button"),
  restartButton: document.getElementById("restart-button"),
  passButton: document.getElementById("pass-button"),
  cancelButton: document.getElementById("cancel-button"),
  splitToggle: document.getElementById("split-toggle"),
  autoTickToggle: document.getElementById("auto-tick-toggle"),
  tickRateInput: document.getElementById("tick-rate-input"),
  statusMessage: document.getElementById("status-message"),
  turnValue: document.getElementById("turn-value"),
  stepValue: document.getElementById("step-value"),
  winnerValue: document.getElementById("winner-value"),
  playersList: document.getElementById("players-list"),
  previewValue: document.getElementById("preview-value"),
  previewList: document.getElementById("preview-list"),
};

const ctx = ui.canvas.getContext("2d");
const state = {
  socket: null,
  snapshot: null,
  status: "connecting",
  hoverCell: null,
  reconnectTimer: null,
  boardRect: { x: 0, y: 0, size: 0, cell: 0 },
  assets: {
    general: loadImage("/assets/images/crownie.png"),
    city: loadImage("/assets/images/citie.png"),
    mountain: loadImage("/assets/images/mountainie.png"),
  },
};

function loadImage(src) {
  const image = new Image();
  image.src = src;
  image.addEventListener("load", render);
  return image;
}

function connectionUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/game`;
}

function setConnectionStatus(status) {
  state.status = status;
  ui.connectionDot.classList.toggle("is-connected", status === "connected");
  ui.connectionDot.classList.toggle("is-connecting", status === "connecting");
  ui.connectionDot.classList.toggle("is-offline", status === "offline");
}

function connect() {
  if (state.socket) {
    state.socket.onclose = null;
    state.socket.close();
  }
  window.clearTimeout(state.reconnectTimer);
  setConnectionStatus("connecting");
  ui.statusMessage.textContent = "Connecting";

  const socket = new WebSocket(connectionUrl());
  state.socket = socket;

  socket.addEventListener("open", () => {
    setConnectionStatus("connected");
    ui.statusMessage.textContent = "Connected";
  });

  socket.addEventListener("message", (event) => {
    try {
      const snapshot = JSON.parse(event.data);
      if (snapshot.type === "snapshot") {
        state.snapshot = snapshot;
        syncControls(snapshot);
        renderHud(snapshot);
        render();
      }
    } catch (error) {
      ui.statusMessage.textContent = `Invalid server payload: ${error.message}`;
    }
  });

  socket.addEventListener("close", () => {
    setConnectionStatus("offline");
    ui.statusMessage.textContent = "Disconnected";
    state.reconnectTimer = window.setTimeout(connect, 2000);
  });

  socket.addEventListener("error", () => {
    setConnectionStatus("offline");
    ui.statusMessage.textContent = "Connection error";
  });
}

function sendCommand(command) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    ui.statusMessage.textContent = "Socket offline";
    return;
  }
  state.socket.send(JSON.stringify(command));
}

function syncControls(snapshot) {
  ui.splitToggle.checked = Boolean(snapshot.split_enabled);
  ui.autoTickToggle.checked = Boolean(snapshot.auto_tick && snapshot.auto_tick.enabled);
  if (document.activeElement !== ui.tickRateInput) {
    ui.tickRateInput.value = String(snapshot.auto_tick ? snapshot.auto_tick.tick_rate : 2);
  }

  const isWatchMode = snapshot.mode === "machine-vs-machine";
  const disabled = isWatchMode || snapshot.game_done;
  ui.passButton.disabled = disabled;
  ui.cancelButton.disabled = isWatchMode;
  ui.splitToggle.disabled = disabled;
  ui.restartButton.disabled = false;
}

function renderHud(snapshot) {
  ui.modeLabel.textContent = snapshot.mode === "machine-vs-machine" ? "Machine watch" : "Human vs model";
  ui.statusMessage.textContent = snapshot.last_message || "Ready";
  ui.turnValue.textContent = String(snapshot.time);
  ui.stepValue.textContent = String(snapshot.step_count);
  ui.winnerValue.textContent = snapshot.winner === null ? "-" : playerName(snapshot, snapshot.winner);
  renderPlayers(snapshot);
  renderPreview(snapshot);
}

function renderPlayers(snapshot) {
  ui.playersList.replaceChildren();
  const maxArmy = Math.max(...snapshot.players.map((player) => player.army), 1);
  const maxLand = Math.max(...snapshot.players.map((player) => player.land), 1);

  for (const player of snapshot.players) {
    const row = document.createElement("div");
    row.className = "player-row";

    const swatch = document.createElement("span");
    swatch.className = "player-swatch";
    swatch.style.backgroundColor = player.color;

    const body = document.createElement("div");
    const name = document.createElement("div");
    name.className = "player-name";
    name.textContent = player.name;

    const bars = document.createElement("div");
    bars.className = "player-bars";
    bars.appendChild(metricBar(player.color, player.army / maxArmy));
    bars.appendChild(metricBar("#e0b64d", player.land / maxLand));
    body.append(name, bars);

    const stats = document.createElement("div");
    stats.className = "player-stats";
    stats.textContent = `${player.army} army / ${player.land} land`;

    row.append(swatch, body, stats);
    ui.playersList.appendChild(row);
  }
}

function metricBar(color, ratio) {
  const track = document.createElement("div");
  track.className = "bar-track";
  const fill = document.createElement("div");
  fill.className = "bar-fill";
  fill.style.width = `${Math.max(2, Math.min(100, ratio * 100))}%`;
  fill.style.backgroundColor = color;
  track.appendChild(fill);
  return track;
}

function renderPreview(snapshot) {
  ui.previewList.replaceChildren();
  const preview = snapshot.policy_preview;
  if (!preview || !preview.candidates.length) {
    ui.previewValue.textContent = "-";
    return;
  }

  ui.previewValue.textContent = `Value ${formatNumber(preview.value)} / ${preview.policy_mode}`;
  for (const candidate of preview.candidates) {
    const item = document.createElement("li");
    item.className = "preview-item";

    const action = document.createElement("span");
    action.className = "preview-action";
    action.textContent = previewLabel(candidate);

    const probability = document.createElement("span");
    probability.className = "preview-probability";
    probability.textContent = `${Math.round(candidate.probability * 100)}%`;

    item.append(action, probability);
    ui.previewList.appendChild(item);
  }
}

function previewLabel(candidate) {
  if (candidate.is_pass) {
    return "Pass";
  }
  const source = candidate.source ? candidate.source.join(",") : "?";
  const target = candidate.target ? candidate.target.join(",") : candidate.direction_label;
  const split = candidate.is_split ? " split" : "";
  return `${source} -> ${target}${split}`;
}

function playerName(snapshot, index) {
  const player = snapshot.players.find((entry) => entry.index === index);
  return player ? player.name : `P${index}`;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return number.toFixed(2);
}

function resizeCanvas() {
  const rect = ui.canvas.getBoundingClientRect();
  const cssWidth = Math.max(280, Math.floor(rect.width));
  const cssHeight = Math.max(280, Math.floor(rect.height));
  const pixelRatio = window.devicePixelRatio || 1;
  const nextWidth = Math.floor(cssWidth * pixelRatio);
  const nextHeight = Math.floor(cssHeight * pixelRatio);

  if (ui.canvas.width !== nextWidth || ui.canvas.height !== nextHeight) {
    ui.canvas.width = nextWidth;
    ui.canvas.height = nextHeight;
  }
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  render();
}

function render() {
  const width = ui.canvas.clientWidth || 900;
  const height = ui.canvas.clientHeight || 900;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0e1116";
  ctx.fillRect(0, 0, width, height);

  if (!state.snapshot) {
    drawEmptyBoard(width, height);
    return;
  }

  const snapshot = state.snapshot;
  const grid = snapshot.grid;
  const padding = Math.max(10, Math.min(18, width * 0.025));
  const boardSize = Math.min(width, height) - padding * 2;
  const cellSize = boardSize / grid.width;
  state.boardRect = { x: (width - boardSize) / 2, y: (height - boardSize) / 2, size: boardSize, cell: cellSize };

  drawCells(snapshot, state.boardRect);
  drawPolicyOverlay(snapshot, state.boardRect);
  drawBoardBorder(state.boardRect);
}

function drawEmptyBoard(width, height) {
  const size = Math.min(width, height) - 36;
  const x = (width - size) / 2;
  const y = (height - size) / 2;
  const cell = size / 8;
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      ctx.fillStyle = (row + col) % 2 === 0 ? "#171c24" : "#1d232d";
      ctx.fillRect(x + col * cell, y + row * cell, cell, cell);
    }
  }
  ctx.fillStyle = "#aeb7c5";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "18px Quicksand, sans-serif";
  ctx.fillText(state.status === "offline" ? "Disconnected" : "Connecting", width / 2, height / 2);
}

function drawCells(snapshot, board) {
  const grid = snapshot.grid;
  const validTargets = new Set(snapshot.valid_targets.map((cell) => cellKey(cell[0], cell[1])));
  const selected = snapshot.selected_cell ? cellKey(snapshot.selected_cell[0], snapshot.selected_cell[1]) : null;

  for (let row = 0; row < grid.height; row += 1) {
    for (let col = 0; col < grid.width; col += 1) {
      const x = board.x + col * board.cell;
      const y = board.y + row * board.cell;
      const visible = Boolean(grid.visible[row][col]);
      const owner = grid.ownership[row][col];
      const key = cellKey(row, col);
      drawCellBase(snapshot, row, col, x, y, board.cell, visible, owner);
      if (visible) {
        drawCellContents(snapshot, row, col, x, y, board.cell);
      }
      if (key === selected) {
        strokeCell(x, y, board.cell, "#e0b64d", 3);
      } else if (validTargets.has(key)) {
        strokeCell(x, y, board.cell, "#3fb56a", 2);
      }
      if (state.hoverCell && key === cellKey(state.hoverCell.row, state.hoverCell.col)) {
        strokeCell(x, y, board.cell, "rgba(255, 255, 255, 0.75)", 1.5);
      }
    }
  }
}

function drawCellBase(snapshot, row, col, x, y, size, visible, owner) {
  if (!visible) {
    ctx.fillStyle = (row + col) % 2 === 0 ? "#10141b" : "#151a22";
    ctx.fillRect(x, y, size, size);
    ctx.fillStyle = "rgba(255, 255, 255, 0.035)";
    ctx.fillRect(x, y, size, size / 2);
    drawGridLine(x, y, size);
    return;
  }

  const base = owner >= 0 ? colorForPlayer(snapshot, owner) : "#3b4655";
  ctx.fillStyle = owner >= 0 ? withAlpha(base, 0.76) : "#323b47";
  ctx.fillRect(x, y, size, size);
  ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
  ctx.fillRect(x, y, size, size * 0.34);
  drawGridLine(x, y, size);
}

function drawCellContents(snapshot, row, col, x, y, size) {
  const grid = snapshot.grid;
  const army = grid.armies[row][col];
  const isMountain = Boolean(grid.mountains[row][col]);
  const isCity = Boolean(grid.cities[row][col]);
  const isGeneral = Boolean(grid.generals[row][col]);

  if (isMountain) {
    drawImageCentered(state.assets.mountain, x, y, size, 0.78);
    return;
  }

  if (isCity) {
    drawImageCentered(state.assets.city, x, y, size, 0.86);
  }
  if (isGeneral) {
    drawImageCentered(state.assets.general, x, y, size, 0.92);
  }
  if (army > 0 || isCity || isGeneral) {
    drawArmyText(army, x, y, size);
  }
}

function drawArmyText(value, x, y, size) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `${Math.max(12, Math.floor(size * 0.28))}px Quicksand, sans-serif`;
  ctx.lineWidth = Math.max(2, size * 0.045);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
  ctx.fillStyle = "#f8fafc";
  ctx.strokeText(String(value), x + size / 2, y + size / 2);
  ctx.fillText(String(value), x + size / 2, y + size / 2);
  ctx.restore();
}

function drawImageCentered(image, x, y, size, alpha) {
  const ready = image.complete && image.naturalWidth > 0;
  const inset = size * 0.19;
  const drawSize = size - inset * 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  if (ready) {
    ctx.drawImage(image, x + inset, y + inset, drawSize, drawSize);
  } else {
    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
    ctx.beginPath();
    ctx.arc(x + size / 2, y + size / 2, drawSize / 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawPolicyOverlay(snapshot, board) {
  const preview = snapshot.policy_preview;
  if (!preview || !preview.candidates.length) {
    return;
  }

  preview.candidates.forEach((candidate, index) => {
    if (!candidate.source || !candidate.target) {
      return;
    }
    const source = cellCenter(board, candidate.source[0], candidate.source[1]);
    const target = cellCenter(board, candidate.target[0], candidate.target[1]);
    const alpha = Math.max(0.18, 0.68 - index * 0.16);
    drawArrow(source.x, source.y, target.x, target.y, `rgba(224, 182, 77, ${alpha})`);
  });
}

function drawArrow(x1, y1, x2, y2, color) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 10;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawBoardBorder(board) {
  ctx.strokeStyle = "#596474";
  ctx.lineWidth = 2;
  ctx.strokeRect(board.x, board.y, board.size, board.size);
}

function drawGridLine(x, y, size) {
  ctx.strokeStyle = "rgba(12, 15, 20, 0.36)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}

function strokeCell(x, y, size, color, lineWidth) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.strokeRect(x + lineWidth / 2, y + lineWidth / 2, size - lineWidth, size - lineWidth);
  ctx.restore();
}

function colorForPlayer(snapshot, owner) {
  const player = snapshot.players.find((entry) => entry.index === owner);
  return player ? player.color : "#596474";
}

function withAlpha(hex, alpha) {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!match) {
    return hex;
  }
  const red = parseInt(match[1], 16);
  const green = parseInt(match[2], 16);
  const blue = parseInt(match[3], 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function cellCenter(board, row, col) {
  return {
    x: board.x + col * board.cell + board.cell / 2,
    y: board.y + row * board.cell + board.cell / 2,
  };
}

function cellFromEvent(event) {
  if (!state.snapshot) {
    return null;
  }
  const rect = ui.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const board = state.boardRect;
  if (x < board.x || y < board.y || x >= board.x + board.size || y >= board.y + board.size) {
    return null;
  }
  const col = Math.floor((x - board.x) / board.cell);
  const row = Math.floor((y - board.y) / board.cell);
  if (row < 0 || col < 0 || row >= state.snapshot.grid.height || col >= state.snapshot.grid.width) {
    return null;
  }
  return { row, col };
}

function isVisibleCell(row, col) {
  return Boolean(state.snapshot && state.snapshot.grid.visible[row][col]);
}

function isValidTarget(row, col) {
  if (!state.snapshot) {
    return false;
  }
  return state.snapshot.valid_targets.some((cell) => cell[0] === row && cell[1] === col);
}

function cellKey(row, col) {
  return `${row}:${col}`;
}

function sendAutoTick() {
  const rate = Math.max(0.1, Number(ui.tickRateInput.value) || 2);
  ui.tickRateInput.value = String(rate);
  sendCommand({
    type: "set_auto_tick",
    enabled: ui.autoTickToggle.checked,
    tick_rate: rate,
  });
}

ui.canvas.addEventListener("click", (event) => {
  const snapshot = state.snapshot;
  if (!snapshot || snapshot.game_done || snapshot.mode === "machine-vs-machine") {
    return;
  }
  const cell = cellFromEvent(event);
  if (!cell || !isVisibleCell(cell.row, cell.col)) {
    return;
  }
  const selected = snapshot.selected_cell;
  if (selected && selected[0] === cell.row && selected[1] === cell.col) {
    sendCommand({ type: "cancel" });
    return;
  }
  if (selected && isValidTarget(cell.row, cell.col)) {
    sendCommand({
      type: "move",
      source: selected,
      target: [cell.row, cell.col],
      split: ui.splitToggle.checked,
    });
    return;
  }
  sendCommand({ type: "select", row: cell.row, col: cell.col });
});

ui.canvas.addEventListener("mousemove", (event) => {
  state.hoverCell = cellFromEvent(event);
  render();
});

ui.canvas.addEventListener("mouseleave", () => {
  state.hoverCell = null;
  render();
});

ui.reconnectButton.addEventListener("click", connect);
ui.restartButton.addEventListener("click", () => sendCommand({ type: "restart" }));
ui.passButton.addEventListener("click", () => sendCommand({ type: "pass" }));
ui.cancelButton.addEventListener("click", () => sendCommand({ type: "cancel" }));
ui.splitToggle.addEventListener("change", () => sendCommand({ type: "set_split", enabled: ui.splitToggle.checked }));
ui.autoTickToggle.addEventListener("change", sendAutoTick);
ui.tickRateInput.addEventListener("change", sendAutoTick);
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    sendCommand({ type: "cancel" });
  }
});

const observer = new ResizeObserver(resizeCanvas);
observer.observe(ui.canvas);
window.addEventListener("resize", resizeCanvas);
resizeCanvas();
connect();
