// The league dashboard's Othello board (see core/league/dashboard.html): a position is 64
// squares a1, b1, ..., h8, 0 empty, 1 black, 2 white, and whose turn it is. Actions are as
// game.py's: the squares row by row, 64 a pass.

const DIRECTIONS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
const PASS = 64;
const NS = "http://www.w3.org/2000/svg";

function start() {
  const squares = new Array(64).fill(0);
  squares[27] = 2; // d4
  squares[28] = 1; // e4
  squares[35] = 1; // d5
  squares[36] = 2; // e5
  return { squares, turn: 1, passed: false, terminated: false, reward: 0 };
}

function flips(squares, me, move) {
  const them = 3 - me;
  const flipped = [];
  const [row, col] = [Math.floor(move / 8), move % 8];
  for (const [dr, dc] of DIRECTIONS) {
    const line = [];
    let [r, c] = [row + dr, col + dc];
    while (r >= 0 && r < 8 && c >= 0 && c < 8 && squares[r * 8 + c] === them) {
      line.push(r * 8 + c);
      r += dr;
      c += dc;
    }
    if (r >= 0 && r < 8 && c >= 0 && c < 8 && squares[r * 8 + c] === me) flipped.push(...line);
  }
  return flipped;
}

// As pgx plays it: the game ends when the board is full, when the player who just moved has taken
// every disc, or on the second of two passes in a row. `reward` is the result for the player to
// move in a position that's over.
function play(position, move) {
  const squares = position.squares.slice();
  const me = position.turn;
  const them = 3 - me;
  if (move !== PASS) {
    squares[move] = me;
    for (const square of flips(position.squares, me, move)) squares[square] = me;
  }
  const [mine, theirs] = [squares.filter((s) => s === me).length, squares.filter((s) => s === them).length];
  const terminated = mine + theirs === 64 || theirs === 0 || (position.passed && move === PASS);
  const reward = !terminated || mine === theirs ? 0 : mine > theirs ? -1 : 1;
  return { squares, turn: them, passed: move === PASS, terminated, reward };
}

// the moves the player to move has, a pass if nothing else; none once the game is over
function legal(position) {
  if (position.terminated) return [];
  const moves = [];
  for (let square = 0; square < 64; square++) {
    if (position.squares[square] === 0 && flips(position.squares, position.turn, square).length) moves.push(square);
  }
  return moves.length ? moves : [PASS];
}

// the network's input, channels first (see export_onnx.py): the discs of the player to move, then
// the other player's
function observe(position, out, offset = 0) {
  for (let square = 0; square < 64; square++) {
    const disc = position.squares[square];
    out[offset + square] = disc === position.turn ? 1 : 0;
    out[offset + 64 + square] = disc === 3 - position.turn ? 1 : 0;
  }
}

function moveName(move) {
  return move === PASS ? "pass" : "abcdefgh"[move % 8] + (Math.floor(move / 8) + 1);
}

function counts(position) {
  return [1, 2].map((side) => position.squares.filter((s) => s === side).length);
}

function node(tag, attributes) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

const SIZE = 40;
const MARGIN = 16;
const BOARD = "#d6bd8f";
const LINE = "#a7824c";
const INK = "#2a1802";
const CREAM = "#fbf4e4";

// the parts of a board that change, kept between draws so that discs can fade between colours
function build(container) {
  const root = node("svg", { viewBox: `0 0 ${MARGIN + 8 * SIZE} ${MARGIN + 8 * SIZE}`, class: "othello-board" });
  root.append(node("rect", { x: MARGIN, y: MARGIN, width: 8 * SIZE, height: 8 * SIZE, fill: BOARD, rx: 2 }));
  for (let i = 0; i <= 8; i++) {
    const at = MARGIN + i * SIZE;
    root.append(node("line", { x1: at, x2: at, y1: MARGIN, y2: MARGIN + 8 * SIZE, stroke: LINE }));
    root.append(node("line", { y1: at, y2: at, x1: MARGIN, x2: MARGIN + 8 * SIZE, stroke: LINE }));
  }
  for (let i = 0; i < 8; i++) {
    const column = node("text", { x: MARGIN + (i + 0.5) * SIZE, y: 11, "text-anchor": "middle", "font-size": 10, fill: "var(--ink-3)" });
    column.textContent = "abcdefgh"[i];
    const row = node("text", { x: 7, y: MARGIN + (i + 0.5) * SIZE + 3, "text-anchor": "middle", "font-size": 10, fill: "var(--ink-3)" });
    row.textContent = String(i + 1);
    root.append(column, row);
  }
  const cells = [];
  for (let i = 0; i < 64; i++) {
    const [cx, cy] = [MARGIN + (i % 8 + 0.5) * SIZE, MARGIN + (Math.floor(i / 8) + 0.5) * SIZE];
    const mark = node("circle", { cx, cy, r: 0, visibility: "hidden" });
    const ring = node("circle", { cx, cy, r: SIZE * 0.44, fill: "none", stroke: "#6f4a92", "stroke-width": 2, visibility: "hidden" });
    const disc = node("circle", { cx, cy, r: SIZE * 0.4, visibility: "hidden", style: "transition: fill .2s ease; filter: drop-shadow(0 1px 1px rgba(42,24,2,.4))" });
    const last = node("circle", { cx, cy, r: SIZE * 0.08, fill: "#c9562c", visibility: "hidden" });
    const label = node("text", { x: cx, y: cy + 3.5, "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: INK, visibility: "hidden", style: "pointer-events:none;font-variant-numeric:tabular-nums" });
    const hit = node("rect", { x: cx - SIZE / 2, y: cy - SIZE / 2, width: SIZE, height: SIZE, fill: "transparent" });
    root.append(mark, ring, disc, last, label, hit);
    cells.push({ mark, ring, disc, last, label, hit });
  }
  container.replaceChildren(root);
  container.boardCells = cells;
  return cells;
}

const show = (element, on) => element.setAttribute("visibility", on ? "visible" : "hidden");

// `marks` by move hold what to show on empty squares: `size` from 0 to 1, its `color` and `label`;
// `played` is the move to ring; `on` holds `enter`, `leave` and `click`, each given a move
function draw(container, position, lastMove, { marks = new Map(), played = null, on = {} } = {}) {
  const cells = container.boardCells ?? build(container);
  cells.forEach((cell, i) => {
    const square = position.squares[i];
    show(cell.disc, square);
    if (square) cell.disc.setAttribute("fill", square === 1 ? INK : CREAM);
    show(cell.last, i === lastMove);
    show(cell.ring, i === played);
    const mark = square ? undefined : marks.get(i);
    show(cell.mark, mark);
    show(cell.label, mark?.label);
    if (mark) {
      cell.mark.setAttribute("r", SIZE * (0.12 + 0.34 * Math.sqrt(mark.size)));
      cell.mark.setAttribute("fill", mark.color);
      cell.label.textContent = mark.label ?? "";
      cell.label.setAttribute("fill", mark.ink ?? INK);
    }
    cell.hit.onpointerenter = on.enter && (() => on.enter(i));
    cell.hit.onpointerleave = on.leave && (() => on.leave(i));
    cell.hit.onclick = on.click && (() => on.click(i));
    cell.hit.style.cursor = mark && on.click ? "pointer" : "default";
  });
}

export default {
  start,
  play,
  legal,
  observe,
  draw,
  moveName,
  counts,
  numActions: 65,
  inputShape: [2, 8, 8],
  sideNames: ["Black", "White"],
  sideColors: [INK, CREAM],
};
