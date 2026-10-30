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
  return { squares, turn: 1 };
}

function play(position, move) {
  const squares = position.squares.slice();
  const me = position.turn;
  const them = 3 - me;
  if (move !== PASS) {
    squares[move] = me;
    const [row, col] = [Math.floor(move / 8), move % 8];
    for (const [dr, dc] of DIRECTIONS) {
      const line = [];
      let [r, c] = [row + dr, col + dc];
      while (r >= 0 && r < 8 && c >= 0 && c < 8 && squares[r * 8 + c] === them) {
        line.push(r * 8 + c);
        r += dr;
        c += dc;
      }
      if (r >= 0 && r < 8 && c >= 0 && c < 8 && squares[r * 8 + c] === me) {
        for (const square of line) squares[square] = me;
      }
    }
  }
  return { squares, turn: them };
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

function draw(container, position, lastMove) {
  const size = 40;
  const margin = 16;
  const root = node("svg", { viewBox: `0 0 ${margin + 8 * size} ${margin + 8 * size}` });
  root.append(node("rect", { x: margin, y: margin, width: 8 * size, height: 8 * size, fill: "#1f6b4f" }));
  for (let i = 0; i <= 8; i++) {
    const at = margin + i * size;
    root.append(node("line", { x1: at, x2: at, y1: margin, y2: margin + 8 * size, stroke: "#134433" }));
    root.append(node("line", { y1: at, y2: at, x1: margin, x2: margin + 8 * size, stroke: "#134433" }));
  }
  for (let i = 0; i < 8; i++) {
    const column = node("text", { x: margin + (i + 0.5) * size, y: 11, "text-anchor": "middle", "font-size": 10, fill: "var(--ink-3)" });
    column.textContent = "abcdefgh"[i];
    const row = node("text", { x: 7, y: margin + (i + 0.5) * size + 3, "text-anchor": "middle", "font-size": 10, fill: "var(--ink-3)" });
    row.textContent = String(i + 1);
    root.append(column, row);
  }
  position.squares.forEach((square, i) => {
    const [cx, cy] = [margin + (i % 8 + 0.5) * size, margin + (Math.floor(i / 8) + 0.5) * size];
    if (square) {
      root.append(node("circle", { cx, cy, r: size * 0.42, fill: square === 1 ? "#111" : "#f2f1ec" }));
    }
    if (i === lastMove) {
      root.append(node("circle", { cx, cy, r: size * 0.09, fill: "#e66767" }));
    }
  });
  container.replaceChildren(root);
}

export default {
  start,
  play,
  draw,
  moveName,
  counts,
  sideNames: ["Black", "White"],
  sideColors: ["#111", "#f2f1ec"],
};
