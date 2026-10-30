// Episode views, one per environment, drawing on a canvas in the page's theme
// (palette.js).
//
// A training run logs an episode as seaglass Arrays of type "turbozero/<env>-episode"
// (core/monitoring/episodes.py); episode.js reads its arrays (episodes.js ndArrays)
// and calls RENDERERS[env](arrays) for a view of them:
//
//   {
//     width, height,       // its size, in the units draw() draws in
//     frames, fps,         // how many frames, and how many a second at 1x
//     draw(ctx, frame),    // draw a frame
//     counter(frame),      // optional: the frame, as the player's counter says it
//     speeds, loop,        // optional: the speeds offered, and false to stop at the end
//     players,             // optional: for a two-player game, [{name, color, ring,
//                          // symbol}] by colour, the first to move in its usual
//                          // opening first; a side's mark is its symbol if it has
//                          // one, else a disc of its colour (a ring with `ring`)
//     pieces(frame, colour),  // optional: how many pieces of a colour are on the board
//   }
//
// The board games' views also draw any position, for the search explorer (search.js):
//
//   {
//     drawBoard(ctx, cell, move),   // the board cell(r, c) gives, move [r, c] marked
//     actionCell(action, cell),     // the cell an action fills on that board, or null
//     actionAt(x, y),               // the action played at a point on the board, or null
//     moveName(action),             // an action as players write it
//     at(r, c), cellSize,           // a cell's centre, and its size
//   }
//
// Arrays are NdArrays (episodes.js). A view should do any per-episode work when it's
// made, not every frame; it reads the theme's colours when it draws.
//
// The board games' arrays are core/monitoring/episodes.py's pgx_two_player_episode:
// board (frames, rows, cols) of 0 for empty, 1 and 2 for the two colours; to_move
// (frames,), the colour to move; agent, the trained agent's colour; values (frames, 2),
// the agent's and its opponent's value estimates; completed and outcome, the agent's
// result once the game is over.

import { BLACK, C4_BLUE, GOLD, OTHELLO_GREEN, RED, WHITE, sunken, theme } from "./palette.js";

export const RENDERERS = {
  othello,
  connect_four: connectFour,
  hex,
  tic_tac_toe: ticTacToe,
};

function disc(ctx, x, y, r, fill, stroke = null, width = 1) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}

function line(ctx, [x0, y0], [x1, y1], color, width) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

// ---------------------------------------------------------------- board games

// a move every 0.9s at 1x
const MOVE_FPS = 1 / 0.9;
const MOVE_SPEEDS = [0.5, 1, 2, 4];

// What's common to the board games: the board at each frame, the cell each move
// filled, lines of `connect` in a row, and the view's counter and speeds. A game's
// view draws a position with drawBoard(ctx, cell, move), and a frame as the board
// at that frame with the move into it.
function boardGame({ board }, { connect = null, passes = false } = {}) {
  const [frames, rows, cols] = board.shape;
  const cell = (t, r, c) => board.get(t, r, c);
  // the cell the move into frame t filled, [r, c], or null (the start, a pass);
  // othello's moves turn other discs over, but only fill the one
  const moves = [null];
  for (let t = 1; t < frames; t++) {
    let move = null;
    for (let r = 0; r < rows && !move; r++) {
      for (let c = 0; c < cols; c++) {
        if (!cell(t - 1, r, c) && cell(t, r, c)) {
          move = [r, c];
          break;
        }
      }
    }
    moves.push(move);
  }
  return {
    frames,
    rows,
    cols,
    cell,
    moves,
    // the cells of `connect` in a row through a move, if it made one
    win: (at, move) => connect && winningLine(at, move, connect, rows, cols),
    // a frame's board, for drawBoard
    board: (t) => (r, c) => cell(t, r, c),
    fps: MOVE_FPS,
    speeds: MOVE_SPEEDS,
    loop: false,
    counter(t) {
      // the start isn't a pass
      return `move ${t}/${frames - 1}${passes && t > 0 && !moves[t] ? " pass" : ""}`;
    },
  };
}

// a board's cells as players name them: files a-h left to right, ranks 1-8 down
const cellName = (r, c) => `${String.fromCharCode(97 + c)}${r + 1}`;

function winningLine(cell, move, n, rows, cols) {
  if (!move) return null;
  const [r0, c0] = move;
  const color = cell(r0, c0);
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    // from the far end of the run through the move
    let r = r0, c = c0;
    while (r - dr >= 0 && r - dr < rows && c - dc >= 0 && c - dc < cols && cell(r - dr, c - dc) === color) {
      r -= dr;
      c -= dc;
    }
    const run = [];
    while (r >= 0 && r < rows && c >= 0 && c < cols && cell(r, c) === color) {
      run.push([r, c]);
      r += dr;
      c += dc;
    }
    if (run.length >= n) return run;
  }
  return null;
}

// coordinates along the edges, as players write moves: files a-h, ranks 1-8
function labels(ctx, game, at, size, pad) {
  ctx.fillStyle = theme("dim");
  ctx.font = `10px ${theme("mono") || "monospace"}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let c = 0; c < game.cols; c++) ctx.fillText(String.fromCharCode(97 + c), at(0, c)[0], pad / 2);
  for (let r = 0; r < game.rows; r++) ctx.fillText(String(r + 1), pad / 2, at(r, 0)[1]);
}

// ---------------------------------------------------------------- othello

const OTHELLO_CELL = 40, OTHELLO_PAD = 18;

function othello(arrays) {
  const game = boardGame(arrays, { passes: true });
  const size = OTHELLO_CELL * 8;
  const W = OTHELLO_PAD + size + 6, H = OTHELLO_PAD + size + 6;
  const at = (r, c) => [OTHELLO_PAD + (c + 0.5) * OTHELLO_CELL, OTHELLO_PAD + (r + 0.5) * OTHELLO_CELL];
  const fills = [BLACK, WHITE];
  return {
    ...game,
    width: W,
    height: H,
    at,
    cellSize: OTHELLO_CELL,
    players: [
      { name: "Black", color: BLACK },
      { name: "White", color: WHITE },
    ],
    // the score
    pieces(t, color) {
      let n = 0;
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) n += game.cell(t, r, c) === color + 1;
      return n;
    },
    // actions are the 64 cells, row by row, then a pass
    actionCell: (a) => (a < 64 ? [Math.floor(a / 8), a % 8] : null),
    actionAt(x, y) {
      const c = Math.floor((x - OTHELLO_PAD) / OTHELLO_CELL), r = Math.floor((y - OTHELLO_PAD) / OTHELLO_CELL);
      return r >= 0 && r < 8 && c >= 0 && c < 8 ? r * 8 + c : null;
    },
    // two characters, like the squares: "ps" for a pass
    moveName: (a) => (a < 64 ? cellName(Math.floor(a / 8), a % 8) : "ps"),
    draw(ctx, t) {
      this.drawBoard(ctx, game.board(t), game.moves[t]);
    },
    drawBoard(ctx, cell, move) {
      ctx.fillStyle = sunken();
      ctx.fillRect(0, 0, W, H);
      labels(ctx, game, at, size, OTHELLO_PAD);
      ctx.fillStyle = OTHELLO_GREEN;
      ctx.fillRect(OTHELLO_PAD, OTHELLO_PAD, size, size);
      for (let i = 0; i <= 8; i++) {
        const x = OTHELLO_PAD + i * OTHELLO_CELL;
        line(ctx, [x, OTHELLO_PAD], [x, OTHELLO_PAD + size], BLACK, 1);
        line(ctx, [OTHELLO_PAD, x], [OTHELLO_PAD + size, x], BLACK, 1);
      }
      // the four dots that mark out the centre
      for (const [r, c] of [[2, 2], [2, 6], [6, 2], [6, 6]]) {
        disc(ctx, OTHELLO_PAD + c * OTHELLO_CELL, OTHELLO_PAD + r * OTHELLO_CELL, 3, BLACK);
      }
      const r = OTHELLO_CELL * 0.4;
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const v = cell(row, col);
          if (v) disc(ctx, ...at(row, col), r, fills[v - 1], BLACK, 1);
        }
      }
      if (move) disc(ctx, ...at(...move), 4, RED);
    },
  };
}

// ---------------------------------------------------------------- connect four

const C4_CELL = 46, C4_PAD = 8;

function connectFour(arrays) {
  const game = boardGame(arrays, { connect: 4 });
  const W = C4_PAD * 2 + 7 * C4_CELL, H = C4_PAD * 2 + 6 * C4_CELL;
  const at = (r, c) => [C4_PAD + (c + 0.5) * C4_CELL, C4_PAD + (r + 0.5) * C4_CELL];
  const fills = [RED, GOLD];
  const R = C4_CELL * 0.38;
  return {
    ...game,
    width: W,
    height: H,
    at,
    cellSize: C4_CELL,
    players: [
      { name: "Red", color: RED },
      { name: "Yellow", color: GOLD },
    ],
    // actions are the columns; a piece drops to the lowest empty cell
    actionCell(a, cell) {
      for (let r = 5; r >= 0; r--) if (!cell(r, a)) return [r, a];
      return null;
    },
    actionAt(x) {
      const c = Math.floor((x - C4_PAD) / C4_CELL);
      return c >= 0 && c < 7 ? c : null;
    },
    moveName: (a) => String(a + 1),
    draw(ctx, t) {
      this.drawBoard(ctx, game.board(t), game.moves[t]);
    },
    drawBoard(ctx, cell, move) {
      ctx.fillStyle = C4_BLUE;
      ctx.beginPath();
      ctx.roundRect(0, 0, W, H, 10);
      ctx.fill();
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 7; c++) {
          const v = cell(r, c);
          disc(ctx, ...at(r, c), R, v ? fills[v - 1] : WHITE);
        }
      }
      if (move) disc(ctx, ...at(...move), 5, WHITE);
      const win = game.win(cell, move);
      if (win) for (const rc of win) disc(ctx, ...at(...rc), R, null, BLACK, 4);
    },
  };
}

// ---------------------------------------------------------------- tic-tac-toe

const TTT_CELL = 80, TTT_PAD = 12;

function ticTacToe(arrays) {
  const game = boardGame(arrays, { connect: 3 });
  const size = 3 * TTT_CELL;
  const W = TTT_PAD * 2 + size, H = TTT_PAD * 2 + size;
  const at = (r, c) => [TTT_PAD + (c + 0.5) * TTT_CELL, TTT_PAD + (r + 0.5) * TTT_CELL];
  const mark = TTT_CELL * 0.28;
  return {
    ...game,
    width: W,
    height: H,
    at,
    cellSize: TTT_CELL,
    players: [
      { name: "X", color: "var(--sg-s1)", symbol: "X" },
      { name: "O", color: "var(--sg-s2)", ring: true, symbol: "O" },
    ],
    // actions are the cells, row by row
    actionCell: (a) => [Math.floor(a / 3), a % 3],
    actionAt(x, y) {
      const c = Math.floor((x - TTT_PAD) / TTT_CELL), r = Math.floor((y - TTT_PAD) / TTT_CELL);
      return r >= 0 && r < 3 && c >= 0 && c < 3 ? r * 3 + c : null;
    },
    moveName: (a) => cellName(Math.floor(a / 3), a % 3),
    draw(ctx, t) {
      this.drawBoard(ctx, game.board(t), game.moves[t]);
    },
    drawBoard(ctx, cell, move) {
      ctx.fillStyle = theme("well");
      ctx.fillRect(0, 0, W, H);
      if (move) {
        ctx.fillStyle = sunken();
        ctx.fillRect(TTT_PAD + move[1] * TTT_CELL, TTT_PAD + move[0] * TTT_CELL, TTT_CELL, TTT_CELL);
      }
      ctx.lineCap = "round";
      for (let i = 1; i < 3; i++) {
        const x = TTT_PAD + i * TTT_CELL;
        line(ctx, [x, TTT_PAD + 4], [x, TTT_PAD + size - 4], theme("edge"), 3);
        line(ctx, [TTT_PAD + 4, x], [TTT_PAD + size - 4, x], theme("edge"), 3);
      }
      const [xColor, oColor] = [theme("s1"), theme("s2")];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const v = cell(r, c);
          const [x, y] = at(r, c);
          if (v === 1) {
            line(ctx, [x - mark, y - mark], [x + mark, y + mark], xColor, 7);
            line(ctx, [x - mark, y + mark], [x + mark, y - mark], xColor, 7);
          } else if (v === 2) {
            ctx.beginPath();
            ctx.arc(x, y, mark, 0, 2 * Math.PI);
            ctx.strokeStyle = oColor;
            ctx.lineWidth = 7;
            ctx.stroke();
          }
        }
      }
      const win = game.win(cell, move);
      if (win) {
        const [a, b] = [at(...win[0]), at(...win.at(-1))];
        // a little past the end cells' marks
        const dx = b[0] - a[0], dy = b[1] - a[1], k = (mark * 1.3) / Math.hypot(dx, dy);
        ctx.globalAlpha = 0.85;
        line(ctx, [a[0] - dx * k, a[1] - dy * k], [b[0] + dx * k, b[1] + dy * k], theme("fg"), 5);
        ctx.globalAlpha = 1;
      }
    },
  };
}

// ---------------------------------------------------------------- hex

const HEX_CELL = 26;
const HEX_NEIGHBOURS = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]];

// The board is a rhombus, each row half a cell to the right of the one above. Red (colour 0)
// connects the top and bottom, Blue the left and right; each side's edge is drawn in its colour.
// Actions are the cells, row by row, then the swap: the first stone taken, and the other player's
// placed at its reflection, (r, c) -> (c, r), which boardGame's moves see as a filled cell unless
// the stone is on the long diagonal.
function hex(arrays) {
  const game = boardGame(arrays);
  const n = game.rows, swap = n * n;
  const fills = [RED, C4_BLUE];
  const dy = (HEX_CELL * Math.sqrt(3)) / 2;
  const padX = HEX_CELL * 1.2, padY = HEX_CELL * 1.1;
  const W = padX * 2 + HEX_CELL * (n - 1) * 1.5 + 8, H = padY * 2 + dy * (n - 1);
  const at = (r, c) => [padX + HEX_CELL * (c + r / 2), padY + dy * r];
  const hexagon = (ctx, x, y, s) => {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 6 + (i * Math.PI) / 3;
      ctx[i ? "lineTo" : "moveTo"](x + s * Math.cos(a), y + s * Math.sin(a));
    }
    ctx.closePath();
  };
  const radius = HEX_CELL / Math.sqrt(3);
  // the stones connecting a side's two edges, if they do
  function chain(cell) {
    for (const colour of [1, 2]) {
      const seen = new Map();
      const starts = [];
      for (let i = 0; i < n; i++) {
        const [r, c] = colour === 1 ? [0, i] : [i, 0];
        if (cell(r, c) === colour) starts.push([r, c, null]);
      }
      const reached = [];
      while (starts.length) {
        const [r, c, from] = starts.pop();
        const key = r * n + c;
        if (seen.has(key)) continue;
        seen.set(key, from);
        if ((colour === 1 ? r : c) === n - 1) reached.push(key);
        for (const [dr, dc] of HEX_NEIGHBOURS) {
          const i = r + dr, j = c + dc;
          if (i >= 0 && i < n && j >= 0 && j < n && cell(i, j) === colour) starts.push([i, j, key]);
        }
      }
      if (reached.length) {
        const path = [];
        for (let key = reached[0]; key !== null; key = seen.get(key)) path.push([Math.floor(key / n), key % n]);
        return path;
      }
    }
    return null;
  }
  return {
    ...game,
    width: W,
    height: H,
    at,
    cellSize: HEX_CELL,
    players: [
      { name: "Red", color: RED },
      { name: "Blue", color: C4_BLUE },
    ],
    pieces(t, color) {
      let count = 0;
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) count += game.cell(t, r, c) === color + 1;
      return count;
    },
    actionCell: (a) => (a < swap ? [Math.floor(a / n), a % n] : null),
    actionAt(x, y) {
      const r = Math.round((y - padY) / dy);
      const c = Math.round((x - padX) / HEX_CELL - r / 2);
      return r >= 0 && r < n && c >= 0 && c < n ? r * n + c : null;
    },
    moveName: (a) => (a < swap ? cellName(Math.floor(a / n), a % n) : "swap"),
    // the swap is the only move that adds no stone
    counter(t) {
      const swapped = t > 0 && this.pieces(t, 0) + this.pieces(t, 1) === this.pieces(t - 1, 0) + this.pieces(t - 1, 1);
      return `move ${t}/${game.frames - 1}${swapped ? " swap" : ""}`;
    },
    draw(ctx, t) {
      this.drawBoard(ctx, game.board(t), game.moves[t]);
    },
    drawBoard(ctx, cell, move) {
      ctx.fillStyle = sunken();
      ctx.fillRect(0, 0, W, H);
      // each side's edges, under the cells along them
      const edge = (from, to, colour) => {
        const [x0, y0] = at(...from), [x1, y1] = at(...to);
        line(ctx, [x0, y0], [x1, y1], colour, radius * 2.6);
      };
      ctx.lineCap = "round";
      edge([0, 0], [0, n - 1], RED);
      edge([n - 1, 0], [n - 1, n - 1], RED);
      edge([0, 0], [n - 1, 0], C4_BLUE);
      edge([0, n - 1], [n - 1, n - 1], C4_BLUE);
      ctx.lineCap = "butt";
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const [x, y] = at(r, c);
          hexagon(ctx, x, y, radius);
          ctx.fillStyle = theme("well") || WHITE;
          ctx.fill();
          ctx.strokeStyle = theme("edge") || BLACK;
          ctx.lineWidth = 1;
          ctx.stroke();
          const v = cell(r, c);
          if (v) disc(ctx, x, y, HEX_CELL * 0.38, fills[v - 1]);
        }
      }
      // coordinates along the top and left, as players write moves
      ctx.fillStyle = theme("dim");
      ctx.font = `9px ${theme("mono") || "monospace"}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < n; i++) {
        const [tx, ty] = at(0, i);
        ctx.fillText(String.fromCharCode(97 + i), tx, ty - radius - 8);
        const [lx, ly] = at(i, 0);
        ctx.fillText(String(i + 1), lx - radius - 9, ly);
      }
      const won = chain(cell);
      if (won) for (const rc of won) disc(ctx, ...at(...rc), HEX_CELL * 0.14, WHITE);
      if (move) disc(ctx, ...at(...move), 4, won ? BLACK : WHITE);
    },
  };
}
