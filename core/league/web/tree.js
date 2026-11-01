// A search's tree, browsed move by move: the moves it explored from a position, best first, each
// with its visits, value, prior and the line it expects after that move.

import { el } from "./dom.js";

// a tree from the worker, indexed: each node's children, best first
export function indexTree(tree) {
  const children = Array.from({ length: tree.n.length }, () => []);
  for (let g = 1; g < tree.n.length; g++) children[tree.parent[g]].push(g);
  for (const list of children) list.sort((a, b) => tree.n[b] - tree.n[a] || tree.q[a] - tree.q[b]);
  return { tree, children };
}

// the node the moves from the root reach, -1 where the tree doesn't
export function nodeAt({ tree, children }, path) {
  let g = 0;
  for (const a of path) {
    g = children[g].find((c) => tree.action[c] === a) ?? -1;
    if (g < 0) break;
  }
  return g;
}

export const positionAt = (board, root, path) => path.reduce((position, a) => board.play(position, a), root);

// a node's q, its value for the player to move `depth` moves from the root, as the first side's
export const forFirst = (tree, q, depth) => ((tree.position.turn === 1) !== (depth % 2 === 1) ? q : -q);

// a value from -1 to 1 for the player choosing: rust (bad) to plum (good), the square root telling
// apart the values near 0 where most are
const NEUTRAL = [0xf3, 0xe6, 0xc8], GOOD = [0x6f, 0x4a, 0x92], BAD = [0xb0, 0x57, 0x1f];
export function valueColor(v) {
  const t = Math.sqrt(Math.min(1, Math.abs(v)));
  const to = v >= 0 ? GOOD : BAD;
  const [r, g, b] = NEUTRAL.map((x, i) => Math.round(x * (1 - t) + to[i] * t));
  return `rgb(${r}, ${g}, ${b})`;
}

const pct = (x) => `${Math.round(100 * x)}%`;
const key = (path) => path?.join(",") ?? "";

function bestLine({ children }, g) {
  const line = [g];
  while (children[line[line.length - 1]].length) line.push(children[line[line.length - 1]][0]);
  return line;
}

// + favours the first side and − the other, then a disc of the side it favours
function advantage(board, v) {
  const size = Math.abs(v).toFixed(3);
  if (!Number(size)) return el("span", { class: "adv" }, size);
  return el("span", { class: "adv" }, (v > 0 ? "+" : "−") + size,
    el("i", { class: "swatch", style: `background:${board.sideColors[v > 0 ? 0 : 1]}` }));
}

export function drawTree(box, { board, ix, path, preview, played, go, setPreview }) {
  const scrolled = box.querySelector(".tree-lines")?.scrollTop ?? 0;
  const { tree, children } = ix;
  const at = path ?? [];
  const g = nodeAt(ix, at);
  const position = positionAt(board, tree.position, at);
  const depth = at.length;

  const crumb = (label, on, to) => el("button", { class: on ? "crumb on" : "crumb", on: { click: () => go(to) } }, label);
  const crumbs = el("div", { class: "tree-crumbs" }, crumb("game", !path, null), el("span", { class: "sep" }, "·"), crumb("root", path?.length === 0, []),
    ...at.flatMap((a, i) => [el("span", { class: "sep" }, "›"), crumb(board.moveName(a), i === at.length - 1, at.slice(0, i + 1))]));
  // a position the search never reached: only the way back
  if (g < 0) { box.replaceChildren(crumbs); return; }

  const kids = children[g];
  const total = kids.reduce((k, c) => k + tree.n[c], 0);
  const rows = kids.map((c) => {
    const line = bestLine(ix, c);
    const share = total ? tree.n[c] / total : 0;
    const to = (i) => [...at, ...line.slice(0, i + 1).map((m) => tree.action[m])];
    const isPlayed = !path && tree.action[c] === played;
    return el("div", { class: isPlayed ? "tree-line played" : "tree-line", on: { pointerleave: () => setPreview(null) } },
      el("div", { class: "stats", on: { pointerenter: () => setPreview(to(0)), click: () => go(to(0)) } },
        el("i", { class: "bar", style: `width:${pct(share)}` }),
        el("b", { class: "mv" }, board.moveName(tree.action[c])),
        el("span", { class: "num" }, `${tree.n[c]} `, el("span", { class: "dim" }, `(${pct(share)})`)),
        el("span", { class: "num" }, advantage(board, forFirst(tree, tree.q[c], depth + 1))),
        el("span", { class: "num dim" }, tree.prior[c].toFixed(2))),
      el("div", { class: "pv" }, ...line.map((m, i) => el("button", {
        class: `tok${i === 0 ? " first" : ""}${key(preview) === key(to(i)) ? " hot" : ""}`,
        on: { pointerenter: () => setPreview(to(i)), click: () => go(to(i)) },
      }, board.moveName(tree.action[m]))),
      tree.terminal[line[line.length - 1]] === 1 ? el("span", { class: "end" }, "end") : null));
  });
  // the legal moves it never explored, with their priors where it has them, the root's
  const explored = new Set(kids.map((c) => tree.action[c]));
  const rest = board.legal(position).filter((a) => !explored.has(a))
    .map((a) => ({ a, prior: depth === 0 && tree.policy ? tree.policy[a] : null }))
    .sort((x, y) => (y.prior ?? 0) - (x.prior ?? 0));
  const lines = el("div", { class: "tree-lines" },
    el("div", { class: "tree-cols" }, el("span", {}), el("span", {}, "visits"), el("span", {}, "value"), el("span", {}, "prior")),
    ...rows,
    ...rest.map(({ a, prior }) => el("div", { class: !path && a === played ? "tree-line unexplored played" : "tree-line unexplored" },
      el("div", { class: "stats" }, el("b", { class: "mv" }, board.moveName(a)), el("span", { class: "num" }, "0"), el("span", {}),
        el("span", { class: "num" }, prior === null ? "" : prior.toFixed(2))))));
  box.replaceChildren(crumbs,
    el("div", { class: "tree-summary" }, tree.terminal[g] === 1 ? el("span", {}, "game over") : null,
      el("span", { class: "dim" }, `${tree.n[g]} visits`), el("b", {}, advantage(board, forFirst(tree, tree.q[g], depth)))),
    lines);
  lines.scrollTop = scrolled;
}
