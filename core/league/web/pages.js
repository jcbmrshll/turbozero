// The pages a click opens: a player's, its games, and every network side by side.

import board from "/game.js";
import { ago, api, backButton, byId, el, elo, perMove, runOf, send, showTip, state, svg, tagOf, tile } from "./dom.js";

const SERIES = 8;
let redraw = () => {};
export function onPlayerChange(f) { redraw = f; }

export function record(a, b) {
  for (const [x, y, w, d, l] of state.summary.pairs) {
    if (x === a && y === b) return { w, d, l };
    if (x === b && y === a) return { w: l, d, l: w };
  }
  return { w: 0, d: 0, l: 0 };
}
function wdl(r) {
  return el("span", {}, el("span", { class: "w" }, String(r.w)), " – ", el("span", { class: "d" }, String(r.d)), " – ", el("span", { class: "l" }, String(r.l)));
}
function points(r) { const n = r.w + r.d + r.l; return n ? (r.w + r.d / 2) / n : null; }

function notes(player) {
  const tagBox = el("div", { class: "edit-tags" });
  const allTags = [...new Set(state.summary.players.flatMap((p) => p.tags))].sort();
  async function setTags(tags) {
    await send("/api/notes", { player: player.id, tags });
    await redraw();
  }
  const adding = el("input", { placeholder: "Tag", list: "all-tags", on: { keydown: (e) => {
    if (e.key === "Enter" && e.target.value.trim()) setTags([...player.tags, e.target.value]);
  } } });
  tagBox.append(...player.tags.map((tag) => el("span", { class: "tag own" }, tag,
    el("button", { class: "x", on: { click: () => setTags(player.tags.filter((t) => t !== tag)) } }, "×"))),
    adding, el("datalist", { id: "all-tags" }, ...allTags.map((tag) => el("option", { value: tag }))));
  const description = el("textarea", { rows: 3, placeholder: "Description", on: { change: async (e) => {
    await send("/api/notes", { player: player.id, description: e.target.value });
    await redraw();
  } } });
  description.value = player.description;
  const facts = Object.entries(player.facts).filter(([, v]) => v !== false)
    .map(([label, value]) => el("div", {}, el("span", { class: "dim" }, label), el("b", {}, value === true ? "yes" : String(value))));
  return el("div", { class: "notes" }, description, tagBox, el("div", { class: "facts" }, ...facts));
}

export async function drawPlayer(main, aId, bId) {
  const a = byId(aId);
  if (!a) { location.hash = ""; return; }
  const players = [...state.summary.players].sort((p, q) => q.elo - p.elo).filter((p) => p.id !== aId);
  const selB = el("select", { on: { change: () => { location.hash = `#player/${aId}${selB.value ? `/${selB.value}` : ""}`; } } },
    el("option", { value: "" }, "Every opponent"),
    ...players.map((p) => { const o = el("option", { value: p.id }, p.name); if (p.id === bId) o.selected = true; return o; }));
  const content = [
    el("div", { class: "controls" }, backButton()),
    el("div", { class: "controls", style: "margin-bottom:4px" }, el("h1", {}, a.name), tagOf(a),
      a.can_play ? el("button", { on: { click: async () => {
        await send("/api/paused", { player: aId, paused: !a.retired });
        await redraw();
      } } }, a.retired ? "Resume" : "Pause") : null),
    el("div", { class: "dim", style: "margin-bottom:14px" }, runOf(a), a.net ? ` · ${a.net}` : ""),
    el("div", { class: "tiles" },
      tile(a.games ? `${elo(a.elo)} ±${a.sd.toFixed(0)}` : "–", "rating"),
      tile(a.games.toLocaleString(), "games"),
      tile(perMove(a), "per move")),
    notes(a),
    el("div", { class: "controls" }, el("span", { class: "dim" }, "vs"), selB),
  ];
  if (bId !== undefined && bId !== null) {
    const r = record(aId, bId);
    const fit = await api(`/api/pair?a=${aId}&b=${bId}`);
    const score = points(r);
    content.push(el("div", { class: "tiles" },
      tile(`${elo(fit.difference)} ±${fit.sd.toFixed(0)}`, "rating difference"),
      el("div", { class: "tile" }, el("b", {}, wdl(r)), el("span", {}, "wins, draws, losses")),
      tile(score === null ? "–" : score.toFixed(3), "score"),
      tile(fit.expected.toFixed(3), "expected")));
  } else {
    const opponents = state.summary.players.filter((p) => { const r = record(aId, p.id); return p.id !== aId && r.w + r.d + r.l > 0; })
      .sort((p, q) => q.elo - p.elo);
    content.push(el("div", { class: "card" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Opponent"), el("th", { class: "num" }, "Rating"), el("th", { class: "num" }, "Difference"),
        el("th", { class: "num" }, "Wins – draws – losses"), el("th", { class: "num" }, "Score"), el("th", { class: "num" }, "Expected"))),
      el("tbody", {}, ...opponents.map((p) => {
        const r = record(aId, p.id), score = points(r);
        const expected = 1 / (1 + 10 ** ((p.elo - a.elo) / 400));
        return el("tr", { class: "link", on: { click: () => { location.hash = `#player/${aId}/${p.id}`; } } },
          el("td", {}, p.name, tagOf(p)),
          el("td", { class: "num" }, elo(p.elo)),
          el("td", { class: "num" }, elo(a.elo - p.elo)),
          el("td", { class: "num" }, wdl(r)),
          el("td", { class: "num" }, score === null ? "" : score.toFixed(3)),
          el("td", { class: "num dim" }, expected.toFixed(3)));
      })))));
  }
  const list = el("div", {});
  content.push(el("h2", {}, "Games"), list);
  main.replaceChildren(el("div", { class: "page" }, ...content));
  await drawGames(list, aId, bId);
}

export async function drawGames(box, player, opponent, offset = 0) {
  const limit = 50;
  const query = new URLSearchParams({ limit, offset });
  if (player !== undefined && player !== null) query.set("player", player);
  if (opponent !== undefined && opponent !== null) query.set("opponent", opponent);
  const { games, total } = await api(`/api/games?${query}`);
  const resultText = (g) => (g.result > 0 ? "1 – 0" : g.result < 0 ? "0 – 1" : "½ – ½");
  box.replaceChildren(
    el("div", { class: "card" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { class: "num" }, "#"),
        el("th", {}, board?.sideNames?.[0] ?? "First"), el("th", {}, board?.sideNames?.[1] ?? "Second"),
        el("th", {}, "Result"), el("th", { class: "num" }, "Score"), el("th", { class: "num" }, "Opening"),
        el("th", { class: "num" }, "Moves"), el("th", { class: "num" }, "Ago"), el("th", {}, "Code"))),
      el("tbody", {}, ...games.map((g) => el("tr", { class: "link", on: { click: () => { location.hash = `#game/${g.id}`; } } },
        el("td", { class: "num dim" }, String(g.id)),
        el("td", {}, g.first), el("td", {}, g.second), el("td", {}, resultText(g)),
        el("td", { class: "num dim" }, g.first_score === null ? "" : `${g.first_score} – ${g.second_score}`),
        el("td", { class: "num dim" }, g.opening < 0 ? "" : String(g.opening)),
        el("td", { class: "num dim" }, String(g.plies)),
        el("td", { class: "num dim" }, ago(g.finished)),
        el("td", { class: "dim" }, g.code_version)))))),
    el("div", { class: "controls", style: "margin-top:10px" },
      el("button", { ...(offset === 0 ? { disabled: "" } : {}), on: { click: () => drawGames(box, player, opponent, Math.max(0, offset - limit)) } }, "Newer"),
      el("button", { ...(offset + limit >= total ? { disabled: "" } : {}), on: { click: () => drawGames(box, player, opponent, offset + limit) } }, "Older"),
      el("span", { class: "dim" }, total ? `${offset + 1}–${Math.min(offset + limit, total)} of ${total.toLocaleString()}` : "0")));
}

export async function drawGamesPage(main) {
  const box = el("div", {});
  main.replaceChildren(el("div", { class: "page" }, el("div", { class: "controls" }, backButton()), box));
  await drawGames(box);
}

export function drawNetworks(main) {
  const nets = new Map();
  for (const p of state.summary.players) {
    if (p.network === null) continue;
    if (!nets.has(p.network)) nets.set(p.network, new Map());
    nets.get(p.network).set(p.sims, p);
  }
  const sims = [...new Set(state.summary.players.filter((p) => p.sims !== null).map((p) => p.sims))].sort((a, b) => a - b);
  const best = (entries) => Math.max(...[...entries.values()].map((p) => p.elo));
  const order = [...nets.entries()].sort((a, b) => best(b[1]) - best(a[1]));
  const color = new Map(order.map(([network], i) => [network, i < SERIES ? `var(--s${i + 1})` : "var(--other)"]));
  const engines = state.summary.players.filter((p) => p.external);

  const table = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Network"), el("th", {}, "Run"), el("th", {}, "Spec"),
      ...sims.map((s) => el("th", { class: "num" }, `${s} sims`)))),
    el("tbody", {}, ...order.map(([network, entries]) => {
      const any = [...entries.values()][0];
      return el("tr", {},
        el("td", {}, el("i", { style: `display:inline-block;width:14px;height:2px;margin-right:7px;vertical-align:middle;background:${color.get(network)}` }), network),
        el("td", { class: "dim" }, runOf(any)), el("td", { class: "dim" }, any.net),
        ...sims.map((s) => {
          const p = entries.get(s);
          return el("td", { class: "num" }, p ? el("a", { class: "name", href: `#player/${p.id}` }, `${elo(p.elo)} `, el("span", { class: "dim" }, `±${p.sd.toFixed(0)}`)) : "");
        }));
    })));

  const chartBox = el("div", { id: "chart" });
  main.replaceChildren(el("div", { class: "page" },
    el("div", { class: "controls" }, backButton()),
    el("div", { class: "card" }, chartBox,
      el("div", { class: "legend" }, ...order.slice(0, SERIES).map(([network]) =>
        el("span", {}, el("i", { style: `background:${color.get(network)}` }), network)),
        order.length > SERIES ? el("span", {}, el("i", { style: "background:var(--other)" }), "Other") : null)),
    el("h2", {}, "Ratings"),
    el("div", { class: "card" }, table)));
  drawChart(chartBox, order, sims, color, engines);
}

function niceStep(raw) {
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * mag).find((m) => m >= raw);
}

function drawChart(box, order, sims, color, engines) {
  if (!sims.length) return;
  const W = 1000, H = 400, L = 56, R = 110, T = 12, B = 30;
  const shown = [...order.flatMap(([, e]) => [...e.values()]), ...engines];
  const ys = shown.map((p) => p.elo);
  const [lo, hi] = ys.length ? [Math.min(...ys), Math.max(...ys)] : [-100, 100];
  const pad = Math.max(20, (hi - lo) * 0.08);
  const y = (v) => T + (hi + pad - v) / (hi - lo + 2 * pad) * (H - T - B);
  const logs = sims.map((s) => Math.log(s));
  const [l0, l1] = [logs[0], logs[logs.length - 1]];
  const x = (s) => (l1 === l0 ? (L + W - R) / 2 : L + (Math.log(s) - l0) / (l1 - l0) * (W - L - R));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}` });
  const step = niceStep((hi - lo + 2 * pad) / 6);
  for (let v = Math.ceil((lo - pad) / step) * step; v <= hi + pad; v += step) {
    root.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--line)" }));
    const t = svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }); t.textContent = elo(v); root.append(t);
  }
  for (const s of sims) {
    const t = svg("text", { x: x(s), y: H - 8, "text-anchor": "middle" }); t.textContent = String(s); root.append(t);
  }
  for (const e of engines) {
    root.append(svg("line", { x1: L, x2: W - R, y1: y(e.elo), y2: y(e.elo), stroke: "var(--ink-3)", "stroke-width": 1 }));
    const t = svg("text", { x: W - R + 6, y: y(e.elo) + 4 }); t.textContent = e.name; root.append(t);
  }
  // the least important first, so the top networks are drawn over them
  for (const [network, entries] of [...order].reverse()) {
    const dots = [...entries.values()].sort((a, b) => a.sims - b.sims);
    if (dots.length > 1) root.append(svg("polyline", {
      points: dots.map((p) => `${x(p.sims)},${y(p.elo)}`).join(" "), fill: "none",
      stroke: color.get(network), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
    for (const p of dots) root.append(svg("circle", {
      cx: x(p.sims), cy: y(p.elo), r: 4, fill: color.get(network), stroke: "var(--panel)", "stroke-width": 2,
    }));
  }
  const cross = svg("line", { y1: T, y2: H - B, stroke: "var(--line-strong)", visibility: "hidden" });
  root.append(cross);
  box.append(root);
  const tip = document.getElementById("tip");
  root.addEventListener("pointermove", (event) => {
    const rect = root.getBoundingClientRect();
    const mx = (event.clientX - rect.left) / rect.width * W;
    const s = sims.reduce((a, b) => (Math.abs(x(b) - mx) < Math.abs(x(a) - mx) ? b : a));
    cross.setAttribute("x1", x(s)); cross.setAttribute("x2", x(s)); cross.setAttribute("visibility", "visible");
    const rows = order.filter(([, e]) => e.has(s)).map(([n, e]) => [n, e.get(s)]).sort((a, b) => b[1].elo - a[1].elo);
    showTip(tip, event, el("div", {}, el("span", {}, `${s} sims`)), ...rows.map(([n, p]) =>
      el("div", {}, el("span", {}, el("i", { style: `background:${color.get(n)}` }), n), el("b", {}, `${elo(p.elo)} ±${p.sd.toFixed(0)}`))));
  });
  root.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
}
