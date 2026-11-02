// The main page's plot: what a player's moves cost on fixed hardware against its rating. Each
// run of timings is one host, shown on its own, with that hardware named under the plot.

import { api, el, elo, showTip, svg } from "./dom.js";

let run = 0;
let int8 = localStorage.getItem("pareto-int8") === "1";
// measured int8 speedups with the search at batch 16, by the network's size; unmeasured sizes
// take the largest nets'. Batch 1 gains nothing. Ratings stay as they are, from bf16
const INT8_SPEEDUP = { "4x64": 1.26, "6x96": 1.38, "8x128": 1.66, "10x128": 1.59 };
const body = el("div", { class: "pareto-body" });
export const pareto = el("section", { class: "panel" }, el("h2", {}, "Runtime against rating"), body);

function seconds(s) {
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  return s < 100 ? `${s.toFixed(1)} s` : `${s.toFixed(0)} s`;
}

function hardware(found) {
  const h = found.hardware;
  const parts = [h.provider, h.cpu ?? h.model, h.cores && `${h.cores} cores`, h.threads && `${h.threads} threads`];
  return parts.filter(Boolean).join(" · ");
}

function decades(lo, hi) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 10 ** 0.5]) ticks.push(m * 10 ** e);
  }
  return ticks.filter((t) => t >= lo && t <= hi);
}

function withInt8(points) {
  const shown = points.map((p) => {
    const size = p.name.match(/^\d+x\d+/)?.[0];
    const speedup = size && p.batch > 1 ? INT8_SPEEDUP[size] ?? 1.6 : 1;
    return { ...p, seconds: p.seconds / speedup, frontier: false };
  });
  let best = -Infinity;
  for (const p of [...shown].sort((a, b) => a.seconds - b.seconds || b.elo - a.elo)) {
    if (p.solid && p.elo > best) { p.frontier = true; best = p.elo; }
  }
  return shown;
}

function plot(points) {
  const W = 640, H = 300, L = 48, R = 12, T = 8, B = 28;
  const xs = points.map((p) => Math.log10(p.seconds));
  const ys = points.map((p) => p.elo);
  const [x0, x1] = [Math.min(...xs) - 0.08, Math.max(...xs) + 0.08];
  const pad = Math.max(30, (Math.max(...ys) - Math.min(...ys)) * 0.06);
  const [y0, y1] = [Math.min(...ys) - pad, Math.max(...ys) + pad];
  const x = (s) => L + (Math.log10(s) - x0) / (x1 - x0) * (W - L - R);
  const y = (v) => T + (y1 - v) / (y1 - y0) * (H - T - B);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}` });
  const label = (attrs, words) => { const t = svg("text", attrs); t.textContent = words; root.append(t); };
  const step = [50, 100, 200, 500, 1000].find((s) => (y1 - y0) / s <= 6) ?? 2000;
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
    root.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--line)" }));
    label({ x: L - 6, y: y(v) + 4, "text-anchor": "end" }, elo(v));
  }
  for (const t of decades(10 ** x0, 10 ** x1)) {
    root.append(svg("line", { x1: x(t), x2: x(t), y1: H - B, y2: H - B + 4, stroke: "var(--line-strong)" }));
    label({ x: x(t), y: H - 8, "text-anchor": "middle" }, seconds(t));
  }
  const frontier = points.filter((p) => p.frontier).sort((a, b) => a.seconds - b.seconds);
  if (frontier.length > 1) {
    const line = frontier.map((p) => `${x(p.seconds)},${y(p.elo)}`).join(" ");
    root.append(svg("polyline", { points: line, fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  const tip = document.getElementById("tip");
  // the frontier's last, so that it's drawn over the rest
  for (const p of [...points].sort((a, b) => a.frontier - b.frontier)) {
    const fill = !p.solid ? "var(--panel)" : p.frontier ? "var(--accent)" : "var(--ink-3)";
    const stroke = p.solid ? "var(--panel)" : "var(--ink-2)";
    const mark = svg("circle", { cx: x(p.seconds), cy: y(p.elo), r: 4.5, fill, stroke, "stroke-width": p.solid ? 2 : 1.5 });
    const target = svg("circle", { cx: x(p.seconds), cy: y(p.elo), r: 10, fill: "transparent", style: "cursor:pointer" });
    target.addEventListener("pointermove", (event) => showTip(tip, event,
      el("div", {}, el("b", {}, p.name)),
      el("div", {}, el("span", {}, "Rating"), el("b", {}, `${elo(p.elo)} ±${p.sd.toFixed(0)}`)),
      el("div", {}, el("span", {}, "Per game"), el("b", {}, seconds(p.seconds))),
      p.batch ? el("div", {}, el("span", {}, "Batch"), el("b", {}, `${p.batch}${p.solid ? "" : " unverified"}`)) : null));
    target.addEventListener("pointerleave", () => { tip.style.display = "none"; });
    target.addEventListener("click", () => { tip.style.display = "none"; location.hash = `#player/${p.player}`; });
    root.append(mark, target);
  }
  return root;
}

export async function drawPareto() {
  const runs = await api("/api/pareto");
  const shown = runs.filter((r) => r.points.length);
  if (!shown.length) { body.replaceChildren(el("div", { class: "empty dim" }, "No timings")); return; }
  run = Math.min(run, shown.length - 1);
  const found = shown[run];
  const choose = shown.length > 1 ? el("select", { on: { change: (e) => { run = Number(e.target.value); drawPareto(); } } },
    ...shown.map((r, i) => { const o = el("option", { value: i }, r.name); if (i === run) o.selected = true; return o; })) : null;
  const toggle = el("input", { type: "checkbox", on: { change: (e) => {
    int8 = e.target.checked;
    localStorage.setItem("pareto-int8", int8 ? "1" : "0");
    drawPareto();
  } } });
  toggle.checked = int8;
  body.replaceChildren(plot(int8 ? withInt8(found.points) : found.points),
    el("div", { class: "under dim" }, choose,
      el("label", {}, toggle, "int8"),
      el("span", {}, hardware(found)),
      el("span", { class: "keys" },
        el("i", { class: "key on" }), "Frontier", el("i", { class: "key" }), "Below", el("i", { class: "key open" }), "Batch unverified")));
}
