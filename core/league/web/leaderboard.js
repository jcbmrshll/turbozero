// The main page's left column: the ratings, filtered by what's known about each player.

import { byId, duration, el, elo, perMove, state, tagOf } from "./dom.js";

const STORED = "league-filters";
let sortKey = "elo", sortDir = -1;
let chosen = {};
let text = "";
let showPaused = false;
try { ({ chosen = {}, text = "", showPaused = false } = JSON.parse(localStorage.getItem(STORED)) ?? {}); } catch {}

const list = el("tbody");
const count = el("span", { class: "dim" });
const head = el("thead");
const facetBar = el("div", { class: "facets" });
const stats = el("div", { class: "stats dim" });
export const leaderboard = el("section", { class: "column" }, stats, facetBar,
  el("div", { class: "scroll" }, el("table", { class: "sticky" }, head, list)));

function status(player) {
  return !player.can_play ? "external" : player.retired ? "paused" : "active";
}

// label to the values each player has, a player's being one or several
function valuesOf(player) {
  return { ...Object.fromEntries(Object.entries(player.facts).map(([k, v]) => [k, [String(v)]])), status: [status(player)], tags: player.tags };
}

// each label and its values, where choosing among them would leave someone out
function facets() {
  const found = new Map(), holders = new Map();
  for (const player of state.summary.players) {
    for (const [label, values] of Object.entries(valuesOf(player))) {
      if (!found.has(label)) found.set(label, new Set());
      for (const value of values) found.get(label).add(value);
      if (values.length) holders.set(label, (holders.get(label) ?? 0) + 1);
    }
  }
  const ordered = (values) => [...values].sort((a, b) => (Number(a) - Number(b)) || a.localeCompare(b));
  return [...found.entries()]
    .filter(([label, values]) => values.size > 1 || (values.size === 1 && holders.get(label) < state.summary.players.length))
    .map(([label, values]) => [label, ordered(values)]);
}

function matches(player) {
  if (player.retired && !showPaused && !chosen.status?.includes(status(player))) return false;
  if (text && !player.name.toLowerCase().includes(text.toLowerCase())) return false;
  const values = valuesOf(player);
  return Object.entries(chosen).every(([label, wanted]) => !wanted.length || (values[label] ?? []).some((v) => wanted.includes(v)));
}

function save() {
  localStorage.setItem(STORED, JSON.stringify({ chosen, text, showPaused }));
}

function drawFacets() {
  const open = new Set([...facetBar.querySelectorAll("details[open]")].map((d) => d.dataset.label));
  const search = el("input", { type: "search", placeholder: "Player", value: text, on: { input: (e) => { text = e.target.value; save(); drawRows(); } } });
  const active = Object.values(chosen).some((wanted) => wanted.length) || text;
  const paused = el("input", { type: "checkbox", on: { change: (e) => { showPaused = e.target.checked; save(); drawRows(); } } });
  paused.checked = showPaused;
  facetBar.replaceChildren(search, el("label", {}, paused, "Paused"), ...facets().map(([label, values]) => {
    const wanted = chosen[label] ?? [];
    const details = el("details", { "data-label": label, class: wanted.length ? "facet on" : "facet" },
      el("summary", {}, wanted.length ? `${label}: ${wanted.join(", ")}` : label),
      el("div", { class: "chips" }, ...values.map((value) => el("button", {
        class: wanted.includes(value) ? "chip on" : "chip",
        on: { click: () => {
          chosen[label] = wanted.includes(value) ? wanted.filter((v) => v !== value) : [...wanted, value];
          save(); drawFacets(); drawRows();
        } },
      }, value))));
    details.open = open.has(label);
    return details;
  }), ...(active ? [el("button", { on: { click: () => { chosen = {}; text = ""; save(); drawFacets(); drawRows(); } } }, "Clear")] : []), count);
  for (const details of facetBar.querySelectorAll("details")) {
    details.addEventListener("toggle", () => {
      if (details.open) for (const other of facetBar.querySelectorAll("details")) if (other !== details) other.open = false;
    });
  }
}

function drawHead() {
  const columns = [["elo", "Rating", true], ["sd", "±", true], ["games", "Games", true], ["seconds_per_move", "Per move", true], ["name", "Player", false]];
  head.replaceChildren(el("tr", {}, el("th", { class: "num" }, "#"), ...columns.map(([key, label, num]) =>
    el("th", {
      class: num ? "num" : "", "data-sort": key,
      on: { click: () => { if (sortKey === key) sortDir = -sortDir; else { sortKey = key; sortDir = -1; } drawHead(); drawRows(); } },
    }, label + (sortKey === key ? (sortDir < 0 ? " ↓" : " ↑") : "")))));
}

function drawRows() {
  const players = state.summary.players.filter(matches).sort((a, b) => {
    // whoever hasn't played has no rating to sort by
    if (!a.games !== !b.games) return a.games ? -1 : 1;
    const [x, y] = [a[sortKey] ?? -Infinity, b[sortKey] ?? -Infinity];
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  count.textContent = `${players.length} of ${state.summary.players.length}`;
  list.replaceChildren(...players.map((p, i) => el("tr", { class: "link", on: { click: () => { location.hash = `#player/${p.id}`; } } },
    el("td", { class: "num dim" }, String(i + 1)),
    el("td", { class: "num" }, p.games ? elo(p.elo) : "–"),
    el("td", { class: "num dim" }, p.games ? p.sd.toFixed(0) : "–"),
    el("td", { class: "num" }, String(p.games)),
    el("td", { class: "num dim" }, perMove(p)),
    el("td", {}, p.name, tagOf(p), ...p.tags.map((tag) => el("span", { class: "tag own" }, tag))))));
}

function drawStats() {
  const { summary } = state;
  const anchor = summary.anchor ? byId(summary.anchor) : null;
  const worst = Math.max(0, ...summary.players.filter((p) => !p.external && !p.retired).map((p) => p.sd));
  const parts = [`${summary.players.length} players`, `${summary.games.toLocaleString()} games`, `${duration(summary.seconds)} playing`,
    `${anchor ? anchor.name : "mean"} at 0`, `worst network ±${worst.toFixed(0)}`];
  if (summary.parts > 1) parts.push(`${summary.parts} parts`);
  stats.textContent = parts.join(" · ");
}

export function drawLeaderboard() {
  drawStats();
  if (facetBar.contains(document.activeElement) && document.activeElement.tagName === "INPUT") { drawRows(); return; }
  drawFacets();
  drawHead();
  drawRows();
}
