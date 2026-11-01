// What every page of the league's dashboard shares: building its elements, talking to its
// server, and what it says about a player.

export const state = { summary: null };

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "on") for (const [e, f] of Object.entries(value)) node.addEventListener(e, f);
    else if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...children);
  return node;
}

export const api = (path) => fetch(path).then((r) => r.json());

// writes something, and resolves to the server's answer, which holds `error` if it refused
export const send = (path, body) =>
  fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());

export const elo = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(0);
export const byId = (id) => state.summary.players.find((p) => p.id === id);

export function ago(t) {
  const s = Math.max(0, state.summary.time - t);
  if (s < 90) return `${s.toFixed(0)}s`;
  if (s < 5400) return `${(s / 60).toFixed(0)}m`;
  if (s < 129600) return `${(s / 3600).toFixed(0)}h`;
  return `${(s / 86400).toFixed(0)}d`;
}

export const duration = (s) => (s < 5400 ? `${(s / 60).toFixed(0)}m` : `${(s / 3600).toFixed(1)}h`);

export function perMove(player) {
  const s = player.seconds_per_move;
  if (s === null || s === undefined) return "–";
  return s < 1 ? `${(s * 1000).toFixed(s < 0.01 ? 1 : 0)} ms` : `${s.toFixed(1)} s`;
}

export function tagOf(player) {
  const label = !player.can_play ? "external" : player.retired ? "paused" : player.external ? "engine" : null;
  return label && el("span", { class: "tag" }, label);
}

export function tile(value, label) {
  return el("div", { class: "tile" }, el("b", {}, value), el("span", {}, label));
}

export function runOf(player) {
  if (!player.url) return player.run;
  return el("a", { class: "name run", href: player.url, target: "_blank", rel: "noreferrer" }, player.run);
}

// how many pages into the dashboard this tab has gone, to know that "back" stays in it
let depth = 0;
window.addEventListener("hashchange", () => { depth += 1; });

export function backButton() {
  return el("button", { on: { click: () => {
    if (depth > 0) { depth -= 2; history.back(); } else location.hash = "";
  } } }, "← Back");
}

export function showTip(tip, event, ...rows) {
  tip.replaceChildren(...rows.filter(Boolean));
  tip.style.display = "block";
  tip.style.left = `${Math.min(event.pageX + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
  tip.style.top = `${event.pageY + 14}px`;
}
