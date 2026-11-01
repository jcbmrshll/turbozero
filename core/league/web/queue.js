// The main page's queue: what's playing, the series asked for in the order they'll be played,
// and the batches just played.

import { ago, byId, el, send, state } from "./dom.js";

const now = el("div", { class: "now" });
const rows = el("tbody");
const recent = el("tbody");
const error = el("span", { class: "error" });
const nameList = el("datalist", { id: "player-names" });
const inputA = el("input", { list: "player-names", placeholder: "Player" });
const inputB = el("input", { list: "player-names", placeholder: "Opponent" });
const inputGames = el("input", { type: "number", min: 2, step: 2, value: 64, class: "games" });
const form = el("form", { class: "add", on: { submit: addMatch } }, inputA, el("span", { class: "dim" }, "vs"), inputB, inputGames, el("button", {}, "Add"), error, nameList);
export const queue = el("section", { class: "panel fill" }, el("h2", {}, "Queue"), now, form,
  el("div", { class: "scroll" },
    el("table", { class: "series" }, rows),
    el("h2", {}, "Recent"), el("table", {}, recent)));

let dragging = null;
let refresh = () => {};
export function onQueueChange(f) { refresh = f; }

async function act(path, body) {
  const answer = await send(path, body);
  error.textContent = answer.error ?? "";
  await refresh();
  return answer;
}

async function addMatch(event) {
  event.preventDefault();
  const named = (input) => state.summary.players.find((p) => p.name === input.value);
  const [a, b] = [named(inputA), named(inputB)];
  if (!a || !b) { error.textContent = `No player ${a ? inputB.value : inputA.value}`; return; }
  const answer = await act("/api/match/add", { a: a.id, b: b.id, games: Number(inputGames.value) });
  if (!answer.error) { inputA.value = ""; inputB.value = ""; }
}

function drawNow() {
  const { playing, worker } = state.summary;
  if (playing) {
    now.className = "now playing";
    now.replaceChildren(el("span", { class: "dot" }), el("b", {}, `${playing.a} vs ${playing.b}`),
      el("span", { class: "dim" }, `${playing.games} games`),
      el("span", { class: "dim" }, playing.request ? `#${playing.request}` : "scheduler"),
      el("span", { class: "dim right" }, ago(playing.started)));
  } else {
    now.className = "now";
    const words = worker.state === "queued" ? `Queued #${worker.job}` : worker.state === "waiting" ? `Waiting: ${worker.reason ?? ""}` : "Not playing";
    now.replaceChildren(el("span", { class: "dot" }), el("span", { class: "dim" }, words));
  }
}

function drawRow(request, i, all) {
  const [a, b] = [byId(request.a), byId(request.b)];
  const row = el("tr", {
    draggable: "true", class: `${request.status === "playing" ? "playing" : ""}`, "data-id": request.id,
    on: {
      dragstart: () => { dragging = request.id; row.classList.add("dragging"); },
      dragend: () => { dragging = null; row.classList.remove("dragging"); },
      dragover: (event) => { event.preventDefault(); row.classList.add("over"); },
      dragleave: () => row.classList.remove("over"),
      drop: (event) => {
        event.preventDefault();
        row.classList.remove("over");
        if (dragging !== null && dragging !== request.id) act("/api/match/move", { id: dragging, index: i });
      },
    },
  },
    el("td", { class: "dim handle" }, "⠿"),
    el("td", { class: "dim num" }, String(i + 1)),
    el("td", { class: "pair" }, `${a.name} vs ${b.name}`),
    el("td", {}, request.status ? el("span", { class: `tag ${request.status === "playing" ? "live" : ""}` }, request.status) : null),
    el("td", { class: "num dim" }, `${request.played} /`),
    el("td", { class: "num" }, el("input", {
      type: "number", class: "games", min: 2, step: 2, value: request.games,
      on: { change: (e) => act("/api/match/games", { id: request.id, games: Number(e.target.value) }) },
    })),
    el("td", { class: "actions" },
      el("button", { ...(i === 0 ? { disabled: "" } : {}), on: { click: () => act("/api/match/move", { id: request.id, index: i - 1 }) } }, "↑"),
      el("button", { ...(i === all.length - 1 ? { disabled: "" } : {}), on: { click: () => act("/api/match/move", { id: request.id, index: i + 1 }) } }, "↓"),
      el("button", { on: { click: () => act("/api/match/cancel", { id: request.id }) } }, "×")));
  return row;
}

export function drawQueue() {
  const { summary } = state;
  drawNow();
  nameList.replaceChildren(...summary.players.filter((p) => p.can_play && !p.retired).map((p) => el("option", { value: p.name })));
  // not from under someone's hands
  if (dragging === null && !(rows.contains(document.activeElement) && document.activeElement.tagName === "INPUT")) {
    rows.replaceChildren(...summary.requests.map(drawRow));
  }
  recent.replaceChildren(...summary.recent.map((b) => el("tr", {},
    el("td", { class: "pair" }, `${b.a} vs ${b.b}`),
    el("td", { class: "num" }, `${b.games} games`),
    el("td", { class: "num dim" }, `${b.seconds.toFixed(0)} s`),
    el("td", { class: "num dim" }, ago(b.finished)))));
}
