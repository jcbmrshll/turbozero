// The league's dashboard: which page the address says, kept up to date with the server.

import { api, el, state } from "./dom.js";
import { drawGamePage, leaveGame } from "./game_view.js";
import { drawLeaderboard, leaderboard } from "./leaderboard.js";
import { drawGamesPage, drawNetworks, drawPlayer, onPlayerChange } from "./pages.js";
import { drawPareto, pareto } from "./pareto.js";
import { drawQueue, onQueueChange, queue } from "./queue.js";

const main = document.getElementById("main");
const columns = el("div", { class: "columns" }, leaderboard, el("section", { class: "column" }, pareto, queue));

function drawWorker() {
  const { worker, playing } = state.summary;
  const box = document.getElementById("worker");
  let text = "Not playing";
  if (playing) text = `${playing.a} vs ${playing.b}`;
  else if (worker.state === "queued") text = `Queued #${worker.job}`;
  else if (worker.state === "waiting") text = `Waiting: ${worker.reason ?? ""}`;
  box.className = playing ? "playing" : worker.state;
  document.getElementById("worker-text").textContent = text;
}

const page = () => (location.hash.slice(1) || "main").split("/");

async function route() {
  const [name, a, b] = page();
  document.body.classList.toggle("fixed", name === "main");
  for (const link of document.querySelectorAll("#nav a")) link.classList.toggle("on", link.getAttribute("href") === `#${name}`);
  if (name !== "game") leaveGame();
  if (name === "networks") drawNetworks(main);
  else if (name === "player") await drawPlayer(main, Number(a), b ? Number(b) : undefined);
  else if (name === "games") await drawGamesPage(main);
  else if (name === "game") await drawGamePage(main, Number(a));
  else {
    main.replaceChildren(columns);
    drawLeaderboard();
    drawQueue();
    await drawPareto();
  }
}

async function refresh() {
  const games = state.summary?.games;
  state.summary = await api("/api/summary");
  document.getElementById("title").textContent = `${state.summary.game} Elo league`;
  drawWorker();
  if (page()[0] === "main") {
    drawQueue();
    // reading the leaderboard isn't disturbed unless there's news
    if (games !== state.summary.games) {
      drawLeaderboard();
      await drawPareto();
    }
  }
}

onQueueChange(refresh);
onPlayerChange(async () => { await refresh(); await route(); });
await refresh();
window.addEventListener("hashchange", route);
await route();
setInterval(refresh, 10000);
