// A game's page: its positions on the board, with a trained network's view of each worked out in
// the browser. Of a game's searches only a summary was kept (`game.searches`, none for older games),
// so the tree shown is searched again here, by default by the player to move's network at the sims
// it played at. An engine compiled to WebAssembly can look instead, if the store has them.

import board from "/game.js";
import { BrowserEngine } from "./browser_engine.js";
import { api, backButton, byId, el, state, svg, tile } from "./dom.js";
import { drawTree, forFirst, indexTree, nodeAt, positionAt, valueColor } from "./tree.js";

const SIMS = [0, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000];
const DEFAULT_SIMS = 128;
// the most iterations pondering goes to, as the tree grows with each one
const PONDER_LIMIT = 50000;
// the most finished searches to keep, for coming back to a position
const CACHED = 200;
// how long (ms) a position stays before an engine searches it: stepping past positions shouldn't
// cost a search each, which takes seconds to end and, for Egaroucid, a second to start again
const ENGINE_SETTLE = 200;
// the most an engine's score can be, in discs
const MAX_SCORE = 64;

let leave = null;
export function leaveGame() {
  if (leave) leave();
  leave = null;
}

// the networks that can run here, the best rated first, one entry each
function loadableNets() {
  const best = new Map();
  for (const p of state.summary.players) {
    if (p.onnx && (!best.has(p.onnx) || p.elo > best.get(p.onnx).elo)) best.set(p.onnx, p);
  }
  return [...best.values()].sort((a, b) => b.elo - a.elo);
}

const resultText = (result) => (result > 0 ? "1 – 0" : result < 0 ? "0 – 1" : "½ – ½");

export async function drawGamePage(main, id) {
  leaveGame();
  const game = await api(`/api/game?id=${id}`);
  // gone elsewhere while it loaded
  if (location.hash !== `#game/${id}`) return;
  if (game.error) { main.replaceChildren(el("div", { class: "page" }, el("div", { class: "controls" }, backButton()), el("p", { class: "dim" }, game.error))); return; }
  const players = [byId(game.first_id), byId(game.second_id)];
  const states = [board.start()];
  for (const move of game.moves) states.push(board.play(states[states.length - 1], move));

  const nets = loadableNets();
  const canAnalyse = state.summary.runtime && nets.length > 0 && !!board.observe;
  const worker = canAnalyse ? new Worker(`/web/engine_worker.js?ort=${encodeURIComponent(state.summary.runtime)}`, { type: "module" }) : null;
  const engines = state.summary.browser_engines && board.engines ? board.engines : null;
  const browserEngine = engines ? new BrowserEngine(state.summary.browser_engines, onEngineMessage) : null;
  leave = () => { worker?.terminate(); browserEngine?.close(); clearTimeout(settling); document.removeEventListener("keydown", onKey); };

  let at = game.opening_plies;
  let engine = null; // the engine looking, if any: none until it's picked, as each takes memory
  const levels = new Map(); // each engine's level set here
  let engineLook = null; // what it's made of the position: { key, id, running, position, step }
  let settling = null;
  let mode = "mover"; // or the name of a network's export
  let override = null; // the sims set here, rather than the game's
  let ponder = false;
  let overlay = "visits";
  let hidden = false;
  let path = null, preview = null;
  let current = null; // the search shown: { key, ix, n, q, value, done, of, running }
  let stops = 0, nextId = 1, searchId = 0, timelineId = 0;
  let msPerEval = null, failure = null;
  const finished = new Map();
  const timeline = new Array(states.length).fill(null);

  // whose network looks at position `i`, and how long for
  function looker(i) {
    if (mode !== "mover") {
      const p = nets.find((n) => n.onnx === mode);
      return { onnx: mode, sims: override ?? DEFAULT_SIMS, name: p.network };
    }
    const turn = states[i].turn - 1;
    const p = [players[turn], players[1 - turn]].find((q) => q?.onnx) ?? nets[0];
    return { onnx: p.onnx, sims: override ?? p.sims ?? DEFAULT_SIMS, name: p.network };
  }

  const boardBox = el("div", { class: "board-wrap" });
  const buttons = game.moves.map((move, i) => el("button", {
    class: i < game.opening_plies ? "opening" : "", on: { click: () => goTo(i + 1) },
  }, `${i + 1}. ${board.moveName(move)}`));
  const slider = el("input", { type: "range", min: 0, max: game.moves.length, value: at, class: "scrub", on: { input: (e) => goTo(Number(e.target.value)) } });
  const firstCount = el("span", { class: "count" }), secondCount = el("span", { class: "count" });
  const prev = el("button", { on: { click: () => goTo(at - 1) } }, "←");
  const next = el("button", { on: { click: () => goTo(at + 1) } }, "→");
  const toStart = el("button", { on: { click: () => goTo(0) } }, "⏮");
  const toEnd = el("button", { on: { click: () => goTo(game.moves.length) } }, "⏭");
  const evalShares = [el("span", { class: "share" }), el("span", { class: "share" })];
  const evalFill = el("div", {});
  const timelineBox = el("div", { class: "timeline" });
  // rows are replaced as a search goes on, so one that's left may never say so
  const treeBox = el("div", { class: "tree-body", on: { pointerleave: () => setPreview(null) } });
  const thinking = el("div", { class: "thinking" });
  const readouts = el("div", { class: "readouts dim" });
  const signed = (v, digits) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;
  const rootValue = el("span", {});
  const rootLabel = el("span", {});

  function drawRootValue() {
    const search = game.searches[at];
    rootLabel.textContent = search?.score !== undefined ? "score" : "root value";
    rootValue.textContent = !search ? "–" : search.score !== undefined ? signed(search.score, 0) : signed(search.value, 3);
  }

  const shownPath = () => (preview ?? path);

  function drawBoard() {
    const shown = shownPath();
    const ix = current?.ix;
    if (engine) {
      board.draw(boardBox, states[at], at ? game.moves[at - 1] : null, { marks: engineMarks(), played: game.moves[at] ?? null });
      drawCounts(states[at]);
      return;
    }
    if (shown && ix) {
      const position = positionAt(board, ix.tree.position, shown);
      board.draw(boardBox, position, shown.length ? shown[shown.length - 1] : (at ? game.moves[at - 1] : null), { marks: preview ? new Map() : marksFor(ix, path ?? []), on: boardEvents() });
      drawCounts(position);
      return;
    }
    board.draw(boardBox, states[at], at ? game.moves[at - 1] : null, { marks: ix ? marksFor(ix, []) : new Map(), played: game.moves[at] ?? null, on: boardEvents() });
    drawCounts(states[at]);
  }
  function drawCounts(position) {
    const counts = board.counts?.(position) ?? ["", ""];
    firstCount.textContent = counts[0]; secondCount.textContent = counts[1];
  }
  function boardEvents() {
    return {
      enter: (move) => { if (path && markable(move)) setPreview([...path, move]); },
      leave: () => { if (path) setPreview(null); },
      click: (move) => { if (markable(move)) go([...(path ?? []), move]); },
    };
  }
  function markable(move) {
    const ix = current?.ix;
    if (!ix) return false;
    const g = nodeAt(ix, path ?? []);
    return g >= 0 && ix.children[g].some((c) => ix.tree.action[c] === move);
  }
  function marksFor(ix, from) {
    const marks = new Map();
    const g = nodeAt(ix, from);
    if (g < 0) return marks;
    const { tree, children } = ix;
    if (overlay === "policy") {
      const priors = from.length === 0 && tree.policy
        ? board.legal(tree.position).map((a) => [a, tree.policy[a], null])
        : children[g].map((c) => [tree.action[c], tree.prior[c], tree.q[c]]);
      for (const [a, prior] of priors) marks.set(a, { size: prior, color: valueColor(prior), label: `${Math.round(prior * 100)}`, ink: prior > 0.3 ? "#fff" : undefined });
      return marks;
    }
    const total = children[g].reduce((k, c) => k + tree.n[c], 0);
    for (const c of children[g]) {
      const v = -tree.q[c];
      marks.set(tree.action[c], { size: total ? tree.n[c] / total : 0, color: valueColor(v), label: String(tree.n[c]), ink: Math.abs(v) > 0.3 ? "#fff" : undefined });
    }
    return marks;
  }

  const engineStep = () => engineLook?.step ?? null;
  const levelOf = (name) => levels.get(name) ?? engines[name].level;
  function engineMarks() {
    const step = engineStep();
    const marks = new Map();
    if (step) marks.set(step.move, { size: 1, color: valueColor(step.score / MAX_SCORE), label: signed(step.score, 0), ink: Math.abs(step.score) > 6 ? "#fff" : undefined });
    return marks;
  }

  function drawEval() {
    if (engine) {
      const step = engineStep();
      const score = step ? (states[at].turn === 1 ? step.score : -step.score) : null;
      evalFill.style.width = `${score === null ? 50 : (score + MAX_SCORE) / (2 * MAX_SCORE) * 100}%`;
      evalShares[0].textContent = score === null ? "" : signed(score, 0);
      evalShares[1].textContent = score === null ? "" : signed(-score, 0);
      return;
    }
    const ix = current?.ix;
    const shown = shownPath() ?? [];
    let value = timeline[at];
    if (ix) {
      const g = nodeAt(ix, shown);
      if (g >= 0) value = forFirst(ix.tree, ix.tree.q[g], shown.length);
    }
    const share = value === null ? 0.5 : (value + 1) / 2;
    evalFill.style.width = `${share * 100}%`;
    evalShares[0].textContent = value === null ? "" : `${Math.round(share * 100)}%`;
    evalShares[1].textContent = value === null ? "" : `${Math.round((1 - share) * 100)}%`;
  }

  function drawTimeline() {
    const W = 640, H = 64, P = 2;
    const count = Math.max(1, states.length - 1);
    const x = (i) => P + i / count * (W - 2 * P);
    const y = (v) => H / 2 - v * (H / 2 - P);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
    root.append(svg("rect", { x: 0, y: 0, width: x(game.opening_plies), height: H, fill: "var(--line)", opacity: 0.5 }));
    root.append(svg("line", { x1: 0, x2: W, y1: H / 2, y2: H / 2, stroke: "var(--line-strong)" }));
    const known = timeline.map((v, i) => [i, v]).filter(([, v]) => v !== null);
    if (known.length > 1) {
      root.append(svg("polyline", { points: known.map(([i, v]) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
    }
    root.append(svg("line", { x1: x(at), x2: x(at), y1: 0, y2: H, stroke: "var(--ink-2)", "vector-effect": "non-scaling-stroke" }));
    root.addEventListener("click", (event) => {
      const rect = root.getBoundingClientRect();
      goTo(Math.round((event.clientX - rect.left) / rect.width * count));
    });
    timelineBox.replaceChildren(root);
  }

  function drawThinking() {
    thinking.replaceChildren();
    if (engine) {
      if (engineLook?.running) {
        const step = engineStep();
        thinking.append(el("i", { class: "shimmer" }), el("span", {}, step ? `depth ${step.depth} at ${step.percent}%` : "loading"));
      }
      return;
    }
    if (current?.running) {
      const fill = ponder ? null : current.done / Math.max(1, current.of);
      thinking.append(el("i", { class: fill === null ? "shimmer" : "" }, fill === null ? null : el("b", { style: `width:${fill * 100}%` })),
        el("span", {}, `${current.done.toLocaleString()} sims`));
    }
  }

  function drawEngineAnalysis() {
    drawThinking();
    const step = engineStep();
    const parts = [`${engines[engine].label} level ${levelOf(engine)}`];
    if (step) parts.push(`depth ${step.depth} at ${step.percent}%`, `${step.nodes.toLocaleString()} nodes`, `${(step.ms / 1000).toFixed(1)} s`);
    if (failure) parts.push(failure);
    readouts.textContent = parts.join(" · ");
    treeBox.replaceChildren(...(step ? [el("div", { class: "engine-line" }, el("b", {}, board.moveName(step.move)), el("span", {}, `${signed(step.score, 0)} discs for ${board.sideNames[states[at].turn - 1]}`))] : []));
  }

  function drawAnalysis() {
    if (engine) { drawEngineAnalysis(); return; }
    const ix = current?.ix;
    const look = canAnalyse ? looker(at) : null;
    drawThinking();
    const parts = [];
    if (look) parts.push(look.name);
    if (current?.value !== undefined && current?.value !== null) parts.push(`net ${current.value >= 0 ? "+" : "−"}${Math.abs(current.value).toFixed(3)}`);
    if (msPerEval !== null) parts.push(`${msPerEval.toFixed(1)} ms/eval`);
    if (failure) parts.push(failure);
    readouts.textContent = parts.join(" · ");
    if (ix) drawTree(treeBox, { board, ix, path, preview, played: game.moves[at] ?? null, go, setPreview });
    else treeBox.replaceChildren();
  }

  function setPreview(to) {
    const same = (a, b) => (a?.join(",") ?? null) === (b?.join(",") ?? null);
    if (same(preview, to)) return;
    preview = to;
    drawBoard(); drawEval();
  }
  function go(to) {
    path = to; preview = null;
    drawBoard(); drawEval(); drawAnalysis();
  }

  function stopSearch() {
    stops += 1;
    worker?.postMessage({ type: "stop", count: stops });
  }

  function analyseWithEngine() {
    clearTimeout(settling);
    browserEngine.stop();
    failure = null;
    const position = states[at];
    const level = levelOf(engine);
    const key = `${engine}|${level}|${at}`;
    if (position.terminated) engineLook = null;
    else if (finished.has(key)) engineLook = finished.get(key);
    else {
      const look = engineLook = { key, id: null, running: true, step: null };
      const name = engine;
      settling = setTimeout(() => { look.id = browserEngine.search(name, position, level); }, ENGINE_SETTLE);
    }
    drawAnalysis();
  }

  function onEngineMessage(data) {
    const look = engineLook;
    if (data.id !== look?.id) return;
    if (data.type === "error") {
      failure = data.message;
      look.running = false;
    } else {
      look.step = data;
      look.running = data.type === "progress";
      if (data.type === "result") {
        finished.set(look.key, look);
        if (finished.size > CACHED) finished.delete(finished.keys().next().value);
      }
    }
    drawBoard(); drawEval(); drawAnalysis();
  }

  function analyse() {
    if (engine) { analyseWithEngine(); return; }
    if (!worker) { drawAnalysis(); return; }
    stopSearch();
    searchId = 0;
    path = null; preview = null;
    const position = states[at];
    if (position.terminated) { current = null; drawAnalysis(); return; }
    const look = looker(at);
    const iterations = ponder ? PONDER_LIMIT : look.sims;
    const key = `${look.onnx}|${iterations}|${at}`;
    if (finished.has(key)) { current = finished.get(key); drawAnalysis(); return; }
    searchId = nextId++;
    current = { key, ix: null, done: 0, of: iterations, running: true, value: null };
    worker.postMessage({ type: "search", id: searchId, net: look.onnx, position, iterations, after: stops });
    drawAnalysis();
  }

  function startTimeline() {
    if (!worker) return;
    timeline.fill(null);
    timelineId = nextId++;
    const groups = new Map();
    states.forEach((position, i) => {
      if (position.terminated) {
        timeline[i] = position.turn === 1 ? position.reward : -position.reward;
        return;
      }
      const { onnx } = looker(i);
      if (!groups.has(onnx)) groups.set(onnx, []);
      groups.get(onnx).push(i);
    });
    for (const [onnx, indices] of groups) {
      worker.postMessage({ type: "values", id: timelineId, net: onnx, indices, positions: indices.map((i) => states[i]) });
    }
    drawTimeline();
  }

  if (worker) {
    worker.onmessage = ({ data }) => {
      if (data.type === "values") {
        if (data.id !== timelineId) return;
        data.indices.forEach((i, k) => { timeline[i] = states[i].turn === 1 ? data.values[k] : -data.values[k]; });
        drawTimeline();
        drawEval();
      } else if (data.type === "error") {
        failure = data.message;
        if (current) current.running = false;
        drawAnalysis();
      } else if ((data.type === "progress" || data.type === "result") && data.id === searchId) {
        msPerEval = data.msPerEval ?? msPerEval;
        if (data.tree) current.ix = indexTree(data.tree);
        current.done = data.type === "result" ? current.of : data.done;
        current.value = data.value;
        current.running = data.type === "progress";
        if (data.type === "result") {
          finished.set(current.key, current);
          if (finished.size > CACHED) finished.delete(finished.keys().next().value);
        }
        // the tree comes a few times a second, and the count far more often
        if (data.tree) { drawBoard(); drawEval(); drawAnalysis(); } else drawThinking();
      }
    };
  }

  function goTo(i) {
    at = Math.max(0, Math.min(game.moves.length, i));
    buttons.forEach((b, k) => b.classList.toggle("on", k === at - 1));
    slider.value = at;
    prev.disabled = at === 0; next.disabled = at === game.moves.length;
    analyse();
    drawBoard(); drawEval(); drawTimeline(); drawRootValue();
    drawSimsControl();
  }

  const options = [...new Set([...SIMS, ...players.map((p) => p?.sims).filter((s) => s !== null && s !== undefined)])].sort((a, b) => a - b);
  const simsBox = el("span", { class: "sims-control" });
  function drawSimsControl() {
    if (engine) {
      const level = levelOf(engine);
      simsBox.className = "sims-control";
      simsBox.replaceChildren(
        el("input", { type: "range", min: 1, max: engines[engine].maxLevel, value: level, on: { change: (e) => { levels.set(engine, Number(e.target.value)); drawSimsControl(); analyse(); drawBoard(); drawEval(); } } }),
        el("span", {}, `level ${level}`));
      return;
    }
    if (!canAnalyse) { simsBox.replaceChildren(); return; }
    const sims = looker(at).sims;
    const nearest = options.reduce((best, s, i) => (Math.abs(s - sims) < Math.abs(options[best] - sims) ? i : best), 0);
    simsBox.className = ponder ? "sims-control off" : "sims-control";
    simsBox.replaceChildren(
      el("input", { type: "range", min: 0, max: options.length - 1, value: nearest, ...(ponder ? { disabled: "" } : {}), on: { change: (e) => { override = options[Number(e.target.value)]; drawSimsControl(); analyse(); drawBoard(); drawEval(); } } }),
      el("span", {}, `${sims.toLocaleString()} sims`),
      ...(override !== null && !ponder ? [el("button", { on: { click: () => { override = null; drawSimsControl(); analyse(); drawBoard(); drawEval(); } } }, "Game")] : []));
  }

  const netSelect = canAnalyse || engines ? el("select", { on: { change: (e) => {
    const [, picked] = /^engine:(.*)$/.exec(e.target.value) ?? [];
    if (engine && !picked) { engineLook = null; browserEngine.close(); clearTimeout(settling); current = null; failure = null; }
    if (picked) { stopSearch(); current = null; searchId = 0; }
    engine = picked ?? null;
    netOnly.forEach((part) => { part.hidden = !canAnalyse || !!engine; });
    if (canAnalyse && !picked && e.target.value !== mode) { mode = e.target.value; finished.clear(); startTimeline(); }
    drawSimsControl(); analyse(); drawBoard(); drawEval();
  } } },
    ...(canAnalyse ? [el("option", { value: "mover" }, "Player to move"), ...nets.map((p) => el("option", { value: p.onnx }, p.network))] : [el("option", { value: "" }, "None")]),
    ...Object.entries(engines ?? {}).map(([name, spec]) => el("option", { value: `engine:${name}` }, spec.label))) : null;
  const overlayGroup = el("span", { class: "segment" }, ...["visits", "policy"].map((name) => el("button", {
    class: overlay === name ? "on" : "", "data-name": name,
    on: { click: () => {
      overlay = name;
      for (const b of overlayGroup.children) b.classList.toggle("on", b.dataset.name === name);
      drawBoard();
    } },
  }, name === "visits" ? "Visits" : "Policy")));
  const panel = el("div", { class: "analysis" });
  const eye = el("button", { on: { click: () => {
    hidden = !hidden;
    panel.classList.toggle("frosted", hidden);
    eye.textContent = hidden ? "Show" : "Hide";
  } } }, "Hide");
  const ponderBox = el("label", { class: "check" }, el("input", { type: "checkbox", on: { change: (e) => { ponder = e.target.checked; drawSimsControl(); analyse(); drawBoard(); drawEval(); } } }), "Ponder");
  const netOnly = [ponderBox, overlayGroup];
  netOnly.forEach((part) => { part.hidden = !canAnalyse; });
  panel.append(
    el("div", { class: "analysis-head" }, el("h2", {}, "Analysis"), thinking, eye),
    el("div", { class: "controls compact" }, netSelect, simsBox, ponderBox, overlayGroup),
    treeBox, readouts);

  const playerName = (p, name) => (p ? el("a", { class: "name", href: `#player/${p.id}` }, name) : name);
  const discs = board.sideColors ?? ["#111", "#eee"];
  main.replaceChildren(el("div", { class: "page wide" },
    el("div", { class: "controls" }, backButton()),
    el("div", { class: "game" },
      el("div", { class: "game-left" },
        el("div", { class: "card" },
          el("div", { class: "side" }, el("span", { class: "disc", style: `background:${discs[0]}` }), playerName(players[0], game.first), firstCount),
          boardBox,
          el("div", { class: "side" }, el("span", { class: "disc", style: `background:${discs[1]}` }), playerName(players[1], game.second), secondCount),
          el("div", { class: "controls scrubber" }, toStart, prev, slider, next, toEnd)),
        canAnalyse || engines ? el("div", { class: "eval" }, el("i", { class: "swatch", style: `background:${discs[0]}` }), evalShares[0], el("div", { class: "eval-bar" }, evalFill), evalShares[1], el("i", { class: "swatch", style: `background:${discs[1]}` })) : null,
        canAnalyse ? timelineBox : null),
      el("div", { class: "game-right" },
        el("div", { class: "tiles" }, tile(resultText(game.result), "result"), game.opening < 0 ? null : tile(String(game.opening), "opening"), tile(game.code_version, "code"), Object.keys(game.searches).length ? tile(rootValue, rootLabel) : null),
        canAnalyse || engines ? panel : null,
        el("div", { class: "card" }, el("div", { class: "moves" }, ...buttons))))));

  function onKey(e) {
    if (e.target.closest?.("input, select, textarea")) return;
    const ix = current?.ix;
    if (path && ix && e.key === "ArrowLeft") go(path.length ? path.slice(0, -1) : null);
    else if (path && ix && e.key === "ArrowRight") {
      const g = nodeAt(ix, path);
      if (g >= 0 && ix.children[g].length) go([...path, ix.tree.action[ix.children[g][0]]]);
    } else if (path && e.key === "Escape") go(null);
    else if (e.key === "ArrowLeft") goTo(at - 1);
    else if (e.key === "ArrowRight") goTo(at + 1);
    else return;
    e.preventDefault();
  }
  document.addEventListener("keydown", onKey);
  startTimeline();
  goTo(at);
}
