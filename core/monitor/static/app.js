// turbozero monitor: run list, live charts, test game viewer.
// No framework; every view is plain DOM built with h(), and labels that come
// from a run (metric names, config values) only ever go in as text.

const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map((v) => `var(${v})`);
const POLL_MS = 2000;
const RUNS_POLL_MS = 5000;
// a running run that hasn't logged for this long is probably dead
const STALE_S = 180;
// the stat tiles, in order, when the run logs them; each tester's outcome follows
const HEADLINE = ["loss", "policy_loss", "value_loss"];
// testers log "<tester name>_avg_outcome"; they share one chart
const OUTCOME = /^(.+)_avg_outcome$/;
// charts that lead the grid; the rest follow in the order they were first logged
const CHART_ORDER = ["avg_outcome", "loss", "policy_loss", "value_loss"];
// the windows offered over the metric charts, in epochs
const CHART_WINDOWS = [50, 200, 1000];

const state = {
  runs: [],
  runId: null,
  run: null,
  rows: [],
  metricsNext: 0,
  media: [],
  mediaNext: 0,
  // media key -> index into that key's entries, or null to follow the latest
  mediaSel: {},
  connected: true,
  // the metric charts show the last this-many epochs, or all of them if null;
  // kept across runs and reloads
  chartWindow: Number(localStorage.getItem("chartWindow")) || null,
};

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const svg = ["svg", "path", "line", "circle", "text", "g", "rect"].includes(tag);
  const el = svg
    ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function fmt(v) {
  if (v == null || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e4) return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);
  if (Number.isInteger(v)) return v.toLocaleString("en");
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toPrecision(3);
}

// axis ticks are already round numbers; just drop float noise and trailing zeros
function fmtTick(v) {
  return Math.abs(v) >= 1e4 ? fmt(v) : String(+v.toPrecision(6));
}

function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function duration(s) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

function started(t) {
  return new Date(t * 1000).toLocaleString("en", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

// "41 / 300 epochs" (epochs done) when the run said how long it will be
function progress(run) {
  if (run.step == null) return "";
  const total = run.num_epochs;
  // a continued run can go past the epochs it first asked for
  return total && run.step < total ? `${run.step + 1} / ${total} epochs` : `epoch ${run.step}`;
}

function status(run) {
  if (run.status === "running" && Date.now() / 1000 - run.updated > STALE_S) return "stale";
  return run.status;
}

const tip = document.getElementById("tip");
function showTip(x, y, ...children) {
  tip.replaceChildren(...children);
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  const left = x + 14 + r.width > innerWidth ? x - 14 - r.width : x + 14;
  const top = Math.min(innerHeight - r.height - 4, Math.max(4, y - r.height / 2));
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}
function hideTip() {
  tip.hidden = true;
}

// ---------------------------------------------------------------- routing

function route() {
  const m = location.pathname.match(/^\/run\/([\w-]+)/);
  return m ? m[1] : null;
}

function go(runId) {
  history.pushState(null, "", runId ? `/run/${runId}` : "/");
  select(runId);
}

window.addEventListener("popstate", () => select(route()));

// ---------------------------------------------------------------- sidebar

function renderSide() {
  const side = document.getElementById("side");
  const byProject = new Map();
  for (const run of state.runs) {
    if (!byProject.has(run.project)) byProject.set(run.project, []);
    byProject.get(run.project).push(run);
  }
  const children = [];
  for (const [project, runs] of byProject) {
    children.push(h("div", { class: "side-hdr", text: project }));
    for (const run of runs) {
      children.push(
        h(
          "a",
          {
            class: `run-row${run.id === state.runId ? " active" : ""}`,
            href: `/run/${run.id}`,
            onclick: (e) => {
              if (e.metaKey || e.ctrlKey) return;
              e.preventDefault();
              go(run.id);
            },
          },
          h("span", { class: `dot ${status(run)}`, title: status(run) }),
          h("span", { class: "name", text: run.name }),
          h("span", { class: "step", text: progress(run) }),
          h("span", { class: "when", text: `${started(run.created)} · ${ago(run.updated)}` }),
        ),
      );
    }
  }
  if (!children.length) children.push(h("div", { class: "side-hdr", text: "no runs" }));
  side.replaceChildren(...children);
}

async function pollRuns() {
  try {
    state.runs = await api("/api/runs");
    setConnected(true);
  } catch {
    setConnected(false);
  }
  renderSide();
  if (!state.runId && !route()) renderEmpty();
}

function setConnected(ok) {
  state.connected = ok;
  const el = document.getElementById("conn");
  el.textContent = ok ? "" : "server unreachable";
  el.classList.toggle("down", !ok);
}

// ---------------------------------------------------------------- run view

const main = document.getElementById("main");
let view = null; // the containers of the current run view

function renderEmpty() {
  if (state.runs.length) {
    // nothing selected: open the newest run
    go(state.runs[0].id);
    return;
  }
  view = null;
  main.replaceChildren(
    h(
      "div",
      { class: "empty" },
      h("h1", { text: "No runs yet" }),
      h("p", { text: "Start a training run with the monitor attached:" }),
      h("p", {}, h("code", { text: "uv run examples/othello.py --monitor" })),
      h("p", { text: "It will show up here as soon as it starts training." }),
    ),
  );
}

async function select(runId) {
  if (!runId) {
    state.runId = null;
    renderSide();
    renderEmpty();
    return;
  }
  Object.assign(state, {
    runId,
    run: null,
    rows: [],
    metricsNext: 0,
    media: [],
    mediaNext: 0,
    mediaSel: {},
  });
  renderSide();
  view = {
    head: h("div", { class: "runhead" }),
    tiles: h("div", { class: "tiles" }),
    mediaTitle: h("div", { class: "stitle", text: "Test games" }),
    media: h("div", { class: "media-grid" }),
    chartsTitle: h("div", { class: "stitle-row" }),
    charts: h("div", { class: "chart-grid" }),
    configTitle: h("div", { class: "stitle", text: "Config" }),
    config: h("div", { class: "config-grid" }),
    mediaCards: {},
  };
  main.replaceChildren(
    h(
      "div",
      { class: "page" },
      view.head,
      view.tiles,
      view.mediaTitle,
      view.media,
      view.chartsTitle,
      view.charts,
      view.configTitle,
      view.config,
    ),
  );
  renderChartsTitle();
  main.scrollTop = 0;
  await pollRun();
}

let polling = false;
async function pollRun() {
  const runId = state.runId;
  if (!runId || polling) return;
  polling = true;
  try {
    const [run, metrics, media] = await Promise.all([
      api(`/api/runs/${runId}`),
      api(`/api/runs/${runId}/metrics?since=${state.metricsNext}`),
      api(`/api/runs/${runId}/media?since=${state.mediaNext}`),
    ]);
    if (runId !== state.runId) return;
    setConnected(true);
    const firstLoad = !state.run;
    state.run = run;
    state.metricsNext = metrics.next;
    state.mediaNext = media.next;
    state.rows.push(...metrics.rows);
    state.media.push(...media.rows);

    renderHead();
    if (firstLoad) renderConfig();
    if (firstLoad || metrics.rows.length) {
      renderTiles();
      renderCharts();
    }
    if (firstLoad || media.rows.length) renderMedia();
  } catch (e) {
    if (String(e.message).startsWith("404")) {
      main.replaceChildren(h("div", { class: "empty" }, h("h1", { text: "No such run" })));
      state.runId = null;
    } else {
      setConnected(false);
    }
  } finally {
    polling = false;
  }
}

function renderHead() {
  const run = state.run;
  const st = status(run);
  const end = run.status === "running" ? Date.now() / 1000 : run.updated;
  const sep = () => h("span", { class: "sep", text: "·" });
  view.head.replaceChildren(
    h("div", { class: "crumbs" }, run.project, h("span", { class: "id", text: run.id })),
    h("h1", { text: run.name }),
    h(
      "div",
      { class: "meta" },
      h("span", { class: `dot ${st}` }),
      h("span", { text: st }),
      sep(),
      run.seed != null && h("span", { text: `seed ${run.seed}` }),
      run.seed != null && sep(),
      h("span", { text: `started ${started(run.created)}` }),
      sep(),
      h("span", { text: duration(end - run.created) }),
      run.step != null && sep(),
      run.step != null && h("span", { text: progress(run) }),
    ),
  );
}

function latest(key) {
  for (let i = state.rows.length - 1; i >= 0; i--) {
    const v = state.rows[i][key];
    if (v != null) return v;
  }
  return null;
}

function renderTiles() {
  const outcomes = metricKeys().filter((k) => OUTCOME.test(k));
  const tiles = [...HEADLINE, ...outcomes].filter((k) => latest(k) != null).map((k) =>
    h(
      "div",
      { class: "tile" },
      h("div", { class: "lab", text: k.replaceAll("_", " ") }),
      h("div", { class: "val", text: fmt(latest(k)) }),
    ),
  );
  view.tiles.replaceChildren(...tiles);
  view.tiles.hidden = !tiles.length;
}

// ---------------------------------------------------------------- charts

// Group metric keys into charts: "a/b" keys chart together under "a", every
// tester's "<name>_avg_outcome" charts together under "avg_outcome", and
// max_/mean_/min_ variants of one quantity chart together under its name.
function groupMetrics(keys) {
  const groups = new Map();
  const add = (group, series, key) => {
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ name: series, key });
  };
  const set = new Set(keys);
  for (const key of keys) {
    const slash = key.lastIndexOf("/");
    const stat = key.match(/^(max|mean|min)_(.+)$/);
    const outcome = key.match(OUTCOME);
    if (slash > 0) {
      add(key.slice(0, slash), key.slice(slash + 1), key);
    } else if (outcome) {
      add("avg_outcome", outcome[1], key);
    } else if (stat && ["max", "mean", "min"].filter((s) => set.has(`${s}_${stat[2]}`)).length > 1) {
      add(stat[2], stat[1], key);
    } else {
      add(key, key, key);
    }
  }
  for (const series of groups.values()) {
    const rank = { max: 0, mean: 1, min: 2 };
    series.sort((a, b) => (rank[a.name] ?? 3) - (rank[b.name] ?? 3));
  }
  const order = [...groups.keys()].sort((a, b) => {
    const ra = CHART_ORDER.indexOf(a), rb = CHART_ORDER.indexOf(b);
    return (ra < 0 ? 99 : ra) - (rb < 0 ? 99 : rb);
  });
  return order.map((name) => ({ name, series: groups.get(name) }));
}

function metricKeys() {
  const keys = [];
  const seen = new Set(["step", "time"]);
  for (const row of state.rows) {
    for (const k of Object.keys(row)) {
      if (!seen.has(k)) {
        seen.add(k);
        keys.push(k);
      }
    }
  }
  return keys;
}

function renderCharts() {
  const groups = groupMetrics(metricKeys());
  view.chartsTitle.hidden = !groups.length;
  // the window ends at the newest epoch logged
  const last = state.rows.at(-1)?.step;
  const from = state.chartWindow && last != null ? last - state.chartWindow + 1 : -Infinity;
  view.charts.replaceChildren(...groups.map((g) => chartCard(g, from)));
}

function setChartWindow(n) {
  state.chartWindow = n > 0 ? Math.floor(n) : null;
  if (state.chartWindow) localStorage.setItem("chartWindow", state.chartWindow);
  else localStorage.removeItem("chartWindow");
  renderChartsTitle();
  if (state.rows.length) renderCharts();
}

// "Metrics", and the window of epochs the charts show: all, a preset, or any N
function renderChartsTitle() {
  const win = state.chartWindow;
  const toggle = (n, text, title) =>
    h("button", { class: win === n ? "on" : null, text, title, onclick: () => setChartWindow(n) });
  const custom = h("input", {
    type: "number",
    min: 1,
    step: 1,
    placeholder: "N",
    title: "show the last N epochs",
    value: win && !CHART_WINDOWS.includes(win) ? win : null,
    class: win && !CHART_WINDOWS.includes(win) ? "on" : null,
    onchange: (e) => setChartWindow(Number(e.target.value)),
    onkeydown: (e) => e.key === "Enter" && e.target.blur(),
  });
  view.chartsTitle.replaceChildren(
    h("div", { class: "stitle", text: "Metrics" }),
    h("span", { class: "spacer" }),
    h(
      "span",
      { class: "scrub" },
      h("span", { class: "dim", text: "last" }),
      toggle(null, "all", "every epoch"),
      CHART_WINDOWS.map((n) => toggle(n, fmt(n), `the last ${n} epochs`)),
      custom,
      h("span", { class: "dim", text: "epochs" }),
    ),
  );
}

function chartCard(group, from) {
  const series = group.series.map((s, i) => ({
    ...s,
    color: SERIES[i % SERIES.length],
    points: state.rows.filter((r) => r[s.key] != null && r.step >= from).map((r) => [r.step, r[s.key]]),
  }));
  const single = series.length === 1;
  const card = h(
    "div",
    { class: "card chart-card" },
    h(
      "div",
      { class: "card-hdr" },
      h("span", { class: "title", text: group.name.replaceAll("_", " ") }),
      h("span", { class: "spacer" }),
      single && h("span", { class: "latest", text: fmt(latest(series[0].key)) }),
    ),
    !single &&
      h(
        "div",
        { class: "legend" },
        series.map((s) =>
          h("span", { class: "key" }, h("i", { style: `background:${s.color}` }), s.name),
        ),
      ),
  );
  const holder = h("div");
  card.append(holder);
  // draw once the card is laid out, so the chart knows its width
  requestAnimationFrame(() => holder.replaceChildren(lineChart(series, holder.clientWidth)));
  return card;
}

function niceTicks(lo, hi, count) {
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 1;
    lo -= pad;
    hi += pad;
  }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const start = Math.floor(lo / step) * step;
  // the last tick must reach hi, or the top of the data runs off the chart
  const ticks = [+start.toPrecision(12)];
  while (ticks[ticks.length - 1] < hi) ticks.push(+(start + ticks.length * step).toPrecision(12));
  return ticks;
}

function lineChart(series, width) {
  const W = Math.max(width, 200), H = 150;
  const all = series.flatMap((s) => s.points);
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  if (!all.length) return svg;

  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).filter(Number.isFinite);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yt = niceTicks(Math.min(...ys), Math.max(...ys), 3);
  // room for the widest y label (9px mono is ~5.5px a character)
  const labelW = Math.max(...yt.map((t) => fmtTick(t).length)) * 5.5;
  const pad = { l: Math.max(28, labelW + 12), r: 10, t: 10, b: 18 };
  const y0 = yt[0], y1 = yt[yt.length - 1];
  const sx = (x) => pad.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * (H - pad.t - pad.b);

  for (const t of yt) {
    svg.append(
      h("line", { class: "grid", x1: pad.l, x2: W - pad.r, y1: sy(t), y2: sy(t) }),
      h("text", { class: "tick", x: pad.l - 6, y: sy(t) + 3, "text-anchor": "end", text: fmtTick(t) }),
    );
  }
  svg.append(h("line", { class: "axis", x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b }));
  const xt = niceTicks(x0, x1, Math.max(2, Math.floor((W - pad.l - pad.r) / 80)))
    .filter((t) => t >= x0 && t <= x1 && Number.isInteger(t));
  for (const t of xt) {
    svg.append(h("text", { class: "tick", x: sx(t), y: H - 5, "text-anchor": "middle", text: fmtTick(t) }));
  }

  for (const s of series) {
    // break the line wherever a value is missing (e.g. a diverged loss, logged as null)
    let d = "", prev = null;
    for (const [x, y] of s.points) {
      if (!Number.isFinite(y)) { prev = null; continue; }
      d += `${prev === null ? "M" : "L"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
      prev = x;
    }
    svg.append(h("path", { class: "line", d, stroke: s.color }));
  }

  // hover: a crosshair snapped to the nearest logged step, every series in one tooltip
  const steps = [...new Set(xs)].sort((a, b) => a - b);
  const lookup = series.map((s) => new Map(s.points));
  const cross = h("line", { class: "cross", y1: pad.t, y2: H - pad.b, visibility: "hidden" });
  const dots = series.map((s) => h("circle", { class: "hover-dot", r: 4, fill: s.color, visibility: "hidden" }));
  svg.append(cross, ...dots);
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const target = x0 + ((px - pad.l) / (W - pad.l - pad.r)) * (x1 - x0);
    let best = steps[0];
    for (const st of steps) if (Math.abs(st - target) < Math.abs(best - target)) best = st;
    cross.setAttribute("x1", sx(best));
    cross.setAttribute("x2", sx(best));
    cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((s, i) => {
      const v = lookup[i].get(best);
      const ok = v != null && Number.isFinite(v);
      dots[i].setAttribute("visibility", ok ? "visible" : "hidden");
      if (!ok) return;
      dots[i].setAttribute("cx", sx(best));
      dots[i].setAttribute("cy", sy(v));
      rows.push([v, s]);
    });
    rows.sort((a, b) => b[0] - a[0]);
    showTip(
      e.clientX,
      e.clientY,
      h("div", { class: "head", text: `epoch ${best}` }),
      ...rows.map(([v, s]) =>
        h(
          "div",
          { class: "row" },
          h("i", { style: `background:${s.color}` }),
          h("b", { text: fmt(v) }),
          series.length > 1 && h("span", { text: s.name }),
        ),
      ),
    );
  });
  svg.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    hideTip();
  });
  return svg;
}

// redraw charts at the new width when the window is resized
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!view) return;
    // leave media alone so gifs don't restart
    if (state.rows.length) renderCharts();
  }, 150);
});

// ---------------------------------------------------------------- media

function mediaByKey() {
  const byKey = new Map();
  for (const m of state.media) {
    if (!byKey.has(m.key)) byKey.set(m.key, []);
    byKey.get(m.key).push(m);
  }
  for (const list of byKey.values()) list.sort((a, b) => a.step - b.step);
  return [...byKey];
}

function renderMedia() {
  const groups = mediaByKey();
  view.mediaTitle.hidden = !groups.length;
  for (const [key, entries] of groups) {
    let card = view.mediaCards[key];
    if (!card) {
      card = {
        hdr: h("div", { class: "card-hdr" }),
        body: h("div", { class: "media-body" }),
        shown: null,
      };
      card.el = h("div", { class: "card" }, card.hdr, card.body);
      view.mediaCards[key] = card;
      view.media.append(card.el);
    }
    updateMediaCard(key, entries, card);
  }
}

function updateMediaCard(key, entries, card) {
  const sel = state.mediaSel[key];
  const idx = sel == null ? entries.length - 1 : sel;
  const entry = entries[idx];
  const set = (i) => {
    state.mediaSel[key] = i >= entries.length - 1 ? null : Math.max(0, i);
    updateMediaCard(key, entries, card);
  };
  card.hdr.replaceChildren(
    h("span", { class: "title", text: key.replaceAll("_", " ") }),
    h("span", { class: "spacer" }),
    h(
      "span",
      { class: "scrub" },
      h("button", { onclick: () => set(idx - 1), disabled: idx === 0, title: "previous", text: "‹" }),
      h("span", { class: "pos", text: `epoch ${entry.step}` }),
      h("button", { onclick: () => set(idx + 1), disabled: idx === entries.length - 1, title: "next", text: "›" }),
      h("button", {
        class: sel == null ? "on" : null,
        onclick: () => set(entries.length - 1),
        title: "follow the newest",
        text: "latest",
      }),
    ),
  );
  // swapping the element restarts a gif, so only do it when the entry changes
  const id = entry.file || `error-${entry.step}`;
  if (card.shown === id) return;
  card.shown = id;
  if (entry.error) {
    card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't render: ${entry.error}` }));
    return;
  }
  const src = `/media/${state.runId}/${encodeURIComponent(entry.file)}`;
  if (entry.file.endsWith(".json")) {
    card.body.replaceChildren();
    fetch(src)
      .then((r) => r.json())
      .then((data) => {
        if (card.shown !== entry.file) return;
        card.body.replaceChildren(h("pre", { text: JSON.stringify(data, null, 2) }));
      });
  } else if (entry.file.endsWith(".mp4")) {
    card.body.replaceChildren(h("video", { src, autoplay: true, loop: true, muted: true, controls: true }));
  } else {
    card.body.replaceChildren(h("img", { src, alt: `${key} at epoch ${entry.step}` }));
  }
}

// ---------------------------------------------------------------- config

function configValue(v) {
  if (v == null) return "–";
  if (Array.isArray(v) && v.every((x) => typeof x !== "object")) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v).replace(/<function (\S+) at 0x[0-9a-f]+>/g, "$1");
}

function renderConfig() {
  const config = state.run.config || {};
  const sections = Object.entries(config).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v));
  const flat = Object.entries(config).filter(([, v]) => !(v && typeof v === "object" && !Array.isArray(v)));
  if (flat.length) sections.unshift(["config", Object.fromEntries(flat)]);
  view.configTitle.hidden = !sections.length;
  view.config.replaceChildren(
    ...sections.map(([name, values]) =>
      h(
        "div",
        { class: "card" },
        h("div", { class: "card-hdr" }, h("span", { class: "title", text: name.replaceAll("_", " ") })),
        h(
          "table",
          {},
          Object.entries(values).map(([k, v]) =>
            h("tr", {}, h("td", { text: k }), h("td", { text: configValue(v) })),
          ),
        ),
      ),
    ),
  );
}

// ---------------------------------------------------------------- start

async function start() {
  await pollRuns();
  const runId = route();
  if (runId) select(runId);
  else renderEmpty();
  setInterval(pollRuns, RUNS_POLL_MS);
  setInterval(() => {
    if (document.visibilityState === "visible") pollRun();
  }, POLL_MS);
}

start();
