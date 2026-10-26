// The renderer for "turbozero/<env>-episode" entries (layout.json): a test's game,
// played on a canvas by its env's view (renderers.js), with each player's value of the
// position under it, or, when the game recorded its searches, the search explorer
// (search.js) beside it.
//
// The page's location keeps where the reader is in the game, under the entry's key, as
// "<entry's step>.<frame>", then ".<move>-<move>..." for a position explored from the
// frame's, so that a reload or a shared link opens it again. A game playing keeps none.
// The speed it plays at is kept as a preference.

import { h } from "./dom.js";
import { RENDERERS } from "./renderers.js";
import { episodePlayer, ndArrays } from "./episodes.js";
import { advantage, hasSearches, searchExplorer, sideMark } from "./search.js";

// how tall an episode's board can be when its searches are explored on it
const EXPLORE_HEIGHT = 520;
const TYPE = /^turbozero\/(.+)-episode$/;

// the bundle's stylesheet, once per frame
function styles() {
  if (document.getElementById("tz-style")) return;
  document.head.append(h("link", { id: "tz-style", rel: "stylesheet", href: new URL("./style.css", import.meta.url).href }));
}

// where a location's state says the reader is in the entry's game: {frame, path}, or null
function parsePlace(text, entry) {
  const m = text?.match(/^(-?\d+)\.(\d+)(?:\.(\d+(?:-\d+)*))?$/);
  if (!m || Number(m[1]) !== entry.step) return null;
  return { frame: Number(m[2]), path: m[3] ? m[3].split("-").map(Number) : [] };
}

const placeText = (entry, frame, path) => `${entry.step}.${frame}${path.length ? `.${path.join("-")}` : ""}`;

export async function render({ el, data, host }) {
  styles();
  let shown = null;

  async function show(entry) {
    shown?.destroy();
    shown = null;
    const env = entry.type?.match(TYPE)?.[1];
    const make = RENDERERS[env];
    if (!make) throw new Error(`no view for ${env ?? entry.type} episodes`);
    const arrays = ndArrays(await host.arrays(entry.file));
    const view = make(arrays);
    shown = mount(el, entry, arrays, view, host, await host.getPref("speed"));
    await shown.locate(await host.location());
  }

  await show(data);
  return {
    async update(change) {
      if (change.data) await show(change.data);
      else if (change.location) await shown?.locate(change.location);
      // a new theme: the canvas reads its colours as it draws
      if (change.theme) shown?.redraw();
    },
    destroy() {
      shown?.destroy();
    },
  };
}

// a game drawn into el: {locate(location), redraw(), destroy()}
function mount(el, entry, arrays, view, host, speed) {
  const opponent = entry.attrs?.opponent;
  const caption = view.players && arrays.values ? gameCaption(view, arrays, opponent) : null;
  // the searches behind the game's moves, explored on its board, with the moves they
  // explored beside it; using the explorer pauses the game
  const explorer = view.players && view.drawBoard && hasSearches(arrays) ? searchExplorer(view, arrays, { who: seat(view, arrays, opponent) }) : null;
  // the explorer's rows for the players stand in for the caption
  const below = explorer ? explorer.evals : caption?.el;
  // shown from where the players started playing, after any opening played for them; a
  // game whose searches can be explored waits there to be explored
  const start = arrays.start?.item() ?? 0;
  const player = episodePlayer(explorer?.view ?? view, {
    below,
    maxHeight: explorer ? EXPLORE_HEIGHT : undefined,
    paused: !!explorer,
    start,
    speed: speed ?? 1,
  });
  explorer?.attach(player);
  if (caption && !explorer) player.onFrame = caption.show;
  el.replaceChildren(h("div", { class: `episode${explorer ? " exploring" : ""}` }, player.el, explorer?.side));
  // drawn once the canvas has a size; the caption needn't wait
  if (!explorer) caption?.show(player.frame);

  // where the reader is, as the location has it: what was last written or followed
  const key = entry.key;
  let place = null;
  const write = (text) => {
    if (text === place) return;
    place = text;
    host.navigate({ state: { [key]: text } });
  };
  player.onSpeed = (s) => host.setPref("speed", s);
  player.onUser = (frame, playing) => write(playing ? null : placeText(entry, frame, explorer?.path() ?? []));
  if (explorer) explorer.onUser = (path) => write(placeText(entry, player.frame, path));

  return {
    async locate(location) {
      const text = location.state?.[key] ?? null;
      if (text === place) return;
      const before = place;
      place = text;
      const at = parsePlace(text, entry);
      if (!at) {
        // back to where the game was before anyone moved it
        if (!before) return;
        player.setFrame(start);
        if (explorer) explorer.go([]);
        else player.play();
        return;
      }
      player.pause();
      player.setFrame(Math.min(at.frame, view.frames - 1));
      explorer?.go(at.path);
    },
    redraw: () => player.redraw(),
    destroy: () => player.destroy(),
  };
}

// the players of a two-player game, 0 the trained agent and 1 its opponent (by its
// name, if the run gave one), as searchExplorer names them
function seat(view, arrays, opponent) {
  const agent = arrays.agent.item();
  return (p) => {
    const colorIndex = p === 0 ? agent : 1 - agent;
    const side = view.players[colorIndex];
    return { ...side, role: p === 0 ? "agent" : opponent || "opponent", colorIndex };
  };
}

// who played which colour in a two-player game, each one's value estimate at the
// frame shown, its pieces on the board if the view counts them, and the result at
// the end
function gameCaption(view, arrays, opponent) {
  const agent = arrays.agent.item();
  const last = view.frames - 1;
  const outcome = arrays.completed.item() ? arrays.outcome.item() : null;
  const result = (sign) => (outcome == null ? "" : outcome * sign > 0 ? "won" : outcome * sign < 0 ? "lost" : "drew");
  const rows = [
    { who: "agent", color: agent, values: 0, sign: 1 },
    { who: opponent || "opponent", color: 1 - agent, values: 1, sign: -1 },
  ].map((p) => {
    const side = view.players[p.color];
    const swatch = sideMark(side);
    const value = h("b", { class: "num" });
    const pieces = view.pieces && h("span", { class: "num dim" });
    const tag = h("span", { class: "result" });
    const el = h("div", { class: "seat" }, swatch, h("span", { class: "who", text: `${p.who} · ${side.name}` }), h("span", { class: "spacer" }), pieces, h("span", { class: "dim", text: "value" }), value, tag);
    return { ...p, el, value, pieces, tag };
  });
  const toMove = arrays.to_move;
  return {
    el: h("div", { class: "seats" }, rows.map((r) => r.el)),
    show(t) {
      for (const r of rows) {
        const v = arrays.values.get(t, r.values);
        // each one's estimate is from its own side; shown for the first player
        r.value.replaceChildren(advantage(r.color === 0 ? v : -v, view.players));
        if (r.pieces) r.pieces.textContent = `${view.pieces(t, r.color)} discs`;
        const over = t === last && outcome != null;
        r.tag.textContent = over ? result(r.sign) : "";
        r.tag.className = `result ${over ? result(r.sign) : ""}`;
        r.el.classList.toggle("to-move", !over && toMove?.get(t) === r.color);
      }
    },
  };
}
