// One of the game's engines compiled to WebAssembly, off the page's thread: which, and the build,
// are in this script's URL. A search holds this thread until it ends, so nothing can reach it; to
// stop one, the page terminates the worker. Its files load when the first search is asked for.

import board from "/game.js";

const params = new URL(self.location.href).searchParams;
const BUILD = `/wasm-engines/${params.get("build")}`;
const spec = board.engines[params.get("engine")];

let engine = null;
// what's being searched: says each step to the page
let reporting = null;

async function load() {
  const init = (await import(`${BUILD}/${spec.module}`)).default;
  const files = await Promise.all(Object.values(spec.files).map(async (name) => {
    const response = await fetch(`${BUILD}/${name}`);
    if (!response.ok) throw new Error(`${name}: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  }));
  engine = await init({
    onResult: (step) => reporting?.(step),
    printErr: (line) => { const step = spec.logged?.(line); if (step) reporting?.(step); },
    print: () => {},
  });
  Object.keys(spec.files).forEach((path, i) => {
    engine.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    engine.FS.writeFile(path, files[i]);
  });
  engine._engine_init();
  self.postMessage({ type: "ready" });
}

function search({ id, position, level }) {
  const { player, opponent, passed } = board.engineProblem(position);
  const start = performance.now();
  let last = null;
  reporting = (step) => {
    // an engine's first report comes before it's searched anything
    if (step.depth === 0) return;
    last = passed ? { ...step, move: 64, score: -step.score } : step;
    self.postMessage({ type: "progress", id, ms: performance.now() - start, ...last });
  };
  engine._engine_search(player, opponent, level);
  self.postMessage({ type: "result", id, ms: performance.now() - start, ...last });
}

// requests wait for the engine to load, in the order they came
let queue = load();
self.onmessage = ({ data }) => {
  queue = queue.then(() => search(data)).catch((e) => {
    self.postMessage({ type: "error", id: data.id, message: e instanceof Error ? e.message : String(e) });
  });
};
