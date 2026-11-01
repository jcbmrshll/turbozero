// Networks in onnxruntime-web, off the page's thread, and the search over them. On plain HTTP, as
// the tailnet's address is, browsers allow neither threads nor WebGPU, so it's one wasm thread.

import board from "/game.js";
import * as ort from "/ort/ort.wasm.min.mjs";
import { Search, snapshot } from "./search.js";

ort.env.wasm.wasmPaths = { wasm: "/ort/ort-wasm-simd-threaded.wasm", mjs: "/ort/ort-wasm-simd-threaded.mjs" };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

// the networks loaded, the latest last: few, as each holds its weights
const MAX_SESSIONS = 3;
const sessions = new Map();
// how often (ms) a search's progress sends the tree
const TREE_EVERY = 250;
// bumped by a stop, which takes effect at once on the search running then
let stopped = 0;
// requests run one at a time, in the order they came
let queue = Promise.resolve();

async function session(net) {
  if (!sessions.has(net)) {
    sessions.set(net, ort.InferenceSession.create(`/onnx/${net}`, { executionProviders: ["wasm"] }));
    if (sessions.size > MAX_SESSIONS) {
      const [oldest] = sessions.keys();
      const old = await sessions.get(oldest);
      sessions.delete(oldest);
      await old.release();
    }
  }
  return sessions.get(net);
}

async function evaluateMany(model, positions) {
  const size = board.inputShape.reduce((a, b) => a * b);
  const input = new Float32Array(positions.length * size);
  positions.forEach((position, i) => board.observe(position, input, i * size));
  const tensor = new ort.Tensor("float32", input, [positions.length, ...board.inputShape]);
  const out = await model.run({ input: tensor });
  tensor.dispose();
  const logits = await out.policy.getData();
  const values = await out.value.getData();
  out.policy.dispose();
  out.value.dispose();
  return { logits, values };
}

const yieldToEvents = () => new Promise((resolve) => {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => resolve();
  channel.port2.postMessage(null);
});

function summaryOf(search, root) {
  return { n: root.n, q: root.q, value: search.value };
}

async function handle(request) {
  if (request.type === "values") {
    // what a game's positions are worth to their players' networks, for its timeline
    const model = await session(request.net);
    for (let at = 0; at < request.positions.length; at += 8) {
      const { values } = await evaluateMany(model, request.positions.slice(at, at + 8));
      self.postMessage({ type: "values", id: request.id, indices: request.indices.slice(at, at + 8), values: [...values] });
      await yieldToEvents();
    }
  } else if (request.type === "search") {
    if (request.after !== stopped) return;
    const model = await session(request.net);
    let evals = 0, evalMs = 0;
    const search = new Search(board, async (position) => {
      const start = performance.now();
      const { logits, values } = await evaluateMany(model, [position]);
      evals += 1;
      evalMs += performance.now() - start;
      return { logits, value: values[0] };
    });
    let treeAt = 0;
    const root = await search.run(request.position, request.iterations, async (done) => {
      const now = performance.now();
      const sendTree = now - treeAt >= TREE_EVERY;
      if (sendTree) treeAt = now;
      self.postMessage({ type: "progress", id: request.id, done, of: request.iterations, ...summaryOf(search, search.root), tree: sendTree ? snapshot(search.root) : undefined, msPerEval: evalMs / evals });
      await yieldToEvents();
      return request.after === stopped;
    });
    if (request.after === stopped) {
      self.postMessage({ type: "result", id: request.id, ...summaryOf(search, root), tree: snapshot(root), msPerEval: evalMs / evals });
    }
  }
}

self.onmessage = (event) => {
  const request = event.data;
  if (request.type === "stop") { stopped = request.count; return; }
  queue = queue.then(() => handle(request)).catch((e) => {
    self.postMessage({ type: "error", id: request.id, message: e instanceof Error ? e.message : String(e) });
  });
};
self.postMessage({ type: "ready" });
