// Searching with an engine compiled to WebAssembly (`wasm_worker.js`), one search at a time. A worker
// is kept while it's idle, for the next position; stopping a search costs its worker, and the next
// search loading the engine again.

export class BrowserEngine {
  constructor(build, onMessage) {
    this.build = build;
    this.onMessage = onMessage;
    this.worker = null;
    this.engine = null;
    this.running = null;
    this.nextId = 1;
  }

  // ends the search running, if any
  stop() {
    if (this.running !== null) this.close();
  }

  close() {
    this.worker?.terminate();
    this.worker = null;
    this.running = null;
  }

  // Returns the search's id, which each of its messages carries.
  search(engine, position, level) {
    if (this.running !== null || engine !== this.engine) this.close();
    if (!this.worker) {
      this.engine = engine;
      const url = `/web/wasm_worker.js?build=${encodeURIComponent(this.build)}&engine=${encodeURIComponent(engine)}`;
      this.worker = new Worker(url, { type: "module" });
      this.worker.onmessage = ({ data }) => {
        if (data.type === "ready") return;
        if (data.id !== this.running) return;
        if (data.type !== "progress") this.running = null;
        this.onMessage(data);
      };
      this.worker.onerror = (e) => {
        const id = this.running;
        this.close();
        this.onMessage({ type: "error", id, message: e.message });
      };
    }
    this.running = this.nextId++;
    this.worker.postMessage({ id: this.running, position, level });
    return this.running;
  }
}
