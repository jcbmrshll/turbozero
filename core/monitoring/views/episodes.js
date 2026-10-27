// Episodes in the browser: the raw arrays a run logged, played frame by frame on a
// canvas by the env's view (renderers.js).

import { h } from "./dom.js";

// ---------------------------------------------------------------- arrays

// an n-dimensional view onto a flat typed array, row-major
export class NdArray {
  constructor(data, shape) {
    this.data = data;
    this.shape = shape;
    this.strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  }
  get length() {
    return this.shape[0];
  }
  get(...index) {
    let at = 0;
    for (let i = 0; i < index.length; i++) at += index[i] * this.strides[i];
    return this.data[at];
  }
  // the array at index i of the first axis
  at(i) {
    const n = this.strides[0];
    return new NdArray(this.data.subarray(i * n, (i + 1) * n), this.shape.slice(1));
  }
  // a 0-d array's value
  item() {
    return this.data[0];
  }
}

// an episode's arrays, {name: NdArray}, from host.arrays' {name: {dtype, shape, data}}
export function ndArrays(arrays) {
  return Object.fromEntries(Object.entries(arrays).map(([name, a]) => [name, new NdArray(a.data, a.shape)]));
}

// ---------------------------------------------------------------- player

const SPEEDS = [0.25, 0.5, 1, 2];
const MAX_HEIGHT = 420;

// keys go to the player under the pointer, without clicking it first, unless
// they're being typed somewhere
let hovered = null;
document.addEventListener("keydown", (ev) => {
  if (!hovered || ev.target.closest?.("input, select, textarea")) return;
  if (hovered.key(ev.key)) ev.preventDefault();
});

// A canvas playing a view (a renderer's, see renderers.js) with play/pause, a
// scrubber, speeds and full screen; space plays and pauses, the arrow keys step a
// frame and f goes full screen, while the pointer is over it. player.onFrame(frame)
// is called with every frame drawn. A view can count its frames its own way,
// counter(frame), offer its own speeds, and stop at its last frame rather than loop
// (loop: false). Anything in `below` goes under the canvas, full screen too. It
// starts playing unless `paused`, from frame `start` (where the game's players started
// playing, after an opening played for them), and plays again from there.
// player.onClick(event), if set, gets clicks on the canvas first, and returns true
// for those it took; the rest play and pause. player.canvas is the canvas, and
// player.redraw() draws the frame again, for a view that changed. `speed` is the speed it
// plays at, if the view offers it; player.onSpeed(speed) is told when it's changed, and
// player.onUser(frame, playing) when someone plays, pauses or moves it (not as it plays).
// player.destroy() stops it.
export function episodePlayer(view, { maxHeight = MAX_HEIGHT, below = null, paused = false, start = 0, speed = 1 } = {}) {
  const canvas = h("canvas");
  canvas.style.aspectRatio = `${view.width} / ${view.height}`;
  // no taller than maxHeight on screen
  canvas.style.maxWidth = `${(maxHeight * view.width) / view.height}px`;
  // and what's under it no wider
  if (below) below.style.maxWidth = canvas.style.maxWidth;
  const ctx = canvas.getContext("2d");
  const still = view.frames <= 1;
  const loops = view.loop !== false;
  const speeds = view.speeds ?? SPEEDS;
  const playBtn = h("button", { class: "play", title: "play/pause (space)" });
  const scrub = h("input", { type: "range", min: 0, max: Math.max(0, view.frames - 1), step: 1, value: 0 });
  const counter = h("span", { class: "pos" });
  const speedBtns = speeds.map((s) => h("button", { text: `${s}×`, onclick: () => (setSpeed(s), player.onSpeed?.(s)) }));
  const fullBtn = document.fullscreenEnabled && h("button", { class: "full", title: "full screen (f)", onclick: () => toggleFull() }, fullIcon());
  const controls = h("div", { class: "scrub player-controls" }, playBtn, scrub, counter, ...speedBtns, fullBtn);
  const el = h("div", { class: "player" }, canvas, below, !still && controls);
  // full screen, as big as fits (style.css .player:fullscreen)
  el.style.setProperty("--aspect", view.width / view.height);

  start = Math.min(Math.max(start, 0), view.frames - 1);
  const player = { el, canvas, frame: start, playing: !still && !paused, speed: 1, onFrame: null, onClick: null, onSpeed: null, onUser: null };
  let scale = 1;
  // the frame's time at the last play or seek, so playback is real time whatever
  // the display's refresh rate
  let anchor = { time: 0, frame: start };

  function draw() {
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    view.draw(ctx, player.frame);
    scrub.value = player.frame;
    counter.textContent = view.counter ? view.counter(player.frame) : `${player.frame + 1}/${view.frames}`;
    player.onFrame?.(player.frame);
  }

  player.setFrame = (frame) => {
    player.frame = ((Math.round(frame) % view.frames) + view.frames) % view.frames;
    anchor = { time: performance.now(), frame: player.frame };
    draw();
  };
  function setPlaying(on) {
    player.playing = on && !still;
    // played again from the end of an episode that doesn't loop: from the start
    if (player.playing && !loops && player.frame === view.frames - 1) player.setFrame(start);
    anchor = { time: performance.now(), frame: player.frame };
    playBtn.textContent = player.playing ? "❚❚" : "▶";
    if (player.playing) loop();
  }
  // someone played, paused or moved it
  const byUser = () => player.onUser?.(player.frame, player.playing);
  player.pause = () => setPlaying(false);
  player.play = () => setPlaying(true);
  player.redraw = () => draw();
  function setSpeed(s) {
    player.speed = s;
    anchor = { time: performance.now(), frame: player.frame };
    speedBtns.forEach((b, i) => b.classList.toggle("on", speeds[i] === s));
  }
  function toggleFull() {
    if (document.fullscreenElement === el) document.exitFullscreen();
    else el.requestFullscreen().catch(() => {});
  }

  // draws while playing and on screen; a card that's scrolled away or on a hidden
  // tab costs nothing
  let visible = false, running = false;
  function loop() {
    if (running || !player.playing || !visible) return;
    running = true;
    const tick = (now) => {
      if (!player.playing || !visible) {
        running = false;
        return;
      }
      // a frame's timestamp can be from just before the anchor was set
      const elapsed = Math.max(0, now - anchor.time) / 1000;
      let frame = anchor.frame + Math.floor(elapsed * view.fps * player.speed);
      if (!loops && frame >= view.frames - 1) {
        player.frame = view.frames - 1;
        draw();
        running = false;
        setPlaying(false);
        return;
      }
      frame %= view.frames;
      if (frame !== player.frame) {
        player.frame = frame;
        draw();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  player.destroy = () => {
    player.playing = false;
    seen.disconnect();
    sized.disconnect();
    if (hovered === player) hovered = null;
  };
  const seen = new IntersectionObserver(([entry]) => {
    // a player taken off the page (for another episode) is done with
    if (!canvas.isConnected) {
      player.destroy();
      return;
    }
    visible = entry.isIntersecting;
    // pick up where it was, not where the clock would have it
    anchor = { time: performance.now(), frame: player.frame };
    loop();
  });
  seen.observe(canvas);
  // drawn at the canvas's size on screen, so it's sharp at any width
  const sized = new ResizeObserver(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    if (width === canvas.width) return;
    canvas.width = width;
    canvas.height = Math.round((width * view.height) / view.width);
    scale = width / view.width;
    draw();
  });
  sized.observe(canvas);

  playBtn.addEventListener("click", () => {
    setPlaying(!player.playing);
    byUser();
  });
  scrub.addEventListener("input", () => {
    setPlaying(false);
    player.setFrame(Number(scrub.value));
    byUser();
  });
  canvas.addEventListener("click", (ev) => {
    if (player.onClick?.(ev)) return;
    setPlaying(!player.playing);
    byUser();
  });
  player.key = (key) => {
    if (still) return false;
    if (key === " ") setPlaying(!player.playing);
    else if (key === "ArrowRight" || key === "ArrowLeft") {
      setPlaying(false);
      const frame = player.frame + (key === "ArrowRight" ? 1 : -1);
      // one that doesn't loop stops at either end
      player.setFrame(loops ? frame : Math.min(Math.max(frame, 0), view.frames - 1));
    } else if (key === "f" && fullBtn) {
      toggleFull();
      return true;
    } else return false;
    byUser();
    return true;
  };
  el.addEventListener("pointerenter", () => (hovered = player));
  el.addEventListener("pointerleave", () => hovered === player && (hovered = null));

  setSpeed(speeds.includes(speed) ? speed : speeds.includes(1) ? 1 : speeds[0]);
  setPlaying(player.playing);
  return player;
}

// four corners pointing out
function fullIcon() {
  return h("svg", { viewBox: "0 0 10 10", width: 10, height: 10 },
    h("path", { d: "M0.5 3.5v-3h3M6.5 0.5h3v3M9.5 6.5v3h-3M3.5 9.5h-3v-3", fill: "none", stroke: "currentColor", "stroke-width": 1.2 }));
}
