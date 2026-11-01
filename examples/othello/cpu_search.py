"""A single game's MCTS for playing on CPU: the tree and Othello's rules in C++ (`cpu_search.cpp`,
compiled on first use), the network in ONNX Runtime (see `export_onnx.py`).

A search visits the tree a batch of traversals at a time, each steered away from those still in
flight by virtual loss, and evaluates their leaves together: a batch of 1 is the sequential search
of `core/evaluators/mcts`. Only for play: self-play and training use `core`'s batched search.
"""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parent / "cpu_search.cpp"
CACHE_DIR = Path(
    os.environ.get("TURBOZERO_CACHE", Path.home() / ".cache" / "turbozero" / "othello")
)
PASS = 64

VL_LOSS, VL_VISIT, VL_DRAW = 0, 1, 2
VIRTUAL_LOSS = {"loss": VL_LOSS, "visit": VL_VISIT, "draw": VL_DRAW}

# (batch, 2, 8, 8) observations -> (batch, 65) policy logits, (batch, 1) values
Eval = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def library_path() -> Path:
    """The compiled search, built for this machine, named after its source."""
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    path = CACHE_DIR / "cpu-search" / f"cpu_search_{digest}.so"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # named per process: several may be compiling at once
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(
            ["g++", "-std=c++17", "-O3", "-march=native", "-shared", "-fPIC"]
            + [str(SOURCE), "-o", str(partial)],
            check=True,
        )
        partial.replace(path)
    return path


def load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(library_path()))
    Search = ctypes.c_void_p
    ints = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    u64 = ctypes.POINTER(ctypes.c_uint64)
    c_int = ctypes.POINTER(ctypes.c_int)
    signatures = {
        "search_new": (
            Search,
            [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int],
        ),
        "search_free": (None, [Search]),
        "search_reset": (None, [Search]),
        "search_begin": (None, [Search, ctypes.c_int]),
        "search_remaining": (ctypes.c_int, [Search]),
        "search_collect": (ctypes.c_int, [Search, ctypes.c_int, ctypes.c_void_p]),
        "search_apply": (None, [Search, ctypes.c_void_p, ctypes.c_void_p]),
        "search_num_legal": (ctypes.c_int, [Search]),
        "search_first_legal": (ctypes.c_int, [Search]),
        "search_best_action": (ctypes.c_int, [Search]),
        "search_visits": (None, [Search, ints]),
        "search_legal": (None, [Search, ints]),
        "search_root_q": (ctypes.c_float, [Search]),
        "search_play": (None, [Search, ctypes.c_int]),
        "search_position": (None, [Search, u64, u64, c_int, c_int]),
        "search_reward": (ctypes.c_float, [Search]),
        "search_collisions": (ctypes.c_long, [Search]),
        "search_total_visits": (ctypes.c_long, [Search]),
        "search_num_nodes": (ctypes.c_long, [Search]),
    }
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


LIB = load_library()


@dataclass(frozen=True)
class SearchSettings:
    """How a player searches.

    Attributes:
        batch: traversals a batch
        virtual_loss: how a traversal in flight counts against its edges: "loss", a lost visit;
            "draw", a drawn one; "visit", a visit in the exploration term only (Leela Chess Zero's)
        weight: visits a traversal in flight counts as
        c_init: PUCT's exploration constant (see `core`'s `PUCTSelector`)
        max_collisions: traversals of a batch allowed to find a leaf already in flight before it
            stops short; None for the batch's size
    """

    batch: int = 1
    virtual_loss: str = "loss"
    weight: float = 1.0
    c_init: float = 1.25
    max_collisions: int | None = None


class Search:
    """A game's tree, from its start: the root follows the game (`play`), keeping what it searched."""

    def __init__(self, settings: SearchSettings | None = None):
        settings = settings or SearchSettings()
        self.settings = settings
        collisions = settings.max_collisions
        self.handle = LIB.search_new(
            VIRTUAL_LOSS[settings.virtual_loss],
            settings.weight,
            settings.c_init,
            settings.batch if collisions is None else collisions,
        )
        LIB.search_reset(self.handle)

    def __del__(self) -> None:
        LIB.search_free(self.handle)

    def begin(self, visits: int) -> None:
        """Starts the search for a move, of `visits` visits of the tree."""
        LIB.search_begin(self.handle, visits)

    @property
    def remaining(self) -> int:
        return LIB.search_remaining(self.handle)

    def collect(self, out: np.ndarray) -> int:
        """Up to a batch of leaves to evaluate, their observations written to `out`, (at least
        batch, 2, 8, 8), float32. Returns how many; `apply` takes their evaluation, and must be
        called, even with none, before the next `collect`."""
        return LIB.search_collect(
            self.handle, self.settings.batch, out.ctypes.data_as(ctypes.c_void_p)
        )

    def apply(self, logits: np.ndarray, values: np.ndarray) -> None:
        """The leaves' (leaves, 65) policy logits and (leaves) values, float32."""
        LIB.search_apply(
            self.handle,
            logits.ctypes.data_as(ctypes.c_void_p),
            values.ctypes.data_as(ctypes.c_void_p),
        )

    def run(self, visits: int, evaluate: Eval) -> None:
        """A whole search: `visits` visits, evaluating their leaves with `evaluate`."""
        self.begin(visits)
        batch = np.empty((self.settings.batch, 2, 8, 8), np.float32)
        while self.remaining > 0:
            count = self.collect(batch)
            logits, values = evaluate(batch[:count])
            self.apply(logits, values)

    def best_action(self) -> int:
        """The most visited action."""
        return LIB.search_best_action(self.handle)

    def num_legal(self) -> int:
        return LIB.search_num_legal(self.handle)

    def first_legal(self) -> int:
        return LIB.search_first_legal(self.handle)

    def visits(self) -> np.ndarray:
        """(65) the visits of each of the root's actions."""
        out = np.zeros(65, np.int32)
        LIB.search_visits(self.handle, out)
        return out

    def legal(self) -> np.ndarray:
        """(65) whether each action is legal at the root."""
        out = np.zeros(65, np.int32)
        LIB.search_legal(self.handle, out)
        return out.astype(bool)

    def root_q(self) -> float:
        return LIB.search_root_q(self.handle)

    def play(self, action: int) -> None:
        LIB.search_play(self.handle, action)

    def position(self) -> tuple[int, int, bool, bool]:
        """(the discs of the player to move, the opponent's, as bits row by row; whether the last
        move was a pass; whether the game's over)."""
        me, opp = ctypes.c_uint64(), ctypes.c_uint64()
        passed, terminated = ctypes.c_int(), ctypes.c_int()
        LIB.search_position(
            self.handle,
            ctypes.byref(me),
            ctypes.byref(opp),
            ctypes.byref(passed),
            ctypes.byref(terminated),
        )
        return me.value, opp.value, bool(passed.value), bool(terminated.value)

    def terminated(self) -> bool:
        return self.position()[3]

    def reward(self) -> float:
        """The result for the player to move, once the game's over."""
        return LIB.search_reward(self.handle)

    def collisions(self) -> int:
        return LIB.search_collisions(self.handle)

    def total_visits(self) -> int:
        return LIB.search_total_visits(self.handle)

    def num_nodes(self) -> int:
        return LIB.search_num_nodes(self.handle)


def compile_model(model: bytes | str, precision: str, cores: int, streams: int, core):
    """An ONNX graph or a saved model, compiled by OpenVINO to compute in `precision` over `streams`
    requests in flight at once, sharing `cores`."""
    if isinstance(model, bytes):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            f.write(model)
            f.flush()
            network = core.read_model(f.name)
    else:
        network = core.read_model(model)
    network.reshape({"input": [-1, 2, 8, 8]})
    return core.compile_model(
        network,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": cores,
            "NUM_STREAMS": streams,
            "INFERENCE_PRECISION_HINT": precision,
        },
    )


def openvino_eval(
    model: bytes | str, cores: int, precision: str = "bf16", streams: int | None = None
) -> Eval:
    """The network of an ONNX graph or a saved model in OpenVINO, computing in `precision`: a batch
    is split over up to `streams` requests in flight at once, sharing `cores`. One to a core (the
    default) is the fastest way found at batches of 8 and up, one in all at a batch of 1. bf16 runs
    on AVX512_BF16 where the CPU has it."""
    import openvino as ov

    streams = streams or cores
    compiled = compile_model(model, precision, cores, streams, ov.Core())
    requests = [compiled.create_infer_request() for _ in range(streams)]

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.empty((0, 65), np.float32), np.empty((0,), np.float32)
        parts = np.array_split(x, min(streams, len(x)))
        used = requests[: len(parts)]
        for request, part in zip(used, parts):
            request.set_input_tensor(ov.Tensor(part))
            request.start_async()
        for request in used:
            request.wait()
        policy = np.concatenate([np.array(r.get_output_tensor(0).data) for r in used])
        value = np.concatenate([np.array(r.get_output_tensor(1).data) for r in used])
        return policy, value[:, 0]

    return evaluate


def ort_session(model: bytes | str, threads: int):
    """An ONNX Runtime session on the CPU, computing on `threads` threads within each call."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


def ort_eval(session) -> Eval:
    """The network of an `ort_session`, as `Search.run` takes it."""

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.empty((0, 65), np.float32), np.empty((0,), np.float32)
        policy, value = session.run(["policy", "value"], {"input": x})
        return policy, value[:, 0]  # type: ignore[index]

    return evaluate
