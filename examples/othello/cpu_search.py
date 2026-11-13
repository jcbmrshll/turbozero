"""A single game's MCTS for playing on CPU: the tree and Othello's rules in C++ (`cpu_search.cpp`,
compiled on first use), the network in ONNX Runtime (see `export_onnx.py`).

A search visits the tree a batch of traversals at a time, each steered away from those still in
flight by virtual loss, and evaluates their leaves together: a batch of 1 is the sequential search
of `core/evaluators/mcts`. Only for play: self-play and training use `core`'s batched search.

With `random_symmetry`, the network sees each leaf in a random one of the board's 8 symmetries, and
its policy is mapped back to the real squares: whatever it gets wrong about one orientation doesn't
repeat across the tree.
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
from native_target import target_flags

SOURCE = Path(__file__).parent / "cpu_search.cpp"
# what it includes of its own
HEADERS = [Path(__file__).parent / "cpu_net.h"]
CACHE_DIR = Path(
    os.environ.get("TURBOZERO_CACHE", Path.home() / ".cache" / "turbozero" / "othello")
)
PASS = 64

VL_LOSS, VL_VISIT, VL_DRAW = 0, 1, 2
VIRTUAL_LOSS = {"loss": VL_LOSS, "visit": VL_VISIT, "draw": VL_DRAW}

# (batch, 2, 8, 8) observations -> (batch, 65) policy logits, (batch, 1) values
Eval = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def openvino_paths() -> tuple[Path, Path]:
    """OpenVINO's headers and its runtime library, from the installed Python package."""
    import openvino

    root = Path(openvino.__file__).parent
    # libopenvino.so.<version> on Linux, libopenvino.<version>.dylib on macOS
    (library,) = [
        *(root / "libs").glob("libopenvino.so.*"),
        *(root / "libs").glob("libopenvino.*.dylib"),
    ]
    return root / "include", library


def library_path() -> Path:
    """The compiled search, built for this machine's level (`native_target`), named after its source, its
    target and the OpenVINO it links."""
    include, openvino = openvino_paths()
    target = target_flags()
    source = (
        b"".join(p.read_bytes() for p in [SOURCE, *HEADERS])
        + str([openvino, target]).encode()
    )
    digest = hashlib.sha256(source).hexdigest()[:16]
    path = CACHE_DIR / "cpu-search" / f"cpu_search_{digest}.so"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # named per process: several may be compiling at once
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(
            ["g++", "-std=c++17", "-O3", *target, "-fopenmp-simd", "-shared", "-fPIC"]
            + [str(SOURCE), "-o", str(partial), f"-I{include}", str(openvino)]
            + [f"-Wl,-rpath,{openvino.parent}", "-lpthread"],
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
    floats = ctypes.POINTER(ctypes.c_float)
    signatures = {
        "search_new": (
            Search,
            [
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
            ],
        ),
        "search_free": (None, [Search]),
        "search_reset": (None, [Search]),
        "search_set_position": (None, [Search, ctypes.c_uint64, ctypes.c_uint64]),
        "search_begin": (None, [Search, ctypes.c_int]),
        "search_remaining": (ctypes.c_int, [Search]),
        "search_collect": (ctypes.c_int, [Search, ctypes.c_int, ctypes.c_void_p]),
        "search_apply": (None, [Search, ctypes.c_void_p, ctypes.c_void_p]),
        "batch_new": (ctypes.c_void_p, []),
        "batch_free": (None, [ctypes.c_void_p]),
        "search_collect_into": (
            ctypes.c_int,
            [Search, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
        ),
        "search_apply_from": (
            None,
            [Search, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        ),
        "search_collect_bits": (
            ctypes.c_int,
            [Search, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
        ),
        "search_ended_in_a_row": (ctypes.c_int, [Search]),
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
        "search_reused": (ctypes.c_long, [Search]),
        "search_symmetric_bits": (ctypes.c_uint64, [ctypes.c_uint64, ctypes.c_int]),
        "search_symmetric_action": (ctypes.c_int, [ctypes.c_int, ctypes.c_int]),
        "engine_new": (
            ctypes.c_void_p,
            [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        ),
        "engine_new_net": (
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
        ),
        "engine_free": (None, [ctypes.c_void_p]),
        "net_supported": (ctypes.c_int, []),
        "net_new": (ctypes.c_void_p, [ctypes.c_int, ctypes.c_int, ctypes.c_int]),
        "net_free": (None, [ctypes.c_void_p]),
        "net_add_block": (None, [ctypes.c_void_p, ctypes.c_int]),
        "net_set_conv": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            + [ctypes.c_int, ctypes.POINTER(ctypes.c_int8), floats, floats]
            + [floats, ctypes.c_int],
        ),
        "net_set_norm": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            + [floats, floats, ctypes.c_float],
        ),
        "net_set_linear": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            + [floats, floats],
        ),
        "net_set_heads": (None, [ctypes.c_void_p, floats]),
        "net_forward": (
            None,
            [ctypes.c_void_p, ctypes.c_int, floats, floats, floats, ctypes.c_int],
        ),
        "engine_search": (
            ctypes.c_long,
            [ctypes.c_void_p, Search, ctypes.c_int, ctypes.c_double, ctypes.c_int],
        ),
        "engine_stats": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
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
        prior_temperature: the softmax temperature of every node's prior, the network's logits
            divided by it: above 1 flattens the prior (see `core`'s `MCTS`)
        max_collisions: traversals of a batch allowed to find a leaf already in flight before it
            stops short; None for the batch's size
        random_symmetry: show the network each leaf in a random one of the board's 8 symmetries,
            drawn from the search's seed
        reuse: give a leaf whose position the game's searches have evaluated before, reached by
            another order of moves, that evaluation rather than the network's (Leela Chess Zero's
            cache of evaluations, here the tree itself): about 14-17% of leaves at 1000-4000
            visits, more near the end. Without `random_symmetry` the search is the same, only
            cheaper.
    """

    batch: int = 1
    virtual_loss: str = "loss"
    weight: float = 1.0
    c_init: float = 1.25
    prior_temperature: float = 1.0
    max_collisions: int | None = None
    random_symmetry: bool = False
    reuse: bool = True

    def __post_init__(self) -> None:
        if self.prior_temperature <= 0:
            raise ValueError(
                f"prior_temperature must be positive, got {self.prior_temperature}"
            )


class Search:
    """A game's tree, from its start: the root follows the game (`play`), keeping what it searched.
    `seed` starts its draws of symmetries, the only randomness it has."""

    def __init__(self, settings: SearchSettings | None = None, seed: int = 0):
        settings = settings or SearchSettings()
        self.settings = settings
        collisions = settings.max_collisions
        self.handle = LIB.search_new(
            VIRTUAL_LOSS[settings.virtual_loss],
            settings.weight,
            settings.c_init,
            settings.prior_temperature,
            settings.batch if collisions is None else collisions,
            settings.random_symmetry,
            settings.reuse,
            seed,
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

    def collect_into(self, batch: "Batch", count: int, out: np.ndarray) -> int:
        """`collect`, of up to `count` leaves, into a `Batch` of the caller's: several can be
        out for evaluation at once, each applied (`apply_from`) when its evaluation comes back,
        in any order. `out` is C-contiguous float32 with room for `count`."""
        assert out.dtype == np.float32 and out.flags.c_contiguous and len(out) >= count
        return LIB.search_collect_into(
            self.handle, batch.handle, count, out.ctypes.data_as(ctypes.c_void_p)
        )

    def collect_bits(self, batch: "Batch", count: int, out: np.ndarray) -> int:
        """`collect_into`, each leaf's observation packed: `out`, C-contiguous uint64 with room
        for `count` rows of two words, the discs of the player to move and the other's (bit i
        the square of row i // 8, column i % 8; see `unpack`)."""
        assert out.dtype == np.uint64 and out.flags.c_contiguous
        assert out.shape[1:] == (2,) and len(out) >= count
        return LIB.search_collect_bits(
            self.handle, batch.handle, count, out.ctypes.data_as(ctypes.c_void_p)
        )

    def apply_from(
        self, batch: "Batch", logits: np.ndarray | None, values: np.ndarray | None
    ) -> None:
        """The evaluations of what `collect_into` put in `batch`, (leaves, 65) and (leaves)
        float32, C-contiguous; None for a batch of no leaves (it may still hold collisions)."""
        if logits is None or values is None:
            LIB.search_apply_from(self.handle, batch.handle, None, None)
            return
        logits = np.ascontiguousarray(logits, np.float32)
        values = np.ascontiguousarray(values, np.float32)
        LIB.search_apply_from(
            self.handle,
            batch.handle,
            logits.ctypes.data_as(ctypes.c_void_p),
            values.ctypes.data_as(ctypes.c_void_p),
        )

    def ended_in_a_row(self) -> int:
        """Traversals in a row, to now, that ended the game with no leaf between."""
        return LIB.search_ended_in_a_row(self.handle)

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

    def set_position(self, me: int, opp: int) -> None:
        """Starts the tree afresh from a position given whole: the discs of the player to move,
        and the opponent's, as bits (bit 0 is a1)."""
        LIB.search_set_position(self.handle, me, opp)

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

    def reused(self) -> int:
        """Leaves that took an evaluation from elsewhere in the tree, of the same position reached
        by another order of moves, rather than the network's."""
        return LIB.search_reused(self.handle)


def pack(x: np.ndarray) -> np.ndarray:
    """(n, 2, 8, 8) observations, 0 or 1, as `Search.collect_bits` writes them: (n, 2) uint64."""
    flat = np.asarray(x).reshape(len(x), 2, 64) > 0.5
    return (
        np.packbits(flat, axis=-1, bitorder="little").view(np.uint64).reshape(len(x), 2)
    )


def unpack(bits: np.ndarray) -> np.ndarray:
    """`pack`'s inverse: (n, 2) uint64 to (n, 2, 8, 8) float32."""
    bytes_ = (
        np.ascontiguousarray(bits, np.uint64).view(np.uint8).reshape(len(bits), 2, 8)
    )
    planes = np.unpackbits(bytes_, axis=-1, bitorder="little")
    return planes.reshape(len(bits), 2, 8, 8).astype(np.float32)


class Batch:
    """The traversals of one collection of a `Search`'s, until their evaluations are applied:
    see `Search.collect_into`."""

    def __init__(self) -> None:
        self.handle = LIB.batch_new()

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            LIB.batch_free(self.handle)


class Engine:
    """A network on `threads` cores, searching one tree at a time with a worker thread to a core
    (see `engine_search` in `cpu_search.cpp`): each collects up to `batch` leaves, evaluates them on
    its own core while the others go on, and backs them up, all in C++. Made by `openvino` or
    `native`."""

    def __init__(self, handle: int, threads: int, batch: int, network=None):
        self.handle = handle
        self.threads = threads
        self.batch = batch
        # what the engine's C++ evaluates with, which must outlive it
        self.network = network

    @classmethod
    def openvino(
        cls, model: str, threads: int, batch: int, precision: str = "f32"
    ) -> "Engine":
        """`model`, a saved OpenVINO model or ONNX graph's path, compiled by OpenVINO for each
        worker and number of leaves, computing what isn't quantized in `precision`."""
        handle = LIB.engine_new(str(model).encode(), precision.encode(), threads, batch)
        if not handle:
            raise RuntimeError(f"couldn't compile {model}")
        return cls(handle, threads, batch)

    @classmethod
    def native(cls, network, threads: int, batch: int) -> "Engine":
        """A `cpu_net.NativeNet`, its workers each evaluating their leaves one at a time."""
        handle = LIB.engine_new_net(network.handle, threads, batch)
        if not handle:
            raise RuntimeError("the native network needs AVX2 or AVX-512 VNNI")
        return cls(handle, threads, batch, network)

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            LIB.engine_free(self.handle)

    def search(
        self,
        search: "Search",
        visits: int = 0,
        seconds: float = 0.0,
        stop_ended: int = 0,
    ) -> int:
        """`visits` visits of `search`'s tree, or with none, as many as `seconds` allow; stopping
        early once `stop_ended` traversals in a row end the game. Returns the leaves evaluated."""
        return LIB.engine_search(
            self.handle, search.handle, visits, seconds, stop_ended
        )

    def stats(self) -> dict[str, float]:
        """Calls of the network, leaves and seconds inferring summed over workers, so far."""
        out = (ctypes.c_double * 3)()
        LIB.engine_stats(self.handle, out)
        return {"calls": out[0], "leaves": out[1], "infer_seconds": out[2]}


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
