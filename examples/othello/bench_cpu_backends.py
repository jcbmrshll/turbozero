"""Network evaluations per second on CPU by backend, on positions of the engine-labelled set,
checking each backend's outputs against JAX's.

    taskset -c 16-23 JAX_PLATFORMS=cpu uv run examples/othello/bench_cpu_backends.py \
        --backend ort jax --net 8x128-128s-8p-w1125-274m --results results.jsonl

A network is named as in the league's store (its `networks` table). Backends:

    jax              one jitted call
    jax-split        the batch split over a thread per core (with XLA_FLAGS to run each on one)
    ort              ONNX Runtime, all cores within each call
    ort-split        ONNX Runtime, the batch split over a thread per core, one core per call
    ov-f32-8         OpenVINO in f32 (or bf16), the batch split over 8 streams (or any number)
    ort-int8         ONNX Runtime on a quantized (int8, QDQ, per-channel) graph
    ort-int8-split   the same, the batch split over a thread per core
"""

import argparse
import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from engine_set import DEFAULT_PATH, EngineSet
from export_onnx import export_checkpoint
from game import load_checkpoint

from core.networks.utils import apply_nn

STORE = Path.home() / ".cache/turbozero/othello/league/league.db"
BATCHES = (1, 8, 64)
Eval = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def network_path(name: str) -> tuple[str, str]:
    store = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    row = store.execute(
        "select path, net from networks where name = ?", (name,)
    ).fetchone()
    return row[0], row[1]


def jax_eval(path: str, net: str) -> Eval:
    """Takes channels-first observations, as the ONNX graph does."""
    network, (params, state) = load_checkpoint(path, net)
    static = eqx.filter(network, eqx.is_inexact_array, inverse=True)
    model = eqx.nn.inference_mode(eqx.combine(params, static))
    apply = jax.jit(lambda x: apply_nn(model, state, jnp.moveaxis(x, 1, -1))[0])

    def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy, value = apply(x)
        return np.asarray(policy), np.asarray(value)

    return run


def ort_eval(model: bytes, threads: int) -> Eval:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])

    def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy, value = session.run(["policy", "value"], {"input": x})
        return np.asarray(policy), np.asarray(value)

    return run


def split(make: Callable[[], Eval], cores: int) -> Eval:
    """An eval of a batch by thread per core, `make` giving each its own."""
    pool = ThreadPoolExecutor(cores)
    evals = [make() for _ in range(cores)]

    def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = [p for p in np.array_split(x, cores) if len(p)]
        out = list(pool.map(lambda e, p: e(p), evals, parts))
        return (
            np.concatenate([o[0] for o in out]),
            np.concatenate([o[1] for o in out]),
        )

    return run


def ov_eval(model: bytes, cores: int, batch: int, precision: str, streams: int) -> Eval:
    """OpenVINO, computing in `precision`: the batch split over `streams` requests in flight at
    once (`request.infer` would hold Python's lock), sharing `cores`."""
    import openvino as ov

    core = ov.Core()
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        f.write(model)
        f.flush()
        m = core.read_model(f.name)
    streams = min(streams, batch)
    if batch % streams:
        raise ValueError(f"batch {batch} doesn't split over {streams} streams")
    m.reshape({"input": [batch // streams, 2, 8, 8]})
    compiled = core.compile_model(
        m,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": cores,
            "NUM_STREAMS": streams,
            "INFERENCE_PRECISION_HINT": precision,
        },
    )
    print("  openvino:", compiled.get_property("INFERENCE_PRECISION_HINT"), flush=True)
    requests = [compiled.create_infer_request() for _ in range(streams)]

    def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = np.split(x, streams)
        for request, part in zip(requests, parts):
            request.set_input_tensor(ov.Tensor(part))
            request.start_async()
        for request in requests:
            request.wait()
        return (
            np.concatenate([np.array(r.get_output_tensor(0).data) for r in requests]),
            np.concatenate([np.array(r.get_output_tensor(1).data) for r in requests]),
        )

    return run


def quantized(model: bytes, calibration: np.ndarray) -> bytes:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.items = iter(calibration[i : i + 1] for i in range(len(calibration)))

        def get_next(self):  # pyright: ignore[reportIncompatibleMethodOverride]
            x = next(self.items, None)
            return None if x is None else {"input": x}

    with tempfile.TemporaryDirectory() as d:
        source, out = f"{d}/f.onnx", f"{d}/q.onnx"
        Path(source).write_bytes(model)
        quantize_static(
            source,
            out,
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        return Path(out).read_bytes()


def make_eval(
    backend: str,
    path: str,
    net: str,
    model: bytes,
    calibration: np.ndarray,
    batch: int,
    cores: int,
) -> Eval:
    if backend in ("jax", "jax-split"):
        make = lambda: jax_eval(path, net)
    elif backend.startswith("ort-int8"):
        q = quantized(model, calibration)
        make = lambda: ort_eval(q, 1 if "split" in backend else cores)
    elif backend.startswith("ort"):
        make = lambda: ort_eval(model, 1 if "split" in backend else cores)
    elif backend.startswith("ov-"):
        # ov-<precision>-<streams>, e.g. ov-f32-8
        _, precision, streams = backend.split("-")
        return ov_eval(model, cores, batch, precision, int(streams))
    else:
        raise ValueError(backend)
    if "split" in backend and batch > 1:
        return split(make, min(cores, batch))
    return make()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--backend", nargs="+", required=True)
    parser.add_argument("--net", nargs="+", required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--label", default="", help="recorded, e.g. the XLA flags")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    observations = np.asarray(
        EngineSet.load(str(DEFAULT_PATH)).scored_positions().observations
    )
    # channels-first, floats, as the graph takes
    observations = np.moveaxis(observations, -1, 1).astype(np.float32)
    calibration, holdout = observations[:512], observations[512:1536]
    cores = len(os.sched_getaffinity(0))
    print(
        f"{cores} cores; jax {jax.__version__}, load {os.getloadavg()[0]:.1f}",
        flush=True,
    )

    for name in args.net:
        path, net = network_path(name)
        model = export_checkpoint(path, net).SerializeToString()
        reference = jax_eval(path, net)(holdout)
        for backend in args.backend:
            for batch in args.batch:
                run = make_eval(backend, path, net, model, calibration, batch, cores)
                x = holdout[:batch]
                run(x)
                start = time.perf_counter()
                calls = 0
                while (seconds := time.perf_counter() - start) < args.seconds:
                    run(x)
                    calls += 1
                rate = calls * batch / seconds
                # the outputs against JAX's, in batches of this size
                outs = [
                    run(holdout[i : i + batch])
                    for i in range(0, 256 - 256 % batch, batch)
                ]
                policy = np.concatenate([o[0] for o in outs])
                value = np.concatenate([o[1] for o in outs])
                value_error = float(np.abs(value - reference[1][: len(value)]).mean())
                logit_error = float(np.abs(policy - reference[0][: len(policy)]).mean())
                agree = float(
                    (
                        policy[:, :64].argmax(1)
                        == reference[0][: len(policy), :64].argmax(1)
                    ).mean()
                )
                record = {
                    "net": name,
                    "backend": backend,
                    "batch": batch,
                    "evals_per_s": round(rate),
                    "value_mae": value_error,
                    "logit_mae": logit_error,
                    "top1_agree": agree,
                    "label": args.label,
                    "load": os.getloadavg()[0],
                }
                print(json.dumps(record), flush=True)
                if args.results is not None:
                    with open(args.results, "a") as f:
                        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
