"""Quantizes a network for playing on CPU: post-training int8 by NNCF, as an OpenVINO model.

    uv run examples/othello/fidelity.py leaves --net 8x128-128s-8p-w1125-274m --out LEAVES
    uv run examples/othello/quantize.py --net 8x128-128s-8p-w1125-274m \
        --positions LEAVES/calibration.npy

A network is named as in the league's store. It's calibrated on `--positions`, an .npy of
channels-first float32 observations, best the leaves of its own searches, or else on positions of
the engine-labelled set. Models are kept in `QUANTIZED_DIR`, named by the network and the variant,
and played as the backend "int8-<variant>" (see `cpu_match.make_eval`).
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from bench_cpu_backends import network_path
from export_onnx import export_checkpoint

QUANTIZED_DIR = (
    Path(
        os.environ.get(
            "TURBOZERO_CACHE", Path.home() / ".cache" / "turbozero" / "othello"
        )
    )
    / "quantized"
)


# the settings that played best: see the README's "Playing on CPU"
BEST = "performance_leaves_heads_minmax"


def engine_positions(count: int) -> np.ndarray:
    """The first `count` positions of the engine-labelled set, as the ONNX graph takes them."""
    from engine_set import DEFAULT_PATH, EngineSet

    observations = np.asarray(
        EngineSet.load(str(DEFAULT_PATH)).scored_positions().observations
    )
    return np.moveaxis(observations, -1, 1).astype(np.float32)[:count]


def quantized_path(name: str, variant: str = BEST) -> Path:
    return QUANTIZED_DIR / f"{name}-{variant}.xml"


def head_ops(model, outputs: list[str]) -> list[str]:
    """The names of the ops computing the given outputs from the trunk's last features, which are
    what the value head's convolution takes."""

    def inputs(op) -> list:
        return [i.get_source_output().get_node() for i in op.inputs()]

    def ancestors(op) -> dict:
        seen = {op.get_friendly_name(): op}
        stack = [op]
        while stack:
            for parent in inputs(stack.pop()):
                if parent.get_friendly_name() not in seen:
                    seen[parent.get_friendly_name()] = parent
                    stack.append(parent)
        return seen

    def result(output: str):
        return next(
            r for r in model.get_results() if output in r.input_value(0).get_names()
        )

    above_value = ancestors(result("value"))
    convs = [op for op in above_value.values() if op.get_type_name() == "Convolution"]
    # the value head's only convolution is the nearest to its output: the walk meets it first
    trunk_end = inputs(convs[0])[0]
    trunk = ancestors(trunk_end)
    names = set()
    for output in outputs:
        names |= set(ancestors(result(output))) - set(trunk)
    return sorted(names)


def quantize_graph(graph: bytes, variant: str, calibration: np.ndarray):
    """The int8 OpenVINO model of an ONNX graph. A `variant` starts with NNCF's preset, "mixed",
    asymmetric activations, or "performance", symmetric, then options after underscores: "value" or
    "heads" keep the value head or both heads in float, "stem" the first convolution, "last" the
    trunk's last two, "noof" gives the first layer's weights all 8 bits (VNNI doesn't overflow),
    "minmax" takes activations' ranges over all the calibration rather than each position's mean,
    "pertensor" a scale per weight rather than per channel, and any other just labels, e.g. the
    calibration."""
    import nncf
    import openvino as ov

    core = ov.Core()
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        f.write(graph)
        f.flush()
        model = core.read_model(f.name)
    model.reshape({"input": [1, 2, 8, 8]})
    presets = {
        "mixed": nncf.QuantizationPreset.MIXED,
        "performance": nncf.QuantizationPreset.PERFORMANCE,
    }
    preset, *options = variant.split("_")
    ignored = []
    if "value" in options:
        ignored = head_ops(model, ["value"])
    if "heads" in options:
        ignored = head_ops(model, ["value", "policy"])
    trunk_convs = [
        op.get_friendly_name()
        for op in model.get_ordered_ops()
        if op.get_type_name() == "Convolution"
        and op.get_friendly_name() not in head_ops(model, ["value", "policy"])
    ]
    if "stem" in options:
        ignored = [*ignored, trunk_convs[0]]
    if "last" in options:
        ignored = [*ignored, *trunk_convs[-2:]]
    extra = {}
    if "minmax" in options:
        extra["activations_range_estimator_params"] = (
            nncf.quantization.range_estimator.RangeEstimatorParametersSet.MINMAX
        )
    if "pertensor" in options:
        extra["weights_quantization_params"] = (
            nncf.quantization.advanced_parameters.QuantizationParameters(
                per_channel=False
            )
        )
    advanced = nncf.AdvancedQuantizationParameters(
        overflow_fix=nncf.OverflowFix.DISABLE
        if "noof" in options
        else nncf.OverflowFix.FIRST_LAYER,
        **extra,
    )
    return nncf.quantize(
        model,
        nncf.Dataset(calibration, lambda x: x[None]),
        preset=presets[preset],
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(calibration),
        ignored_scope=nncf.IgnoredScope(names=ignored, validate=False),
        advanced_parameters=advanced,
    )


def quantize(name: str, variant: str, calibration: np.ndarray) -> Path:
    """Writes the int8 model of the network `name`."""
    import openvino as ov

    path, net = network_path(name)
    graph = export_checkpoint(path, net).SerializeToString()
    quantized = quantize_graph(graph, variant, calibration)
    out = quantized_path(name, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, out, compress_to_fp16=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--net", nargs="+", required=True)
    parser.add_argument("--variant", default=BEST)
    parser.add_argument("--positions", type=Path, default=None)
    parser.add_argument("--calibration", type=int, default=512)
    args = parser.parse_args()
    calibration = (
        engine_positions(args.calibration)
        if args.positions is None
        else np.load(args.positions)[: args.calibration]
    )
    for name in args.net:
        print(quantize(name, args.variant, calibration), flush=True)


if __name__ == "__main__":
    main()
