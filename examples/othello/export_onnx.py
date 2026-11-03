"""Exports a checkpoint's network to ONNX, for playing on CPU (see `cpu_search.py`).

    uv run examples/othello/export_onnx.py CHECKPOINT --net NET OUT.onnx

The graph is written by hand from the network's layers: its input is a float32 batch of
channels-first observations, (batch, 2, 8, 8), its outputs the policy logits, (batch, 65), and the
value, (batch, 1). BatchNorm, and the norm-free networks' fixed scales, are folded into the
convolutions, as in the network's inference mode (see `core.networks.azresnet.conv_norm`); the other
normalizations (LayerNorm, GroupNorm, RMSNorm) follow them.
"""

import argparse
import os
from pathlib import Path

# a CPU script: JAX would otherwise take the GPU's memory, and the league would give way
if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import numpy as np
import onnx
from game import load_checkpoint
from onnx import TensorProto, helper, numpy_helper

from core.networks.azresnet import (
    AZResnet,
    FixedScale,
    GroupNorm,
    Norm,
    PooledHeads,
    ResidualBlock,
    inference_affine,
)

OPSET = 17
IR_VERSION = 8


class Graph:
    """The nodes and weights of an ONNX graph as they're added."""

    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.weights: list[onnx.TensorProto] = []
        self.count = 0

    def name(self, stem: str) -> str:
        self.count += 1
        return f"{stem}_{self.count}"

    def weight(self, array, stem: str = "w") -> str:
        name = self.name(stem)
        self.weights.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def node(self, op: str, *inputs: str, **attrs) -> str:
        out = self.name(op.lower())
        self.nodes.append(helper.make_node(op, list(inputs), [out], **attrs))
        return out

    def relu(self, x: str) -> str:
        return self.node("Relu", x)

    def add(self, x: str, y: str) -> str:
        return self.node("Add", x, y)

    def conv(self, layer: eqx.nn.Conv2d, x: str, weight=None, bias=None) -> str:
        """`layer`, a convolution with no bias, with `weight` and `bias` if given in its place."""
        weight = np.asarray(layer.weight if weight is None else weight, np.float32)
        kernel = weight.shape[-1]
        inputs = [x, self.weight(weight)]
        if bias is not None:
            inputs.append(self.weight(np.asarray(bias, np.float32)))
        return self.node(
            "Conv",
            *inputs,
            kernel_shape=[kernel, kernel],
            pads=[kernel // 2] * 4,
        )

    def conv_norm(
        self, layer: eqx.nn.Conv2d, norm: Norm, state: eqx.nn.State, x: str
    ) -> str:
        """A convolution and its normalization: BatchNorm or a fixed scale folded into it, or
        followed by the others."""
        if isinstance(norm, eqx.nn.BatchNorm):
            scale, offset = inference_affine(norm, state)
            weight = layer.weight * scale.reshape(-1, 1, 1, 1)
            return self.conv(layer, x, weight, offset)
        if isinstance(norm, FixedScale):
            return self.conv(layer, x, norm.conv_weight(layer.weight), norm.bias)
        return self.group_norm(norm, self.conv(layer, x))

    def group_norm(self, norm: GroupNorm, x: str) -> str:
        """`norm` of (batch, channels, height, width) features, in plain ops: each group of
        channels to (batch, groups, rest)."""
        grouped = self.node(
            "Reshape", x, self.weight(np.array([0, norm.groups, -1]), "shape")
        )
        if norm.center:
            mean = self.node("ReduceMean", grouped, axes=[2], keepdims=1)
            grouped = self.node("Sub", grouped, mean)
        square = self.node("ReduceMean", self.node("Mul", grouped, grouped), axes=[2])
        eps = self.weight(np.array(norm.eps, np.float32), "eps")
        root = self.node("Sqrt", self.add(square, eps))
        normalized = self.node(
            "Reshape", self.node("Div", grouped, root), self.node("Shape", x)
        )
        channelwise = np.asarray(norm.weight, np.float32).reshape(-1, 1, 1)
        scaled = self.node("Mul", normalized, self.weight(channelwise, "scale"))
        offset = np.asarray(norm.bias, np.float32).reshape(-1, 1, 1)
        return self.add(scaled, self.weight(offset, "offset"))

    def linear(self, layer: eqx.nn.Linear, x: str) -> str:
        assert layer.bias is not None
        return self.node(
            "Gemm",
            x,
            self.weight(np.asarray(layer.weight, np.float32)),
            self.weight(np.asarray(layer.bias, np.float32)),
            transB=1,
        )

    def board_pool(self, x: str) -> str:
        """Each channel's mean and maximum over the board (see `azresnet.board_pool`)."""
        mean = self.node("ReduceMean", x, axes=[2, 3], keepdims=0)
        maximum = self.node("ReduceMax", x, axes=[2, 3], keepdims=0)
        return self.node("Concat", mean, maximum, axis=1)

    def board_bias(self, bias: str) -> str:
        """(batch, channels) to add to (batch, channels, 8, 8) features."""
        return self.node("Unsqueeze", bias, self.weight(np.array([2, 3]), "axes"))

    def slice_channels(self, x: str, start: int, end: int) -> str:
        return self.node(
            "Slice",
            x,
            self.weight(np.array([start]), "start"),
            self.weight(np.array([end]), "end"),
            self.weight(np.array([1]), "axis"),
        )

    def block(self, block: ResidualBlock, state: eqx.nn.State, x: str) -> str:
        y = self.conv_norm(block.conv1, block.norm1, state, x)
        if block.pool_linear is not None:
            channels = block.conv1.out_channels
            assert isinstance(channels, int)
            kept = channels - block.pool_channels
            pooled = self.slice_channels(y, kept, channels)
            y = self.slice_channels(y, 0, kept)
            pooled = self.board_pool(self.relu(pooled))
            bias = self.linear(block.pool_linear, pooled)
            y = self.add(y, self.board_bias(bias))
        y = self.relu(y)
        y = self.conv_norm(block.conv2, block.norm2, state, y)
        return self.relu(self.add(x, y))

    def heads(self, heads: PooledHeads, state: eqx.nn.State, x: str) -> tuple[str, str]:
        policy = self.conv_norm(heads.policy_conv, heads.policy_norm, state, x)
        pooled = self.conv_norm(
            heads.policy_pool_conv, heads.policy_pool_norm, state, x
        )
        pooled = self.board_pool(self.relu(pooled))
        bias = self.linear(heads.policy_pool_linear, pooled)
        policy = self.relu(self.add(policy, self.board_bias(bias)))
        squares = self.conv(heads.policy_square_conv, policy)
        squares = self.node("Flatten", squares, axis=1)
        policy = self.node(
            "Concat", squares, self.linear(heads.pass_linear, pooled), axis=1
        )
        value = self.conv_norm(heads.value_conv, heads.value_norm, state, x)
        value = self.board_pool(self.relu(value))
        value = self.relu(self.linear(heads.value_hidden, value))
        value = self.node("Tanh", self.linear(heads.value_linear, value))
        return policy, value


def to_onnx(network: AZResnet, state: eqx.nn.State) -> onnx.ModelProto:
    """The ONNX graph of `network`, of basic blocks and pooled heads, with its `state` (BatchNorm's,
    if it has any)."""
    if not isinstance(network.heads, PooledHeads):
        raise TypeError("only pooled heads export")
    graph = Graph()
    x = "input"
    x = graph.relu(graph.conv_norm(network.stem_conv, network.stem_norm, state, x))
    for block in network.blocks:
        if not isinstance(block, ResidualBlock):
            raise TypeError("only basic blocks export")
        x = graph.block(block, state, x)
    policy, value = graph.heads(network.heads, state, x)
    graph.nodes.append(helper.make_node("Identity", [policy], ["policy"]))
    graph.nodes.append(helper.make_node("Identity", [value], ["value"]))
    model = helper.make_model(
        helper.make_graph(
            graph.nodes,
            "othello",
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, ["batch", 2, 8, 8]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "policy", TensorProto.FLOAT, ["batch", 65]
                ),
                helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch", 1]),
            ],
            graph.weights,
        ),
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    # the version of the file format, which runtimes lag behind onnx's own default for
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def export_checkpoint(path: str, net: str) -> onnx.ModelProto:
    """The ONNX graph of the `net` network in the checkpoint at `path`."""
    network, (params, state) = load_checkpoint(path, net)
    static = eqx.filter(network, eqx.is_inexact_array, inverse=True)
    return to_onnx(eqx.nn.inference_mode(eqx.combine(params, static)), state)


def save_onnx(path: str, net: str, out: Path) -> None:
    """Writes the ONNX graph of the checkpoint at `path` to `out`, whole or not at all, on the
    CPU, so that it can't get in the way of anything on the GPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = export_checkpoint(path, net)
    partial = out.with_name(f"{out.name}.partial")
    onnx.save(model, partial)
    os.chmod(partial, 0o444)
    partial.rename(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--net", required=True, help="the checkpoint's train.py --net")
    parser.add_argument("out", help="the .onnx file to write")
    args = parser.parse_args()
    save_onnx(args.checkpoint, args.net, Path(args.out))


if __name__ == "__main__":
    main()
