import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass
class BoardTransformerConfig:
    """Configuration for a board transformer, after Leela Chess Zero's: a token per square.

    Attributes:
        policy_head_out_size: output size of the policy head (number of actions): the board's
            squares, row by row, then a pass
        num_layers: number of transformer layers
        dim: width of each square's token
        num_heads: attention heads per layer
        ffn_dim: hidden width of each layer's feed-forward network, by default `dim` (Leela's: 1 to
            2 times `dim`)
        attention_bias: give each layer's attention a learned bias for each pair of squares and
            head: position information inside attention, which Leela found worth about half again
            the network's size, for a few percent more compute
        smolgen: give each layer's attention Leela's smolgen bias as well: a bias for each pair of
            squares and head generated from the whole board. Each square's token is compressed to
            `smolgen_compress` channels, all of them map through a hidden layer of `smolgen_hidden`
            units to `smolgen_head` per head, and a linear map shared by all layers makes those the
            head's bias for every pair of squares
        smolgen_compress: see `smolgen`
        smolgen_hidden: see `smolgen`
        smolgen_head: see `smolgen`
        norm: "post", LayerNorm after each residual sum, with DeepNorm's scaling of the residual and
            initialization (Leela's); or "pre", LayerNorm at the start of each residual branch, and
            once more before the heads
        value_channels: channels each square's token is projected to for the value head, which
            takes them all (as AlphaZero's value head takes its convolution's 1-channel board)
        value_hidden_size: width of the value head's hidden layer
        inference_dtype: dtype the matrix multiplications and activations compute in, in inference
            mode (see `eqx.nn.inference_mode`), e.g. "bfloat16" for faster self-play
        train_dtype: dtype the matrix multiplications compute in, in training, e.g. "bfloat16" for
            mixed precision: the parameters, the residual stream, LayerNorm, the attention softmax
            and the heads stay in float32
        remat: in training, keep only each layer's input for the backward pass, and compute its
            activations again from it (see `jax.checkpoint`)
    """

    policy_head_out_size: int
    num_layers: int
    dim: int
    num_heads: int = 8
    ffn_dim: int | None = None
    attention_bias: bool = True
    smolgen: bool = False
    smolgen_compress: int = 32
    smolgen_hidden: int = 256
    smolgen_head: int = 256
    norm: Literal["post", "pre"] = "post"
    value_channels: int = 8
    value_hidden_size: int = 256
    inference_dtype: str = "float32"
    train_dtype: str = "float32"
    remat: bool = False


class Dense(eqx.Module):
    """An affine map on the last axis that computes in a given dtype."""

    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int, gain: float = 1.0, *, key):
        """Initialized as Xavier-normal with `gain`, the bias at 0."""
        std = gain * math.sqrt(2 / (in_size + out_size))
        self.weight = std * jax.random.normal(key, (in_size, out_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: jax.Array, dtype: jnp.dtype) -> jax.Array:
        return x.astype(dtype) @ self.weight.astype(dtype) + self.bias.astype(dtype)


class LayerNorm(eqx.Module):
    """LayerNorm over the last axis; its statistics in float32, its output in the input's dtype."""

    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones((dim,))
        self.bias = jnp.zeros((dim,))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x.astype(jnp.float32)
        mean = y.mean(axis=-1, keepdims=True)
        var = ((y - mean) ** 2).mean(axis=-1, keepdims=True)
        y = (y - mean) * jax.lax.rsqrt(var + self.eps)
        return (y * self.weight + self.bias).astype(x.dtype)


class Smolgen(eqx.Module):
    """Leela's smolgen: a layer's attention bias for each head and pair of squares, generated from
    the whole board (see `BoardTransformerConfig.smolgen`). Up to the linear map, shared by all
    layers, from each head's features to its biases."""

    compress: Dense
    hidden: Dense
    hidden_norm: LayerNorm
    heads: Dense
    heads_norm: LayerNorm
    num_heads: int = eqx.field(static=True)

    def __init__(self, config: BoardTransformerConfig, squares: int, *, key: jax.Array):
        compress_key, hidden_key, heads_key = jax.random.split(key, 3)
        self.compress = Dense(config.dim, config.smolgen_compress, key=compress_key)
        self.hidden = Dense(
            squares * config.smolgen_compress, config.smolgen_hidden, key=hidden_key
        )
        self.hidden_norm = LayerNorm(config.smolgen_hidden)
        self.heads = Dense(
            config.smolgen_hidden,
            config.num_heads * config.smolgen_head,
            key=heads_key,
        )
        self.heads_norm = LayerNorm(config.smolgen_head)
        self.num_heads = config.num_heads

    def __call__(self, x: jax.Array, generator: Dense, dtype: jnp.dtype) -> jax.Array:
        """The biases, (heads, squares, squares), from the tokens `x` (squares, dim) and the
        shared map `generator`."""
        squares = x.shape[0]
        y = self.compress(x, dtype).reshape(-1)
        y = self.hidden_norm(jax.nn.mish(self.hidden(y, dtype)))
        y = self.heads(y, dtype).reshape(self.num_heads, -1)
        y = self.heads_norm(jax.nn.mish(y))
        return generator(y, dtype).reshape(self.num_heads, squares, squares)


class Layer(eqx.Module):
    """A transformer layer: multi-head self-attention over the squares, then a feed-forward network
    with Mish, each a residual branch."""

    qkv: Dense
    out: Dense
    bias: jax.Array | None
    smolgen: Smolgen | None
    ffn_in: Dense
    ffn_out: Dense
    norm1: LayerNorm
    norm2: LayerNorm
    num_heads: int = eqx.field(static=True)
    post_norm: bool = eqx.field(static=True)
    # DeepNorm's scaling of the residual in post-norm layers (1 in pre-norm ones)
    alpha: float = eqx.field(static=True)

    def __init__(self, config: BoardTransformerConfig, squares: int, *, key: jax.Array):
        dim, heads = config.dim, config.num_heads
        if dim % heads:
            raise ValueError(f"width {dim} doesn't split into {heads} heads")
        q_key, k_key, v_key, out_key, in_key, ffn_key, smolgen_key = jax.random.split(
            key, 7
        )
        post_norm = config.norm == "post"
        # DeepNorm (for an encoder of N layers): residual scaled by (2N)^(1/4), and the value,
        # output and feed-forward weights initialized smaller by (8N)^(-1/4)
        n = config.num_layers
        self.alpha = (2 * n) ** 0.25 if post_norm else 1.0
        beta = (8 * n) ** -0.25 if post_norm else 1.0
        # queries and keys at gain 1, values at beta: the three as one matrix
        q, k, v = (
            Dense(dim, dim, gain, key=key_)
            for gain, key_ in ((1.0, q_key), (1.0, k_key), (beta, v_key))
        )
        self.qkv = eqx.tree_at(
            lambda d: (d.weight, d.bias),
            q,
            (
                jnp.concatenate([q.weight, k.weight, v.weight], axis=1),
                jnp.zeros((3 * dim,)),
            ),
        )
        self.out = Dense(dim, dim, beta, key=out_key)
        self.bias = (
            jnp.zeros((heads, squares, squares)) if config.attention_bias else None
        )
        self.smolgen = (
            Smolgen(config, squares, key=smolgen_key) if config.smolgen else None
        )
        ffn_dim = config.ffn_dim or dim
        self.ffn_in = Dense(dim, ffn_dim, beta, key=in_key)
        self.ffn_out = Dense(ffn_dim, dim, beta, key=ffn_key)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.num_heads = heads
        self.post_norm = post_norm

    def attention(
        self, x: jax.Array, generator: Dense | None, dtype: jnp.dtype
    ) -> jax.Array:
        squares, dim = x.shape
        head_dim = dim // self.num_heads
        qkv = self.qkv(x, dtype).reshape(squares, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        logits = jnp.einsum("qhd,khd->hqk", q, k).astype(jnp.float32)
        logits = logits / math.sqrt(head_dim)
        if self.bias is not None:
            logits = logits + self.bias
        if self.smolgen is not None:
            assert generator is not None
            logits = logits + self.smolgen(x, generator, dtype).astype(jnp.float32)
        weights = jax.nn.softmax(logits, axis=-1).astype(dtype)
        y = jnp.einsum("hqk,khd->qhd", weights, v).reshape(squares, dim)
        return self.out(y, dtype)

    def ffn(self, x: jax.Array, dtype: jnp.dtype) -> jax.Array:
        return self.ffn_out(jax.nn.mish(self.ffn_in(x, dtype)), dtype)

    def __call__(
        self, x: jax.Array, generator: Dense | None, dtype: jnp.dtype
    ) -> jax.Array:
        """x is the residual stream, (squares, dim), in float32 in training and in the inference
        dtype in inference; `generator` smolgen's map shared by all layers, if they have smolgen."""
        if self.post_norm:
            attention = self.attention(x, generator, dtype)
            x = self.norm1(self.alpha * x + attention.astype(x.dtype))
            return self.norm2(self.alpha * x + self.ffn(x, dtype).astype(x.dtype))
        x = x + self.attention(self.norm1(x), generator, dtype).astype(x.dtype)
        return x + self.ffn(self.norm2(x), dtype).astype(x.dtype)


def checkpointed(
    layer: Layer, x: jax.Array, generator: Dense | None, dtype: jnp.dtype
) -> jax.Array:
    """Applies `layer` under `jax.checkpoint`: differentiating it saves only its inputs, and
    computes its activations again in the backward pass."""
    params, static = eqx.partition(layer, eqx.is_array)

    @jax.checkpoint
    def apply(params: Layer, x: jax.Array, generator: Dense | None) -> jax.Array:
        return eqx.combine(params, static)(x, generator, dtype)

    return apply(params, x, generator)


class BoardTransformer(eqx.Module):
    """A transformer with a token per board square (see `BoardTransformerConfig`).

    Its policy is a logit per square from that square's token, and a pass logit from the tokens'
    mean; its value a hidden layer on every square's token, projected to a few channels.

    It has no state, but takes and returns one as the networks with BatchNorm do (see
    `core.networks.utils.apply_nn`): create it with `eqx.nn.make_with_state(BoardTransformer)(
    config, input_shape, key=key)`, which gives an empty state.
    """

    embed: Dense
    position: jax.Array
    layers: list[Layer]
    smolgen_generator: Dense | None
    final_norm: LayerNorm | None
    policy: Dense
    pass_logit: Dense
    value_proj: Dense
    value_hidden: Dense
    value_out: Dense
    # set by `eqx.nn.inference_mode`
    inference: bool
    inference_dtype: str = eqx.field(static=True)
    train_dtype: str = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(
        self,
        config: BoardTransformerConfig,
        input_shape: tuple[int, int, int],
        *,
        key: jax.Array,
    ):
        """
        Args:
            config: network configuration
            input_shape: shape of a single (unbatched) input, (height, width, channels)
            key: rng used to initialize parameters
        """
        height, width, in_channels = input_shape
        squares = height * width
        if config.policy_head_out_size != squares + 1:
            raise ValueError(
                "the policy is a logit per square and one to pass: "
                f"{squares + 1} actions, not {config.policy_head_out_size}"
            )
        keys = jax.random.split(key, config.num_layers + 8)
        dim = config.dim
        self.embed = Dense(in_channels, dim, key=keys[0])
        self.position = 0.02 * jax.random.normal(keys[1], (squares, dim))
        self.layers = [Layer(config, squares, key=k) for k in keys[8:]]
        # smolgen's map from each head's features to its biases, shared by all layers: small, so
        # that the biases start small
        self.smolgen_generator = (
            Dense(config.smolgen_head, squares * squares, 0.1, key=keys[7])
            if config.smolgen
            else None
        )
        self.final_norm = LayerNorm(dim) if config.norm == "pre" else None
        self.policy = Dense(dim, 1, key=keys[2])
        self.pass_logit = Dense(dim, 1, key=keys[3])
        self.value_proj = Dense(dim, config.value_channels, key=keys[4])
        self.value_hidden = Dense(
            squares * config.value_channels, config.value_hidden_size, key=keys[5]
        )
        self.value_out = Dense(config.value_hidden_size, 1, key=keys[6])
        self.inference = False
        self.inference_dtype = config.inference_dtype
        self.train_dtype = config.train_dtype
        self.remat = config.remat

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        dtype = jnp.dtype(self.inference_dtype if self.inference else self.train_dtype)
        # the residual stream is float32 in training, the inference dtype in inference
        stream = dtype if self.inference else jnp.dtype(jnp.float32)
        # a token per square, from its contents and its position
        x = x.reshape(-1, x.shape[-1])
        x = (self.embed(x, dtype) + self.position.astype(dtype)).astype(stream)
        for layer in self.layers:
            if self.remat and not self.inference:
                x = checkpointed(layer, x, self.smolgen_generator, dtype)
            else:
                x = layer(x, self.smolgen_generator, dtype)
        if self.final_norm is not None:
            x = self.final_norm(x)

        f32 = jnp.dtype(jnp.float32)
        squares = self.policy(x, dtype).reshape(-1).astype(f32)
        passing = self.pass_logit(x.astype(f32).mean(axis=0), f32)
        policy = jnp.concatenate([squares, passing])

        value = jax.nn.mish(self.value_proj(x, dtype)).reshape(-1)
        value = jax.nn.mish(self.value_hidden(value, f32))
        value = jnp.tanh(self.value_out(value, f32))
        return (policy, value), state
