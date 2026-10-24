"""Networks named by a short spec, e.g. on the command line (see `network_config`)."""

import dataclasses
import types
import typing
from typing import Any, Literal

import equinox as eqx
import jax

from core.networks.azresnet import AZResnet, AZResnetConfig
from core.networks.transformer import BoardTransformer, BoardTransformerConfig

NetworkConfig = AZResnetConfig | BoardTransformerConfig

# each kind of network: its module, its configuration, and the fields a spec's DEPTHxWIDTH sets
KINDS: dict[str, tuple[type[Any], type[Any], str, str]] = {
    "resnet": (AZResnet, AZResnetConfig, "num_blocks", "num_channels"),
    "transformer": (BoardTransformer, BoardTransformerConfig, "num_layers", "dim"),
}


def _parse(name: str, annotation: Any, text: str) -> Any:
    """`text` as a value of the configuration field `name`, whose type is `annotation`."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        # optional fields, e.g. int | None
        (annotation,) = [a for a in typing.get_args(annotation) if a is not type(None)]
    if typing.get_origin(annotation) is Literal:
        choices = typing.get_args(annotation)
        if text not in choices:
            raise ValueError(f"{name} is one of {', '.join(choices)}, not {text!r}")
        return text
    if annotation is bool:
        if text not in ("true", "false"):
            raise ValueError(f"{name} is true or false, not {text!r}")
        return text == "true"
    return annotation(text)


def network_config(
    spec: str, policy_head_out_size: int, **options: Any
) -> NetworkConfig:
    """The configuration a spec names: KIND:DEPTHxWIDTH, then any of the configuration's fields as
    ,FIELD=VALUE. For example
    - "resnet:10x128": AlphaZero's ResNet (see `AZResnetConfig`), 10 blocks of 128 channels
    - "resnet:10x128,pool_every=3,heads=pooled": with KataGo's global pooling
    - "resnet:12x192,block=bottleneck": of KataGo's bottleneck blocks
    - "transformer:8x256,num_heads=8,ffn_dim=256": a transformer (see `BoardTransformerConfig`),
      8 layers of width 256

    Args:
        spec: the spec
        policy_head_out_size: the number of actions
        options: more fields, e.g. the dtypes, which the spec's override

    Raises:
        ValueError: if the spec isn't one
    """
    kind, _, rest = spec.partition(":")
    if kind not in KINDS:
        raise ValueError(
            f"networks are one of {', '.join(KINDS)}, not {kind!r} ({spec})"
        )
    _, config_type, depth_field, width_field = KINDS[kind]
    size, *settings = rest.split(",")
    try:
        depth, width = (int(n) for n in size.split("x"))
    except ValueError:
        raise ValueError(f"{spec}: the size is DEPTHxWIDTH, e.g. 10x128") from None
    fields = {
        f.name: f
        for f in dataclasses.fields(config_type)
        if f.name not in ("policy_head_out_size", depth_field, width_field)
    }
    hints = typing.get_type_hints(config_type)
    values = dict(options)
    for setting in settings:
        name, _, text = setting.partition("=")
        if name not in fields:
            raise ValueError(
                f"{spec}: {kind} networks have no {name!r}, but {', '.join(fields)}"
            )
        values[name] = _parse(name, hints[name], text)
    return config_type(
        policy_head_out_size=policy_head_out_size,
        **{depth_field: depth, width_field: width},
        **values,
    )


def make_network(
    config: NetworkConfig, input_shape: tuple[int, int, int], key: jax.Array
) -> tuple[eqx.Module, eqx.nn.State]:
    """A network and its state (see `core.networks.utils.apply_nn`), from its configuration."""
    for module, config_type, _, _ in KINDS.values():
        if isinstance(config, config_type):
            return eqx.nn.make_with_state(module)(config, input_shape, key=key)
    raise TypeError(f"not a network configuration: {config!r}")
