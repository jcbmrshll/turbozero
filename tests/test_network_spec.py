import jax
import pytest

from core.networks.azresnet import AZResnet, AZResnetConfig
from core.networks.spec import make_network, network_config
from core.networks.transformer import BoardTransformer, BoardTransformerConfig


def test_a_spec_names_a_network_its_size_and_its_fields():
    config = network_config(
        "resnet:12x96,block=bottleneck,bottleneck_inner=2,heads=pooled", 65
    )

    assert config == AZResnetConfig(
        policy_head_out_size=65,
        num_blocks=12,
        num_channels=96,
        block="bottleneck",
        bottleneck_inner=2,
        heads="pooled",
    )


def test_the_spec_overrides_the_options_it_names():
    config = network_config(
        "transformer:4x64,attention_bias=false,train_dtype=float32",
        65,
        inference_dtype="bfloat16",
        train_dtype="bfloat16",
    )

    assert config == BoardTransformerConfig(
        policy_head_out_size=65,
        num_layers=4,
        dim=64,
        attention_bias=False,
        inference_dtype="bfloat16",
        train_dtype="float32",
    )


@pytest.mark.parametrize(
    "spec, error",
    [
        ("mlp:2x8", "networks are one of resnet, transformer"),
        ("resnet:10", "the size is DEPTHxWIDTH"),
        ("resnet:10x128,depth=3", "resnet networks have no 'depth'"),
        ("resnet:10x128,heads=katago", "heads is one of alphazero, pooled"),
        ("transformer:2x8,attention_bias=1", "attention_bias is true or false"),
        ("resnet:10x128,pool_every=x", "invalid literal for int"),
    ],
)
def test_specs_that_name_no_network_are_refused(spec, error):
    with pytest.raises(ValueError, match=error):
        network_config(spec, 65)


@pytest.mark.parametrize(
    "spec, module",
    [("resnet:1x4", AZResnet), ("transformer:1x4,num_heads=2", BoardTransformer)],
)
def test_make_network_builds_the_configured_module(spec, module):
    network, _ = make_network(
        network_config(spec, 10), (3, 3, 2), jax.random.PRNGKey(0)
    )

    assert isinstance(network, module)
