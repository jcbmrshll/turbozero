import sys
from pathlib import Path

import equinox as eqx
import jax
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from game import load_checkpoint, make_network, make_optimizer


def save_checkpoint(path: Path, net: str):
    """A checkpoint as the trainer saves one: params and network state first, then the
    optimizer's state."""
    network, network_state = make_network(net)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = make_optimizer(total_steps=1).init(params)
    eqx.tree_serialise_leaves(path, (params, network_state, opt_state))
    return params


def test_load_checkpoint_loads_its_net(tmp_path):
    path = tmp_path / "0.eqx"
    saved = save_checkpoint(path, "resnet:1x4")
    _, (params, _) = load_checkpoint(str(path), "resnet:1x4")
    assert jax.tree.all(jax.tree.map(lambda a, b: (a == b).all(), params, saved))


@pytest.mark.parametrize(
    "trained, net",
    [
        ("resnet:1x4", "resnet:1x4,heads=pooled"),
        ("resnet:1x4", "resnet:1x4,pool_every=1,pool_channels=2"),
        ("resnet:1x4,pool_every=1,pool_channels=2,heads=pooled", "resnet:1x4"),
    ],
)
def test_load_checkpoint_names_the_net_on_mismatch(tmp_path, trained, net):
    path = tmp_path / "0.eqx"
    save_checkpoint(path, trained)
    with pytest.raises(ValueError, match=f"doesn't hold a {net} network.*--net"):
        load_checkpoint(str(path), net)
