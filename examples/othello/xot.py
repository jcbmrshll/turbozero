"""XOT openings: 10,784 eight-move Othello openings, each ending in a position Edax
(searching 16 moves ahead) judges nearly even, within 2 discs.

Starting games from them instead of the initial position gives deterministic players
varied, balanced games: the usual way to test Othello programs against each other (e.g.
OLIVAW against Edax, https://arxiv.org/abs/2103.17228), each opening played twice with
the colors swapped.

The list, `xot-openings.txt`, is Matthias Berg's "large" XOT list
(https://berg.earthlingz.de/xot/), as used by Egaroucid, NBoard and others.
"""

from pathlib import Path

import numpy as np
from game import action

XOT_PATH = Path(__file__).parent / "xot-openings.txt"


def load_xot(path: str | Path = XOT_PATH) -> np.ndarray:
    """The XOT openings as actions, one row of 8 per opening (black moves first).

    Args:
        path: (optional) an opening list, one opening a line in square names, e.g.
            "f5d6c4d3c2b3b4b5"
    """
    lines = Path(path).read_text().split()
    return np.array(
        [[action(line[i : i + 2]) for i in range(0, len(line), 2)] for line in lines],
        dtype=np.int32,
    )
