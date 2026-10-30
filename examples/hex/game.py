"""Hex pieces shared by the scripts in this directory: the environment at a board size and how
turbozero talks to it, board symmetries, the network and optimizer, evaluators for test games,
cell names, and loading checkpoints.

The board is an N x N rhombus, cells (r, c) numbered row by row, r * N + c; (r, c) neighbours (r, c
± 1), (r ± 1, c) and (r + 1, c - 1), (r - 1, c + 1). Colour 0 moves first and connects the top
and bottom rows, colour 1 the left and right columns. Action N * N is the swap (pie rule), legal
only on the second move: pgx takes the first stone away, and gives the player swapping a stone
at its reflection over the long diagonal, (r, c) -> (c, r). Colour 0 then moves, as after any
move of colour 1: players keep their colours, and what changes is whose the stone is.
"""

from dataclasses import dataclass, replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx.hex

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import BaseExperience
from core.networks.spec import make_network as build_network
from core.networks.spec import network_config
from core.training.train import weight_decay_mask
from core.types import StepMetadata

# AdamW's decay of a weight each step is learning rate * WEIGHT_DECAY * weight, as Othello's
WEIGHT_DECAY = 0.5

# steps down the tree an MCTS traversal takes between checks for its leaf (see
# MCTS.traversal_steps), as Othello's
TRAVERSAL_STEPS = 8


@dataclass(frozen=True)
class SizeDefaults:
    """What train.py runs at a board size unless told otherwise.

    Attributes:
        net: the network, a spec as `core.networks.spec.network_config` takes
        epochs: epochs in a run
    """

    net: str
    epochs: int


# Othello's protocol throughout (1024 games at once, 64 sims a move, 128 moves and 128 training
# steps of 4096 samples an epoch, so 4 passes per position), with the network and the run's length
# scaled with the board: forward passes cost in proportion to the cells, and games get longer
SIZES = {
    7: SizeDefaults("resnet:4x64,pool_every=2,heads=pooled", 50),
    9: SizeDefaults("resnet:6x96,pool_every=2,heads=pooled", 100),
    11: SizeDefaults("resnet:6x128,pool_every=2,heads=pooled", 150),
}


class SizedHex(pgx.hex.Hex):
    """pgx's Hex at any size. pgx's `GameState` holds an 11 x 11 board by default whatever the
    `size` (pgx/_src/games/hex.py), as does its state's legal action mask, and everything else
    takes the `size`: this just gives each a new game's right shape."""

    def _init(self, key):
        state = super()._init(key)
        cells = self.size * self.size
        return state.replace(  # pyright: ignore[reportAttributeAccessIssue]
            _x=state._x._replace(board=jnp.zeros(cells, jnp.int32)),
            legal_action_mask=jnp.ones(cells + 1, jnp.bool_).at[-1].set(False),
        )


def metadata(state) -> StepMetadata:
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


def state_to_nn_input(state):
    """Converts an environment state to the network's input. pgx provides this as
    `state.observation`, with four planes: the player to move's stones, the opponent's, all ones
    if the player to move is colour 1 (which sides it connects), and all ones if it can swap."""
    return state.observation


def make_symmetry_transform_fn(size: int, rotate: bool, reflect: bool):
    """A DataTransformFn that maps a training sample through one of the board's symmetries.
    Training draws one at random for each sample it takes from the replay buffer. Hex has two
    kinds, and the samples' values, which are the player to move's, don't change under either:

    - `rotate`, a half turn, (r, c) -> (N - 1 - r, N - 1 - c): each player's sides map to
      themselves
    - `reflect`, over the long diagonal, (r, c) -> (c, r), with the colours exchanged: colour 0's
      sides map to colour 1's

    The swap action stays where it is under both: the stone a swap places is the reflection of
    the stone it takes, and both symmetries commute with that reflection. A reflected sample
    from early in a game shows what no pgx game can (colour 1 on an empty board, colour 0 able to
    swap); it's the same game with the colours' names exchanged, so it teaches nothing false.
    """
    # idxs[r, c] is the cell that ends up at (r, c)
    idxs = np.arange(size * size).reshape(size, size)
    if reflect:
        idxs = idxs.T
    if rotate:
        idxs = idxs[::-1, ::-1]
    action_ids = np.append(idxs.flatten(), size * size)

    def transform_fn(experience: BaseExperience) -> BaseExperience:
        obs = experience.observation_nn
        if reflect:
            obs = jnp.swapaxes(obs, -3, -2)
            obs = obs.at[..., 2].set(jnp.logical_not(obs[..., 2]))
        if rotate:
            obs = jnp.flip(obs, axis=(-3, -2))
        return replace(
            experience,
            observation_nn=obs,
            policy_mask=experience.policy_mask[..., action_ids],
            policy_weights=experience.policy_weights[..., action_ids],
        )

    return transform_fn


def make_optimizer(
    total_steps: int,
    lr: float = 1e-3,
    lr_final: float = 1e-4,
    weight_decay: float = WEIGHT_DECAY,
) -> optax.GradientTransformation:
    """AdamW, with the learning rate decaying from `lr` to `lr_final` over `total_steps`
    training steps, along a cosine, and weight decay on the network's weights, not its biases or
    BatchNorm scales and offsets (see `weight_decay_mask`). Its hyperparameters are injected, for
    the trainer to record them and log the learning rate."""
    return optax.inject_hyperparams(optax.adamw, static_args=("mask",))(
        learning_rate=optax.cosine_decay_schedule(
            lr, decay_steps=max(total_steps, 1), alpha=lr_final / lr
        ),
        weight_decay=weight_decay,
        mask=weight_decay_mask,
    )


class HexGame:
    """The game at one board size, and what depends on the size.

    Attributes:
        size: cells a side
        env: the pgx environment
        swap: the swap action
        symmetry_transform_fns: every board symmetry but the identity, which training draws from
            along with it
    """

    def __init__(self, size: int = 11):
        self.size = size
        self.env = SizedHex(size=size)
        self.swap = size * size
        self.symmetry_transform_fns = [
            make_symmetry_transform_fn(size, rotate, reflect)
            for rotate, reflect in ((True, False), (False, True), (True, True))
        ]

    # turbozero interfaces with an environment through a step fn (state, action) and an init fn
    # (key), each returning the new state along with the StepMetadata it needs
    def step_fn(self, state, action):
        state = self.env.step(state, action)
        return state, metadata(state)

    def init_fn(self, key):
        state = self.env.init(key)
        return state, metadata(state)

    @property
    def max_moves(self) -> int:
        """The longest a game can be: every cell filled, and a swap."""
        return self.swap + 1

    def cell_name(self, action: int) -> str:
        """The name of an action, as players and GTP write it: "swap", or column letter then
        row number, e.g. 19 at size 9 -> "b3"."""
        if action == self.swap:
            return "swap"
        return "abcdefghijklmnopqrstuvwxyz"[action % self.size] + str(
            action // self.size + 1
        )

    def action(self, name: str) -> int:
        """The action of a cell's name, e.g. "b3" (or "B3") at size 9 -> 19; "swap" for the swap."""
        name = name.strip().lower()
        if name == "swap":
            return self.swap
        return (int(name[1:]) - 1) * self.size + "abcdefghijklmnopqrstuvwxyz".index(
            name[0]
        )

    def reflected(self, action: int) -> int:
        """The cell of an action reflected over the long diagonal: where a swap of a stone on
        that cell puts it."""
        return (action % self.size) * self.size + action // self.size

    def shortest_path_eval(self, obs):
        """A baseline that doesn't use a neural network at all: values a position by how many
        more stones the opponent needs to connect its sides than the player to move does, and
        favours the cells on either's shortest paths. `obs` is a batch of one, as
        `make_nn_eval_fn_no_params_callable` gives it."""
        n = self.size
        me, opp, colour = obs[0, ..., 0], obs[0, ..., 1], obs[0, 0, 0, 2]

        def from_top(cost):
            # the stones it takes to reach each cell from the top row
            def relax(_, dist):
                pad = jnp.pad(dist, 1, constant_values=n * n)
                nearest = jnp.stack(
                    [
                        pad[1:-1, :-2],  # (r, c - 1)
                        pad[1:-1, 2:],  # (r, c + 1)
                        pad[:-2, 1:-1],  # (r - 1, c)
                        pad[2:, 1:-1],  # (r + 1, c)
                        pad[:-2, 2:],  # (r - 1, c + 1)
                        pad[2:, :-2],  # (r + 1, c - 1)
                    ]
                ).min(axis=0)
                return jnp.minimum(dist, nearest + cost)

            start = jnp.full((n, n), n * n).at[0].set(cost[0])
            # a shortest path could take every cell, but winds little on a board this size
            return jax.lax.fori_loop(0, 3 * n, relax, start)

        def paths(mine, theirs, top_bottom):
            # the stones the player needs to connect its sides, and to do so through each cell;
            # only which sides those are differs between the players, so work on a board where
            # they're the top and bottom. A half turn leaves the neighbours as they are
            own = jnp.where(top_bottom, mine, mine.T)
            blocked = jnp.where(top_bottom, theirs, theirs.T)
            cost = jnp.where(own, 0, jnp.where(blocked, n * n, 1))
            top = from_top(cost)
            bottom = from_top(cost[::-1, ::-1])[::-1, ::-1]
            through = top + bottom - cost
            return top[-1].min(), jnp.where(top_bottom, through, through.T)

        my_stones, my_paths = paths(me, opp, ~colour)
        their_stones, their_paths = paths(opp, me, colour)
        value = jnp.clip((their_stones - my_stones) / n, -1, 1)
        # how many stones a cell is off the nearer of the shortest paths, as logits
        off_path = jnp.minimum(my_paths - my_stones, their_paths - their_stones)
        logits = jnp.append(-2.0 * off_path.flatten(), 0.0)
        return logits[None], value[None]

    def make_network(
        self,
        net: str | None = None,
        seed: int = 0,
        inference_dtype: str = "float32",
        train_dtype: str = "float32",
        remat: bool = False,
    ):
        """The network a spec names (see core.networks.spec.network_config), by default the
        board size's (`SIZES`). It's created along with its state (BatchNorm's, if it has any):
        returns (network, state). It computes in `inference_dtype` when it isn't training, e.g.
        in self-play, and in `train_dtype` in training (mixed precision, see AZResnetConfig).
        With `remat`, training computes each block's activations again in the backward pass
        instead of keeping them."""
        config = network_config(
            net or SIZES[self.size].net,
            self.env.num_actions,
            inference_dtype=inference_dtype,
            train_dtype=train_dtype,
            remat=remat,
        )
        # pgx types observation_shape as Tuple[int, ...]; here it's (size, size, 4)
        shape = cast(tuple[int, int, int], self.env.observation_shape)
        return build_network(config, shape, jax.random.PRNGKey(seed))

    def load_checkpoint(
        self, path: str, net: str | None = None, inference_dtype: str = "float32"
    ):
        """Loads the network saved in a training checkpoint (see `train.py`). `net` must name
        the network it was trained with, the run's --net.

        Raises:
            ValueError: if the checkpoint doesn't hold such a network

        Returns:
            (network, (params, network state)): the network, and the parameters and state to
            evaluate it with, as `core.evaluators.evaluation_fns.make_nn_eval_fn` takes them
        """
        net = net or SIZES[self.size].net
        network, network_state = self.make_network(net, inference_dtype=inference_dtype)
        params = eqx.filter(network, eqx.is_inexact_array)
        # a checkpoint is a TrainState's arrays in order, params and network state first: read
        # just those, so that it loads whatever optimizer trained it
        with open(path, "rb") as f:
            try:
                params, network_state = eqx.tree_deserialise_leaves(
                    f, (params, network_state)
                )
            # a mismatch fails wherever the layouts first differ, as a shape error or worse
            except Exception as e:
                raise ValueError(
                    f"{path} doesn't hold a {net} network at size {self.size}: pass the --size "
                    "and --net its run trained with"
                ) from e
        return network, (params, network_state)

    def make_test_evaluator(
        self, eval_fn, num_iterations: int = 64, noise: bool = True
    ) -> MCTS:
        """Evaluator used in test games: temperature 0 to always play the most-visited action.
        Opponents share these settings, so that only the quality of their policy/value
        estimates and their search budget differ.

        With `noise`, AlphaZero's Dirichlet noise is mixed into the root policy, so that two
        deterministic players don't replay the same game; turn it off to measure an evaluator
        at full strength when the games get their variety elsewhere (see `openings.py`)."""
        return AlphaZero(MCTS)(
            eval_fn=eval_fn,
            num_iterations=num_iterations,
            # the tree is kept between moves: room for the subtree a search reuses, and its own
            # iterations on top (as self-play's, see train.py's --tree-factor)
            max_nodes=2 * num_iterations,
            branching_factor=self.env.num_actions,
            action_selector=PUCTSelector(),
            temperature=0.0,
            dirichlet_epsilon=0.25 if noise else 0.0,
            traversal_steps=TRAVERSAL_STEPS,
        )
