# Hex

AlphaZero on Hex at 7 x 7, 9 x 9 and 11 x 11: training, and evaluating the result against the
strong open-source engine MoHex 2.0
([Benzene](https://github.com/alcalyn/benzene-vanilla-cmake)) and, at 11 x 11, pgx's pretrained
model.

| File | |
|---|---|
| `train.py` | Trains a network by self-play, testing it against a random player, a shortest-path evaluator and (at 11 x 11) pgx's model, scoring it against MoHex's scores of fixed positions, and playing MoHex if asked |
| `vs_engine.py` | Plays a checkpoint against MoHex at several levels, from openings (and train.py's network as it trains) |
| `eval_pgx.py` | Plays an 11 x 11 checkpoint against pgx's pretrained model at several search budgets |
| `engines.py` | MoHex, driven over GTP |
| `setup_mohex.sh`, `benzene.patch` | Build MoHex for `vs_engine.py`, without root |
| `engine_set.py`, `label_positions.py` | The engine-labelled set: fixed positions with every legal move scored by MoHex, to build (train.py scores the network on it every epoch) |
| `openings.py`, `make_openings.py`, `openings-<size>.txt` | The opening lists, to start test games from, and to build them |
| `game.py` | What the scripts share: the environment at a size, board symmetries, network, evaluators |

## The game

Actions are the cells, row by row, then the swap, `size * size`. Colour 0 moves first and connects
the top and bottom, colour 1 the left and right; a cell (r, c) neighbours (r, c ± 1), (r ± 1, c),
(r + 1, c - 1) and (r - 1, c + 1). Hex has no draws. A game takes at most `size * size + 1` moves.

pgx's `Hex(size=N)` crashes for N other than 11, as its default state holds an 11 x 11 board.
`game.SizedHex` gives a new game a board and a legal action mask of the right shape; nothing else
in pgx assumes 11.

The swap (pie rule) is legal only on the second move. pgx takes the first stone away and gives the
player swapping one at its reflection over the long diagonal, (r, c) -> (c, r), and colour 0 moves
next, so players keep their colours. MoHex's own swap leaves the stone and exchanges the colours
instead, which is the same position reflected; `engines.py` sets MoHex's board up as pgx's is after
a swap, so the two boards are always the same.

The observation has four planes: the player to move's stones, the opponent's, all ones if the
player to move is colour 1, and all ones if it can swap.

Training maps each sample through one of the board's 4 symmetries, drawn at random: the half turn,
the reflection over the long diagonal with the colours exchanged, and both together.
`tests/test_hex_game.py` plays real games and their images side by side through pgx, swaps
included, and they stay each other's images to the end.

## Training

```
uv run --extra cuda examples/hex/train.py --size 11 --monitor --ckpt-dir runs/hex11 \
    --notes "the baseline" --vs-engine-last 3
uv run --extra cuda examples/hex/train.py --size 7 --monitor --ckpt-dir runs/hex7 \
    --notes "the baseline" --vs-engine-last 3
```

A fresh worktree's jax is CPU-only: run `./install.sh` first. `--name` names the run on the server,
and `--notes` says what it's testing and why, at the top of its page.

Each size's defaults (`game.SIZES`) keep Othello's protocol: 1024 games at once, 64 MCTS iterations
a move, an epoch of 128 moves in each game and 128 training steps of 4096 samples, so 4 passes per
position. The network and the length of the run scale with the board:

| Size | `--net` | `--epochs` |
|---|---|---|
| 7 | `resnet:4x64,pool_every=2,heads=pooled` | 50 |
| 9 | `resnet:6x96,pool_every=2,heads=pooled` | 100 |
| 11 | `resnet:6x128,pool_every=2,heads=pooled` | 150 |

Self-play starts from the empty board, with the swap rule (`--standard-starts` below 1 starts some
games from the openings instead). The other flags are Othello's: `--sims`, `--sims-schedule`,
`--train-steps`, `--train-batch`, `--buffer`, `--lr`, `--inference-dtype`, `--tree-positions`,
`--policy-target`, `--value-target-q` and the rest, and see
[Othello's README](../othello/README.md) for what they do. Resuming and forking work the same:
`--save-state-at`, `--save-state-every`, `--resume`, `--init-from`.

Board size changes game length, which Othello can't: the run's page plots `moves` and `episodes`,
and their ratio is the positions per game. Times the 4 passes per position, that's the views each
game's outcome gets.

Every `--eval-every` epochs (5 by default) it plays 128 test games against each of a random player,
the shortest-path evaluator (`HexGame.shortest_path_eval`: both players' shortest connections,
searching 64 iterations a move, which beat a random player in every game we tried) and at 11 x 11
pgx's pretrained model at 64 iterations, logged as `random_avg_outcome`, `path_avg_outcome`
and `pgx_avg_outcome`. With `--monitor` the run's page draws a game against each on a hex board,
each move's search to explore beside it.

## Openings

Hex's first player wins with perfect play, so a match has to be fair by its construction. Test
games and games against MoHex start from an opening, each played with both colours. A list holds
every single first stone, after which the player to move has the swap, as in a real game, and
two-stone positions that MoHex judges close to even (20,000 playouts, from 0.35 to 0.65, nothing
proven). `make_openings.py` builds them:

```
taskset -c 16-23 uv run examples/hex/make_openings.py --size 7 --workers 8
```

## Evaluating a checkpoint

```
examples/hex/setup_mohex.sh                       # once
uv run --extra cuda examples/hex/vs_engine.py runs/hex11/149.eqx --size 11 --sims 256 --levels 100 1000 10000
uv run --extra cuda examples/hex/eval_pgx.py runs/hex11/149.eqx --pgx-sims 64 256 1024
```

`setup_mohex.sh` builds MoHex and what it needs (Boost, Berkeley DB) from source into
`~/.cache/turbozero/hex/mohex`; it takes minutes, and needs no root. Benzene is old code, and one
function that can run off its end without returning crashes when compiled by a recent GCC:
`benzene.patch` fixes that.

At level L MoHex searches L playouts a move, on one thread, however long they take, so a busy
machine doesn't weaken it. Each process takes about 500 MB, so `vs_engine.py` runs as many as fit
in a quarter of the available memory, on the cores it may use (`taskset`). Our agent searches at
temperature 0, without root noise. Each result is checked against MoHex's own board.
`--monitor --run ID` logs a checkpoint's results to its training run.

During training, `--vs-engine-every N` plays after every N epochs and `--vs-engine-last K` after
each of the last K, by default 256 games (128 openings) at level 1000 and 256 iterations a move,
logged as `mohex_level1000_sims256` with `_se`, `_win` and `_loss`, a game to step through as
`mohex_level1000_sims256_game`, and the mean of the last K as `_last<K>`. `mohex_seconds` is
the time they took, to take out of the run's wall time.

### Against MoHex's scores of fixed positions

Without search or games: thousands of fixed positions from MoHex's games against itself, each legal
move of which MoHex has scored. `train.py` scores the raw network on them every epoch, as Othello
does (`--engine-set`, `--engine-eval-every`), logging for all positions and for each phase (a
share of the board's cells filled: `opening` below 20%, `midgame` to 40%, `endgame` above) `engine_top1`,
`engine_regret`, `engine_exp_regret`, `engine_value_mse` and `engine_value_sign`.
A score here is a win or loss value, from -1 to 1, not a disc margin, so a regret of 2 is giving
a won game away.

```
taskset -c 16-23 uv run examples/hex/label_positions.py --size 7 --solve --workers 8
taskset -c 16-23 uv run examples/hex/label_positions.py --size 11 --level 100000 --workers 8
```

- At 7 x 7, `--solve` has MoHex's solver find the winning moves exactly, in seconds a position. A
  move it doesn't consider (dead or dominated) scores as a loss, though it may win where a better
  one does, so `engine_top1` and the regrets are a little pessimistic.
- At 9 x 9 and 11 x 11, `--level` has its search value each move instead: estimates, not truth.

Sets go to `~/.cache/turbozero/hex/engine-sets/mohex-<size>.npz`; without one, training warns and goes
on unscored.

## What to expect

**No full training run has been done at any size.** The defaults above and the training commands
are untested beyond short smoke runs, so there are no numbers yet for how strong a full run gets,
or how long it takes.

What has been run, on one RTX 5080, from the commands above cut short:

| Size | Epochs | Time, all included | Loss | Against random, 128 games | Against MoHex level 1000 |
|---|---|---|---|---|---|
| 7 | 16 | 3 minutes | 3.06 to 1.67 | 1.00 | 0.03 (32 games, 64 sims) |
| 11 | 3 | 5 minutes | 3.00 to 2.18 | 0.92 | 0.00 (16 games, 32 sims) |

These show the commands work and the network starts to learn, not its strength. At 7 x 7, in those
16 epochs, it went from losing nearly every game to the shortest-path evaluator to winning all 128,
and `engine_top1` on the exactly solved set rose from 0.57 to 0.78. At 9 x 9, only a tiny network on
the CPU has been run, to check that the pieces fit together.

## Credits

MoHex and Benzene are by the Hex group at the University of Alberta, under the LGPL; pgx's model is
from [pgx](https://github.com/sotetsuk/pgx). The setup script builds them from their repositories;
they aren't part of turbozero. Jones 2021, ["Scaling Scaling Laws with Board
Games"](https://arxiv.org/abs/2104.03113), ran AlphaZero on Hex with MoHex as its anchor.
