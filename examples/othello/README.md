# Othello

AlphaZero on Othello: training, and evaluating the result against pgx's pretrained
Othello model and against two strong open-source Othello engines,
[Edax](https://github.com/abulmo/edax-reversi) and
[Egaroucid](https://github.com/Nyanyan/Egaroucid).

| File | |
|---|---|
| `train.py` | Trains a network by self-play, testing it on a ladder of opponents as it goes |
| `eval_pgx.py` | Plays a checkpoint against pgx's pretrained model at several search budgets |
| `vs_engine.py` | Plays a checkpoint against Edax or Egaroucid at several search depths, from XOT openings |
| `engines.py` | Edax and Egaroucid, each driven over GTP |
| `setup_edax.sh`, `setup_egaroucid.sh` | Build the engines for `vs_engine.py` |
| `game.py` | What the scripts share: the environment, board symmetries, network, evaluators |
| `xot.py`, `xot-openings.txt` | The XOT opening list |

## Training

```
uv run turbozero-monitor                            # optional: a dashboard, in another shell
uv run examples/othello/train.py --monitor --ckpt-dir runs/othello
```

Self-play runs in 1024 games at once, searching 64 MCTS iterations a move, and each
position is stored with its 7 symmetric copies (rotations and reflections). Games start
from XOT openings rather than the standard start, which keeps self-play varied; every
test game starts from one too (`--standard-starts` mixes in the standard start). Every
5 epochs, the network plays a ladder of opponents, easiest first: a random player, a
greedy tile counter, then pgx's pretrained model searching 1, 4, 16, 64 and 256
iterations a move against our 64. It moves up past each rung it scores at least 0.55
against (a draw counts half). A 10-block network passes every rung within about 60
epochs; past that, play it against Edax instead (`--eval-every 0` turns the ladder
off, and `watch.py`, below, plays an engine as it trains). See `--help` for the
network size, search budget and number of epochs.

On one RTX 5080, with the defaults (a 6-block, 128-channel network), an epoch takes
about 63 seconds. One 200-epoch run (with epochs then taking about 70 seconds) passed
every rung of the ladder by epoch 65. With `--inference-dtype bfloat16` the network
computes in bfloat16 in self-play and test games (it still trains in float32), and an
epoch takes about 36 seconds; searching with the same checkpoint, bfloat16 and float32
play evenly.

### Value targets from the search

By default the network learns each played position's value from the game's outcome z.
Early in a game z says little about the position, while the search's value q is a
better assessment, if limited by the search's horizon. Two options use q instead, after
OLIVAW (Norelli & Panconesi, section IV-B) and "Lessons from implementing AlphaZero"
(Young, Prasad & Abrams):

- `--value-target-q W` trains played positions on `(1 - W)·z + W·q`, with q the root
  value of the search that chose the move (for the player to move).
- `--tree-positions K` also trains on up to K positions from each self-play search tree:
  nodes visited at least `--tree-min-visits` times, sampled in proportion to their visits
  (`--tree-select most-visited` takes the most visited, as OLIVAW did). Each trains on
  the visit distribution over its children and its q, and gets the 7 symmetric copies
  too. They're kept in a replay buffer of their own (`--tree-buffer`, by default K times
  `--buffer`, about 1.4 GB per K on the GPU), and make up a share of each training
  batch: `--tree-ratio` tree positions per played position, 1 (half the batch) by
  default, halving every `--tree-half-life` epochs if given. Self-play reuses the
  played move's subtree in its next search, so a node along the expected line can be
  stored by several searches in a row; `--tree-discarded-only` only stores nodes outside
  that subtree, so each position is stored at most once, by the last search it's in.

The monitor shows, each epoch, the tree positions stored (`tree_positions`, and
`tree_positions_per_move` out of K), their mean visit count (`tree_mean_visits`), and the
share of each training batch they made up (`tree_batch_fraction`).

## Evaluating a checkpoint

Checkpoints from `train.py` are named after their epoch (`199.eqx` is the last of a
200-epoch run). Pass the network size if it isn't the default.

```
uv run examples/othello/eval_pgx.py runs/othello/199.eqx --pgx-sims 64 256 1024

examples/othello/setup_edax.sh                      # once
examples/othello/setup_egaroucid.sh                 # once
uv run examples/othello/vs_engine.py runs/othello/199.eqx --engine edax --sims 400 --levels 4 6 8 10
uv run examples/othello/vs_engine.py runs/othello/199.eqx --engine egaroucid --levels 2 4 6
```

`vs_engine.py` plays the engine with its opening book off, at a fixed search depth
("level") and one thread per game, with many games at once: each has its own engine
process (an Edax process takes about 170 MB and an Egaroucid one about 1.3 GB, so it
runs as many as fit in a quarter of the available memory, up to 128).
The engines and our agent (at temperature 0, without root noise) are all deterministic,
so games start from **XOT openings**: a list of 10,784 eight-move openings that each end
in a position Edax judges nearly even (the same ones for every checkpoint, given the
same `--seed`). Each opening is played twice, with the colors swapped. This is how
OLIVAW, an AlphaZero-style Othello program, was compared with Edax
([Norelli & Panconesi, 2022](https://arxiv.org/abs/2103.17228)).

The run above scored, out of 256 games against each depth (128 openings), searching
400 MCTS iterations a move:

| Edax depth | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Score | 0.79 | 0.49 | 0.23 | 0.07 |

### During training

`train.py` keeps only its 2 newest checkpoints; `--keep-every 25` also keeps every 25th.
`watch.py` plays each one against an engine as it's saved (and the last one when
training ends), so a long run's strength shows before it ends:

```
uv run examples/othello/train.py --keep-every 25 --ckpt-dir runs/othello --monitor &
uv run examples/othello/watch.py runs/othello --pid $! --levels 4 6 8
```

Results go to `runs/othello/vs_engine.jsonl`, and with `--monitor URL --run ID` (the id
in the run's monitor link) to the run's monitor page. Our agent searches on the GPU next
to training, taking only the memory it needs: with a 10-block network, 256 games against
each of 3 levels take about 2 minutes and slow training's epochs by about half while
they run.

## Credits

`xot-openings.txt` is Matthias Berg's large XOT list
([berg.earthlingz.de/xot](https://berg.earthlingz.de/xot/)), as also distributed with
Egaroucid, NBoard and othello-sensei. Edax (Richard Delorme) and Egaroucid (Takuto
Yamana) are under the GPL; the setup scripts build them from their repositories, they
aren't part of turbozero.
