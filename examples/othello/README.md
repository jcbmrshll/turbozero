# Othello

AlphaZero on Othello: training, and evaluating the result against pgx's pretrained
Othello model and against [Edax](https://github.com/abulmo/edax-reversi), a strong
open-source Othello engine.

| File | |
|---|---|
| `train.py` | Trains a network by self-play, testing it on a ladder of opponents as it goes |
| `eval_pgx.py` | Plays a checkpoint against pgx's pretrained model at several search budgets |
| `vs_edax.py` | Plays a checkpoint against Edax at several search depths, from XOT openings |
| `setup_edax.sh` | Builds Edax for `vs_edax.py` |
| `game.py` | What the scripts share: the environment, board symmetries, network, evaluators |
| `xot.py`, `xot-openings.txt` | The XOT opening list |

## Training

```
uv run turbozero-monitor                            # optional: a dashboard, in another shell
uv run examples/othello/train.py --monitor --ckpt-dir runs/othello
```

Self-play runs in 1024 games at once, searching 64 MCTS iterations a move, and each
position is stored with its 7 symmetric copies (rotations and reflections). Every 5
epochs, the network plays a ladder of opponents, easiest first: a random player, a
greedy tile counter, then pgx's pretrained model searching 1, 4, 16, 64 and 256
iterations a move against our 64. It moves up past each rung it scores at least 0.55
against (a draw counts half). See `--help` for the network size, search budget and
number of epochs.

On one RTX 5080, with the defaults (a 6-block, 128-channel network), an epoch takes
about 70 seconds and a 200-epoch run about 4 hours. One such run passed every rung of
the ladder by epoch 65.

## Evaluating a checkpoint

Checkpoints from `train.py` are named after their epoch (`199.eqx` is the last of a
200-epoch run). Pass the network size if it isn't the default.

```
uv run examples/othello/eval_pgx.py runs/othello/199.eqx --pgx-sims 64 256 1024

examples/othello/setup_edax.sh                      # once
uv run examples/othello/vs_edax.py runs/othello/199.eqx --sims 400 --levels 4 6 8 10
```

`vs_edax.py` plays Edax with its opening book off, at a fixed search depth ("level")
and one thread per game, all games against a level at once. Edax and our agent (at
temperature 0, without root noise) are both deterministic, so games start from **XOT
openings**: a list of 10,784 eight-move openings that each end in a position Edax judges
nearly even. Each opening is played twice, with the colors swapped. This is how OLIVAW,
an AlphaZero-style Othello program, was compared with Edax
([Norelli & Panconesi, 2022](https://arxiv.org/abs/2103.17228)).

The run above scored, out of 256 games against each depth (128 openings), searching
400 MCTS iterations a move:

| Edax depth | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Score | 0.79 | 0.49 | 0.23 | 0.07 |

## Credits

`xot-openings.txt` is Matthias Berg's large XOT list
([berg.earthlingz.de/xot](https://berg.earthlingz.de/xot/)), as also distributed with
Egaroucid, NBoard and othello-sensei. Edax is Richard Delorme's, under the GPL;
`setup_edax.sh` builds it from its repository, it isn't part of turbozero.
