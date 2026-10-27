# Othello

AlphaZero on Othello: training, and evaluating the result against pgx's pretrained
Othello model and against two strong open-source Othello engines,
[Edax](https://github.com/abulmo/edax-reversi) and
[Egaroucid](https://github.com/Nyanyan/Egaroucid).

| File | |
|---|---|
| `train.py` | Trains a network by self-play, testing it on a ladder of opponents as it goes |
| `eval_pgx.py` | Plays a checkpoint against pgx's pretrained model at several search budgets |
| `vs_engine.py` | Plays a checkpoint against Edax or Egaroucid at several levels, from XOT openings |
| `engines.py` | Edax and Egaroucid, each driven over GTP, and what their searches made of each position, from their logs |
| `setup_edax.sh`, `setup_egaroucid.sh` | Build the engines for `vs_engine.py` |
| `bench_nets.py` | Times networks on their own: a forward pass at self-play's batch, a training step at training's |
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
off, and `watch.py`, below, plays an engine as it trains). The monitor also plots
`episode_draw`, the fraction of self-play games drawn, which `game.py`'s
`episode_metrics` reports through the Trainer's `episode_metrics_fn`. See `--help` for
the network size, search budget and number of epochs.

The self-play search budget can grow over a run: `--sims-schedule 0:32,50:64,150:128`
searches 32 iterations a move from epoch 0, 64 from epoch 50 and 128 from epoch 150,
instead of `--sims` throughout. OLIVAW did this in three stages, doubling each time:
100 iterations a move, 200 from about generation 4 and 400 from about generation 11,
of 20 (`0:100,4:200,11:400`, counting its generations as epochs).
Cheap searches are enough while the network learns the basics, and deeper ones refine
its policy later. Each change compiles self-play again, as the first epoch does, so
keep changes few; games in progress carry on, searching from new, empty trees. Test
games, `eval_pgx.py`, `vs_engine.py` and `watch.py` keep their own budgets. The
monitor plots the budget as `search_iterations`.

`--buffer` sets replay memory, the samples kept per environment, which training samples
from. Self-play adds 1024 an epoch (128 moves, each with its 7 symmetric copies), so the
default 3000 is about the last 3 epochs. The monitor plots the samples training can
sample as `buffer_samples`.

The buffer's size sets how old the data training samples is, not how often each sample
is trained on: on average, that's the samples training takes each epoch over the samples
self-play adds, whatever the size. With the defaults, 128 steps of 4096 samples
against 1024 environments adding 1024 samples each, it's 0.5. `--train-steps` changes
it (or `--train-batch`).

On one RTX 5080, with the defaults (a 6-block, 128-channel network), an epoch takes
about 63 seconds. One 200-epoch run (with epochs then taking about 70 seconds) passed
every rung of the ladder by epoch 65. With `--inference-dtype bfloat16` the network
computes in bfloat16 in self-play and test games (it still trains in float32), and an
epoch takes about 36 seconds; searching with the same checkpoint, bfloat16 and float32
play evenly.

### The network

`--net` names the network as KIND:DEPTHxWIDTH, then any of its configuration's fields
as ,FIELD=VALUE (see `core/networks/spec.py`). The default, `resnet:6x128`, is the
residual network from the AlphaZero paper with 6 blocks of 128 channels. Its variants
(`AZResnetConfig`):

- `heads=pooled`: KataGo's policy and value heads, which pool features over the whole
  board (their mean and maximum) instead of flattening a 1-channel map.
- `pool_every=N`: KataGo's global pooling bias in every Nth block: some of the
  block's channels, pooled over the board, bias the rest. Othello's flips run along
  whole lines, and this gives every square board-wide context for almost no compute.
- `block=bottleneck`: KataGo's nested bottleneck blocks, a 1x1 convolution down to
  half the width, a basic block (`bottleneck_inner` of them) and a 1x1 convolution
  back up, which cost under a third of a basic block.

`transformer:LAYERSxWIDTH` is a transformer with a token per square, after Leela Chess
Zero's (`BoardTransformerConfig`): learned attention biases, optionally smolgen. On
the GPU it is much slower per FLOP than the convolutions: `transformer:8x256` takes
1.4 times the FLOPs of `resnet:10x128` and 2.9 times the forward time, and needs
`--remat` to train at the default batch.

`bench_nets.py` times networks as named here:

```
uv run examples/othello/bench_nets.py resnet:10x128 resnet:10x128,pool_every=3,heads=pooled
```

### Training memory

Most of training's GPU memory goes to the activations its backward pass keeps, which grow
with the network and the batch: a 10-block, 128-channel network fits on a 16 GB card at
the default batch of 4096 (about 12 GB at its peak, replay memory included), but 15 × 128
and 10 × 192 run out of memory. Halving `--train-batch` (and doubling `--train-steps`)
fits, but it changes training too. Two options cut the memory and keep the batch:

- `--remat` keeps only each residual block's input for the backward pass, and computes
  the activations inside the block again from it (gradient checkpointing). It costs an
  extra forward pass through the blocks, roughly a third more training compute, and
  gives the same training, up to float rounding.
- `--grad-accum M` splits each batch into M micro-batches, computes their gradients one
  after another, and makes one update with their mean: activations take 1/M the memory,
  for the same compute (if smaller convolutions keep the GPU as busy). The update is the
  whole batch's, up to float rounding, except for BatchNorm: each micro-batch is
  normalized by its own mean and variance, as if training on batches of 4096 / M.
  BatchNorm's running statistics, which self-play uses, are updated once per step, by
  the mean of the micro-batches' updates: their running mean follows the whole batch's
  mean, and their running variance the micro-batches' mean variance (a little smaller
  than the whole batch's). Micro-batch i takes samples i, i + M, i + 2M, ... of the
  batch, so each holds its share of tree positions (`--tree-positions`). M must divide
  `--train-batch`.

Both apply to training alone (self-play and test games are unchanged), and can be
combined. Compiled for the CPU, a training step at batch 4096 needs this much temporary
memory (`memory_analysis()` of the compiled step; the GPU's numbers differ, but scale
alike):

| Network | Neither | `--remat` | `--grad-accum 2` | `--grad-accum 4` | `--remat --grad-accum 2` |
|---|---|---|---|---|---|
| 10 × 128 | 10.8 GiB | 3.1 GiB | 5.4 GiB | 2.7 GiB | 1.5 GiB |
| 15 × 128 | 15.8 GiB | 4.4 GiB | 7.9 GiB | 3.9 GiB | 2.1 GiB |
| 10 × 192 | 16.1 GiB | 4.7 GiB | 8.1 GiB | 4.0 GiB | 2.3 GiB |

Try `--remat` first, since it doesn't change training; `--grad-accum` is for
networks too big even with it.

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
  too. They make up a share of each training batch: `--tree-ratio` tree positions per
  played position, 1 (half the batch) by default, halving every `--tree-half-life` epochs
  if given. Self-play reuses the played move's subtree in its next search, so a node along
  the expected line can be stored by several searches in a row, and is often played soon
  after; `--tree-discarded-only` only stores nodes outside that subtree, so each position
  is stored at most once, by the last search it's in.

Tree positions follow replay memory (`--buffer`): training samples those stored in the
moves its played positions come from (3000 samples is the last 375 moves, at 8 samples a
move), however many positions each move stored. They're kept in a replay buffer of their
own, by default K times `--buffer`, which holds that span even when every move stores K;
`--tree-buffer` sets its size instead (a move stores fewer than K when too few nodes qualify, and about 1.3 of 2
with `--tree-discarded-only`, so a smaller buffer can still hold the whole span).

Each K adds to GPU memory what raising `--buffer` by as much again would: 1.45 GB per K at
`--buffer 3000`. Training holds two copies of the replay memory (the epoch's, and the one
before it, for the self-play metrics), and preallocates 75% of the GPU, 12 GB of a 16 GB
card. At `--buffer 3000`, K = 1 fits (it's the memory of `--buffer 6000`, which runs);
K = 2 is that of `--buffer 9000`, which hasn't been tried. Keeping the buffer in host
memory would make every training step copy its minibatch over; for a larger K, give
`--tree-buffer` less than K times `--buffer` instead.

The monitor shows, each epoch, the tree positions stored (`tree_positions`, and
`tree_positions_per_move` out of K), their mean visit count (`tree_mean_visits`), how many
training could sample (`tree_buffer_samples`), and the share of each training batch they
made up (`tree_batch_fraction`).

### Resuming and forking runs

Checkpoints hold only the network and optimizer state. `--save-state-at 50,150` also
saves the whole training state once 50 and 150 epochs are done, as `state-50.npz` and
`state-150.npz` in `--ckpt-dir`: the network, optimizer state, replay buffers, games in
progress with their search trees, the ladder's rung and the rng. `--save-state-every 10`
saves one every 10 epochs and keeps only the newest of these. A state takes about as much
disk space as replay memory takes GPU memory, about 1.45 GB per 3000 samples per
environment, and K times that again with `--tree-positions K`.

`--resume` continues the run in `--ckpt-dir` from its newest state; pass the run's own
arguments. It carries on exactly as if it hadn't stopped (on the CPU the result is
bit-identical; on a GPU, convolution autotuning may differ between processes). It logs
to the same monitor run, which then holds the epochs between the state and the stop
twice, and deletes the checkpoints saved after the state, which it saves again.

```
uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long
uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long --resume
```

`--init-from` forks a new run from a state (a `state-<epoch>.npz`, or a directory to take
the newest from). The fork is a run of its own: it starts at epoch 0, with its own
`--epochs`, seed and schedules (`--lr` to `--lr-final` and `--sims-schedule`, both
counted from its epoch 0), new games and search trees, and its
own `--ckpt-dir`. It starts from the saved network, Adam's moments, and the replay
buffers, so its first epochs train on as much data as the run it was forked from did
(the entries of the games in progress are dropped). `--init-without-buffers` starts with
empty buffers instead. The network size must match. The replay memory may differ in
size: each environment's newest samples are kept, as many as fit. The monitor's config
records the parent state as `parent`, and so does every state the fork saves.

This makes it cheap to compare settings late in training. Short runs from scratch only
show which settings learn fastest early on. Instead, train one long trunk at a constant
learning rate (`--lr-final` equal to `--lr`), save its state, and fork short runs from it
that differ in one setting each, alongside a control fork with the trunk's settings:

```
uv run examples/othello/train.py --epochs 150 --lr 1e-3 --lr-final 1e-3 \
    --save-state-at 150 --eval-every 0 --ckpt-dir runs/trunk
uv run examples/othello/train.py --epochs 30 --init-from runs/trunk/state-150.npz \
    --eval-every 0 --ckpt-dir runs/trunk150-control
uv run examples/othello/train.py --epochs 30 --init-from runs/trunk/state-150.npz \
    --sims 128 --eval-every 0 --ckpt-dir runs/trunk150-sims128
```

## Evaluating a checkpoint

Checkpoints from `train.py` are named after their epoch (`199.eqx` is the last of a
200-epoch run). Pass the run's `--net` if it isn't the default.

```
uv run examples/othello/eval_pgx.py runs/othello/199.eqx --pgx-sims 64 256 1024

examples/othello/setup_edax.sh                      # once
examples/othello/setup_egaroucid.sh                 # once
uv run examples/othello/vs_engine.py runs/othello/199.eqx --engine edax --sims 400 --levels 4 6 8 10
uv run examples/othello/vs_engine.py runs/othello/199.eqx --engine egaroucid --levels 2 4 6
```

`vs_engine.py` plays the engine with its opening book off and one thread per game, at a
fixed level: at level L (up to 10) both engines search L moves ahead, and solve the game
exactly once 2L or fewer squares are empty. It plays many games at once, each with its own
engine process (an Edax process takes about 170 MB and an Egaroucid one about 1.3 GB, so it
runs as many as fit in a quarter of the available memory, up to 128).
The engines and our agent (at temperature 0, without root noise) are all deterministic,
so games start from **XOT openings**: a list of 10,784 eight-move openings that each end
in a position Edax judges nearly even (the same ones for every checkpoint, given the
same `--seed`). Each opening is played twice, with the colors swapped. This is how
OLIVAW, an AlphaZero-style Othello program, was compared with Edax
([Norelli & Panconesi, 2022](https://arxiv.org/abs/2103.17228)).

The run above scored, out of 256 games against each level (128 openings), searching
400 MCTS iterations a move:

| Edax level | 4 | 6 | 8 | 10 |
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
in the run's monitor link) to the run's monitor page, as `edax_level4_sims64` and so on,
each with a game at that level, `edax_level4_sims64_game`, to step through: our agent's search
of each position, and the engine's scores (its predicted disc margin / 64): Edax's of every
legal move of every position, each with the line it expects after it (from its own search
where it moved, and from a separate Edax process elsewhere, so as not to change its moves),
Egaroucid's of the moves it played.
Restarted, it evaluates only the levels a checkpoint has no result for yet, at these `--sims`
and `--openings`. Our agent searches on the GPU next to training, taking only the memory it
needs: with a 10-block network, 256 games against each of 3 levels take about 2 minutes and
slow training's epochs by about half while they run.

## Credits

`xot-openings.txt` is Matthias Berg's large XOT list
([berg.earthlingz.de/xot](https://berg.earthlingz.de/xot/)), as also distributed with
Egaroucid, NBoard and othello-sensei. Edax (Richard Delorme) and Egaroucid (Takuto
Yamana) are under the GPL; the setup scripts build them from their repositories, they
aren't part of turbozero.
