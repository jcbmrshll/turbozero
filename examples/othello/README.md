# Othello

AlphaZero on Othello: training, and evaluating the result against pgx's pretrained
Othello model and against two strong open-source Othello engines,
[Edax](https://github.com/abulmo/edax-reversi) and
[Egaroucid](https://github.com/Nyanyan/Egaroucid).

| File | |
|---|---|
| `train.py` | Trains a network by self-play, scoring it against Edax's scores of fixed positions as it goes, and in games against Edax if asked |
| `eval_pgx.py` | Plays a checkpoint against pgx's pretrained model at several search budgets |
| `vs_engine.py` | Plays a checkpoint against Edax or Egaroucid at several levels, from XOT openings (and train.py's network as it trains) |
| `engines.py` | Edax and Egaroucid, each driven over GTP, and what their searches made of each position, from their logs |
| `setup_edax.sh`, `setup_egaroucid.sh` | Build the engines for `vs_engine.py` |
| `engine_set.py`, `label_positions.py`, `score_engine_set.py` | The engine-labelled set: fixed positions with every legal move scored by Edax, to build, and to score checkpoints on (train.py scores the network on it every epoch) |
| `bench_nets.py` | Times networks on their own: a forward pass at self-play's batch, a training step at training's |
| `game.py` | What the scripts share: the environment, board symmetries, network, evaluators |
| `xot.py`, `xot-openings.txt` | The XOT opening list |

## Training

```
uv run examples/othello/train.py --monitor --ckpt-dir runs/othello --notes "the baseline" \
    --vs-engine-last 3   # Edax games after each of the last 3 epochs
```

`--name` names the run on the server, and `--notes` says what it's testing and why, at the
top of its page.

Self-play runs in 1024 games at once, searching 64 MCTS iterations a move. Replay memory
stores each position once, and training maps each position it samples through one of the
board's 8 symmetries (rotations and reflections), drawn at random. Games start
from XOT openings rather than the standard start, which keeps self-play varied; every
test game starts from one too (`--standard-starts` mixes in the standard start).

A run is tracked by the engine-labelled set: every epoch, `train.py` scores the raw network
on Edax's scores of 5000 fixed positions, in about 20 ms, and logs `engine_exp_regret` (the
discs its policy gives up against Edax's best move) and the rest of the `engine_*` metrics
(see [below](#against-edaxs-scores-of-fixed-positions)). They're much finer than games: a
difference between two checkpoints is good to about ±0.015 discs. Games against Edax
measure strength itself, with `--vs-engine-every` and `--vs-engine-last` (see
[During training](#during-training)).

`--eval-every N` also plays test games every N epochs, on a ladder of opponents, easiest
first: a random player, a greedy tile counter, then pgx's pretrained model searching 1, 4,
16, 64 and 256 iterations a move against our 64. It moves up past each rung it scores at
least 0.55 against (a draw counts half). A 10-block network passes every rung within about
60 epochs. Each test plays 128 games a rung it tries, so it costs real time; it's off by
default.

The run's page also plots `episode_draw`, the fraction of self-play games drawn, which
`game.py`'s `episode_metrics` reports through the Trainer's `episode_metrics_fn`. See `--help` for
the network size, search budget and number of epochs.

The self-play search budget can grow over a run: `--sims-schedule 0:32,50:64,150:128`
searches 32 iterations a move from epoch 0, 64 from epoch 50 and 128 from epoch 150,
instead of `--sims` throughout. OLIVAW did this in three stages, doubling each time:
100 iterations a move, 200 from about generation 4 and 400 from about generation 11,
of 20 (`0:100,4:200,11:400`, counting its generations as epochs).
Cheap searches are enough while the network learns the basics, and deeper ones refine
its policy later. Each change compiles self-play again, as the first epoch does, so
keep changes few; games in progress carry on, searching from new, empty trees. Test
games, `eval_pgx.py` and the games against engines keep their own budgets. The
run's page plots the budget as `search_iterations`.

Each self-play environment searches at its own pace, a network evaluation a cycle, so a
search needn't take a fixed number of iterations: with `--count-reused-visits` it stops
once its root has `--sims` + 1 visits, those of the subtree it kept from the last move's
search included. A move whose subtree the last search explored well then takes fewer
iterations, about half of them with a trained network, and every root still ends with the
visits of a full search. An epoch is still the compute of 128 full searches in each
environment, so it plays more moves (logged as `moves` and `cycles_per_move`), and fills
the replay buffer faster. `--min-iterations` sets the fewest a search takes, and
`--advance-every` how often finished searches play their moves (5 cycles by default with
`--count-reused-visits`; a search done in between goes on until then).

By default every search starts from the subtree of the move played that the last search
explored, and adds `--sims` iterations to its visits. `--no-persist-tree` starts each from
an empty tree instead: an ablation of that reuse (a target then comes from `--sims` visits
alone, all of them fresh, and the root's noise steers a whole search).

`--traversal-steps` sets how many steps down the tree a search's traversal takes between
checks for having reached its leaf: the search is the same whatever the number, but on GPU
each check waits on the host, and the batch's traversals go on until its deepest is done.
The default, 8, was the fastest of 1, 2, 4, 8 and 16 on one RTX 5080 (1024 games, 64 to 256
sims), self-play 1.2 to 1.25 times as fast as with 1.

`--buffer` sets replay memory, the positions kept per environment, which training samples
from. Self-play adds 128 an epoch (a position a move), so the default 375 is about the last
3 epochs. The run's page plots the positions training can sample as `buffer_samples`, and
how many of them are distinct, a position's symmetric images counting as one, as
`buffer_distinct_positions`.

The buffer's size sets how old the data training samples is, not how often each position
is trained on: on average, that's the samples training takes each epoch over the positions
self-play adds, whatever the size. With the defaults, 128 steps of 4096 samples
against 1024 environments adding 128 positions each, it's 4, each time in a symmetry drawn
at random. `--train-steps` changes it (or `--train-batch`).

The network computes in bfloat16 by default: in self-play and test games
(`--inference-dtype`), and its convolutions in training (`--train-dtype`, mixed precision:
its parameters, BatchNorm and heads stay in float32). On one RTX 5080, with the plain
`resnet:6x128` (below) computing in float32 throughout, an epoch takes about 63 seconds, and
one 200-epoch run (with `--eval-every 5`) passed every rung of the ladder by epoch 65.
bfloat16 in self-play and test games brings an epoch to about 36 seconds; searching with
the same checkpoint, bfloat16 and float32 play evenly. bfloat16 training then takes each training step in less
than half the time (a 10-block, 128-channel network's from 157 to 72 ms), and trained as
well as float32 in a two-seed comparison.

### The network

`--net` names the network as KIND:DEPTHxWIDTH, then any of its configuration's fields
as ,FIELD=VALUE (see `core/networks/spec.py`). `resnet:6x128` is the residual network
from the AlphaZero paper with 6 blocks of 128 channels. Its variants
(`AZResnetConfig`):

- `heads=pooled`: KataGo's policy and value heads, which pool features over the whole
  board (their mean and maximum) instead of flattening a 1-channel map.
- `pool_every=N`: KataGo's global pooling bias in every Nth block: some of the
  block's channels, pooled over the board, bias the rest. Othello's flips run along
  whole lines, and this gives every square board-wide context for almost no compute.
- `block=bottleneck`: KataGo's nested bottleneck blocks, a 1x1 convolution down to
  half the width, a basic block (`bottleneck_inner` of them) and a 1x1 convolution
  back up, which cost under a third of a basic block.

The default, `resnet:6x128,pool_every=2,heads=pooled` (`DEFAULT_NET` in `game.py`), has
both kinds of global pooling. Trained for 50 epochs at 64 sims a move, then searching
1000 iterations a move against Edax level 8 (512 games), `resnet:10x128` with the same
pooling scored 0.73 in two seeds, where the plain `resnet:10x128` scored 0.57 to 0.61 in
three, in the same training time (about 47 minutes). The pooled 6 × 128 scored 0.71, in
64% of that time. On the engine-labelled set (below) the pooled networks' regret was
1.85 to 1.93 discs, the plain ones' 2.07 to 2.12.

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
  the visit distribution over its children and its q, in a random symmetry each time
  training samples it, as played positions are. They make up a share of each training batch: `--tree-ratio` tree positions per
  played position, 1 (half the batch) by default, halving every `--tree-half-life` epochs
  if given. Self-play reuses the played move's subtree in its next search, so a node along
  the expected line can be stored by several searches in a row, and is often played soon
  after; `--tree-discarded-only` only stores nodes outside that subtree, so each position
  is stored at most once, by the last search it's in.

Tree positions follow replay memory (`--buffer`): training samples those stored in the
moves its played positions come from (the last 375 at the default `--buffer`), however many positions each move stored. They're kept in a replay buffer of their
own, by default K times `--buffer`, which holds that span even when every move stores K;
`--tree-buffer` sets its size instead (a move stores fewer than K when too few nodes qualify, and about 1.3 of 2
with `--tree-discarded-only`, so a smaller buffer can still hold the whole span).

Each K adds to GPU memory what raising `--buffer` by as much again would: 0.18 GB per K at
`--buffer 375` (about 480 bytes a position, in each of 1024 environments). Training holds
two copies of the replay memory (the epoch's, and the one before it, for the self-play
metrics), and preallocates 75% of the GPU, 12 GB of a 16 GB card. 6000 positions per
environment, played and tree positions together (2.9 GB), are known to fit. Keeping the
buffer in host memory would make every training step copy its minibatch over; for a much
larger K, give `--tree-buffer` less than K times `--buffer` instead.

The run's page shows, each epoch, the tree positions stored (`tree_positions`, and
`tree_positions_per_move` out of K), their mean visit count (`tree_mean_visits`), how many
training could sample (`tree_buffer_samples`), and the share of each training batch they
made up (`tree_batch_fraction`).

### Policy target from the search

By default the network learns each played position's policy from the search's visit
distribution, as in AlphaZero. `--policy-target completed-q` trains it on Gumbel MuZero's
improved policy instead (Danihelka et al., "Policy improvement by planning with Gumbel",
ICLR 2022): `softmax(logits + σ(completed Q))`, with the network's logits at the root
before the root's Dirichlet noise, and the children's values (unvisited children's
estimated from the visited ones') rescaled to [0, 1] and scaled by
`(--maxvisit-init + the most visited child's visits) · --value-scale`. The noise then
steers only which moves the search explores, not the target. Tree positions keep the
visit distribution, so the two options don't go together.

`policy_loss`, the cross-entropy against those targets, is their entropy (logged as
`policy_target_entropy`), which no network can lower, plus the KL divergence from them to the
network's policy (`policy_kl`). The search sets the first: a deeper search, a kept tree or
the completed-Q target train toward sharper or flatter targets. So `policy_loss` doesn't
compare across runs that differ in those; `policy_kl` says how well each network fits its own
targets, and the engine-set metrics how good its policy is.

### Resuming and forking runs

Checkpoints hold only the network and optimizer state. `--save-state-at 50,150` also
saves the whole training state once 50 and 150 epochs are done, as `state-50.npz` and
`state-150.npz` in `--ckpt-dir`: the network, optimizer state, replay buffers, games in
progress with their search trees, the ladder's rung (with `--eval-every`) and the rng.
`--save-state-every 10` saves one every 10 epochs and keeps only the newest of these. A state takes about as much
disk space as replay memory takes GPU memory, about 0.18 GB per 375 positions per
environment, and K times that again with `--tree-positions K`.

`--resume` continues the run in `--ckpt-dir` from its newest state; pass the run's own
arguments. It carries on exactly as if it hadn't stopped (on the CPU the result is
bit-identical; on a GPU, convolution autotuning may differ between processes). It logs
to the same seaglass run, which then holds the epochs between the state and the stop
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
size: each environment's newest samples are kept, as many as fit. The seaglass run's config
records the parent state as `parent`, and so does every state the fork saves.

This makes it cheap to compare settings late in training. Short runs from scratch only
show which settings learn fastest early on. Instead, train one long trunk at a constant
learning rate (`--lr-final` equal to `--lr`), save its state, and fork short runs from it
that differ in one setting each, alongside a control fork with the trunk's settings:

```
uv run examples/othello/train.py --epochs 150 --lr 1e-3 --lr-final 1e-3 \
    --save-state-at 150 --ckpt-dir runs/trunk
uv run examples/othello/train.py --epochs 30 --init-from runs/trunk/state-150.npz \
    --ckpt-dir runs/trunk150-control
uv run examples/othello/train.py --epochs 30 --init-from runs/trunk/state-150.npz \
    --sims 128 --ckpt-dir runs/trunk150-sims128
```

## Evaluating a checkpoint

Checkpoints from `train.py` are named after their epoch (`199.eqx` is the last of a
200-epoch run). Pass the run's `--net` if it isn't the default: `--net resnet:6x128` for
a run trained before the pooled network became the default.

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

`train.py` tracks a run by the engine-labelled set every epoch (below), and plays it against
Edax when asked to: `--vs-engine-every N` after every N epochs, and `--vs-engine-last K` after
each of the last K epochs. By default it plays 512 games (256 XOT openings) at level 8,
searching 1000 MCTS iterations a move, the same games every time and in every run:

```
uv run examples/othello/train.py --epochs 50 --ckpt-dir runs/othello --monitor --vs-engine-last 3
uv run examples/othello/train.py --ckpt-dir runs/othello --monitor --vs-engine-every 25 \
    --vs-engine-levels 4 6 8 --vs-engine-sims 400
```

`--vs-engine-levels`, `--vs-engine-sims` and `--vs-engine-openings` set the games, and
`--vs-engine egaroucid` plays Egaroucid instead. Each level's result is logged to the run at
the epoch of the network that played (counting from 0, so `--vs-engine-every 25` logs at
epochs 24, 49, ...), as `edax_level8_sims1000`, the score, with its standard error
`edax_level8_sims1000_se` and the fractions of games won, drawn and lost (`_win`, `_draw`,
`_loss`), and a game at that level, `edax_level8_sims1000_game`, to step through: our agent's
search of each position, and the engine's scores (its predicted disc margin / 64): Edax's of
every legal move of every position, each with the line it expects after it (from its own
search where it moved, and from a separate Edax process elsewhere, so as not to change its
moves), Egaroucid's of the moves it played. Results also go to `runs/othello/vs_engine.jsonl`,
a line per level and epoch.

One network's score drifts from its neighbours' by about ±0.035, more than the ±0.02 of
512 games' own noise, so `--vs-engine-last K` also logs the mean of the last K networks'
scores at the last epoch, as `edax_level8_sims1000_last3` (for K = 3), with its standard
error (`_last3_se`), each game's results averaged over the networks first, since they all
play the same games.

The games are played between epochs, in the training process: our agent's search shares
the run's GPU memory and its compiled search, and the engine processes (single-threaded, one
a game) run on the cores the run may use, so `taskset -c 0-7` keeps them on those 8, as many
searching at once as there are cores. Training waits for them: each evaluation's time is
logged as `edax_seconds`, to take out of the run's wall time. On one RTX 5080, with an
8-block network, a level at the defaults took about 7.5 minutes before our agent's searches
computed in bfloat16 and searched only on its own moves (each about twice as fast). A resumed run (`--resume`) goes on logging them to
the same run, at the same epochs, and keeps the results of the last networks it played
before it stopped for their mean.

To evaluate a checkpoint after training, `vs_engine.py` logs to its run with `--monitor
--run ID` (the id in the run's seaglass link), at the checkpoint's epoch:

```
uv run examples/othello/vs_engine.py runs/othello/49.eqx --levels 8 --sims 1000 --openings 256 \
    --stop-below -1 --monitor --run ID
```

### Against Edax's scores of fixed positions

Games against engines measure strength, but coarsely: 512 games give a score within about
±0.02, and a run's score moves by more than that from training randomness alone. The
engine-labelled set measures the network more finely, without search or games: thousands of
fixed positions from self-play, past the XOT opening and with a choice of move, each legal
move of which Edax has scored (the disc margin it expects at the end, searching 24 moves
ahead and solving the last ~31 squares exactly).

`train.py` scores the network on it every epoch and logs, for all positions and for each
game phase (`opening`, 40-51 empty squares; `midgame`, 25-39; `endgame`, 1-24):

- `engine_top1`: how often the network's most likely move is one of Edax's best
- `engine_regret`: discs its most likely move gives up against Edax's best
- `engine_exp_regret`: the same, averaged over its policy: smoother, and about three times
  less noisy than `engine_regret` (a difference between two checkpoints on the same set is
  good to about ±0.015 discs). But it also measures how sharp the policy is, which the
  search and its policy target set (more sims, or `--policy-target completed-q`, train a
  sharper or flatter one): compare it only between runs with the same search and target,
  and across them, `engine_regret` and `engine_value_mse`
- `engine_value_mse`: its value's squared error against the outcome Edax predicts (±1, 0 for a
  draw)
- `engine_value_sign`: how often its value calls the winner Edax does

Scoring takes a forward pass over the positions, about 20 ms for 5000 on the GPU.
`--engine-set PATH` scores on another set, and `--engine-eval-every N` sets the cadence (0
for never). Without a set at the default path, `~/.cache/turbozero/othello/engine-sets/edax-l24.npz`,
training warns and goes on unscored; a missing `--engine-set` is an error.

To build a set, from the replay buffers of saved training states (`--save-state-at`); each
`--source` is a state, optionally only some of its environments:

```
examples/othello/setup_edax.sh                       # once
taskset -c 16-23 uv run examples/othello/label_positions.py \
    --source az10x128=runs/trunk/state-150.npz:896-1024 \
    --source pooled4x64=runs/p4x64/state-450.npz --per-source 2500 --level 24 --workers 8
```

At level 24 a position takes about 2 seconds on average (up to a minute), so 5000 took 84
minutes on 8 Edax processes. To score saved checkpoints on a set:

```
uv run examples/othello/score_engine_set.py runs/othello/49.eqx runs/othello/99.eqx --net resnet:10x128
```

## Credits

`xot-openings.txt` is Matthias Berg's large XOT list
([berg.earthlingz.de/xot](https://berg.earthlingz.de/xot/)), as also distributed with
Egaroucid, NBoard and othello-sensei. Edax (Richard Delorme) and Egaroucid (Takuto
Yamana) are under the GPL; the setup scripts build them from their repositories, they
aren't part of turbozero.

## The Elo league

`elo_league.py` keeps a rating for every network it's given, at each search budget, and for
Edax and Egaroucid, from full games played whenever the GPU is free (see `core/league/`). Every
game is stored; ratings are fit to all of them, with an uncertainty for each player.
