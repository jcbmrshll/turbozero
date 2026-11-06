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
| `setup_edax.sh`, `setup_egaroucid.sh` | Build the engines for `vs_engine.py` and `clock_match.py` |
| `clock_match.py` | Games on a real clock, one at a time on the same cores, each side at its best settings |
| `cpu_search.py`, `cpu_search.cpp` | A single game's MCTS for playing on CPU: a tree in C++, leaves evaluated in batches by OpenVINO or ONNX Runtime |
| `cpu_match.py` | Games between two such players at fixed visits a move, and their speed in evals a second |
| `export_onnx.py` | Exports a checkpoint's network to ONNX |
| `bench_cpu_backends.py` | Evals a second by CPU backend and batch, each checked against JAX's outputs |
| `quantize.py`, `fidelity.py`, `bench_quantized.py` | Quantizes a network to int8 for playing on CPU, and measures its outputs and speed against bf16 |
| `engine_set.py`, `label_positions.py`, `score_engine_set.py` | The engine-labelled set: fixed positions with every legal move scored by Edax, to build, and to score checkpoints on (train.py scores the network on it every epoch) |
| `distill.py` | Trains a network on a saved training state's replay buffer, with no self-play, on a teacher network's values or the games' outcomes |
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

`--playout-cap P:FAST` is KataGo's playout cap randomization: each move is searched with
`--sims` iterations with probability P, and otherwise with FAST, without the root's noise.
Only the full searches' positions are stored for training; the fast ones are played, and
count towards their game's outcome. At the same compute, that plays more moves and so more
games (KataGo: P 0.25, 600 and 100 playouts). `--selfplay-cycles` holds an epoch's
self-play compute fixed across settings, in cycles per environment (by default 128 full
searches, 128 × (`--sims` + 1)); size `--buffer` by the share of moves stored, for the
same span of self-play, and pass a small `--advance-every` whose multiples the searches'
cycles (iterations + 1) are, so that no search waits for its move to be played. For
example, 25% of moves at 127 iterations and 75% at 43, at the compute of uniform 64:
`--sims 127 --playout-cap 0.25:43 --selfplay-cycles 8320 --advance-every 4 --buffer 281`.

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
from. Self-play adds 128 an epoch (a position a move), so the default 1125 is about the last
9 epochs. The run's page plots the positions training can sample as `buffer_samples`, and
how many of them are distinct, a position's symmetric images counting as one, as
`buffer_distinct_positions`.

The buffer's size sets how old the data training samples is, not how often each position
is trained on: on average, that's the samples training takes each epoch over the positions
self-play adds, whatever the size. With the defaults, 256 steps of 4096 samples
against 1024 environments adding 128 positions each, it's 8, each time in a symmetry drawn
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
- `norm=`: the normalization. `batch`, BatchNorm, is the default. The others treat each
  position on its own, so the network has no state and training and play compute the
  same function: `channel` normalizes each square's channels (a transformer's per-token
  LayerNorm), `layer`, `group` (`norm_groups`) and `rms` the channels and the board
  together; `fixed` (KataGo's fixed variance initialization) and `ws` (weight
  standardization) don't normalize, but scale each convolution to unit variance at
  initialization, and fold into the convolutions as BatchNorm does.
- `layout=pre`: the pre-activation residual network: a normalization and a ReLU before
  each convolution, branches added to a trunk that is never normalized, and one more
  normalization before the heads. Every branch reads unit-scale input however deep the
  network, and each branch's last convolution starts at zero, so the network starts as
  the stem and the heads. Not with `fixed` or `ws`.

  On the 5080, `norm=channel,layout=pre` costs 4% of self-play and 10% of a train step
  against BatchNorm (8x128), and 10-13% of CPU play in bf16; `layer`, `group` and `rms`
  cost 16-30%. At 16x64, 40 epochs at 1e-3, it learned faster than BatchNorm (engine
  regret 5.7 against 7.3 at epoch 8), where `ws` trailed and random-initialized
  branches started 10-15 epochs late. In the post-activation layout, `ws`'s residual
  stream grows multiplicatively with depth, and at 4x64, 68 minutes, it lost to
  BatchNorm in direct games by 49 ±12 Elo at 128 visits (`fixed` and `rms` about 110).

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
moves its played positions come from (the last 1125 at the default `--buffer`), however many positions each move stored. They're kept in a replay buffer of their
own, by default K times `--buffer`, which holds that span even when every move stores K;
`--tree-buffer` sets its size instead (a move stores fewer than K when too few nodes qualify, and about 1.3 of 2
with `--tree-discarded-only`, so a smaller buffer can still hold the whole span).

Each K adds to GPU memory what raising `--buffer` by as much again would: 0.54 GB per K at
`--buffer 1125` (about 480 bytes a position, in each of 1024 environments). Training holds
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
`--epochs`, seed and schedules (the learning rate's and `--sims-schedule`, both
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

The learning rate holds `--lr`, then drops 10x at 70% of the run and again at 90% (`--lr-drops`).
To read several budgets from one run, fork that same schedule: train the trunk at a constant
learning rate to 70% of a budget, then fork the last 30%, with `--lr 1e-4 --lr-drops 0.667:1e-5`.

A drop at a share of the budget comes too late in long runs: at 1e-3, 6x96 and 8x128 nets stop
gaining after about 120-160 epochs whatever the budget. `--lr-plateau` drops when training
stops improving instead: from `--lr` to 1e-4, then 1e-5 (or the rates it's given), each time the
training loss (`loss`) improved by less than `--lr-plateau-threshold` (0.004) over the last
`--lr-plateau-window` (40) epochs, by the slope of a least-squares line through them, once
`--lr-plateau-min-epochs` (40) have been spent at the rate. `--epochs` caps the run, and
`--lr-plateau-stop` ends it early once the last rate plateaus. Each drop is logged as `lr_drop`
(the new rate), at the epoch that made it, with that epoch's `lr_plateau_improvement`. A run
resumed with `--resume` carries on with the rule where it stopped. A fork starts again at
its own `--lr`: to carry on at 1e-4, fork with `--lr 1e-4 --lr-plateau 1e-5`.

```
uv run examples/othello/train.py --epochs 600 --lr-plateau --save-state-every 10 --ckpt-dir runs/plateau
```

### Growing the network

Self-play costs far more than training. KataGo started with small nets and trained the next
size up on the same games, switching when it caught up. `--grow NET` (repeat it for several,
smallest first) does the same: `--net` plays self-play, and the next network trains on the same
replay buffer each epoch without playing. When it takes over, the one before stops, and the
one after that, if any, starts training from scratch on the buffer as it is. So two networks
train at a time.

The next network takes over when its `--grow-metric` on the engine-labelled set
(`engine_value_mse`, scored every epoch) is at or below the playing one's. Each is averaged over
the last `--grow-window` (10) epochs, and the next must have trained `--grow-min-epochs` (10) first.
With `--grow-switch-at EPOCHS`, it takes over after fixed numbers of epochs instead. Either way,
a playing network whose `--lr-plateau-stop` rule is done hands over rather than stopping the
run. A waiting network trains at `--lr`, and follows `--lr-plateau` from its first rate once it
plays.

Each network's checkpoints go to `--ckpt-dir/<its size>`, e.g. `runs/grow/2x32` and `runs/grow/4x64`.
Saved states hold every network, in `--ckpt-dir`. Each network's metrics are logged as
`<its size>/<metric>`, and the playing network's also without the prefix. The run logs which
network played (`growth_playing`), the next's mean metric minus the playing one's (`growth_gap`),
and the switch (`growth_switch`). Every run logs its compute per epoch, to compare runs at equal
GPU time: `selfplay_seconds`, `train_seconds` and `test_seconds` (with `--grow`, each network's
as `<its size>/train_seconds` and `<its size>/test_seconds`), `epoch_seconds`, and the run's total,
`run_seconds`.

```
uv run examples/othello/train.py --net resnet:2x32,pool_every=2,heads=pooled,pool_channels=8 \
    --grow resnet:4x64,pool_every=2,heads=pooled,pool_channels=16 --engine-eval-every 1 \
    --lr-plateau --lr-plateau-stop --epochs 600 --ckpt-dir runs/grow
```

### Training on Modal

`modal_jobs.py` runs train.py on Modal's GPUs, several at once, and brings the results to the
paths a local run would write them to, so analysis and the Elo league don't need to know:

```sh
# steps run in order on one GPU, from the commit named (it must be on GitHub)
uv run examples/othello/modal_jobs.py launch --name trunk-a --gpu L40S --hours 4 --commit 7b52872 \
    --step "examples/othello/train.py --epochs 93 --lr-final 1e-3 --save-state-at 93 --monitor --name a --notes why --ckpt-dir $C/trunk > $L/trunk.log" \
    --step "examples/othello/train.py --epochs 19 --lr 1e-4 --lr-final 1e-4 --init-from $C/trunk/state-93.npz --ckpt-dir $C/fork > $L/fork.log"
uv run examples/othello/modal_jobs.py list [--billed today]
uv run examples/othello/modal_jobs.py stop trunk-a       # or --all; --now doesn't wait for the run to end as stopped
uv run examples/othello/modal_jobs.py resume trunk-a     # after it ended short; --hours for a new limit
uv run examples/othello/modal_jobs.py sync               # `launch` keeps one running, in tmux session modal-sync
```

A step is the command as you'd write it here, with `> LOG`; paths must be under
`~/.cache/turbozero`. `--init-from` names a `state-<EPOCH>.npz`, here, in the bucket, or from an
earlier step. `--vs-engine-*` is refused, as the image has no Edax. A job runs without this machine.
Files go to the bucket as they're written: the sync brings them here, and a `--monitor` run appears
on slick's seaglass while it runs, tagged with `$SEAGLASS_TAGS` and `modal`, its chip in its
provenance. `modal-job.json` beside the checkpoints records commit, chip, flags and wall time.
Nothing leaves the bucket on its own. An interrupted job continues from its newest state, saved
every `--resume-every` epochs (10) and kept in the bucket only; it repeats the epochs since, so its
log and metrics hold those epochs twice, the later ones the ones that count, as with `--resume`.

What each chip did on 8x128, 128 sims a move, 8 passes, window 1125 (72 s/epoch on the RTX 5080),
with 2 cores and 16 GiB, at Modal's list prices of 2026-10-31:

| `--gpu` | s/epoch | speed vs 5080 | $/h | $ per 5080-hour |
|---|---|---|---|---|
| L40S | 48 | 1.49 | 2.17 | 1.46 |
| RTX-PRO-6000 | 34.5 | 2.09 | 3.25 | 1.56 |
| H100! | 29 | 2.47 | 4.17 | 1.69 |
| A100-40GB | 56 | 1.29 | 2.32 | 1.80 |
| A10 | 129 | 0.56 | 1.32 | 2.37 |

L40S costs least for the work done on this net; H100! gives results soonest. The advantage shrinks
on smaller nets: 4x64 at the same settings took 18.9 s/epoch on an L40S against 17.8 on the RTX 5080
(0.94x), so time a small net before budgeting it. Budgets stay in epochs, computed from RTX 5080
timings, as chips differ in speed.

Modal also bills what jobs upload, $0.04/GB: about $0.36/h with `--resume-every 25` on 4x64 at window
1125, and `list` leaves it out (`list --billed` has it). Raise `--resume-every` to send fewer states,
at the risk of more to redo after an interruption. Bringing a state of about 0.8 GB here costs about
$0.10 in GCS egress, which is why states saved only to continue from stay in the bucket.

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

### On a clock

`clock_match.py` plays games the way a contest would: one at a time, the same seconds for each
side for the whole game, on the same cores. Each engine runs its own time management at its
strongest settings: Edax over GTP (`time_settings`, then `time_left` before each move), and
Egaroucid over its console protocol (`-time`, `play`, `go`), since its GTP has no clock.

```
taskset -c 16-23 uv run examples/othello/clock_match.py --a edax:threads=8 --b egaroucid:threads=8 \
    --seconds 10 --openings 100
```

which amounts to these engine processes, each with a fresh process for every game:

```
edax/bin/edax -gtp -q -l 60 -n 8 -book-usage on -h 25    # in edax/
egaroucid/bin/Egaroucid_for_Console.out -quiet -noboard -threads 8 -time 10    # in egaroucid/bin/
```

- **Run them with the machine quiet.** Eight busy cores on the other core complex slowed the
  engines' 8-thread searches by 30-45%, more than any choice of cores or build. Which cores
  matters less: on an idle Ryzen 9 9900X3D, 16-23 (eight physical cores across both complexes)
  and 0-5,12,13 (the V-cache complex, two cores doubled) were within 7% for the engines, though
  16-23 ran a network's batched search 1.45 times faster. An idle sibling is part of the core:
  16-23 needs 4-11 idle.
- **Clocks of 2 to 10 seconds a game suit them.** Edax used about 85% of each, and Egaroucid
  95-100%, passing it by at most 14 ms (hence `GRACE`, 50 ms). At 30 seconds Egaroucid used only 59%:
  its time management keeps a large margin. Clocks are whole seconds: Egaroucid's `-time` is.
- **Opening books are on**, as they ship. From XOT openings they hardly matter: of 300 openings,
  4 start in Egaroucid's book, and Edax ships an empty one.
- **Neither engine ponders** over these protocols, and between moves neither uses any CPU.

A level is no stand-in for a clock. On real positions, from XOT openings, with 8 threads on 16-23:

| | Seconds a move: mean | 99th percentile | Worst | Seconds a game: mean | Worst |
|---|---|---|---|---|---|
| Edax level 16 | 0.018 | 0.31 | 0.67 | 0.47 | 1.0 |
| Edax level 20 | 0.046 | 0.75 | 1.35 | 1.2 | 2.0 |
| Edax level 24 | 0.11 | 1.3 | 2.2 | 2.9 | 6.1 |
| Edax level 28 | 0.64 | 8.0 | 17 | 16.7 | 34 |
| Egaroucid level 16 | 0.020 | 0.31 | 0.49 | 0.53 | 0.75 |
| Egaroucid level 20 | 0.048 | 0.40 | 0.73 | 1.3 | 3.1 |
| Egaroucid level 24 | 0.16 | 1.5 | 7.9 | 4.3 | 11.4 |
| Egaroucid level 28 | 0.76 | 7.9 | 20 | 19.9 | 56 |

More threads make a level no stronger, only faster (at level 16, 150 games each: 8 threads
against 1 scored 0.49 for Edax and 0.50 for Egaroucid, each about 0.04 either way), and one
search took 3.5 times less time on 8 threads than on 1 at level 20.

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

## Playing on CPU

Self-play's search is built for thousands of games at once on a GPU. Alone on CPU it spends about
12 times what the network costs: 86 evals a second for an 8x128 net whose forward pass alone does
1056. `cpu_search.py` plays one game on CPU instead: `clock_match.py` plays it, as `net:`, against
another or an engine on a per-game clock.

```
taskset -c 16-23 uv run examples/othello/clock_match.py --seconds 10 --openings 20 \
    --a net:net=8x128-128s-8p-w1125-274m,batch=16,virtual_loss=visit,random_symmetry=1,taper=24 \
    --b edax:threads=8
```

The network is named as in the league's store. Its search collects 16 traversals at a time, each
steered off the others by virtual visits, and evaluates their leaves together, split over a stream
per core of OpenVINO, in bfloat16. With `random_symmetry=1` it shows the network each leaf in a
random one of the board's 8 symmetries, which costs nothing measurable and gained about 30 Elo at
4000 visits a move, in self-play and against Edax, though nothing shown at 1000. On eight physical cores of a Ryzen 9 9900X3D, one game at a time:

| 8x128, evals a second | batch 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| With the search, OpenVINO bf16 | 2796 | 9906 | 11002 | 12358 |

| Network only, batch 1 / 8 / 64 | 4x64 | 6x96 | 8x128 | 10x128 |
|---|---|---|---|---|
| JAX | 5077 / 15133 / 29386 | 2051 / 5936 / 10819 | 1056 / 3158 / 5191 | 703 / 2462 / 3940 |
| ONNX Runtime, f32 | 6552 / 12253 / 16418 | 5088 / 9902 / 13898 | 2266 / 4775 / 6839 | 1600 / 3335 / 3658 |
| OpenVINO bf16 | 6289 / 43221 / 97480 | 4296 / 22975 / 34898 | 3210 / 12192 / 16458 | 2777 / 11204 / 13189 |

Each of these is one run, on a machine with other work on it. Batch 1 is on one stream, the others
on a stream per core. Physical cores matter more than the CPU's cache: the same net at batch 32 does
12358 on eight cores and 8556 on six with two of them doubled up, V-cache or not.

At 1000 visits a move, batches of 8 and 16 lost nothing against sequential search in 200 games each
(+17 and +19, standard errors 16 and 18), batch 32 lost 28 and batch 64 lost 60. On a 4 s clock
batch 16 took 86.5% of 100 games from the same net searching sequentially.

### What a player costs

`timing_benchmark.py` times every player alone: each game is a player against itself from XOT
openings, and only the side to move is charged, move by move. Stop everything else on the machine
first (other load costs engines 30-45%); it won't start if any CPU is doing real work.

```
uv run examples/othello/timing_benchmark.py prepare     # the nets, as ONNX, once
taskset -c 16-23 uv run examples/othello/timing_benchmark.py work run-1 --hours 3.5
uv run examples/othello/timing_benchmark.py summarize run-1 --publish
```

`--publish` writes the league's `timings.json`; each move's record stays under
`~/.cache/turbozero/othello/timing/`. One side's seconds a game on 8 threads of those cores, with
bf16 networks, on 2026-11-01:

| Sims | 4x64 | 6x96 | 8x128 | 10x128 |
|---|---|---|---|---|
| 128, sequential | 0.52 | 0.73 | 0.92 | 1.11 |
| 1000, sequential | 3.58 | 4.98 | 6.50 | 7.73 |
| 1000, batch 16 | 0.46 | 0.90 | 1.67 | 2.05 |
| 4000, batch 32 | 1.35 | 3.02 | 5.26 | 6.53 |
| 16000, batch 32 | 4.91 | 11.4 | 22.8 | 26.4 |

| Level | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|
| Edax | 0.087 | 0.240 | 0.268 | 1.07 | 2.54 | 14.0 |
| Egaroucid | 0.021 | 0.188 | 0.392 | 1.25 | 4.15 | 18.4 |

A line through 128, 1000 and 4000 sims predicted seconds a move at 16000 to within 9.1%. An engine's
slowest move at L20 and up took 33-97 times its mean.

### Int8

Quantized after training, a network plays stronger on a clock than in bf16, where its leaves come in
batches. Build the model, calibrated on the leaves of its own searches, and play it as a `backend`:

```
uv run examples/othello/fidelity.py leaves --net 8x128-128s-8p-w1125-274m --out LEAVES
uv run examples/othello/quantize.py --net 8x128-128s-8p-w1125-274m --positions LEAVES/calibration.npy
taskset -c 16-23 uv run examples/othello/clock_match.py --seconds 4 --openings 100 --b edax:threads=8 \
    --a net:net=8x128-128s-8p-w1125-274m,batch=16,virtual_loss=visit,taper=24,backend=int8-performance_leaves_heads_minmax
```

Int8 against bf16, on the same eight cores with nothing else running, batch 16 with the search; games
from XOT openings, 1000 visits a move, or 4 s a side:

| Network | Evals a second | Elo, equal visits | Elo, equal clock |
|---|---|---|---|
| 4x64 | 52827 vs 41778, 1.26x | | |
| 6x96 | 28019 vs 20302, 1.38x | -3 ± 19 (200 games) | +63 ± 19 (200) |
| 8x128 | 17298 vs 10429, 1.66x | -9 ± 12 (400) | +83 ± 21 (200) |
| 10x128 | 14146 vs 8876, 1.59x | | |

At a batch of 1 it's no faster (0.94-0.99x). The gain is OpenVINO's int8 kernels on AVX512_VNNI: on
ONNX Runtime int8 only matched bf16, and a CPU without VNNI will differ. The 4x64 net's outputs follow
bf16's least closely, and it hasn't played.

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

A network's players keep a tree of 2 x sims nodes unless they're added with a bigger one
(`add-player NET --sims 4000 --tree-factor 4`, a player of its own, `NET@4000-t4`). For a session
of requested series only, with the GPU to itself: `play --requests-only --batch-seconds 900`.

The dashboard (`elo_league.py dashboard`) lists the players, with filters read from their names and
settings, beside the queue of series and a plot of runtime against rating. A player's page holds
its description and tags. A network's budgets share theirs.

A game's page runs trained networks in the browser, on their ONNX exports, through onnxruntime-web
served from the store (`elo_league.py fetch-ort`, once). No game's search was kept, so its tree is
searched again from each position, by default by the player to move's network at the sims it played
at, with the same PUCT as core's MCTS. Games kept their tree between moves, and this doesn't, so a
position's tree isn't the one the game had.

The same page can run Edax and Egaroucid in the browser, compiled to WebAssembly from the commits the
setup scripts pin (`elo_league.py build-browser-engines`, once; `setup_wasm_engines.sh`), with no book.
It shows each one's best move and its score in discs for the player to move. They run on one thread:
threads need cross-origin isolation headers and HTTPS, which the tailnet's plain HTTP address doesn't
have. A search can't be interrupted, so stopping one ends its worker: Edax is back in about 0.1 s, and
Egaroucid in about 1 s, as it builds its tables again. Edax takes about 130 MB. Egaroucid takes about
565 MB, with a hash table of 2^23 entries for native's 2^25; its keys are native's, cut to size.
`wasm/check_against_native.py` compares them with the native engines on the league's own positions.
These numbers are for exploring a game: they aren't ratings, and they stay off the Pareto plot.

Every run and every network registered with the league is exported to ONNX, which is cheap (about
0.3 s and 9 MB for an 8x128 once the process is up). A failed export only says so, and
`elo_league.py export-onnx` tries the store's missing ones again.

The plot reads `timings.json` beside the store, which a benchmark writes atomically:

```json
{"runs": [{"name": "...", "hardware": {"cpu": "...", "cores": 8, "threads": 8}, "software": {},
  "configs": [
    {"net": "resnet:8x128,...", "sims": 1000, "batch": 16, "batch_strength": "measured_no_loss",
     "source": "measured", "plot": true, "games": 24, "seconds_per_game": 2.1},
    {"engine": "edax", "level": 20,
     "source": "measured", "plot": true, "games": 24, "seconds_per_game": 1.4}]}]}
```

A run is one host, and the plot shows one at a time. `seconds_per_game` is one side's mean. Only
configs that are `measured` and marked `plot` appear, joined to players by architecture and sims or
engine and level. A `batch_strength` of `unmeasured` draws a point hollow and off the frontier.
