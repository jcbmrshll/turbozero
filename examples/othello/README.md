# Othello

AlphaZero on Othello: training, and evaluating the result against strong open-source Othello
engines, chiefly [Edax](https://github.com/abulmo/edax-reversi) and
[Egaroucid](https://github.com/Nyanyan/Egaroucid).

| File | |
|---|---|
| `train.py` | Trains a network by self-play, validating its weight average against an engine's scores of fixed positions as it goes |
| `evaluate.py` | Every evaluation game, between any two players (networks or engines), here, on the GPU, on Google Cloud or on Modal, kept in one store |
| `eval_players.py`, `eval_games.py` | Its players, each a config with an id of its digest, and its games: one at a time on clocks or fixed budgets, or in batches on the GPU |
| `engines.py` | Edax and Egaroucid, each driven over GTP, and what their searches made of each position, from their logs; and Melody, Neural Reversi, ymatioun and Edax-FM |
| `setup_edax.sh`, `setup_egaroucid.sh`, `setup_melody.sh`, `setup_neural_reversi.sh`, `setup_ymatioun.sh`, `setup_edax_fm.sh` | Build the engines |
| `clock_match.py` | The referee of `evaluate.py`'s games: each side on its own clock or budget, stopped and solved at a depth if asked |
| `ggs.py` | GGS's random starts, its archive's, and its clock and scoring |
| `gcp_cpu_jobs.py`, `gcp_cpu_runner.py`, `modal_hosts.py` | Commands on Google Cloud Spot VMs of one CPU model (8 Zen 5 cores, AVX-512 VNNI), for timed games, or in Modal containers |
| `cpu_search.py`, `cpu_search.cpp` | A single game's MCTS for playing on CPU: a tree in C++, its leaves evaluated in batches by OpenVINO or ONNX Runtime, or searched by a thread to a core (`Engine`) |
| `cpu_net.py`, `cpu_net.h` | A network's int8 forward pass in C++ (AVX-512 VNNI, AVX2 or ARM's NEON), for `Engine`: no OpenVINO, closer to the float network |
| `cpu_match.py` | Our player on the CPU, and its speed in evals a second |
| `endgame.py`, `endgame.cpp` | The endgame solved exactly in C++, for a player (`solve=N`) and in batches for labels |
| `export_onnx.py` | Exports a checkpoint's network to ONNX |
| `bench_cpu_backends.py` | Evals a second by CPU backend and batch, each checked against JAX's outputs |
| `quantize.py`, `fidelity.py`, `bench_quantized.py` | Quantizes a network to int8 for playing on CPU, and measures its outputs and speed against bf16 |
| `engine_set.py`, `label_positions.py`, `score_engine_set.py` | The engine-labelled set: fixed positions with every legal move scored by a strong engine, to build, and to score checkpoints on (train.py validates the weight average on it every epoch) |
| `distill.py` | Trains a network on a saved training state's replay buffer or a dataset, with no self-play, on a teacher network's values or the games' outcomes |
| `teacher_dataset.py` | A dataset of whole self-play games from a kept move log (see `core/memory/move_log.py`), labelled with exact endgame results |
| `bench_nets.py` | Times networks on their own: a forward pass at self-play's batch, a training step at training's |
| `game.py` | What the scripts share: the environment, board symmetries, network, evaluators |
| `xot.py`, `xot-openings.txt` | The XOT opening list |

## Training

```
uv run examples/othello/train.py --monitor --ckpt-dir runs/othello --notes "the baseline"
```

`--name` names the run on the server, and `--notes` says what it's testing and why, at the
top of its page.

Self-play runs in 1024 games at once, searching 128 MCTS iterations a move. Replay memory
stores each position once, and training maps each position it samples through one of the
board's 8 symmetries (rotations and reflections), drawn at random. Games start
from XOT openings rather than the standard start, which keeps self-play varied
(`--standard-starts` mixes in the standard start).

Self-play plays with a weight average of the network (`--average-epochs`, below), and the run
is tracked by validating that average on the engine-labelled set: every epoch, `train.py`
scores it on an engine's scores of fixed positions, in about 20 ms, and logs
`avg4/engine_exp_regret` (the discs its policy gives up against the engine's best move) and the
rest of the `avg4/engine_*` metrics
(see [below](#against-edaxs-scores-of-fixed-positions)). They're much finer than games: a
difference between two checkpoints is good to about ±0.015 discs. Games measure strength
itself: `evaluate.py` plays checkpoints against engines and each other (see [Evaluating a
checkpoint](#evaluating-a-checkpoint)), and the Elo league rates them.

The run's page also plots `episode_draw`, the fraction of self-play games drawn, which
`game.py`'s `episode_metrics` reports through the Trainer's `episode_metrics_fn`. See `--help` for
the network size, search budget and number of epochs.

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

`--buffer-window N` trains on only each environment's newest N of the positions kept, as if
`--buffer` were N, while the older ones stay stored. A branch can then widen the window at once,
onto positions the run already has, where a bigger `--buffer` would start empty and take as
many epochs to fill: keep the largest buffer a run may use (`--buffer 9000 --buffer-window 1125`)
and branch with a wider window to try one.

The buffer's size sets how old the data training samples is, not how often each position
is trained on: on average, that's the samples training takes each epoch over the positions
self-play adds, whatever the size. With the defaults, 256 steps of 4096 samples
against 1024 environments adding 128 positions each, it's 8, each time in a symmetry drawn
at random. `--train-steps` changes it (or `--train-batch`).

The network computes in bfloat16 by default: in self-play
(`--inference-dtype`), and its convolutions in training (`--train-dtype`, mixed precision:
its parameters, BatchNorm and heads stay in float32). On one RTX 5080, with the plain
`resnet:6x128` (below) computing in float32 throughout, an epoch takes about 63 seconds.
bfloat16 in self-play brings an epoch to about 36 seconds; searching with
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
  batch. M must divide `--train-batch`.

Both apply to training alone (self-play is unchanged), and can be
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

### Exact endgame labels

By default every self-play position with 20 or fewer empty squares gets an exact value
target: an exact solver's result for it (win, draw or loss for the player to move), instead of
the result the game had when played out. Each position before a game's first one at 20 or fewer
takes that position's result. The labels lose the noise of endgame mistakes: in F2 e440's
self-play (the 8x128 at 511/127 sims), the played result differs from the exact one at 16-20
empties in 17% of games, mostly a win or loss played as a draw or the reverse.
`--solve-empties N` moves the cut to N empty squares; `--solve-empties off` plays games out and
labels them with their played results.

The solver (`--solver native`, endgame.py, a win/draw/loss solve; `edax` for comparison)
runs on the CPU while the GPU trains and plays. Positions stored in an epoch's self-play are solved
through its training and the next epoch's self-play, and trained on from then. A new run's first
epoch solves them at once, since it has nothing else to train on. On the 4x64 at 128 sims, an
epoch stores 43,000-60,000 positions at 20 or fewer empties, solved in 13-19 s on 14 cores, and
self-play waited under 0.25 s an epoch for them (`--solve-threads`, default every core the process
may use; on Modal, `--cpu 16`). `solve_positions` counts every solve, and `solve_games`,
`solve_draw` and `solve_missed` count the games.

`--solve-play-past MEAN` ends games early instead: each game stops past N before each further
move with probability 1 / (1 + MEAN), MEAN moves on average, a geometric number, and skips those
moves' searches to play more games in the same compute. Positions up to N keep the solved result
at N; each position stored past N is solved on its own. With 0, games stop at N and never produce
positions below it, though searches from just above N still reach them and the network values
their leaves there. The default, `end`, plays every game to its end.

### Policy target from the search

The network learns each played position's policy from the search's visit distribution, as in
AlphaZero.

`policy_loss`, the cross-entropy against those targets, is their entropy (logged as
`policy_target_entropy`), which no network can lower, plus the KL divergence from them to the
network's policy (`policy_kl`). The search sets the first: a deeper search trains toward
sharper targets. So `policy_loss` doesn't
compare across runs that differ in those; `policy_kl` says how well each network fits its own
targets, and the engine-set metrics how good its policy is.

### Resuming, branching and forking runs

`--save-weights-every N` (1 by default) saves the weight average every N epochs as
`--ckpt-dir/avg4/<epoch>.eqx`, its weights and BatchNorm statistics: what a player loads, and
nothing for training to continue from. `--save-state-at 50,150` also saves the whole training
state once 50 and 150 epochs are done, as `state-50.npz` and `state-150.npz` in `--ckpt-dir`: the
network, optimizer state, weight average, replay buffer, games in progress with their search
trees and the rng. `--save-state-every 10` saves one every 10 epochs and keeps only the newest of
these. A state takes about as much disk space as replay memory takes GPU memory, about 0.18 GB
per 375 positions per environment.

`--resume` continues the run in `--ckpt-dir` from its newest state; pass the run's own
arguments. It carries on exactly as if it hadn't stopped (on the CPU the result is
bit-identical; on a GPU, convolution autotuning may differ between processes). It logs
to the same seaglass run, which then holds the epochs between the state and the stop
twice, and deletes the weights saved after the state, which it saves again.

```
uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long
uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long --resume
```

The learning rate is constant, `--lr`. `--branch-from` continues a run from a state in a new
`--ckpt-dir` with other settings: a branch of the run rather than a new one, and the way to drop
the learning rate. It goes on at the state's epoch (`--epochs` counts the epochs before it too),
with the network, its optimizer state (taking the branch's `--lr`, `--weight-decay` and
`--momentum`), the weight average, the buffer, the games in progress and the rng. Only a search of
another size starts its trees anew, its moves in progress searched again; with the run's own
settings a branch continues it exactly. A different `--buffer` resizes the replay buffer at the
branch's start, each environment keeping its newest positions; a different `--buffer-window` only
changes which of them training samples. The branch logs to a seaglass run of its own, whose config
records the state as `parent`.

```
uv run examples/othello/train.py --epochs 250 --lr 1e-4 --weight-decay 0.05 \
    --branch-from runs/trunk/state-150.npz --save-state-at 250 --ckpt-dir runs/trunk150-1e-4
```

Under a normalization most weights' scale doesn't change the network's output, and AdamW settles
their norm about sqrt(learning rate / weight decay): a 10x drop with the decay held shrinks it
about 3x over the following epochs, which raises the effective step (about the learning rate /
the norm) back and costs the drop much of its gain. A branch at a 10x lower `--lr` holds the norm
with a 10x higher `--weight-decay`, as above (0.5 at 1e-3, 0.05 at 1e-4: AdamW's decay of a weight
each step is the learning rate times the decay).

`--init-from` forks a new run from a state (a `state-<epoch>.npz`, or a directory to take
the newest from). The fork is a run of its own: it starts at epoch 0, with its own
`--epochs` and seed, and its own `--ckpt-dir`. It starts from the saved network, optimizer
state, weight average and replay buffer, games in progress included, so its first epochs train
on as much data as the run it was forked from did. `--init-without-buffers` starts with an empty
buffer and new games instead. The network size and `--envs` must match. The replay memory may
differ in size: each environment's newest samples are kept, as many as fit. The seaglass run's
config records the parent state as `parent`, and so does every state the fork saves.

This makes it cheap to compare settings late in training. Short runs from scratch only
show which settings learn fastest early on. Instead, train one long trunk, save its state, and
fork short runs from it that differ in one setting each, alongside a control fork with the
trunk's settings:

```
uv run examples/othello/train.py --epochs 150 --save-state-at 150 --ckpt-dir runs/trunk
uv run examples/othello/train.py --epochs 30 --init-from runs/trunk/state-150.npz \
    --ckpt-dir runs/trunk150-control
uv run examples/othello/train.py --epochs 30 --init-from runs/trunk/state-150.npz \
    --sims 256 --ckpt-dir runs/trunk150-sims256
```

### Weight averages

At a constant learning rate the network's weights jitter from epoch to epoch, and so does its
play: KataGo played and rated averaged weights. Here self-play plays with an average of the
network's weights over about `--average-epochs` epochs (4 by default): the mean of the epochs so
far, then an exponential moving average that moves 1/H of the way to each epoch's weights, its
horizon growing from 1 early in the run (see `average_share` in core/training/train.py). Each
epoch, before self-play, the average moves toward the network's weights and gets BatchNorm
statistics of its own from `--average-norm-batches` micro-batches of the replay buffer through its
weights. Training itself is unchanged. The engine-labelled set validates the average
(`avg4/engine_*`), and the checkpoints `--save-weights-every` saves are its own. `--average-epochs
1` plays with, validates and saves the network's own weights.

### Memorization: held-out games

A position is trained on many times while it's in the replay buffer (about 14 at 256 steps of
4096 with `--playout-cap 0.25:64` and `--buffer 2250`), and the network fits what it can of each
game's outcome there. The training loss can't tell that fit from learning.
`--holdout-envs 32` plays 32 of the self-play environments' games like the others but never
trains on them, and every epoch, after training, scores the network's weights (`*_raw`) and the
weight average self-play played with (`*_avg`) on
`--holdout-positions` of their positions and as many of the trained-on games' positions of the
same ages:

- `heldout_*` and `seen_*`: policy KL and value loss on each, and `memorized_*` the difference.
  The two sets are different games, so a few thousand held-out games' own outcomes put an offset
  in the value loss that `*_se` (from micro-batch spread) doesn't include: about 0.004 at 32 of
  1024 environments, about 5,400 games. A checkpoint that never saw any of them scored
  +0.0039 ± 0.0027 worse on the last 32 environments' games than on the rest. So value readings
  under about 0.005 aren't resolvable at that size (hold out more environments for better), while
  the policy KL, whose targets are per position, is cleaner.

Training is unchanged apart from training on 3% fewer games. A train loop starts the held-out
buffers empty, since a saved state's may hold positions trained on, so a fork or resume measures
again from its start.

What a run's games can still teach its network, offline: `distill.py train --data STATE
--heldout-envs 64 --init-from CKPT` keeps training a checkpoint on a saved state's games, with
some environments' whole games held out (see distill.py). Once the held-out loss stops falling
while the trained-on probe's keeps falling, more passes over those games only memorize them.
Start from a checkpoint older than the state's positions, so that both sets start unseen.

### Training on Modal

`modal_jobs.py` runs train.py on Modal's GPUs, several at once, and brings the results to the
paths a local run would write them to, so analysis and the Elo league don't need to know:

```sh
# steps run in order on one GPU, from the commit named (it must be on GitHub)
uv run examples/othello/modal_jobs.py launch --name trunk-a --gpu L40S --hours 4 --commit 7b52872 \
    --step "examples/othello/train.py --epochs 93 --save-state-at 93 --monitor --name a --notes why --ckpt-dir $C/trunk > $L/trunk.log" \
    --step "examples/othello/train.py --epochs 112 --lr 1e-4 --weight-decay 0.05 --branch-from $C/trunk/state-93.npz --ckpt-dir $C/branch > $L/branch.log"
uv run examples/othello/modal_jobs.py list [--billed today]
uv run examples/othello/modal_jobs.py stop trunk-a       # or --all; --now doesn't wait for the run to end as stopped
uv run examples/othello/modal_jobs.py resume trunk-a     # after it ended short; --hours for a new limit
uv run examples/othello/modal_jobs.py sync               # `launch` keeps one running, in tmux session modal-sync
uv run examples/othello/modal_jobs.py extend trunk-a --step "..."   # more steps, after its others
```

A job can grow while it runs: `extend` appends steps, which it runs after the others, and with
`launch --follow 20` it waits up to 20 minutes after its last step for more before ending. A
run in segments (each a step that branches from the last one's final state, with new settings)
is then one job on one machine: each segment starts from the state on its own disk.
`--states-in-bucket` keeps saved states in the bucket, where later steps and other jobs (forks)
start from them, rather than bringing them here, and removes all but the newest segments' from
the job's disk.

A step is the command as you'd write it here, with `> LOG`; paths must be under
`~/.cache/turbozero`. `--init-from` and `--branch-from` name a `state-<EPOCH>.npz`, here, in the
bucket, or from an earlier step: a staircase of learning rates is steps that each branch from the
last one's final state. A job runs without this machine.
Files go to the bucket as they're written: the sync brings them here, and a `--monitor` run appears
on slick's seaglass while it runs, tagged with `$SEAGLASS_TAGS` and `modal`, its chip in its
provenance. `modal-job.json` beside the weights records commit, chip, flags and wall time.
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

Every game between two players, networks or engines, goes through `evaluate.py`, and every game is
kept in one store, `~/.cache/turbozero/othello/evals/evals.db`:

```
examples/othello/setup_edax.sh                      # once
examples/othello/setup_egaroucid.sh                 # once

# a network on the CPU at fixed visits, against Edax at level 24, every game solved from 24 empties
uv run examples/othello/evaluate.py run --cores 16-23 --openings 50 --solve-at 24 \
    --a net:net=NAME,visits=26400,engine=1,backend=native,virtual_loss=visit --b edax:level=24
# each on a 10 s clock, on 4 Google Cloud VMs
uv run examples/othello/evaluate.py run --on gcp --jobs 4 --openings 100 \
    --a net:net=NAME,clock=10,engine=1,backend=native --b edax:clock=10
# a checkpoint on the GPU, many games at once, against Edax at level 8 (the league's way)
uv run --extra cuda examples/othello/evaluate.py run --openings 256 \
    --a net:checkpoint=runs/othello/avg4/199.eqx,spec=resnet:8x128,device=gpu-batched,visits=400 \
    --b edax:level=8
uv run examples/othello/evaluate.py matches                  # every match, and how far it got
uv run examples/othello/evaluate.py show MATCH [--games]     # what it showed
uv run examples/othello/evaluate.py players -v               # every player, with its config
uv run examples/othello/evaluate.py page                     # the web page, on port 8031
```

**The page.** Every player's Elo, ± and record, the latest games, and a page per player with its
record against each opponent. Elo is fit to every game, whatever its match's setup. Players linked
by games, directly or through others, make a pool, fit on its own with its mean at 0: Elos from
different pools don't compare.

**Players.** A player is `kind:key=value,...` (`eval_players.py`), or a table of a `--config` file
(TOML or JSON, with the match's other settings): a network, `net:`, the league's by `net=NAME` or
any checkpoint by `checkpoint=PATH,spec=NET`; or an engine, `edax`, `egaroucid`, `melody`,
`neural-reversi`, `ymatioun` or `edax-fm`. Each side has its own budget: a network `visits` a move
or a `clock`, an engine a `level` or a `clock`, a contest's seconds for the whole game (`10`) or
GGS's (`01:00//02:00`). A network plays on a `device`: `cpu` (`cpu_search.py`, with `engine`,
`backend`, `batch` and the rest, see [Playing on CPU](#playing-on-cpu)) or `gpu-batched` (core's
MCTS, many games moving in step, at fixed visits against engines at a level or other such networks, with a tree of 4 x visits
nodes). On every device `prior_temperature` is the softmax temperature of the search's prior,
default 2: flatter than the network's policy, which self-play trains at 1.
Engines get `threads` (all the cores by default, one in batched games) and `args`.

**Ids.** A player is everything that decides how it plays: its config, defaults filled in, with a
network's checkpoint by its sha256 (and a quantized model by its file's), and an engine's build by
its files'. Its id is the config's digest, so the same settings are the same player wherever they
play, and any change, a level, a thread, a rebuilt engine, is a new player. What it ran on isn't
part of it: each game records its host (CPU model and ISA extensions, the cores it had, the GPU,
where: `local`, `gcp:MACHINE`, `modal:GPU:CORESc`) and the machine's load.

**Matches.** A match plays `--openings` starts, each with both colors, drawn by `--seed` from
`--starts` (`xot`, the default; GGS's `random:14`; or `file:PATH`, see [GGS
conditions](#ggs-conditions)). With `--solve-at N` every game stops at its first position with N
empties or fewer, solved by `endgame.py` for both sides (for a win, a draw or a loss; `--solve-exact`
for the margin); without it, games are played to the end. A match's id is its setup's and where
it's played: running it again continues it, and `--tag` starts another of the same. Each game keeps
its moves, every move's seconds (where moves are timed alone: not in GPU batches, whose timing is
the match's), how it ended (`end`, `solved`, `timeout`), and its host.

**Where.** `--on local` plays here, on `--cores` (else every core this process may use), one game
at a time, or `--jobs N` processes at once, each on its share of the cores (for fixed budgets: timed
games in parallel share the machine). `--on gcp` plays on `--jobs` Spot VMs of one CPU model (see
[On Google Cloud CPUs](#on-google-cloud-cpus)), and `--on modal` in Modal containers (`--gpu`,
`--modal-cpu`, `--modal-memory`): Modal's CPUs are a mix of models, so its seconds don't compare,
and some have no AVX-512, which the engines' default builds need. Play engines built for x86-64-v3
there, e.g. `EDAX_DIR=~/.cache/turbozero/othello/engines-2026-10/edax-v3 ARCH=x86-64-v3
examples/othello/setup_edax.sh` (and `ARCH=x86-64-v3 EGAROUCID_DIR=...egaroucid-v3` for
`setup_egaroucid.sh`), then `edax:dir=...,level=8`: another build, so another player. Cloud hosts send their games to the bucket as
they go, and `run` brings them into the store while it waits; `evaluate.py sync MATCH` brings in
what's there later. JAX stays off the GPU unless a player is on it.

The engines play with their opening book off at a fixed level: at level L (up to 10) both engines
search L moves ahead, and solve the game exactly once 2L or fewer squares are empty. In batched
games each has its own engine process (an Edax process takes about 170 MB and an Egaroucid one about
1.3 GB, so it runs as many as fit in a quarter of the available memory, up to 128). The engines and
our agent (at temperature 0, without root noise) are all deterministic, so games start from **XOT
openings**: a list of 10,784 eight-move openings that each end in a position Edax judges nearly even
(the same ones for every checkpoint, given the same `--seed`). Each opening is played twice, with
the colors swapped. This is how OLIVAW, an AlphaZero-style Othello program, was compared with Edax
([Norelli & Panconesi, 2022](https://arxiv.org/abs/2103.17228)).

A 10x128 checkpoint scored, out of 256 games against each level (128 openings), searching 400 MCTS
iterations a move:

| Edax level | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Score | 0.79 | 0.49 | 0.23 | 0.07 |

### On a clock

On a clock, `evaluate.py` plays games the way a contest would: one at a time, the same seconds for
each side for the whole game, on the same cores. Each engine runs its own time management at its
strongest settings: Edax over GTP (`time_settings`, then `time_left` before each move), and
Egaroucid over its console protocol (`-time`, `play`, `go`), since its GTP has no clock.

```
uv run examples/othello/evaluate.py run --cores 16-23 --openings 100 \
    --a edax:clock=10,threads=8 --b egaroucid:clock=10,threads=8
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

### More engines

Four more engines, each plausibly near Edax's strength or above, play in `evaluate.py`, on a clock
or (but for ymatioun) at a level. Each `setup_*.sh` builds one at a pinned commit
into `~/.cache/turbozero/othello/engines-2026-11/`, for a standard ISA level, never `-march=native`.
None has an opening book.

| Engine | Name | Licence | Protocol | Levels | Notes |
|---|---|---|---|---|---|
| [Melody](https://github.com/HCyrano/Melody), Bruno Causse | `melody` | GPLv3 | its own, IOStd: a whole position and a depth or the milliseconds left | Edax's (`engines.edax_level`) | AVX2 at most. Its time management keeps up to 15 s in reserve and takes at least 1 s a move: it lost on time at 10 s a game, not at 60 |
| [Neural Reversi](https://github.com/natsutteatsuiyone/neural-reversi) | `neural-reversi` | GPLv3 | GTP, with `setboard` | its own, up to 30 | Edax-like search, neural network evaluation; `time_left` in whole seconds |
| [ymatioun](https://github.com/ymatioun/reversi), Youri Matiounine | `ymatioun` | MIT | its own, like UCI: `position fen`, `go time <ms>` | none: clock only | threads fixed at build (`bin/ymatioun-t<N>`, `THREADS="1 8"`); the setup script patches its weights' paths, threads, hash size and page-rounded allocations |
| [Edax-FM](https://github.com/Yoka0014/edax-fm-reversi) | `edax-fm` | GPLv3 | Edax's GTP | Edax's | Edax 4.6 with a factorization-machines layer; "Kalmia" on GGS |

```
examples/othello/setup_melody.sh                    # once each
uv run examples/othello/evaluate.py run --cores 8 --openings 50 \
    --a melody:clock=60,threads=1 --b edax:clock=60,threads=1
```

### GGS conditions

Programs play each other on GGS, Michael Buro's game server, mostly in `s8r14` synchro matches:
two games from one random 14-disc position, colours swapped, on a clock of `ini/inc/ext` (most
often `01:00//02:00`: a minute a game, and once that's out, a loss by 2 discs at best and two
more minutes to finish). `ggs.py` has its random starts, clock and scoring, and `evaluate.py`
plays them:

```
uv run examples/othello/evaluate.py run --cores 16-23 --starts random:14 --openings 50 \
    --a net:net=NAME,engine=1,clock=01:00//02:00 --b edax:threads=8,clock=01:00//02:00
```

- **The starts are far from XOT's.** The server makes half of them by dropping discs near the
  centre and half by random moves from 5 central discs (`ggs.random_start`). The two ways give
  different positions, and only their even mix matches the archive's 43,000 `s8r14` starts of
  2024-2026 (disc balance, squares filled, and how lopsided Edax finds them). Edax at level 12
  puts a third of them at 8 discs or more for one side, where every XOT opening is within 2.
  None can be reached from the standard start, so each player is given the board
  (`set_position`): Edax by `loadsgf` of a GGF file, Egaroucid by its console's `setboard`, a
  network by a fresh tree.
- **Pairs are the unit.** Top programs split 90-95% of GGS's pairs 1-1, so a match's signal is in
  the decisive pairs and in the two games' disc results summed, which is how GGS scores a pair.
  `evaluate.py show` reports both (`pairs`), with the Elo of the decisive pairs alone.
- **One game at a time by default.** On GGS a pair's games run at once, in lockstep: each round,
  the player to move in each game moves, at the same time. `--synchro` plays them so, on the
  same cores, with a second network for a net player.
- `ggs.py extract` takes an archive's starts (e.g. `Othello.2010-11-2026.ggf.bz2` from
  https://skatgame.net/mburo/ggs/game-archive/Othello/) for `--starts file:PATH`, and `ggs.py
  compare` sets them beside `ggs.random_start`'s.

### On Google Cloud CPUs

Timed games need the same CPU every time, and Modal's are a mix (some AMD Zen 3 without AVX-512,
where the native network runs its AVX2 kernel at about a quarter of the speed). `gcp_cpu_jobs.py`
runs commands on Spot VMs of one model: `n4d-highcpu-16` with one thread a core, 8 physical cores
of AMD Turin (EPYC 9B45, Zen 5, AVX-512 VNNI), the same generation as razorback's 9900X3D but at a
lower clock (4.1 GHz at most). `evaluate.py run --on gcp` plays a match on them; underneath, any
command can run there:

```sh
uv run examples/othello/gcp_cpu_jobs.py setup --project PROJECT        # once; the bucket is modal_jobs.py's
uv run examples/othello/gcp_cpu_jobs.py run --name check-1 --jobs jobs.json --out results.jsonl \
    --vms 2 --files quantized/4x64-...-native.npz
uv run examples/othello/gcp_cpu_jobs.py follow check-1                 # lines jobs add to their outputs
uv run examples/othello/gcp_cpu_jobs.py list
uv run examples/othello/gcp_cpu_jobs.py cleanup                        # every job's VMs
```

`jobs.json` is a list of `{"name": ..., "cmd": [...], "outputs": [...]}`: each command runs in
`examples/othello` with the environment's python first on `PATH`, and the last line of JSON it
prints is its result. Files named in `outputs` (e.g. `evaluate.py play`'s games) go to the
bucket every 20 seconds while the job runs, which `follow` prints line by line, and come back
beside `--out`, in `OUT-files/SLOT/NAME/`, when it ends. The jobs are dealt out over the VMs; with
`--each`, every VM runs every job, to compare VMs. A VM takes this checkout's files as they are
(new and uncommitted ones too), the engines' builds and each `--files` path under
`~/.cache/turbozero/othello`, waits for its CPUs to go idle (Ubuntu's own startup work skewed a
first timed job by 11%), runs its jobs, sending each result to the bucket as it ends, and deletes
itself. A preempted VM is replaced, the replacement skipping what's done; `run` again with the
same `--name` continues a job cut short. Every VM is deleted after `--hours` whatever it's doing,
and `run` deletes any left when it ends.

The first VM of a `uv.lock` builds the environment (g++ from Ubuntu's mirror, uv, the venv from
PyPI: 3-13 minutes) and leaves it in the bucket; VMs after it start from that, ready about 40
seconds after their startup script begins (a minute or so after they're created).

Two VMs playing the same games agreed within 1-4% on each game's time where both played the same
moves. The engines took about 1.45x their time on razorback's CPUs 16-23 (1.24-1.7x by level),
Egaroucid a little more than Edax, so these VMs need their own timings rather than razorback's
scaled.

### Against an engine's scores of fixed positions

Games against engines measure strength, but coarsely: 512 games give a score within about
±0.02, and a run's score moves by more than that from training randomness alone. The
engine-labelled set measures the network more finely, without search or games: thousands of
fixed positions past the XOT opening and with a choice of move, each legal move of which an
engine has scored: the disc margin it expects at the end. Sets record what scored each
position (`labeller_names`): Edax at level 24 (24 moves ahead, the last squares exact), or
Neural Reversi at level 26 (exact up to 30 empty squares, 26 moves ahead beyond).

`train.py` scores the weight average on it every epoch and logs, as `avg4/engine_*`, for all
positions and for each game phase (`opening`, 40-51 empty squares; `midgame`, 25-39; `endgame`,
1-24):

- `engine_top1`: how often the network's most likely move is one of the engine's best
- `engine_regret`: discs its most likely move gives up against the engine's best
- `engine_exp_regret`: the same, averaged over its policy: smoother, and about three times
  less noisy than `engine_regret` (a difference between two checkpoints on the same set is
  good to about ±0.015 discs). But it also measures how sharp the policy is, which the
  search sets (more sims train a sharper one): compare it only between runs with the same search,
  and across them, `engine_regret` and `engine_value_mse`
- `engine_value_mse`: its value's squared error against the outcome the engine predicts (±1, 0 for a
  draw)
- `engine_value_sign`: how often its value calls the winner the engine does

`score_engine_set.py` also reports the same metrics as `filtered_engine_*` over only the positions
the network decides: from the XOT opening's to the exact solver's, which plays and labels
positions with 20 empty squares or fewer. They leave out positions whose moves the solver makes in
play, and are reported for all of them and in halves (`opening`, 36-51 empty squares; `midgame`,
21-35).

Scoring takes a forward pass over the positions, about 20 ms for 5000 on the GPU.
`--engine-set PATH` scores on another set, and `--validate-every N` sets the cadence (0
for never). Without a set at the default path, `~/.cache/turbozero/othello/engine-sets/nr26-evals-29k.npz`,
training warns and goes on unscored; a missing `--engine-set` is an error.

The default set has 29,100 positions with 21-51 empty squares from the evals store's games of
2026-11 (networks against six engines at 30 s, the engines' round robin, and 2048-visit head to
heads between 4x64 and ray-filter networks), scored by Neural Reversi at level 26. Its positions
are close to even (73% within 2 discs of a draw), so its values and errors run higher than the
earlier Edax set's (`edax-l24-5k.npz`, 5,000 self-play positions): compare runs on one set. On
29 rated 11x64 checkpoints its `value_mse` ranked them by 2048-visit Elo at Spearman 0.78
(0.54 on the Edax set), `top1` at 0.81 (0.13).

To build a set, from the games in the evals store (evaluation games, which nothing trains on):
sample positions from kinds of match, score every legal move of each with Neural Reversi
(`setup_neural_reversi.sh`, with `ARCH=x86-64-v3` for Modal's CPUs) on Modal's CPUs, and put
them together (see label_positions.py for the sources and the defaults):

```
uv run examples/othello/label_positions.py positions --out ~/.cache/turbozero/othello/sets/positions.npz \
    --source 'engine30-ray11=1300:net-engine:ray11x64' --source 'engines30=2700:engine-engine' \
    --source 'h2h-ray11-ray9n=4300:net-net:ray11x64/ray9x64n'
uv run examples/othello/label_positions.py cloud ~/.cache/turbozero/othello/sets/positions.npz \
    --name set1 --hosts 100 --out ~/.cache/turbozero/othello/sets/labels/results.jsonl
uv run examples/othello/label_positions.py assemble ~/.cache/turbozero/othello/sets/positions.npz \
    ~/.cache/turbozero/othello/sets/labels --out ~/.cache/turbozero/othello/engine-sets/new.npz
```

At level 26, Neural Reversi scores a position's moves exactly up to 30 empty squares (on a
razorback core 8 s at 21-26, 20-160 s at 27-29, minutes at 30) and 26 moves ahead beyond
(about 4 s); Modal's cores take about twice as long. 30,000 positions spread over 21-51 empty
squares took about 600 core-hours on Modal, about $45.
To score saved checkpoints on a set:

```
uv run examples/othello/score_engine_set.py runs/othello/avg4/49.eqx runs/othello/avg4/99.eqx --net resnet:10x128
```

## Playing on CPU

Self-play's search is built for thousands of games at once on a GPU. Alone on CPU it spends about
12 times what the network costs: 86 evals a second for an 8x128 net whose forward pass alone does
1056. `cpu_search.py` plays one game on CPU instead: `evaluate.py` plays it, as `net:`, against
another or an engine, on a per-game clock or at fixed visits.

```
uv run examples/othello/evaluate.py run --cores 16-23 --openings 20 --b edax:clock=10,threads=8 \
    --a net:net=8x128-128s-8p-w1125-274m,clock=10,batch=16,virtual_loss=visit,random_symmetry=1,taper=24
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

A player's cost is timed alone: a match of the player against itself from XOT openings, only the
side to move charged, move by move. Stop everything else on the machine first (other load costs
engines 30-45%); `--require-idle` won't start unless the cores and their SMT siblings are idle. The
store keeps every move's seconds, and `timings` sums them up by host, for the league's plot:

```
uv run examples/othello/evaluate.py run --cores 16-23 --require-idle --openings 12 \
    --a edax:level=20,threads=8 --b edax:level=20,threads=8
uv run examples/othello/evaluate.py timings                          # the hosts with timed games
uv run examples/othello/evaluate.py timings --host HOST --name razorback-8t-16-23 --publish
```

`--publish` puts the host's run in the league's `timings.json`, replacing one of the same name: each
player at a fixed budget that played itself there, its games played out. One side's seconds a game
on 8 threads of those cores, with bf16 networks, on 2026-11-01:

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
uv run examples/othello/evaluate.py run --cores 16-23 --openings 100 --b edax:clock=4,threads=8 \
    --a net:net=8x128-128s-8p-w1125-274m,clock=4,batch=16,virtual_loss=visit,taper=24,backend=int8-performance_leaves_heads_minmax
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

### The engine and the native network

A network's `engine=1` searches with a thread to a core, all in C++ (`cpu_search.Engine`): each
collects its own few leaves, evaluates them on its own core while the others carry on, and backs them
up, the tree shared under a lock. The batch above waits on its slowest stream, and goes through Python,
at every step; here the network's time is 90-97% of the search's. `backend=native` evaluates with
`cpu_net.py`'s network, our own int8 convolutions with float between them, a leaf at a time, so each
thread needs only one leaf in flight. The library is built for a standard level of the CPU's
architecture, as engines ship builds for (`native_target.py`). On x86: AVX-512 with VNNI (x86-64-v4)
where the CPU has them, else AVX2 (x86-64-v3), with AVX-VNNI where the CPU has it (Intel's client
chips from Alder Lake, which lack AVX-512). On ARM: ARMv8.2 with the dot product and the 8-bit matrix
multiply (i8mm: Apple's M2 on, Graviton 3 on, Axion), else with the dot product alone (Apple's M1,
Graviton 2). `TURBOZERO_CPU_LEVEL` (`v4`, `v3-vnni`, `v3`, `arm-i8mm`, `arm-dotprod`) picks one,
e.g. to play as an AVX2 build would. Every kernel sums exactly the same integers:

- AVX2's 3x3 convolutions run Winograd's F(2x2, 3x3) in integers: each 2x2 tile's 4x4 input patch and
  the weights are transformed into values that fit 16 bits, multiplied in pairs summed into 32 bits,
  and transformed back, in 16 products where the direct convolution takes 36. With AVX-VNNI they run
  directly, as AVX-512's do at half the width: `vpdpbusd` takes 4 byte products a lane, so the direct
  convolution's 36 products cost 9 instructions where Winograd's 16 cost 8, without the transforms.
- NEON's multiply signed bytes: the inputs go in less 128, and 128 times each output's weight sum is
  added back. `sdot` takes 4 channels of a square against 4 outputs' weights; i8mm's `smmla` takes 8
  channels of two neighbouring squares against two outputs' weights, twice the products an instruction,
  from the input laid out in such pairs.

Evals a second on one core, a leaf at a time:

| Core, kernel | 8x128 | 4x64 |
|---|---|---|
| Zen 5, AVX-512 VNNI | 3615 | 22264 |
| Zen 5, AVX2 Winograd (a direct 16-bit kernel) | 1729 (933) | 11618 (6650) |
| Axion (Neoverse V2), i8mm (`sdot`) | 1442 (974) | 7868 (5931) |
| Apple M4, i8mm (`sdot`) | 1450 (1490) | 9330 (9690) |

Our engine's seconds a game over the engine's, at the visits that tie with each level on Zen 5 (the
8 pairings of Edax and Egaroucid at levels 20 and 24 with 8x128 and 4x64 nets, 4 threads a side, whole
cores):

| Machine, our build | Against Edax | Against Egaroucid |
|---|---|---|
| Zen 5 (EPYC 9B45), AVX-512 | 0.89-1.05 | 0.89-1.02 |
| Zen 3 (EPYC 7B13), AVX2 Winograd (direct 16-bit) | 1.92-2.34 (3.29-4.61) | 2.06-2.43 (3.60-4.25) |
| Axion, i8mm (`sdot`) | 1.34-1.61 (1.94-2.35) | 0.84-1.08 (1.20-1.43) |
| Apple M4 (MacBook Air, in use during the run), i8mm | 2.08-2.72 | 2.10-2.58 |

Egaroucid's SIMD is x86's: on ARM it runs its generic build, as it ships for Apple's chips. The
engines' branchy search runs best on Apple's cores (Edax at level 20 1.9 s a game, Zen 5 2.0-2.3,
Axion 2.6), while the network's 128-bit NEON doesn't, so the M4 is our worst case.
Build it once a network, calibrated as `quantize.py` is:

```
uv run examples/othello/cpu_net.py --net 8x128-128s-8p-w1125-274m --positions LEAVES/calibration.npy
uv run examples/othello/evaluate.py run --cores 16-23 --openings 100 --b edax:clock=4,threads=8 \
    --a net:net=8x128-128s-8p-w1125-274m,clock=4,engine=1,backend=native,batch=1,virtual_loss=visit,taper=24
```

A leaf whose position the game's searches have already evaluated, reached by another order of moves,
takes that evaluation from the tree rather than the network's (`reuse`, on by default): 14-17% of
leaves at 1000-4000 visits, about 27% under 10 empties. Without random symmetry the search is
otherwise the same.

The native network's convolutions take each input channel's own scale, folded into the weights
(SmoothQuant, alpha 0.75), then the tensor's, from the minimum and maximum seen in calibration; their
normalization of each square's channels runs in the pass that quantizes the next one's input, where
OpenVINO runs separate ops (`profile_native.cpp` times each stage). The stem's weights take 16 bits,
since its sums are looked up from a table of 16-bit integers anyway. After quantizing, each
convolution's biases, in order through the network, and the value's last bias are moved so that their
outputs' means on the calibration leaves match the float network's (bias correction): rounding errors
that don't average out otherwise add up to a shift of the whole value. On five ray and 4x64 nets (2026-11),
the two cut the value's mean error on held-out search leaves from 0.017-0.028 to 0.013-0.017, and on
the engine-labelled set from 0.014-0.024 to 0.010-0.014; a normalized ray 9x64's from 0.028 to 0.015,
of which its 8-bit stem's rounding had made a shift of about -0.012. Against the float
network on 2000 search leaves, before both (value's mean error, its 99th percentile, top move):

| Network | Native | OpenVINO int8 |
|---|---|---|
| 8x128 | 0.010 / 0.052 / 97.9% | 0.043 / 0.191 / 96.6% |
| 4x64 | 0.017 / 0.095 / 96.7% | 0.079 / 0.323 / 91.1% |
| 8x128, channel norm | 0.024 / 0.116 / 95.2% | 0.078 / 0.352 / 86.7% |
| 4x64, channel norm | 0.024 / 0.094 / 92.8% | 0.058 / 0.223 / 80.2% |

Evals a second on one core, the native network a leaf at a time against OpenVINO int8's best batch:
8x128 3908 against 4060 at 4; 4x64 26362 against 24840; 8x128 with channel norm 3548 against 3345;
4x64 with channel norm 21970 against 16170. Channel norm costs OpenVINO int8 19-37% over BatchNorm,
the native network 9-17%.

On a 4 s clock, 4 physical cores (both sides the same 4 threads), XOT openings with both colours, 200
games each (standard errors over the openings' pairs of games):

| Player a against player b | Network | Elo for a |
|---|---|---|
| Engine, OpenVINO int8, 2 leaves a thread, reuse; against the batch of 16, no reuse | 8x128 | +133 ± 18 |
| The same engine, reuse against none | 8x128 | +44 ± 14 |
| Engine with the native network, 1 leaf a thread; against OpenVINO's engine, 2 a thread | 8x128 | -2 ± 17 |
| The same | 8x128, channel norm | +31 ± 20 |
| Engine, native network, 1 leaf a thread, reuse; against the batch of 16, no reuse | 6x96 | +141 ± 19 |
| The same | 4x64 | +215 ± 21 |

Leaves in flight cost a little search quality, and how much depends on the net and the visits. At
equal visits, 8 threads against the batch of 16, the native network on both sides, 300-400 games each:

| Leaves a thread | 1 | 2 | 4 |
|---|---|---|---|
| 8x128, 1000 visits | +9 ± 11 | -10 ± 13 | |
| 8x128, 4000 | +9 ± 9 | +5 ± 12 | -9 ± 11 |
| 6x96, 4000 | -13 ± 12 | -6 ± 12 | -22 ± 11 |
| 4x64, 4000 | -31 ± 12 | -9 ± 12 | +13 ± 14 |

The native network costs nothing per core to run a leaf at a time, so the choice is search quality
alone: 1 a thread for 8x128; small nets at high visits want 2-4. It matched bf16 at
equal visits: -3 ± 12 on 4x64 and on 8x128 with channel norm, 400 games each.

Against engines the gain is smaller than against the same net. At 10 s a game, 4 physical cores and
4 threads a side, the 700-minute 8x128 scored 35.5% against Edax with the engine and native network
(1 leaf a thread) and 31.5% with the batch of 16, on the same 100 games' openings and colours: +4.0
± 3.8 points, about +31 ± 30 Elo. With random symmetry it scored 37.5%, and against Egaroucid 41%.
Edax spent 0.03 s a game on the last 23 empty squares, the net 1.6 s, but the games were decided
earlier: solved exactly, perfect play from 24 empties would have scored 36% instead of 34% (100
games), and from 28, 37.5%. The net gives up about 5 discs from 24 empties, and rarely the result.

Below AVX2 (or the dot product on ARM) the library builds without the native network, and `Engine` runs OpenVINO.

### Solving the endgame

With `solve=N`, a `net:` player solves positions with N or fewer empty squares exactly instead of
searching them (`endgame.py`), on the same threads, its time charged to the clock:

```bash
uv run examples/othello/evaluate.py run --cores 16-23 --openings 100 --b edax:clock=10,threads=8 \
    --a net:net=8x128-128s-8p-w1125-274m,clock=10,engine=1,backend=native,batch=1,virtual_loss=visit,solve=20
```

By default it solves for a win, a draw or a loss, about ten times faster than the margin
(`solve_exact=1`) and all a result needs: it plays a winning or drawing move, and a lost position is
searched as usual, the network's best try against a fallible opponent. On a clock the game plans
its solve as Edax does: from the most empties, up to N, whose expected solve takes at most a quarter
of the game's clock (`solve_share`), the moves before it sharing the clock less what's kept for
it: that expected cost, at least 4 s or 5% of what's left (`Solving.kept`). The expected
cost (`solve_cost`, seconds at 24 empties on 8 threads, doubling each empty more) is by default
Edax's own exact solver's, 0.19 s, the speed we assume ours will reach; ours took about 2.0 s for a
win, a draw or a loss there (2026-11-10). A solve may take all but a tenth of what's left, else it
gives up and the move is searched; once one succeeds the moves after it take milliseconds, its
transposition table kept from move to move. With `--solve-at N`, give the network `solve=N` too: it
then budgets its clock for the moves before it. `solve_batch`
solves many positions at once, a thread to each, for training labels.

The solver is alpha-beta over bitboards: the last four empties tried in place, odd quadrants
first; above them moves ordered fastest first (fewest replies), principal variation search, cuts by
the opponent's stable discs, and from 8 empties a transposition table, which from 10 is looked at
for each child's bound before searching it. Its exact scores match Edax's `-solve` (tests run the
comparison where Edax is built). On one core it visits about three times Edax's nodes for an exact
solve at 18-20 empties, at a similar rate a node; Edax narrows its window with its evaluation first.

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

`timings.json`, beside the store, holds what players' moves cost on fixed hardware. A benchmark
writes it atomically:

```json
{"runs": [{"name": "...", "hardware": {"cpu": "...", "cores": 8, "threads": 8}, "software": {},
  "configs": [
    {"net": "resnet:8x128,...", "sims": 1000, "batch": 16, "batch_strength": "measured_no_loss",
     "source": "measured", "plot": true, "games": 24, "seconds_per_game": 2.1},
    {"engine": "edax", "level": 20,
     "source": "measured", "plot": true, "games": 24, "seconds_per_game": 1.4}]}]}
```

A run is one host. `seconds_per_game` is one side's mean. Configs are joined to players by
architecture and sims or engine and level; a `batch_strength` of `unmeasured` means the batch size
hasn't been shown to lose no strength.
