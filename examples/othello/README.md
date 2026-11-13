# Othello

AlphaZero on Othello: training, and evaluating the result against strong open-source Othello
engines, chiefly [Edax](https://github.com/abulmo/edax-reversi) and
[Egaroucid](https://github.com/Nyanyan/Egaroucid).

| File | |
|---|---|
| `train.py` | Trains a network by self-play, scoring it against Edax's scores of fixed positions as it goes, and in games against Edax if asked |
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
| `engine_set.py`, `label_positions.py`, `score_engine_set.py` | The engine-labelled set: fixed positions with every legal move scored by Edax, to build, and to score checkpoints on (train.py scores the network on it every epoch) |
| `distill.py` | Trains a network on a saved training state's replay buffer or a dataset, with no self-play, on a teacher network's values or the games' outcomes |
| `teacher_dataset.py` | A dataset of whole self-play games from the moves a run kept (`--log-moves`), labelled with exact endgame results |
| `bench_nets.py` | Times networks on their own: a forward pass at self-play's batch, a training step at training's |
| `game.py` | What the scripts share: the environment, board symmetries, network, evaluators |
| `xot.py`, `xot-openings.txt` | The XOT opening list |

## Training

```
uv run examples/othello/train.py --monitor --ckpt-dir runs/othello --notes "the baseline"
```

`--name` names the run on the server, and `--notes` says what it's testing and why, at the
top of its page.

Self-play runs in 1024 games at once, searching 64 MCTS iterations a move. Replay memory
stores each position once, and training maps each position it samples through one of the
board's 8 symmetries (rotations and reflections), drawn at random. Games start
from XOT openings rather than the standard start, which keeps self-play varied
(`--standard-starts` mixes in the standard start).

A run is tracked by the engine-labelled set: every epoch, `train.py` scores the raw network
on Edax's scores of 5000 fixed positions, in about 20 ms, and logs `engine_exp_regret` (the
discs its policy gives up against Edax's best move) and the rest of the `engine_*` metrics
(see [below](#against-edaxs-scores-of-fixed-positions)). They're much finer than games: a
difference between two checkpoints is good to about ±0.015 discs. Games measure strength
itself: `evaluate.py` plays checkpoints against engines and each other (see [Evaluating a
checkpoint](#evaluating-a-checkpoint)), and the Elo league rates them.

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
keep changes few; games in progress carry on, searching from new, empty trees.
Evaluation games keep their own budgets. The run's page plots the budget as
`search_iterations`.

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
`resnet:6x128` (below) computing in float32 throughout, an epoch takes about 63 seconds.
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

### Exact endgame labels

`--solve-empties N` stops each self-play game once its position has N or fewer empty
squares. That position is searched and stored like any other, and an exact solver's result
for it (win, draw or loss for the player to move) becomes every position's value target,
instead of the result the game would have had if played out. Self-play skips the searches
for the last N moves and plays more games in the same compute. The labels lose the noise of
endgame mistakes: in F2 e440's self-play (the 8x128 at 511/127 sims), the played result
differs from the exact one at 16-20 empties in 17% of games, mostly a win or loss played
as a draw or the reverse.

The solver (`--solver native`, endgame.py, a win/draw/loss solve; `edax` for comparison)
runs on the CPU while the GPU trains and plays. Games stopped in an epoch's self-play are solved
through its training and the next epoch's self-play, and trained on from then. A new run's first epoch solves them at once,
since it has nothing else to train on. At N = 20, a solve averages about 80 ms a core on
F2's positions, and an epoch at F2's settings should stop about 4,600 games (152,000 moves at
about 33 a game from an XOT opening), so the solver needs
3-4 cores to keep up with a 5080 (`--solve-threads`, default every core the process may
use; on Modal, `--cpu 8`). `solve_wait_seconds` shows how long self-play waited for it,
and `solve_games`, `solve_draw` and `solve_missed` count the games.

Stopped games never produce positions with fewer than N empty squares, though searches from
just above N still reach them, and their leaves there are valued by the network.
`--solve-play-past MEAN` plays each game on past N, ending before each further move with
probability 1 / (1 + MEAN): MEAN moves on average, a geometric number. Positions up to N keep the
solved result at N; each position stored past N is solved on its own and gets its own exact
result. Solves below N are cheap (24 ms at 18 empties, a few below that), so at MEAN 3.5 the
solver's time goes up about 15% a game. `solve_positions` counts every solve. `--solve-share P`
stops only a fraction P of games and plays the rest to the end, unsolved.

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
progress with their search trees and the rng.
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

`--branch-from` continues a run from a state in a new `--ckpt-dir`, with other self-play
settings: a branch of the run rather than a new one. It goes on at the state's epoch (`--epochs`
counts the epochs before it too), with every network and optimizer state, the learning rate
rule's and growth's progress, the buffers, the games in progress and the rng. Only a search of
another size starts its trees anew, its moves in progress searched again; with the run's own
settings a branch continues it exactly. Two branches of one state, one at `--sims 128` and one
at `--sims 64`, compare the two at the same point of the same run. It works for runs with
`--grow`, which can't be forked, and `--take-over` hands self-play to the next network from
the branch's first epoch. A different `--buffer` resizes the replay buffer at the branch's
start, each environment keeping its newest positions. The branch logs to a seaglass run of its
own, whose config records the state as `parent`.

```
uv run examples/othello/train.py --epochs 180 --sims 128 --branch-from runs/trunk/state-150.npz \
    --save-state-at 180 --ckpt-dir runs/trunk150-sims128
```

### Growing the network

Self-play costs far more than training. KataGo started with small nets and trained the next
size up on the same games, switching when it caught up. `--grow NET` (repeat it for several,
smallest first) does the same: `--net` plays self-play, and the next network trains on the same
replay buffer each epoch without playing. When it takes over, the one before stops, and the
one after that, if any, starts training from scratch on the buffer as it is. So two networks
train at a time.

The next network takes over when its `--grow-metric` on the engine-labelled set
(`engine_value_mse`, scored every epoch) is at or below the playing one's, or with
`--grow-metric loss` its training loss on the data both train on, as KataGo's rule had it.
`--grow-margin 0.01` makes the next lead by 1% of the playing one's metric, and `--grow-guard
engine_value_mse` makes it also no worse on the engine set's value error. Each is averaged over
the last `--grow-window` (10) epochs, and the next must have trained `--grow-min-epochs` (10) first.
With `--grow-switch-at EPOCHS`, it takes over after fixed numbers of epochs instead. Either way,
a playing network whose `--lr-plateau-stop` rule is done hands over rather than stopping the
run. A waiting network trains at `--lr`, and follows `--lr-plateau` from its first rate once it
plays. `--grow-train-retired` keeps the networks that played before training on the replay buffer,
`--train-steps` each an epoch, without playing; a resumed or branched run turns it on or off.
`--grow-train-steps N` has the waiting network take N training steps an epoch
rather than `--train-steps`, e.g. what its own play would afford for the same compute. With
`--grow-manual`, nothing takes over by itself: the run logs `growth_caught_up` (1 once
the rule would switch), saved states record it too, and the next network takes over in a branch
of the run, with `--branch-from` and `--take-over` (see the run controller below). A branch can
also grow past the networks the run started with: `--grow` networks after the saved state's
last start from scratch, as one that hasn't joined a run yet does, so a run whose last network
plays can go on into a bigger one.

Each network's checkpoints go to `--ckpt-dir/<its size>`, e.g. `runs/grow/2x32` and `runs/grow/4x64`.
Saved states hold every network, in `--ckpt-dir`. Each network's metrics are logged as
`<its size>/<metric>`, and the playing network's also without the prefix. The run logs which
network played (`growth_playing`), the next's mean metric minus the playing one's (`growth_gap`),
and the switch (`growth_switch`). Every run logs its compute per epoch, to compare runs at equal
GPU time: `selfplay_seconds`, `train_seconds` and `test_seconds` (with `--grow`, each network's
as `<its size>/train_seconds` and `<its size>/test_seconds`), `epoch_seconds`, and the run's total,
`run_seconds`. It also logs it in network evaluations, which don't depend on the hardware:
`selfplay_evals`, the positions self-play evaluated (search cycles times environments, since
every environment pays every cycle), and `train_samples`, the samples trained on (with
`--grow`, each network's as `<its size>/train_samples`, and the evaluations under the playing
one's name too).

```
uv run examples/othello/train.py --net resnet:2x32,pool_every=2,heads=pooled,pool_channels=8 \
    --grow resnet:4x64,pool_every=2,heads=pooled,pool_channels=16 --engine-eval-every 1 \
    --lr-plateau --lr-plateau-stop --epochs 600 --ckpt-dir runs/grow
```

### Weight averages

At a constant learning rate the network's weights jitter from epoch to epoch, and so does its
play: KataGo played and rated averaged weights. `--average-epochs 4 16` keeps averages of each
network's weights over about 4 and 16 epochs: the mean of the epochs so far, then an exponential
moving average that moves 1/H of the way to each epoch's weights. Training itself is unchanged,
and so is the run's rng: a run with averages is the run without them, plus each average's
checkpoints in `avg<H>` in the network's checkpoint directory (e.g. `runs/grow/8x128/avg4`) and
its scores on the engine-labelled set, `avg<H>/engine_*`. Saved states keep them; a state saved
without them starts them at its weights. `--selfplay-average 4` has self-play play with the
4-epoch average instead of the current weights.

What the jitter costs can be watched, off by default: `--fresh-positions N` scores the playing
network every epoch on N of the positions its self-play just stored, before it trains on them,
its raw weights and its self-play average: their policy KL and value loss (`fresh_*`), and the
raw weights' excess (`fresh_*_gap`). N is a multiple of the micro-batch (`--train-batch` /
`--grad-accum`), e.g. 16 of them. It draws its samples from the run's rng, so a run with it
plays other games than the same run without.

### Keeping every move: datasets of games

`--log-moves DIR` keeps every move self-play plays, fast searches' too, in `DIR` as
`moves-<EPOCH>.npy` (see `core/memory/move_log.py`): the position, the search's policy target and
root value, the legal moves, the move played, whether the search was full, the ply, and the game
it's in, about 190 bytes a move. Training is unchanged. The log isn't part of saved states, so a
run keeping its moves doesn't `--resume` into the same directory: start another into a new one.

`--train-steps 0` trains nothing: the network and its weight averages only play. With
`--init-from` a state and `--selfplay-average`, a fixed teacher plays its own self-play settings'
games, for a dataset that students and architectures train on from disk instead of from live
self-play:

```bash
T=~/.cache/turbozero/othello/solo8x128/s14t-s13te340-lr2e6-511s-solve20-past3.5-tandem4x64-5e5
uv run examples/othello/train.py --net resnet:8x128,pool_every=2,heads=pooled,pool_channels=32,norm=channel,layout=pre \
    --init-from $T/state-380.npz --fork-into 8x128 --init-without-buffers \
    --train-steps 0 --average-epochs 4 --selfplay-average 4 \
    --sims 511 --playout-cap 0.25:127 --tree-factor 4 --advance-every 128 --selfplay-cycles 33280 \
    --solve-empties 20 --solve-play-past 3.5 --engine-eval-every 0 \
    --epochs 160 --ckpt-dir runs/teacher --log-moves runs/teacher/moves
uv run examples/othello/teacher_dataset.py build --moves runs/teacher/moves --out runs/dataset \
    --solve-empties 20 --threads 8
```

`build` puts the games back together, drops those the run ended in the middle of, and labels
every position with its game's exact result for the player to move: positions with 20 empties or
fewer are each solved on the CPU (win, draw or loss, about 37 ms of a core each), and those before
a game's first one take its result, as self-play's exact labels do. It holds out whole games
(`--heldout`, 3% by default). The dataset is a directory of columns, a `.npy` per field and a row
per position, game by game (see `COLUMNS` in `teacher_dataset.py`), so a later relabelling (e.g.
a deeper search of some positions) is another column beside them. `distill.py train --data DATASET
--heldout DATASET` trains on its training games' full searches and measures on its held-out ones;
`--all-positions`, `--min-empties` and `--max-empties` change which positions.

### The run controller

`controller.py` runs a long growing run in steps, each its own entry in slick's GPU queue, so
other work can be slotted in between them, and decides between steps what the run does next.
Each period:

1. **Trunk**: the run trains on at its current settings, for `trunk_turnovers` turnovers of
   its replay buffer, the next network training beside the one playing, on its games. No
   network takes over during it (`--grow-manual`); the next's training loss against the
   playing one's is logged (`growth_gap`, `growth_caught_up`), but nothing switches on it. The
   next network trains the steps its own play would afford for the same compute
   (`--grow-train-steps`): the playing one's steps times the ratio of the two regimes'
   benchmarked epoch costs, each regime with its own waiting network (solved from the last
   back). With 8x128 playing at 255/63 sims, 12x144 takes 146 of 256 steps; the smaller steps
   of the ladder about 40%. So when the test plays the two, the bigger one has trained about
   what the run would have spent on it as the one playing.
2. **Size**: if there's a next network, it plays the one playing, both as they trained in the
   trunk, each with its weights averaged over the last `average_epochs` (4) epochs
   (`--average-epochs`; see weight averages above). If it scores `margin_se` (2) standard
   errors of its series or more, the run grows: the next network takes over from the trunk's
   state at the next trunk (`--take-over`), and the period needs no branches. Growing comes
   first: the search only goes up while the next network doesn't pay its way, a bigger network
   being what makes more search worth it.
3. **Sims**, otherwise: two branches of the run (`--branch-from`) at equal benchmarked compute,
   in which no network takes over either. `double` searches twice the iterations, both budgets
   with playout cap randomization, for `branch_turnovers` turnovers of its buffer; `stay` keeps
   the search for as many epochs as cost the same. If `double`'s playing network scores
   `margin_se` standard errors or more against `stay`'s, the run goes on from `double`, else
   from `stay`; `double`'s next network plays `stay`'s too, for the record. One branch is
   thrown away: longer trunks (`trunk_turnovers`) test less often. With `selfplay_average`,
   self-play plays the averages too (`--selfplay-average`).
4. **Ladder**: the network the run kept plays the league's Edax players in turn (`ladder`,
   edax-L1 up to edax-L24), a series each, from the rung the last period lost to, on to the next
   while it scores over half. Its games put the kept line on the league's scale.

Every network is rated at 128 sims, the best model per search; for the record, the run also
plays them at equal time. `clock_sims` gives each size the sims that take it about 1 s of one
side's clock a game on the contest CPU path (bf16 OpenVINO, 8 threads, batched search, from the
league's `timings.json`): 4x64 3000, 6x96 1100, 8x128 520, 12x144 250. The size step plays the next
network against the one playing at those sims too (`size on the clock`), and the kept network
climbs a second ladder of its own at its clock's sims (`pNNN-clock-ladder`). Those players have
4 tree nodes an iteration (`clock_tree_factor`): at thousands of sims, the default 2 costs
strength. Nothing is decided on the clock's games.

With `train_retired`, the networks that played before keep training on the run's games, as many
steps as the one playing (train.py `--grow-train-retired`), and the period's first test (the size
step, or the sims step once the last network plays) plays each of them against the one playing
(`retired <size>`): whether a smaller network keeps up on a bigger one's
games, or falls behind for want of capacity. It isn't compute-optimal (about 10% more compute,
15% with three retired), so it's counted in the run's compute and on its own
(`retired_seconds`), but not in the next network's steps or any decision. It isn't part of a
saved state: changed in controller.json, it holds from the next step, and a retired network
takes up its training where it left it.

With `"window"` in the config, the replay window grows with the positions the kept line of
states has stored per environment, n, by the replay-window task's rule: n until `c` (256), then
`c (1 + beta ((n / c)^alpha - 1) / alpha)` (KataGo's shape, `alpha` 0.75, `beta` 0.075, which
reaches about 1125 at an 8x128 run's 30M positions). A step starts with the buffer resized once
the target is 25% past its capacity. With playout cap randomization n counts the positions
stored. Periods are in turnovers of the buffer, so they lengthen as it grows, and as the
networks do; `trunk_hours` caps a trunk at that many GPU hours by the cost model, so the
tests come about as often throughout.

Compute is counted in network evaluations (self-play's positions evaluated, the samples each
network trains on) and in GPU time: the first step times each network's forward pass at
self-play's batch and its training step at training's (`bench_nets.py`), and each training
step measures what the search's own bookkeeping adds per cycle (its epochs' self-play time
less the forward passes, about 20% at 8x128; each search's median, the nearest budget's for a
search not run yet). A step's time is those times its cycles and steps: branches are matched
on it, and the waiting network's steps follow it. Compiling and tests aren't counted. The losing branch counts in what the run spent (`spent`), not
in the line of states it kept (`kept`). Decisions, with their series, go to `decisions.jsonl`.

The learning rate (`lr`, 5e-4) is constant through the run, as KataGo held its until its final
net's last days. The config is JSON: the networks, smallest first, the search to start with
(`full`, or with playout cap randomization `full`, `fast`, `full_probability` and `cycles`,
train.py's `--selfplay-cycles`), train.py's other arguments, and the steps' frozen worktree
(with `./install.sh`'s CUDA environment) and the slick task's worktree to queue from. See
`controller.Config` for the rest.

```json
{
  "name": "grow1",
  "nets": ["resnet:4x64,pool_every=2,heads=pooled,pool_channels=16",
           "resnet:6x96,pool_every=2,heads=pooled,pool_channels=24"],
  "search": {"full": 127, "fast": 31, "full_probability": 0.25, "cycles": 8320},
  "buffer": 256,
  "window": {"c": 256},
  "lr": 5e-4,
  "worktree": "~/worktrees/turbozero/grow1-frozen",
  "queue_cwd": "~/worktrees/turbozero/<the task's worktree>"
}
```

```
uv run examples/othello/controller.py init runs/grow1 --config grow1.json
uv run examples/othello/controller.py start runs/grow1
uv run examples/othello/controller.py status runs/grow1
uv run examples/othello/controller.py stop runs/grow1             # after the steps queued now
uv run examples/othello/controller.py retry runs/grow1 p003-double
uv run examples/othello/controller.py advance runs/grow1           # what a step's end queues
```

`advance` also catches a run up after an update of the controller, e.g. a run from before the
ladder plays each period's ladder, in order.

A rating step runs the league's player in its own queue entry, playing only requested series,
from the league's frozen worktree. If something queued at its priority or above takes the GPU
first, the step goes back in the queue. Its players are retired once their series are done.

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
earlier step. A job runs without this machine.
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
    --a net:checkpoint=runs/othello/199.eqx,spec=resnet:8x128,device=gpu-batched,visits=400 \
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
`backend`, `batch` and the rest, see [Playing on CPU](#playing-on-cpu)), `gpu` (a game at a time,
its leaves batched on the GPU, `gpu_play.py`) or `gpu-batched` (core's MCTS, many games moving in
step, at fixed visits against engines at a level or other such networks, with a tree of 4 x visits
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
  and transformed back, in 16 products where the direct convolution takes 36.
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
float epilogues normalize each square's channels where OpenVINO runs separate ops. Against the float
network on 2000 search leaves (value's mean error, its 99th percentile, top move):

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
of the game's clock (`solve_share`), the moves before it sharing the clock less that. The expected
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
