# Connect Four

AlphaZero on Connect Four: training, and evaluating the result against the truth. The game is
solved, so [Pascal Pons's solver](https://github.com/PascalPons/connect4) scores every move of any
position exactly, and a network's mistakes can be counted rather than estimated.

| File | |
|---|---|
| `train.py` | Trains a network by self-play, scoring it against the solver's scores of fixed positions as it goes, and in games against the solver |
| `vs_solver.py` | Plays a checkpoint against the solver at several levels, from fixed openings (and train.py's network as it trains) |
| `solver.py` | The solver, driven over its command line, and the player it makes at each level |
| `setup_solver.sh` | Builds the solver, and labels the set of positions |
| `solver_set.py`, `label_positions.py`, `score_solver_set.py` | The solver-labelled set: fixed positions with every legal move scored exactly, to build, and to score checkpoints on (train.py scores the network on it every epoch) |
| `game.py` | What the scripts share: the environment, the board's symmetry, network, evaluators |
| `openings.py`, `openings.txt`, `build_openings.py` | The opening list, and how it was built |

pgx has no pretrained Connect Four model, so unlike the Othello example there's no `eval_pgx.py`;
the solver is a far better yardstick.

## Training

```
examples/connect_four/setup_solver.sh     # once
uv run examples/connect_four/train.py --monitor --ckpt-dir runs/connect_four --notes "the baseline" \
    --vs-engine-last 3   # games against the solver after each of the last 3 epochs, too
```

`--name` names the run on the server, and `--notes` says what it's testing and why, at the
top of its page. On a GPU, install with `./install.sh`, and use `uv run --extra cuda` in place of
`uv run`.

Self-play runs in 1024 games at once, searching 64 MCTS iterations a move, for 32 moves in each an
epoch, and the network takes 32 training steps of 4096 samples: 4 passes over each new position,
as in the Othello example. Replay memory stores each position once, and training maps each
position it samples through the board's mirror image or leaves it, at random. Games start from the
empty board, varied by sampling each move from the search's visits; `--standard-starts` sets
the fraction that do, the rest starting from the opening list. See `--help` for the network size,
search budget and number of epochs.

A run is tracked by the solver-labelled set: every epoch, `train.py` scores the raw network on 5001
fixed positions (see [below](#against-the-solvers-scores-of-fixed-positions)). Games against the
solver measure strength, after every 10 epochs by default (`--vs-engine-every`, `--vs-engine-last`,
see [During training](#during-training)).

`--eval-every N` plays 128 test games against a random player every N epochs (10 by default), logged
as `random_avg_outcome`, `random_win_rate` and `random_loss_rate`, our agent searching 64 iterations
a move.

With `--monitor` the run's page shows the metrics and games against the solver and the random player, to
step through, with what each move's search explored. It also plots `episode_draw`, the fraction of
self-play games drawn, which `game.py`'s `episode_metrics` reports through the Trainer's
`episode_metrics_fn`.

`--search weighted` searches with weighted MCTS
([core/evaluators/mcts/weighted_mcts.py](../../core/evaluators/mcts/weighted_mcts.py)) instead of
AlphaZero's PUCT MCTS, in self-play and in our agent's test games: it backs up a softmax-weighted
sum of child Q-values, with its temperature set by `--q-temperature`, instead of a plain average.

```
uv run examples/connect_four/train.py --search weighted --monitor
```

The rest of the flags are the Othello example's, and work the same: see [its
README](../othello/README.md) for `--sims-schedule`, `--count-reused-visits`, `--no-persist-tree`,
`--traversal-steps`, `--buffer`, `--inference-dtype`, `--remat`, `--grad-accum`,
`--value-target-q`, `--tree-positions`, `--policy-target`, and for `--save-state-at`,
`--save-state-every`, `--resume` and `--init-from`, which continue a run and fork others from it:

```
uv run examples/connect_four/train.py --epochs 100 --save-state-every 10 --ckpt-dir runs/long
uv run examples/connect_four/train.py --epochs 100 --save-state-every 10 --ckpt-dir runs/long --resume
uv run examples/connect_four/train.py --epochs 30 --init-from runs/long --ckpt-dir runs/long-fork
```

### The network

`--net` names the network as KIND:DEPTHxWIDTH, then any of its configuration's fields as
,FIELD=VALUE (see `core/networks/spec.py`). The default, `resnet:4x64` (`DEFAULT_NET` in
`game.py`), is the residual network from the AlphaZero paper with 4 blocks of 64 channels.
`pool_every=N` adds KataGo's global pooling. The Othello example's `heads=pooled` doesn't apply:
those heads output a logit per square, and here an action is a column.

## Evaluating a checkpoint

Checkpoints from `train.py` are named after their epoch (`99.eqx` is the last of a 100-epoch run).
Pass the run's `--net` if it isn't the default.

```
examples/connect_four/setup_solver.sh     # once
uv run examples/connect_four/vs_solver.py runs/connect_four/99.eqx --levels 0 2 4 6 8 42 --sims 256
uv run examples/connect_four/score_solver_set.py runs/connect_four/99.eqx
```

At level L the solver looks L moves ahead and plays what it sees there, exactly as a minimax search
that deep with no evaluation would: it takes the soonest win it sees, puts off the latest loss if it
sees nothing else, and otherwise picks at random. Level 0 is a random player, level 1 takes a win,
level 2 also stops the opponent's, and level 42 is perfect. Its scores are the true ones either way:
a level only hides those of the games that end too far ahead.

Games start from the **openings** of `openings.txt`: every 6-move opening whose position scores
within 2 (drawn, or won only with the board nearly full), 3407 of them, mirror images and move
orders reaching the same position counting once. They give our agent, at temperature 0 and without
root noise, varied games. Each is played twice, with the colors swapped.

Each opening's two games have a known result under perfect play, one win and one loss or two draws,
so **a perfect player scores exactly 0.5 against level 42**: that's the ceiling, and whatever is
missing from it is games our agent let slip.

### During training

`train.py` plays the solver after every `--vs-engine-every` epochs (10 by default), and after each
of the last `--vs-engine-last` epochs. By default it plays 512 games (256 openings) at each of levels
0, 4, 8 and 42, searching 256 MCTS iterations a move, the same openings every time and in every run.
`--vs-engine-levels`, `--vs-engine-sims` and `--vs-engine-openings` set the games. If the solver
isn't installed, it warns and goes on without them.

Each level's result is logged to the run at the epoch of the network that played (counting from 0,
so `--vs-engine-every 10` logs at epochs 9, 19, ...), as `solver_level42_sims256`, the score, with its
standard error `solver_level42_sims256_se` and the fractions of games won, drawn and lost (`_win`,
`_draw`, `_loss`), and a game at that level, `solver_level42_sims256_game`, to step through: our
agent's search of each position, and the solver's verdict on it, how every legal move's game ends
under perfect play with the line that gets there. Results also go to
`runs/connect_four/vs_solver.jsonl`, a line per level and epoch. `--vs-engine-last K` also logs the
mean of the last K networks' scores at the last epoch, as `solver_level42_sims256_last3` (for K = 3),
with its standard error (`_last3_se`).

The games are played between epochs, in the training process, and training waits for them: their
time is logged as `solver_seconds`. The solver's processes (8 at most, 120 MB each) run on the cores
the run may use, so `taskset -c 0-7` keeps them on those 8.

To evaluate a checkpoint after training, `vs_solver.py` logs to its run with `--monitor --run ID` (the
id in the run's seaglass link), at the checkpoint's epoch.

### Against the solver's scores of fixed positions

The solver-labelled set measures the network without search or games: 5001 fixed positions with
every legal move scored exactly. `label_positions.py` samples them, with no training run needed,
from games the solver plays with some of its moves at random: a third with 10% of them (positions a
strong player meets), a third with 30%, and a third wholly random (positions a young network
meets). Every position has a choice of move, and mirror images and move orders that reach the same
position count once. It takes about 15 seconds on 8 cores, and `setup_solver.sh` runs it.

A move's score from the solver is for the player making it: 0 for a draw, positive for a win, the
sooner the bigger (a win with the last disc of the board scores 1, with the player's fourth disc 18),
negative for a loss. `train.py` scores the network on the set every epoch in two ways, each logged
for all positions, for each game phase (`opening`, 29-42 empty squares; `midgame`, 15-28; `endgame`,
1-14) and for `critical` positions, those whose outcome depends on the move (about half):

- `engine_*`, by the solver's scores, the same metrics as the Othello example's against Edax, in
  which winning later than a move could gives some up
- `outcome_*`, by how each move's game ends alone (+1, 0, -1), which is what AlphaZero's value
  trains towards, with no reward for winning soon. `outcome_top1_critical`, how often the network's
  first choice keeps the game's result where the move matters, is the one to watch

Each has:

- `top1`: how often the network's most likely move is one of the best
- `regret`: how much score its most likely move gives up against the best
- `exp_regret`: the same, averaged over its policy: smoother, but it also measures how sharp the
  policy is, which the search and its policy target set, so compare it only between runs with the
  same search and target
- `value_mse`: its value's squared error against the game's true result (±1, 0 for a draw): this
  game's special value, an error against the truth rather than an engine's guess or a noisy game's
  outcome
- `value_sign`: how often its value calls the winner correctly, over positions that aren't draws

`--engine-set PATH` scores on another set, and `--engine-eval-every N` sets the cadence (0 for never).
Without a set at the default path, `~/.cache/turbozero/connect_four/solver-set.npz`, training warns
and goes on unscored; a missing `--engine-set` is an error. To score saved checkpoints:

```
uv run examples/connect_four/score_solver_set.py runs/connect_four/49.eqx runs/connect_four/99.eqx
```

## What to expect

The default 100-epoch run hasn't been done yet, so there are no measured numbers for its strength or
its time. A perfect player would score 1 on `outcome_top1` and 0.5 against level 42.

A 20-epoch smoke run of the command above (`--epochs 20`, on one RTX 5080) took under 3 minutes, of
which the games against the solver took 105 seconds. That shows the command works, not how strong the
network gets, and its learning rate decayed over only those 20 epochs:

| | After 1 epoch | After 10 | After 20 |
|---|---|---|---|
| `loss` | 2.42 | 2.09 | 1.90 |
| `outcome_top1_critical` | 0.45 | 0.72 | 0.81 |
| `engine_value_mse` | 0.83 | 0.63 | 0.62 |
| Score against level 4, searching 256 | | 0.93 | 0.96 |
| Score against level 8, searching 256 | | 0.80 | 0.88 |
| Score against level 42 (0.5 at best), searching 256 | | 0.03 | 0.08 |

Searching 64 iterations a move, it won all 128 games against a random player after a single epoch, so
that test says little about learning.

## Credits

The solver is Pascal Pons's ([connect4.gamesolver.org](http://connect4.gamesolver.org)), under the
AGPL; the setup script builds it and downloads its opening book from its repository, they aren't part
of turbozero.
