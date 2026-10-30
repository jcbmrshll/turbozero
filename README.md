# *turbozero* 🏎️ 🏎️ 🏎️ 🏎️

📣 If you're looking for the old PyTorch version of turbozero, it's been moved here: [turbozero_torch](https://github.com/jcbmrshll/turbozero_torch) 📣

#### *`turbozero`* is a vectorized implementation of [AlphaZero](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) written in JAX

It contains:
* Monte Carlo Tree Search with subtree persistence
* Batched Replay Memory
* A complete, customizable training/evaluation loop

#### *`turbozero`* is *_fast_* and *_parallelized_*:
 * every consequential part of the training loop is JIT-compiled
 * self-play and evaluation episodes are batched/vmapped with hardware-acceleration in mind

#### *`turbozero`* is *_extendable_*:
 * see an [idea on twitter](https://twitter.com/ptrschmdtnlsn/status/1748800529608888362) for a simple tweak to MCTS?
      * [implement it](https://github.com/jcbmrshll/turbozero/blob/main/core/evaluators/mcts/weighted_mcts.py) then [test it](https://github.com/jcbmrshll/turbozero/blob/main/examples/connect_four/train.py) by extending core components (`--search weighted`)
  
#### *`turbozero`* is *_flexible_*:
 * easy to integrate with you custom JAX environment or neural network architecture.
      * networks are [Equinox](https://github.com/patrick-kidger/equinox) modules, trained with [Optax](https://github.com/google-deepmind/optax); see [`apply_nn`](https://github.com/jcbmrshll/turbozero/blob/main/core/networks/utils.py) for the calling convention
 * Use the provided training and evaluation utilities, or pick and choose the components that you need.

To get started, check out the [Othello example](https://github.com/jcbmrshll/turbozero/tree/main/examples/othello), whose training script walks through each component

## Installation
`turbozero` uses [`uv`](https://docs.astral.sh/uv/) for dependency management. With `uv` installed, run:
```
./install.sh
```
This creates a `.venv` with all dependencies. If an NVIDIA GPU is present it installs the CUDA 13 build of JAX (requires NVIDIA driver >= 580), otherwise the CPU build. Re-run `./install.sh` rather than a bare `uv sync`, which would remove the GPU libraries. For other accelerators, see https://docs.jax.dev/en/latest/installation.html.

## Examples
Example training scripts live in `examples/`:
```
uv run examples/othello/train.py   # AlphaZero, scored every epoch against Edax's scores of fixed positions
uv run examples/hex/train.py       # AlphaZero on Hex at 7 x 7, 9 x 9 or 11 x 11, measured against MoHex
uv run examples/connect_four/train.py   # AlphaZero, measured against a perfect solver; --search weighted for weighted MCTS
uv run examples/tic_tac_toe.py     # sanity check: trains in a minute or two on CPU, tested against a random player
```
Pass `--help` to see their options.

## Monitoring runs
Training runs can log their metrics, config and test games to a [seaglass](https://github.com/jcbmrshll/seaglass) server. The training loop hands off raw test games (the board after each move, each player's value estimates, and what their searches explored) and the run's page plays them in the browser, so training never waits on drawing or the network:
```
uv run examples/othello/train.py --monitor   # logs to $SEAGLASS_URL, else slick's server (http://slick/seaglass)
```
Inside a slick task `$SEAGLASS_URL` and `$SEAGLASS_TAGS` are set, so the run lists under the task that started it. `--monitor URL` logs to another server, e.g. a standalone one: `uv run seaglass serve --dir runs` serves http://localhost:8008, storing runs in `./runs`. `--notes "..."` says what a run is testing and why, on its page (or set `$SEAGLASS_NOTES`, which any run reads).

In your own scripts, pass `monitor=Monitor(project="...")` (from `core.monitoring`) to the `Trainer`, and `episode_fn=pgx_two_player_episode()` (from `core.monitoring.episodes`) to testers of Othello, Hex, Connect Four or tic-tac-toe to see their games. `core.monitoring.Monitor` is a `seaglass.Monitor` that ships turbozero's views (`core/monitoring/views/`, a seaglass bundle) with each run, so any seaglass server draws its games and search trees. Another game needs a board in `core.monitoring.episodes.BOARDS`, and a view in `core/monitoring/views/renderers.js` with its episode type in `layout.json`. Each run declares what the metrics turbozero logs are, from `core/monitoring/declared.json`: a description the run's page shows by each, which way is better, and the headline metrics its tiles show. A new metric needs a description there. Logging never interrupts training: if the server is down, the run keeps going and says so once.

## Issues
If you use this project and encounter an issue, error, or undesired behavior, please submit a [GitHub Issue](https://github.com/jcbmrshll/turbozero/issues) and I will do my best to resolve it as soon as I can. You may also contact me directly via `hello@jacob.land`.

## Contributing 
Contributions, improvements, and fixes are more than welcome! For now I don't have a formal process for this, other than creating a [Pull Request](https://github.com/jcbmrshll/turbozero/pulls). For large changes, consider creating an [Issue](https://github.com/jcbmrshll/turbozero/issues) beforehand.

If you are interested in contributing but don't know what to work on, please reach out. I have plenty of things you could do.

CI runs `uv run ruff check .`, `uv run ruff format --check .` and `uv run pyright` alongside the tests. To run ruff (lint + format) automatically on each commit, install the pre-commit hooks once:
```
uv run pre-commit install
```

Bulk reformat commits are listed in `.git-blame-ignore-revs`; GitHub's blame view skips them automatically, and `git config blame.ignoreRevsFile .git-blame-ignore-revs` does the same for local `git blame`.

## References
Papers/Repos I found helpful.

Repositories:
* [google-deepmind/mctx](https://github.com/google-deepmind/mctx): Monte Carlo tree search in JAX
* [sotetsuk/pgx](https://github.com/sotetsuk/pgx): Vectorized RL game environments in JAX
* [instadeepai/flashbax](https://github.com/instadeepai/flashbax): Accelerated Replay Buffers in JAX
* [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel): RL algorithms

Papers:
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
* [Revisiting Fundamentals of Experience Replay](https://arxiv.org/abs/2007.06700)


## Cite This Work
If you found this work useful, please cite it with:
```
@software{turbozero,
  author = {Marshall, Jacob},
  title = {{turbozero: fast + parallel AlphaZero}},
  url = {https://github.com/jcbmrshll/turbozero}
}
```
