# *turbozero* 🏎️ 🏎️ 🏎️ 🏎️

📣 If you're looking for the old PyTorch version of turbozero, it's been moved here: [turbozero_torch](https://github.com/jcbmrshll/turbozero_torch) 📣

#### *`turbozero`* is a vectorized implementation of [AlphaZero](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) written in JAX

It contains:
* Monte Carlo Tree Search with subtree persistence
* Batched Replay Memory
* A complete, customizable training/evaluation loop

#### *`turbozero`* is *_fast_* and *_parallelized_*:
 * every consequential part of the training loop is JIT-compiled
 * parititions across multiple GPUs by default when available 🚀 NEW! 🚀
 * self-play and evaluation episodes are batched/vmapped with hardware-acceleration in mind

#### *`turbozero`* is *_extendable_*:
 * see an [idea on twitter](https://twitter.com/ptrschmdtnlsn/status/1748800529608888362) for a simple tweak to MCTS?
      * [implement it](https://github.com/jcbmrshll/turbozero/blob/main/core/evaluators/mcts/weighted_mcts.py) then [test it](https://github.com/jcbmrshll/turbozero/blob/main/notebooks/weighted_mcts.ipynb) by extending core components
  
#### *`turbozero`* is *_flexible_*:
 * easy to integrate with you custom JAX environment or neural network architecture.
 * Use the provided training and evaluation utilities, or pick and choose the components that you need.

To get started, check out the [Hello World Notebook](https://github.com/jcbmrshll/turbozero/blob/main/notebooks/hello_world.ipynb)

## Installation
`turbozero` uses `poetry` for dependency management, you can install it with:
```
pip install poetry==1.7.1
```
Then, to install dependencies:
```
poetry install
```
If you're using a GPU/TPU/etc., after running the previous command you'll need to install the device-specific version of JAX.

For a GPU w/ CUDA 12:
```
poetry source add jax https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
to point poetry towards JAX cuda releases, then use
```
poetry add jax[cuda12_pip]==0.4.35
```
to install the CUDA 12 release for JAX. See https://jax.readthedocs.io/en/latest/installation.html for other devices/cuda versions.

I have tested this project with CUDA 11 and CUDA 12.

To launch an ipython kernel, run:
```
poetry run python -m ipykernel install --user --name turbozero
```

## Issues
If you use this project and encounter an issue, error, or undesired behavior, please submit a [GitHub Issue](https://github.com/jcbmrshll/turbozero/issues) and I will do my best to resolve it as soon as I can. You may also contact me directly via `hello@jacob.land`.

## Contributing 
Contributions, improvements, and fixes are more than welcome! For now I don't have a formal process for this, other than creating a [Pull Request](https://github.com/jcbmrshll/turbozero/pulls). For large changes, consider creating an [Issue](https://github.com/jcbmrshll/turbozero/issues) beforehand.

If you are interested in contributing but don't know what to work on, please reach out. I have plenty of things you could do.

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
