"""Shared setup for the turbozero test suite."""

import os

# tests always run on CPU, never the GPU
os.environ["JAX_PLATFORMS"] = "cpu"
# pytest-xdist workers already use one core each, so each runs XLA single-threaded
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
    ).strip()

import jax

# compiling dominates the suite's runtime, so compiled programs persist across runs (and
# worktrees); JAX_COMPILATION_CACHE_DIR overrides where
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.expanduser("~/.cache/turbozero/jax-tests"),
    )
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_max_size", 2**30)
