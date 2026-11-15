"""Shared setup for the turbozero test suite."""

import os

# tests run on the CPU unless JAX_PLATFORMS asks for another platform, e.g.
# JAX_PLATFORMS=cuda; without preallocation the pytest-xdist workers share a GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# pytest-xdist gives each worker a core, so each runs XLA single-threaded
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
    ).strip()

import jax

# compiling dominates the suite's runtime, so compiled programs persist across
# runs and worktrees; JAX_COMPILATION_CACHE_DIR overrides where
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.expanduser("~/.cache/turbozero/jax-tests"),
    )
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_max_size", 2**30)
