import os

import jax


def test_jax_runs_on_the_requested_platform():
    platform = os.environ["JAX_PLATFORMS"].split(",")[0]
    if platform in ("cuda", "rocm"):
        platform = "gpu"
    assert jax.default_backend() == platform
