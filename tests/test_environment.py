import jax


def test_suite_runs_on_cpu():
    # set up in conftest.py: tests always run on CPU, never the GPU
    assert jax.default_backend() == "cpu"
