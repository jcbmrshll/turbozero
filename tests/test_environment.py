import jax


def test_suite_runs_on_cpu():
    assert jax.default_backend() == "cpu"
