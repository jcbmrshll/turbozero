import jax


def test_suite_runs_on_two_cpu_devices():
    # set up in conftest.py: tests always run on CPU, with two simulated devices for the pmap paths
    assert jax.default_backend() == "cpu"
    assert jax.local_device_count() == 2
