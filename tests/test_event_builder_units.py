"""Unit boundary between the host pipeline (meters) and the kernel (cm).

``root_reader`` and ``event_generation`` carry photon origins in meters, while
``_simulation_with_data_impl`` divides ``photon_origins`` by 100. The bucketed
tracer is the only place that conversion happens, so pin it here: when it was
missing, every propagated photon started 100x too close to the origin and no
test noticed.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from lucid.sources.event_builder import _trace_event_bucketed


class _CapturingSimulator:
    """Stands in for the JIT kernel and records what it was handed."""

    def __init__(self, n_sensors):
        self.n_sensors = n_sensors
        self.seen = []

    def __call__(self, track_params, key, photonsim_data):
        self.seen.append(np.asarray(photonsim_data['photon_origins']))
        n = self.n_sensors
        empty_f = jnp.zeros(0, dtype=jnp.float32)
        empty_i = jnp.zeros(0, dtype=jnp.int32)
        return (jnp.zeros(n, dtype=jnp.float32), jnp.zeros(n, dtype=jnp.float32),
                jnp.zeros(n, dtype=jnp.float32), empty_f, empty_f, empty_f,
                empty_i, empty_i)


def test_bucketed_tracer_converts_meters_to_cm():
    n_photons, n_sensors = 5, 3
    origins_m = np.array([[12.32238, -8.39766, -3.59919]] * n_photons,
                         dtype=np.float32)
    sim = _CapturingSimulator(n_sensors)

    _trace_event_bucketed(
        sim,
        origins_m,
        np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_photons, 1)),
        np.zeros(n_photons, dtype=np.float32),
        np.full(n_photons, 400.0, dtype=np.float32),
        np.zeros(n_photons, dtype=np.int32),
        n_sensors, (8,), __import__("jax").random.PRNGKey(0),
    )

    assert sim.seen, "kernel was never called"
    got = sim.seen[0][:n_photons]
    np.testing.assert_allclose(got, origins_m * 100.0, rtol=1e-5)
