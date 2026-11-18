"""Unit boundary between the host pipeline and the kernel: metres on both sides.

``root_reader`` and ``interaction.place_raw_entry`` carry photon origins in metres and
``_simulation_with_data_impl`` takes them as given. The bucketed tracer used to
multiply by 100 for a kernel that divided by 100; when one side of such a pair
goes missing every photon starts 100x off and no other test notices, so pin the
pass-through here.
"""
import numpy as np

import jax.numpy as jnp

from lucid.sources.interaction import _trace_event_bucketed


class _CapturingSimulator:
    """Stands in for the JIT kernel and records what it was handed."""

    def __init__(self, n_sensors):
        self.n_sensors = n_sensors
        self.seen = []

    def __call__(self, key, photonsim_data):
        self.seen.append(np.asarray(photonsim_data['photon_origins']))
        n = self.n_sensors
        empty_f = jnp.zeros(0, dtype=jnp.float32)
        empty_i = jnp.zeros(0, dtype=jnp.int32)
        return (jnp.zeros(n, dtype=jnp.float32), jnp.zeros(n, dtype=jnp.float32),
                jnp.zeros(n, dtype=jnp.float32), empty_f, empty_f, empty_f,
                empty_i, empty_i)


def test_bucketed_tracer_hands_the_kernel_metres_unchanged():
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
    np.testing.assert_array_equal(got, origins_m)
