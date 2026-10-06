"""Tests for the differentiable JUNO laser time observation grid."""

import jax
import jax.numpy as jnp
import numpy as np

from lucid.juno_laser.histogram import weighted_time_histogram
from lucid.juno_laser.time_grid import TimeGrid


TIME_GRID = TimeGrid(0.0, 4.0, 2.0)


def _histogram(times, weights=None):
    """Return one-sensor masses on the [0, 4) ns two-bin grid."""
    times = jnp.asarray(times, dtype=jnp.float32)
    if weights is None:
        weights = jnp.ones(times.shape, dtype=jnp.float32)
    return weighted_time_histogram(
        weights,
        jnp.zeros(times.shape, dtype=jnp.int32),
        times,
        n_sensors=1,
        time_grid=TIME_GRID,
    )[0]


def test_hits_at_bin_centers_are_not_shifted():
    """The two mass entries represent centers at 1 ns and 3 ns."""
    np.testing.assert_allclose(_histogram([1.0]), [1.0, 0.0])
    np.testing.assert_allclose(_histogram([3.0]), [0.0, 1.0])


def test_window_edges_preserve_all_in_window_mass():
    """Finite-window clipping must fold interpolation mass onto edge bins."""
    upper = np.nextafter(np.float32(4.0), np.float32(0.0))
    lower_hist = _histogram([0.0])
    upper_hist = _histogram([upper])
    np.testing.assert_allclose(lower_hist, [1.0, 0.0])
    np.testing.assert_allclose(upper_hist, [0.0, 1.0])
    np.testing.assert_allclose(lower_hist.sum(), 1.0)
    np.testing.assert_allclose(upper_hist.sum(), 1.0)


def test_every_in_window_hit_preserves_its_weight():
    """Soft assignment must conserve the accepted weighted PE mass."""
    times = jnp.asarray([0.0, 0.3, 1.0, 2.0, 3.7], dtype=jnp.float32)
    weights = jnp.asarray([0.2, 0.4, 0.6, 0.8, 1.0], dtype=jnp.float32)
    histogram = _histogram(times, weights)
    np.testing.assert_allclose(histogram.sum(), weights.sum(), rtol=1e-6)


def test_linear_assignment_preserves_the_interior_time_moment():
    """Interpolation between centers must reproduce an interior hit time."""
    centers = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    histogram = _histogram([1.4])
    reconstructed_time = jnp.sum(histogram * centers) / histogram.sum()
    np.testing.assert_allclose(reconstructed_time, 1.4, atol=1e-6)


def test_interior_time_moment_has_unit_time_gradient():
    """The soft observation grid must retain an interior timing gradient."""
    centers = jnp.asarray([1.0, 3.0], dtype=jnp.float32)

    def reconstructed_time(time):
        histogram = _histogram(jnp.asarray([time]))
        return jnp.sum(histogram * centers) / histogram.sum()

    gradient = jax.grad(reconstructed_time)(jnp.asarray(1.4))
    np.testing.assert_allclose(gradient, 1.0, atol=1e-6)
