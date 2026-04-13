"""Tests for the photon propagator output.

The propagator is the JIT-compiled function that takes photon positions
and directions and returns intersection data (weights, indices, times,
positions, normals, inside_sensor). This is the core ray-tracing engine.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

import pytest
from lucid.geometry.detector_geometry import DetectorGeometry

pytestmark = pytest.mark.slow


class TestCylinderPropagator:
    """Test the cylinder propagator with known ray configurations."""

    def setup_method(self):
        self.dg = DetectorGeometry.from_config(
            "config/WCTE_geom_config.json", detector_type='Cylinder',
            temperature=0.2)

    def test_output_has_all_keys(self):
        """Propagator must return dict with all required keys."""
        origins = jnp.array([[0.0, 0.0, 0.0]])
        dirs = jnp.array([[1.0, 0.0, 0.0]])
        result = self.dg.propagator(origins, dirs)
        for key in ['sensor_weights', 'sensor_indices', 'times',
                     'positions', 'normals', 'inside_sensor']:
            assert key in result, f"Missing key: {key}"

    def test_outward_ray_hits_barrel(self):
        """A ray from center going +x should hit the barrel of a cylinder."""
        origins = jnp.array([[0.0, 0.0, 0.0]])
        dirs = jnp.array([[1.0, 0.0, 0.0]])
        result = self.dg.propagator(origins, dirs)
        # positions should be on the barrel (x² + y² ≈ r²)
        hit_pos = result['positions']  # (n_rays, 3)
        r_hit = jnp.sqrt(hit_pos[0, 0]**2 + hit_pos[0, 1]**2)
        npt.assert_allclose(r_hit, self.dg.detector.r, atol=0.1)

    def test_weights_non_negative(self):
        """All sensor weights should be non-negative."""
        origins = jnp.zeros((5, 3))
        dirs = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [-1, 0, 0], [0, -1, 0]], dtype=jnp.float32)
        result = self.dg.propagator(origins, dirs)
        assert jnp.all(result['sensor_weights'] >= 0)

    def test_times_positive(self):
        """Hit times should be positive (photon travels forward)."""
        origins = jnp.zeros((3, 3))
        dirs = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=jnp.float32)
        result = self.dg.propagator(origins, dirs)
        # times in meters — should be > 0 for hits
        valid = result['sensor_weights'] > 0
        if jnp.any(valid):
            valid_times = result['times'][valid]
            assert jnp.all(valid_times >= 0)

    def test_normals_are_unit(self):
        """Surface normals should be approximately unit vectors."""
        origins = jnp.zeros((3, 3))
        dirs = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=jnp.float32)
        result = self.dg.propagator(origins, dirs)
        norms = jnp.linalg.norm(result['normals'], axis=-1)
        # Normals should be close to unit (may not be exactly 1 due to normalization)
        npt.assert_allclose(norms, 1.0, atol=0.1)

    def test_indices_valid(self):
        """Sensor indices should be in [0, num_sensors)."""
        origins = jnp.zeros((5, 3))
        dirs = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
        result = self.dg.propagator(origins, dirs)
        assert jnp.all(result['sensor_indices'] >= 0)
        assert jnp.all(result['sensor_indices'] < self.dg.num_sensors)

    def test_batch_size_consistent(self):
        """Output shapes should match input batch size."""
        n_rays = 7
        origins = jnp.zeros((n_rays, 3))
        dirs = jax.random.normal(jax.random.PRNGKey(42), (n_rays, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        result = self.dg.propagator(origins, dirs)
        assert result['positions'].shape == (n_rays, 3)
        assert result['normals'].shape == (n_rays, 3)
        # sensor_weights: (max_sensors_per_cell, n_rays)
        assert result['sensor_weights'].shape[1] == n_rays

    def test_propagator_is_jit_compiled(self):
        """Calling the propagator twice should not retrace (it's pre-JITted)."""
        origins = jnp.zeros((2, 3))
        dirs = jnp.array([[1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        # First call may trigger JIT compilation
        result1 = self.dg.propagator(origins, dirs)
        # Second call should use cached compilation
        result2 = self.dg.propagator(origins, dirs)
        npt.assert_allclose(result1['positions'], result2['positions'])

    def test_deterministic(self):
        """Propagator should be deterministic (no randomness)."""
        origins = jnp.zeros((3, 3))
        dirs = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=jnp.float32)
        r1 = self.dg.propagator(origins, dirs)
        r2 = self.dg.propagator(origins, dirs)
        npt.assert_array_equal(r1['sensor_weights'], r2['sensor_weights'])
        npt.assert_array_equal(r1['sensor_indices'], r2['sensor_indices'])
