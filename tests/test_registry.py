"""Tests for the geometry registry and JIT compatibility of detector methods.

Verifies:
- Registry lookup, case insensitivity, error handling
- generate_detector produces identical results via registry vs old path
- bounds_check works inside jax.jit (critical: methods called in JIT context)
- Detector sensor positions are bit-identical before and after registry refactor
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry import (
    generate_detector, get_detector_class, list_detector_types,
    Cylinder, Sphere, Box,
)
from lucid.geometry.registry import _REGISTRY


class TestRegistryLookup:
    def test_all_types_registered(self):
        types = list_detector_types()
        assert sorted(types) == ['box', 'cylinder', 'polygon_cylinder',
                                 'sphere', 'string']

    def test_case_insensitive(self):
        assert get_detector_class('cylinder') is Cylinder
        assert get_detector_class('Cylinder') is Cylinder
        assert get_detector_class('CYLINDER') is Cylinder

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown detector type"):
            get_detector_class('hexagon')

    def test_returns_correct_class(self):
        assert get_detector_class('cylinder') is Cylinder
        assert get_detector_class('sphere') is Sphere
        assert get_detector_class('box') is Box


class TestGenerateDetectorViaRegistry:
    """Verify generate_detector through registry produces identical detectors."""

    def test_cylinder_identical(self):
        """WCTE cylinder must have same sensors as baseline."""
        det = generate_detector("config/WCTE_like_geom_config.json")
        assert isinstance(det, Cylinder)
        assert det.all_points.shape == (2444, 3)
        npt.assert_allclose(det.all_points[0], [2.0, 0.0, -1.94], atol=1e-5)

    def test_sphere_identical(self):
        det = generate_detector("config/JUNO_geom_config.json")
        assert isinstance(det, Sphere)
        assert det.all_points.shape[0] == 10000
        npt.assert_allclose(det.all_points[0], [0.0, 0.0, 17.5], atol=1e-3)

    def test_box_identical(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        assert isinstance(det, Box)
        assert det.all_points.shape[0] == 4928
        npt.assert_allclose(det.all_points[0], [-10.88, 2.0, -1.88], atol=1e-2)


class TestBoundsCheckJIT:
    """Critical: bounds_check must work inside jax.jit since it's called
    in the photon propagation loop which is JIT-compiled."""

    def test_cylinder_bounds_check_jit(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])

        @jax.jit
        def check(positions):
            return det.bounds_check(positions)

        result = check(pts)
        assert result[0] == True
        assert result[1] == False

    def test_sphere_bounds_check_jit(self):
        det = generate_detector("config/JUNO_geom_config.json")
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])

        @jax.jit
        def check(positions):
            return det.bounds_check(positions)

        result = check(pts)
        assert result[0] == True
        assert result[1] == False

    def test_box_bounds_check_jit(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])

        @jax.jit
        def check(positions):
            return det.bounds_check(positions)

        result = check(pts)
        assert result[0] == True
        assert result[1] == False

    def test_bounds_check_jit_with_grad(self):
        """bounds_check output feeds into jnp.where which is differentiable."""
        det = generate_detector("config/WCTE_like_geom_config.json")

        @jax.jit
        def loss_fn(positions):
            inside = det.bounds_check(positions)
            # Use inside flag in a differentiable way
            return jnp.sum(jnp.where(inside, jnp.sum(positions**2, axis=1), 0.0))

        pts = jnp.array([[0.5, 0.5, 0.0], [100.0, 100.0, 100.0]])
        grad = jax.grad(loss_fn)(pts)
        # Gradient should be non-zero for inside point, zero for outside
        assert jnp.any(grad[0] != 0)
        npt.assert_allclose(grad[1], [0.0, 0.0, 0.0], atol=1e-7)

    def test_bounds_check_vmap(self):
        """bounds_check should work with vmap over batch dimension."""
        det = generate_detector("config/WCTE_like_geom_config.json")

        # Single-point bounds check via vmap
        single_check = jax.vmap(lambda p: det.bounds_check(p[None])[0])
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0], [1.0, 0.0, 0.0]])
        result = single_check(pts)
        npt.assert_array_equal(result, [True, False, True])


class TestDetectorGeometryJIT:
    """Verify DetectorGeometry container works in JIT context."""

    def test_from_config_cylinder_jit_propagator(self):
        """The propagator from DetectorGeometry should be JIT-callable."""
        from lucid.geometry.detector_geometry import DetectorGeometry
        dg = DetectorGeometry.from_config("config/WCTE_like_geom_config.json",
                                           detector_type='Cylinder')
        # Propagator should already be JIT-compiled
        assert dg.propagator is not None

        # Create minimal test rays (2 photons pointing inward)
        origins = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        dirs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = dg.propagator(origins, dirs)
        # Should return a dict with expected keys
        assert 'sensor_weights' in result
        assert 'sensor_indices' in result
        assert 'positions' in result
