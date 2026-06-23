"""Phase 1 tests for the nested two-sphere (JUNO-like) geometry.

Covers the geometry class, the forward ray/two-sphere intersection kernel, and the
nested propagator (outer-sphere sensor lookup + inner-interface masking).
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry import generate_detector, NestedSphere, Sphere, list_detector_types
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.propagation.nested_sphere import (
    intersect_two_spheres_forward, batch_intersect_two_spheres,
)

CONFIG = "config/JUNO_nested_geom_config.json"
R_IN, R_OUT = 17.5, 19.5


class TestNestedSphereGeometry:
    def test_registered(self):
        assert 'nested_sphere' in list_detector_types()

    def test_is_sphere_subclass(self):
        # NestedSphere reuses all Sphere placement/grid machinery at the outer radius.
        assert issubclass(NestedSphere, Sphere)

    def test_generate(self):
        det = generate_detector(CONFIG)
        assert isinstance(det, NestedSphere)
        assert det.r_inner == R_IN
        assert det.r_outer == R_OUT
        assert det.all_points.shape[0] == 10000
        # sensors sit on the OUTER sphere
        npt.assert_allclose(np.linalg.norm(det.all_points, axis=1), R_OUT, atol=1e-3)

    def test_inner_lt_outer_enforced(self):
        with pytest.raises(ValueError, match="inner_radius"):
            NestedSphere(20.0, 19.5, 100, 0.25)

    def test_region_of(self):
        det = generate_detector(CONFIG)
        pts = jnp.array([[0., 0., 0.], [18.5, 0., 0.], [17.4, 0., 0.]])
        # 0 = inner medium (r < r_inner), 1 = outer medium
        npt.assert_array_equal(np.asarray(det.region_of(pts)), [0, 1, 0])


class TestTwoSphereIntersection:
    def test_center_outward_hits_inner(self):
        t, hi, p, n = intersect_two_spheres_forward(
            jnp.array([0., 0., 0.]), jnp.array([1., 0., 0.]), R_IN, R_OUT)
        assert bool(hi) is True
        npt.assert_allclose(float(t), R_IN, atol=1e-4)
        npt.assert_allclose(np.asarray(p), [R_IN, 0, 0], atol=1e-4)
        npt.assert_allclose(np.asarray(n), [1, 0, 0], atol=1e-4)

    def test_shell_outward_hits_outer(self):
        t, hi, p, n = intersect_two_spheres_forward(
            jnp.array([18.5, 0., 0.]), jnp.array([1., 0., 0.]), R_IN, R_OUT)
        assert bool(hi) is False
        npt.assert_allclose(float(t), R_OUT - 18.5, atol=1e-4)

    def test_shell_inward_hits_inner(self):
        t, hi, p, n = intersect_two_spheres_forward(
            jnp.array([18.5, 0., 0.]), jnp.array([-1., 0., 0.]), R_IN, R_OUT)
        assert bool(hi) is True
        npt.assert_allclose(float(t), 18.5 - R_IN, atol=1e-4)

    def test_inner_edge_outward_exits_inner(self):
        t, hi, p, n = intersect_two_spheres_forward(
            jnp.array([17.0, 0., 0.]), jnp.array([1., 0., 0.]), R_IN, R_OUT)
        assert bool(hi) is True
        npt.assert_allclose(float(t), R_IN - 17.0, atol=1e-4)

    def test_jit_and_vmap(self):
        f = jax.jit(lambda o, d: batch_intersect_two_spheres(o, d, R_IN, R_OUT))
        o = jnp.array([[0., 0., 0.], [18.5, 0., 0.]])
        d = jnp.array([[1., 0., 0.], [-1., 0., 0.]])
        t, hi, p, n = f(o, d)
        npt.assert_array_equal(np.asarray(hi), [True, True])


class TestNestedPropagator:
    def test_build_and_dict_keys(self):
        dg = DetectorGeometry.from_config(CONFIG, detector_type='nested_sphere')
        assert dg.is_nested
        assert dg.r_inner == R_IN and dg.r_outer == R_OUT
        assert dg.medium_outer is not None
        res = dg.propagator(jnp.array([[0., 0., 0.]]), jnp.array([[1., 0., 0.]]))
        for k in ('sensor_weights', 'sensor_indices', 'times', 'positions',
                  'normals', 'inside_sensor', 'hit_interface'):
            assert k in res

    def test_interface_masks_sensors(self):
        dg = DetectorGeometry.from_config(CONFIG, detector_type='nested_sphere')
        o = jnp.array([[0., 0., 0.], [18.5, 0., 0.], [18.5, 0., 0.]])
        d = jnp.array([[1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
        res = dg.propagator(o, d)
        npt.assert_array_equal(np.asarray(res['hit_interface']), [True, False, True])
        w = np.asarray(res['sensor_weights']).sum(axis=0)
        # interface-hit rays carry NO sensor charge; the outward shell ray does
        assert w[0] == 0.0
        assert w[2] == 0.0
        assert w[1] > 0.0
        # interface rays' geometry hit is on the inner sphere
        npt.assert_allclose(np.linalg.norm(np.asarray(res['positions'])[0]), R_IN, atol=1e-3)
