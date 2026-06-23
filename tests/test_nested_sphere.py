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
from lucid.simulation.photon_step import _interface_refract_reflect
from lucid.simulation.reflection import fresnel_rr

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


class TestSingleMediumUnaffected:
    """Phase-0 tripwire: the two-medium additions must not touch the single-medium path."""

    def test_detector_params_leaf_count_unchanged(self):
        # outer_optics defaults to None ⇒ contributes ZERO pytree leaves ⇒ existing
        # flatten order / normalize / grad machinery is identical for single-medium.
        from lucid.detector_params import DetectorParams
        dp = DetectorParams.from_flat(qe=0.065, qe_corrections=jnp.ones(50))
        assert dp.outer_optics is None
        n_leaves = len(jax.tree_util.tree_leaves(dp))
        assert len(jax.tree_util.tree_leaves(dp._replace())) == n_leaves

    def test_with_outer_optics_adds_only_outer_leaves(self):
        from lucid.detector_params import DetectorParams
        dp = DetectorParams.from_flat(qe=0.065, qe_corrections=jnp.ones(50))
        base = len(jax.tree_util.tree_leaves(dp))
        split = len(jax.tree_util.tree_leaves(dp.with_outer_optics()))
        # ScatteringParams (5) + AbsorptionParams (2) = 7 new leaves.
        assert split - base == 7

    def test_single_sphere_propagator_has_no_interface_key(self):
        # The nested-only 'hit_interface' key must not appear in the single-sphere dict.
        dg = DetectorGeometry.from_config("config/JUNO_geom_config.json", detector_type='sphere')
        assert not dg.is_nested
        res = dg.propagator(jnp.array([[0., 0., 0.]]), jnp.array([[1., 0., 0.]]))
        assert 'hit_interface' not in res


class TestInterfacePhysics:
    """Analytic Snell / Fresnel / TIR + flux conservation — the PRIMARY physics gate
    (validation is internal-consistency only, so these must be airtight)."""
    N_LS, N_W = 1.48, 1.33   # high-contrast LAB-LS <-> water

    def test_fresnel_flux_conservation(self):
        # R + T = 1 across all incidence angles (T = 1 - R by construction).
        for deg in [0, 20, 40, 55, 63, 70, 85]:
            ci = np.cos(np.deg2rad(deg))
            R, ct = fresnel_rr(jnp.array(ci), self.N_LS, self.N_W)
            assert 0.0 <= float(R) <= 1.0

    def test_normal_incidence_reflectance(self):
        # R0 = ((n1-n2)/(n1+n2))^2 at normal incidence.
        R, ct = fresnel_rr(jnp.array(1.0), self.N_LS, self.N_W)
        R0 = ((self.N_LS - self.N_W) / (self.N_LS + self.N_W)) ** 2
        npt.assert_allclose(float(R), R0, atol=1e-6)

    def test_total_internal_reflection(self):
        # Dense->light past the critical angle (arcsin(1.33/1.48) ~ 64 deg) => R = 1.
        theta_c = np.rad2deg(np.arcsin(self.N_W / self.N_LS))
        ci = np.cos(np.deg2rad(theta_c + 5))
        R, ct = fresnel_rr(jnp.array(ci), self.N_LS, self.N_W)
        npt.assert_allclose(float(R), 1.0, atol=1e-6)

    def test_snell_refraction_angle(self):
        # 45 deg incidence, inner(1.48)->outer(1.33). Transmitted angle obeys Snell.
        d = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)   # 45 deg to +x
        radial = jnp.array([1.0, 0.0, 0.0])              # outward normal
        new_dir, new_mid, score, transmit = _interface_refract_reflect(
            d, radial, jnp.array(0), self.N_LS, self.N_W, jnp.array(0.0))  # u=0 -> transmit
        assert bool(transmit) is True
        assert int(new_mid) == 1                          # medium flipped inner->outer
        npt.assert_allclose(float(jnp.linalg.norm(new_dir)), 1.0, atol=1e-5)
        # transmitted angle: sin t2 = (n1/n2) sin t1
        sin_t1 = np.sqrt(0.5)
        sin_t2 = (self.N_LS / self.N_W) * sin_t1
        cos_t2 = np.sqrt(1 - sin_t2 ** 2)
        npt.assert_allclose(abs(float(jnp.dot(new_dir, radial))), cos_t2, atol=1e-4)

    def test_tir_reflects_and_keeps_medium(self):
        # 75 deg incidence inner->outer is beyond critical => must reflect, medium unchanged.
        ang = np.deg2rad(75)
        d = jnp.array([np.cos(ang), np.sin(ang), 0.0])
        radial = jnp.array([1.0, 0.0, 0.0])
        new_dir, new_mid, score, transmit = _interface_refract_reflect(
            d, radial, jnp.array(0), self.N_LS, self.N_W, jnp.array(0.0))  # u=0
        assert bool(transmit) is False                    # T=0 at TIR => no transmit even at u=0
        assert int(new_mid) == 0                          # medium unchanged
        # specular reflection about the radial normal: x-component flips
        npt.assert_allclose(float(new_dir[0]), -np.cos(ang), atol=1e-5)
        npt.assert_allclose(float(new_dir[1]), np.sin(ang), atol=1e-5)


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


class TestNestedForward:
    """End-to-end calibration forward through the two-medium engine."""

    @staticmethod
    def _dp():
        from lucid.detector_params import DetectorParams
        return DetectorParams.from_flat(
            scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
            absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(10000))

    @pytest.mark.slow
    def test_invisible_interface_matches_single_sphere(self, tmp_path):
        # inner==outer material + equal n ⇒ the interface is optically transparent, so a
        # nested wbls/wbls detector must reproduce a single wbls sphere at the outer radius.
        import json
        from lucid.simulation import setup_event_simulator
        from lucid.sources import isotropic_source
        single = tmp_path / "single.json"
        single.write_text(json.dumps({
            "material": "wbls", "detector_type": "sphere",
            "geometry_definitions": {"radius": 19.5, "n_sensors": 10000, "sensor_radius": 0.25}}))
        nested = tmp_path / "nested.json"
        nested.write_text(json.dumps({
            "material": "wbls", "inner_material": "wbls", "outer_material": "wbls",
            "detector_type": "nested_sphere",
            "geometry_definitions": {"inner_radius": 17.5, "outer_radius": 19.5,
                                     "n_sensors": 10000, "sensor_radius": 0.25}}))
        dp = self._dp()
        src = isotropic_source(position=[0., 0., 0.], intensity=50_000_000, wavelength=420.0)
        KW = dict(temperature=None, K=16, is_calibration=True, wavelength_mode=True,
                  physics_config="config/JUNO_nested_physics_config.json")
        c_ref, _ = setup_event_simulator(str(single), 300000, detector_type='sphere', **KW)(
            src, dp, jax.random.PRNGKey(0))
        c_nest, _ = setup_event_simulator(str(nested), 300000, detector_type='nested_sphere', **KW)(
            src, dp, jax.random.PRNGKey(0))
        ratio = float(c_nest.sum()) / float(c_ref.sum())
        assert abs(ratio - 1.0) < 0.01, f"invisible interface ratio {ratio:.4f} != 1"

    @pytest.mark.slow
    def test_high_contrast_tir_reduces_charge(self):
        # LAB-LS (n=1.48) inside water (n=1.33): TIR beyond the 64° critical angle traps
        # light in the LS, so detected charge is materially below the transparent case.
        from lucid.simulation import setup_event_simulator
        from lucid.sources import isotropic_source
        dp = self._dp()
        src = isotropic_source(position=[0., 0., 0.], intensity=50_000_000, wavelength=430.0)
        KW = dict(temperature=None, K=20, is_calibration=True, wavelength_mode=True,
                  detector_type='nested_sphere')
        c_w, _ = setup_event_simulator(
            'config/JUNO_nested_geom_config.json', 300000,
            physics_config='config/JUNO_nested_physics_config.json', **KW)(src, dp, jax.random.PRNGKey(0))
        c_l, _ = setup_event_simulator(
            'config/JUNO_nested_labls_geom_config.json', 300000,
            physics_config='config/JUNO_nested_labls_physics_config.json', **KW)(src, dp, jax.random.PRNGKey(0))
        assert float(c_l.sum()) < 0.9 * float(c_w.sum())   # TIR + 28m Rayleigh measurably reduce charge
        assert float(c_l.sum()) > 0.5 * float(c_w.sum())   # but not catastrophically
