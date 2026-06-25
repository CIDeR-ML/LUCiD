"""PR2 generic two-medium structure — invariant tests.

Promotes the build-time gates to the suite: the unified factory step is byte-identical to
the legacy single-medium step; the surface list reproduces the concentric-sphere
intersection; the split-prefix invariant byte-identity rests on; surfaces gradients are
NaN-free; and the outer_media param layout (mask/bounds/save) round-trips.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.simulation import photon_step as P
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.reflection import ScalarReflection
from lucid.propagation import surfaces as SF
from lucid.propagation.nested_sphere import intersect_two_spheres_forward

REFL = ScalarReflection(wall_rate=jnp.float32(0.2), sensor_rate=jnp.float32(0.2))


def _rand_inputs(n, seed):
    r = np.random.RandomState(seed)

    def unit(a):
        return jnp.asarray(a / np.linalg.norm(a, axis=-1, keepdims=True), jnp.float32)

    return dict(
        pos=jnp.asarray(r.uniform(-15, 15, (n, 3)), jnp.float32),
        direction=unit(r.normal(size=(n, 3))),
        normal=unit(r.normal(size=(n, 3))),
        time=jnp.asarray(r.uniform(0, 50, n), jnp.float32),
        sdist=jnp.asarray(r.uniform(0.1, 30, n), jnp.float32),
        scat=jnp.asarray(r.uniform(20, 300, n), jnp.float32),
        mie=jnp.asarray(r.uniform(1e3, 1e4, n), jnp.float32),
        g=jnp.full(n, 0.95, jnp.float32),
        absl=jnp.asarray(r.uniform(20, 300, n), jnp.float32),
        hit_sensor=jnp.asarray(r.rand(n) < 0.4),
        lam=jnp.full(n, 420.0, jnp.float32),
        keys=jax.random.split(jax.random.PRNGKey(seed), n),
        sol=jnp.full(n, 0.2254, jnp.float32),
    )


def _vrun_single(fn, a):
    f = lambda *x: fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], REFL,
                      x[8], x[9], x[10], x[11], x[12])
    return jax.vmap(f, in_axes=(0,) * 13)(
        a['pos'], a['direction'], a['time'], a['sdist'], a['normal'], a['scat'],
        a['mie'], a['g'], a['absl'], a['hit_sensor'], a['lam'], a['keys'], a['sol'])


class TestFactorySingleMediumByteIdentity:
    """make_photon_step(..., has_interface=False) == the legacy single-medium step, bit-for-bit."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    def test_sample(self, seed):
        a = _rand_inputs(256, seed)
        leg = _vrun_single(P.photon_iteration_sample, a)
        fac = _vrun_single(make_photon_step('sample', False), a)
        assert all(bool(jnp.array_equal(x, y)) for x, y in zip(leg, fac))

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    def test_update_factors(self, seed):
        a = _rand_inputs(256, seed)
        leg = _vrun_single(P.make_photon_iteration_update_factors_safe(), a)
        fac = _vrun_single(make_photon_step('update_factors', False), a)
        assert all(bool(jnp.array_equal(x, y)) for x, y in zip(leg, fac))


class TestSplitPrefixInvariant:
    """Byte-identity of the single-medium path rests on JAX split(k,N)[:M]==split(k,M)."""

    def test_prefix(self):
        for s in range(50):
            k = jax.random.PRNGKey(s)
            assert bool(jnp.array_equal(jax.random.split(k, 6), jax.random.split(k, 7)[:6]))
            assert bool(jnp.array_equal(jax.random.split(k, 8), jax.random.split(k, 9)[:8]))


class TestSurfacesEquivalence:
    """surfaces.nearest_interface reproduces the legacy concentric two-sphere intersection."""

    def test_matches_legacy(self):
        R_IN, R_OUT = 17.5, 19.5
        centers = jnp.zeros((1, 3))
        radii = jnp.asarray([R_IN])
        r = np.random.RandomState(0)
        u = r.normal(size=(2000, 3))
        o = jnp.asarray((r.uniform(0, R_OUT, 2000)[:, None] * u / np.linalg.norm(u, axis=1, keepdims=True)), jnp.float32)
        d = jnp.asarray(r.normal(size=(2000, 3)), jnp.float32)
        d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
        t_leg, hi_leg, _, _ = jax.vmap(lambda a, b: intersect_two_spheres_forward(a, b, R_IN, R_OUT))(o, d)

        def hit(a, b):
            t_out, _ = SF.sphere_forward_t(a, b, jnp.zeros(3), R_OUT)
            _t, _w, h, _p, _n = SF.nearest_interface(a, b, centers, radii, t_outer=t_out)
            return h

        hi_new = jax.vmap(hit)(o, d)
        assert int(jnp.sum(hi_leg != hi_new)) == 0


class TestSurfacesGradientFinite:
    """The eps-inside-sqrt fix: NaN-free gradients at tangent/miss rays + mixed stacks."""

    def test_tangent_and_miss(self):
        d = jnp.array([1., 0., 0.]); c = jnp.array([0., 0., 0.])
        for o in (jnp.array([0., 1., 0.]), jnp.array([0., 5., 0.])):   # tangent, clean miss
            g = jax.grad(lambda o: SF.sphere_forward_t(o, d, c, 1.0)[0])(o)
            assert bool(jnp.all(jnp.isfinite(g)))

    def test_mixed_hit_miss_stack(self):
        C = jnp.array([[0., 0., 0.], [100., 0., 0.]]); R = jnp.array([1., 1.])
        o = jnp.array([-5., 0., 0.]); d = jnp.array([1., 0., 0.])
        g = jax.grad(lambda C: SF.nearest_interface(o, d, C, R)[0])(C)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_region_of_empty(self):
        out = SF.region_of_spheres(jnp.zeros((4, 3)), jnp.zeros((0, 3)), jnp.zeros((0,)))
        assert out.shape == (4,) and int(out.sum()) == 0


class TestOuterMediaParams:
    """The outer_media (N-native per-region optics) layout: mask, bounds, save/load."""

    def _dp(self):
        from lucid.detector_params import DetectorParams
        return DetectorParams.from_flat(scatter_length=50., absorption_length=50.,
                                        qe=0.065, qe_corrections=jnp.ones(20))

    def test_default_single_medium(self):
        dp = self._dp()
        assert dp.outer_media == () and dp.outer_optics is None

    def test_with_outer_adds_seven_leaves(self):
        dp = self._dp()
        base = len(jax.tree_util.tree_leaves(dp))
        assert len(jax.tree_util.tree_leaves(dp.with_outer_optics())) - base == 7

    def test_mask_covers_outer(self):
        from lucid.detector_params import make_optimization_mask
        m = make_optimization_mask(self._dp().with_outer_optics(), {'scatter_length'})
        assert bool(m.outer_media[0].scattering.scatter_length)
        assert not bool(m.outer_media[0].absorption.absorption_length)

    def test_bounds_normalize_roundtrip(self):
        from lucid.detector_params import default_bounds, normalize_params, denormalize_params
        dp = self._dp().with_outer_optics()
        bmin, bmax = default_bounds(20, n_outer_media=1)
        back = denormalize_params(normalize_params(dp, bmin, bmax), bmin, bmax)
        assert bool(jnp.allclose(back.outer_media[0].scattering.scatter_length,
                                 dp.outer_media[0].scattering.scatter_length))

    def test_save_load_roundtrip(self, tmp_path):
        from lucid.detector_params import MediumParams, save_detector_params, load_detector_params
        dp = self._dp().with_outer_media(MediumParams(
            scattering=self._dp().scattering._replace(scatter_length=jnp.asarray(77.0)),
            absorption=self._dp().absorption._replace(absorption_length=jnp.asarray(123.0))))
        fp = str(tmp_path / "dp.json")
        save_detector_params(dp, fp)
        loaded = load_detector_params(fp, num_sensors=20)
        assert float(loaded.outer_media[0].scattering.scatter_length) == 77.0
        assert float(loaded.outer_media[0].absorption.absorption_length) == 123.0


class TestCosThetaOptIn:
    """cosθ acceptance is OPT-IN: off by default, on only when the config sets the flag."""

    def _propagator(self, apply_flag):
        import os, json, tempfile
        from lucid.geometry.detector_geometry import DetectorGeometry
        cfg = {"material": "water", "detector_type": "sphere",
               "geometry_definitions": {"radius": 5.0, "n_sensors": 600, "sensor_radius": 0.3}}
        if apply_flag:
            cfg["apply_angular_acceptance"] = True
        p = os.path.join(tempfile.mkdtemp(), "g.json")
        json.dump(cfg, open(p, "w"))
        return DetectorGeometry.from_config(p, detector_type="sphere", temperature=None).propagator

    def _detected(self, prop, origin, n=8000, seed=0):
        r = np.random.RandomState(seed)
        d = r.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(np.asarray(origin, np.float32), (n, 3))
        res = prop(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32))
        return float(jnp.sum(res['sensor_weights']))

    def test_default_off_overcounts_relative_to_opt_in(self):
        # Off-centre source near the wall: without cosθ (default) grazing rays over-count, so the
        # detected weight exceeds the cosθ-projected (opt-in) value. Proves the flag gates cosθ.
        off = self._detected(self._propagator(False), [0, 0, 4.0])
        on = self._detected(self._propagator(True), [0, 0, 4.0])
        assert off > on * 1.1
