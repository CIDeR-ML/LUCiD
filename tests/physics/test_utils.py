"""lucid/utils.py: the direction <-> angles codec, the shortest-arc rotation, and the
pseudo-data smearing models.

The smears are what the paper's pseudo-data are made with (``smear_times`` for the TTS,
``smear_charges_SK_like`` for ``Abe_2013``, ``smear_charges_spe`` for ``Bellamy_94``), so their
magnitudes are pinned, not only their signs.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.utils import (cartesian_to_spherical, rotation_onto, smear_charges_SK_like,
                         smear_charges_spe, smear_times, spherical_to_cartesian)

N = 20_000      # draws per magnitude check: the std of a sample std is ~1/sqrt(2N) = 0.5%


class TestSphericalToCartesian:
    def test_north_pole(self):
        result = spherical_to_cartesian(0.0, 0.0)
        npt.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-6)

    def test_equator_x(self):
        result = spherical_to_cartesian(jnp.pi / 2, 0.0)
        npt.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_equator_y(self):
        result = spherical_to_cartesian(jnp.pi / 2, jnp.pi / 2)
        npt.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_south_pole(self):
        result = spherical_to_cartesian(jnp.pi, 0.0)
        npt.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-6)

    def test_generic_point(self):
        """theta = 60 deg, phi = 30 deg: (sin60 cos30, sin60 sin30, cos60) = (3/4, sqrt3/4, 1/2),
        a point where no component is 0 or +-1 (the old check here was only |v| = 1, which a
        sin/cos swap also satisfies)."""
        result = spherical_to_cartesian(np.pi / 3, np.pi / 6)
        npt.assert_allclose(result, [0.75, np.sqrt(3) / 4, 0.5], atol=1e-6)


class TestCartesianToSpherical:
    """The inverse: angles of a unit direction, theta in [0, pi] and phi in [-pi, pi]."""

    @pytest.mark.parametrize('xp, atol', [(np, 1e-15), (jnp, 1e-6)])
    def test_round_trip_from_angles(self, xp, atol):
        """decode(encode) on a grid off the poles; phi outside [-pi, pi] comes back wrapped."""
        theta, phi = np.meshgrid([0.2, 1.0, np.pi / 2, 2.9], [-3.0, -0.5, 0.0, 1.2, 3.1, 4.0, 7.0])
        direction = spherical_to_cartesian(xp.asarray(theta), xp.asarray(phi), xp)
        t, p = cartesian_to_spherical(direction, xp)
        npt.assert_allclose(t, theta, atol=atol)
        npt.assert_allclose(p, np.angle(np.exp(1j * phi)), atol=atol)

    @pytest.mark.parametrize('xp', [np, jnp])
    def test_round_trip_from_directions(self, xp):
        """encode(decode) returns the direction, the poles and the -x axis (phi = +-pi) included."""
        rng = np.random.default_rng(0)
        d = rng.normal(size=(50, 3))
        d = np.concatenate([d, [[0, 0, 1], [0, 0, -1], [-1, 0, 0], [-1, -0.0, 0], [0, -1, 0]]])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        back = spherical_to_cartesian(*cartesian_to_spherical(xp.asarray(d.T), xp), xp)
        npt.assert_allclose(back, d.T, atol=1e-15 if xp is np else 1e-6)

    def test_poles_and_clipping(self):
        """theta is exactly 0 and pi at the poles; a z a rounding above 1 is clipped, not NaN."""
        assert cartesian_to_spherical(np.array([0., 0., 1.]), np)[0] == 0.0
        assert cartesian_to_spherical(np.array([0., 0., -1.]), np)[0] == np.pi
        assert cartesian_to_spherical(np.array([0., 0., 1. + 1e-15]), np)[0] == 0.0

    def test_particle_params_encodes_through_the_codec(self):
        """``ParticleParams.from_cartesian`` is this codec on the normalised direction, to the bit,
        for unit and unnormalised inputs, and its ``direction`` decodes back to ``d / |d|``."""
        from lucid.detector_params import ParticleParams
        rng = np.random.default_rng(1)
        d = rng.normal(size=(2000, 3)) * rng.uniform(0.1, 10.0, size=(2000, 1))
        d = jnp.asarray(np.concatenate([d / np.linalg.norm(d, axis=1, keepdims=True), d]),
                        jnp.float32)
        p = jax.vmap(lambda v: ParticleParams.from_cartesian(1.0, jnp.zeros(3), v))(d)
        theta, phi = jax.vmap(lambda v: cartesian_to_spherical(v / jnp.linalg.norm(v)))(d)
        npt.assert_array_equal(p.theta, theta)
        npt.assert_array_equal(p.phi, phi)
        npt.assert_allclose(jax.vmap(lambda q: q.direction)(p),
                            d / jnp.linalg.norm(d, axis=1, keepdims=True), atol=5e-6)


def _directions(rng, n=200):
    """Random unit directions plus the poles, and directions 1e-3 .. 1e-9 rad off each pole."""
    d = rng.normal(size=(n, 3))
    near = [[np.cos(p) * e, np.sin(p) * e, s] for e in (1e-3, 1e-6, 1e-9)
            for p in (0.3, 2.0) for s in (1.0, -1.0)]
    d = np.concatenate([d, [[0, 0, 1], [0, 0, -1]], near])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class TestRotationOnto:
    """The shortest-arc rotation taking direction ``a`` onto ``b``."""

    @pytest.mark.parametrize('xp, atol', [(np, 1e-14), (jnp, 2e-6)])
    def test_proper_rotation_taking_a_onto_b(self, xp, atol):
        """Orthonormal and det +1 to rounding, maps a to b and fixes a x b, for b random and near
        +-z (a = +z) and for random pairs; the inputs need not be unit. Near b = -a the arc is
        ill-conditioned: rounding of the inputs moves R by ~eps / |a + b| (unit a, b), so the
        last two hold to that."""
        rng = np.random.default_rng(0)
        b = _directions(rng)
        a = np.concatenate([np.tile([0.0, 0.0, 1.0], (len(b), 1)), rng.normal(size=b.shape)])
        b = np.concatenate([b, b]) * rng.uniform(0.5, 2.0, size=(2 * len(b), 1))
        for ai, bi in zip(a, b):
            R = np.asarray(rotation_onto(xp.asarray(ai), xp.asarray(bi), xp), np.float64)
            npt.assert_allclose(R @ R.T, np.eye(3), atol=atol)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=atol)
            a_unit, b_unit = ai / np.linalg.norm(ai), bi / np.linalg.norm(bi)
            conditioned = atol * (1 + 1 / max(np.linalg.norm(a_unit + b_unit), 1e-30))
            npt.assert_allclose(R @ a_unit, b_unit, atol=conditioned)
            axis = np.cross(a_unit, b_unit)
            npt.assert_allclose(R @ axis, axis, atol=conditioned)

    @pytest.mark.parametrize('xp', [np, jnp])
    def test_identity_and_the_half_turn(self, xp):
        """a onto a is the identity, exactly; +z onto -z is the documented half turn about +y."""
        z = xp.asarray([0.0, 0.0, 1.0])
        npt.assert_array_equal(rotation_onto(z, z, xp), np.eye(3))
        npt.assert_allclose(rotation_onto(z, -z, xp), np.diag([-1.0, 1.0, -1.0]), atol=1e-7)

    def test_gradient_matches_finite_differences(self):
        """jax.grad (float32) of w . R(a, b) v, in a and in b, against central differences of the
        float64 numpy path (h = 1e-6), for random pairs and for b at and within 1e-3 rad of +z
        with a = +z: the bisector form has no 0/0 there. Finite at b = -a, where the arc's axis
        is undefined."""
        rng = np.random.default_rng(1)
        v, w = rng.normal(size=3), rng.normal(size=3)

        def f(a, b, xp):
            return xp.dot(w, rotation_onto(a, b, xp) @ v)

        def central(a, b, h=1e-6):
            eye = np.eye(3)
            return [np.array([(f(*(x + h * e if k == i else y for k, y in enumerate((a, b))), np)
                               - f(*(x - h * e if k == i else y for k, y in enumerate((a, b))), np))
                              / (2 * h) for e in eye]) for i, x in enumerate((a, b))]

        grad = jax.jit(jax.grad(lambda a, b: f(a, b, jnp), argnums=(0, 1)))
        z = np.array([0.0, 0.0, 1.0])
        near_z = [z] + [np.array([np.cos(p) * e, np.sin(p) * e, 1.0]) for e in (1e-3, 1e-5)
                        for p in (0.0, 1.0, 2.5, 4.0)]
        pairs = [(z, b) for b in near_z] + [(rng.normal(size=3), rng.normal(size=3))
                                            for _ in range(20)]
        for a, b in pairs:
            got = grad(jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32))
            for g, want in zip(got, central(a, b)):
                npt.assert_allclose(np.asarray(g), want, atol=2e-5 * (1 + np.abs(want).max()))
        at_antipode = grad(jnp.asarray(z, jnp.float32), jnp.asarray(-z, jnp.float32))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in at_antipode)


class TestSmearTimes:
    def test_spread_is_the_time_resolution_and_the_mean_is_kept(self):
        """Gaussian noise of std ``time_resolution``: std within 3% (the sample's own scatter is
        0.5%) and the mean within 5 standard errors. At sigma = 2 ns, sigma^2 would give 4 ns;
        at 0.4 ns, 0.16 ns."""
        times = jnp.full(N, 100.0)
        for sigma in (0.4, 2.0):
            out = np.asarray(smear_times(times, sigma, jax.random.PRNGKey(42)), np.float64)
            assert out.std() == pytest.approx(sigma, rel=0.03), sigma
            assert abs(out.mean() - 100.0) < 5 * sigma / np.sqrt(N), sigma

    def test_default_is_the_sk_time_resolution(self):
        """The docstring's default, 0.4 ns (SK, arXiv:1307.0162)."""
        out = np.asarray(smear_times(jnp.full(N, 0.0), key=jax.random.PRNGKey(1)), np.float64)
        assert out.std() == pytest.approx(0.4, rel=0.03)


class TestSmearChargesSKLike:
    def test_fractional_resolution_per_charge_regime(self):
        """sigma/Q = 1.2% below 20 p.e., 0.75% below 130, 0.5% above: the coefficients the
        docstring attributes to arXiv:1307.0162 Table 2 (not re-checked against the paper here).
        Measured at Q = 10, 50 and 500, each within 3% of its own sigma."""
        for q, frac in ((10.0, 0.012), (50.0, 0.0075), (500.0, 0.005)):
            out = np.asarray(smear_charges_SK_like(jnp.full(N, q), jax.random.PRNGKey(7)),
                             np.float64)
            assert out.std() / q == pytest.approx(frac, rel=0.03), q
            assert abs(out.mean() - q) < 5 * frac * q / np.sqrt(N), q

    def test_matches_the_digitizer_copy_of_the_same_model(self):
        """``lucid.simulation.digitizer.charge_resolution_sigma('sk_like')`` is a second copy of
        this formula, kept for the ``basic`` digitizer's byte-parity with the historical output.
        Same key, same normal draw: the two must agree everywhere, including on which side of
        each regime boundary (20, 130) a charge falls."""
        from lucid.simulation.digitizer import charge_resolution_sigma
        q = jnp.array([0.5, 19.99, 20.0, 129.9, 130.0, 500.0])
        key = jax.random.PRNGKey(3)
        sigma = charge_resolution_sigma(np.asarray(q), 'sk_like')
        noise = np.asarray(jax.random.normal(key, shape=q.shape))
        want = np.clip(np.asarray(q) + noise * sigma, 0.0, None)
        npt.assert_allclose(np.asarray(smear_charges_SK_like(q, key)), want, rtol=1e-6)

    def test_never_negative_and_needs_a_key(self):
        out = smear_charges_SK_like(jnp.array([0.0, 0.01, 1.0]), jax.random.PRNGKey(0))
        assert bool(jnp.all(out >= 0))
        with pytest.raises(ValueError, match='key'):
            smear_charges_SK_like(jnp.array([1.0]))


class TestSmearChargesSPE:
    def test_mean_charge_is_the_photoelectron_count(self):
        """``<Q> = N``: the Bellamy mixture is normalised by its fit_mean, which for the SK
        parameters is the mixture mean (1-w) mu + w tau = 0.85232 vs fit_mean 0.852 (0.04%).

        The spread is the compound sum of N iid single-p.e. draws,
        Var Q = N [(1-w)(sig^2 + mu^2) + 2 w tau^2 - m1^2] / fit_mean^2 (textbook, independent of
        the module's binomial split). Q = 50 p.e. on 2e4 sensors: mean to 0.3%, std to 3%."""
        from lucid.simulation.digitizer import _SPE_SK as spe
        w, mu, sig, tau, fm = spe['w'], spe['mu'], spe['sig'], spe['tau'], spe['fit_mean']
        m1 = (1 - w) * mu + w * tau
        assert m1 == pytest.approx(fm, rel=1e-3)
        var1 = (1 - w) * (sig ** 2 + mu ** 2) + w * 2 * tau ** 2 - m1 ** 2

        n_pe = 50.0
        out = np.asarray(smear_charges_spe(jnp.full(N, n_pe), jax.random.PRNGKey(5)), np.float64)
        assert out.mean() == pytest.approx(n_pe * m1 / fm, rel=3e-3)
        assert out.std() == pytest.approx(np.sqrt(n_pe * var1) / fm, rel=0.03)

    def test_needs_a_key(self):
        with pytest.raises(ValueError, match='key'):
            smear_charges_spe(jnp.array([1.0]))
