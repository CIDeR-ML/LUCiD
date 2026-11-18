"""Physics tests for the pure optics functions in ``lucid.simulation.optics``.

Every check here has an independent oracle: the law of reflection, an
orthonormal right-handed frame, the truncated-exponential mean, the Rayleigh
CDF ``F(mu) = (mu^3 + 3 mu + 4) / 8`` and its moments, and the Lambertian
``E[cos] = 2/3``. Nothing here builds a detector, so it all runs in the default
lane.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.simulation.optics import (
    compute_reflection_direction, sample_scatter_distance, solve_rayleigh_inverse_cdf,
    compute_scatter_direction, sample_cosine_hemisphere,
)
from lucid.utils import create_local_frame, normalize


# Two oblique photon directions: one takes create_local_frame's |z_x| < 0.9 branch,
# the other its |z_x| >= 0.9 branch. An axis-aligned direction cannot tell the
# local->world map from its inverse (frame vs frame.T), so none is used for
# direction-sampling tests.
OBLIQUE_DIRS = [
    np.array([1.0, 2.0, -2.0]) / 3.0,
    np.array([0.95, 0.1, 0.3]) / np.linalg.norm([0.95, 0.1, 0.3]),
]


def rayleigh_second_moment(inc):
    """E[d d^T] for Rayleigh scattering about ``inc``.

    With mu = d.inc, E[mu^2] = 3/8 * int (1 + mu^2) mu^2 dmu = 2/5; the azimuth is
    uniform, so the transverse part is (1 - 2/5)/2 = 3/10 in each direction:
    E[d d^T] = 3/10 I + (2/5 - 3/10) inc inc^T. Isotropic scattering gives I/3.
    """
    return 0.3 * np.eye(3) + 0.1 * np.outer(inc, inc)


class TestNormalizePhysics:
    def test_values(self):
        npt.assert_allclose(normalize(jnp.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0], atol=1e-6)

    def test_zero_vector_stays_zero(self):
        """The eps floor returns 0, not NaN, for a zero vector."""
        npt.assert_array_equal(normalize(jnp.zeros(3)), np.zeros(3))

    def test_batch_support(self):
        """(N, 3) batches are normalised row by row, not over the whole array."""
        v = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        npt.assert_allclose(normalize(v), [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], atol=1e-6)

    def test_gradient_is_the_projector(self):
        """d(v/|v|)/dv = (I - u u^T)/|v|, the textbook Jacobian of a normaliser."""
        v = jnp.array([1.0, 2.0, 3.0])
        J = jax.jacobian(normalize)(v)
        u = np.asarray(v) / np.linalg.norm(v)
        npt.assert_allclose(J, (np.eye(3) - np.outer(u, u)) / np.linalg.norm(v), atol=1e-6)


class TestReflectionPhysics:
    """Law of reflection: d' = d - 2 (d.n) n."""

    def test_45_degrees_and_head_on(self):
        n = jnp.array([0.0, 1.0, 0.0])
        d = jnp.array([1.0, -1.0, 0.0]) / jnp.sqrt(2.0)
        npt.assert_allclose(compute_reflection_direction(d, n),
                            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), atol=1e-6)
        npt.assert_allclose(compute_reflection_direction(jnp.array([0.0, -1.0, 0.0]), n),
                            [0.0, 1.0, 0.0], atol=1e-6)

    def test_angle_of_incidence_equals_reflection(self):
        normal = jnp.array([0.0, 1.0, 0.0])
        for angle_deg in [15, 30, 45, 60, 75]:
            theta = jnp.deg2rad(angle_deg)
            incident = jnp.array([jnp.sin(theta), -jnp.cos(theta), 0.0])
            reflected = compute_reflection_direction(incident, normal)
            npt.assert_allclose(-jnp.dot(incident, normal), jnp.dot(reflected, normal),
                                atol=1e-5, err_msg=f"Failed at {angle_deg} degrees")

    def test_reflection_reverses_normal_component(self):
        """The normal component flips sign, the tangential part is unchanged, and the
        normal's own sign or length does not matter."""
        incident = normalize(jnp.array([1.0, -1.0, 0.3]))
        n = jnp.array([0.0, 1.0, 0.0])
        reflected = compute_reflection_direction(incident, n)
        npt.assert_allclose(jnp.dot(reflected, n), -jnp.dot(incident, n), atol=1e-5)
        tangent_in = incident - jnp.dot(incident, n) * n
        tangent_out = reflected - jnp.dot(reflected, n) * n
        npt.assert_allclose(tangent_in, tangent_out, atol=1e-5)
        npt.assert_allclose(compute_reflection_direction(incident, -3.0 * n), reflected, atol=1e-6)


class TestLocalFramePhysics:
    """create_local_frame returns an orthonormal right-handed frame whose third row is z."""

    @pytest.mark.parametrize("z", [[0., 0., 1.], [1., 0., 0.], [1., 1., 1.], [3., -1., 2.],
                                   [0.95, 0.1, 0.3]])
    def test_orthonormal_right_handed_z_preserved(self, z):
        z = normalize(jnp.array(z))
        frame = create_local_frame(z)
        npt.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-6)
        npt.assert_allclose(jnp.linalg.det(frame), 1.0, atol=1e-5)
        npt.assert_allclose(frame[2], z, atol=1e-6)

    @pytest.mark.parametrize("z", OBLIQUE_DIRS + [np.array([0., 0., -1.])])
    def test_the_sources_place_the_azimuth_in_this_frame(self, z):
        """The cascade source's (theta, phi) about its axis is the local direction mapped by this
        frame, as the Cherenkov cone and the laser are: one azimuth origin across lucid."""
        from lucid.sources.cascade import _cone_to_world
        theta = jnp.array([0.3, 0.7, 2.0, 0.7])
        phi = jnp.array([0.0, 1.0, 4.0, np.pi])
        local = jnp.stack([jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi),
                           jnp.cos(theta)], axis=1)
        npt.assert_allclose(_cone_to_world(jnp.asarray(z), theta, phi),
                            local @ create_local_frame(jnp.asarray(z)), atol=1e-6)


class TestScatterDistancePhysics:
    """sample_scatter_distance draws from exp(-x/S) truncated to [0, D]."""

    def test_always_less_than_D(self):
        D, S = 5.0, 2.0
        d = jax.vmap(lambda k: sample_scatter_distance(D, S, k))(
            jax.random.split(jax.random.PRNGKey(0), 1000))
        assert float(d.min()) >= 0.0 and float(d.max()) < D + 1e-6

    def test_mean_matches_truncated_exponential(self):
        """E[x | x < D] = S - D e^{-D/S} / (1 - e^{-D/S}) = 1.4483 for D=5, S=2.
        The sample std is about 1.3, so the 5e4-draw mean is good to 0.006 (1 sigma)."""
        D, S = 5.0, 2.0
        samples = jax.vmap(lambda k: sample_scatter_distance(D, S, k))(
            jax.random.split(jax.random.PRNGKey(42), 50000))
        e = np.exp(-D / S)
        npt.assert_allclose(jnp.mean(samples), S - D * e / (1 - e), atol=0.02)

    def test_large_scatter_length_uniform_like(self):
        """S >> D: the truncated exponential is uniform on [0, D], mean D/2."""
        D, S = 1.0, 1000.0
        samples = jax.vmap(lambda k: sample_scatter_distance(D, S, k))(
            jax.random.split(jax.random.PRNGKey(0), 10000))
        npt.assert_allclose(jnp.mean(samples), D / 2, atol=0.01)


class TestRayleighPhaseFunction:
    """The inverse CDF of P(mu) = 3/8 (1 + mu^2), whose CDF is F(mu) = (mu^3 + 3 mu + 4)/8."""

    def test_endpoints_and_median(self):
        npt.assert_allclose(solve_rayleigh_inverse_cdf(0.0), -1.0, atol=1e-5)
        npt.assert_allclose(solve_rayleigh_inverse_cdf(0.5), 0.0, atol=1e-5)
        npt.assert_allclose(solve_rayleigh_inverse_cdf(1.0), 1.0, atol=1e-5)

    def test_is_the_inverse_of_the_analytic_cdf(self):
        """F(F^-1(u)) = u on a grid: a pointwise check of the Cardano solution."""
        u = jnp.linspace(0.0, 1.0, 1001)
        mu = np.asarray(jax.vmap(solve_rayleigh_inverse_cdf)(u), dtype=np.float64)
        npt.assert_allclose((mu**3 + 3 * mu + 4) / 8, np.asarray(u), atol=2e-6)
        assert np.all(np.diff(mu) >= 0.0)

    def test_histogram_matches_analytic_bin_masses(self):
        """Midpoint quantiles u_i = (i + 1/2)/N give a histogram whose bin masses equal
        F(b_{k+1}) - F(b_k) to within 1/N = 5e-5 per bin. Isotropic scattering misses by
        up to 0.0094 per bin, so the tolerance 5e-4 separates the two by 19x."""
        N = 20000
        u = (jnp.arange(N) + 0.5) / N
        mu = np.asarray(jax.vmap(solve_rayleigh_inverse_cdf)(u))
        bins = np.linspace(-1.0, 1.0, 51)
        hist, _ = np.histogram(mu, bins=bins)
        F = (bins**3 + 3 * bins + 4) / 8
        npt.assert_allclose(hist / N, np.diff(F), atol=5e-4)

    def test_second_moment_is_two_fifths(self):
        """E[mu^2] = 2/5 for Rayleigh (isotropic: 1/3)."""
        N = 20000
        mu = jax.vmap(solve_rayleigh_inverse_cdf)((jnp.arange(N) + 0.5) / N)
        npt.assert_allclose(jnp.mean(mu**2), 0.4, atol=1e-4)

    def test_symmetry(self):
        """P(mu) = P(-mu), so F^-1(u) = -F^-1(1 - u)."""
        u = jnp.linspace(0.0, 0.5, 26)
        npt.assert_allclose(jax.vmap(solve_rayleigh_inverse_cdf)(u),
                            -jax.vmap(solve_rayleigh_inverse_cdf)(1.0 - u), atol=1e-4)


class TestScatterDirectionPhysics:
    """compute_scatter_direction samples Rayleigh scattering about the photon direction."""

    @pytest.mark.parametrize("inc", OBLIQUE_DIRS, ids=["frame-branch-x", "frame-branch-y"])
    def test_moments_about_an_oblique_direction(self, inc):
        """E[d] = 0 and E[d d^T] = 3/10 I + 1/10 inc inc^T (see rayleigh_second_moment).

        At N = 2e5 the per-entry standard error is <= 7e-4 (sqrt(Var mu^2) = 0.31),
        so atol 4e-3 is ~6 sigma. Isotropic scattering misses the inc-inc projection
        by 0.067; scattering about a fixed z axis misses entries by up to ~0.09.
        """
        keys = jax.random.split(jax.random.PRNGKey(42), 200_000)
        d = np.asarray(jax.jit(jax.vmap(
            lambda k: compute_scatter_direction(jnp.asarray(inc), k)))(keys), np.float64)
        npt.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
        npt.assert_allclose(d.mean(axis=0), 0.0, atol=6e-3)
        npt.assert_allclose(d.T @ d / len(d), rayleigh_second_moment(inc), atol=4e-3)


class TestCosineHemispherePhysics:
    """Lambertian sampling about a normal: p(theta) ~ cos theta sin theta."""

    @pytest.mark.parametrize("normal", [np.array([0.0, 0.0, 1.0])] + OBLIQUE_DIRS,
                             ids=["z", "oblique-x", "oblique-y"])
    def test_hemisphere_and_moments(self, normal):
        """All samples lie on the normal's side, E[cos] = 2/3, E[cos^2] = 1/2, and the
        tangential mean is zero (a transposed frame tilts the mean off the normal)."""
        keys = jax.random.split(jax.random.PRNGKey(0), 50000)
        s = np.asarray(jax.jit(jax.vmap(
            lambda k: sample_cosine_hemisphere(jnp.asarray(normal), k)))(keys), np.float64)
        c = s @ normal
        assert c.min() >= -1e-6
        # std of cos is sqrt(1/2 - 4/9) = 0.24 -> 1e-3 standard error at 5e4 draws
        npt.assert_allclose(c.mean(), 2.0 / 3.0, atol=5e-3)
        npt.assert_allclose((c**2).mean(), 0.5, atol=5e-3)
        npt.assert_allclose(s.mean(axis=0), 2.0 / 3.0 * normal, atol=8e-3)
