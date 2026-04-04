"""Physics-level tests for optics functions.

These tests verify the physical correctness of the optics implementations,
not just that they run or match reference values.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.simulation.optics import (
    normalize, compute_reflection_direction, create_local_frame,
    sample_scatter_distance, solve_rayleigh_inverse_cdf,
    compute_scatter_direction, sample_cosine_hemisphere,
    jax_normalize,
)
from lucid.utils import jax_rotate_vector


class TestReflectionPhysics:
    """Verify reflection obeys the law of reflection."""

    def test_angle_of_incidence_equals_reflection(self):
        """The angle between incident ray and normal should equal the angle
        between reflected ray and normal."""
        normal = jnp.array([0.0, 1.0, 0.0])
        for angle_deg in [15, 30, 45, 60, 75]:
            theta = jnp.deg2rad(angle_deg)
            incident = jnp.array([jnp.sin(theta), -jnp.cos(theta), 0.0])
            reflected = compute_reflection_direction(incident, normal)
            cos_in = -jnp.dot(incident, normal)
            cos_out = jnp.dot(reflected, normal)
            npt.assert_allclose(cos_in, cos_out, atol=1e-5,
                                err_msg=f"Failed at {angle_deg} degrees")

    def test_reflection_in_plane_of_incidence(self):
        """Reflected ray must lie in the plane defined by incident ray and normal."""
        normal = jnp.array([0.0, 0.0, 1.0])
        incident = normalize(jnp.array([1.0, 0.5, -1.0]))
        reflected = compute_reflection_direction(incident, normal)
        # Cross product of (incident × normal) and (reflected × normal) should be parallel
        # to normal, meaning reflected is in the incidence plane
        cross_in = jnp.cross(incident, normal)
        cross_out = jnp.cross(reflected, normal)
        # They should be parallel (or anti-parallel)
        ratio = cross_in / (cross_out + 1e-10)
        npt.assert_allclose(ratio[0], ratio[1], atol=1e-4)

    def test_reflection_reverses_normal_component(self):
        """The component along the normal should flip sign; tangent stays."""
        normal = jnp.array([0.0, 1.0, 0.0])
        incident = normalize(jnp.array([1.0, -1.0, 0.3]))
        reflected = compute_reflection_direction(incident, normal)
        # Normal component flips
        npt.assert_allclose(jnp.dot(reflected, normal),
                            -jnp.dot(incident, normal), atol=1e-5)
        # Tangential components preserved
        tangent_in = incident - jnp.dot(incident, normal) * normal
        tangent_out = reflected - jnp.dot(reflected, normal) * normal
        npt.assert_allclose(tangent_in, tangent_out, atol=1e-5)


class TestLocalFramePhysics:
    """Verify create_local_frame produces valid orthonormal frames."""

    def test_orthogonality(self):
        """All three basis vectors must be mutually orthogonal."""
        for z in [jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.]),
                  normalize(jnp.array([1., 1., 1.]))]:
            frame = create_local_frame(z)
            npt.assert_allclose(jnp.dot(frame[0], frame[1]), 0.0, atol=1e-6)
            npt.assert_allclose(jnp.dot(frame[0], frame[2]), 0.0, atol=1e-6)
            npt.assert_allclose(jnp.dot(frame[1], frame[2]), 0.0, atol=1e-6)

    def test_all_unit_vectors(self):
        for z in [jnp.array([0., 0., 1.]), normalize(jnp.array([3., -1., 2.]))]:
            frame = create_local_frame(z)
            for i in range(3):
                npt.assert_allclose(jnp.linalg.norm(frame[i]), 1.0, atol=1e-5)

    def test_right_handed(self):
        """Frame should be right-handed: det(frame) = +1."""
        for z in [jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.]),
                  normalize(jnp.array([1., 2., 3.]))]:
            frame = create_local_frame(z)
            det = jnp.linalg.det(frame)
            npt.assert_allclose(det, 1.0, atol=1e-5)

    def test_z_axis_preserved(self):
        """The third row of the frame should be the normalized input."""
        z = normalize(jnp.array([3., -1., 2.]))
        frame = create_local_frame(z)
        npt.assert_allclose(frame[2], z, atol=1e-5)


class TestScatterDistancePhysics:
    """Verify truncated exponential sampling physics."""

    def test_always_less_than_D(self):
        """Scatter distance must be < D (photon scatters before surface)."""
        key = jax.random.PRNGKey(0)
        D, S = 5.0, 2.0
        for i in range(100):
            k = jax.random.fold_in(key, i)
            d = sample_scatter_distance(D, S, k)
            assert d < D + 1e-6

    def test_mean_matches_truncated_exponential(self):
        """Statistical mean should match the truncated exponential E[x | x < D]."""
        key = jax.random.PRNGKey(42)
        D, S = 5.0, 2.0
        samples = jax.vmap(lambda k: sample_scatter_distance(D, S, k))(
            jax.random.split(key, 50000))
        # E[X | X < D] = S - D * exp(-D/S) / (1 - exp(-D/S))
        exp_term = jnp.exp(-D / S)
        expected_mean = S - D * exp_term / (1 - exp_term)
        npt.assert_allclose(jnp.mean(samples), expected_mean, atol=0.02)

    def test_large_scatter_length_uniform_like(self):
        """When S >> D, scatter distance should be nearly uniform in [0, D]."""
        key = jax.random.PRNGKey(0)
        D, S = 1.0, 1000.0
        samples = jax.vmap(lambda k: sample_scatter_distance(D, S, k))(
            jax.random.split(key, 10000))
        npt.assert_allclose(jnp.mean(samples), D / 2, atol=0.05)


class TestRayleighPhaseFunction:
    """Verify the Rayleigh scattering distribution P(cosθ) ∝ (1 + cos²θ)."""

    def test_inverse_cdf_matches_distribution(self):
        """Sample many values and check the histogram matches (1+cos²θ)."""
        u_vals = jnp.linspace(0.001, 0.999, 10000)
        cos_thetas = jax.vmap(solve_rayleigh_inverse_cdf)(u_vals)
        # Bin the cos_theta values and compare to expected PDF
        bins = jnp.linspace(-1, 1, 51)
        hist, _ = jnp.histogram(cos_thetas, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        expected = 1 + centers**2  # unnormalized PDF
        expected_norm = expected / jnp.sum(expected)
        hist_norm = hist / jnp.sum(hist)
        npt.assert_allclose(hist_norm, expected_norm, atol=0.02)

    def test_symmetry(self):
        """P(cosθ) = P(-cosθ) since 1+cos²θ is symmetric."""
        for u in [0.1, 0.2, 0.3, 0.4]:
            mu_low = solve_rayleigh_inverse_cdf(u)
            mu_high = solve_rayleigh_inverse_cdf(1.0 - u)
            npt.assert_allclose(mu_low, -mu_high, atol=1e-4)


class TestCosineHemispherePhysics:
    """Verify cosine-weighted hemisphere sampling."""

    def test_all_on_correct_hemisphere(self):
        """All samples should have positive dot product with the normal."""
        key = jax.random.PRNGKey(0)
        normal = jnp.array([0.0, 0.0, 1.0])
        keys = jax.random.split(key, 500)
        samples = jax.vmap(lambda k: sample_cosine_hemisphere(normal, k))(keys)
        dots = jax.vmap(lambda s: jnp.dot(s, normal))(samples)
        assert jnp.all(dots >= -1e-6), "Sample on wrong hemisphere"

    def test_mean_cos_theta_is_two_thirds(self):
        """For cosine-weighted: E[cosθ] = 2/3 (integral of cos²θ sinθ)."""
        key = jax.random.PRNGKey(42)
        normal = jnp.array([0.0, 0.0, 1.0])
        keys = jax.random.split(key, 50000)
        samples = jax.vmap(lambda k: sample_cosine_hemisphere(normal, k))(keys)
        cos_thetas = jax.vmap(lambda s: jnp.dot(s, normal))(samples)
        npt.assert_allclose(jnp.mean(cos_thetas), 2.0 / 3.0, atol=0.01)

    def test_works_for_arbitrary_normal(self):
        """Should work correctly for non-axis-aligned normals."""
        key = jax.random.PRNGKey(0)
        normal = normalize(jnp.array([1.0, 1.0, 1.0]))
        keys = jax.random.split(key, 1000)
        samples = jax.vmap(lambda k: sample_cosine_hemisphere(normal, k))(keys)
        dots = jax.vmap(lambda s: jnp.dot(s, normal))(samples)
        assert jnp.all(dots >= -1e-5)
        npt.assert_allclose(jnp.mean(dots), 2.0 / 3.0, atol=0.03)


class TestScatterDirectionPhysics:
    """Verify Rayleigh scattering direction sampling."""

    def test_mean_cos_theta_matches_rayleigh(self):
        """E[cosθ] for Rayleigh P(μ)∝(1+μ²) should be 0 (symmetric distribution)."""
        key = jax.random.PRNGKey(42)
        incident = jnp.array([0.0, 0.0, 1.0])
        keys = jax.random.split(key, 20000)
        scattered = jax.vmap(lambda k: compute_scatter_direction(incident, k))(keys)
        cos_thetas = jax.vmap(lambda s: jnp.dot(s, incident))(scattered)
        npt.assert_allclose(jnp.mean(cos_thetas), 0.0, atol=0.02)

    def test_azimuthal_symmetry(self):
        """Scattering should be azimuthally symmetric around the incident direction."""
        key = jax.random.PRNGKey(0)
        incident = jnp.array([0.0, 0.0, 1.0])
        keys = jax.random.split(key, 10000)
        scattered = jax.vmap(lambda k: compute_scatter_direction(incident, k))(keys)
        # Mean x and y components should be ~0 (azimuthal symmetry)
        npt.assert_allclose(jnp.mean(scattered[:, 0]), 0.0, atol=0.02)
        npt.assert_allclose(jnp.mean(scattered[:, 1]), 0.0, atol=0.02)


class TestRotateVectorPhysics:
    def test_rotation_preserves_dot_with_axis(self):
        """Component along the rotation axis should be unchanged."""
        vec = jnp.array([1.0, 2.0, 3.0])
        axis = normalize(jnp.array([0.0, 1.0, 0.0]))
        rotated = jax_rotate_vector(vec, axis, 1.23)
        npt.assert_allclose(jnp.dot(rotated, axis), jnp.dot(vec, axis), atol=1e-5)

    def test_small_angle_approximation(self):
        """For small angles, rotated ≈ original + angle * (axis × vector)."""
        vec = jnp.array([1.0, 0.0, 0.0])
        axis = jnp.array([0.0, 0.0, 1.0])
        angle = 0.001
        rotated = jax_rotate_vector(vec, axis, angle)
        expected = vec + angle * jnp.cross(axis, vec)
        npt.assert_allclose(rotated, expected, atol=1e-5)


class TestNormalizePhysics:
    def test_gradient_exists_for_nonzero(self):
        """normalize should be differentiable for non-zero vectors."""
        def loss(v):
            return jnp.sum(normalize(v))
        g = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_batch_support(self):
        """Should handle (N, 3) batches correctly."""
        v = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        result = normalize(v)
        npt.assert_allclose(jnp.linalg.norm(result, axis=-1), [1.0, 1.0], atol=1e-5)
