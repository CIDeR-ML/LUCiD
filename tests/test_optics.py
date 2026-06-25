"""Tests for optics functions (simulation.py lines 33-147)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation import (
    normalize, jax_normalize, compute_reflection_direction,
    create_local_frame, sample_scatter_distance, solve_rayleigh_inverse_cdf,
    compute_scatter_direction, sample_cosine_hemisphere, jax_rotate_vector,
)


# ── Deterministic tests ─────────────────────────────────────────────

class TestNormalize:
    def test_unit_vector(self):
        result = normalize(jnp.array([3.0, 4.0, 0.0]))
        npt.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-6)

    def test_already_unit(self):
        v = jnp.array([0.0, 1.0, 0.0])
        npt.assert_allclose(normalize(v), [0.0, 1.0, 0.0], atol=1e-7)

    def test_zero_vector(self):
        result = normalize(jnp.array([0.0, 0.0, 0.0]))
        npt.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-7)

    def test_output_norm_is_one(self):
        v = jnp.array([2.5, -3.1, 0.7])
        result = normalize(v)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-5)


class TestJaxNormalize:
    def test_basic(self):
        result = jax_normalize(jnp.array([3.0, 4.0, 0.0]))
        npt.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-6)

    def test_zero_vector(self):
        result = jax_normalize(jnp.array([0.0, 0.0, 0.0]))
        npt.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-7)

    def test_negative_components(self):
        v = jnp.array([-3.0, 4.0, 0.0])
        result = jax_normalize(v)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-5)


class TestReflection:
    def test_45_degree_off_wall(self):
        incident = jnp.array([1.0, -1.0, 0.0]) / jnp.sqrt(2.0)
        normal = jnp.array([0.0, 1.0, 0.0])
        result = compute_reflection_direction(incident, normal)
        expected = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_head_on(self):
        incident = jnp.array([0.0, -1.0, 0.0])
        normal = jnp.array([0.0, 1.0, 0.0])
        result = compute_reflection_direction(incident, normal)
        npt.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_preserves_magnitude(self):
        incident = jnp.array([1.0, -1.0, 0.0]) / jnp.sqrt(2.0)
        normal = jnp.array([0.0, 1.0, 0.0])
        result = compute_reflection_direction(incident, normal)
        npt.assert_allclose(jnp.linalg.norm(result), jnp.linalg.norm(incident), atol=1e-6)


class TestLocalFrame:
    def test_z_axis(self):
        frame = create_local_frame(jnp.array([0.0, 0.0, 1.0]))
        # frame[2] should be the z-axis itself
        npt.assert_allclose(frame[2], [0.0, 0.0, 1.0], atol=1e-6)
        # orthogonality
        npt.assert_allclose(jnp.dot(frame[0], frame[1]), 0.0, atol=1e-6)
        npt.assert_allclose(jnp.dot(frame[0], frame[2]), 0.0, atol=1e-6)

    def test_x_axis(self):
        frame = create_local_frame(jnp.array([1.0, 0.0, 0.0]))
        npt.assert_allclose(frame[2], [1.0, 0.0, 0.0], atol=1e-6)
        npt.assert_allclose(jnp.dot(frame[0], frame[1]), 0.0, atol=1e-6)


class TestRayleighInverseCDF:
    def test_median(self):
        assert abs(solve_rayleigh_inverse_cdf(0.5) - 0.0) < 1e-5

    def test_min(self):
        assert abs(solve_rayleigh_inverse_cdf(0.0) - (-1.0)) < 1e-5

    def test_max(self):
        assert abs(solve_rayleigh_inverse_cdf(1.0) - 1.0) < 1e-5

    def test_monotonic(self):
        u_vals = jnp.linspace(0.0, 1.0, 50)
        results = jax.vmap(solve_rayleigh_inverse_cdf)(u_vals)
        diffs = jnp.diff(results)
        assert jnp.all(diffs >= -1e-6), "Rayleigh inverse CDF should be monotonically non-decreasing"


class TestRotateVector:
    def test_90_around_z(self):
        vec = jnp.array([1.0, 0.0, 0.0])
        axis = jnp.array([0.0, 0.0, 1.0])
        result = jax_rotate_vector(vec, axis, jnp.pi / 2)
        npt.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_360_is_identity(self):
        vec = jnp.array([0.3, 0.5, 0.8])
        axis = jnp.array([0.0, 0.0, 1.0])
        result = jax_rotate_vector(vec, axis, 2 * jnp.pi)
        npt.assert_allclose(result, vec, atol=1e-5)

    def test_preserves_norm(self):
        vec = jnp.array([1.0, 2.0, 3.0])
        axis = jnp.array([0.0, 1.0, 0.0])
        result = jax_rotate_vector(vec, axis, 1.23)
        npt.assert_allclose(jnp.linalg.norm(result), jnp.linalg.norm(vec), atol=1e-5)


# ── Stochastic tests (exact match with fixed seed on CPU) ───────────

class TestScatterDistance:
    def test_bounded_by_D(self):
        key = jax.random.PRNGKey(0)
        for i in range(10):
            k = jax.random.fold_in(key, i)
            d = sample_scatter_distance(3.0, 2.0, k)
            assert d <= 3.0 + 1e-6, f"Scatter distance {d} exceeds D=3.0"


class TestScatterDirection:
    def test_output_is_unit(self):
        key = jax.random.PRNGKey(7)
        result = compute_scatter_direction(jnp.array([1.0, 0.0, 0.0]), key)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-5)


class TestCosineHemisphere:
    def test_output_is_unit(self):
        key = jax.random.PRNGKey(99)
        result = sample_cosine_hemisphere(jnp.array([0.0, 1.0, 0.0]), key)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-5)

    def test_positive_dot_with_normal(self):
        """Cosine hemisphere sample should be on the same side as the normal."""
        key = jax.random.PRNGKey(0)
        normal = jnp.array([0.0, 0.0, 1.0])
        for i in range(20):
            k = jax.random.fold_in(key, i)
            s = sample_cosine_hemisphere(normal, k)
            assert jnp.dot(s, normal) >= -1e-6, "Sample on wrong side of hemisphere"
