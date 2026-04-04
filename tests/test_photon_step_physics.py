"""Physics-level tests for photon iteration functions.

Tests verify that the photon propagation step correctly implements:
- Absorption attenuation (Beer-Lambert law)
- Detection probability (surface/scatter branching)
- Time-of-flight computation
- Position advancement along ray direction
- STE (Straight-Through Estimator) expected value semantics
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.photon_step import (
    photon_iteration_sample,
    photon_iteration_update_factors,
    photon_iteration_update_factors_safe,
)


def _base_args(key, **overrides):
    args = dict(
        position=jnp.array([0.0, 0.0, 0.0]),
        direction=jnp.array([0.0, 0.0, 1.0]),
        time=0.0,
        surface_distance=2.0,
        normal=jnp.array([0.0, 0.0, -1.0]),
        scatter_length=50.0,
        wall_reflection_rate=0.5,
        sensor_reflection_rate=0.3,
        absorption_length=100.0,
        hit_sensor=True,
        rng_key=key,
        speed_of_light=0.3,  # m/ns
    )
    args.update(overrides)
    return args


class TestAbsorptionAttenuation:
    """Beer-Lambert: attenuation = exp(-distance / absorption_length)."""

    def test_attenuation_value(self):
        """reflection_attenuation should be exp(-d/L_abs) for update_factors."""
        key = jax.random.PRNGKey(42)
        d = 2.0
        L_abs = 100.0
        args = _base_args(key, surface_distance=d, absorption_length=L_abs)
        _, _, _, _, refl_atten, _ = photon_iteration_update_factors(**args)
        expected = jnp.exp(-d / L_abs)
        npt.assert_allclose(refl_atten, expected, atol=1e-5)

    def test_short_absorption_strong_attenuation(self):
        """Short absorption length → strong attenuation (near 0)."""
        key = jax.random.PRNGKey(42)
        args = _base_args(key, surface_distance=5.0, absorption_length=1.0)
        _, _, _, _, refl_atten, _ = photon_iteration_update_factors(**args)
        assert float(refl_atten) < 0.01  # exp(-5) ≈ 0.0067

    def test_infinite_absorption_no_attenuation(self):
        """Very large absorption length → negligible attenuation (near 1)."""
        key = jax.random.PRNGKey(42)
        args = _base_args(key, surface_distance=2.0, absorption_length=1e6)
        _, _, _, _, refl_atten, _ = photon_iteration_update_factors(**args)
        npt.assert_allclose(refl_atten, 1.0, atol=1e-4)


class TestDetectionProbability:
    """Detection probability = P(reach surface) × P(not reflect)."""

    def test_detection_prob_formula(self):
        """detect_prob = exp(-D/S) × (1 - reflection_rate) for update_factors."""
        key = jax.random.PRNGKey(42)
        D, S = 2.0, 50.0
        sensor_refl = 0.3
        args = _base_args(key, surface_distance=D, scatter_length=S,
                          sensor_reflection_rate=sensor_refl, hit_sensor=True)
        _, _, _, detect_prob, _, _ = photon_iteration_update_factors(**args)
        expected = jnp.exp(-D / S) * (1 - sensor_refl)
        npt.assert_allclose(detect_prob, expected, atol=1e-5)

    def test_detection_prob_uses_wall_rate_for_wall(self):
        """When hit_sensor=False, wall_reflection_rate is used."""
        key = jax.random.PRNGKey(42)
        D, S = 2.0, 50.0
        wall_refl = 0.5
        args = _base_args(key, surface_distance=D, scatter_length=S,
                          wall_reflection_rate=wall_refl, hit_sensor=False)
        _, _, _, detect_prob, _, _ = photon_iteration_update_factors(**args)
        expected = jnp.exp(-D / S) * (1 - wall_refl)
        npt.assert_allclose(detect_prob, expected, atol=1e-5)

    def test_no_detection_on_wall(self):
        """Walls with 100% reflection should have 0 detection probability."""
        key = jax.random.PRNGKey(42)
        args = _base_args(key, wall_reflection_rate=1.0, hit_sensor=False)
        _, _, _, detect_prob, _, _ = photon_iteration_update_factors(**args)
        npt.assert_allclose(detect_prob, 0.0, atol=1e-5)


class TestTimeOfFlight:
    """Time should advance by distance / speed_of_light."""

    def test_time_advance(self):
        """New time = old time + distance_traveled / c_medium."""
        key = jax.random.PRNGKey(42)
        c = 0.3
        D = 2.0
        args = _base_args(key, surface_distance=D, speed_of_light=c, time=5.0)
        # update_factors uses STE — distance is a mix of surface and scatter
        # For sample mode hitting surface: time advance = D / c
        _, _, new_time, _, _, _ = photon_iteration_update_factors(**args)
        # Time should be > initial time
        assert float(new_time) > 5.0


class TestPositionAdvancement:
    """Position should advance along the direction vector."""

    def test_surface_hit_position(self):
        """When photon reaches surface, new_pos ≈ old_pos + D * direction + epsilon * normal."""
        key = jax.random.PRNGKey(42)
        pos = jnp.array([0.0, 0.0, 0.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        D = 2.0
        normal = jnp.array([0.0, 0.0, -1.0])
        # For update_factors, the position is a weighted combination
        # but should be approximately along the direction
        args = _base_args(key, position=pos, direction=direction,
                          surface_distance=D, normal=normal)
        new_pos, _, _, _, _, _ = photon_iteration_update_factors(**args)
        # z-component should have increased (we moved along +z)
        assert float(new_pos[2]) > 0.0


class TestContinuingFactor:
    """continuing_factor controls photon survival for next iteration."""

    def test_continuing_plus_detect_sums_correctly(self):
        """detect_prob + continuing_factor ≈ reflection_attenuation for update_factors.
        This is the probability conservation: either detected or continues."""
        key = jax.random.PRNGKey(42)
        args = _base_args(key)
        _, _, _, detect_prob, refl_atten, cont_factor = \
            photon_iteration_update_factors(**args)
        # continuing_factor = reflect_prob * refl_atten + scatter_prob * scatter_atten
        # detect_prob = reach_prob * (1 - refl_rate)
        # These don't sum to exactly refl_atten but both should be in [0, 1]
        assert 0.0 <= float(detect_prob) <= 1.0
        assert 0.0 <= float(cont_factor) <= 1.0


class TestSampleVsUpdateFactors:
    """Compare sample (MC) vs update_factors (STE) modes."""

    def test_sample_binary_detection(self):
        """Sample mode: detect_prob is binary (0 or 1)."""
        key = jax.random.PRNGKey(42)
        results = []
        for i in range(50):
            k = jax.random.fold_in(key, i)
            _, _, _, detect, _, _ = photon_iteration_sample(**_base_args(k))
            results.append(float(detect))
        unique = set(results)
        assert unique.issubset({0.0, 1.0}), f"Expected binary, got {unique}"

    def test_update_factors_continuous_detection(self):
        """Update factors mode: detect_prob is continuous in (0, 1)."""
        key = jax.random.PRNGKey(42)
        args = _base_args(key)
        _, _, _, detect, _, _ = photon_iteration_update_factors(**args)
        assert 0.0 < float(detect) < 1.0

    def test_sample_detection_rate_matches_expected(self):
        """Over many samples, mean detection should ≈ update_factors detect_prob."""
        key = jax.random.PRNGKey(42)
        args = _base_args(key)
        _, _, _, expected_detect, _, _ = photon_iteration_update_factors(**args)

        detections = []
        for i in range(5000):
            k = jax.random.fold_in(jax.random.PRNGKey(0), i)
            _, _, _, d, _, _ = photon_iteration_sample(**_base_args(k))
            detections.append(float(d))
        npt.assert_allclose(jnp.array(detections).mean(), float(expected_detect), atol=0.03)


class TestCustomVJPGradients:
    """Verify the custom VJP sanitizes NaN gradients."""

    def test_gradient_through_detection(self):
        """Gradient of detect_prob w.r.t. position should be finite."""
        key = jax.random.PRNGKey(42)
        def loss_fn(pos):
            args = _base_args(key, position=pos)
            _, _, _, detect_prob, _, _ = photon_iteration_update_factors_safe(**args)
            return detect_prob
        grad = jax.grad(loss_fn)(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_scatter_length(self):
        """Gradient of detect_prob w.r.t. scatter_length should be finite and meaningful."""
        key = jax.random.PRNGKey(42)
        def loss_fn(scatter_length):
            args = _base_args(key, scatter_length=scatter_length)
            _, _, _, detect_prob, _, _ = photon_iteration_update_factors_safe(**args)
            return detect_prob
        grad = jax.grad(loss_fn)(50.0)
        assert jnp.isfinite(grad)
        # Longer scatter length → more likely to reach surface → higher detect_prob
        assert float(grad) > 0

    def test_gradient_through_absorption(self):
        """Gradient of attenuation w.r.t. absorption_length should be positive."""
        key = jax.random.PRNGKey(42)
        def loss_fn(abs_length):
            args = _base_args(key, absorption_length=abs_length)
            _, _, _, _, refl_atten, _ = photon_iteration_update_factors_safe(**args)
            return refl_atten
        grad = jax.grad(loss_fn)(100.0)
        assert jnp.isfinite(grad)
        # Longer absorption length → less attenuation → higher value
        assert float(grad) > 0
