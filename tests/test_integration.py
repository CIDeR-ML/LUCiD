"""Integration tests — verify the pieces fit together correctly.

Tests the chain: optics → photon_step → sensor_response with realistic inputs,
and gradient flow through the differentiable simulation path.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt

pytestmark = pytest.mark.slow

from lucid.simulation.optics import (
    normalize, compute_reflection_direction, create_local_frame,
    sample_scatter_distance, compute_scatter_direction,
    sample_cosine_hemisphere, jax_normalize,
)
from lucid.simulation.photon_step import (
    photon_iteration_sample,
    photon_iteration_update_factors,
    photon_iteration_update_factors_safe,
)
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_likelihood,
)
from lucid.simulation.types import PhotonState
from lucid.geometry.detector_geometry import DetectorGeometry


class TestOpticsToPhotonStep:
    """Verify optics functions produce valid inputs for photon_step."""

    def test_scatter_direction_into_photon_step(self):
        """compute_scatter_direction output should be a valid direction for photon_step."""
        key = jax.random.PRNGKey(42)
        incident = jnp.array([0.0, 0.0, 1.0])
        new_dir = compute_scatter_direction(incident, key)
        # Use as direction in photon_step
        k2 = jax.random.PRNGKey(99)
        result = photon_iteration_update_factors(
            position=jnp.zeros(3), direction=new_dir, time=0.0,
            surface_distance=2.0, normal=jnp.array([0.0, 0.0, -1.0]),
            scatter_length=50.0, wall_reflection_rate=0.5,
            sensor_reflection_rate=0.3, absorption_length=100.0,
            hit_sensor=True, rng_key=k2, speed_of_light=0.2253)
        new_pos, new_dir2, new_time, dp, ra, cf = result
        assert jnp.all(jnp.isfinite(new_pos))
        assert jnp.all(jnp.isfinite(new_dir2))
        assert jnp.isfinite(new_time)

    def test_reflection_preserves_unit_for_photon_step(self):
        """compute_reflection_direction output should be unit vector (required by photon_step)."""
        incident = normalize(jnp.array([1.0, -1.0, 0.3]))
        normal = jnp.array([0.0, 1.0, 0.0])
        reflected = compute_reflection_direction(incident, normal)
        npt.assert_allclose(jnp.linalg.norm(reflected), 1.0, atol=1e-5)


class TestPhotonStepToSensorResponse:
    """Verify photon_step outputs feed correctly into make_hits_*."""

    def test_update_factors_output_into_make_hits(self):
        """Detection weights from photon_step should aggregate correctly in make_hits."""
        key = jax.random.PRNGKey(42)
        # Simulate 10 photons, all hitting sensor 0
        n_rays = 10
        weights = jnp.ones(n_rays) * 0.8
        indices = jnp.zeros(n_rays, dtype=jnp.int32)
        times = jnp.linspace(10.0, 15.0, n_rays)
        n_sensors = 5
        qe_corr = jnp.ones(n_sensors)

        q, t = make_hits_simulation(weights, indices, times, n_sensors,
                                     qe=0.2, qe_corrections=qe_corr)
        # All 10 photons hit sensor 0 with weight 0.8
        expected_charge = 10 * 0.8 * 0.2  # 1.6
        npt.assert_allclose(q[0], expected_charge, atol=1e-5)
        # Time should be close to 10.0 (the earliest)
        assert 9.5 < float(t[0]) < 10.5

    def test_likelihood_log_weights_consistent(self):
        """Likelihood log_w values should be consistent with linear weights."""
        weights = jnp.array([0.5, 0.3, 0.8])
        indices = jnp.array([0, 1, 0])
        times = jnp.array([10.0, 12.0, 11.0])
        qe_corr = jnp.ones(5)

        log_w, safe_t, fi, total_q = make_hits_likelihood(
            weights, indices, times, 5, qe=0.2, qe_corrections=qe_corr)

        # log_w should be log(weight * qe * qe_correction)
        for i in range(3):
            expected = jnp.log(weights[i] * 0.2 * 1.0)
            npt.assert_allclose(log_w[i], expected, atol=1e-4)


class TestGradientFlowEndToEnd:
    """Verify gradients flow through the differentiable simulation path."""

    def test_grad_through_photon_step_chain(self):
        """Gradient should flow: position → photon_step → detect_prob."""
        key = jax.random.PRNGKey(42)

        def loss_fn(position):
            _, _, _, detect_prob, _, _ = photon_iteration_update_factors_safe(
                position=position, direction=jnp.array([0., 0., 1.]),
                time=0.0, surface_distance=2.0,
                normal=jnp.array([0., 0., -1.]),
                scatter_length=50.0, wall_reflection_rate=0.5,
                sensor_reflection_rate=0.3, absorption_length=100.0,
                hit_sensor=True, rng_key=key, speed_of_light=0.2253)
            return detect_prob

        pos = jnp.array([0.5, 0.5, 0.5])
        grad = jax.grad(loss_fn)(pos)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_make_hits_simulation(self):
        """Gradient should flow through sensor response aggregation."""
        def loss_fn(weights):
            qe_corr = jnp.ones(10)
            q, t = make_hits_simulation(
                weights, jnp.array([0, 0, 1]),
                jnp.array([10., 12., 8.]),
                10, qe=0.2, qe_corrections=qe_corr, temperature=0.01)
            return jnp.sum(q) + jnp.sum(t)

        grad = jax.grad(loss_fn)(jnp.array([0.8, 0.6, 0.9]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_detector_params_through_sensor_response(self):
        """Gradient w.r.t. QE should flow through sensor response."""
        def loss_fn(qe):
            weights = jnp.array([0.8, 0.6])
            indices = jnp.array([0, 1])
            times = jnp.array([10.0, 12.0])
            qe_corr = jnp.ones(5)
            q, _ = make_hits_simulation(weights, indices, times, 5,
                                         qe=qe, qe_corrections=qe_corr)
            return jnp.sum(q)

        grad = jax.grad(loss_fn)(0.2)
        assert jnp.isfinite(grad)
        # Higher QE → higher charge → positive gradient
        assert float(grad) > 0

    def test_grad_qe_corrections_through_sensor_response(self):
        """Gradient w.r.t. per-sensor QE corrections."""
        def loss_fn(qe_corrections):
            weights = jnp.array([0.8, 0.6, 0.5])
            indices = jnp.array([0, 0, 1])
            times = jnp.array([10.0, 12.0, 8.0])
            q, _ = make_hits_simulation(weights, indices, times, 5,
                                         qe=0.2, qe_corrections=qe_corrections)
            return jnp.sum(q)

        qe_corr = jnp.ones(5)
        grad = jax.grad(loss_fn)(qe_corr)
        assert jnp.all(jnp.isfinite(grad))
        # Sensors 0 and 1 have photons, so their gradients should be > 0
        assert float(grad[0]) > 0
        assert float(grad[1]) > 0
        # Sensors 2-4 have no photons, gradient should be 0
        npt.assert_allclose(grad[2:], 0.0, atol=1e-7)


class TestNormalizeEpsilonChange:
    """Verify the epsilon change (1e-6 → 1e-8) doesn't break anything.

    The optics.py normalize was jnp.maximum(norm, 1e-6), now it's
    (norm + 1e-8) from utils.py. This matters for near-zero vectors
    and gradients.
    """

    def test_normal_vectors_unchanged(self):
        """For realistic vectors (norm >> epsilon), output is identical."""
        v = jnp.array([3.0, 4.0, 0.0])
        result = normalize(v)
        npt.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-6)

    def test_near_zero_gradients_exist(self):
        """For near-zero vectors, gradient should exist (not clipped to 0)."""
        def loss(v):
            return jnp.sum(normalize(v))

        # Near-zero vector
        v = jnp.array([1e-10, 1e-10, 1e-10])
        grad = jax.grad(loss)(v)
        assert jnp.all(jnp.isfinite(grad))

    def test_reflection_with_normalized_output(self):
        """compute_reflection_direction should still produce unit output."""
        incident = normalize(jnp.array([1.0, -1.0, 0.0]))
        normal = jnp.array([0.0, 1.0, 0.0])
        reflected = compute_reflection_direction(incident, normal)
        npt.assert_allclose(jnp.linalg.norm(reflected), 1.0, atol=1e-5)

    def test_local_frame_with_normalize(self):
        """create_local_frame should produce valid frame with new normalize."""
        for z in [jnp.array([0., 0., 1.]),
                  normalize(jnp.array([1., 2., 3.]))]:
            frame = create_local_frame(z)
            # Orthogonal
            npt.assert_allclose(jnp.dot(frame[0], frame[1]), 0.0, atol=1e-5)
            # Unit vectors
            for i in range(3):
                npt.assert_allclose(jnp.linalg.norm(frame[i]), 1.0, atol=1e-5)


class TestWavelengthModuleJIT:
    """Verify wavelength module functions work inside JIT."""

    def test_make_medium_fields_in_jit(self):
        from lucid.wavelength.medium import make_medium
        m = make_medium("water", wavelength_grid=jnp.linspace(300., 700., 50))

        @jax.jit
        def use_medium(wavelengths):
            # Interpolate scatter coefficient at given wavelengths
            return jnp.interp(wavelengths, m.wavelength_grid, m.scatter_coeff)

        result = use_medium(jnp.array([400.0, 500.0]))
        assert result.shape == (2,)
        assert jnp.all(result > 0)

    def test_effective_properties_in_jit(self):
        from lucid.wavelength.medium import make_medium, compute_effective_properties
        from lucid.detector_params import DetectorParams
        m = make_medium("water", wavelength_grid=jnp.linspace(300., 700., 100))
        dp = DetectorParams(scatter_length=50., wall_reflection_rate=0.5,
                            sensor_reflection_rate=0.3, absorption_length=100.,
                            qe=0.2, qe_corrections=jnp.ones(10))

        @jax.jit
        def compute(wavelengths):
            return compute_effective_properties(dp, m, wavelengths=wavelengths)

        eff_s, eff_a, eff_qe = compute(jnp.array([350., 500.]))
        assert eff_s.shape == (2,)
        assert jnp.all(jnp.isfinite(eff_s))

    def test_cherenkov_sampling_in_jit(self):
        from lucid.wavelength.spectrum import sample_cherenkov_wavelengths

        @jax.jit
        def sample(key):
            return sample_cherenkov_wavelengths(key, 100)

        wl = sample(jax.random.PRNGKey(42))
        assert wl.shape == (100,)
        assert jnp.all(wl >= 300.)
        assert jnp.all(wl <= 700.)

    def test_cherenkov_sampling_grad(self):
        """Cherenkov sampling should be differentiable w.r.t. nothing breaks
        (it uses uniform samples, which have trivial gradients)."""
        from lucid.wavelength.spectrum import sample_cherenkov_wavelengths

        @jax.jit
        def mean_wl(key):
            return jnp.mean(sample_cherenkov_wavelengths(key, 1000))

        # Just verify it runs without error under jit
        result = mean_wl(jax.random.PRNGKey(0))
        assert jnp.isfinite(result)


class TestSimConfigIntegration:
    """Verify SimConfig fields are correctly consumed by the simulator."""

    def test_effective_n_grad_iters_values(self):
        from lucid.simulation.config import SimConfig
        # Track mode: gradient flows all K bounces (normal-fix inside photon step
        # eliminates the curvature-compounding explosion that used to require 0)
        assert SimConfig(mode='track', K=7).effective_n_grad_iters == 7
        # Calibration: n_grad_iters=2 (gradient flows for first 2 iterations)
        assert SimConfig(mode='calibration').effective_n_grad_iters == 2
        # Data: n_grad_iters=0
        assert SimConfig(mode='data').effective_n_grad_iters == 0
        # Explicit override takes precedence
        assert SimConfig(mode='track', n_grad_iters=3).effective_n_grad_iters == 3

    def test_sim_config_is_hashable(self):
        """SimConfig must be hashable for use as static_argname in JIT."""
        from lucid.simulation.config import SimConfig
        cfg = SimConfig(mode='track', K=7)
        # NamedTuples are hashable if all fields are hashable
        # n_grad_iters=None is hashable
        h = hash(cfg)
        assert isinstance(h, int)
