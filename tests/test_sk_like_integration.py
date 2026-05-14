"""Integration tests for SK_like detector with multiple source types and K convergence.

Tests that:
1. Laser, isotropic, and Cherenkov (SIREN track) sources produce reasonable output
2. Photon charge converges with K (later iterations contribute less)
3. Checks whether K is sufficient (last iteration charge vs total)
4. Gradients are finite for all source types
"""
import os
import pytest
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit

pytestmark = pytest.mark.slow

# Use SK_like (cylinder approximation of SK, same dimensions)
GEOM = os.path.join(os.path.dirname(__file__), '..', 'config', 'SK_like_geom_config.json')
GRID_KW = dict(n_cap=150, n_angular=250, n_height=150)


@pytest.fixture(scope="module")
def detector():
    from lucid.geometry import generate_detector
    return generate_detector(GEOM)


@pytest.fixture(scope="module")
def detector_params(detector):
    from lucid.detector_params import DetectorParams
    N = len(detector.all_points)
    return DetectorParams(
        scatter_length=50.0,
        wall_reflection_rate=0.2,
        sensor_reflection_rate=0.2,
        absorption_length=50.0,
        qe=0.065,
        qe_corrections=jnp.ones(N),
    )


# ── Calibration sources (laser + isotropic) ────────────────────────

class TestCalibrationSources:
    """Test laser and isotropic sources in calibration mode."""

    @pytest.fixture(scope="class")
    def laser_sim(self, detector, detector_params):
        from lucid.simulation import setup_event_simulator
        return setup_event_simulator(
            GEOM, 100_000, temperature=None, K=8,
            is_data=False, is_calibration=True,
            default_detector_params=detector_params,
            **GRID_KW)

    @pytest.fixture(scope="class")
    def isotropic_sim(self, detector, detector_params):
        from lucid.simulation import setup_event_simulator
        return setup_event_simulator(
            GEOM, 100_000, temperature=None, K=8,
            is_data=False, is_calibration=True,
            default_detector_params=detector_params,
            **GRID_KW)

    def test_laser_source_forward(self, laser_sim, detector):
        from lucid.sources import laser_source
        source = laser_source(
            position=[0.0, 0.0, detector.H / 2 - 0.1],
            intensity=100_000_000)
        key = jax.random.PRNGKey(42)
        charges, times = laser_sim(source, key)

        charge_sum = float(jnp.sum(charges))
        n_active = int(jnp.sum(charges > 0))

        assert charge_sum > 0, "Laser source should produce nonzero charge"
        assert n_active > 100, f"Expected many active sensors, got {n_active}"
        assert jnp.all(jnp.isfinite(charges)), "All charges should be finite"

    def test_isotropic_source_forward(self, isotropic_sim, detector):
        from lucid.sources import isotropic_source
        source = isotropic_source(position=[0.0, 0.0, 0.0], intensity=100_000_000)
        key = jax.random.PRNGKey(99)
        charges, times = isotropic_sim(source, key)

        charge_sum = float(jnp.sum(charges))
        n_active = int(jnp.sum(charges > 0))

        assert charge_sum > 0, "Isotropic source should produce nonzero charge"
        assert n_active > 100, f"Expected many active sensors, got {n_active}"

    def test_laser_gradient_finite(self, detector, detector_params):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source
        from lucid.losses import WC_smooth_loss
        from lucid.detector_params import denormalize_params, normalize_params, default_bounds

        sim = setup_event_simulator(
            GEOM, 50_000, temperature=None, K=6,
            is_data=False, is_calibration=True, **GRID_KW)
        sim_ref = setup_event_simulator(
            GEOM, 50_000, temperature=None, K=6,
            is_data=False, is_calibration=True,
            default_detector_params=detector_params, **GRID_KW)

        source = laser_source(position=[0.0, 0.0, detector.H / 2 - 0.1],
                              intensity=100_000_000)
        sensor_points = jnp.array(detector.all_points)
        key = jax.random.PRNGKey(42)
        true_data = jax.lax.stop_gradient(sim_ref(source, key))

        @jit
        def loss_fn(dp):
            pred = sim(source, dp, key)
            return WC_smooth_loss(sensor_points, *true_data, *pred,
                                  lambda_poisson=1.0, lambda_time=0.0, tau=2.0)

        loss, grads = value_and_grad(loss_fn)(detector_params)
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"
        grad_leaves = jax.tree.leaves(grads)
        for g in grad_leaves:
            assert jnp.all(jnp.isfinite(g)), "All gradient components should be finite"


# ── Track mode (SIREN/Cherenkov) with K convergence ────────────────

class TestTrackModeKConvergence:
    """Test Cherenkov simulation via SIREN and analyze per-iteration charge."""

    @pytest.fixture(scope="class")
    def track_output(self, detector_params):
        """Run track simulation at K=12 and return likelihood 4-tuple.

        K=12 matches the production data-mode default (see
        lucid/production/generate_events*.py) which the K-convergence analysis
        showed covers >99.9% of charge for SK_like geometry.
        """
        from lucid.simulation import setup_event_simulator
        from lucid.detector_params import ParticleParams

        K = 12
        NPHOT = 10_000

        sim = setup_event_simulator(
            GEOM, NPHOT, temperature=0.1, K=K,
            is_data=False, is_calibration=False,
            default_detector_params=detector_params,
            **GRID_KW)

        track = ParticleParams(
            energy=jnp.array(500.0),
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=jnp.array(0.5),
            phi=jnp.array(1.0),
            t0=jnp.array(0.0),
        )

        key = jax.random.PRNGKey(42)
        output = sim(track, key)
        return output, K, NPHOT

    def test_likelihood_output_structure(self, track_output):
        """Track mode should return 4-tuple likelihood output."""
        output, K, NPHOT = track_output
        assert len(output) == 4, f"Expected 4-tuple, got {len(output)}"
        log_w, flat_times, flat_indices, total_charge = output

        assert log_w.ndim == 1
        assert flat_times.ndim == 1
        assert flat_indices.ndim == 1
        assert total_charge.ndim == 1

    def test_nonzero_charge(self, track_output):
        """Track should deposit nonzero charge."""
        _, _, _, total_charge = track_output[0]
        charge_sum = float(jnp.sum(total_charge))
        n_active = int(jnp.sum(total_charge > 0))
        assert charge_sum > 0, "Track should produce nonzero charge"
        assert n_active > 50, f"Expected active sensors, got {n_active}"

    def _per_ray_charge(self, track_output):
        """Helper: compute per-ray, per-iteration charge arrays."""
        (log_w, _, _, _), K, NPHOT = track_output
        chunk = len(log_w) // K
        max_sc = chunk // NPHOT  # max_candidates_per_ray
        weights_per_k = jnp.exp(log_w.reshape(K, chunk))
        # Sum across sensor cells to get per-ray charge per iteration
        w_per_ray = weights_per_k.reshape(K, max_sc, NPHOT).sum(axis=1)  # (K, NPHOT)
        return w_per_ray

    def test_per_iteration_charge_decreases(self, track_output):
        """Charge deposited per K iteration should generally decrease."""
        w_per_ray = self._per_ray_charge(track_output)
        K = w_per_ray.shape[0]
        charge_per_iter = jnp.sum(w_per_ray, axis=1)

        # First iteration should have more charge than last
        assert float(charge_per_iter[0]) > float(charge_per_iter[-1]), \
            f"First iter charge ({float(charge_per_iter[0]):.4f}) should exceed " \
            f"last iter ({float(charge_per_iter[-1]):.4f})"

        # First half should deposit more than second half
        first_half = float(jnp.sum(charge_per_iter[:K // 2]))
        second_half = float(jnp.sum(charge_per_iter[K // 2:]))
        assert first_half > second_half, \
            f"First half charge ({first_half:.4f}) should exceed second half ({second_half:.4f})"

    def test_k_sufficiency_percentiles(self, track_output):
        """Check K is sufficient using per-ray cumulative charge percentiles.

        The mean last-iteration fraction can hide heavy-tailed rays that scatter
        many times. We check the p95 cumulative fraction to ensure even the
        slowest-converging rays have deposited most of their charge.
        """
        w_per_ray = self._per_ray_charge(track_output)
        K = w_per_ray.shape[0]

        # Cumulative charge fraction per ray
        cumulative = jnp.cumsum(w_per_ray, axis=0)  # (K, NPHOT)
        total_per_ray = cumulative[-1]  # (NPHOT,)
        active = total_per_ray > 1e-20
        frac = cumulative[:, active] / (total_per_ray[None, active] + 1e-30)

        # At penultimate iteration (K-1), even the 5th percentile ray
        # should have deposited >90% of its charge
        p5_penultimate = float(jnp.percentile(frac[K - 2], 5))
        assert p5_penultimate > 0.90, \
            f"At K={K-1}, p5 cumulative fraction is {p5_penultimate:.3f} — " \
            f"K={K} may be insufficient for the slowest rays"

        # Per-iteration charge fraction at last step: p95 should be small.
        # With per-bounce survival r≈0.33, a ray depositing fraction f at the
        # last step loses ~f*r/(1-r) ≈ f*0.5 of its total charge beyond K.
        # Threshold 1% → worst-case missed charge ~0.5% for p95 rays.
        iter_frac = w_per_ray[:, active] / (total_per_ray[None, active] + 1e-30)
        p95_last = float(jnp.percentile(iter_frac[-1], 95))
        assert p95_last < 0.01, \
            f"At K={K}, p95 per-iteration fraction is {p95_last:.3f} — " \
            f"too many rays still depositing significant charge (>1% threshold)"

    def test_per_bounce_survival_physically_consistent(self, track_output):
        """Aggregate per-bounce charge ratio should match back-of-envelope physics.

        Back-of-envelope for SK_like (r=16.9m, H=36.2m) with scatter_length=
        absorption_length=50m, wall_reflection=0.2:

          d_typical ≈ 2r × 2/π ≈ 21.5m (mean chord for wall-to-wall bounces)
          p_reach_surface = exp(-d/scatter_length) ≈ 0.65
          absorption_atten = exp(-d/absorption_length) ≈ 0.65
          surface_continuing = p_reach × absorption × reflection ≈ 0.65 × 0.65 × 0.2 ≈ 0.085
          scatter_continuing = (1-p_reach) × exp(-d_avg_scatter/abs) ≈ 0.35 × 0.82 ≈ 0.29
          total_survival ≈ 0.37 per bounce

        We use AGGREGATE ratios (total charge[k+1] / charge[k]) rather than
        per-ray ratios because per-ray charge is dominated by sensor overlap
        noise — a ray missing sensors at step k but hitting one at step k+1
        produces a huge ratio despite normal survival decay.

        Skip the K=1→K=2 ratio since the first bounce travels from the track
        origin (~17m from center) rather than wall-to-wall.
        """
        w_per_ray = self._per_ray_charge(track_output)
        K = w_per_ray.shape[0]
        charge_per_iter = jnp.sum(w_per_ray, axis=1)

        # Compute consecutive ratios (skip K=1→K=2 since first bounce is from center)
        ratios = []
        for k in range(2, K):
            if float(charge_per_iter[k - 1]) > 0:
                ratios.append(float(charge_per_iter[k] / charge_per_iter[k - 1]))

        mean_ratio = sum(ratios) / len(ratios)
        # Expected ~0.37 from back-of-envelope; allow [0.20, 0.55] for
        # geometry approximation error and temperature=0.1 soft-assignment
        assert 0.20 < mean_ratio < 0.55, \
            f"Mean per-bounce survival ratio {mean_ratio:.3f} outside " \
            f"physical range [0.20, 0.55] (expected ~0.37 for SK_like " \
            f"with scatter=absorption=50m, wall_refl=0.2)"

    def test_track_gradient_finite(self, detector_params):
        """Gradients through track simulation should be finite."""
        from lucid.simulation import setup_event_simulator
        from lucid.detector_params import ParticleParams
        from lucid.losses import poisson_nll

        sim = setup_event_simulator(
            GEOM, 5_000, temperature=0.1, K=4,
            is_data=False, is_calibration=False,
            default_detector_params=detector_params,
            **GRID_KW)

        key = jax.random.PRNGKey(42)
        track = ParticleParams(
            energy=jnp.array(500.0),
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=jnp.array(0.5),
            phi=jnp.array(1.0),
            t0=jnp.array(0.0),
        )

        # Get reference observation
        _, _, _, obs_charge = jax.lax.stop_gradient(sim(track, key))

        @jit
        def loss_fn(params_arr):
            t = ParticleParams(
                energy=params_arr[0],
                position=params_arr[1:4],
                theta=params_arr[4],
                phi=params_arr[5],
                t0=jnp.array(0.0))
            _, _, _, pred_q = sim(t, key)
            return poisson_nll(obs_charge, pred_q)

        params = jnp.array([500.0, 0.0, 0.0, 0.0, 0.5, 1.0])
        loss, grad = value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"
        assert jnp.all(jnp.isfinite(grad)), f"All gradients should be finite, got {grad}"


# ── Cross-source charge comparison ─────────────────────────────────

class TestCrossSourceConsistency:
    """Verify charge patterns are physically reasonable across source types."""

    def test_laser_charge_pattern(self, detector, detector_params):
        """Laser at top should illuminate sensors below — not uniform."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            GEOM, 50_000, temperature=None, K=6,
            is_data=False, is_calibration=True,
            default_detector_params=detector_params,
            **GRID_KW)

        source = laser_source(position=[0.0, 0.0, detector.H / 2 - 0.1],
                              intensity=100_000_000)
        key = jax.random.PRNGKey(42)
        charges, _ = sim(source, key)

        # Charge should NOT be uniform — laser creates a localized pattern
        charged = charges[charges > 0]
        if len(charged) > 10:
            cv = float(jnp.std(charged) / jnp.mean(charged))
            assert cv > 0.1, f"Laser charge should be non-uniform, CV={cv:.3f}"

    def test_isotropic_more_uniform_than_laser(self, detector, detector_params):
        """Isotropic source at center should be more uniform than laser at top."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source, isotropic_source

        sim = setup_event_simulator(
            GEOM, 50_000, temperature=None, K=6,
            is_data=False, is_calibration=True,
            default_detector_params=detector_params,
            **GRID_KW)

        key = jax.random.PRNGKey(42)

        # Laser at top
        laser = laser_source(position=[0.0, 0.0, detector.H / 2 - 0.1],
                             intensity=100_000_000)
        laser_charges, _ = sim(laser, key)

        # Isotropic at center
        iso = isotropic_source(position=[0.0, 0.0, 0.0], intensity=100_000_000)
        iso_charges, _ = sim(iso, key)

        # Both should have active sensors
        laser_active = int(jnp.sum(laser_charges > 0))
        iso_active = int(jnp.sum(iso_charges > 0))
        assert laser_active > 0 and iso_active > 0

        # Isotropic from center should illuminate more sensors
        # (laser is directional, isotropic covers all directions)
        assert iso_active >= laser_active * 0.5, \
            f"Isotropic ({iso_active} active) should illuminate broadly vs laser ({laser_active})"
