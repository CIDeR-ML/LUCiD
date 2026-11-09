"""Integration tests for SK_like detector with multiple source types and K convergence.

Tests that:
1. Laser, isotropic, and Cherenkov (SIREN track) sources produce reasonable output
2. Photon charge converges with K (later iterations contribute less)
3. Checks whether K is sufficient (last iteration charge vs total)
4. Gradients equal central finite differences (laser: qe, wall reflection; track: energy, vertex)
"""
import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad, jit

from tests.helpers.params import central_difference, leaf, with_leaf
from tests.helpers.paths import REPO

pytestmark = pytest.mark.slow

# Use SK_like (cylinder approximation of SK, same dimensions)
GEOM = os.path.join(REPO, 'config', 'SK_like_geom_config.json')
GRID_KW = dict(n_cap=150, n_angular=250, n_height=150)


@pytest.fixture(scope="module")
def detector():
    from lucid.geometry import generate_detector
    return generate_detector(GEOM)


@pytest.fixture(scope="module")
def detector_params(detector):
    from lucid.detector_params import DetectorParams
    N = len(detector.all_points)
    return DetectorParams.from_flat(
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
            GEOM, 100_000, source='light_source', temperature=None, K=8,
            default_detector_params=detector_params,
            **GRID_KW)

    @pytest.fixture(scope="class")
    def isotropic_sim(self, detector, detector_params):
        from lucid.simulation import setup_event_simulator
        return setup_event_simulator(
            GEOM, 100_000, source='light_source', temperature=None, K=8,
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

    def test_laser_gradient_matches_central_differences(self, detector, detector_params):
        """On the full SK_like detector, d loss / d qe and d loss / d wall_reflection_rate equal
        central finite differences of the same jitted loss at a fixed key, and point back to the
        truth.

        The data are the forward at ``detector_params`` (qe 0.065, wall 0.2) with the same key;
        the gradient is taken at qe 0.08 and wall 0.3, where it is not zero. Tolerance 0.5 %.
        """
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source
        from lucid.losses import WC_smooth_loss

        sim = setup_event_simulator(
            GEOM, 50_000, source='light_source', temperature=None, K=6,
            **GRID_KW)
        sim_ref = setup_event_simulator(
            GEOM, 50_000, source='light_source', temperature=None, K=6,
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

        start = with_leaf(with_leaf(detector_params, 'qe', 0.08), 'wall_reflection_rate', 0.3)
        loss, grads = value_and_grad(loss_fn)(start)
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"
        for field, truth in (('qe', 0.065), ('wall_reflection_rate', 0.2)):
            fd = central_difference(loss_fn, start, field)
            ad = leaf(grads, field)
            assert abs(ad - fd) <= 5e-3 * abs(fd), (field, ad, fd)
            assert ad * (leaf(start, field) - truth) > 0, \
                f'{field}: the gradient must point back to the truth'


# ── Track mode (SIREN/Cherenkov) with K convergence ────────────────

@pytest.mark.requires_emitter('muon', 'water')
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
            GEOM, NPHOT, source='particle', temperature=0.1, K=K,
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

    def test_likelihood_output_is_a_per_photon_decomposition_of_the_total(self, track_output,
                                                                          detector):
        """``(log_w, flat_times, flat_indices, total_charge)``: three aligned per-photon arrays
        and a per-sensor total that is exactly their sum, ``total_charge[s] = sum over photons
        with flat_indices == s of exp(log_w)``.

        The time likelihood reads the per-photon arrays and the charge likelihood reads the
        total, so the two must describe the same light: a total built from other weights (QE
        dropped, the wrong sensor axis), or indices outside the detector, fails here. A photon
        with no weight carries the -1e10 sentinel and time 0.
        """
        output, K, NPHOT = track_output
        assert len(output) == 4, f"Expected 4-tuple, got {len(output)}"
        log_w, flat_times, flat_indices, total_charge = (np.asarray(a) for a in output)
        ns = len(detector.all_points)

        assert total_charge.shape == (ns,)
        assert log_w.shape == flat_times.shape == flat_indices.shape
        assert log_w.shape[0] % (K * NPHOT) == 0           # K iterations x candidates x rays
        assert flat_indices.min() >= 0 and flat_indices.max() < ns

        valid = log_w > -1e9
        assert valid.any()
        np.testing.assert_array_equal(flat_times[~valid], 0.0)
        assert (flat_times[valid] > 0).all()
        w = np.where(valid, np.exp(log_w.astype(np.float64)), 0.0)
        per_sensor = np.bincount(flat_indices, weights=w, minlength=ns)
        np.testing.assert_allclose(per_sensor, total_charge, rtol=2e-5,
                                   atol=1e-6 * total_charge.max())

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

    # (parameter index in [E, x, y, z, theta, phi], displacement from the truth)
    _RESTORING = [(1, 0.5), (1, -0.5), (2, 0.5), (2, -0.5), (3, 0.5), (3, -0.5),
                  (4, 0.05), (4, -0.05), (5, 0.05), (5, -0.05)]

    def test_track_gradient_points_back_to_the_truth(self, detector_params):
        """Displace one track parameter from the truth (vertex by 0.5 m either way, each angle by
        0.05 rad either way): its gradient component has the sign that moves it back.

        The data are the forward at the truth with the same key, so the Poisson NLL is at its
        minimum there and each one-parameter displacement must see a restoring gradient, in its
        mean over four keys. This is a sign check, not a finite-difference one: at 5000 photons
        and a fixed key the forward is piecewise in the track (photons change sensor), and central
        differences miss the autodiff gradient by 30 % to 10x at every step tried. Since the SIREN
        decode stopped adding a uniform light floor (C3A2), one key's polar-angle component takes
        the wrong sign in about 1 of 4 keys (measured 3/8 over keys 42, 0, 1, 2 at 5000 and 20000
        photons; the 4-key mean restores at both). The energy is left out of
        the sign check (through this charge-only loss its gradient is dominated by Monte-Carlo
        noise; measured: the wrong sign at -50 MeV) but every component, energy included, must be
        finite. Measured smallest restoring component: 2.2 per metre.
        """
        from lucid.simulation import setup_event_simulator
        from lucid.detector_params import ParticleParams
        from lucid.losses import poisson_nll

        sim = setup_event_simulator(
            GEOM, 5_000, source='particle', temperature=0.1, K=4,
            default_detector_params=detector_params,
            **GRID_KW)

        keys = [jax.random.PRNGKey(k) for k in (42, 0, 1, 2)]

        def track(p):
            return ParticleParams(energy=p[0], position=p[1:4], theta=p[4], phi=p[5],
                                  t0=jnp.array(0.0))

        truth = jnp.array([500.0, 0.0, 0.0, 0.0, 0.5, 1.0])
        observed = [jax.lax.stop_gradient(sim(track(truth), key))[3] for key in keys]

        @jit
        def loss_fn(params_arr, obs_charge, key):
            _, _, _, pred_q = sim(track(params_arr), key)
            return poisson_nll(obs_charge, pred_q)

        grad_fn = jit(jax.grad(loss_fn))
        wrong = []
        for i, d in self._RESTORING:
            full = sum(grad_fn(truth.at[i].add(d), obs, key) for obs, key in zip(observed, keys))
            assert jnp.all(jnp.isfinite(full)), f'non-finite gradient (energy included) at {i}+{d}: {full}'
            g = float(full[i])
            if not g * d > 0:
                wrong.append((i, d, g))
        assert not wrong, f'(parameter, displacement, gradient) not restoring: {wrong}'


# ── Cross-source charge comparison ─────────────────────────────────

class TestCrossSourceConsistency:
    """Verify charge patterns are physically reasonable across source types."""

    def test_laser_charge_pattern(self, detector, detector_params):
        """Laser at top should illuminate sensors below — not uniform."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            GEOM, 50_000, source='light_source', temperature=None, K=6,
            default_detector_params=detector_params,
            **GRID_KW)

        source = laser_source(position=[0.0, 0.0, detector.H / 2 - 0.1],
                              intensity=100_000_000)
        key = jax.random.PRNGKey(42)
        charges, _ = sim(source, key)

        # Charge should NOT be uniform — laser creates a localized pattern
        charged = charges[charges > 0]
        assert len(charged) > 10, \
            f"a 1e8-intensity laser must light more than 10 PMTs, lit {len(charged)}"
        cv = float(jnp.std(charged) / jnp.mean(charged))
        assert cv > 0.1, f"Laser charge should be non-uniform, CV={cv:.3f}"

    def test_isotropic_more_uniform_than_laser(self, detector, detector_params):
        """Isotropic source at center should be more uniform than laser at top."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source, isotropic_source

        sim = setup_event_simulator(
            GEOM, 50_000, source='light_source', temperature=None, K=6,
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
