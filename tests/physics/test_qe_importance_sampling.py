"""Tests for wavelength_sampling='cherenkov_qe' (Method B, QE-importance sampling).

Covers:
- The pure sampler in lucid/wavelength/spectrum.py: <QE>_C against an independent integral and
  the sampled law against QE(lambda)/lambda^2 (KS).
- Setup-time errors for forbidden combinations.
- End-to-end equivalence in expectation between Method A and Method B.
- Variance reduction in expected-value mode.
- Gradients w.r.t. detector.qe and qe_corrections: Q is linear in each, so g qe = Q and g_i = Q_i.
- Explicit wavelengths override the sampler (per-call precedence rule).
"""
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from lucid.wavelength.medium import load_qe_curve
from lucid.wavelength.spectrum import (
    build_qe_weighted_cherenkov_sampler,
)

from tests.helpers.params import calib_dp
from tests.helpers.stats import ks_distance
from tests.helpers.paths import REPO


GEOM = os.path.join(REPO, 'config', 'WCTE_like_geom_config.json')
PHYSICS = os.path.join(REPO, 'config', 'SK_physics_config.json')
SK_QE_PATH = os.path.join(REPO, 'config', 'pmt', 'SK_QE.json')


# ── Pure sampler ───────────────────────────────────────────────────────────

LO, HI = 300.0, 648.22
N_KS = 400_000
KS_CRIT = 1.95 / np.sqrt(N_KS)          # 99.9 % Kolmogorov critical value, 0.0031


def _frank_tamm_cdf(x):
    return (1 / LO - 1 / x) / (1 / LO - 1 / HI)


class TestQEWeightedSampler:
    """The sampler draws lambda ~ QE(lambda)/lambda^2 and returns
    <QE>_C = int QE/lambda^2 / int 1/lambda^2."""

    def test_constant_qe_is_returned_exactly(self):
        """QE(lambda) = c gives <QE>_C = c whatever the spectrum. (A 1/lambda weight in the
        numerator gives 430 c: the ratio stops being dimensionless.)"""
        _, mean_qe = build_qe_weighted_cherenkov_sampler(
            lambda lam: 0.3 * jnp.ones_like(lam), LO, HI)
        np.testing.assert_allclose(mean_qe, 0.3, rtol=1e-6)

    def test_constant_qe_samples_follow_frank_tamm(self):
        """With a constant QE the importance sampler must reduce to the 1/lambda^2 law."""
        sample_fn, _ = build_qe_weighted_cherenkov_sampler(
            lambda lam: 0.3 * jnp.ones_like(lam), LO, HI)
        wl = sample_fn(jax.random.PRNGKey(0), N_KS)
        assert ks_distance(wl, _frank_tamm_cdf) < KS_CRIT

    def test_sk_mean_qe_matches_an_independent_integral(self):
        """<QE>_C for the SK curve against a float64 trapezoid on 2e5 points (measured
        agreement 8e-6)."""
        from scipy.integrate import trapezoid
        qe_fn = load_qe_curve(SK_QE_PATH)
        _, mean_qe = build_qe_weighted_cherenkov_sampler(qe_fn, LO, HI)
        x = np.linspace(LO, HI, 200001)
        q = np.asarray(qe_fn(jnp.asarray(x)), np.float64)
        ref = trapezoid(q / x**2, x) / trapezoid(1 / x**2, x)
        np.testing.assert_allclose(mean_qe, ref, rtol=1e-4)

    def test_sk_samples_follow_qe_times_frank_tamm(self):
        """Samples follow QE(lambda)/lambda^2 (KS against a fine-grid CDF; measured 0.0023
        at n = 4e5: the cumsum point mass above is small here because QE(300 nm) is)."""
        qe_fn = load_qe_curve(SK_QE_PATH)
        sample_fn, _ = build_qe_weighted_cherenkov_sampler(qe_fn, LO, HI)
        wl = np.asarray(sample_fn(jax.random.PRNGKey(1), N_KS))
        assert wl.min() >= LO and wl.max() <= HI
        x = np.linspace(LO, HI, 200001)
        p = np.asarray(qe_fn(jnp.asarray(x)), np.float64) / x**2
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
        cdf /= cdf[-1]
        assert ks_distance(wl, lambda v: np.interp(v, x, cdf)) < KS_CRIT

    def test_sampler_errors_when_qe_zero_everywhere(self):
        """If QE is identically zero on the sampling range, refuse to build."""
        def zero_qe(wl):
            return jnp.zeros_like(wl)
        with pytest.raises(ValueError, match="zero everywhere"):
            build_qe_weighted_cherenkov_sampler(zero_qe, 300.0, 700.0)


# ── Setup-time errors ──────────────────────────────────────────────────────

@pytest.mark.slow                 # builds simulators
class TestSetupTimeErrors:
    def test_error_when_wavelength_mode_off(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = calib_dp(len(det.all_points))
        with pytest.raises(ValueError, match="wavelength_mode=True"):
            setup_event_simulator(
                GEOM, 1000, source='light_source', temperature=None, K=2,
                default_detector_params=dp,
                wavelength_mode=False,
                wavelength_sampling='cherenkov_qe')

    def test_error_when_no_qe_curve(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = calib_dp(len(det.all_points))
        with pytest.raises(ValueError, match="requires a QE curve"):
            # physics_config not passed → no QE curve loaded.
            setup_event_simulator(
                GEOM, 1000, source='light_source', temperature=None, K=2,
                default_detector_params=dp,
                wavelength_mode=True,
                wavelength_sampling='cherenkov_qe')

    def test_error_when_photons_are_given(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = calib_dp(len(det.all_points))
        with pytest.raises(ValueError, match="incompatible with.*source=.photons."):
            setup_event_simulator(
                GEOM, 1000, source='photons', temperature=None, K=2,
                default_detector_params=dp,
                physics_config=PHYSICS,
                wavelength_mode=True,
                wavelength_sampling='cherenkov_qe')

    def test_error_on_bad_value(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = calib_dp(len(det.all_points))
        with pytest.raises(ValueError, match="cherenkov_qe"):
            setup_event_simulator(
                GEOM, 1000, source='light_source', temperature=None, K=2,
                default_detector_params=dp,
                physics_config=PHYSICS,
                wavelength_mode=True,
                wavelength_sampling='bogus')


# ── End-to-end equivalence ─────────────────────────────────────────────────

@pytest.fixture(scope='module')
def detector():
    from lucid.geometry import generate_detector
    return generate_detector(GEOM)


@pytest.fixture(scope='module')
def base_dp(detector):
    return calib_dp(len(detector.all_points))


def _build_sim(sampling, base_dp):
    from lucid.simulation import setup_event_simulator
    return setup_event_simulator(
        GEOM, 20000, source='light_source', temperature=None, K=4,
        default_detector_params=base_dp,
        physics_config=PHYSICS,
        wavelength_mode=True,
        wavelength_sampling=sampling)


@pytest.mark.slow                 # builds simulators
class TestMethodBEquivalenceExpected:
    """In expected-value mode, Method A and B should agree in the large-N
    limit within a few times the larger standard error."""

    def test_expected_value_equivalence(self, detector, base_dp):
        from lucid.sources import isotropic_source
        sim_a = _build_sim('cherenkov', base_dp)
        sim_b = _build_sim('cherenkov_qe', base_dp)

        src = isotropic_source(position=[0., 0., 0.], intensity=1.0)
        # Per-key totals; the tolerance is 4 combined standard errors of the two 6-key means
        # (measured: A and B differ by 0.11 %, the bound is ~1.6 %), capped at 3 %.
        keys = jax.random.split(jax.random.PRNGKey(0), 6)
        tot_a = np.array([float(jnp.sum(sim_a(src, k)[0])) for k in keys])
        tot_b = np.array([float(jnp.sum(sim_b(src, k)[0])) for k in keys])
        assert tot_a.mean() > 0.0
        se = np.sqrt(tot_a.var(ddof=1) / len(keys) + tot_b.var(ddof=1) / len(keys))
        diff = abs(tot_a.mean() - tot_b.mean())
        assert diff < min(4.0 * se, 0.03 * tot_a.mean()), (
            f"Method A vs B totals disagreed: {tot_a.mean()} vs {tot_b.mean()} "
            f"({diff / tot_a.mean() * 100:.2f} %, 4 s.e. = {4 * se / tot_a.mean() * 100:.2f} %)")

    def test_method_b_variance_is_lower_or_similar(self, detector, base_dp):
        """Per-sensor std across runs should not exceed Method A by more than
        a small tolerance — Method B is expected to be at least as good."""
        from lucid.sources import isotropic_source
        sim_a = _build_sim('cherenkov', base_dp)
        sim_b = _build_sim('cherenkov_qe', base_dp)

        src = isotropic_source(position=[0., 0., 0.], intensity=1.0)
        keys = jax.random.split(jax.random.PRNGKey(7), 8)
        A = jnp.stack([sim_a(src, k)[0] for k in keys])
        B = jnp.stack([sim_b(src, k)[0] for k in keys])

        # Compare std over the sensors that receive meaningful light.
        mean_a = jnp.mean(A, axis=0)
        mask = mean_a > 0.01 * float(jnp.max(mean_a))
        std_a = jnp.std(A, axis=0)[mask]
        std_b = jnp.std(B, axis=0)[mask]
        # Method B should not be materially worse than A. A 50% slack absorbs
        # MC noise on 8 keys but still catches a regression.
        assert float(jnp.median(std_b / (std_a + 1e-30))) < 1.5


@pytest.mark.slow                 # builds simulators
class TestExplicitWavelengthOverride:
    """When the caller supplies a wavelength, 'cherenkov_qe' must be ignored;
    per-photon qe_fn(λ) is used, not the scalar <QE>_C."""

    def test_scalar_wavelength_uses_curve_not_mean(self, detector, base_dp):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        # Method B set at setup — should not apply since source pins λ.
        sim = setup_event_simulator(
            GEOM, 5000, source='light_source', temperature=None, K=2,
            default_detector_params=base_dp,
            physics_config=PHYSICS,
            wavelength_mode=True,
            wavelength_sampling='cherenkov_qe')

        # QE at 375 is ~21 %, QE at 600 is ~1 %. Detected charges must track
        # that ratio — a scalar <QE>_C weight would make them equal.
        src_peak = laser_source(position=[0., 0., 0.], intensity=1e6, wavelength=375.0)
        src_red  = laser_source(position=[0., 0., 0.], intensity=1e6, wavelength=600.0)
        key = jax.random.PRNGKey(3)
        c_peak, _ = sim(src_peak, key)
        c_red, _  = sim(src_red,  key)

        ratio = float(jnp.sum(c_peak) / (jnp.sum(c_red) + 1e-30))
        assert ratio > 3.0, \
            f"peak/red ratio {ratio:.2f} — flag leaked into explicit-λ path"


@pytest.mark.slow                 # builds simulators
class TestGradients:
    """QE-importance sampling must preserve gradients w.r.t. the fittable
    QE parameters (global scalar, per-sensor corrections). QE curve shape
    is frozen at setup time (by design)."""

    def test_charge_is_linear_in_detector_qe(self, detector, base_dp):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import isotropic_source

        sim = setup_event_simulator(
            GEOM, 5000, source='light_source', temperature=None, K=2,
            physics_config=PHYSICS,
            wavelength_mode=True,
            wavelength_sampling='cherenkov_qe')
        src = isotropic_source(position=[0., 0., 0.], intensity=1.0)
        key = jax.random.PRNGKey(5)

        def loss(qe_scalar):
            dp = base_dp._replace(
                response=base_dp.response._replace(qe=qe_scalar))
            c, _ = sim(src, dp, key)
            return jnp.sum(c)

        # The expected charge is linear in detector.qe, so Euler's identity g * qe = Q holds.
        q, g = jax.value_and_grad(loss)(0.2)
        assert float(q) > 0.0
        np.testing.assert_allclose(float(g) * 0.2, float(q), rtol=1e-4)

    def test_qe_correction_gradient_is_the_sensor_charge(self, detector, base_dp):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import isotropic_source

        sim = setup_event_simulator(
            GEOM, 5000, source='light_source', temperature=None, K=2,
            physics_config=PHYSICS,
            wavelength_mode=True,
            wavelength_sampling='cherenkov_qe')
        src = isotropic_source(position=[0., 0., 0.], intensity=1.0)
        key = jax.random.PRNGKey(6)
        N = base_dp.per_pmt.qe_corrections.shape[0]

        def loss(corrections):
            dp = base_dp._replace(
                per_pmt=base_dp.per_pmt._replace(qe_corrections=corrections))
            c, _ = sim(src, dp, key)
            return jnp.sum(c)

        # Each correction multiplies its own sensor's charge only, so at corrections = 1 the
        # gradient of the total is each sensor's charge: g_i = Q_i.
        g = jax.grad(loss)(jnp.ones(N))
        q = np.asarray(sim(src, base_dp._replace(
            per_pmt=base_dp.per_pmt._replace(qe_corrections=jnp.ones(N))), key)[0])
        assert q.sum() > 0.0
        np.testing.assert_allclose(np.asarray(g), q, rtol=1e-4, atol=1e-6 * q.max())
