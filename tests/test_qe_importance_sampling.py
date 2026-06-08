"""Tests for wavelength_sampling='cherenkov_qe' (Method B, QE-importance sampling).

Covers:
- The pure sampler in lucid/wavelength/spectrum.py: shape/support/normalization.
- Setup-time errors for forbidden combinations.
- End-to-end equivalence in expectation between Method A and Method B.
- Variance reduction in expected-value mode.
- Gradients w.r.t. detector.qe and qe_corrections remain finite and sensible.
- Explicit wavelengths override the sampler (per-call precedence rule).
"""
import os

import pytest
import jax
import jax.numpy as jnp

from lucid.wavelength.medium import load_qe_curve
from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths,
    build_qe_weighted_cherenkov_sampler,
)

pytestmark = pytest.mark.slow

GEOM = os.path.join(os.path.dirname(__file__), '..', 'config', 'WCTE_like_geom_config.json')
PHYSICS = os.path.join(os.path.dirname(__file__), '..', 'config', 'SK_physics_config.json')
SK_QE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'pmt', 'SK_QE.json')


# ── Pure sampler ───────────────────────────────────────────────────────────

class TestQEWeightedSampler:
    def test_sampler_support_and_shape(self):
        qe_fn = load_qe_curve(SK_QE_PATH)
        sample_fn, mean_qe = build_qe_weighted_cherenkov_sampler(
            qe_fn, 300.0, 648.22)
        key = jax.random.PRNGKey(0)
        wl = sample_fn(key, 5000)
        assert wl.shape == (5000,)
        assert jnp.all(wl >= 300.0)
        assert jnp.all(wl <= 648.22)
        # <QE>_C for SK over [300, 648] should land in a reasonable range.
        assert 0.03 < mean_qe < 0.15

    def test_sampler_biases_toward_qe_peak(self):
        """Median under QE-weighted dist should be near the QE/λ² peak (~360 nm),
        not the Cherenkov median (~413 nm over [300, 700])."""
        qe_fn = load_qe_curve(SK_QE_PATH)
        sample_fn, _ = build_qe_weighted_cherenkov_sampler(
            qe_fn, 300.0, 648.22)
        key = jax.random.PRNGKey(1)
        wl = sample_fn(key, 50000)
        med = float(jnp.median(wl))
        # QE-peak · 1/λ² peak sits around 360 nm for SK; give a generous window.
        assert 330.0 < med < 410.0

    def test_sampler_errors_when_qe_zero_everywhere(self):
        """If QE is identically zero on the sampling range, refuse to build."""
        def zero_qe(wl):
            return jnp.zeros_like(wl)
        with pytest.raises(ValueError, match="zero everywhere"):
            build_qe_weighted_cherenkov_sampler(zero_qe, 300.0, 700.0)


# ── Setup-time errors ──────────────────────────────────────────────────────

class TestSetupTimeErrors:
    def _make_dp(self, n_sensors):
        from lucid.detector_params import DetectorParams
        return DetectorParams.from_flat(
            scatter_length=50., wall_reflection_rate=0.2,
            sensor_reflection_rate=0.2, absorption_length=50.,
            qe=0.2, qe_corrections=jnp.ones(n_sensors))

    def test_error_when_wavelength_mode_off(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = self._make_dp(len(det.all_points))
        with pytest.raises(ValueError, match="wavelength_mode=True"):
            setup_event_simulator(
                GEOM, 1000, temperature=None, K=2,
                is_calibration=True, default_detector_params=dp,
                wavelength_mode=False,
                wavelength_sampling='cherenkov_qe')

    def test_error_when_no_qe_curve(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = self._make_dp(len(det.all_points))
        with pytest.raises(ValueError, match="requires a QE curve"):
            # physics_config not passed → no QE curve loaded.
            setup_event_simulator(
                GEOM, 1000, temperature=None, K=2,
                is_calibration=True, default_detector_params=dp,
                wavelength_mode=True,
                wavelength_sampling='cherenkov_qe')

    def test_error_when_is_data(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = self._make_dp(len(det.all_points))
        with pytest.raises(ValueError, match="incompatible with.*is_data"):
            setup_event_simulator(
                GEOM, 1000, temperature=None, K=2,
                is_data=True, default_detector_params=dp,
                physics_config=PHYSICS,
                wavelength_mode=True,
                wavelength_sampling='cherenkov_qe')

    def test_error_on_bad_value(self):
        from lucid.simulation import setup_event_simulator
        from lucid.geometry import generate_detector
        det = generate_detector(GEOM)
        dp = self._make_dp(len(det.all_points))
        with pytest.raises(ValueError, match="cherenkov_qe"):
            setup_event_simulator(
                GEOM, 1000, temperature=None, K=2,
                is_calibration=True, default_detector_params=dp,
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
    from lucid.detector_params import DetectorParams
    N = len(detector.all_points)
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2,
        sensor_reflection_rate=0.2, absorption_length=50.0,
        qe=0.2, qe_corrections=jnp.ones(N))


def _build_sim(sampling, base_dp, use_expected_value):
    from lucid.simulation import setup_event_simulator
    return setup_event_simulator(
        GEOM, 20000, temperature=None, K=4,
        is_calibration=True, default_detector_params=base_dp,
        physics_config=PHYSICS,
        wavelength_mode=True,
        use_expected_value=use_expected_value,
        wavelength_sampling=sampling)


class TestMethodBEquivalenceExpected:
    """In expected-value mode, Method A and B should agree in the large-N
    limit within a few times the larger standard error."""

    def test_expected_value_equivalence(self, detector, base_dp):
        from lucid.sources import isotropic_source
        sim_a = _build_sim('cherenkov', base_dp, use_expected_value=True)
        sim_b = _build_sim('cherenkov_qe', base_dp, use_expected_value=True)

        src = isotropic_source(position=[0., 0., 0.], intensity=1.0)
        # Average over several keys to beat down MC noise in both estimators.
        keys = jax.random.split(jax.random.PRNGKey(0), 6)
        cA = jnp.mean(jnp.stack([sim_a(src, k)[0] for k in keys]), axis=0)
        cB = jnp.mean(jnp.stack([sim_b(src, k)[0] for k in keys]), axis=0)

        # Total expected charge should match to a few % (MC noise).
        tot_a = float(jnp.sum(cA))
        tot_b = float(jnp.sum(cB))
        assert tot_a > 0.0
        rel = abs(tot_a - tot_b) / max(tot_a, tot_b)
        assert rel < 0.08, f"Method A vs B totals disagreed: {tot_a} vs {tot_b} ({rel*100:.1f}%)"

    def test_method_b_variance_is_lower_or_similar(self, detector, base_dp):
        """Per-sensor std across runs should not exceed Method A by more than
        a small tolerance — Method B is expected to be at least as good."""
        from lucid.sources import isotropic_source
        sim_a = _build_sim('cherenkov', base_dp, use_expected_value=True)
        sim_b = _build_sim('cherenkov_qe', base_dp, use_expected_value=True)

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


class TestExplicitWavelengthOverride:
    """When the caller supplies a wavelength, 'cherenkov_qe' must be ignored;
    per-photon qe_fn(λ) is used, not the scalar <QE>_C."""

    def test_scalar_wavelength_uses_curve_not_mean(self, detector, base_dp):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        # Method B set at setup — should not apply since source pins λ.
        sim = setup_event_simulator(
            GEOM, 5000, temperature=None, K=2,
            is_calibration=True, default_detector_params=base_dp,
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


class TestGradients:
    """QE-importance sampling must preserve gradients w.r.t. the fittable
    QE parameters (global scalar, per-sensor corrections). QE curve shape
    is frozen at setup time (by design)."""

    def test_grad_detector_qe_finite(self, detector, base_dp):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import isotropic_source
        from lucid.detector_params import DetectorParams

        sim = setup_event_simulator(
            GEOM, 5000, temperature=None, K=2,
            is_calibration=True, physics_config=PHYSICS,
            wavelength_mode=True,
            wavelength_sampling='cherenkov_qe')
        src = isotropic_source(position=[0., 0., 0.], intensity=1.0)
        key = jax.random.PRNGKey(5)

        def loss(qe_scalar):
            dp = base_dp._replace(
                response=base_dp.response._replace(qe=qe_scalar))
            c, _ = sim(src, dp, key)
            return jnp.sum(c)

        g = jax.grad(loss)(0.2)
        assert jnp.isfinite(g)
        # Higher detector.qe should give more charge.
        assert float(g) > 0.0

    def test_grad_qe_corrections_finite(self, detector, base_dp):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import isotropic_source
        from lucid.detector_params import DetectorParams

        sim = setup_event_simulator(
            GEOM, 5000, temperature=None, K=2,
            is_calibration=True, physics_config=PHYSICS,
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

        g = jax.grad(loss)(jnp.ones(N))
        assert jnp.all(jnp.isfinite(g))
        # Every correction is multiplicative on its sensor's charge, so all
        # entries should be non-negative (most strictly positive for hit sensors).
        assert float(jnp.min(g)) >= 0.0
