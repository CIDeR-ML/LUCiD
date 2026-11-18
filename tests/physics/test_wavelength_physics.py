"""Physics tests for the wavelength-dependent water model and the spectra and phase functions.

References for the absolute values pinned here:

- Symmetric (Rayleigh + symmetric Mie) scattering: SK calibration paper (Abe et al.,
  arXiv:1307.0162), Eq. 16, alpha_sym = P4/lambda^4 (1 + P5/lambda^2), with the Table 3
  values P4 = 8.51e7, P5 = 1.14e5 (nm units, 1/m) as recorded in config/materials/water.json.
- Absorption at lambda >= 465 nm: Pope & Fry (1997), Appl. Opt. 36, 8710, Table 3
  (omlc.org/spectra/water/data/pope97.txt, converted 1/cm -> 1/m), which the water JSON
  says it carries; below the 464 nm splice the SK paper's Eq. 14-15 power law.
- Cherenkov spectrum: Frank-Tamm dN/dlambda ~ 1/lambda^2.
- Henyey-Greenstein: p(mu) = (1-g^2) / (2 (1+g^2-2g mu)^{3/2}), CDF
  F(mu) = (1-g^2)/(2g) [(1+g^2-2g mu)^{-1/2} - (1+g)^{-1}], E[mu] = g, E[mu^2] = (1+2g^2)/3.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.wavelength.medium import make_medium
from lucid.wavelength.optical_model import evaluate_optical_model
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths
from lucid.wavelength.scattering import (
    hg_sample_cos_theta, hg_logpdf, rayleigh_logpdf, compute_mie_scatter_direction,
)

from tests.helpers.params import lab_dp
from tests.helpers.stats import ks_distance

SK_P4, SK_P5 = 8.51e7, 1.14e5          # SK paper Table 3 (symmetric scattering)
SK_P0, SK_P1, SK_P2, SK_P3 = 0.624, 2.96e7, 3.24e-2, 10.9   # SK paper Table 3 (absorption)
# Pope & Fry (1997) Table 3, pure-water absorption coefficient a(lambda), 1/m.
POPE_FRY_M_INV = {500.0: 0.0204, 550.0: 0.0565, 600.0: 0.2224, 650.0: 0.3400, 700.0: 0.6240}


class TestRayleighScaling:
    """SK symmetric scattering: alpha_sym = P4/lambda^4 (1 + P5/lambda^2)."""

    def test_lambda4_ratio(self):
        """alpha(350)/alpha(700) = 16 (1 + P5/350^2)/(1 + P5/700^2) = 25.06. Dropping the
        P5 term gives exactly 16; a P5/lambda term gives 16.6."""
        m = make_medium("water", wavelength_grid=jnp.array([350.0, 700.0]))
        ratio = float(m.scatter_coeff[0] / m.scatter_coeff[1])
        npt.assert_allclose(ratio, 16 * (1 + SK_P5 / 350**2) / (1 + SK_P5 / 700**2), rtol=1e-5)

    def test_absolute_length_at_400nm(self):
        """L_sym(400 nm) = 1/alpha_sym = 175.7 m from Eq. 16 with Table 3. Pins the P4
        scale, which the ratio cannot see."""
        m = make_medium("water", wavelength_grid=jnp.array([400.0]))
        expected = 1.0 / (SK_P4 / 400.0**4 * (1 + SK_P5 / 400.0**2))
        npt.assert_allclose(1.0 / float(m.scatter_coeff[0]), expected, rtol=1e-5)
        npt.assert_allclose(expected, 175.66, rtol=1e-3)

    def test_positive_and_monotonic_decrease(self):
        wl = jnp.linspace(300.0, 700.0, 50)
        m = make_medium("water", wavelength_grid=wl)
        assert jnp.all(m.scatter_coeff > 0) and jnp.all(m.mie_scatter_coeff > 0)
        assert jnp.all(jnp.diff(m.scatter_coeff) < 0)


class TestAbsorptionPhysics:
    """Water absorption: the SK fit in the blue, Pope & Fry (1997) in the red."""

    @pytest.mark.parametrize("lam", sorted(POPE_FRY_M_INV))
    def test_red_absorption_is_pope_and_fry(self, lam):
        """Above the splice alpha_abs = P0 P1/lambda^4 + a_PF(lambda). The P0 P1 term adds
        1.4 % at 500 nm and < 0.4 % from 550 nm on, so 2 % holds the curve to Pope & Fry.
        A 1/cm-vs-1/m or 10x slip, or the pre-2026-07 mistranscribed table (red knots
        10-20 % high), fails."""
        m = make_medium("water", wavelength_grid=jnp.array([lam]))
        npt.assert_allclose(float(m.absorption_coeff[0]), POPE_FRY_M_INV[lam], rtol=0.02)

    @pytest.mark.parametrize("lam", [350.0, 400.0, 420.0, 440.0])
    def test_blue_absorption_is_the_sk_power_law(self, lam):
        """Below the 464 nm blend window (452-476 nm) the curve is SK Eq. 14-15:
        alpha_abs = P0 P1/lambda^4 + P0 P2 (lambda/500)^P3, Table 3 values.
        (L_abs(400 nm) = 400 m: the SK in-situ fit is more transparent in the blue than
        Pope & Fry's laboratory water; that is the fit, not a transcription slip.)"""
        m = make_medium("water", wavelength_grid=jnp.array([lam]))
        expected = SK_P0 * SK_P1 / lam**4 + SK_P0 * SK_P2 * (lam / 500.0) ** SK_P3
        npt.assert_allclose(float(m.absorption_coeff[0]), expected, rtol=1e-4)

    def test_continuous_across_the_splice(self):
        """The power law and Pope & Fry are blended over 452-476 nm: no step larger than
        1 % between neighbours 0.25 nm apart (measured 0.79 %, set by the steep absorption
        edge itself)."""
        wl = jnp.linspace(440.0, 490.0, 201)
        a = np.asarray(make_medium("water", wavelength_grid=wl).absorption_coeff)
        assert np.max(np.abs(np.diff(a)) / a[:-1]) < 0.01

    def test_transparency_peak_in_the_blue(self):
        wl = jnp.linspace(350.0, 600.0, 251)
        m = make_medium("water", wavelength_grid=wl)
        assert jnp.all(m.absorption_coeff > 0)
        peak_wl = float(wl[jnp.argmax(1.0 / m.absorption_coeff)])
        assert 350 <= peak_wl < 500, peak_wl


class TestEffectivePropertyPhysics:
    def test_shorter_wavelength_scatters_more(self):
        """Photons at shorter wavelengths should have shorter effective scatter length."""
        dp = lab_dp()
        wl = jnp.linspace(300.0, 700.0, 100)
        m = make_medium("water", wavelength_grid=wl)
        eff_s = evaluate_optical_model(dp, jnp.array([350.0, 600.0]), m, 2).scatter_len
        assert float(eff_s[0]) < float(eff_s[1])


class TestCherenkovSpectrumPhysics:
    """Frank-Tamm: dN/dlambda ~ 1/lambda^2, so F(lambda) = (1/l_min - 1/lambda)/(1/l_min - 1/l_max)."""

    def test_analytical_mean(self):
        """E[lambda] = ln(l_max/l_min) / (1/l_min - 1/l_max) = 444.8 nm on [300, 700]
        (uniform: 500, 1/lambda: 472). The std is ~110 nm, so the 2e5 mean is good to 0.25."""
        lmin, lmax = 300.0, 700.0
        wl = sample_cherenkov_wavelengths(jax.random.PRNGKey(42), 200000, lmin, lmax)
        npt.assert_allclose(jnp.mean(wl), np.log(lmax / lmin) / (1 / lmin - 1 / lmax), atol=1.0)

    @pytest.mark.parametrize("lmin,lmax", [(300.0, 700.0), (400.0, 500.0)])
    def test_distribution_matches_frank_tamm(self, lmin, lmax):
        """In range, and a KS distance against the closed-form CDF below the 99.9 %
        critical value 1.95/sqrt(2e5) = 0.0044 (a log-uniform sampler misses by 0.06)."""
        wl = np.asarray(sample_cherenkov_wavelengths(jax.random.PRNGKey(7), 200000, lmin, lmax))
        assert wl.min() >= lmin and wl.max() <= lmax
        ks = ks_distance(wl, lambda x: (1 / lmin - 1 / x) / (1 / lmin - 1 / lmax))
        assert ks < 0.0044, ks


class TestHGPhaseFunctionPhysics:
    """Henyey-Greenstein sampling and its log-density (the DiCE score for g)."""

    @pytest.mark.parametrize("g", [0.1, 0.5, 0.8, 0.95])
    def test_moments(self, g):
        """E[mu] = g and E[mu^2] = (1 + 2g^2)/3. Midpoint-quantile u makes the sample
        means quadrature-accurate."""
        u = (jnp.arange(50000) + 0.5) / 50000
        mu = np.asarray(jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u, g), np.float64)
        npt.assert_allclose(mu.mean(), g, atol=2e-3)
        npt.assert_allclose((mu**2).mean(), (1 + 2 * g * g) / 3, atol=2e-3)

    @pytest.mark.parametrize("g", [0.3, 0.9])
    def test_sampler_matches_the_closed_form_cdf(self, g):
        u = jax.random.uniform(jax.random.PRNGKey(3), (200000,))
        mu = jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u, g)

        def F(m):
            return (1 - g * g) / (2 * g) * ((1 + g * g - 2 * g * m) ** -0.5 - 1 / (1 + g))
        assert ks_distance(mu, F) < 0.0044

    def test_near_isotropic_at_small_g(self):
        """g = 0.01: nearly isotropic, E[mu^2] = 1/3. (g is clamped to [1e-4, 1 - 1e-4],
        so negative g is not supported.)"""
        u = (jnp.arange(20000) + 0.5) / 20000
        mu = jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u, 0.01)
        npt.assert_allclose(jnp.mean(mu), 0.01, atol=2e-3)
        npt.assert_allclose(jnp.mean(mu**2), 1 / 3, atol=2e-3)

    @pytest.mark.parametrize("g", [0.2, 0.7, 0.95])
    def test_logpdf_is_normalised_and_is_the_sampled_density(self, g):
        """exp(hg_logpdf) integrates to 1 over mu in [-1, 1] and equals dF/dmu of the
        closed-form CDF, so the DiCE score is the density the sampler draws from."""
        from scipy.integrate import trapezoid
        mu = np.linspace(-1.0, 1.0, 200001)
        p = np.exp(np.asarray(hg_logpdf(jnp.asarray(mu), g), np.float64))
        npt.assert_allclose(trapezoid(p, mu), 1.0, rtol=1e-4)
        exact = (1 - g * g) / 2 * (1 + g * g - 2 * g * mu) ** -1.5
        npt.assert_allclose(p, exact, rtol=1e-4)

    def test_rayleigh_logpdf_is_normalised(self):
        from scipy.integrate import trapezoid
        mu = np.linspace(-1.0, 1.0, 20001)
        npt.assert_allclose(trapezoid(np.exp(np.asarray(rayleigh_logpdf(jnp.asarray(mu)))), mu),
                            1.0, rtol=1e-5)

    def test_differentiable_wrt_g(self):
        """d E[mu]/dg = 1 (E[mu] = g), through the reparameterised sampler."""
        u = (jnp.arange(20000) + 0.5) / 20000

        def mean_cos(g):
            return jnp.mean(jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u, g))
        npt.assert_allclose(float(jax.grad(mean_cos)(0.5)), 1.0, atol=5e-3)


class TestMieScatterDirection:
    def test_moments_about_an_oblique_direction(self):
        """compute_mie_scatter_direction scatters about the photon direction:
        E[d] = g inc and E[d d^T] = (1-m2)/2 I + (3 m2 - 1)/2 inc inc^T, m2 = (1+2g^2)/3.
        Standard errors at 2e5 keys are < 2e-3."""
        g = 0.8
        inc = np.array([1.0, 2.0, -2.0]) / 3.0
        keys = jax.random.split(jax.random.PRNGKey(0), 200_000)
        d = np.asarray(jax.jit(jax.vmap(
            lambda k: compute_mie_scatter_direction(jnp.asarray(inc), k, g)))(keys), np.float64)
        npt.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
        npt.assert_allclose(d.mean(axis=0), g * inc, atol=6e-3)
        m2 = (1 + 2 * g * g) / 3
        npt.assert_allclose(d.T @ d / len(d),
                            (1 - m2) / 2 * np.eye(3) + (3 * m2 - 1) / 2 * np.outer(inc, inc),
                            atol=6e-3)
