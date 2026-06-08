"""Tests for the Spectrum abstraction (Monochromatic / PowerLaw / QEWeighted)."""
import os

import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths, Monochromatic, PowerLaw, QEWeighted,
)
from lucid.wavelength.medium import load_qe_curve

_SK_QE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'pmt', 'SK_QE.json')
_KEY = jax.random.PRNGKey(0)


class TestMonochromatic:
    def test_constant_wavelength(self):
        s = Monochromatic(405.0)
        lam = s.sample(_KEY, 1000)
        npt.assert_allclose(lam, 405.0)
        assert s.mean_qe is None


class TestPowerLaw:
    def test_p2_byte_identical_to_cherenkov(self):
        """PowerLaw(2) must reproduce sample_cherenkov_wavelengths exactly."""
        s = PowerLaw(2.0)
        lam_s = s.sample(_KEY, 5000, 300.0, 700.0)
        lam_ref = sample_cherenkov_wavelengths(_KEY, 5000, 300.0, 700.0)
        npt.assert_array_equal(lam_s, lam_ref)
        assert s.mean_qe is None

    def test_in_range(self):
        for p in (1.5, 2.0, 4.0):
            lam = PowerLaw(p).sample(_KEY, 10000, 350.0, 500.0)
            assert float(lam.min()) >= 350.0 - 1e-3
            assert float(lam.max()) <= 500.0 + 1e-3

    def test_steeper_power_favours_short_wavelengths(self):
        m2 = float(jnp.mean(PowerLaw(2.0).sample(_KEY, 20000, 300.0, 700.0)))
        m4 = float(jnp.mean(PowerLaw(4.0).sample(_KEY, 20000, 300.0, 700.0)))
        assert m4 < m2   # steeper 1/λ^4 pulls the mean toward shorter λ


class TestQEWeighted:
    def test_mean_qe_and_sampling_in_band(self):
        qe_fn = load_qe_curve(_SK_QE_PATH)
        s = QEWeighted(qe_fn, 350.0, 600.0)
        assert s.mean_qe is not None and 0.0 < s.mean_qe < 1.0
        lam = s.sample(_KEY, 10000)
        assert float(lam.min()) >= 350.0 - 1e-3
        assert float(lam.max()) <= 600.0 + 1e-3
