"""Tests for the per-photon optical model seam (lucid.wavelength.optical_model).

These lock the two regimes of ``evaluate_optical_model`` and the exact lengths it
must reproduce so the simulator's ``_get_optical_arrays`` stays byte-identical to the
pre-extraction inline code.
"""
import os

import jax.numpy as jnp
import numpy.testing as npt

from lucid.detector_params import DetectorParams
from lucid.wavelength.medium import make_medium, load_qe_curve
from lucid.wavelength.optical_model import evaluate_optical_model, OpticalArrays

_SK_QE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'pmt', 'SK_QE.json')


def _dp(ns=10):
    return DetectorParams.from_flat(
        scatter_length=50.0, mie_scatter_length=1000.0, g=0.9,
        wall_reflection_rate=0.5, sensor_reflection_rate=0.3,
        absorption_length=100.0, qe=0.2, qe_corrections=jnp.ones(ns),
    )


class TestScalarMode:
    def test_broadcasts_scalars(self):
        """wavelengths=None → DetectorParams scalars broadcast to (n,), qe None."""
        dp = _dp()
        m = make_medium("water")
        n = 7
        oa = evaluate_optical_model(dp, None, m, n)
        assert isinstance(oa, OpticalArrays)
        assert oa.scatter_len.shape == (n,)
        assert oa.mie_len.shape == (n,)
        assert oa.abs_len.shape == (n,)
        npt.assert_allclose(oa.scatter_len, 50.0)
        npt.assert_allclose(oa.mie_len, 1000.0)
        npt.assert_allclose(oa.abs_len, 100.0)
        assert oa.qe is None


class TestWavelengthMode:
    def test_lengths_are_inverse_medium_coeff(self):
        """length(λ) = 1 / coeff(λ); matches the medium reference exactly."""
        dp = _dp()
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        wavelengths = jnp.array([350.0, 400.0, 500.0, 600.0])
        oa = evaluate_optical_model(dp, wavelengths, m, wavelengths.shape[0])

        sc = jnp.interp(wavelengths, m.wavelength_grid, m.scatter_coeff)
        ac = jnp.interp(wavelengths, m.wavelength_grid, m.absorption_coeff)
        asym = jnp.interp(wavelengths, m.wavelength_grid, m.mie_scatter_coeff)
        npt.assert_allclose(oa.scatter_len, 1.0 / (sc + 1e-30), rtol=1e-6)
        npt.assert_allclose(oa.abs_len, 1.0 / (ac + 1e-30), rtol=1e-6)
        npt.assert_allclose(oa.mie_len, 1.0 / (asym + 1e-30), rtol=1e-6)
        assert oa.qe is None

    def test_clamps_to_grid(self):
        """λ outside the grid is clamped to the grid endpoints (no extrapolation)."""
        dp = _dp()
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        oa_lo = evaluate_optical_model(dp, jnp.array([100.0]), m, 1)
        oa_edge = evaluate_optical_model(dp, jnp.array([300.0]), m, 1)
        npt.assert_allclose(oa_lo.scatter_len, oa_edge.scatter_len, rtol=1e-6)

    def test_qe_fn_applied_per_photon(self):
        """A qe_fn is evaluated at the (clamped) per-photon wavelengths."""
        dp = _dp()
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        qe_fn = load_qe_curve(_SK_QE_PATH)
        wavelengths = jnp.array([350.0, 400.0, 500.0])
        oa = evaluate_optical_model(dp, wavelengths, m, 3, qe_fn=qe_fn)
        assert oa.qe is not None
        wl_clamped = jnp.clip(wavelengths, m.wavelength_grid[0], m.wavelength_grid[-1])
        npt.assert_allclose(oa.qe, qe_fn(wl_clamped), rtol=1e-6)
