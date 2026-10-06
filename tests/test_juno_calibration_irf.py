"""Tests for measured PMT-family IRFs in JUNO laser calibration."""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from lucid.geometry.juno_pmt import HAMAMATSU, HIGH_QE_NNVT, NNVT
from lucid.juno_laser.irf import (
    convolve_irf_masses,
    load_type_average_irf,
    resample_irf_bank,
)


def _write_campaign(tmp_path):
    """Write a tiny campaign IRF and matching PMT table."""
    dtype = np.dtype([
        ("sn", "S8"),
        ("load", "f4"),
        ("fwhm", "f4"),
        ("t0", "f4"),
        ("dt", "f4"),
        ("mass", "f4", (9,)),
    ])
    rows = np.zeros(4, dtype=dtype)
    rows["sn"] = (b"H001", b"H001", b"H002", b"N001")
    rows["load"] = (1.0, 2.0, 1.0, 1.0)
    rows["dt"] = 0.5
    rows["mass"][0, 2:5] = (0.2, 0.6, 0.2)
    rows["mass"][1, 3:6] = (0.1, 0.8, 0.1)
    rows["mass"][2, 4:7] = (0.2, 0.6, 0.2)
    rows["mass"][3, 2:7] = (0.1, 0.2, 0.4, 0.2, 0.1)
    irf_path = tmp_path / "irf.h5"
    with h5py.File(irf_path, "w") as output:
        output["hist"] = rows
    map_path = tmp_path / "pmt.parquet"
    pd.DataFrame({
        "SN": ("H001", "H002", "N001"),
        "copy_number": (0, 1, 2),
        "pmt_type": (1, 1, 2),
    }).to_parquet(map_path, index=False)
    return irf_path, map_path


def test_type_average_irf_selects_load_centers_and_normalizes(tmp_path):
    """Campaign reduction must select, align and average reproducibly."""
    irf_path, map_path = _write_campaign(tmp_path)
    bank, metadata = load_type_average_irf(irf_path, map_path)
    kernels = np.asarray(bank.kernels)
    assert kernels.shape == (2, 9)
    np.testing.assert_allclose(np.sum(kernels, axis=1), 1.0, rtol=1e-6)
    assert np.argmax(kernels[HAMAMATSU]) == 4
    assert np.argmax(kernels[NNVT]) == 4
    assert metadata["matched_hamamatsu"] == 2
    assert metadata["matched_nnvt"] == 1
    assert metadata["native_bin_width_ns"] == 0.5


def test_irf_resampling_preserves_probability_and_center():
    """Changing numerical spacing must preserve each response mass."""
    kernels = jnp.asarray((
        (0.0, 0.1, 0.8, 0.1, 0.0),
        (0.1, 0.2, 0.4, 0.2, 0.1),
    ))
    from lucid.juno_laser.irf import IrfKernelBank

    resampled = resample_irf_bank(IrfKernelBank(kernels, 0.5), 0.25)
    values = np.asarray(resampled.kernels)
    np.testing.assert_allclose(np.sum(values, axis=1), 1.0, rtol=1e-6)
    assert all(np.argmax(row) == row.size // 2 for row in values)


def test_irf_convolution_is_family_specific_and_differentiable():
    """Each PMT uses its family kernel and gradients pass to optical mass."""
    from lucid.juno_laser.irf import IrfKernelBank

    bank = IrfKernelBank(jnp.asarray((
        (0.0, 1.0, 0.0),
        (0.25, 0.5, 0.25),
    )), 1.0)
    masses = jnp.zeros((3, 7)).at[:, 3].set((1.0, 2.0, 3.0))
    types = jnp.asarray((HAMAMATSU, NNVT, HIGH_QE_NNVT))
    result = convolve_irf_masses(masses, types, bank)
    np.testing.assert_allclose(result[0], masses[0], atol=1e-7)
    np.testing.assert_allclose(result[1, 2:5], (0.5, 1.0, 0.5))
    np.testing.assert_allclose(result[2, 2:5], (0.75, 1.5, 0.75))
    gradient = jax.grad(
        lambda value: jnp.sum(
            convolve_irf_masses(value, types, bank) ** 2
        )
    )(masses)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)
