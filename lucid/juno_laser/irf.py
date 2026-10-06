"""Measured PMT impulse-response kernels for JUNO laser calibration."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from lucid.geometry.juno_pmt import HAMAMATSU, HIGH_QE_NNVT, NNVT


class IrfKernelBank(NamedTuple):
    """Discrete centered IRF probability masses for two PMT families."""

    kernels: jnp.ndarray
    bin_width_ns: float


def _decode_serial(value) -> str:
    """Return an HDF5 serial-number field as a stripped string."""
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


def _center_waveform(mass, bin_width_ns, target_centers_ns):
    """Interpolate one waveform onto a peak-centered common time axis."""
    values = np.asarray(mass, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("IRF mass must be a finite one-dimensional array")
    if np.any(values < 0.0) or not np.any(values > 0.0):
        raise ValueError("IRF mass must be non-negative with positive sum")
    peak = int(np.argmax(values))
    relative_time = (
        np.arange(values.size, dtype=np.float64) - peak
    ) * bin_width_ns
    centered = np.interp(
        target_centers_ns, relative_time, values, left=0.0, right=0.0
    )
    return centered / np.sum(centered)


def load_type_average_irf(irf_path, pmt_map_path):
    """Load peak-centered, highest-load mean IRFs for two PMT families.

    The current campaign file covers only a subset of channels.  We select
    the highest-load record for each serial number, match it to the JUNO PMT
    table, center each waveform on its maximum, normalize it, and then take
    an equal-weight mean within the Hamamatsu and NNVT families.
    """
    with h5py.File(Path(irf_path), "r") as source:
        rows = source["hist"][:]
    if not len(rows):
        raise ValueError("IRF table is empty")
    best = {}
    for row in rows:
        serial = _decode_serial(row["sn"])
        if serial not in best or row["load"] > best[serial]["load"]:
            best[serial] = row
    widths = {float(row["dt"]) for row in best.values()}
    lengths = {len(row["mass"]) for row in best.values()}
    if len(widths) != 1 or len(lengths) != 1:
        raise ValueError("IRF records must share one sampling grid")
    native_width = widths.pop()
    native_length = lengths.pop()
    if native_width <= 0.0:
        raise ValueError("IRF bin width must be positive")

    pmt_table = pd.read_parquet(
        pmt_map_path, columns=["SN", "copy_number", "pmt_type"]
    )
    pmt_table = pmt_table[
        (pmt_table["copy_number"] >= 0)
        & (pmt_table["copy_number"] < 17612)
    ]
    type_by_serial = {
        str(serial).strip(): int(pmt_type)
        for serial, pmt_type in zip(
            pmt_table["SN"], pmt_table["pmt_type"]
        )
        if pd.notna(serial)
    }
    half = (native_length - 1) // 2
    centers = np.arange(-half, half + 1, dtype=np.float64) * native_width
    grouped = {HAMAMATSU: [], NNVT: []}
    external_to_internal = {1: HAMAMATSU, 2: NNVT}
    for serial, row in best.items():
        family = external_to_internal.get(type_by_serial.get(serial))
        if family is None:
            continue
        grouped[family].append(
            _center_waveform(row["mass"], native_width, centers)
        )
    if not all(grouped.values()):
        raise ValueError("IRF data do not contain both PMT families")
    kernels = np.stack(
        [np.mean(grouped[family], axis=0) for family in (HAMAMATSU, NNVT)]
    )
    kernels /= np.sum(kernels, axis=1, keepdims=True)
    metadata = {
        "irf_path": str(Path(irf_path).resolve()),
        "pmt_map_path": str(Path(pmt_map_path).resolve()),
        "selection": "highest load per serial; peak-centered; family mean",
        "native_bin_width_ns": native_width,
        "matched_hamamatsu": len(grouped[HAMAMATSU]),
        "matched_nnvt": len(grouped[NNVT]),
    }
    return IrfKernelBank(jnp.asarray(kernels), native_width), metadata


def resample_irf_bank(bank: IrfKernelBank, bin_width_ns: float):
    """Resample centered IRF masses to a numerical time-grid spacing."""
    if bin_width_ns <= 0.0:
        raise ValueError("target IRF bin width must be positive")
    kernels = np.asarray(bank.kernels, dtype=np.float64)
    source_half = (kernels.shape[1] - 1) // 2
    source_centers = (
        np.arange(-source_half, source_half + 1) * bank.bin_width_ns
    )
    extent = float(np.max(np.abs(source_centers)))
    target_half = max(int(np.floor(extent / bin_width_ns)), 1)
    target_centers = (
        np.arange(-target_half, target_half + 1) * bin_width_ns
    )
    densities = kernels / bank.bin_width_ns
    result = np.stack([
        np.interp(
            target_centers,
            source_centers,
            density,
            left=0.0,
            right=0.0,
        ) * bin_width_ns
        for density in densities
    ])
    result /= np.sum(result, axis=1, keepdims=True)
    return IrfKernelBank(jnp.asarray(result, dtype=jnp.float32), bin_width_ns)


def convolve_irf_masses(bin_masses, pmt_type_codes, bank: IrfKernelBank):
    """Convolve PMT-by-time masses with differentiable family IRFs."""
    masses = jnp.asarray(bin_masses)
    types = jnp.asarray(pmt_type_codes, dtype=jnp.int32)
    kernels = jnp.asarray(bank.kernels, dtype=masses.dtype)
    if masses.ndim != 2 or kernels.ndim != 2 or kernels.shape[0] != 2:
        raise ValueError("masses and two-family IRF kernels must be 2D")
    if types.shape != (masses.shape[0],):
        raise ValueError("PMT type codes must match the sensor axis")
    family = jnp.where(types == HAMAMATSU, HAMAMATSU, NNVT)
    selected = kernels[family]
    full_size = masses.shape[1] + kernels.shape[1] - 1
    fft_size = 1 << (full_size - 1).bit_length()
    transformed_mass = jnp.fft.rfft(masses, n=fft_size, axis=1)
    transformed_irf = jnp.fft.rfft(selected, n=fft_size, axis=1)
    full = jnp.fft.irfft(
        transformed_mass * transformed_irf, n=fft_size, axis=1
    )[:, :full_size]
    start = (kernels.shape[1] - 1) // 2
    # Real FFT round-off can produce tiny negative values around exact zeros.
    # Probability masses must remain non-negative for PPP sampling and rates.
    return jnp.maximum(
        full[:, start:start + masses.shape[1]],
        jnp.asarray(0.0, dtype=masses.dtype),
    )


def make_irf_response_forward(ensemble_forward, pmt_type_codes, bank):
    """Wrap an optical ensemble with a measured PMT IRF response."""
    host_types = np.asarray(pmt_type_codes, dtype=np.int32)
    supported = np.isin(host_types, (HAMAMATSU, NNVT, HIGH_QE_NNVT))
    if host_types.ndim != 1 or not np.all(supported):
        raise ValueError("IRF response found unsupported PMT type codes")
    types = jnp.asarray(host_types, dtype=jnp.int32)

    def response(theta, keys):
        predictions = ensemble_forward(theta, keys)
        return jax.lax.map(
            lambda prediction: convolve_irf_masses(
                prediction, types, bank
            ),
            predictions,
        )

    return response
