"""Measured per-channel PMT IRF kernels from the characterization campaign.

The IRF file stores one kernel per (load, tube) in the ``hist`` dataset:
the dark-subtracted single-PE response of a characterized tube -- pulse,
fibre and transit-time spread together -- as mass in 0.5 ns bins on the
trigger-copy time axis.  Because the kernel already contains the TTS,
enabling it replaces the per-channel Gaussian TTS convolution in the
comparison pipeline.

Only the characterized tubes have a measured kernel, and the comparison
plots are restricted to exactly those channels (no population-average
inference).  A tube measured in several loads contributes its
largest-load row.  The kernel is re-centered on its peak, because the
file's time axis is relative to the trigger-copy crossing, not to the
PE time.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import compare_sim as base

DEFAULT_IRF_H5 = (
    "/junofs/PmtCharacterization/IRF/jpwaptool_ser_baseline.h5"
)


def read_irf_hist(path):
    """Read the ``hist`` dataset of an IRF file.

    Input: IRF h5 path.
    Output: numpy struct array with fields sn/load/fwhm/t0/dt/mass.
    """
    with h5py.File(path, "r") as handle:
        return handle["hist"][:]


def _resample_mass(t0, dt, mass, bin_width):
    """Rebin bin masses onto a new width, exactly when possible.

    Input: fine-grid start/width, the mass array and the target width.
    Output: mass on the target grid (starting at t0); exact group sums
    for integer width ratios, otherwise the piecewise-linear density
    integrated at the new bin centers.
    """
    ratio = bin_width / dt
    nearest = int(round(ratio))
    if nearest >= 1 and abs(ratio - nearest) < 1e-9:
        pad = (-len(mass)) % nearest
        padded = np.pad(mass, (0, pad))
        return padded.reshape(-1, nearest).sum(axis=1)
    density = mass / dt
    fine_centers = t0 + (np.arange(len(mass)) + 0.5) * dt
    count = int(np.ceil(len(mass) * dt / bin_width))
    centers = t0 + (np.arange(count) + 0.5) * bin_width
    return np.interp(centers, fine_centers, density) * bin_width


def center_on_peak(mass):
    """Roll a kernel so that its maximum sits at the center index.

    Input: mass array.  Output: odd-length array whose argmax is at
    (len - 1) // 2; even inputs gain one trailing zero bin.
    """
    values = np.asarray(mass, dtype=np.float64)
    if len(values) % 2 == 0:
        values = np.concatenate([values, [0.0]])
    peak = int(np.argmax(values))
    center = (len(values) - 1) // 2
    return np.roll(values, center - peak)


def characterized_channels(path, pmt_map_path):
    """Return the sorted copy numbers whose serial has an IRF row.

    Input: IRF h5 path and the PMT map parquet path (SN column).
    Output: sorted list of int copy numbers.
    """
    rows = read_irf_hist(path)
    serials = {row["sn"].decode() for row in rows}
    table = pd.read_parquet(pmt_map_path, columns=["SN", "copy_number"])
    channels = {
        int(copy_number)
        for sn, copy_number in zip(table["SN"], table["copy_number"])
        if pd.notna(sn)
        and str(sn) in serials
        and 0 <= copy_number < base.FULL_CHANNEL_COUNT
    }
    return sorted(channels)


def build_channel_kernels(path, pmt_map_path, bin_width):
    """Build the measured IRF kernel of every characterized channel.

    Input: IRF h5 path, PMT map parquet path (SN -> copy_number) and
    the target histogram bin width [ns].
    Output: (kernels, meta, orphans).  kernels maps copy_number to the
    odd-length unit-sum kernel centered on its peak; meta maps
    copy_number to {"sn", "load", "fwhm"}; orphans lists the IRF
    serials absent from the PMT map.  A tube measured in several loads
    contributes its largest-load row.
    """
    rows = read_irf_hist(path)
    if len(rows) == 0:
        raise ValueError("no IRF rows in the file")
    if (
        len({float(v) for v in rows["t0"]}) != 1
        or len({float(v) for v in rows["dt"]}) != 1
        or len({len(m) for m in rows["mass"]}) != 1
    ):
        raise ValueError("IRF rows must share one time grid")
    best = {}
    for row in rows:
        sn = row["sn"].decode()
        if sn not in best or row["load"] > best[sn]["load"]:
            best[sn] = row
    table = pd.read_parquet(pmt_map_path, columns=["SN", "copy_number"])
    # The PMT map also lists non-CD entries (copy_number >= 17612, some
    # sharing a serial with a CD channel); only CD channels are compared.
    channel_by_sn = {
        str(sn): int(copy_number)
        for sn, copy_number in zip(table["SN"], table["copy_number"])
        if pd.notna(sn) and 0 <= copy_number < base.FULL_CHANNEL_COUNT
    }
    kernels = {}
    meta = {}
    orphans = []
    for sn in sorted(best):
        row = best[sn]
        channel = channel_by_sn.get(sn)
        if channel is None:
            orphans.append(sn)
            continue
        kernel = center_on_peak(
            _resample_mass(
                float(row["t0"]), float(row["dt"]),
                np.asarray(row["mass"], dtype=np.float64),
                float(bin_width),
            )
        )
        kernels[channel] = kernel / kernel.sum()
        meta[channel] = {
            "sn": sn,
            "load": int(row["load"]),
            "fwhm": float(row["fwhm"]),
        }
    if not kernels:
        raise ValueError("no IRF rows match the PMT map serial numbers")
    return kernels, meta, orphans


def irf_descriptor(path, kernels):
    """Serialize the IRF identity for the aligned-cache key.

    Input: IRF file path and the kernels mapping from
    build_channel_kernels.
    Output: "file:<basename>:ch<channel count>".
    """
    return f"file:{Path(path).name}:ch{len(kernels)}"