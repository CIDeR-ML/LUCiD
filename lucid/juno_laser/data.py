"""Input handling for JUNO laser per-PE calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class PerPeObservation:
    """Binned PE observations for one source position."""

    counts: np.ndarray
    n_events: int
    n_pe: int
    time_min_ns: float
    time_max_ns: float
    bin_width_ns: float

    @property
    def n_sensors(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[1])


def event_files(data_dir: str | Path, max_events: int = 0) -> list[Path]:
    """Return sorted event NPZ files, excluding trajectory-only files."""
    root = Path(data_dir).expanduser()
    files = sorted(path for path in root.glob("*.npz")
                   if "traj" not in path.name.lower())
    if max_events > 0:
        files = files[:max_events]
    if not files:
        raise FileNotFoundError(f"No event NPZ files found in {root}")
    return files


def _event_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = {name for name in ("hit_pmt_ids", "pe_hit_times") if name not in data.files}
        if missing:
            raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
        ids = np.asarray(data["hit_pmt_ids"], dtype=np.int64)
        times = np.asarray(data["pe_hit_times"], dtype=np.float64)
    if ids.ndim != 1 or times.ndim != 1 or ids.shape != times.shape:
        raise ValueError(f"{path}: hit_pmt_ids and pe_hit_times must be matching 1-D arrays")
    return ids, times


def load_per_pe_histogram(data_dir: str | Path, *, n_sensors: int,
                          time_min_ns: float, time_max_ns: float,
                          bin_width_ns: float, max_events: int = 0) -> PerPeObservation:
    """Read PE sequences and accumulate a PMT/time-bin count histogram.

    The event axis is intentionally retained only through ``n_events``.  The
    returned counts are sufficient for the binned Poisson likelihood used by
    v42 while preserving every accepted PE and its time bin.
    """
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    if time_max_ns <= time_min_ns:
        raise ValueError("time_max_ns must exceed time_min_ns")
    if bin_width_ns <= 0:
        raise ValueError("bin_width_ns must be positive")
    n_bins = int(np.ceil((time_max_ns - time_min_ns) / bin_width_ns))
    counts = np.zeros((n_sensors, n_bins), dtype=np.float64)
    n_events = 0
    n_pe = 0

    for path in event_files(data_dir, max_events=max_events):
        ids, times = _event_arrays(path)
        bins = np.floor((times - time_min_ns) / bin_width_ns).astype(np.int64)
        valid = ((ids >= 0) & (ids < n_sensors) & (bins >= 0) &
                 (bins < n_bins) & np.isfinite(times))
        linear = ids[valid] * n_bins + bins[valid]
        if linear.size:
            counts += np.bincount(linear, minlength=n_sensors * n_bins).reshape(counts.shape)
        n_pe += int(valid.sum())
        n_events += 1

    return PerPeObservation(
        counts=counts,
        n_events=n_events,
        n_pe=n_pe,
        time_min_ns=float(time_min_ns),
        time_max_ns=float(time_max_ns),
        bin_width_ns=float(bin_width_ns),
    )

