"""Single validated time-grid convention for JUNO laser calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """Uniform finite observation grid with cell-centered basis times."""

    time_min_ns: float
    time_max_ns: float
    bin_width_ns: float
    n_bins: int = field(init=False)
    edges: np.ndarray = field(init=False, repr=False, compare=False)
    centers: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate exact tiling and precompute immutable grid arrays."""
        values = np.asarray(
            (self.time_min_ns, self.time_max_ns, self.bin_width_ns),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("time-grid values must be finite")
        if self.time_max_ns <= self.time_min_ns:
            raise ValueError("time_max_ns must exceed time_min_ns")
        if self.bin_width_ns <= 0.0:
            raise ValueError("bin_width_ns must be positive")
        ratio = (
            float(self.time_max_ns) - float(self.time_min_ns)
        ) / float(self.bin_width_ns)
        n_bins = int(round(ratio))
        tolerance = 1.0e-9 * max(1.0, abs(ratio))
        if n_bins <= 0 or not np.isclose(
            ratio, n_bins, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                "observation window must contain an integer number of bins: "
                f"width/bin_width={ratio:.17g}"
            )
        edges = float(self.time_min_ns) + np.arange(
            n_bins + 1, dtype=np.float64
        ) * float(self.bin_width_ns)
        centers = 0.5 * (edges[:-1] + edges[1:])
        edges.setflags(write=False)
        centers.setflags(write=False)
        object.__setattr__(self, "time_min_ns", float(self.time_min_ns))
        object.__setattr__(self, "time_max_ns", float(self.time_max_ns))
        object.__setattr__(self, "bin_width_ns", float(self.bin_width_ns))
        object.__setattr__(self, "n_bins", n_bins)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "centers", centers)

    def validate_mass_shape(self, masses, *, name: str = "bin masses"):
        """Require a PMT-by-time array to use this grid's time dimension."""
        shape = np.shape(masses)
        if len(shape) != 2 or shape[1] != self.n_bins:
            raise ValueError(
                f"{name} must have time dimension {self.n_bins}, got {shape}"
            )
