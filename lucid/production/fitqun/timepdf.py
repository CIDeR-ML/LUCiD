"""Direct-light time PDF — the ``<cell>_hist.root`` stage-1 inputs.

fiTQun's time likelihood is conditioned on how much light a PMT was predicted
to see: a PMT expecting 0.1 p.e. and one expecting 100 have very different
first-hit time distributions. So the tuning histogram is two-dimensional,

    x = t_c, the hit time corrected for time of flight from the track midpoint
    y = log10(mu), the predicted charge at that PMT

exactly as ``Utilities/timepdf/makehistWCSim.cc`` fills it, and ``fittpdf.cc``
then fits a mean and a width per momentum in slices of log10(mu).

The time correction, which is the whole content of the x axis:

    t_c = t_hit - t_offset - R_mid * n / c - s_mid / c

with ``s_mid = smax/2`` the track midpoint, ``R_mid`` the distance from that
midpoint to the PMT, and ``n`` the water refractive index used for the
time-of-flight subtraction. What is left is the physical spread: emission
along the track, dispersion, and the PMT's transit-time jitter.

Two deliberate differences from the reference chain, both consequences of this
being a tune *to LUCiD*:

* **Direct light only** comes from running with scattering and reflection off
  rather than from WCSim's ``/fqTune/mode killScatterRef``.
* **The predicted charge on the y axis** is LUCiD's, not fiTQun's. The
  reference links against fiTQun and calls ``Get1Rmudist``, which needs the
  Cherenkov profile and charge PDF to have been tuned already. Taking LUCiD's
  own expected charge makes the first pass self-consistent and removes the
  circular dependency; if you want a second iteration indexed by the tuned
  fiTQun's mu, pass it in as ``mu`` and everything else here is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import binning, rootio

# Speed of light in vacuum, cm/ns -- the constant makehistWCSim.cc uses.
C_CM_PER_NS = 29.9792458


def corrected_time(hit_time_ns: np.ndarray, pmt_pos_cm: np.ndarray, *,
                   vertex_cm: np.ndarray, direction: np.ndarray,
                   s_max_cm: float, t0_ns: float = 0.0,
                   n_water: float = 1.38) -> np.ndarray:
    """``t_c`` for a set of hits from one track.

    ``direction`` is the unit track direction; ``s_max_cm`` the track length the
    Cherenkov profile is defined over (``gsthr`` in the profile file), so the
    midpoint subtraction matches what fiTQun does at reconstruction time.
    """
    vertex = np.asarray(vertex_cm, dtype=np.float64)
    u = np.asarray(direction, dtype=np.float64)
    u = u / np.linalg.norm(u)
    s_mid = 0.5 * s_max_cm
    mid = vertex + s_mid * u
    r_mid = np.linalg.norm(np.asarray(pmt_pos_cm, dtype=np.float64) - mid, axis=-1)
    return (np.asarray(hit_time_ns, dtype=np.float64) - t0_ns
            - r_mid * n_water / C_CM_PER_NS - s_mid / C_CM_PER_NS)


class TimePdfAccumulator:
    """Fills the ``htimepdf`` TH2D over many events of one (particle, momentum)."""

    def __init__(self, t_edges: Optional[np.ndarray] = None,
                 logmu_edges: Optional[np.ndarray] = None):
        self.t_edges = binning.tpdf_t_edges() if t_edges is None else np.asarray(t_edges)
        self.logmu_edges = (binning.tpdf_logmu_edges() if logmu_edges is None
                            else np.asarray(logmu_edges))
        shape = (len(self.t_edges) - 1, len(self.logmu_edges) - 1)
        self.counts = np.zeros(shape, dtype=np.float64)
        self.sumw2 = np.zeros(shape, dtype=np.float64)
        self.n_events = 0

    def fill(self, t_corrected: np.ndarray, mu: np.ndarray,
             weights: Optional[np.ndarray] = None) -> None:
        """Add one event's hits. ``mu`` is the predicted charge at each hit PMT."""
        t = np.asarray(t_corrected, dtype=np.float64)
        m = np.asarray(mu, dtype=np.float64)
        # A PMT with zero predicted charge has no defined log10(mu); it is also
        # not something fiTQun ever evaluates the time likelihood at.
        good = m > 0
        t, m = t[good], m[good]
        w = (np.ones_like(t) if weights is None
             else np.asarray(weights, dtype=np.float64)[good])
        bins = [self.t_edges, self.logmu_edges]
        h, _, _ = np.histogram2d(t, np.log10(m), bins=bins, weights=w)
        h2, _, _ = np.histogram2d(t, np.log10(m), bins=bins, weights=w * w)
        self.counts += h
        self.sumw2 += h2
        self.n_events += 1

    def __add__(self, other: "TimePdfAccumulator") -> "TimePdfAccumulator":
        if self.counts.shape != other.counts.shape:
            raise ValueError("time-PDF accumulators have different binning")
        out = TimePdfAccumulator(self.t_edges, self.logmu_edges)
        out.counts = self.counts + other.counts
        out.sumw2 = self.sumw2 + other.sumw2
        out.n_events = self.n_events + other.n_events
        return out

    def to_root(self):
        return rootio.th2(
            "htimepdf", self.t_edges, self.logmu_edges, self.counts, sumw2=self.sumw2,
            title="Direct-light time residual vs predicted charge",
            xtitle="t_{c} (ns)", ytitle="log_{10}#mu")

    def write(self, path) -> Path:
        """Write ``<cell>_hist.root``, the name ``combhists``/``fittpdf`` expect."""
        rootio.write(path, {"htimepdf": self.to_root()})
        return Path(path)


def cell_name(pdg: int, momentum_mev: float, subjob: int = 0) -> str:
    """``<pdg>_<p>_0_<subjob>_0`` — the reference chain's cell directory naming."""
    return f"{int(pdg)}_{int(round(momentum_mev))}_0_{int(subjob)}_0"
