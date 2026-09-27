"""Water attenuation length, measured the way the reference measures it.

``fiTQun.WaterAttenuationLength<det>`` is a single number standing in for a whole
wavelength-dependent optical model, and it cannot be derived from that model
without choosing a convention: for LUCiD's water the reciprocal mean coefficient
gives 6516 cm, the mean length 8705 cm, and one exponential fitted to the
spectrum-averaged transmission anything from 6810 cm at 5 m to 7718 cm at 40 m,
because the surviving spectrum hardens with distance.

``Utilities/scattable/AttenL`` settles it by measurement rather than algebra:
histogram the source-to-PMT distance of every detected photon, and of the
unscattered subset, then fit

    direct(R) / all(R) = a0 * exp(-R / L)

over 0-5000 cm, keeping only sources further than ``dwall_min`` from the wall
(``AttenLLooper.cc:164``). That is what this reproduces, so the number a tune
carries comes from the same simulation whose photons fiTQun will be fitting.

**Reflections must be excluded from both histograms.** The reference does it with
``isct < 1000``; LUCiD reports one ``indirect`` flag that ORs scattering with
reflection (``photon_step.py:149``), so run the sample with reflection switched
off -- ``config/SK_WAND_physics_config_noreflect.json`` -- or split the flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Reference fit range and source cut, from AttenLLooper.cc / fit_AttenL.cpp.
R_MAX_CM = 5000.0
N_R_BINS = 100
DWALL_MIN_CM = 200.0


@dataclass
class AttenuationFit:
    length_cm: float
    a0: float
    n_direct: int
    n_all: int
    covered_cm: float          # largest R with enough statistics to fit

    def __str__(self) -> str:
        return (f"L = {self.length_cm:.1f} cm (a0 = {self.a0:.4f}) from "
                f"{self.n_direct:,}/{self.n_all:,} direct/all photons out to "
                f"{self.covered_cm:.0f} cm")


def histograms(distance_cm, indirect, dwall_cm, *, r_max_cm=R_MAX_CM,
               n_bins=N_R_BINS, dwall_min_cm=DWALL_MIN_CM):
    """``(edges, all, direct)`` -- the two histograms the fit consumes."""
    distance_cm = np.asarray(distance_cm, dtype=np.float64)
    indirect = np.asarray(indirect, dtype=bool)
    keep = np.asarray(dwall_cm, dtype=np.float64) > dwall_min_cm
    edges = np.linspace(0.0, r_max_cm, n_bins + 1)
    h_all, _ = np.histogram(distance_cm[keep], bins=edges)
    h_dir, _ = np.histogram(distance_cm[keep & ~indirect], bins=edges)
    return edges, h_all, h_dir


def fit(edges, h_all, h_direct, *, min_entries: int = 50) -> AttenuationFit:
    """Fit ``a0*exp(-R/L)`` to the direct fraction.

    Done as a straight line in log space rather than by a non-linear fit, so the
    result is deterministic and has no starting-value sensitivity. Bins are
    weighted by their binomial error on the ratio; bins with too little
    statistics are dropped instead of dominating the fit with a huge error bar.
    """
    h_all = np.asarray(h_all, dtype=np.float64)
    h_direct = np.asarray(h_direct, dtype=np.float64)
    centres = 0.5 * (np.asarray(edges)[1:] + np.asarray(edges)[:-1])

    ok = (h_all >= min_entries) & (h_direct > 0)
    if ok.sum() < 3:
        raise ValueError("not enough populated bins to fit an attenuation length")
    p = h_direct[ok] / h_all[ok]
    # binomial error on p, propagated to ln p
    sigma_lnp = np.sqrt(np.maximum(p * (1.0 - p) / h_all[ok], 1e-12)) / p
    w = 1.0 / sigma_lnp**2
    x = centres[ok]
    y = np.log(p)
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    slope = (w * (x - mx) * (y - my)).sum() / (w * (x - mx) ** 2).sum()
    if not slope < 0:
        raise ValueError("direct fraction does not fall with distance")
    return AttenuationFit(length_cm=float(-1.0 / slope),
                          a0=float(np.exp(my - slope * mx)),
                          n_direct=int(h_direct.sum()), n_all=int(h_all.sum()),
                          covered_cm=float(x.max()))


def measure(distance_cm, indirect, dwall_cm, **kw) -> AttenuationFit:
    """Histogram and fit in one step."""
    hist_kw = {k: kw.pop(k) for k in ("r_max_cm", "n_bins", "dwall_min_cm") if k in kw}
    return fit(*histograms(distance_cm, indirect, dwall_cm, **hist_kw), **kw)
