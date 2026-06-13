"""Reconstruction result analysis — resolution stats, bootstrap CI, vertex residual decomposition.

Moved inside from the inline notebook copies (`bootstrap_percentile_ci`, `compute_statistics`,
the longitudinal/transverse split were each re-defined in ~5 notebooks). Pure numpy, no JAX —
operates on arrays of per-event fit results. Use from the recon notebooks for convergence and
resolution summaries.
"""
import numpy as np


def bootstrap_ci(x, ci=68.0, n_boot=2000, seed=0):
    """Bootstrap confidence interval on the MEDIAN of ``x`` → ``(lo, hi)``."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n_boot, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, (100 - ci) / 2)), float(np.percentile(meds, (100 + ci) / 2))


def resolution_stats(err, ci=68.0, seed=0):
    """median / mean / RMS / ``ci``%-containment of an error array, plus a bootstrap CI on the median.

    ``err`` is a 1-D array of per-event errors (signed or magnitude). Returns a dict.
    """
    e = np.asarray(err, float)
    blo, bhi = bootstrap_ci(np.abs(e), ci, seed=seed)
    return dict(median=float(np.median(e)), mean=float(e.mean()),
                rms=float(np.sqrt((e ** 2).mean())),
                containment=float(np.percentile(np.abs(e), ci)),
                median_ci=(blo, bhi), n=int(e.size))


def vertex_residual(fit_vtx, true_vtx, true_dir):
    """Decompose a vertex error into LONGITUDINAL (along the track direction) and TRANSVERSE
    (perpendicular) components, in the input units. Returns ``(lon, tra)``.

    The longitudinal direction is the hard one for charge reconstruction (the Cherenkov ring is
    vertex-degenerate along the track); splitting the residual this way is the standard recon
    diagnostic.
    """
    d = np.asarray(true_dir, float); d = d / (np.linalg.norm(d) + 1e-12)
    dv = np.asarray(fit_vtx, float) - np.asarray(true_vtx, float)
    lon = float(dv @ d)
    return lon, float(np.linalg.norm(dv - lon * d))


def angular_error_deg(fit_dir, true_dir):
    """Opening angle (degrees) between a fitted and a true direction."""
    a = np.asarray(fit_dir, float); b = np.asarray(true_dir, float)
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
