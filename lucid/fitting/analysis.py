"""Reconstruction result analysis — track errors, resolution stats, the bootstrap, vertex residual
decomposition.

Moved inside from the inline notebook copies (`bootstrap_percentile_ci`, `compute_statistics`,
the longitudinal/transverse split were each re-defined in ~5 notebooks). Pure numpy, no JAX —
operates on arrays of per-event fit results. Use from the recon notebooks for convergence and
resolution summaries.
"""
import numpy as np

from lucid.fitting.recon import vec9_dir


def bootstrap(x, statistic, *, n_boot, seed):
    """``n_boot`` bootstrap replicates of ``statistic``: ``x`` (events along the first axis)
    resampled with replacement, the indices drawn from ``numpy.random.default_rng(seed)``.

    Their standard deviation is the statistic's standard error; :func:`bootstrap_ci` takes an
    interval from them.
    """
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    return np.array([statistic(x[i]) for i in idx])


def bootstrap_ci(x, statistic, *, ci, n_boot, seed):
    """Percentile-bootstrap ``ci``% interval ``(lo, hi)`` of ``statistic(x)``: the central ``ci``%
    of its :func:`bootstrap` replicates.

    The percentile interval assumes no symmetry, which the statistics quoted here (a median, a
    68% containment of non-negative errors) do not have.
    """
    lo, hi = np.percentile(bootstrap(x, statistic, n_boot=n_boot, seed=seed),
                           [(100 - ci) / 2, (100 + ci) / 2])
    return float(lo), float(hi)


def resolution_stats(err, ci=68.0, seed=0):
    """median / mean / RMS / ``ci``%-containment of an error array, plus a bootstrap CI on the median.

    ``err`` is a 1-D array of per-event errors (signed or magnitude). Returns a dict.

    ``median``, ``mean`` and ``median_ci`` describe the array AS GIVEN, so on a signed array they
    measure bias; ``containment`` is a magnitude by definition and is always taken on ``|err|``.
    The bootstrap must use the same array as ``median``, or on signed input the CI need not
    contain the median it is reported with.
    """
    e = np.asarray(err, float)
    blo, bhi = bootstrap_ci(e, np.median, ci=ci, n_boot=2000, seed=seed)
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
    """Opening angle (degrees) between fitted and true directions, along the last axis."""
    a = np.asarray(fit_dir, float); b = np.asarray(true_dir, float)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def track_errors(vec9, truth_vec9, true_dir):
    """Errors of a track fit against the truth: ``[vertex cm, direction deg, dE MeV, dt0 ns]``
    along the last axis.

    ``vec9`` is one fitted 9-vector or a stack of them (a fit trajectory, ``(T, 9)``);
    ``truth_vec9`` and ``true_dir`` are the event's truth. The vertex error is the distance to the
    true vertex and the direction error the opening angle to ``true_dir``; ``dE = E - E_true`` and
    ``dt0 = t0 - t0_true`` keep their sign, so a bias stays visible (take ``abs`` for a resolution).
    """
    v = np.asarray(vec9, float); truth = np.asarray(truth_vec9, float)
    return np.stack([np.linalg.norm(v[..., 1:4] - truth[1:4], axis=-1) * 100.0,
                     angular_error_deg(vec9_dir(v), true_dir),
                     v[..., 0] - truth[0], v[..., 8] - truth[8]], -1)
