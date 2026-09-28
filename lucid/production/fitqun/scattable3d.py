"""fiTQun's 3D scattering table, as ``Utilities/scattable/scattable3d`` builds it.

Restoring fiTQun's default ``UseScatteredLight = 4`` ("6D for the 1R fit, 3D
otherwise") needs ``fiTQun_scattable3d_<config>.root``. It is a much smaller
object than the 6D table: two histograms over per-photon quantities, split on
exactly the same direct/indirect flag (``ScatTable3dLooper.cc:178``, ``isct != 0``,
which counts reflections as indirect just as the 6D table does).

Variables, from ``ScatTable3dLooper.cc:158-176``:

``R``
    emission point to PMT distance.
``costh``
    cosine between the photon's emission direction and the direction from the
    emission point to the PMT.
``dwall``
    distance from the *emission point* to the nearest wall.

The reference hardcodes SK's 1690/1810 cm for the wall; the detector geometry is
taken from the builder here so the same code works for any cylinder.
"""
from __future__ import annotations

import numpy as np

#: ScatTable3dLooper.cc:101-106.
N_R_BINS = 100
R_MAX_CM = 5000.0
N_WALL_BINS = 100
WALL_MAX_CM = 2000.0
N_COSTH_BINS = 100


def edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.linspace(0.0, R_MAX_CM, N_R_BINS + 1),
            np.linspace(0.0, WALL_MAX_CM, N_WALL_BINS + 1),
            np.linspace(-1.0, 1.0, N_COSTH_BINS + 1))


def observables(src_cm, src_dir, pmt_cm, *, det_radius_cm, det_halfheight_cm):
    """``(R, costh, dwall)`` for detected photons, in the reference's convention."""
    d = pmt_cm - src_cm
    R = np.linalg.norm(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        costh = np.einsum("ij,ij->i", d, src_dir) / R
    dwall = np.minimum(det_radius_cm - np.hypot(src_cm[:, 0], src_cm[:, 1]),
                       det_halfheight_cm - np.abs(src_cm[:, 2]))
    return R, costh, dwall


def histograms(R, costh, dwall, indirect):
    """``(indirect 3D, direct 2D)`` -- ``hsct3d`` and ``hdir2d``."""
    er, ew, ec = edges()
    ind = np.asarray(indirect, dtype=bool)
    h3, _ = np.histogramdd((R[ind], dwall[ind], costh[ind]), bins=(er, ew, ec))
    h2, _, _ = np.histogram2d(R[~ind], dwall[~ind], bins=(er, ew))
    return h3, h2


def empty() -> tuple[np.ndarray, np.ndarray]:
    return (np.zeros((N_R_BINS, N_WALL_BINS, N_COSTH_BINS)),
            np.zeros((N_R_BINS, N_WALL_BINS)))


#: MakeScatTable3d.C:45 -- cells whose isotropic reference is thinner than this
#: are zeroed rather than divided, to keep the ratio from exploding.
MIN_ISO_PER_BIN = 10.0


def build(shards, out_path, *, object_name: str = "hscattable3D"):
    """Merge per-job shards into fiTQun's 3D scattering table.

    Reproduces ``MakeScatTable3d.C``: sum the indirect 3D and direct 2D
    histograms over shards, then divide each (R, dwall) cell's angular
    distribution by the direct count spread evenly over the angle bins. The
    result is an indirect/direct ratio per solid angle, which is what fiTQun
    multiplies its direct-light prediction by.
    """
    import uproot

    tot_i = tot_d = None
    edges = None
    for path in shards:
        with np.load(path) as z:
            if tot_i is None:
                tot_i = z["hsct3d"].astype(np.float64)
                tot_d = z["hdir2d"].astype(np.float64)
                edges = (z["r_edges"], z["wall_edges"], z["costh_edges"])
            else:
                tot_i += z["hsct3d"]
                tot_d += z["hdir2d"]
    if tot_i is None:
        raise ValueError("no shards given")

    n_theta = tot_i.shape[2]
    iso = tot_d / n_theta                                   # per angle bin
    ok = iso > MIN_ISO_PER_BIN
    out = np.zeros_like(tot_i)
    np.divide(tot_i, iso[:, :, None], out=out, where=ok[:, :, None])

    with uproot.recreate(out_path) as f:
        f[object_name] = (out, *edges)
    return {"path": str(out_path), "cells_filled": int(ok.sum()),
            "cells_total": int(ok.size), "indirect": float(tot_i.sum()),
            "direct": float(tot_d.sum())}
