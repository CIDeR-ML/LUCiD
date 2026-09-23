"""Grids and axis definitions for the fiTQun tuning tables.

Every grid here is the one the reference WCTE tune was built on, kept as a
data file next to this module rather than transcribed, so the provenance is
checkable by diff:

    data/cprofile_momenta.dat   Utilities/cprofile/CprofileMomRepList.dat
    data/charge_mu_bins.txt     Utilities/chrgpdf/workdir/mutbl.txt
    data/charge_q_bins.txt      Utilities/chrgpdf/workdir/qbins_sk1.txt
    data/timepdf_momenta.json   recovered from the per-cell job directories
                                under WCSim_v1.12.19/Utilities/TuningFiles/timepdf

fiTQun indexes particle types by PDG code and works in **momentum** (MeV/c)
throughout; the generators convert to kinetic energy per particle when they
drive the simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .particles import PDG_MASSES, PDG_NAMES

DATA_DIR = Path(__file__).parent / "data"

__all__ = ["PDG_MASSES", "PDG_NAMES"]


def kinetic_energy_mev(pdg: int, momentum_mev: float) -> float:
    """Kinetic energy for a given momentum — what ``/gun/energy`` would take.

    Prefer driving PhotonSim with ``/gun/momentumAmp`` directly (G4's own gun
    messenger provides it, same command WCSim's tuning macros use); this is
    here for cross-checks and for callers that only have an energy knob.
    """
    m = PDG_MASSES[pdg]
    return float(np.sqrt(momentum_mev**2 + m * m) - m)


def _read_floats(path: Path) -> np.ndarray:
    return np.array([float(tok) for tok in path.read_text().split()], dtype=np.float64)


def cprofile_momenta() -> tuple[np.ndarray, np.ndarray]:
    """Cherenkov-profile momentum grid as ``(momenta, reps)``, sorted by momentum.

    The reference file is a ``<reps> <momentum>`` table where ``reps`` is the
    relative statistics weight of that point; duplicated momenta are summed.
    """
    toks = _read_floats(DATA_DIR / "cprofile_momenta.dat").reshape(-1, 2)
    reps, mom = toks[:, 0], toks[:, 1]
    uniq, inverse = np.unique(mom, return_inverse=True)
    summed = np.zeros_like(uniq)
    np.add.at(summed, inverse, reps)
    return uniq, summed


def charge_mu_grid() -> np.ndarray:
    """Predicted-charge points the charge PDF f(q|mu) is sampled at (p.e.)."""
    return _read_floats(DATA_DIR / "charge_mu_bins.txt")


def charge_q_edges() -> np.ndarray:
    """Observed-charge bin edges for the per-mu charge histograms (p.e.).

    The reference file ends at a 1500 sentinel that ``makeChargePDFplot.C``
    stops on rather than using as an edge; it is dropped here for the same
    reason, leaving the last real edge as the histogram's upper bound.
    """
    edges = _read_floats(DATA_DIR / "charge_q_bins.txt")
    return edges[:-1] if edges[-1] == 1500.0 else edges


def timepdf_momenta(pdg: int) -> np.ndarray:
    """Momentum grid the direct-light time PDF is tuned on, for one PDG."""
    grids = json.loads((DATA_DIR / "timepdf_momenta.json").read_text())
    return np.array(grids[str(int(pdg))], dtype=np.float64)


# --- Cherenkov-profile integral-table axes -----------------------------------
# I_n(R0, cos(theta0); p) is tabulated on a uniform (R0, cos theta0) grid, both
# axes shared across particle types — fiTQun_shared reads the binning off the
# first table it loads and asserts the rest match, and interpolates linearly,
# so the spacing must be uniform. R0 is the vertex-to-PMT distance in cm.
R0_MIN_CM, R0_MAX_CM, N_R0_BINS = 0.0, 5000.0, 100
COSTH0_MIN, COSTH0_MAX, N_COSTH0_BINS = -1.0, 1.0, 100


def r0_edges() -> np.ndarray:
    return np.linspace(R0_MIN_CM, R0_MAX_CM, N_R0_BINS + 1)


def costh0_edges() -> np.ndarray:
    return np.linspace(COSTH0_MIN, COSTH0_MAX, N_COSTH0_BINS + 1)


# --- Emission-profile (s, cos theta) accumulation grid -----------------------
# rho(s, cos theta) is built in these bins and integrated along each (R0,
# cos theta0) line of sight to give I_n. Finer than the I_n grid on purpose:
# it is a pure reduction of the PhotonSim photon list and costs only memory.
N_S_BINS = 500
N_COSTH_BINS = 500


def costh_edges() -> np.ndarray:
    return np.linspace(-1.0, 1.0, N_COSTH_BINS + 1)


# --- Direct-light time PDF ---------------------------------------------------
# htimepdf axes, verbatim from Utilities/timepdf/makehistWCSim.cc:
#   x = corrected hit time residual (ns), y = log10(predicted charge mu).
TPDF_T_MIN, TPDF_T_MAX, TPDF_N_T_BINS = -100.0, 100.0, 400
TPDF_LOGMU_MIN, TPDF_LOGMU_MAX, TPDF_N_LOGMU_BINS = -2.0, 3.0, 125


def tpdf_t_edges() -> np.ndarray:
    return np.linspace(TPDF_T_MIN, TPDF_T_MAX, TPDF_N_T_BINS + 1)


def tpdf_logmu_edges() -> np.ndarray:
    return np.linspace(TPDF_LOGMU_MIN, TPDF_LOGMU_MAX, TPDF_N_LOGMU_BINS + 1)


# --- Photosensor angular response --------------------------------------------
# epsilon(cos eta) is fitted on [0, 1] (a PMT cannot see light from behind);
# the reference angResp TF1 carries 6 parameters over that range.
ANGRESP_N_BINS = 100


def angresp_edges() -> np.ndarray:
    return np.linspace(0.0, 1.0, ANGRESP_N_BINS + 1)
