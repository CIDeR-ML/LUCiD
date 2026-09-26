"""Grids and axis definitions for the fiTQun tuning tables.

Every grid here is the one the reference WCTE tune was built on, kept as a
data file next to this module rather than transcribed, so the provenance is
checkable by diff:

    data/cprofile_momenta.dat        Utilities/cprofile/CprofileMomRepList.dat
    data/cprofile_momenta_<pdg>.txt  the tune's own CProf_<pdg>_fit_WCSim.root,
                                     whose gNphot carries one point per cell
    data/charge_mu_bins.txt          Utilities/chrgpdf/workdir/mutbl.txt
    data/charge_q_bins.txt           Utilities/chrgpdf/workdir/qbins_sk1.txt
    data/timepdf_momenta.json        recovered from the per-cell job directories
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


def cprofile_momenta(pdg: int) -> np.ndarray:
    """Cherenkov-profile momentum grid for one PDG (MeV/c), ascending.

    The grid is **per particle**, and each starts at that particle's Cherenkov
    threshold -- fiTQun's own ``pCherenkovThr`` with ``nphase = 1.334`` puts it
    at 0.6 / 120 / 158 MeV/c for e- / mu- / pi+, and the grids start at 1 / 120 /
    156. Sharing one list across PDGs costs the electron every cell below
    120 MeV/c, where it has 108 of its 659 points.
    """
    return _read_floats(DATA_DIR / f"cprofile_momenta_{int(pdg)}.txt")


def cprofile_momentum_reps() -> tuple[np.ndarray, np.ndarray]:
    """``(momenta, reps)`` from ``CprofileMomRepList.dat`` -- the mu- grid only.

    ``reps`` is the relative statistics weight the reference gave each point
    (duplicated momenta are summed). It drives how many events a cell gets, not
    which cells exist; the grid itself comes from :func:`cprofile_momenta`.
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


def charge_mu_labels() -> list:
    """The mu grid as the **literal text** of ``mutbl.txt``.

    ``gen2d.cc`` opens ``Form("%s_pdf.root", mustr[i])`` with the token it read
    straight from that file, so the filenames have to carry its exact spelling:
    ``1.0``, not ``1``. Formatting the float instead loses the nine whole-number
    entries and ``gen2d.cc`` then dereferences a null TFile.
    """
    return (DATA_DIR / "charge_mu_bins.txt").read_text().split()


def charge_q_edges() -> np.ndarray:
    """Observed-charge bin edges for the per-mu charge histograms (p.e.).

    All 481 values are used, the trailing 1500 included. ``makeChargePDFplot.C``
    breaks its read loop on 1500 *before* incrementing the index, so
    ``nqbins=480`` while ``qbinEdg[480]=1500`` is still handed to the TH1D
    constructor -- 480 bins spanning [0, 1500], last bin [1495, 1500].
    """
    return _read_floats(DATA_DIR / "charge_q_bins.txt")


def timepdf_momenta(pdg: int) -> np.ndarray:
    """Momentum grid the direct-light time PDF is tuned on, for one PDG."""
    grids = json.loads((DATA_DIR / "timepdf_momenta.json").read_text())
    return np.array(grids[str(int(pdg))], dtype=np.float64)


# --- Cherenkov-profile integral-table axes -----------------------------------
# Utilities/cprofile/integcprofile.cc: nR0bin=401, nth0bin=201, R0max=5000, with
# edges R0binEdgs[i]=i*R0max/(nR0bin-1) and th0binEdgs[i]=i*2/(nth0bin-1)-1 --
# so 12.5 cm and 0.01 steps, and one extra edge closing each axis at 5012.5 and
# 1.01 (which is what the shipped tables carry). fiTQun reads the values at bin
# *low edges* and interpolates trilinearly between them. The same file switches
# to nR0bin=1201, nth0bin=101, R0max=30000 when called with its HK flag.
R0_MIN_CM, R0_MAX_CM, N_R0_POINTS = 0.0, 5000.0, 401
COSTH0_MIN, COSTH0_MAX, N_COSTH0_POINTS = -1.0, 1.0, 201


def r0_points() -> np.ndarray:
    """The R0 values I_n is evaluated at (cm)."""
    return np.linspace(R0_MIN_CM, R0_MAX_CM, N_R0_POINTS)


def costh0_points() -> np.ndarray:
    return np.linspace(COSTH0_MIN, COSTH0_MAX, N_COSTH0_POINTS)


# --- Emission-profile (s, cos theta) accumulation grid -----------------------
# The emission angle is histogrammed in 500 bins over [-1, 1] and the flight
# distance in 2200 bins over [-500, 5000] cm -- a fixed axis, not one adapted per
# momentum, keeping the negative-s region (light emitted behind the vertex).
#
# The 2.5 cm bin width is pinned: genhist.cc reports s_max as a bin *centre*, and
# every gsthr value in the shipped tables is an odd multiple of 1.25 cm. The
# range endpoints are not -- the raw (s, cos theta) histogram is filled inside
# skdetsim, which is not part of the tuning repo. They only need to cover the
# profile (the largest shipped s_max is 3736 cm) and leave the width at 2.5.
S_MIN_CM, S_MAX_CM, N_S_BINS = -500.0, 5000.0, 2200
N_COSTH_BINS = 500


def s_edges() -> np.ndarray:
    return np.linspace(S_MIN_CM, S_MAX_CM, N_S_BINS + 1)


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
# epsilon(cos eta) is histogrammed on [0, 1] (a PMT cannot see light from
# behind) in 25 bins -- angularResponsePlotter.cc's nBins, confirmed by the
# shipped angResp TF1's fNpfits = 25. fit_cos.C then fits 6 parameters.
ANGRESP_N_BINS = 25

# Shell radii and half-width, from angularResponsePlotter_v1.C's !isNuPRISM
# branch. A NuPRISM-shaped detector takes the other branch instead, which is a
# different list, not a scaling of this one.
ANGRESP_SHELL_DR_CM = 50.0
ANGRESP_SHELL_RADII_CM = tuple(float(r) for r in range(100, 1501, 100))
ANGRESP_SHELL_RADII_NUPRISM_CM = tuple(float(r) for r in range(50, 301, 50))


def angresp_edges() -> np.ndarray:
    return np.linspace(0.0, 1.0, ANGRESP_N_BINS + 1)
