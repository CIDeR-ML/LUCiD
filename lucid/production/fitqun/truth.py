"""Per-event truth for comparing a reconstruction against the simulation.

LUCiD splits truth across two modalities, and neither alone is enough:

* ``labl`` carries the interaction vertex, the primary PDG codes and the primary
  energies, but no direction.
* ``step`` carries per-segment ``start``/``dir``/``beta_start`` and a track index,
  but no PDG.

Two things about this were established by measurement, because guessing either
one silently corrupts every resolution:

**The primary is ``track_idx == 0``**, not the track named by
``primary_track_ids_data``. On a single-muon event that field reads 1, and track 1
is a delta ray: 20 segments beginning 3 cm off the vertex, beta 0.79, about one
Cherenkov photon each. Track 0 has 5264 segments, starts exactly at the labl
vertex, holds a stable direction and emits ~90 photons per segment.

**``primary_energies_data`` is KINETIC energy.** A pion event reads 200.33 MeV
where its mass is 139.6 and the generator draws kinetic energy from 200 MeV up,
so the field cannot be a total energy. The two readings then agree: for a muon at
beta = 0.9973 the momentum from beta is 1434.8 MeV/c and from the kinetic energy
1432.0 MeV/c, a 0.2% difference. Momentum is taken from the energy (exact) and
beta is kept as a consistency check.

Positions are METRES here, as LUCiD stores them; the fiTQun comparison converts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .particles import PDG_MASSES


@dataclass
class TruthTrack:
    """The primary a single-ring fit should be recovering."""
    event: int
    pdg: int
    vertex_m: np.ndarray      # (3,)
    direction: np.ndarray     # (3,) unit
    momentum_mev: float
    t0_ns: float
    contained: bool

    @property
    def kinetic_mev(self) -> float:
        m = PDG_MASSES[abs(self.pdg)]
        return float(np.sqrt(self.momentum_mev**2 + m*m) - m)


def _momentum_from_kinetic(kinetic_mev: float, pdg: int) -> float:
    m = PDG_MASSES[abs(int(pdg))]
    e = float(kinetic_mev) + m
    return float(np.sqrt(max(e * e - m * m, 0.0)))


def _momentum_from_beta(beta: float, pdg: int) -> float:
    """Cross-check only; see the module docstring."""
    b = float(np.clip(beta, 0.0, 1.0 - 1e-9))
    m = PDG_MASSES[abs(int(pdg))]
    return float(b / np.sqrt(1.0 - b * b) * m)


def read_tracks(labl_path, step_path, *, max_events=None,
                beta_tol: float = 0.05) -> list:
    """One :class:`TruthTrack` per event, for single-primary events.

    Events are skipped rather than guessed at when the interaction has more than
    one primary (which primary a single-ring fit "should" have found is not
    defined), when the PDG has no mass in :data:`PDG_MASSES`, or when the
    momentum from the kinetic energy and from beta disagree by more than
    ``beta_tol`` -- that disagreement means the track picked is not the primary.
    """
    import h5py

    out = []
    with h5py.File(labl_path, "r") as fl, h5py.File(step_path, "r") as fs:
        keys = sorted(k for k in fl if k.startswith("event_"))
        if max_events is not None:
            keys = keys[:max_events]
        for iev, key in enumerate(keys):
            if key not in fs:
                continue
            inter = fl[key]["per_interaction"]
            pdgs = np.asarray(inter["primary_pdgs_data"])
            if len(pdgs) != 1:
                continue
            pdg = int(pdgs[0])
            if abs(pdg) not in PDG_MASSES:
                continue
            vtx = np.array([float(inter["vertex_x"][0]),
                            float(inter["vertex_y"][0]),
                            float(inter["vertex_z"][0])], dtype=np.float64)
            t0 = float(np.asarray(inter["t0"])[0])

            # The primary is track 0; within it, the earliest segment.
            st = fs[key]
            trk = np.asarray(st["track_idx"])
            sel = np.flatnonzero(trk == 0)
            if sel.size == 0:
                continue
            first = sel[np.argmin(np.asarray(st["time"])[sel])]

            d = np.array([float(np.asarray(st["dir_x"])[first]),
                          float(np.asarray(st["dir_y"])[first]),
                          float(np.asarray(st["dir_z"])[first])], dtype=np.float64)
            n = np.linalg.norm(d)
            if not n > 0:
                continue
            kinetic = float(np.asarray(inter["primary_energies_data"])[0])
            p = _momentum_from_kinetic(kinetic, pdg)
            p_beta = _momentum_from_beta(float(np.asarray(st["beta_start"])[first]), pdg)
            if not (p > 0) or abs(p_beta - p) > beta_tol * p:
                continue
            out.append(TruthTrack(
                event=iev, pdg=pdg, vertex_m=vtx, direction=d / n,
                momentum_mev=p, t0_ns=t0,
                contained=bool(np.asarray(fl[key]["per_event"]["contained"]))))
    return out


def write_text(tracks, path, *, m_to_cm: float = 100.0):
    """Write tracks in the plain form ``lucid_to_wcsim`` reads.

    Same split as the geometry export: Python owns the HDF5 layout and what
    counts as the primary, the C++ owns the WCSim ROOT format. Positions are
    converted to **cm** here so the converter needs no unit knowledge.

    ``makehistWCSim.cc`` reads these back off the event as ``GetIpnu``,
    ``GetP``, ``GetDir`` and ``GetStart``, so a tune's time-PDF stage cannot run
    on a file written without them.
    """
    from pathlib import Path

    lines = []
    for t in tracks:
        v = np.asarray(t.vertex_m, dtype=np.float64) * m_to_cm
        d = np.asarray(t.direction, dtype=np.float64)
        lines.append(f"track {t.event} {t.pdg} "
                     f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {t.t0_ns:.6f} "
                     f"{d[0]:.8f} {d[1]:.8f} {d[2]:.8f} {t.momentum_mev:.6f}")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
