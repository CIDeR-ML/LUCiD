"""Cherenkov emission profile and its integral tables (``CProf_<pdg>_WCSim.root``).

This is the one fiTQun table that is a property of the particle and the water
alone — no PMTs, no vessel — which is exactly what PhotonSim simulates. So it
is built here from PhotonSim output directly rather than from a full detector
simulation.

**The profile.** Fire a particle from the origin along +z in bare water and
histogram its Cherenkov photons in

    s        distance along the track axis of the emission point (cm)
    cos(th)  angle between the photon direction and the track axis

normalised to unit integral, ``g(s, cos th)`` is fiTQun's Cherenkov profile.
Off-axis emission (multiple scattering, secondaries) folds into the same
histogram through the projection onto the track axis, which is the
approximation fiTQun's single-track predicted charge is built on.

**The tables.** fiTQun does not use ``g`` directly. Writing the geometric
factor along the track as ``J(s) = Omega(R) T(R) eps(cos eta)`` and
interpolating it quadratically through ``s = 0, smax/2, smax``, the predicted
charge at a PMT seen from the vertex at distance ``R0`` and angle ``th0``
(``fiTQun.cc``, ``Get1Rmudist``) is

    mu = Phi0 * QEeff * nphot(p) * (j0*I_0 + j1*I_1 + j2*I_2)

so the tables are the moments of the profile along that line of sight:

    I_n(R0, cos th0; p) = int_0^smax  ds  s^n  g(s, cos th(s))
    R(s)^2   = R0^2 + s^2 - 2 R0 s cos th0
    cos th(s) = (R0 cos th0 - s) / R(s)

with two companions for the scattered-light term, taken over the profile's
``s`` marginal (normalised to 1, which is why fiTQun's isotropic expression
carries a bare ``j0`` where the direct one carries ``j0*I_0``):

    hI_iso_1 = int ds s g(s)        hI_iso_2 = int ds s^2 g(s)

``gNphot`` is the mean photon yield per primary and ``gsthr`` the track length
the profile is defined over. Both are read back by ``fiTQun_shared::LoadProfiles``.

The tables are evaluated at **bin low edges** on every axis, because that is
where fiTQun reads them (``R0bins[j] = hI3d->GetXaxis()->GetBinLowEdge(j+1)``)
and what its trilinear interpolation assumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import uproot

from . import binning, rootio

# s_max is not known until the photons have been looked at, and the profile's
# s axis runs from 0 to s_max -- so the reduction is two passes over the file:
# a cheap 1D pass in absolute s to find s_max, then the 2D pass on the final
# grid. Binning the first pass onto a fixed generous range instead would alias
# badly, since one grid has to cover both a 1 m and a 50 m track.
_N_S_SCAN = 200_000

# s_max is the quantile of the emission-distance distribution that LUCiD
# already uses for its own s_max parametrisation (PhotonSim/tools/smax), so the
# two stay one definition rather than two that drift.
_SMAX_QUANTILE = 0.9999


@dataclass
class ProfileCell:
    """The reduced emission profile for one (particle, momentum) sample."""
    pdg: int
    momentum_mev: float
    s_edges: np.ndarray          # (n_s+1,) cm, spanning [0, s_max]
    costh_edges: np.ndarray      # (n_costh+1,)
    density: np.ndarray          # (n_s, n_costh) normalised: sum(density*ds*dc) == 1
    s_max_cm: float
    n_photons: float             # mean Cherenkov photons per primary
    n_events: int

    def s_marginal(self) -> np.ndarray:
        """``g(s)``, the profile integrated over angle (unit integral in s)."""
        dc = self.costh_edges[1] - self.costh_edges[0]
        return self.density.sum(axis=1) * dc

    def save(self, path) -> Path:
        """Per-cell intermediate, so the scan can fan out and merge later."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, pdg=self.pdg, momentum_mev=self.momentum_mev,
            s_edges=self.s_edges, costh_edges=self.costh_edges,
            density=self.density, s_max_cm=self.s_max_cm,
            n_photons=self.n_photons, n_events=self.n_events)
        return path

    @classmethod
    def load(cls, path) -> "ProfileCell":
        with np.load(path) as d:
            return cls(pdg=int(d["pdg"]), momentum_mev=float(d["momentum_mev"]),
                       s_edges=d["s_edges"], costh_edges=d["costh_edges"],
                       density=d["density"], s_max_cm=float(d["s_max_cm"]),
                       n_photons=float(d["n_photons"]), n_events=int(d["n_events"]))

    def __add__(self, other: "ProfileCell") -> "ProfileCell":
        """Merge two samples of the same cell, weighting by event count."""
        if (self.pdg, self.momentum_mev) != (other.pdg, other.momentum_mev):
            raise ValueError("can only merge profiles of the same (pdg, momentum)")
        if not np.allclose(self.s_edges, other.s_edges):
            raise ValueError("profiles have different s binning; merge before cropping")
        wa, wb = self.n_events, other.n_events
        tot = wa + wb
        return ProfileCell(
            pdg=self.pdg, momentum_mev=self.momentum_mev,
            s_edges=self.s_edges, costh_edges=self.costh_edges,
            density=(self.density * wa + other.density * wb) / tot,
            s_max_cm=self.s_max_cm,
            n_photons=(self.n_photons * wa + other.n_photons * wb) / tot,
            n_events=tot,
        )


def accumulate(photonsim_path, *, pdg: int, momentum_mev: float,
               direction=(0.0, 0.0, 1.0), s_hi_cm: float = 100000.0,
               n_s_bins: int = binning.N_S_BINS,
               n_costh_bins: int = binning.N_COSTH_BINS,
               s_max_cm: Optional[float] = None,
               quantile: float = _SMAX_QUANTILE,
               step_size: str = "200 MB") -> ProfileCell:
    """Reduce one PhotonSim file to a :class:`ProfileCell`.

    Reads ``OpticalPhotonsRaw`` in chunks, so a high-momentum sample never has
    to fit in memory. ``s_hi_cm`` only has to be an upper bound on the track
    length; the profile is cropped to the s_max derived from the data, or to
    ``s_max_cm`` when the caller pins it (which also skips the scan pass, e.g.
    to keep the split jobs of one cell consistent).
    """
    axis = np.asarray(direction, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    costh_edges = np.linspace(-1.0, 1.0, n_costh_bins + 1)

    with uproot.open(photonsim_path) as f:
        n_events = int(f["OpticalPhotons"].num_entries)
        raw = f["OpticalPhotonsRaw"]

        if s_max_cm is None:
            scan_edges = np.linspace(0.0, s_hi_cm, _N_S_SCAN + 1)
            scan = np.zeros(_N_S_SCAN, dtype=np.float64)
            for s, _ in _iterate_photons(raw, axis, step_size, positions_only=True):
                scan += np.histogram(s, bins=scan_edges)[0]
            if scan.sum() == 0:
                raise ValueError(f"{photonsim_path}: no Cherenkov photons found")
            s_max_cm = _quantile_from_hist(scan, scan_edges, quantile)

        s_edges = np.linspace(0.0, float(s_max_cm), n_s_bins + 1)
        counts = np.zeros((n_s_bins, n_costh_bins), dtype=np.float64)
        n_photons = 0
        for s, costh in _iterate_photons(raw, axis, step_size):
            n_photons += s.size
            counts += np.histogram2d(s, costh, bins=[s_edges, costh_edges])[0]

    if n_photons == 0:
        raise ValueError(f"{photonsim_path}: no Cherenkov photons found")

    ds = s_edges[1] - s_edges[0]
    dc = costh_edges[1] - costh_edges[0]
    norm = counts.sum() * ds * dc
    if norm <= 0:
        raise ValueError(f"{photonsim_path}: no photons inside [0, s_max]")

    return ProfileCell(pdg=pdg, momentum_mev=float(momentum_mev),
                       s_edges=s_edges, costh_edges=costh_edges,
                       density=counts / norm, s_max_cm=float(s_max_cm),
                       n_photons=n_photons / max(n_events, 1), n_events=n_events)


def _iterate_photons(raw, axis: np.ndarray, step_size: str,
                     positions_only: bool = False):
    """Yield ``(s, cos theta)`` per chunk of the raw photon tree.

    ``s`` is the projection of the emission point onto the track axis and
    ``cos theta`` the photon direction's angle to it -- fiTQun's profile
    variables. PhotonSim writes positions in mm; the tables are in cm.
    """
    pos_branches = ["PhotonPosX", "PhotonPosY", "PhotonPosZ"]
    dir_branches = ["PhotonDirX", "PhotonDirY", "PhotonDirZ"]
    branches = pos_branches if positions_only else pos_branches + dir_branches
    for chunk in raw.iterate(branches, step_size=step_size, library="np"):
        # Branches are jagged (one entry per chunk of photons); flatten.
        pos = np.stack([np.concatenate(chunk[b]) for b in pos_branches], axis=1)
        s = (pos @ axis) * 0.1
        if positions_only:
            yield s, None
            continue
        dirs = np.stack([np.concatenate(chunk[b]) for b in dir_branches], axis=1)
        yield s, dirs @ axis


def _quantile_from_hist(marginal: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Linear-interpolated quantile of a binned distribution."""
    cdf = np.cumsum(marginal)
    total = cdf[-1]
    if total <= 0:
        raise ValueError("empty s distribution")
    return float(np.interp(q * total, np.concatenate([[0.0], cdf]), edges))


def integral_tables(cells: list[ProfileCell], *,
                    r0_edges: Optional[np.ndarray] = None,
                    costh0_edges: Optional[np.ndarray] = None,
                    chunk_r0: int = 10):
    """I_n(R0, cos th0; p) for n = 0,1,2, plus the isotropic-source moments.

    Returns ``(I, iso1, iso2, r0_lo, costh0_lo, momenta)`` where ``I`` has shape
    ``(3, nR0, ncosth0, nmom)`` and the axis arrays are the bin **low edges**
    the values are evaluated at.
    """
    cells = sorted(cells, key=lambda c: c.momentum_mev)
    if not cells:
        raise ValueError("no profile cells")
    r0_edges = binning.r0_edges() if r0_edges is None else np.asarray(r0_edges)
    costh0_edges = binning.costh0_edges() if costh0_edges is None else np.asarray(costh0_edges)
    r0 = r0_edges[:-1]
    c0 = costh0_edges[:-1]

    n_mom = len(cells)
    out = np.zeros((3, len(r0), len(c0), n_mom), dtype=np.float64)
    iso1 = np.zeros(n_mom)
    iso2 = np.zeros(n_mom)

    for im, cell in enumerate(cells):
        s = 0.5 * (cell.s_edges[1:] + cell.s_edges[:-1])
        ds = cell.s_edges[1] - cell.s_edges[0]
        g_s = cell.s_marginal()
        iso1[im] = float((g_s * s * ds).sum())
        iso2[im] = float((g_s * s**2 * ds).sum())

        for lo in range(0, len(r0), chunk_r0):
            sl = slice(lo, min(lo + chunk_r0, len(r0)))
            block = _line_of_sight_moments(cell, r0[sl], c0, s, ds)
            out[:, sl, :, im] = block

    return out, iso1, iso2, r0, c0, np.array([c.momentum_mev for c in cells])


def _line_of_sight_moments(cell: ProfileCell, r0: np.ndarray, c0: np.ndarray,
                           s: np.ndarray, ds: float) -> np.ndarray:
    """``int ds s^n g(s, cos th(s))`` for a block of R0 values, vectorised."""
    R0 = r0[:, None, None]
    C0 = c0[None, :, None]
    S = s[None, None, :]

    R2 = R0 * R0 + S * S - 2.0 * R0 * S * C0
    R = np.sqrt(np.maximum(R2, 0.0))
    # A PMT exactly on the track at that s has no defined viewing angle; it
    # contributes nothing (the solid angle J carries the 1/R^2 that diverges
    # there, and fiTQun clamps mu at zero anyway).
    with np.errstate(divide="ignore", invalid="ignore"):
        costh = np.where(R > 0.0, (R0 * C0 - S) / np.where(R > 0.0, R, 1.0), -1.0)
    np.clip(costh, -1.0, 1.0, out=costh)

    # Linear interpolation of g across cos(theta) at each s bin.
    edges = cell.costh_edges
    dc = edges[1] - edges[0]
    fidx = (costh - edges[0]) / dc - 0.5
    np.clip(fidx, 0.0, cell.density.shape[1] - 1.0, out=fidx)
    i0 = np.floor(fidx).astype(np.intp)
    np.clip(i0, 0, cell.density.shape[1] - 2, out=i0)
    # In the outer half-bins fidx sits beyond the last bin centre; clamping the
    # weight holds the profile flat there instead of extrapolating it.
    w = np.clip(fidx - i0, 0.0, 1.0)

    s_idx = np.broadcast_to(np.arange(len(s)), costh.shape)
    g = cell.density[s_idx, i0] * (1.0 - w) + cell.density[s_idx, i0 + 1] * w

    return np.stack([(g * ds).sum(axis=-1),
                     (g * s * ds).sum(axis=-1),
                     (g * s * s * ds).sum(axis=-1)])


def write_cprofile(path, pdg: int, cells: list[ProfileCell], **kwargs) -> None:
    """Write ``CProf_<pdg>_WCSim.root`` as ``fiTQun_shared::LoadProfiles`` reads it."""
    I, iso1, iso2, r0, c0, momenta = integral_tables(cells, **kwargs)

    # fiTQun reads the axes off bin low edges, so the stored edges must be the
    # evaluation points with one extra edge to close the last bin.
    r0_ax = _edges_from_low(r0)
    c0_ax = _edges_from_low(c0)
    mom_ax = _edges_from_low(momenta)

    objects = {}
    for n in range(3):
        objects[f"hI3d_{n}"] = rootio.th3(
            f"hI3d_{n}", r0_ax, c0_ax, mom_ax, I[n],
            title=f"I_{n}(R0, cos#theta_{{0}}; p)",
            xtitle="R0 (cm)", ytitle="cos #theta_{0}", ztitle="p (MeV/c)")
    for n, iso in ((1, iso1), (2, iso2)):
        objects[f"hI_iso_{n}"] = rootio.th1(
            f"hI_iso_{n}", mom_ax, iso,
            title=f"isotropic-source moment <s^{n}>", xtitle="p (MeV/c)")

    cells = sorted(cells, key=lambda c: c.momentum_mev)
    objects["gNphot"] = rootio.tgraph(
        "gNphot", momenta, [c.n_photons for c in cells], "photons per primary")
    objects["gsthr"] = rootio.tgraph(
        "gsthr", momenta, [c.s_max_cm for c in cells], "s_max (cm)")

    rootio.write(path, objects)


def _edges_from_low(low: np.ndarray) -> np.ndarray:
    """Turn evaluation points (read by ROOT as bin low edges) into bin edges."""
    low = np.asarray(low, dtype=np.float64)
    if len(low) == 1:
        return np.array([low[0], low[0] + 1.0])
    last = low[-1] + (low[-1] - low[-2])
    return np.concatenate([low, [last]])
