"""Closure test binned by the photon's INCIDENCE ANGLE on the PMT face.

fiTQun predicts charge through ``J(R, coseta, Zsrc, Zpmt) = Omega(R) * T(R) *
epsilon(coseta)``. Every closure test so far binned by ``cos(theta0)`` -- the
angle at the TRACK, between the track direction and the vertex->PMT vector.
That is a different variable from the one ``epsilon`` depends on, which is eta,
the angle between the arriving photon and the PMT's own normal.

The two are not interchangeable, and eta is the last untested term: it shifts
systematically with distance, so a wrong ``epsilon(eta)`` shows up as an
R-dependent deficit -- exactly the residual left after attenuation and
scattering were each exonerated.

For direct light the arrival direction is the unit vector from the vertex to the
PMT, so eta is fixed by geometry alone: no model, no fit.

    closure_vs_eta.py <abl-dir> [<abl-dir> ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COLS = "event icab flgHit flgMask R cth mu muscat q tHit tau tres qeeff".split()


def load_geom(path: Path):
    """PMT positions and inward normals, indexed by icab."""
    pos, nrm = [], []
    for line in open(path):
        f = line.split()
        if f and f[0] == "pmt":
            pos.append([float(x) for x in f[2:5]])
            nrm.append([float(x) for x in f[5:8]])
    return np.asarray(pos), np.asarray(nrm)


def load_truth(path: Path) -> dict:
    """event -> vertex (cm). Columns: track event pdg vx vy vz t0 dx dy dz p."""
    out = {}
    for line in open(path):
        f = line.split()
        if f and f[0] == "track":
            out[int(f[1])] = np.array([float(f[3]), float(f[4]), float(f[5])])
    return out


def load_dump(path: Path) -> dict:
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) != len(COLS):
            continue
        try:
            rows.append([float(x) for x in f])
        except ValueError:
            continue
    d = np.asarray(rows)
    c = {n: d[:, i] for i, n in enumerate(COLS)}
    keep = c["flgMask"] == 0          # fiTQun's active set; unhit PMTs stay in
    return {k: v[keep] for k, v in c.items()}


def cos_eta(c: dict, pos, nrm, vtx: dict) -> np.ndarray:
    """cos of the incidence angle on the PMT face, per dump row."""
    icab = c["icab"].astype(int)
    v = np.stack([vtx[int(e)] for e in c["event"]])
    u = pos[icab] - v
    u /= np.linalg.norm(u, axis=1)[:, None]
    # Normals point INTO the water, so a photon arriving head-on has
    # dot(u, n) = -1; flip so head-on is +1.
    return -np.einsum("ij,ij->i", u, nrm[icab])


def closure(c: dict, sel) -> tuple[float, float, int]:
    m = (c["mu"] + c["muscat"])[sel]
    q = c["q"][sel]
    if len(m) == 0 or m.mean() <= 0:
        return float("nan"), float("nan"), len(m)
    return q.mean() / m.mean(), q.std() / np.sqrt(len(q)) / m.mean(), len(m)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path, help="abl_<name>/ directories")
    a = p.parse_args(argv)

    runs = []
    for d in a.dirs:
        c = load_dump(d / "pmt.txt")
        pos, nrm = load_geom(d / "geom.txt")
        ce = cos_eta(c, pos, nrm, load_truth(d / "truth.txt"))
        runs.append((d.name.replace("abl_", ""), c, ce))

    print(f"  cos(eta) range over the active set: "
          f"{min(ce.min() for _, _, ce in runs):.3f} .. "
          f"{max(ce.max() for _, _, ce in runs):.3f}\n")

    edges = np.linspace(0.0, 1.0, 11)
    hdr = "".join(f"{n[:11]:>12}" for n, _, _ in runs)
    print(f"  {'cos(eta) bin':<18}{hdr}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = ""
        for _, c, ce in runs:
            r, _, n = closure(c, (ce >= lo) & (ce < hi))
            row += f"{r:12.3f}" if n > 50 else f"{'-':>12}"
        print(f"  {lo:6.2f}-{hi:<11.2f}{row}")

    # A ratio says nothing about how much light is involved. Weight each bin by
    # its share of the event's charge, so a large ratio on a negligible sliver
    # is not mistaken for a large error.
    print("\n  how much light is in each bin, and the absolute miss (baseline)")
    _, c, ce = runs[0]
    mu_tot = (c["mu"] + c["muscat"]).sum()
    q_tot = c["q"].sum()
    print(f"  {'cos(eta) bin':<18}{'%of pred':>10}{'%of obs':>9}"
          f"{'ratio':>8}{'excess pe':>11}{'%of all q':>11}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ce >= lo) & (ce < hi)
        m = (c["mu"] + c["muscat"])[sel].sum()
        q = c["q"][sel].sum()
        if sel.sum() <= 50:
            continue
        print(f"  {lo:6.2f}-{hi:<11.2f}{100*m/mu_tot:10.2f}{100*q/q_tot:9.2f}"
              f"{q/m if m > 0 else float('nan'):8.3f}{q-m:11.0f}{100*(q-m)/q_tot:11.2f}")
    print(f"  TOTAL predicted {mu_tot:.0f} pe, observed {q_tot:.0f} pe, "
          f"ratio {q_tot/mu_tot:.3f}")

    # Is the eta dependence just R in disguise? Hold R fixed and re-look.
    print("\n  same, restricted to one R shell (1500-2500 cm)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = ""
        for _, c, ce in runs:
            sel = (ce >= lo) & (ce < hi) & (c["R"] >= 1500) & (c["R"] < 2500)
            r, _, n = closure(c, sel)
            row += f"{r:12.3f}" if n > 50 else f"{'-':>12}"
        print(f"  {lo:6.2f}-{hi:<11.2f}{row}")


if __name__ == "__main__":
    main()
