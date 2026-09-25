"""Drive the indirect-light tables over the 3 MeV electron sample.

The table class and its ratio live in :mod:`lucid.production.fitqun.scattable`;
this fills them from propagated photons. One MC pass produces both halves, split
on the per-photon scatter flag exactly as the reference splits on ``isct``:

* photons that scattered or reflected go into the 6D **scattered** table,
  binned in (zs, rs, t, ast, ct, phi);
* photons that arrived directly go into the **direct** table, which the
  reference bins only in (zs, rs, t, ast) -- direct light has no
  source-direction dependence worth resolving, and ``DivideUnnormalized4D``
  later spreads its count over the ct*phi cells.

The six coordinates are computed the way ``scatTableLooper`` computes them, in
cm and radians:

    zs   source z                      rs   source distance from the axis
    t    PMT z (barrel) or its distance from the axis (end caps)
    ast  DeltaPhi(PMT, source) about the detector axis
    ct   source direction's z component
    phi  DeltaPhi(source direction, source->PMT vector)

Which surface a PMT belongs to follows fiTQun's *live* selection, a cut on the
PMT's orientation (``|dir_z| > 0.8``), not on its position -- the position cut
that appears nearby in ``fiTQun.cc`` is commented out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import scattable

# LUCiD propagation reports positions in meters; fiTQun's tables are in cm.
M_TO_CM = 100.0


def _delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``TVector3::DeltaPhi`` semantics: the difference wrapped to (-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def coordinates(source_pos: np.ndarray, source_dir: np.ndarray,
                pmt_pos: np.ndarray, *, is_cap: np.ndarray) -> tuple:
    """The six table coordinates for a set of (source, PMT) pairs."""
    source_pos = np.asarray(source_pos, dtype=np.float64)
    source_dir = np.asarray(source_dir, dtype=np.float64)
    pmt_pos = np.asarray(pmt_pos, dtype=np.float64)

    zs = source_pos[:, 2]
    rs = np.hypot(source_pos[:, 0], source_pos[:, 1])
    # The PMT coordinate is whichever one varies over its surface.
    t = np.where(is_cap, np.hypot(pmt_pos[:, 0], pmt_pos[:, 1]), pmt_pos[:, 2])

    phi_pmt = np.arctan2(pmt_pos[:, 1], pmt_pos[:, 0])
    phi_src = np.arctan2(source_pos[:, 1], source_pos[:, 0])
    ast = _delta_phi(phi_pmt, phi_src)

    ct = source_dir[:, 2]
    to_pmt = pmt_pos - source_pos
    phi = _delta_phi(np.arctan2(source_dir[:, 1], source_dir[:, 0]),
                     np.arctan2(to_pmt[:, 1], to_pmt[:, 0]))
    return zs, rs, t, ast, ct, phi


def make_tables(nbins: dict, bounds: dict) -> dict:
    """Empty scattered/direct pairs for each surface.

    The direct table collapses ct and phi to a single bin, which is what makes
    it the 4D partner ``ratio_to`` expects.
    """
    out = {}
    for surface in scattable.SURFACES:
        nb = tuple(nbins[surface])
        bd = tuple(bounds[surface])
        direct_nb = nb[:4] + (1, 1)
        out[surface] = {
            "scattered": scattable.ScatTable(surface, nb, bd),
            "direct": scattable.ScatTable(surface, direct_nb, bd),
        }
    return out


def fill(tables: dict, chunk: dict, *, pmt_positions_m: np.ndarray,
         pmt_dir_z: np.ndarray) -> None:
    """Add one propagated chunk to the tables, split on the scatter flag.

    Positions arrive in METERS, as LUCiD propagation reports them, and are
    converted here to the cm that ``scatTableLooper``'s coordinates use.
    """
    detected = np.asarray(chunk["detected"], dtype=bool)
    if not detected.any():
        return
    dev = chunk.get("deviated")
    if dev is None:
        raise ValueError(
            "the scattering table needs the per-photon 'deviated' flag to "
            "separate indirect from direct light in a single pass")
    dev = np.asarray(dev, dtype=bool)[detected]

    sid = np.asarray(chunk["sensor_id"])[detected]
    src_pos = np.asarray(chunk["emission_pos"])[detected] * M_TO_CM
    src_dir = np.asarray(chunk["emission_dir"])[detected]
    pmt_positions = np.asarray(pmt_positions_m, dtype=np.float64) * M_TO_CM

    surface = scattable.surface_for(pmt_dir_z[sid])
    is_cap = surface != "sidescattable"
    coords = coordinates(src_pos, src_dir, pmt_positions[sid], is_cap=is_cap)

    for name in scattable.SURFACES:
        on_surface = surface == name
        if not on_surface.any():
            continue
        for key, mask in (("scattered", on_surface & dev),
                          ("direct", on_surface & ~dev)):
            if mask.any():
                tables[name][key].fill(*(c[mask] for c in coords))


def finalise(tables: dict) -> dict:
    """Form the scattered/direct ratio fiTQun looks up, per surface."""
    return {name: pair["scattered"].ratio_to(pair["direct"])
            for name, pair in tables.items()}


def write(tables: dict, path, metadata: Optional[dict] = None) -> Path:
    """Write the ratios for ``tools/fitqun/build_scattable_converter.sh``."""
    return scattable.write_hdf5(path, finalise(tables), metadata)
