"""Indirect-light tables — the scattered/reflected fraction fiTQun adds to mu.

Direct light gets fiTQun the bulk of the predicted charge; the rest is light
that scattered in the water or reflected off a surface before reaching a PMT.
fiTQun carries that as a multiplicative ratio looked up in a 6-dimensional
table (``fiTQun::GetScatRatio``), one table per surface the PMT sits on:

    zs    source z (cm)
    rs    source distance from the detector axis (cm)
    t     the PMT coordinate that varies over that surface --
          z for the barrel, distance from the axis for the end caps
    ast   azimuthal angle between source and PMT
    ct    cosine of the source direction's polar angle
    phi   azimuthal angle of the source direction about the source-to-PMT line

Bins are uniform on every axis and the flat index runs ``zs`` fastest, matching
``TScatTable::GetIndex``/``fillbininfo`` so the array can be handed to that
class element for element.

**Why this one does not write ROOT directly.** ``TScatTable`` is a user-defined
class with its own dictionary, not a stock ROOT container; a file holding one
carries the class's streamer info, which uproot cannot synthesise. So the table
is written to HDF5 here and converted by
``tools/fitqun/build_scattable_converter.sh``, which builds the dictionary from
the ``TScatTable`` sources in the fiTQun Utilities checkout and emits
``fiTQun_scattablesF_<config>.root``. The LUCiD container already carries ROOT
and HDF5, so the conversion runs there -- no separate fiTQun build needed.

The round-trip is verified: element order and all six axis definitions survive,
and ROOT's own ``GetIndex()`` agrees with the dimension-0-fastest convention
:meth:`ScatTable.flat` writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DIM_NAMES = ("zs", "rs", "t", "ast", "ct", "phi")
N_DIMS = len(DIM_NAMES)

# The three surfaces a PMT can sit on; fiTQun picks between them by PMT z.
SURFACES = ("side", "top", "bot")


@dataclass
class ScatTable:
    """A 6D uniform-binned table with ``TScatTable``'s indexing convention."""
    name: str
    nbins: tuple           # (n_zs, n_rs, n_t, n_ast, n_ct, n_phi)
    bounds: tuple          # ((lo, hi), ...) per dimension
    table: np.ndarray = field(default=None)

    def __post_init__(self):
        if len(self.nbins) != N_DIMS or len(self.bounds) != N_DIMS:
            raise ValueError(f"a scattering table has {N_DIMS} dimensions")
        # TScatTable hard-caps each axis at 50 bins (maxnbins); exceeding it
        # aborts the C++ side at load, so refuse here where it is debuggable.
        if any(n > 50 for n in self.nbins):
            raise ValueError(f"{self.name}: TScatTable allows at most 50 bins per axis")
        if self.table is None:
            self.table = np.zeros(tuple(self.nbins), dtype=np.float64)
        elif self.table.shape != tuple(self.nbins):
            raise ValueError(f"{self.name}: table shape {self.table.shape} != {tuple(self.nbins)}")

    def bin_index(self, values: Sequence[np.ndarray]) -> list:
        """Per-dimension bin indices, clamping out-of-range the way ``Fill`` does."""
        out = []
        for idim, v in enumerate(values):
            n = self.nbins[idim]
            if n <= 1:
                out.append(np.zeros(np.shape(v), dtype=np.intp))
                continue
            lo, hi = self.bounds[idim]
            idx = np.floor((np.asarray(v, dtype=np.float64) - lo) / ((hi - lo) / n))
            out.append(np.clip(idx, 0, n - 1).astype(np.intp))
        return out

    def fill(self, *values, weights=None) -> None:
        """Accumulate weights. Entries outside any axis range are dropped."""
        if len(values) != N_DIMS:
            raise ValueError(f"fill takes {N_DIMS} coordinate arrays")
        vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in values]
        inside = np.ones(vals[0].shape, dtype=bool)
        for idim, v in enumerate(vals):
            if self.nbins[idim] <= 1:
                continue
            lo, hi = self.bounds[idim]
            inside &= (v >= lo) & (v <= hi)
        idx = self.bin_index([v[inside] for v in vals])
        w = (np.ones(int(inside.sum())) if weights is None
             else np.atleast_1d(np.asarray(weights, dtype=np.float64))[inside])
        np.add.at(self.table, tuple(idx), w)

    def flat(self) -> np.ndarray:
        """The table in ``TScatTable``'s element order (dimension 0 fastest)."""
        return self.table.ravel(order="F")

    def ratio_to(self, direct: "ScatTable") -> "ScatTable":
        """Scattered/direct ratio, which is what fiTQun actually looks up."""
        if self.nbins != direct.nbins:
            raise ValueError("cannot take a ratio of tables with different binning")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(direct.table > 0, self.table / np.where(direct.table > 0, direct.table, 1.0), 0.0)
        return ScatTable(self.name, self.nbins, self.bounds, r)

    def __add__(self, other: "ScatTable") -> "ScatTable":
        if self.nbins != other.nbins or self.bounds != other.bounds:
            raise ValueError("cannot merge tables with different binning")
        return ScatTable(self.name, self.nbins, self.bounds, self.table + other.table)


def write_hdf5(path, tables: dict, metadata: Optional[dict] = None) -> Path:
    """Write ``{surface: ScatTable}`` for ``tools/fitqun/h5_to_scattable.C``."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["dim_names"] = list(DIM_NAMES)
        f.attrs["index_order"] = "dimension 0 fastest (TScatTable::GetIndex)"
        for key, value in (metadata or {}).items():
            f.attrs[key] = value
        for surface, tbl in tables.items():
            g = f.create_group(surface)
            g.attrs["name"] = tbl.name
            g.attrs["nbins"] = np.asarray(tbl.nbins, dtype=np.int32)
            g.attrs["bounds"] = np.asarray(tbl.bounds, dtype=np.float64)
            g.create_dataset("table", data=tbl.flat(), compression="gzip")
    return path


def read_hdf5(path) -> dict:
    """Inverse of :func:`write_hdf5` — mainly so tests can round-trip."""
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        for surface in f:
            g = f[surface]
            nbins = tuple(int(n) for n in g.attrs["nbins"])
            bounds = tuple(tuple(float(x) for x in b) for b in g.attrs["bounds"])
            table = np.asarray(g["table"]).reshape(nbins, order="F")
            out[surface] = ScatTable(str(g.attrs["name"]), nbins, bounds, table)
    return out
