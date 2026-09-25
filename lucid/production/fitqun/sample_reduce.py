"""One pass over a propagated shard: the angular response and the indirect-light tables.

Both tables come from the same isotropic 3 MeV electron sample -- the reference
reuses one production for them -- so a shard is read once and fed to both
reductions. The alternative is two passes over the same terabytes.

Counts are kept **sparse**. The three 6D scattering tables span ~110 million
bins, of which one job's few million detected photons can occupy at most a few
per cent; the dense form would be 881 MB per shard against ~64 MB here, and the
shards only have to survive until the merge. The merged table is densified once
at the end, which is where fiTQun's fixed-size array is actually needed.

Equivalence with the dense path is not argued, it is tested: filling a
:class:`~lucid.production.fitqun.scattable.ScatTable` directly and densifying
the sparse counts must give the same array, bin for bin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import angular, scattable
from .scattable import ScatTable
from .scattable_driver import coordinates

M_TO_CM = 100.0


@dataclass
class SparseCounts:
    """The occupied bins of one :class:`ScatTable`, as (flat index, count).

    ``index`` is a C-order ``ravel_multi_index`` over ``nbins`` -- an internal
    convention only. ``TScatTable``'s element order is applied by
    :meth:`ScatTable.flat` at write time, so densifying here and letting that
    method do its own ordering keeps the file contract in exactly one place.
    """
    name: str
    nbins: tuple
    bounds: tuple
    index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    @classmethod
    def from_values(cls, name: str, nbins: tuple, bounds: tuple,
                    values: Sequence[np.ndarray],
                    weights: Optional[np.ndarray] = None) -> "SparseCounts":
        """Bin coordinates the way :meth:`ScatTable.fill` does, but sparsely."""
        proto = ScatTable(name, nbins, bounds)
        vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in values]
        # Same out-of-range rule as fill: an entry outside ANY axis is dropped,
        # not clamped. Axes collapsed to one bin accept everything.
        inside = np.ones(vals[0].shape, dtype=bool)
        for idim, v in enumerate(vals):
            if nbins[idim] <= 1:
                continue
            lo, hi = bounds[idim]
            inside &= (v >= lo) & (v <= hi)
        idx = proto.bin_index([v[inside] for v in vals])
        w = (np.ones(int(inside.sum()), dtype=np.float64) if weights is None
             else np.atleast_1d(np.asarray(weights, dtype=np.float64))[inside])
        flat = np.ravel_multi_index(tuple(idx), tuple(nbins)).astype(np.int64)
        return cls(name, tuple(nbins), tuple(bounds), *_coalesce(flat, w))

    def __add__(self, other: "SparseCounts") -> "SparseCounts":
        if self.nbins != other.nbins or self.bounds != other.bounds:
            raise ValueError("cannot merge sparse counts with different binning")
        idx = np.concatenate([self.index, other.index])
        cnt = np.concatenate([self.count, other.count])
        return SparseCounts(self.name, self.nbins, self.bounds, *_coalesce(idx, cnt))

    def to_dense(self) -> ScatTable:
        table = np.zeros(int(np.prod(self.nbins)), dtype=np.float64)
        table[self.index] = self.count
        return ScatTable(self.name, self.nbins, self.bounds,
                         table.reshape(tuple(self.nbins)))

    @property
    def occupancy(self) -> float:
        return self.index.size / float(np.prod(self.nbins))


def _coalesce(index: np.ndarray, count: np.ndarray):
    """Sum duplicate indices so a shard holds one row per occupied bin."""
    if index.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    uniq, inv = np.unique(index, return_inverse=True)
    return uniq.astype(np.int64), np.bincount(inv, weights=count).astype(np.float64)


@dataclass
class SampleShard:
    """What one job contributes: sparse scattering counts plus angular histograms."""
    scattered: dict                      # surface -> SparseCounts
    direct: dict                         # surface -> SparseCounts
    angular_counts: dict                 # shell_r_cm -> (n_bins,)
    angular_sumw2: dict
    n_photons: int = 0
    n_detected: int = 0
    n_indirect: int = 0

    def __add__(self, other: "SampleShard") -> "SampleShard":
        return SampleShard(
            scattered={k: self.scattered[k] + other.scattered[k] for k in self.scattered},
            direct={k: self.direct[k] + other.direct[k] for k in self.direct},
            angular_counts={k: self.angular_counts[k] + other.angular_counts[k]
                            for k in self.angular_counts},
            angular_sumw2={k: self.angular_sumw2[k] + other.angular_sumw2[k]
                           for k in self.angular_sumw2},
            n_photons=self.n_photons + other.n_photons,
            n_detected=self.n_detected + other.n_detected,
            n_indirect=self.n_indirect + other.n_indirect,
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {"n_photons": self.n_photons, "n_detected": self.n_detected,
               "n_indirect": self.n_indirect,
               "surfaces": np.array(list(self.scattered), dtype=object),
               "shells": np.array(list(self.angular_counts), dtype=np.float64)}
        for kind, d in (("sca", self.scattered), ("dir", self.direct)):
            for s, c in d.items():
                out[f"{kind}:{s}:index"] = c.index
                out[f"{kind}:{s}:count"] = c.count
                out[f"{kind}:{s}:nbins"] = np.asarray(c.nbins)
                out[f"{kind}:{s}:bounds"] = np.asarray(c.bounds, dtype=np.float64)
        for r, c in self.angular_counts.items():
            out[f"ang:{r:g}:counts"] = c
            out[f"ang:{r:g}:sumw2"] = self.angular_sumw2[r]
        np.savez_compressed(path, **out)
        return path

    @classmethod
    def load(cls, path) -> "SampleShard":
        z = np.load(path, allow_pickle=True)
        surfaces = [str(s) for s in z["surfaces"]]
        shells = [float(r) for r in z["shells"]]

        def _counts(kind, s):
            return SparseCounts(s, tuple(int(n) for n in z[f"{kind}:{s}:nbins"]),
                                tuple(tuple(b) for b in z[f"{kind}:{s}:bounds"]),
                                z[f"{kind}:{s}:index"], z[f"{kind}:{s}:count"])

        return cls(
            scattered={s: _counts("sca", s) for s in surfaces},
            direct={s: _counts("dir", s) for s in surfaces},
            angular_counts={r: z[f"ang:{r:g}:counts"] for r in shells},
            angular_sumw2={r: z[f"ang:{r:g}:sumw2"] for r in shells},
            n_photons=int(z["n_photons"]), n_detected=int(z["n_detected"]),
            n_indirect=int(z["n_indirect"]),
        )


def reduce_shard(shotgun_path, *, pmt_positions_m: np.ndarray, pmt_dir_z: np.ndarray,
                 det_radius_cm: float, det_halfheight_cm: float, pmt_radius_cm: float,
                 shell_radii_cm: Sequence[float], shell_dr_cm: float = 50.0,
                 n_angular_bins: int = 25) -> SampleShard:
    """Read one shotgun output and reduce it to a :class:`SampleShard`."""
    import h5py
    from .angular_driver import sensor_axes

    with h5py.File(str(shotgun_path), "r") as f:
        pp = f["per_photon"]
        detected = pp["detected"][:].reshape(-1)
        sensor_id = pp["sensor_id"][:].reshape(-1)
        if "indirect" not in pp:
            raise ValueError(
                f"{shotgun_path}: no per-photon 'indirect' flag; the direct/"
                "indirect split both tables rest on cannot be recovered")
        indirect = pp["indirect"][:].reshape(-1)
        if "source" not in f:
            raise ValueError(f"{shotgun_path}: no source block; emission points lost")
        emission_m = f["source/origins"][:].reshape(-1, 3)
        emit_dir = f["source/directions"][:].reshape(-1, 3)

    n_photons = int(detected.size)
    det = detected.astype(bool)
    sid = sensor_id[det]
    ind = indirect[det].astype(bool)
    src_pos_cm = emission_m[det] * M_TO_CM
    src_dir = emit_dir[det]
    pmt_pos_cm = np.asarray(pmt_positions_m, dtype=np.float64) * M_TO_CM

    # --- indirect-light tables ------------------------------------------------
    surface = scattable.surface_for(pmt_dir_z[sid])
    is_cap = surface != "sidescattable"
    coords = coordinates(src_pos_cm, src_dir, pmt_pos_cm[sid], is_cap=is_cap)

    scattered, direct = {}, {}
    for name in scattable.SURFACES:
        nb = (scattable.NBINS_SIDE if name == "sidescattable"
              else scattable.NBINS_CAP)
        bd = scattable.axis_bounds(name, det_radius_cm=det_radius_cm,
                                   det_halfheight_cm=det_halfheight_cm,
                                   pmt_radius_cm=pmt_radius_cm)
        on = surface == name
        scattered[name] = SparseCounts.from_values(
            name, nb, bd, [c[on & ind] for c in coords])
        # The direct partner collapses the two source-direction axes; that is
        # what makes it the 4D table DivideUnnormalized4D expects.
        direct[name] = SparseCounts.from_values(
            name, nb[:4] + (1, 1), bd, [c[on & ~ind] for c in coords])

    # --- angular response -----------------------------------------------------
    axes = sensor_axes(pmt_pos_cm, det_radius_cm=det_radius_cm,
                       det_halfheight_cm=det_halfheight_cm)
    ang_c, ang_s = {}, {}
    keep = ~ind                                   # direct light only, as isct==0
    for r in shell_radii_cm:
        _, c, s2 = angular.measure(
            src_pos_cm[keep], pmt_pos_cm[sid[keep]], axes[sid[keep]],
            shell_r_cm=float(r), shell_dr_cm=shell_dr_cm,
            det_radius_cm=det_radius_cm, det_halfheight_cm=det_halfheight_cm,
            n_bins=n_angular_bins)
        ang_c[float(r)] = c
        ang_s[float(r)] = s2

    return SampleShard(scattered=scattered, direct=direct,
                       angular_counts=ang_c, angular_sumw2=ang_s,
                       n_photons=n_photons, n_detected=int(det.sum()),
                       n_indirect=int(ind.sum()))
