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


class DenseAccumulator:
    """Sums many :class:`SparseCounts` into one dense table.

    ``SparseCounts.__add__`` re-sorts the running total for every addition, so
    merging hundreds of shards costs O(n_shards * N log N) once the total
    saturates. Scattering into a preallocated array is O(entries) with no sort,
    at the price of holding one dense table (a few hundred MB) in the merge
    process -- which has to be materialised for the output anyway.
    """

    def __init__(self, name: str, nbins: tuple, bounds: tuple):
        self.name, self.nbins, self.bounds = name, tuple(nbins), tuple(bounds)
        self.table = np.zeros(int(np.prod(nbins)), dtype=np.float64)

    def add(self, counts: "SparseCounts") -> "DenseAccumulator":
        if tuple(counts.nbins) != self.nbins or tuple(counts.bounds) != self.bounds:
            raise ValueError("cannot merge counts with different binning")
        np.add.at(self.table, counts.index, counts.count)
        return self

    def to_table(self) -> ScatTable:
        return ScatTable(self.name, self.nbins, self.bounds,
                         self.table.reshape(self.nbins))


def merge_shards(paths) -> "SampleShard":
    """Sum shard files, densifying the scattering counts as they stream in."""
    acc_s, acc_d, ang_c, ang_s = {}, {}, {}, {}
    n_photons = n_detected = n_indirect = 0
    for path in paths:
        shard = SampleShard.load(path)
        for name, c in shard.scattered.items():
            acc_s.setdefault(name, DenseAccumulator(name, c.nbins, c.bounds)).add(c)
        for name, c in shard.direct.items():
            acc_d.setdefault(name, DenseAccumulator(name, c.nbins, c.bounds)).add(c)
        for r, c in shard.angular_counts.items():
            ang_c[r] = ang_c.get(r, 0) + c
            ang_s[r] = ang_s.get(r, 0) + shard.angular_sumw2[r]
        n_photons += shard.n_photons
        n_detected += shard.n_detected
        n_indirect += shard.n_indirect
    return MergedSample(
        scattered={k: v.to_table() for k, v in acc_s.items()},
        direct={k: v.to_table() for k, v in acc_d.items()},
        angular_counts=ang_c, angular_sumw2=ang_s,
        n_photons=n_photons, n_detected=n_detected, n_indirect=n_indirect)


@dataclass
class MergedSample:
    """The summed sample: scattering tables already dense, angular histograms."""
    scattered: dict                      # surface -> ScatTable
    direct: dict                         # surface -> ScatTable
    angular_counts: dict
    angular_sumw2: dict
    n_photons: int = 0
    n_detected: int = 0
    n_indirect: int = 0

    def ratios(self) -> dict:
        return {n: self.scattered[n].ratio_to(self.direct[n]) for n in self.scattered}


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


class ShardBuilder:
    """Reduces propagated photons into a :class:`SampleShard`, chunk by chunk.

    The reduction is incremental so the per-photon arrays never have to exist
    all at once, on disk or in memory. A job propagates a group, folds it in
    here, and drops it.
    """

    def __init__(self, *, pmt_positions_m: np.ndarray, pmt_dir_z: np.ndarray,
                 det_radius_cm: float, det_halfheight_cm: float,
                 pmt_radius_cm: float, shell_radii_cm: Sequence[float],
                 shell_dr_cm: float = 50.0, n_angular_bins: int = 25):
        from .angular_driver import sensor_axes

        self.pmt_pos_cm = np.asarray(pmt_positions_m, dtype=np.float64) * M_TO_CM
        self.pmt_dir_z = np.asarray(pmt_dir_z)
        self.det_radius_cm = det_radius_cm
        self.det_halfheight_cm = det_halfheight_cm
        self.shell_radii_cm = [float(r) for r in shell_radii_cm]
        self.shell_dr_cm = shell_dr_cm
        self.n_angular_bins = n_angular_bins
        self.axes = sensor_axes(self.pmt_pos_cm, det_radius_cm=det_radius_cm,
                                det_halfheight_cm=det_halfheight_cm)
        self.nbins, self.bounds = {}, {}
        for name in scattable.SURFACES:
            self.nbins[name] = (scattable.NBINS_SIDE if name == "sidescattable"
                                else scattable.NBINS_CAP)
            self.bounds[name] = scattable.axis_bounds(
                name, det_radius_cm=det_radius_cm,
                det_halfheight_cm=det_halfheight_cm, pmt_radius_cm=pmt_radius_cm)
        self.shard = SampleShard(
            scattered={n: SparseCounts(n, self.nbins[n], self.bounds[n])
                       for n in scattable.SURFACES},
            direct={n: SparseCounts(n, self.nbins[n][:4] + (1, 1), self.bounds[n])
                    for n in scattable.SURFACES},
            angular_counts={r: np.zeros(n_angular_bins) for r in self.shell_radii_cm},
            angular_sumw2={r: np.zeros(n_angular_bins) for r in self.shell_radii_cm})

    def add(self, *, detected, sensor_id, indirect, emission_pos_m, emission_dir):
        """Fold one propagated group in. Arrays are flat, one entry per photon."""
        det = np.asarray(detected, dtype=bool).reshape(-1)
        self.shard.n_photons += int(det.size)
        if not det.any():
            return self
        sid = np.asarray(sensor_id).reshape(-1)[det]
        ind = np.asarray(indirect, dtype=bool).reshape(-1)[det]
        src_pos_cm = np.asarray(emission_pos_m).reshape(-1, 3)[det] * M_TO_CM
        src_dir = np.asarray(emission_dir).reshape(-1, 3)[det]
        self.shard.n_detected += int(det.sum())
        self.shard.n_indirect += int(ind.sum())

        surface = scattable.surface_for(self.pmt_dir_z[sid])
        coords = coordinates(src_pos_cm, src_dir, self.pmt_pos_cm[sid],
                             is_cap=surface != "sidescattable")
        for name in scattable.SURFACES:
            on = surface == name
            if not on.any():
                continue
            self.shard.scattered[name] = self.shard.scattered[name] + \
                SparseCounts.from_values(name, self.nbins[name], self.bounds[name],
                                         [c[on & ind] for c in coords])
            # The direct partner collapses the two source-direction axes; that
            # is what makes it the 4D table DivideUnnormalized4D expects.
            self.shard.direct[name] = self.shard.direct[name] + \
                SparseCounts.from_values(name, self.nbins[name][:4] + (1, 1),
                                         self.bounds[name],
                                         [c[on & ~ind] for c in coords])

        keep = ~ind                               # direct light only, as isct==0
        if keep.any():
            for r in self.shell_radii_cm:
                _, c, s2 = angular.measure(
                    src_pos_cm[keep], self.pmt_pos_cm[sid[keep]],
                    self.axes[sid[keep]], shell_r_cm=r,
                    shell_dr_cm=self.shell_dr_cm,
                    det_radius_cm=self.det_radius_cm,
                    det_halfheight_cm=self.det_halfheight_cm,
                    n_bins=self.n_angular_bins)
                self.shard.angular_counts[r] += c
                self.shard.angular_sumw2[r] += s2
        return self

    def result(self) -> SampleShard:
        return self.shard


def reduce_shard(shotgun_path, **kwargs) -> SampleShard:
    """Reduce a saved per-photon file. Kept for shards already on disk."""
    import h5py

    with h5py.File(str(shotgun_path), "r") as f:
        pp = f["per_photon"]
        if "indirect" not in pp:
            raise ValueError(
                f"{shotgun_path}: no per-photon 'indirect' flag; the direct/"
                "indirect split both tables rest on cannot be recovered")
        if "source" not in f:
            raise ValueError(f"{shotgun_path}: no source block; emission points lost")
        return ShardBuilder(**kwargs).add(
            detected=pp["detected"][:], sensor_id=pp["sensor_id"][:],
            indirect=pp["indirect"][:],
            emission_pos_m=f["source/origins"][:],
            emission_dir=f["source/directions"][:]).result()
