"""
Unified builder for the SIREN training-input .h5 lookup tables.

Two variants share one base:

  * `PhotonLookupBuilder`  → `photon_lookup_table.h5`  (Cherenkov: opening angle × s/s_max)
  * `DedxLookupBuilder`    → `dedx_lookup_table.h5`    (energy loss:    dE/dx × s/s_max)

Both read per-cell `<data-dir>/<material>/<particle>/<E>MeV/photonsim.root` files
emitted by `lucid/production/s3df_jobs/siren_inputs/`. The 3rd axis is now the
dimensionless `s / s_max(E)` with `s_max(E) = A · E^B` fit per (material,
particle); the fit comes from `PhotonSim/data/<m>/<p>/smax_fit.csv` and is
welded into the .h5's `metadata.attrs` so downstream LUCiD code (training,
inference) never has to look outside the .h5.

Console-script entry points (declared in pyproject.toml):

  * `lucid-build-photon-table`  → `main_photon`
  * `lucid-build-dedx-table`    → `main_dedx`

See `docs/SIREN_TRAINING_INPUTS.md` for the full pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import uproot
from tqdm import tqdm

logger = logging.getLogger(__name__)

FORMAT_VERSION = "2.0"
CELL_DIR_RE = re.compile(r"^(\d+)MeV$")
SMAX_FIT_COLS = (
    "A", "B", "fit_min_mev", "fit_max_mev",
    "quantile", "quantile_multiplier", "generated_at_utc",
)


# --- smax_fit.csv reader (private to the builder) ---------------------------


def _find_photonsim_dir(*, override: Optional[Path], data_dir: Path) -> Path:
    """Resolve the PhotonSim checkout that holds `data/<m>/<p>/smax_fit.csv`.

    Search order: explicit `--photonsim-dir`, $PHOTONSIM_DEV_PATH, then walk up
    from `data_dir` looking for a sibling `PhotonSim/data/` directory.
    """
    if override is not None:
        return override.resolve()
    env = os.environ.get("PHOTONSIM_DEV_PATH")
    if env:
        return Path(env).resolve()
    for parent in [data_dir.resolve(), *data_dir.resolve().parents]:
        cand = parent / "PhotonSim"
        if (cand / "data").is_dir():
            return cand.resolve()
        cand = parent.parent / "PhotonSim" if parent.parent != parent else None
        if cand is not None and (cand / "data").is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not locate a PhotonSim checkout with data/. Pass --photonsim-dir "
        "or set PHOTONSIM_DEV_PATH."
    )


def _read_smax_fit_csv(photonsim_dir: Path, material: str,
                       particle: str) -> Dict[str, Any]:
    """Return all seven `smax_fit.csv` columns as a dict.

    Raises FileNotFoundError / ValueError if the fit is missing or malformed.
    The returned dict is what gets welded verbatim into the .h5 metadata
    (with each key prefixed `smax_`).
    """
    path = photonsim_dir / "data" / material / particle / "smax_fit.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No s_max fit at {path}. Run PhotonSim's tools/smax/analyze_smax.py "
            f"for ({material!r}, {particle!r}) first."
        )
    with path.open() as fh:
        row = next(csv.DictReader(fh), None)
    if not row:
        raise ValueError(f"s_max fit at {path} is empty.")
    try:
        return {
            "A":                   float(row["A"]),
            "B":                   float(row["B"]),
            "fit_min_mev":         int(row["fit_min_mev"]),
            "fit_max_mev":         int(row["fit_max_mev"]),
            "quantile":            float(row["quantile"]),
            "quantile_multiplier": float(row["quantile_multiplier"]),
            "generated_at_utc":    str(row["generated_at_utc"]),
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"s_max fit at {path} is incomplete: {exc}") from exc


# --- Builder base ------------------------------------------------------------


class LookupTableBuilder(ABC):
    """Shared logic for the photon and dE/dx .h5 builders.

    Subclasses define the histogram name, the table-key string, the x-axis
    label/range, and the per-event "averaged units" string. Everything else —
    cell discovery, ROOT reading, schema, validation — is shared.
    """

    # ---- subclass contract -------------------------------------------------
    HIST_NAME: str         # ROOT TKey of the 2D histogram to read
    TABLE_KEY: str         # base name for h5 datasets (e.g. "photon_table")
    X_AXIS_NAME: str       # "angle" or "dedx"
    X_AXIS_BINS: int       # 500 for both current variants
    X_RANGE: Tuple[float, float]      # (0, π) or (0, 1000)
    AVERAGE_UNITS: str     # "photons/event" or "entries/event"
    TOTAL_KEY: str         # "total_photons" or "total_entries"
    DEFAULT_OUTPUT_NAME: str   # filename if --output is a directory

    DISTANCE_BINS = 500
    DISTANCE_RANGE = (0.0, 1.0)

    def __init__(self, *, data_dir: Path, material: str, particle: str,
                 photonsim_dir: Optional[Path] = None,
                 require_list: Optional[List[int]] = None):
        self.data_dir = Path(data_dir)
        self.material = material
        self.particle = particle
        self.cell_root = self.data_dir / material / particle
        if not self.cell_root.is_dir():
            raise FileNotFoundError(f"No data at {self.cell_root}")

        self.photonsim_dir = _find_photonsim_dir(
            override=photonsim_dir, data_dir=self.data_dir,
        )
        self.smax = _read_smax_fit_csv(self.photonsim_dir, material, particle)

        self.require_list = (
            sorted(set(int(e) for e in require_list)) if require_list else None
        )

        # Filled by build():
        self.energy_values: List[int] = []
        self.events_per_file: Dict[int, int] = {}
        self.raw_table: Optional[np.ndarray] = None
        self.average_table: Optional[np.ndarray] = None
        self.x_edges: Optional[np.ndarray] = None
        self.distance_edges: Optional[np.ndarray] = None

    # ---- cell discovery + per-cell read ------------------------------------

    def _discover_cells(self) -> List[int]:
        """Sorted unique list of energies with `photonsim.root` present."""
        found: Dict[int, Path] = {}
        for child in self.cell_root.iterdir():
            m = CELL_DIR_RE.match(child.name)
            if not m or not child.is_dir():
                continue
            rf = child / "photonsim.root"
            if rf.is_file() and rf.stat().st_size > 0:
                found[int(m.group(1))] = rf
            else:
                logger.warning("Skipping %s (no photonsim.root)", child)
        if self.require_list is not None:
            on_disk = set(found.keys())
            wanted = set(self.require_list)
            missing = sorted(wanted - on_disk)
            extra = sorted(on_disk - wanted)
            if missing:
                raise FileNotFoundError(
                    f"--require-list: {len(missing)} cells missing on disk: "
                    f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
                )
            if extra:
                logger.warning(
                    "--require-list: %d on-disk cells not in config list "
                    "(will be ignored): %s",
                    len(extra), extra[:8],
                )
                for e in extra:
                    found.pop(e, None)
        return sorted(found.keys())

    def _read_cell(self, energy_mev: int) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, int]:
        """Open one cell's photonsim.root and return (counts, x_edges,
        distance_edges, n_events). Raises on missing hist or tree."""
        path = self.cell_root / f"{energy_mev}MeV" / "photonsim.root"
        with uproot.open(path) as f:
            if "OpticalPhotons" not in f:
                raise KeyError(f"OpticalPhotons tree missing in {path}")
            n_events = int(f["OpticalPhotons"].num_entries)
            if self.HIST_NAME not in f:
                raise KeyError(
                    f"{self.HIST_NAME} missing in {path}. The cell was "
                    f"probably run without `/output/smax` (no normalised hist)."
                )
            hist = f[self.HIST_NAME]
            counts = np.asarray(hist.values(), dtype=np.float64)
            xedges = np.asarray(hist.axes[0].edges(), dtype=np.float64)
            yedges = np.asarray(hist.axes[1].edges(), dtype=np.float64)
        return counts, xedges, yedges, n_events

    # ---- main build pass ---------------------------------------------------

    def build(self) -> None:
        self.energy_values = self._discover_cells()
        if not self.energy_values:
            raise RuntimeError(f"No usable cells under {self.cell_root}")

        n_e = len(self.energy_values)
        self.raw_table = np.zeros(
            (n_e, self.X_AXIS_BINS, self.DISTANCE_BINS), dtype=np.float64,
        )
        self.average_table = np.zeros_like(self.raw_table)

        logger.info(
            "Building %s table: %d cells, %d..%d MeV",
            self.TABLE_KEY, n_e, self.energy_values[0], self.energy_values[-1],
        )

        for idx, energy in enumerate(tqdm(self.energy_values, desc="cells")):
            counts, xedges, yedges, n_events = self._read_cell(energy)
            if idx == 0:
                self.x_edges = xedges
                self.distance_edges = yedges
                # Hard checks — fail fast if PhotonSim wasn't in /output/smax mode.
                if not (abs(yedges[0] - 0.0) < 1e-12 and
                        abs(yedges[-1] - 1.0) < 1e-12):
                    raise ValueError(
                        f"distance edges are not [0, 1] in {energy} MeV "
                        f"({yedges[0]}..{yedges[-1]}). This is the absolute-s "
                        f"hist, not the s/s_max one — re-run Stage 1 with "
                        f"/output/smax baked into the macro."
                    )
            else:
                if not (np.allclose(xedges, self.x_edges, atol=1e-12) and
                        np.allclose(yedges, self.distance_edges, atol=1e-12)):
                    raise ValueError(
                        f"Axis edges drift between cells at E={energy} MeV"
                    )

            self.raw_table[idx] = counts
            self.events_per_file[energy] = n_events
            if n_events > 0:
                self.average_table[idx] = counts / n_events

    # ---- h5 write ----------------------------------------------------------

    def save(self, output: Path) -> Path:
        """Write the h5. If `output` is a directory, use DEFAULT_OUTPUT_NAME."""
        output = Path(output)
        if output.is_dir() or output.suffix != ".h5":
            output.mkdir(parents=True, exist_ok=True)
            h5_path = output / self.DEFAULT_OUTPUT_NAME
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            h5_path = output

        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        d_centers = 0.5 * (self.distance_edges[:-1] + self.distance_edges[1:])
        e_arr = np.asarray(self.energy_values, dtype=np.int32)
        e_edges = _energy_edges_from_centers(e_arr)

        # `events_per_file` ints: i4 is fine for ≤2.1B events/cell. Bump to i8
        # if ever needed.
        events_data = np.array(
            sorted(self.events_per_file.items()),
            dtype=[("energy", "i4"), ("events", "i4")],
        )

        with h5py.File(h5_path, "w") as f:
            data = f.create_group("data")
            data.create_dataset(
                f"{self.TABLE_KEY}_raw", data=self.raw_table,
                compression="gzip", compression_opts=4,
            )
            data.create_dataset(
                f"{self.TABLE_KEY}_average", data=self.average_table,
                compression="gzip", compression_opts=4,
            )

            coords = f.create_group("coordinates")
            coords.create_dataset("energy_values", data=e_arr)
            coords.create_dataset("energy_centers", data=e_arr)
            coords.create_dataset("energy_edges", data=e_edges)
            coords.create_dataset(f"{self.X_AXIS_NAME}_edges", data=self.x_edges)
            coords.create_dataset(f"{self.X_AXIS_NAME}_centers", data=x_centers)
            coords.create_dataset("distance_edges", data=self.distance_edges)
            coords.create_dataset("distance_centers", data=d_centers)

            meta = f.create_group("metadata")
            meta.attrs["material"] = self.material
            meta.attrs["particle"] = self.particle
            meta.attrs["distance_axis"] = "s_over_smax"
            meta.attrs["format_version"] = FORMAT_VERSION
            meta.attrs["photonsim_version"] = "unknown"  # TODO: pull from ROOT once exported
            meta.attrs["normalization"] = "average"
            meta.attrs["average_units"] = self.AVERAGE_UNITS
            meta.attrs[f"{self.X_AXIS_NAME}_bins"] = self.X_AXIS_BINS
            meta.attrs["distance_bins"] = self.DISTANCE_BINS
            meta.attrs[f"{self.X_AXIS_NAME}_range_min"] = float(self.X_RANGE[0])
            meta.attrs[f"{self.X_AXIS_NAME}_range_max"] = float(self.X_RANGE[1])
            meta.attrs["distance_range_min"] = float(self.DISTANCE_RANGE[0])
            meta.attrs["distance_range_max"] = float(self.DISTANCE_RANGE[1])
            meta.attrs["table_shape"] = np.asarray(self.average_table.shape, dtype=np.int32)
            meta.attrs[self.TOTAL_KEY] = float(self.raw_table.sum())
            for k, v in self.smax.items():
                meta.attrs[f"smax_{k}"] = v
            self._write_extra_attrs(meta)

            meta.create_dataset("events_per_file", data=events_data)

        logger.info("Wrote %s", h5_path)
        return h5_path

    def _write_extra_attrs(self, meta_group: h5py.Group) -> None:
        """Override for variant-specific attrs."""

    # ---- post-write self-check --------------------------------------------

    def validate(self, h5_path: Path) -> None:
        with h5py.File(h5_path, "r") as f:
            req = [
                f"data/{self.TABLE_KEY}_raw",
                f"data/{self.TABLE_KEY}_average",
                "coordinates/energy_values",
                "coordinates/energy_centers",
                "coordinates/energy_edges",
                f"coordinates/{self.X_AXIS_NAME}_edges",
                f"coordinates/{self.X_AXIS_NAME}_centers",
                "coordinates/distance_edges",
                "coordinates/distance_centers",
                "metadata/events_per_file",
            ]
            for k in req:
                if k not in f:
                    raise AssertionError(f"missing key: {k}")

            attrs = dict(f["metadata"].attrs)
            for k in ("smax_A", "smax_B", "smax_fit_min_mev",
                      "smax_fit_max_mev", "smax_quantile",
                      "smax_quantile_multiplier", "smax_generated_at_utc"):
                if k not in attrs:
                    raise AssertionError(f"missing metadata attr: {k}")
            if attrs.get("distance_axis") != b"s_over_smax" and \
               attrs.get("distance_axis") != "s_over_smax":
                raise AssertionError(
                    f"distance_axis != 's_over_smax' ({attrs.get('distance_axis')!r})"
                )
            if str(attrs.get("format_version", "")).strip("b'\"") != FORMAT_VERSION:
                raise AssertionError(
                    f"format_version != {FORMAT_VERSION} "
                    f"({attrs.get('format_version')!r})"
                )

            d_edges = f["coordinates/distance_edges"][:]
            if not (d_edges[0] == 0.0 and d_edges[-1] == 1.0):
                raise AssertionError(
                    f"distance_edges endpoints are not [0, 1]: "
                    f"{d_edges[0]}..{d_edges[-1]}"
                )

            energy_values = f["coordinates/energy_values"][:]
            avg = f[f"data/{self.TABLE_KEY}_average"]
            raw = f[f"data/{self.TABLE_KEY}_raw"]
            expected = (len(energy_values), self.X_AXIS_BINS, self.DISTANCE_BINS)
            if avg.shape != expected:
                raise AssertionError(
                    f"table shape {avg.shape} != expected {expected}"
                )

            events = {int(r["energy"]): int(r["events"])
                      for r in f["metadata/events_per_file"][:]}
            rng = np.random.default_rng(0)
            valid_idx = [i for i, e in enumerate(energy_values)
                         if events.get(int(e), 0) > 0]
            if not valid_idx:
                raise AssertionError("no cell has events > 0")
            i = int(rng.choice(valid_idx))
            E = int(energy_values[i])
            avg_sum = float(avg[i].sum())
            raw_sum = float(raw[i].sum())
            expect = raw_sum / events[E]
            if not np.isclose(avg_sum, expect, rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"round-trip mismatch at E={E} MeV: "
                    f"avg.sum={avg_sum}, raw.sum/events={expect}"
                )

            # Source spot-check — element-wise equality against the .root file.
            src = self.cell_root / f"{E}MeV" / "photonsim.root"
            with uproot.open(src) as g:
                src_counts = np.asarray(g[self.HIST_NAME].values(),
                                        dtype=np.float64)
            if not np.array_equal(raw[i], src_counts):
                raise AssertionError(
                    f"raw[{i}] does not match source {src}:{self.HIST_NAME}"
                )

            A, B = float(attrs["smax_A"]), float(attrs["smax_B"])
            logger.info(
                "validate(): OK. smax: A=%.6f B=%.6f  fit_min=%s  fit_max=%s  "
                "s_max@1GeV = %.1f mm",
                A, B, attrs["smax_fit_min_mev"], attrs["smax_fit_max_mev"],
                A * 1000.0 ** B,
            )

            # Cross-check the same smax row against the on-disk CSV.
            csv_smax = _read_smax_fit_csv(
                self.photonsim_dir, self.material, self.particle,
            )
            for k, v in csv_smax.items():
                got = attrs[f"smax_{k}"]
                got_norm = got.decode() if isinstance(got, bytes) else got
                if isinstance(v, float):
                    ok = np.isclose(float(got_norm), v, rtol=1e-12, atol=0)
                else:
                    ok = str(got_norm) == str(v)
                if not ok:
                    raise AssertionError(
                        f"smax_{k} drift: h5={got!r}  csv={v!r}"
                    )


# --- Concrete subclasses ----------------------------------------------------


class PhotonLookupBuilder(LookupTableBuilder):
    HIST_NAME = "PhotonHist_AngleDistanceNorm"
    TABLE_KEY = "photon_table"
    X_AXIS_NAME = "angle"
    X_AXIS_BINS = 500
    X_RANGE = (0.0, float(np.pi))
    AVERAGE_UNITS = "photons/event"
    TOTAL_KEY = "total_photons"
    DEFAULT_OUTPUT_NAME = "photon_lookup_table.h5"


class DedxLookupBuilder(LookupTableBuilder):
    HIST_NAME = "dEdxHist_DistanceNorm"
    TABLE_KEY = "dedx_table"
    X_AXIS_NAME = "dedx"
    X_AXIS_BINS = 500
    X_RANGE = (0.0, 1000.0)
    AVERAGE_UNITS = "entries/event"
    TOTAL_KEY = "total_entries"
    DEFAULT_OUTPUT_NAME = "dedx_lookup_table.h5"

    def _write_extra_attrs(self, meta_group: h5py.Group) -> None:
        meta_group.attrs["data_type"] = "dedx"


# --- Helpers ----------------------------------------------------------------


def _energy_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Build h5-friendly edges for the non-uniform energy grid.

    Interior edges are midpoints of consecutive centers; the two outer edges
    extend the end-half-bins outward symmetrically. Never used by dataset.py
    (which reads centers), but kept for downstream plotting tools.
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 0:
        return np.array([], dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mids = 0.5 * (c[:-1] + c[1:])
    left = c[0] - 0.5 * (c[1] - c[0])
    right = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.concatenate([[left], mids, [right]])


def _load_require_list(path: Path) -> List[int]:
    """Extract `energy_list_MeV` from a siren_inputs config JSON."""
    with Path(path).open() as fh:
        cfg = json.load(fh)
    if "energy_list_MeV" not in cfg:
        raise ValueError(f"{path} has no `energy_list_MeV` field")
    return [int(e) for e in cfg["energy_list_MeV"]]


# --- CLI --------------------------------------------------------------------


def _build_parser(prog: str, what: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=f"Build the {what} SIREN training-input .h5 lookup table.",
    )
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Root containing <material>/<particle>/<E>MeV/photonsim.root")
    p.add_argument("--material", required=True,
                   help="e.g. water")
    p.add_argument("--particle", required=True,
                   help="e.g. mu-, e-")
    p.add_argument("--output", required=True, type=Path,
                   help="Output .h5 file path (or a directory).")
    p.add_argument("--photonsim-dir", type=Path, default=None,
                   help="PhotonSim checkout that holds data/<m>/<p>/smax_fit.csv. "
                        "Defaults to $PHOTONSIM_DEV_PATH or a sibling of --data-dir.")
    p.add_argument("--require-list", type=Path, default=None,
                   help="siren_inputs config JSON; assert its energy_list_MeV "
                        "is a subset of the on-disk cells.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the post-write self-check.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _run(builder_cls, prog: str, what: str,
         argv: Optional[List[str]] = None) -> int:
    args = _build_parser(prog, what).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    require_list = (_load_require_list(args.require_list)
                    if args.require_list else None)

    builder = builder_cls(
        data_dir=args.data_dir, material=args.material, particle=args.particle,
        photonsim_dir=args.photonsim_dir, require_list=require_list,
    )
    builder.build()
    h5_path = builder.save(args.output)

    if not args.no_validate:
        try:
            builder.validate(h5_path)
        except AssertionError as exc:
            logger.error("validate(): FAIL — %s", exc)
            return 1
    return 0


def main_photon(argv: Optional[List[str]] = None) -> int:
    return _run(PhotonLookupBuilder, "lucid-build-photon-table",
                "photon angle vs s/s_max", argv)


def main_dedx(argv: Optional[List[str]] = None) -> int:
    return _run(DedxLookupBuilder, "lucid-build-dedx-table",
                "dE/dx vs s/s_max", argv)


if __name__ == "__main__":
    # `python -m lucid.siren.training.photonsim_data.build_tables --kind photon ...`
    raise SystemExit(
        "Invoke via the console scripts `lucid-build-photon-table` or "
        "`lucid-build-dedx-table`."
    )
