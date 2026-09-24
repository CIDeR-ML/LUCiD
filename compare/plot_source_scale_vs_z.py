#!/usr/bin/env python
"""Scan the auto source-intensity scale over source-z simulation sets.

For every source position with both a LUCiD ``z<N>`` directory (N is z in
units of 0.01 m, e.g. ``z300`` = 3 m) and a JUNOsw ``<M>`` directory
(M is z in mm, e.g. ``3000`` = 3 m), the simulation amplitudes are
aggregated over one fixed channel selection with the same readers as
compare_sim_vs_real.py, and the correction

    s(z) = sum_C A_JUNOsw / (qe * sum_C A_LUCiD)

is recorded (qe: LUCiD-only equivalent 532 nm QE).  Only simulation data
enter the scale, so no FSMP reading is needed.  Output: per-z amplitude
caches, a CSV table and a vector PDF curve of s(z).
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare import compare_sim_vs_real as response
from compare import laser_compare_metrics as metrics

LUCID_DIR_PATTERN = re.compile(r"^z(\d+)$")
JUNOSW_DIR_PATTERN = re.compile(r"^(\d+)$")

DEFAULT_LUCID_ROOT = Path(
    "/junofs/C/liukainan/LUCiD/simulation_updated_KnownStructures"
)
DEFAULT_JUNOSW_ROOT = Path("/mnt/stage/haoch/junores/laser/532")
DEFAULT_CHANNEL_CSV = (
    DEFAULT_LUCID_ROOT
    / "compare_sim-restored/15040"
    / "channel_selection_100_tts_dark_on_irf_off_all.csv"
)

CSV_FIELDS = [
    "z_m",
    "lucid_amplitude_pe_per_event",
    "junosw_amplitude_pe_per_event",
    "lucid_over_junosw_ratio",
    "source_intensity_scale",
    "channel_count",
]


def z_m_from_lucid_dir(name):
    """Source z in meters from a LUCiD ``z<N>`` directory name.

    Input: directory basename; N encodes z in units of 0.01 m.
    Output: float z in meters, or None for names that are not a plain
    ``z<N>`` directory (suffixed variants are rejected).
    """
    match = LUCID_DIR_PATTERN.match(name)
    return int(match.group(1)) / 100.0 if match else None


def z_m_from_junosw_dir(name):
    """Source z in meters from a JUNOsw ``<M>`` directory name.

    Input: directory basename; M encodes z in millimeters.
    Output: float z in meters, or None for non-numeric names.
    """
    match = JUNOSW_DIR_PATTERN.match(name)
    return int(match.group(1)) / 1000.0 if match else None


def paired_z_directories(lucid_root, junosw_root):
    """Match LUCiD and JUNOsw z directories by source position.

    Input: the two simulation roots.
    Output: sorted list of (z_m, lucid_dir, junosw_dir) tuples present
    in both frameworks.
    """
    lucid = {}
    for entry in Path(lucid_root).iterdir():
        z_m = z_m_from_lucid_dir(entry.name)
        if entry.is_dir() and z_m is not None:
            lucid[z_m] = entry
    pairs = []
    for entry in Path(junosw_root).iterdir():
        z_m = z_m_from_junosw_dir(entry.name)
        if entry.is_dir() and z_m is not None and z_m in lucid:
            pairs.append((z_m, lucid[z_m], entry))
    return sorted(pairs)


def load_channel_selection(path):
    """Fixed channel list from a channel-selection CSV.

    Input: CSV path with a ``channel`` column.
    Output: sorted int64 ndarray of channel ids.
    """
    frame = pd.read_csv(path)
    return np.sort(frame["channel"].to_numpy(dtype=np.int64))


def lucid_amplitude(sim_dir, channels, edges, chunk_size):
    """Per-channel LUCiD PE totals and event counts.

    Input: simulation directory, selected channels, histogram edges and
    reader chunk size. Output: (total_pe, events) int64 ndarrays.
    """
    files = response.iter_sim_files(sim_dir)
    if not files:
        raise FileNotFoundError(f"No LUCiD Parquet files in {sim_dir}")
    _, _, total_pe, events = response.read_lucid_sim_histograms(
        files, edges, chunk_size, channels
    )
    return total_pe, events


def junosw_amplitude(sim_dir, channels, edges, chunk_size, group, mode):
    """Per-channel JUNOsw PE totals and event counts.

    Input: simulation directory, selected channels, histogram edges,
    reader chunk size, HDF5 group and Track/CDHit match mode.
    Output: (total_pe, events) int64 ndarrays.
    """
    files = response.iter_junosw_sim_files(sim_dir)
    if not files:
        raise FileNotFoundError(f"No JUNOsw HDF5 files in {sim_dir}")
    args = SimpleNamespace(
        junosw_sim_group=group,
        sim_chunk_size=chunk_size,
        junosw_sim_match_mode=mode,
    )
    _, _, total_pe, events = response._read_junosw_histograms(
        files, args, edges, channels
    )
    return total_pe, events


def _load_or_compute_amplitude(cache_path, compute, channels, force):
    """Return cached (total_pe, events) or compute and cache them.

    Input: cache path, zero-argument compute callable, the channel list
    the cache must match and a force-recompute flag.
    Output: (total_pe, events) ndarrays; raises on channel mismatch.
    """
    if cache_path.exists() and not force:
        with np.load(cache_path) as cache:
            if not np.array_equal(cache["channels"], channels):
                raise ValueError(f"Channel mismatch in cache {cache_path}")
            return cache["total_pe"], cache["events"]
    total_pe, events = compute()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, channels=channels, total_pe=total_pe, events=events
    )
    print(f"Wrote {cache_path}")
    return total_pe, events


def source_scale_row(z_m, lucid_dir, junosw_dir, channels, args):
    """Amplitude totals and source-intensity scale for one z point.

    Input: source position, both simulation directories, the channel
    selection and parsed CLI args (data range, bins, reader settings,
    output directory, QE factor, force flag).
    Output: dict row for the CSV table and the scale curve.
    """
    edges = np.linspace(
        args.data_range[0], args.data_range[1], args.bins + 1
    )
    lucid_total, lucid_events = _load_or_compute_amplitude(
        args.output_dir / "cache" / f"lucid_{lucid_dir.name}.npz",
        lambda: lucid_amplitude(
            lucid_dir, channels, edges, args.sim_chunk_size
        ),
        channels,
        args.force,
    )
    junosw_total, junosw_events = _load_or_compute_amplitude(
        args.output_dir / "cache" / f"junosw_{junosw_dir.name}.npz",
        lambda: junosw_amplitude(
            junosw_dir,
            channels,
            edges,
            args.sim_chunk_size,
            args.junosw_sim_group,
            args.junosw_sim_match_mode,
        ),
        channels,
        args.force,
    )
    sources = [
        {
            "simulation_total_pe": lucid_total,
            "simulation_events": lucid_events,
            "kind": np.asarray("lucid"),
        },
        {
            "simulation_total_pe": junosw_total,
            "simulation_events": junosw_events,
            "kind": np.asarray("junosw"),
        },
    ]
    totals = response.source_amplitude_totals(sources)
    qe_factors = metrics.simulation_qe_factors(
        sources, apply_lucid_qe=True, lucid_qe_eff=args.qe
    )
    scale = float(
        response.auto_source_intensity_scale(sources, qe_factors)[0]
    )
    print(
        f"z={z_m:6.2f} m: LUCiD={totals[0]:.5g} PE/event, "
        f"JUNOsw={totals[1]:.5g} PE/event, scale={scale:.6g}"
    )
    return {
        "z_m": z_m,
        "lucid_amplitude_pe_per_event": totals[0],
        "junosw_amplitude_pe_per_event": totals[1],
        "lucid_over_junosw_ratio": totals[0] / totals[1],
        "source_intensity_scale": scale,
        "channel_count": len(channels),
    }


def write_scale_csv(path, rows):
    """Write the per-z amplitude and scale table.

    Input: output CSV path and row dicts from source_scale_row.
    Output: None (writes the file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {path}")


def plot_scale_vs_z(rows, output, qe):
    """Vector PDF of the source-intensity scale versus source z.

    Input: row dicts from source_scale_row, output PDF path and the
    LUCiD-only equivalent QE factor.
    Output: None (writes the PDF).
    """
    z_m = np.array([row["z_m"] for row in rows])
    scale = np.array([row["source_intensity_scale"] for row in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(z_m, scale, "o-", color="#3969ac", label="LUCiD vs JUNOsw")
        for x, y in zip(z_m, scale):
            axis.annotate(
                f"{y:.4f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
            )
        axis.set_xlabel("Source z [m]")
        axis.set_ylabel("Source intensity scale")
        axis.set_title(
            "Auto source-intensity scale vs source z\n"
            f"(LUCiD-only equivalent QE={qe:g}, "
            f"{int(rows[0]['channel_count'])} channels)"
        )
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    print(f"Wrote {output}")


def parse_args(argv=None):
    """Parse command-line arguments.

    Input: optional argument list. Output: argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lucid-root", type=Path, default=DEFAULT_LUCID_ROOT
    )
    parser.add_argument(
        "--junosw-root", type=Path, default=DEFAULT_JUNOSW_ROOT
    )
    parser.add_argument(
        "--channel-selection", type=Path, default=DEFAULT_CHANNEL_CSV
    )
    parser.add_argument(
        "--z",
        type=float,
        nargs="*",
        default=None,
        metavar="Z_M",
        help="Restrict the scan to these source z positions in meters "
        "(default: all common z points)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("compare_sim/scale_vs_z")
    )
    parser.add_argument("--bins", type=int, default=600)
    parser.add_argument(
        "--data-range",
        type=float,
        nargs=2,
        default=(-100.0, 500.0),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument("--sim-chunk-size", type=int, default=10_000_000)
    parser.add_argument("--junosw-sim-group", default="Sim")
    parser.add_argument(
        "--junosw-sim-match-mode",
        choices=("auto", "event", "global"),
        default="auto",
    )
    parser.add_argument(
        "--qe",
        type=float,
        default=metrics.JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """Aggregate every common z point and write the CSV and PDF.

    Input: optional argument list. Output: process exit code.
    """
    args = parse_args(argv)
    channels = load_channel_selection(args.channel_selection)
    pairs = paired_z_directories(args.lucid_root, args.junosw_root)
    if not pairs:
        raise RuntimeError("No common z directories found")
    print(f"Channel selection: {len(channels)} channels")
    print("z points:", ", ".join(f"{z:g}" for z, _, _ in pairs))
    if args.z is not None:
        pairs = [
            pair
            for pair in pairs
            if any(np.isclose(pair[0], z, rtol=0.0, atol=1e-9)
                   for z in args.z)
        ]
        if not pairs:
            raise RuntimeError("No requested z point has both datasets")
    rows = [
        source_scale_row(z_m, lucid_dir, junosw_dir, channels, args)
        for z_m, lucid_dir, junosw_dir in pairs
    ]
    write_scale_csv(args.output_dir / "source_scale_vs_z.csv", rows)
    plot_scale_vs_z(rows, args.output_dir / "source_scale_vs_z.pdf", args.qe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
