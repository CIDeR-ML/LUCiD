#!/usr/bin/env python
"""Per-PMT relative PE counts versus theta: three-panel PDF per run.

For a given run, plot FSMP, JUNOsw and LUCiD per-10-degree-group per-PMT
mean PE counts, each normalized so the flat mean of its 18 group means is
one ("line mean = 1"), with mean +/- SEM error bars.  The three panels
are stacked in a single PDF, mirroring the reference style of
/mnt/stage/liukainan/examples/run15040_theta.png: grid lines, theta 0-180
degrees with 20-degree major ticks, per-panel title and legend.  All
plotted numbers are written to a CSV next to the PDF for inspection.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare import compare_sim as base

THETA_STEP_DEG = 10.0
DATA_ROOT = Path("/junofs/C/liukainan/LUCiD/simulation_updated_SupportRods")
JUNOSW_VERSION = "J26.5.0"


def load_theta(pmt_map, n_channels=17612):
    """Return per-channel polar angles looked up by copy_number.

    Input: PMT map parquet path and the channel count.  Output: theta in
    degrees for copy numbers 0..n_channels-1, NaN when unmapped.
    """
    table = pq.read_table(pmt_map, columns=["copy_number", "theta"])
    copy_numbers = table["copy_number"].to_numpy(zero_copy_only=False)
    theta_all = table["theta"].to_numpy(zero_copy_only=False)
    position = {int(c): i for i, c in enumerate(copy_numbers)}
    return np.asarray(
        [
            theta_all[position[c]] if c in position else np.nan
            for c in range(n_channels)
        ],
        dtype=np.float64,
    )


def group_mean_sem(counts, theta, step=THETA_STEP_DEG):
    """Return per-group per-PMT mean and SEM of channel counts.

    Input: per-channel counts and polar angles in degrees; NaN angles
    are skipped and the 180-degree boundary joins the last group.
    Output: (mean, sem, n_channels) per group; empty groups are NaN.
    """
    counts = np.asarray(counts, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    lows = np.arange(0.0, 180.0, step)
    mean = np.full(len(lows), np.nan)
    sem = np.full(len(lows), np.nan)
    number = np.zeros(len(lows), dtype=int)
    for index, lo in enumerate(lows):
        hi = lo + step
        mask = theta >= lo
        mask &= theta <= hi if hi >= 180.0 else theta < hi
        values = counts[mask]
        values = values[np.isfinite(values)]
        number[index] = len(values)
        if len(values) > 1:
            mean[index] = values.mean()
            sem[index] = values.std(ddof=1) / np.sqrt(len(values))
        elif len(values) == 1:
            mean[index] = values[0]
    return mean, sem, number


def normalize_flat(means):
    """Normalize so the flat mean of the group means is one.

    Input: per-group means (NaN groups are ignored).  Output: the
    relative curve whose unweighted mean over finite groups is one; a
    global count scale cancels.
    """
    means = np.asarray(means, dtype=np.float64)
    return means / np.nanmean(means)


def load_fsmp_counts(root, mode):
    """Load FSMP per-channel counts from the all-channel cache.

    Input: cache directory and mode ("allpe" reads fsmp_total_pe from the
    raw all-PE cache, "firstpe" reads fsmp_histogram_entries from the
    first-PE cache).  Output: counts array; a missing cache raises
    FileNotFoundError naming the run directory.
    """
    root = Path(root)
    if mode == "allpe":
        path = root / "fsmp_data_all_raw_pe_scale_all.npz"
        key = "fsmp_total_pe"
    else:
        path = root / "fsmp_data_all_firstpe_pe_scale_all.npz"
        key = "fsmp_histogram_entries"
    if not path.is_file():
        raise FileNotFoundError(
            f"FSMP {mode} cache not found: {path}; "
            "build it with the pe_scale raw-cache job for this run"
        )
    with np.load(path) as data:
        return data[key].astype(np.float64)


def load_sim_counts(root, filename):
    """Load simulation per-channel counts from an all-channel cache.

    Input: cache directory and the npz filename.  Output: the
    simulation_total_pe array as float64.
    """
    with np.load(Path(root) / filename) as data:
        return data["simulation_total_pe"].astype(np.float64)


def write_table(path, table):
    """Write the per-group counts/mean/rel table as a CSV.

    Input: CSV path and a dict of column name -> 18-value array.
    Output: no return; the CSV is written to disk.
    """
    names = list(table)
    lines = [",".join(names)]
    for index in range(18):
        lines.append(
            ",".join(f"{table[name][index]:.6g}" for name in names)
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def plot_panels(panels, run, output):
    """Write the stacked three-panel PDF and a PNG preview.

    Input: list of (title_source, legend_label, rel, sem, color, marker)
    per panel, the run number and the output PDF path.  Output: no
    return; the PDF and PNG are written to disk.
    """
    base.configure_plot_style()
    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9.5})
    centers = np.arange(THETA_STEP_DEG / 2.0, 180.0, THETA_STEP_DEG)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    for ax, (source, label, rel, sem, color, marker) in zip(axes, panels):
        ax.errorbar(
            centers, rel, yerr=sem, color=color, marker=marker, ms=5,
            lw=1.2, capsize=2.5, elinewidth=0.9, label=label,
        )
        ax.axhline(1.0, ls=":", color="0.6", lw=0.9)
        ax.grid(True, which="major", ls="-", alpha=0.35)
        ax.minorticks_on()
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        lo = np.nanmin(rel - sem)
        hi = np.nanmax(rel + sem)
        pad = 0.06 * max(hi - lo, 0.02)
        ax.set_ylim(min(lo, 1.0) - pad, max(hi, 1.0) + pad)
        ax.set_title(
            f"RUN {run}: {source}, 10-degree theta groups, mean +/- SEM"
        )
        ax.legend(loc="best")
        ax.set_ylabel("Relative PE count (line mean = 1)")
        ax.set_xlim(0.0, 180.0)
        ax.set_xticks(np.arange(0.0, 181.0, 20.0))
    axes[-1].set_xlabel(r"$\theta$ [deg]")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args(argv=None):
    """Parse command-line arguments for the three-panel theta plot.

    Input: optional argument list.  Output: parsed arguments with cache
    and output defaults derived from the run number.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=15040)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="directory holding the all-channel caches of this run",
    )
    parser.add_argument(
        "--pmt-map",
        type=Path,
        default=Path("/junofs/C/junoid_J25.7.3.pq"),
    )
    parser.add_argument(
        "--fsmp-counts",
        choices=["allpe", "firstpe"],
        default="allpe",
        help="FSMP counting mode; allpe matches the sim all-hit counts",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cache_root is None:
        args.cache_root = DATA_ROOT / "compare_sim-pe_scale" / str(args.run)
    if args.output is None:
        args.output = (
            args.cache_root
            / "raw_all_pe_independent"
            / f"run{args.run}_theta_3panel.pdf"
        )
    return args


def main(argv=None):
    """Build the curves, write the CSV table and plot the three panels.

    Input: optional argument list.  Output: no return; the CSV, PDF and
    PNG are written and key numbers are printed.
    """
    args = parse_args(argv)
    root = args.cache_root
    fsmp = load_fsmp_counts(root, args.fsmp_counts)
    junosw = load_sim_counts(root, "junosw_sim_data_all_raw_pe_scale.npz")
    lucid = load_sim_counts(root, "sim_data_all_raw_pe_scale_0.npz")
    theta = load_theta(args.pmt_map)

    fsmp_title = "FSMP" + (" (first PE)"
                           if args.fsmp_counts == "firstpe" else "")
    specs = [
        (fsmp_title, "FSMP", fsmp, "tab:orange", "x"),
        ("JUNOsw", f"JUNOsw {JUNOSW_VERSION}", junosw, "tab:green", "+"),
        ("LUCiD", "LUCiD", lucid, "tab:blue", "^"),
    ]
    panels = []
    table = {
        "theta_center_deg": np.arange(5.0, 180.0, THETA_STEP_DEG),
    }
    for title, legend, counts, color, marker in specs:
        mean, sem, _ = group_mean_sem(counts, theta)
        rel = normalize_flat(mean)
        rel_sem = sem / np.nanmean(mean)
        key = title.split()[0].lower()
        table[f"{key}_mean"] = mean
        table[f"{key}_sem"] = sem
        table[f"{key}_rel"] = rel
        panels.append((title, legend, rel, rel_sem, color, marker))
    output = Path(args.output)
    write_table(output.with_suffix(".csv"), table)
    plot_panels(panels, args.run, output)
    for title, _, rel, _, _, _ in panels:
        print(
            f"{title:18s} rel[5 deg]={rel[0]:7.4f}  "
            f"rel[95 deg]={rel[9]:7.4f}  rel[175 deg]={rel[-1]:7.4f}"
        )


if __name__ == "__main__":
    main()