#!/usr/bin/env python
"""Plot all-PE FSMP-to-simulation count-ratio distributions.

By default the FSMP numerator is dark-subtracted from the cached pre-peak
sideband and exposure-deficient channels are filtered out;
``--no-fsmp-dark-subtract`` and ``--no-fsmp-min-pe-filter`` restore the
raw reference ratio.  ``--pe-per-event`` switches the ratio to per-event
averaged PE counts on both sides.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Default exposure filter for the dark-subtracted FSMP numerator.
FSMP_MIN_PE_PER_EVENT = 0.01

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare import compare_sim as base
from compare import compare_sim_vs_real as response
from compare.pe_scale_io import (
    ensure_raw_pe_scale_caches,
    fsmp_dark_counts,
    load_fsmp_raw_cache,
    load_simulation_raw_cache,
    raw_pe_scale_inputs,
)


def _groups(theta):
    """Return non-empty ten-degree theta masks plus the all-PMT mask."""
    theta = np.asarray(theta, dtype=np.float64)
    valid = np.isfinite(theta) & (theta >= 0.0) & (theta <= 180.0)
    groups = []
    for low in range(0, 180, 10):
        high = low + 10
        upper = theta <= high if high == 180 else theta < high
        mask = valid & (theta >= low) & upper
        if np.any(mask):
            close = "]" if high == 180 else ")"
            groups.append((f"theta in [{low}, {high}{close} deg", mask))
    groups.append(("all PMTs", valid))
    return groups


def _edges(values, bins):
    """Return finite linear histogram edges without an arbitrary x range."""
    finite = values[np.isfinite(values) & (values >= 0.0)]
    if not len(finite):
        return np.linspace(0.0, 1.0, bins + 1)
    low, high = float(np.min(finite)), float(np.max(finite))
    if np.isclose(low, high):
        margin = max(0.05, 0.05 * abs(low))
        low, high = max(0.0, low - margin), high + margin
    return np.linspace(low, high, bins + 1)


def plot_raw_pe_scale(data, run, output_dir, bins):
    """Write one theta-binned PE-scale PDF for every simulation source.

    Input: all-PE count-ratio arrays returned by ``raw_pe_scale_inputs``.
    When ``pe_scale_dark`` is present the dark-subtracted ratio is drawn.
    Output: list of written PDF paths.
    """
    base.configure_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    colors = {"lucid": "tab:orange", "junosw": "tab:green"}
    dark = "pe_scale_dark" in data
    per_event = bool(np.asarray(data.get("pe_per_event", False)))
    tag = "darksub" if dark else "raw"
    if per_event:
        tag += "_pevt"
    for index, (label, kind) in enumerate(
        zip(data["simulation_labels"], data["simulation_kinds"])
    ):
        output = output_dir / f"pe_scale_{tag}_all_pe_{kind}_{index + 1}.pdf"
        color = colors.get(str(kind).lower(), "tab:red")
        with PdfPages(output) as pdf:
            for title, mask in _groups(data["theta"]):
                if dark:
                    scale = data["pe_scale_dark"][index, mask]
                    good = data["pe_scale_dark_source_good_mask"][index]
                    good = good[mask]
                else:
                    scale = data["pe_scale"][index, mask]
                    good = data["pe_scale_source_good_mask"][index, mask]
                good &= np.isfinite(scale)
                finite = scale[good]
                mean = float(np.mean(finite)) if len(finite) else np.nan
                std = float(np.std(finite)) if len(finite) else np.nan
                fig, axis = plt.subplots(figsize=base.FIGURE_SIZE)
                axis.hist(
                    finite,
                    bins=_edges(finite, bins),
                    histtype="step",
                    linewidth=1.8,
                    color=color,
                )
                if len(finite):
                    axis.axvline(mean, color=color, linestyle="--")
                accepted = np.count_nonzero(good)
                note = f"N={accepted}\nMean={mean:.6g}\nStd={std:.6g}"
                if dark and data.get("fsmp_min_pe_per_event", 0.0) > 0.0:
                    cut = data["fsmp_min_pe_per_event"]
                    note += f"\ncut: corr>={cut:g} PE/evt"
                axis.text(
                    0.98,
                    0.98,
                    note,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize="x-small",
                    bbox={"facecolor": "white", "edgecolor": "0.8"},
                )
                if per_event and dark:
                    xlabel = (
                        r"$\tilde\kappa_p=[(N_{\mathrm{FSMP},p}"
                        r"-N_{\mathrm{dark},p})/E_{\mathrm{FSMP}}]"
                        r"/(q_sN_{\mathrm{MC},p}/E_{\mathrm{MC}})$"
                    )
                elif per_event:
                    xlabel = (
                        r"$\tilde\kappa_p=[N_{\mathrm{FSMP},p}"
                        r"/E_{\mathrm{FSMP}}]"
                        r"/(q_sN_{\mathrm{MC},p}/E_{\mathrm{MC}})$"
                    )
                elif dark:
                    xlabel = (
                        r"$\kappa_p=(N_{\mathrm{FSMP},p}"
                        r"-N_{\mathrm{dark},p})/(q_sN_{\mathrm{MC},p})$"
                    )
                else:
                    xlabel = (
                        r"$\kappa_p=N_{\mathrm{FSMP},p}"
                        r"/(q_sN_{\mathrm{MC},p})$"
                    )
                axis.set(
                    title=f"RUN {run}: {title}\nFSMP / {label} PE scale",
                    xlabel=xlabel,
                    ylabel="Channels",
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Wrote {output}")
        outputs.append(output)
    return outputs


def parse_args(argv=None):
    """Parse raw PE-scale cache, build and PDF settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--fsmp-cache", type=Path, required=True)
    parser.add_argument(
        "--simulation-cache", type=Path, nargs="+", required=True
    )
    parser.add_argument("--pmt-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--lucid-qe-effective", type=float, default=0.097)
    parser.add_argument(
        "--no-fsmp-dark-subtract",
        dest="fsmp_dark_subtract",
        action="store_false",
        help="keep the raw reference numerator (no dark subtraction)",
    )
    parser.add_argument(
        "--dark-margin-ns",
        type=float,
        default=30.0,
        help="safety margin before the peak for the dark sideband",
    )
    parser.add_argument(
        "--no-fsmp-min-pe-filter",
        dest="fsmp_min_pe_filter",
        action="store_false",
        help="disable the default minimal FSMP exposure channel filter",
    )
    parser.add_argument(
        "--pe-per-event",
        dest="pe_per_event",
        action="store_true",
        help="plot the per-event averaged PE ratio instead of run totals",
    )
    parser.add_argument(
        "--build-missing",
        action="store_true",
        help="build invalid or absent all-channel raw caches before plotting",
    )
    parser.add_argument("--lucid-cache", type=Path)
    parser.add_argument("--junosw-cache", type=Path)
    parser.add_argument("--fsmp-dir", type=Path)
    parser.add_argument("--lucid-sim-dir", type=Path)
    parser.add_argument("--junosw-sim-dir", type=Path)
    parser.add_argument(
        "--pmt",
        type=Path,
        default=Path(response.DEFAULT_PMT_FILE),
    )
    parser.add_argument(
        "--rtraw-dir",
        type=Path,
        default=Path(response.DEFAULT_RTRAW_DIR),
    )
    parser.add_argument("--sim-chunk-size", type=int, default=10_000_000)
    parser.add_argument("--junosw-sim-group", default="Sim")
    parser.add_argument(
        "--junosw-sim-match-mode",
        choices=("auto", "event", "global"),
        default="auto",
    )
    parser.add_argument("--junosw-sim-file-glob", default="*.h5")
    parser.add_argument("--lucid-label", default="LUCiD (basis)")
    parser.add_argument("--junosw-label", default="JUNOsw (J26.5.0)")
    parser.add_argument("--fsmp-file-limit", type=int, default=None)
    parser.add_argument("--lucid-file-limit", type=int, default=None)
    parser.add_argument("--junosw-file-limit", type=int, default=None)
    parser.add_argument("--fsmp-first-pe", action="store_true")
    args = parser.parse_args(argv)
    if args.bins <= 0:
        parser.error("--bins must be positive")
    if not np.isfinite(args.dark_margin_ns) or args.dark_margin_ns < 0.0:
        parser.error("--dark-margin-ns must be a non-negative number")
    for name in (
        "sim_chunk_size",
        "fsmp_file_limit",
        "lucid_file_limit",
        "junosw_file_limit",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.build_missing:
        if args.lucid_cache is None:
            parser.error("--lucid-cache is required with --build-missing")
        if args.junosw_cache is None:
            parser.error("--junosw-cache is required with --build-missing")
    return args


def main(argv=None):
    """Load raw all-channel caches and write PE-scale PDFs."""
    args = parse_args(argv)
    if args.build_missing:
        ensure_raw_pe_scale_caches(args)
    fsmp = load_fsmp_raw_cache(
        args.fsmp_cache,
        args.pmt_map,
        with_histogram=args.fsmp_dark_subtract,
    )
    sources = [
        load_simulation_raw_cache(path, fsmp["channels"])
        for path in args.simulation_cache
    ]
    min_pe_per_event = (
        FSMP_MIN_PE_PER_EVENT if args.fsmp_min_pe_filter else 0.0
    )
    dark_pe = None
    if args.fsmp_dark_subtract:
        dark_pe, dark_info = fsmp_dark_counts(
            fsmp["total_pe"],
            fsmp["histogram"],
            fsmp["histogram_edges"],
            margin_ns=args.dark_margin_ns,
        )
        print(
            f"FSMP dark sideband: global peak "
            f"{dark_info['global_peak_ns']:.1f} ns, "
            f"margin {dark_info['margin_ns']:.1f} ns, "
            f"low-count channels {dark_info['low_count_channels']}, "
            f"no-sideband channels {dark_info['no_window_channels']}"
        )
    if args.fsmp_dark_subtract and args.fsmp_min_pe_filter:
        print(
            f"FSMP channel filter: corrected numerator >= "
            f"{FSMP_MIN_PE_PER_EVENT:g} PE/event"
        )
    data = raw_pe_scale_inputs(
        fsmp,
        sources,
        args.lucid_qe_effective,
        dark_pe=dark_pe,
        min_pe_per_event=min_pe_per_event,
        per_event=args.pe_per_event,
    )
    plot_raw_pe_scale(data, args.run, args.output_dir, args.bins)


if __name__ == "__main__":
    main()
