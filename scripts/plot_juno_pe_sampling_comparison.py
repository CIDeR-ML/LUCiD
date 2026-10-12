#!/usr/bin/env python
"""Plot the legacy-direct and histogram-resampled PE-time spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from lucid.juno_laser.eventwise import EventwisePeObservation
from lucid.juno_laser.sampling_comparison import (
    histogram_distance,
    summarize_observation,
)
from lucid.juno_laser.time_grid import TimeGrid


def _load_observation(handle, name, time_grid, n_sensors, n_events):
    """Load one sampled PE stream from the diagnostic HDF5 file."""
    group = handle[name]
    return EventwisePeObservation(
        event_ids=group["event_ids"][...],
        pmt_ids=group["pmt_ids"][...],
        pe_times_ns=group["times_ns"][...],
        n_events=n_events,
        n_sensors=n_sensors,
        time_grid=time_grid,
    )


def _histogram_values(times, edges, spectrum_mode):
    """Return either raw PE counts or a unit-area histogram density."""
    values, _ = np.histogram(times, bins=edges)
    if spectrum_mode == "counts":
        return values.astype(np.float64)
    if spectrum_mode == "density":
        width = np.diff(edges)
        return values / max(np.sum(values), 1) / width
    raise ValueError(f"unsupported spectrum mode: {spectrum_mode}")


def render(
    input_path,
    output_pdf,
    output_json,
    plot_bin_ns,
    *,
    x_min_ns=None,
    x_max_ns=None,
    log_y=False,
    spectrum_mode="density",
):
    """Render a two-page PDF and machine-readable comparison summary."""
    with h5py.File(input_path, "r") as handle:
        if handle.attrs.get("schema") != "lucid_pe_sampling_comparison_v1":
            raise ValueError("input is not a PE sampling comparison file")
        metadata = json.loads(handle.attrs["metadata_json"])
        time_grid = TimeGrid(
            metadata["time_min_ns"],
            metadata["time_max_ns"],
            metadata["time_bin_ns"],
        )
        direct = _load_observation(
            handle, "direct", time_grid, metadata["n_sensors"],
            metadata["n_events"],
        )
        binned = _load_observation(
            handle, "binned", time_grid, metadata["n_sensors"],
            metadata["n_events"],
        )
        raw_mass = float(np.sum(handle["raw_hits/weights"][...]))
        binned_mass = float(np.sum(handle["bin_masses"][...]))
    if direct.n_pe == 0 or binned.n_pe == 0:
        raise ValueError(
            "both samplers need non-empty PE samples; increase proposal "
            "photons, transport depth, intensity or pseudo-events"
        )
    edges = np.arange(
        time_grid.time_min_ns,
        time_grid.time_max_ns + 0.5 * plot_bin_ns,
        plot_bin_ns,
    )
    direct_spectrum = _histogram_values(
        direct.pe_times_ns, edges, spectrum_mode
    )
    binned_spectrum = _histogram_values(
        binned.pe_times_ns, edges, spectrum_mode
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    distance = histogram_distance(
        direct.pe_times_ns, binned.pe_times_ns, edges
    )
    direct_summary = summarize_observation(direct)
    binned_summary = summarize_observation(binned)
    pmt_direct = np.bincount(
        direct.pmt_ids, minlength=direct.n_sensors
    ).astype(np.float64)
    pmt_binned = np.bincount(
        binned.pmt_ids, minlength=binned.n_sensors
    ).astype(np.float64)
    pmt_direct /= max(np.sum(pmt_direct), 1.0)
    pmt_binned /= max(np.sum(pmt_binned), 1.0)
    if (x_min_ns is None) != (x_max_ns is None):
        raise ValueError("both display-window limits must be provided")
    if x_min_ns is not None and x_max_ns <= x_min_ns:
        raise ValueError("display-window upper limit must exceed lower limit")
    summary = {
        "schema": "lucid_pe_sampling_comparison_summary_v1",
        "metadata": metadata,
        "raw_expected_mass": raw_mass,
        "binned_expected_mass": binned_mass,
        "relative_mass_difference": (
            (binned_mass - raw_mass) / max(raw_mass, 1.0e-30)
        ),
        "direct": direct_summary,
        "binned": binned_summary,
        "time_spectrum_distance": distance,
        "pmt_probability_total_variation": float(
            0.5 * np.sum(np.abs(pmt_binned - pmt_direct))
        ),
        "plot_bin_ns": float(plot_bin_ns),
        "display_window_ns": (
            None if x_min_ns is None else [float(x_min_ns), float(x_max_ns)]
        ),
        "spectrum_y_scale": "log" if log_y else "linear",
        "spectrum_normalization": (
            "none_raw_pe_counts"
            if spectrum_mode == "counts"
            else "unit_area_density"
        ),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")

    positive = direct_spectrum + binned_spectrum > 0.0
    if x_min_ns is not None:
        lower, upper = float(x_min_ns), float(x_max_ns)
    elif np.any(positive):
        active_centers = centers[positive]
        lower = float(np.quantile(
            np.concatenate((direct.pe_times_ns, binned.pe_times_ns)), 0.001
        ))
        upper = float(np.quantile(
            np.concatenate((direct.pe_times_ns, binned.pe_times_ns)), 0.999
        ))
        lower = max(lower - 2.0, float(active_centers[0]))
        upper = min(upper + 2.0, float(active_centers[-1]))
    else:
        lower, upper = time_grid.time_min_ns, time_grid.time_max_ns

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
        axes[0].step(
            centers,
            direct_spectrum,
            where="mid",
            label="Direct per-hit Poisson",
        )
        axes[0].step(
            centers, binned_spectrum, where="mid",
            label="Soft-bin + categorical",
        )
        y_label = (
            "PE count per bin"
            if spectrum_mode == "counts"
            else "Normalized PE density [ns$^{-1}$]"
        )
        axes[0].set_ylabel(y_label)
        axes[0].set_title("PE-time sampling comparison (no IRF, no dark noise)")
        if log_y:
            axes[0].set_yscale("log")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        difference = binned_spectrum - direct_spectrum
        axes[1].step(centers, difference, where="mid", color="#7f3c8d")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel("PE time [ns]")
        difference_label = (
            "Binned - direct [PE/bin]"
            if spectrum_mode == "counts"
            else "Binned - direct [ns$^{-1}$]"
        )
        axes[1].set_ylabel(difference_label)
        axes[1].set_xlim(lower, upper)
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        direct_counts = np.bincount(
            direct.event_ids, minlength=direct.n_events
        )
        binned_counts = np.bincount(
            binned.event_ids, minlength=binned.n_events
        )
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
        direct_cdf = np.cumsum(direct_spectrum)
        binned_cdf = np.cumsum(binned_spectrum)
        direct_cdf /= max(direct_cdf[-1], 1.0)
        binned_cdf /= max(binned_cdf[-1], 1.0)
        axes[0].plot(centers, binned_cdf - direct_cdf)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_xlim(lower, upper)
        axes[0].set_xlabel("PE time [ns]")
        axes[0].set_ylabel("CDF(binned) - CDF(direct)")
        axes[0].set_title("Cumulative time-spectrum difference")
        axes[0].grid(alpha=0.25)
        count_edges = np.histogram_bin_edges(
            np.concatenate((direct_counts, binned_counts)), bins="auto"
        )
        axes[1].hist(
            direct_counts, bins=count_edges, histtype="step", density=True,
            label="Direct per-hit Poisson",
        )
        axes[1].hist(
            binned_counts, bins=count_edges, histtype="step", density=True,
            label="Soft-bin + categorical",
        )
        axes[1].set_xlabel("PE count per event")
        axes[1].set_ylabel("Probability density")
        axes[1].set_title("Event PE-count distribution")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
        figure.suptitle(
            "TV={:.4g}, KS(hist)={:.4g}, W1(hist)={:.4g} ns".format(
                distance["total_variation"],
                distance["kolmogorov_smirnov_histogram"],
                distance["wasserstein_histogram_ns"],
            )
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    return summary


def main():
    """Parse paths and render the PE sampling comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--plot-bin-ns", type=float, default=0.25)
    parser.add_argument("--x-min-ns", type=float)
    parser.add_argument("--x-max-ns", type=float)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument(
        "--spectrum-mode",
        choices=("density", "counts"),
        default="density",
    )
    args = parser.parse_args()
    if args.plot_bin_ns <= 0.0:
        parser.error("--plot-bin-ns must be positive")
    render(
        args.input,
        args.output_pdf,
        args.output_json,
        args.plot_bin_ns,
        x_min_ns=args.x_min_ns,
        x_max_ns=args.x_max_ns,
        log_y=args.log_y,
        spectrum_mode=args.spectrum_mode,
    )


if __name__ == "__main__":
    main()
