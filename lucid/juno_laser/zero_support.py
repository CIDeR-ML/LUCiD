"""Render one exact PPP zero-support counterexample.

The diagnostic is intentionally small: it stores only the selected source,
event and PMT, their truth/fit time-basis masses, and the observed PE times.
It never changes the likelihood or inserts a numerical rate floor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _mixture_rate(
    masses,
    times_ns,
    centers_ns,
    *,
    sigma_ns,
    truncate_sigma,
    exposure,
    time_offset_ns,
    background_rate_per_ns,
):
    """Evaluate the same truncated Gaussian mixture used by the PPP."""
    masses = np.asarray(masses, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.float64)
    centers = np.asarray(centers_ns, dtype=np.float64)
    result = np.full(times.shape, background_rate_per_ns)
    normalization = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_ns)
    for mass, center in zip(masses, centers):
        if mass == 0.0:
            continue
        z = (times - time_offset_ns - center) / sigma_ns
        selected = np.abs(z) <= truncate_sigma
        result[selected] += (
            exposure
            * mass
            * normalization
            * np.exp(-0.5 * z[selected] ** 2)
        )
    return result


def render_zero_support_example(
    observation,
    truth_bin_masses,
    fit_bin_masses,
    fit_pe_rate,
    *,
    sigma_ns,
    truncate_sigma,
    background_rate_per_ns,
    output_prefix,
    label,
):
    """Write JSON and PDF for the first observed PE with zero fit rate."""
    fit_pe_rate = np.asarray(fit_pe_rate, dtype=np.float64)
    zero_indices = np.flatnonzero(fit_pe_rate <= 0.0)
    if not zero_indices.size:
        raise ValueError("zero-support diagnostic requires a zero PE rate")
    pe_index = int(zero_indices[0])
    event_id = int(observation.event_ids[pe_index])
    pmt_id = int(observation.pmt_ids[pe_index])
    pe_time_ns = float(observation.pe_times_ns[pe_index])
    exposure = float(observation.event_exposures[event_id])
    time_offset_ns = float(observation.event_time_offsets_ns[event_id])
    same_event_pmt = (
        (observation.event_ids == event_id)
        & (observation.pmt_ids == pmt_id)
    )
    observed_times = np.asarray(
        observation.pe_times_ns[same_event_pmt], dtype=np.float64
    )
    centers = np.asarray(observation.time_grid.centers, dtype=np.float64)
    truth_masses = np.asarray(
        truth_bin_masses[pmt_id], dtype=np.float64
    )
    fit_masses = np.asarray(fit_bin_masses[pmt_id], dtype=np.float64)
    time_min = float(observation.time_min_ns)
    time_max = float(observation.time_max_ns)
    full_times = np.linspace(time_min, time_max, 10001, endpoint=False)
    half_width = max(10.0, 10.0 * sigma_ns)
    zoom_min = max(time_min, pe_time_ns - half_width)
    zoom_max = min(time_max, pe_time_ns + half_width)
    zoom_times = np.linspace(zoom_min, zoom_max, 4001)

    def rates(times):
        """Return truth and fit rates on one plotting time array."""
        common = {
            "sigma_ns": sigma_ns,
            "truncate_sigma": truncate_sigma,
            "exposure": exposure,
            "time_offset_ns": time_offset_ns,
            "background_rate_per_ns": background_rate_per_ns,
        }
        return (
            _mixture_rate(truth_masses, times, centers, **common),
            _mixture_rate(fit_masses, times, centers, **common),
        )

    truth_full, fit_full = rates(full_times)
    truth_zoom, fit_zoom = rates(zoom_times)
    truth_at_pe = _mixture_rate(
        truth_masses,
        np.asarray([pe_time_ns]),
        centers,
        sigma_ns=sigma_ns,
        truncate_sigma=truncate_sigma,
        exposure=exposure,
        time_offset_ns=time_offset_ns,
        background_rate_per_ns=background_rate_per_ns,
    )[0]
    fit_at_pe = float(fit_pe_rate[pe_index])
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "lucid_ppp_zero_support_example_v1",
        "label": label,
        "pe_index": pe_index,
        "event_id": event_id,
        "pmt_id": pmt_id,
        "pe_time_ns": pe_time_ns,
        "truth_rate_at_pe_per_ns": float(truth_at_pe),
        "fit_rate_at_pe_per_ns": fit_at_pe,
        "event_exposure": exposure,
        "event_time_offset_ns": time_offset_ns,
        "sigma_ns": float(sigma_ns),
        "truncate_sigma": float(truncate_sigma),
        "kernel_half_width_ns": float(sigma_ns * truncate_sigma),
        "background_rate_per_ns": float(background_rate_per_ns),
        "n_observed_pe_same_event_pmt": int(observed_times.size),
        "truth_pmt_mass": float(np.sum(truth_masses)),
        "fit_pmt_mass": float(np.sum(fit_masses)),
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with PdfPages(prefix.with_suffix(".pdf")) as pdf:
        figure, axes = plt.subplots(3, 1, figsize=(11.0, 10.0))
        axes[0].plot(full_times, truth_full, label="Truth intensity")
        axes[0].plot(full_times, fit_full, label="Initial-fit intensity")
        axes[0].vlines(
            observed_times,
            0.0,
            max(float(np.max(truth_full)), 1.0) * 0.025,
            color="black",
            alpha=0.15,
            linewidth=0.5,
            label="Observed PE times",
        )
        axes[0].axvline(pe_time_ns, color="red", linestyle="--")
        axes[0].set_ylabel(r"$\lambda_{ekp}(t)$ [PE/ns]")
        axes[0].set_title("Full observation window (linear scale)")

        positive = np.concatenate(
            (truth_full[truth_full > 0.0], fit_full[fit_full > 0.0])
        )
        display_floor = (
            float(np.min(positive)) * 0.2 if positive.size else 1.0e-30
        )
        axes[1].plot(
            full_times,
            np.where(truth_full > 0.0, truth_full, np.nan),
            label="Truth intensity",
        )
        axes[1].plot(
            full_times,
            np.where(fit_full > 0.0, fit_full, np.nan),
            label="Initial-fit intensity",
        )
        axes[1].scatter(
            [pe_time_ns],
            [display_floor],
            color="red",
            marker="x",
            s=70,
            label="Observed PE with fit rate = 0",
            zorder=5,
        )
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=display_floor * 0.5)
        axes[1].set_ylabel(r"$\lambda_{ekp}(t)$ [PE/ns]")
        axes[1].set_title("Full observation window (log scale)")

        axes[2].plot(
            zoom_times,
            np.where(truth_zoom > 0.0, truth_zoom, np.nan),
            label="Truth intensity",
        )
        axes[2].plot(
            zoom_times,
            np.where(fit_zoom > 0.0, fit_zoom, np.nan),
            label="Initial-fit intensity",
        )
        axes[2].axvline(
            pe_time_ns,
            color="red",
            linestyle="--",
            label=f"Zero-support PE: {pe_time_ns:.6g} ns",
        )
        axes[2].scatter(
            [pe_time_ns], [display_floor], color="red", marker="x", s=70
        )
        axes[2].set_yscale("log")
        axes[2].set_ylim(bottom=display_floor * 0.5)
        axes[2].set_xlabel("Observed PE time [ns]")
        axes[2].set_ylabel(r"$\lambda_{ekp}(t)$ [PE/ns]")
        axes[2].set_title(
            f"Zoom: event {event_id}, PMT {pmt_id}; "
            f"fit $\\lambda(t_j)$ = {fit_at_pe:.3g}"
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle(
            "PPP zero-support counterexample\n"
            f"{label}; sigma={sigma_ns:g} ns, "
            f"truncation={truncate_sigma:g} sigma"
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    return summary
