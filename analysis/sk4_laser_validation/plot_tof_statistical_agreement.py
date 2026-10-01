#!/usr/bin/env python3
"""Plot canonical TOF curves and residuals using event/batch statistical SEM."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lucid_ballistic_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_canonical_tof import (
    TIME_EDGES_NS, map_cables, prepare_curve, section_masks,
)


def fraction_sem(samples, totals):
    """Delta-method SEM for a normalized fraction from independent samples."""
    fraction = samples.sum(axis=0) / totals.sum()
    centered = samples - totals[:, None] * fraction
    sem = (
        centered.std(axis=0, ddof=1)
        / np.sqrt(len(totals))
        / totals.mean()
    )
    return fraction, sem


def binned_samples(sample_index, display_time, sensor, region, n_samples):
    n_bins = len(TIME_EDGES_NS) - 1
    selected = region[sensor]
    bins = np.searchsorted(
        TIME_EDGES_NS, display_time[selected], side="right") - 1
    valid = (bins >= 0) & (bins < n_bins)
    encoded = sample_index[selected][valid] * n_bins + bins[valid]
    return encoded, selected, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sk-response", required=True, type=Path)
    ap.add_argument("--lucid", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--title", required=True)
    args = ap.parse_args()

    with np.load(args.lucid) as data:
        xyz = data["xyz"].astype(float)
        cables = data["cable_id"].astype(int)
        lu_time = data["time"].astype(float)
        lu_weight = data["weight"].astype(float)
        lu_sensor = data["sensor"].astype(int)
        lu_batch = data["batch_index"].astype(int)
        lu_batch_totals = data["batch_charge"].sum(axis=1).astype(float)
        rays_per_batch = int(data["rays_per_batch"])
    with np.load(args.sk_response) as data:
        events = np.sort(data["event_numbers"].astype(int))
        sk_event = data["pe_event"].astype(int)
        sk_time = data["pe_time"].astype(float)
        sk_cable = data["pe_pmt"].astype(int)

    n_batches = len(lu_batch_totals)
    if (len(lu_batch) != len(lu_weight)
            or lu_batch.min(initial=0) < 0
            or lu_batch.max(initial=-1) >= n_batches):
        raise ValueError("LUCiD file lacks valid exact batch indices")
    sk_event_index = np.searchsorted(events, sk_event)
    sk_totals = np.bincount(
        sk_event_index, minlength=len(events)).astype(float)
    # Saved deposition weights already include the 1/n_batches averaging.
    lu_scaled_totals = lu_batch_totals / n_batches

    sk_sensor = map_cables(cables, sk_cable)
    sk_display, sk_prompt = prepare_curve(
        sk_time, np.ones_like(sk_time), sk_sensor, xyz)
    lu_display, lu_prompt = prepare_curve(
        lu_time, lu_weight, lu_sensor, xyz)
    names, masks = section_masks(xyz)
    centers = 0.5 * (TIME_EDGES_NS[:-1] + TIME_EDGES_NS[1:])
    n_bins = len(centers)

    curves = {}
    for name, region in zip(names, masks):
        sk_encoded, _, _ = binned_samples(
            sk_event_index, sk_display, sk_sensor, region, len(events))
        sk_samples = np.bincount(
            sk_encoded, minlength=len(events) * n_bins,
        ).reshape(len(events), n_bins)

        lu_selected = region[lu_sensor]
        lu_bins = np.searchsorted(
            TIME_EDGES_NS, lu_display[lu_selected], side="right") - 1
        lu_valid = (lu_bins >= 0) & (lu_bins < n_bins)
        lu_encoded = (
            lu_batch[lu_selected][lu_valid] * n_bins + lu_bins[lu_valid])
        lu_samples = np.bincount(
            lu_encoded,
            weights=lu_weight[lu_selected][lu_valid],
            minlength=n_batches * n_bins,
        ).reshape(n_batches, n_bins)

        sk_fraction, sk_sem = fraction_sem(sk_samples, sk_totals)
        lu_fraction, lu_sem = fraction_sem(lu_samples, lu_scaled_totals)
        combined_sem = np.hypot(sk_sem, lu_sem)
        residual = sk_fraction - lu_fraction
        defined = np.isfinite(combined_sem) & (combined_sem > 0.0)
        pull = np.divide(
            residual, combined_sem,
            out=np.full_like(residual, np.nan), where=defined)
        curves[name] = {
            "sk": sk_fraction, "sk_sem": sk_sem,
            "lucid": lu_fraction, "lucid_sem": lu_sem,
            "residual": residual, "combined_sem": combined_sem,
            "pull": pull, "defined": defined,
        }

    fig, axes = plt.subplots(
        7, 2, figsize=(13.5, 15.5), sharex=True,
        gridspec_kw={"width_ratios": [1.65, 1.0], "hspace": 0.13,
                     "wspace": 0.12},
    )
    canonical_max = max(
        max(curves[name]["sk"].max(), curves[name]["lucid"].max())
        for name in names[:6])
    bottom_max = max(
        curves["Bottom"]["sk"].max(), curves["Bottom"]["lucid"].max())
    summary_sections = {}
    for row, name in enumerate(names):
        values = curves[name]
        left, right = axes[row]
        left.step(centers, values["sk"], where="mid", color="tab:blue",
                  lw=1.5, label="SKDetSim accepted PE", zorder=2)
        left.fill_between(
            centers, values["sk"] - values["sk_sem"],
            values["sk"] + values["sk_sem"], step="mid",
            color="tab:blue", alpha=0.16, linewidth=0)
        left.step(centers, values["lucid"], where="mid",
                  color="tab:orange", ls="--", lw=1.6,
                  label="LUCiD weighted average response", zorder=3)
        left.fill_between(
            centers, values["lucid"] - values["lucid_sem"],
            values["lucid"] + values["lucid_sem"], step="mid",
            color="tab:orange", alpha=0.18, linewidth=0)
        left.set_ylim(
            0, 1.15 * (bottom_max if name == "Bottom" else canonical_max))
        left.set_ylabel("fraction total PE\n/ 10 ns bin", fontsize=8)
        left.text(0.98, 0.84, name, transform=left.transAxes,
                  ha="right", va="top", fontweight="bold")
        left.grid(alpha=0.2)

        right.axhspan(-1, 1, color="0.92", zorder=0)
        for level, style in ((0, "-"), (-3, ":"), (3, ":")):
            right.axhline(level, color="0.45", lw=0.8, ls=style)
        right.plot(centers[values["defined"]],
                   values["pull"][values["defined"]], "o-",
                   color="tab:purple", ms=2.7, lw=0.8)
        right.set_ylabel("pull", fontsize=8)
        right.grid(alpha=0.18)
        finite_pull = values["pull"][values["defined"]]
        above_three = np.flatnonzero(
            values["defined"] & (np.abs(values["pull"]) > 3.0))
        summary_sections[name] = {
            "defined_bins": int(len(finite_pull)),
            "rms_pull": float(np.sqrt(np.mean(finite_pull ** 2))),
            "max_absolute_pull": float(np.max(np.abs(finite_pull))),
            "bins_above_3sigma": int(len(above_three)),
            "above_3sigma": [
                {
                    "time_bin_ns": [float(TIME_EDGES_NS[index]),
                                    float(TIME_EDGES_NS[index + 1])],
                    "pull": float(values["pull"][index]),
                    "sk_fraction": float(values["sk"][index]),
                    "lucid_fraction": float(values["lucid"][index]),
                    "combined_sem": float(values["combined_sem"][index]),
                }
                for index in above_three
            ],
        }
        right.text(
            0.98, 0.84,
            "RMS={:.2f}; max|pull|={:.2f}".format(
                summary_sections[name]["rms_pull"],
                summary_sections[name]["max_absolute_pull"]),
            transform=right.transAxes, ha="right", va="top", fontsize=8)

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("Target-TOF-subtracted time [ns]")
    axes[-1, 1].set_xlabel("Target-TOF-subtracted time [ns]")
    for ax in axes.ravel():
        ax.set_xlim(TIME_EDGES_NS[0], TIME_EDGES_NS[-1])
    fig.suptitle(
        args.title + "\nShaded bands: event/batch SEM; pull uses combined SEM\n"
        + f"Prompts: SK {sk_prompt:.1f} ns, LUCiD {lu_prompt:.1f} ns",
        fontsize=12, y=0.998)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.09, right=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    sk_total_mean = float(sk_totals.mean())
    sk_total_sem = float(sk_totals.std(ddof=1) / np.sqrt(len(sk_totals)))
    lu_total_mean = float(lu_batch_totals.mean())
    lu_total_sem = float(
        lu_batch_totals.std(ddof=1) / np.sqrt(n_batches))
    summary = {
        "sk_events": int(len(events)),
        "lucid_batches": int(n_batches),
        "lucid_rays_per_batch": rays_per_batch,
        "weighting": (
            "LUCiD bin contents are sums of deposition weights; SEM uses "
            "independent weighted batches and the normalized-ratio delta method"
        ),
        "sk_prompt_ns": float(sk_prompt),
        "lucid_prompt_ns": float(lu_prompt),
        "sk_total_pe_per_pulse": sk_total_mean,
        "sk_total_pe_sem": sk_total_sem,
        "lucid_total_pe_per_pulse": lu_total_mean,
        "lucid_total_pe_sem": lu_total_sem,
        "sections": summary_sections,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output)
    print(args.summary)


if __name__ == "__main__":
    main()
