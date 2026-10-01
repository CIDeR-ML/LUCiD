#!/usr/bin/env python3
"""Compare B3/B4 Mie-only TOF peaks before and after increasing statistics."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lucid_ballistic_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_canonical_tof import (
    TIME_EDGES_NS, histograms, map_cables, prepare_curve, section_masks,
)


def fraction_sem(samples, totals):
    fraction = samples.sum(axis=0) / totals.sum()
    sem = ((samples - totals[:, None] * fraction).std(axis=0, ddof=1)
           / np.sqrt(len(totals)) / totals.mean())
    return fraction, sem


def sample_curves(sk_path, lucid_path):
    with np.load(lucid_path) as data:
        xyz = data["xyz"].astype(float)
        cables = data["cable_id"].astype(int)
        lucid_time = data["time"].astype(float)
        lucid_weight = data["weight"].astype(float)
        lucid_sensor = data["sensor"].astype(int)
        lucid_totals = data["batch_charge"].sum(axis=1).astype(float)
        saved_batch_index = (
            data["batch_index"].astype(int)
            if "batch_index" in data else None
        )
    with np.load(sk_path) as data:
        events = np.sort(data["event_numbers"].astype(int))
        sk_event = data["pe_event"].astype(int)
        sk_time = data["pe_time"].astype(float)
        sk_cable = data["pe_pmt"].astype(int)

    sk_sensor = map_cables(cables, sk_cable)
    names, masks = section_masks(xyz)
    sk_display, _ = prepare_curve(
        sk_time, np.ones_like(sk_time), sk_sensor, xyz)
    lucid_display, _ = prepare_curve(
        lucid_time, lucid_weight, lucid_sensor, xyz)

    sk_event_index = np.searchsorted(events, sk_event)
    sk_totals = np.bincount(sk_event_index, minlength=len(events)).astype(float)

    n_batches = len(lucid_totals)
    if saved_batch_index is not None:
        batch_index = saved_batch_index
        if (len(batch_index) != len(lucid_weight)
                or batch_index.min(initial=0) < 0
                or batch_index.max(initial=-1) >= n_batches):
            raise ValueError("invalid saved LUCiD batch_index")
    else:
        # Legacy files store each batch consecutively but omit its index.
        # Recover boundaries by matching cumulative deposition weights to the
        # saved batch totals. New outputs save the exact index above.
        targets = np.cumsum(lucid_totals / n_batches)
        boundaries = np.searchsorted(np.cumsum(lucid_weight), targets) + 1
        boundaries[-1] = len(lucid_weight)
        batch_index = np.empty(len(lucid_weight), dtype=np.int16)
        for i, (start, stop) in enumerate(
                zip(np.r_[0, boundaries[:-1]], boundaries)):
            batch_index[start:stop] = i
    scaled_lucid_totals = lucid_totals / n_batches

    curves = {}
    n_bins = len(TIME_EDGES_NS) - 1
    for section in ("B3", "B4"):
        region = masks[names.index(section)]
        sk_selected = region[sk_sensor]
        lu_selected = region[lucid_sensor]
        sk_bin = np.searchsorted(TIME_EDGES_NS, sk_display[sk_selected], side="right") - 1
        lu_bin = np.searchsorted(TIME_EDGES_NS, lucid_display[lu_selected], side="right") - 1
        sk_ok = (sk_bin >= 0) & (sk_bin < n_bins)
        lu_ok = (lu_bin >= 0) & (lu_bin < n_bins)
        sk_samples = np.bincount(
            sk_event_index[sk_selected][sk_ok] * n_bins + sk_bin[sk_ok],
            minlength=len(events) * n_bins).reshape(len(events), n_bins)
        lu_samples = np.bincount(
            batch_index[lu_selected][lu_ok] * n_bins + lu_bin[lu_ok],
            weights=lucid_weight[lu_selected][lu_ok],
            minlength=n_batches * n_bins).reshape(n_batches, n_bins)
        curves[section] = (
            *fraction_sem(sk_samples, sk_totals),
            *fraction_sem(lu_samples, scaled_lucid_totals),
        )
    return curves


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sk-low", required=True, type=Path)
    ap.add_argument("--lucid-low", required=True, type=Path)
    ap.add_argument("--sk-high", required=True, type=Path)
    ap.add_argument("--lucid-high", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    datasets = [
        ("5,000 SK events / 8M LUCiD rays", sample_curves(args.sk_low, args.lucid_low)),
        ("20,000 SK events / 32M LUCiD rays", sample_curves(args.sk_high, args.lucid_high)),
    ]
    centers = 0.5 * (TIME_EDGES_NS[:-1] + TIME_EDGES_NS[1:])
    limits = {"B3": (1310, 1440), "B4": (1360, 1490)}
    high_curves = datasets[1][1]
    peak_bins = {}
    for section in ("B3", "B4"):
        shown = ((centers >= limits[section][0])
                 & (centers <= limits[section][1]))
        sk_high, _, lu_high, _ = high_curves[section]
        peak_bins[section] = np.flatnonzero(shown)[
            np.argmax(0.5 * (sk_high[shown] + lu_high[shown]))]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    for row, section in enumerate(("B3", "B4")):
        for col, (label, curves) in enumerate(datasets):
            sk, sk_sem, lu, lu_sem = curves[section]
            ax = axes[row, col]
            ax.step(centers, sk, where="mid", color="tab:blue", lw=1.8,
                    label="SKDetSim accepted PE", zorder=2)
            ax.fill_between(centers, sk-sk_sem, sk+sk_sem, step="mid",
                            color="tab:blue", alpha=0.16, linewidth=0)
            ax.step(centers, lu, where="mid", color="tab:orange", ls="--",
                    lw=1.8, label="LUCiD average response", zorder=3)
            ax.fill_between(centers, lu-lu_sem, lu+lu_sem, step="mid",
                            color="tab:orange", alpha=0.18, linewidth=0)
            peak = peak_bins[section]
            pull = (sk[peak]-lu[peak]) / np.hypot(sk_sem[peak], lu_sem[peak])
            ax.text(0.98, 0.92,
                    f"fixed peak bin {TIME_EDGES_NS[peak]:.0f}–"
                    f"{TIME_EDGES_NS[peak+1]:.0f} ns: pull = {pull:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
            ax.set(xlim=limits[section], title=f"{section}: {label}",
                   ylabel="Fraction of detector PE / 10 ns bin")
            ax.grid(alpha=0.22)
            if row == 1:
                ax.set_xlabel("Target-TOF-subtracted time [ns]")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Mie-only B3/B4 peaks: shaded bands are statistical SEM")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
