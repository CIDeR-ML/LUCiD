#!/usr/bin/env python3
"""Make the standard target-TOF display for one SKDetSim/LUCiD test.

The first six panels reproduce the calibration notebook convention: Top and
five equal-height barrel bands, normalized by each curve's detector-wide
response.  A seventh, separately scaled panel shows the bottom cap because the
ballistic direct-light signal is concentrated there and is excluded from the
canonical six-panel view.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lucid_ballistic_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGET_M = np.array([-0.25, -6.945, -18.1])
LIGHT_SPEED_M_PER_NS = 0.217289524
PROMPT_BIN_NS = 2.0
DISPLAY_OFFSET_NS = 1500.0
TIME_EDGES_NS = np.arange(1100.0, 1600.0 + 10.0, 10.0)


def prompt_time(times, weights):
    """Return the center of the highest-weight 2 ns detector-wide bin."""
    valid = np.isfinite(times) & np.isfinite(weights) & (weights > 0)
    indices = np.floor(times[valid] / PROMPT_BIN_NS).astype(np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    charge = np.bincount(inverse, weights=weights[valid])
    return (unique[int(np.argmax(charge))] + 0.5) * PROMPT_BIN_NS


def section_masks(xyz):
    top = xyz[:, 2] > 18.0
    bottom = xyz[:, 2] < -18.0
    barrel = ~(top | bottom)
    barrel_z = xyz[barrel, 2]
    edges = np.linspace(barrel_z.max(), barrel_z.min(), 6)
    masks = [top]
    for i in range(5):
        upper, lower = edges[i:i + 2]
        lower_test = xyz[:, 2] > lower if i < 4 else xyz[:, 2] >= lower
        masks.append(barrel & (xyz[:, 2] <= upper) & lower_test)
    masks.append(bottom)
    return ["Top", "B1", "B2", "B3", "B4", "B5", "Bottom"], masks


def map_cables(cables, hit_cables):
    lookup = np.full(max(cables.max(), hit_cables.max()) + 1, -1, dtype=int)
    lookup[cables] = np.arange(len(cables))
    indices = lookup[hit_cables]
    if np.any(indices < 0):
        missing = np.unique(hit_cables[indices < 0])
        raise ValueError(f"SK cable IDs absent from LUCiD geometry: {missing[:10]}")
    return indices


def prepare_curve(times, weights, sensors, xyz):
    distances = np.linalg.norm(xyz[sensors] - TARGET_M, axis=1)
    residual_time = times - distances / LIGHT_SPEED_M_PER_NS
    prompt = prompt_time(residual_time, weights)
    return residual_time - prompt + DISPLAY_OFFSET_NS, prompt


def histograms(display_time, weights, sensors, masks, denominator=None):
    result = np.zeros((len(masks), len(TIME_EDGES_NS) - 1), dtype=float)
    for i, mask in enumerate(masks):
        selected = mask[sensors]
        result[i] = np.histogram(
            display_time[selected], bins=TIME_EDGES_NS, weights=weights[selected]
        )[0]
    if denominator is None:
        denominator = weights.sum()
    return result / denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sk-response", required=True, type=Path)
    parser.add_argument("--lucid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--accepted-only", action="store_true",
        help="compare SKDetSim accepted PE with LUCiD and omit digitized charge",
    )
    parser.add_argument(
        "--summary", type=Path,
        help="optional JSON file for numerical SK accepted-PE/LUCiD comparisons",
    )
    parser.add_argument(
        "--normalization", choices=("detector-total", "pe-per-pulse"),
        default="detector-total",
        help=("detector-total compares normalized TOF shape; pe-per-pulse "
              "retains the absolute yield in each 10 ns bin"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error("output exists; pass --force to replace it")

    with np.load(args.lucid) as data:
        xyz = data["xyz"].astype(float)
        cables = data["cable_id"].astype(int)
        lu_time = data["time"].astype(float)
        lu_weight = data["weight"].astype(float)
        lu_sensor = data["sensor"].astype(int)

    with np.load(args.sk_response) as data:
        sk_event_count = len(data["event_numbers"])
        pe_time = data["pe_time"].astype(float)
        pe_cable = data["pe_pmt"].astype(int)
        dig_time = data["digitized_time"].astype(float)
        dig_weight = data["digitized_charge"].astype(float)
        dig_cable = data["digitized_pmt"].astype(int)

    pe_sensor = map_cables(cables, pe_cable)
    dig_sensor = map_cables(cables, dig_cable)
    names, masks = section_masks(xyz)

    curve_inputs = [
        ("SKDetSim accepted PE", "tab:blue", "-", 2, pe_time,
         np.ones_like(pe_time), pe_sensor, sk_event_count),
    ]
    if not args.accepted_only:
        curve_inputs.append(
            ("SKDetSim digitized charge", "tab:red", "-", 1, dig_time,
             dig_weight, dig_sensor, sk_event_count)
        )
    curve_inputs.append(
        ("LUCiD average response", "tab:orange", "--", 3, lu_time,
         lu_weight, lu_sensor, 1.0)
    )

    curves = []
    for (label, color, style, zorder, times, weights, sensors,
         pulse_denominator) in curve_inputs:
        display, prompt = prepare_curve(times, weights, sensors, xyz)
        denominator = (
            weights.sum()
            if args.normalization == "detector-total"
            else pulse_denominator
        )
        curves.append({
            "label": label, "color": color, "style": style, "zorder": zorder,
            "hist": histograms(
                display, weights, sensors, masks, denominator=denominator),
            "prompt": prompt, "total": float(weights.sum()),
            "plotted_total": float(weights.sum() / denominator),
        })

    fig, axes = plt.subplots(
        7, 1, figsize=(9.0, 13.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1.15], "hspace": 0.12},
    )
    canonical_max = max(float(c["hist"][:6].max(initial=0)) for c in curves)
    bottom_max = max(float(c["hist"][6].max(initial=0)) for c in curves)
    for panel, (ax, name, mask) in enumerate(zip(axes, names, masks)):
        for curve in curves:
            values = curve["hist"][panel]
            ax.step(
                TIME_EDGES_NS, np.r_[values, values[-1]], where="post",
                label=curve["label"], color=curve["color"],
                linestyle=curve["style"], linewidth=1.6,
                zorder=curve["zorder"],
            )
        ax.axvline(DISPLAY_OFFSET_NS, color="k", linewidth=0.7, alpha=0.5)
        ax.text(
            0.02 if panel == 6 else 0.985, 0.82,
            f"{panel + 1}. {name} ({int(mask.sum()):,} PMTs)"
            + ("\nseparate vertical scale" if panel == 6 else ""),
            transform=ax.transAxes, ha="left" if panel == 6 else "right",
            va="top", fontsize=8.5,
            fontweight="bold",
        )
        ymax = bottom_max if panel == 6 else canonical_max
        ax.set_ylim(0, 1.15 * ymax if ymax > 0 else 1.0)
        ylabel = ("frac. total\np.e. / bin"
                  if args.normalization == "detector-total"
                  else "p.e. / pulse\n/ 10 ns bin")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.22)
        if panel == 6:
            ax.spines["top"].set_linewidth(1.8)
            ax.spines["top"].set_color("0.35")

    prompt_text = " | ".join(
        f"{c['label'].replace('SKDetSim ', 'SK ').replace('LUCiD average response', 'LUCiD')}: "
        f"{c['prompt']:.1f} ns" for c in curves
    )
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlim(TIME_EDGES_NS[0], TIME_EDGES_NS[-1])
    axes[-1].set_xlabel(
        "Target-TOF-subtracted time [ns] (each prompt aligned to 1500 ns)"
    )
    fig.suptitle(
        args.title + "\n" +
        ("Each curve normalized by its detector-wide total PE\n"
         if args.normalization == "detector-total"
         else "Absolute yield retained in PE per pulse per 10 ns bin\n") +
        "Canonical Top+B1–B5 panels; bottom cap added as direct-light supplement\n" +
        "Prompts before alignment — " + prompt_text,
        y=0.995, fontsize=11,
    )
    fig.subplots_adjust(top=0.925, bottom=0.065, left=0.11, right=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"output={args.output}")
    for curve in curves:
        print(
            f"{curve['label']}: prompt={curve['prompt']:.3f} ns, "
            f"detector_total={curve['total']:.6g}, "
            f"plotted_total={curve['plotted_total']:.6g}"
        )

    sk_hist = curves[0]["hist"]
    lucid_hist = curves[-1]["hist"]
    difference = sk_hist - lucid_hist
    half_l1 = float(0.5 * np.abs(difference).sum())
    metric_name = (
        "tof_histogram_tv"
        if args.normalization == "detector-total"
        else "tof_histogram_half_l1_pe_per_pulse"
    )
    summary = {
        "normalization": args.normalization,
        "time_window_ns": [float(TIME_EDGES_NS[0]), float(TIME_EDGES_NS[-1])],
        "bin_width_ns": float(np.diff(TIME_EDGES_NS)[0]),
        "sk_prompt_ns": curves[0]["prompt"],
        "lucid_prompt_ns": curves[-1]["prompt"],
        metric_name: half_l1,
        "tof_histogram_l1": float(np.abs(difference).sum()),
        "section_tv": {
            name: float(0.5 * np.abs(difference[i]).sum())
            for i, name in enumerate(names)
        },
    }
    metric_label = ("TOF histogram TV" if args.normalization == "detector-total"
                    else "TOF histogram half-L1 [PE/pulse]")
    print(f"SK accepted/LUCiD {metric_label}={half_l1:.8g}")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
