#!/usr/bin/env python3
"""Plot warmed timings for the SK PMT stability and routing changes."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lucid_ballistic_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    data = json.loads(args.summary.read_text())
    pre = np.r_[data["pre_a"]["all_seconds"][1:],
                data["pre_b"]["all_seconds"][1:]]
    post = np.r_[data["post_a"]["all_seconds"][1:],
                 data["post_b"]["all_seconds"][1:]]
    series = [pre, post]
    labels = ["Before float32 fix\n(a9672cb)",
              "Stable local solve\n(a268c31)"]
    colors = ["tab:blue", "tab:orange"]
    if "wall_selected_forward" in data:
        series.append(np.asarray(
            data["wall_selected_forward"]["warm_seconds"]))
        labels.append("Stable solve + selected-PMT\nforward path")
        colors.append("tab:green")

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    rng = np.random.default_rng(20261001)
    for i, (values, color) in enumerate(zip(series, colors)):
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(i + jitter, values, color=color, s=28, alpha=0.75, zorder=2)
        median = np.median(values)
        ax.hlines(median, i - 0.25, i + 0.25, color="black", lw=2.2, zorder=3)
        ax.text(i, median + 0.08, f"median {median:.3f} s", ha="center", fontsize=10)
    if "wall_selected_forward" in data:
        speedup = data["wall_selected_forward"]["speedup_vs_stable"]
        subtitle = f"Selected-PMT forward path is {speedup:.2f}x faster than stable baseline"
    else:
        increase = data["comparison"]["runtime_increase_percent"]
        subtitle = f"Stable-solver runtime increase = {increase:.1f}%"
    ax.set(
        xticks=np.arange(len(series)),
        xticklabels=labels,
        ylabel="Seconds per 500,000 rays",
        title=("Stable SK PMT intersection runtime\n"
               + subtitle),
        xlim=(-0.55, len(series) - 0.45),
    )
    ax.grid(axis="y", alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
