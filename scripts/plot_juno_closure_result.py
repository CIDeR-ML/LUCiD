#!/usr/bin/env python
"""Render reproducible diagnostics from one closure result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_result(path):
    """Load and validate the minimal closure-result plotting contract."""
    with Path(path).open() as stream:
        result = json.load(stream)
    if not result.get("completed") or not result.get("stages"):
        raise ValueError("closure result must be completed and contain stages")
    return result


def summarize_result(result):
    """Return compact optimization diagnostics for machine inspection."""
    stages = []
    for stage in result["stages"]:
        losses = np.asarray(stage["loss_history"], dtype=np.float64)
        mc_history = np.asarray(
            stage.get("mc_validation_history", ()), dtype=np.float64
        )
        stages.append({
            "name": stage["name"],
            "n_updates": int(len(losses) - 1),
            "initial_train_loss_per_pe": float(losses[0]),
            "best_train_loss_per_pe": float(np.min(losses)),
            "best_train_step": int(np.argmin(losses)),
            "last_train_loss_per_pe": float(losses[-1]),
            "best_mc_validation_loss_per_pe": (
                None
                if not mc_history.size
                else float(np.min(mc_history[:, 1]))
            ),
            "best_mc_validation_step": (
                None
                if not mc_history.size
                else int(mc_history[np.argmin(mc_history[:, 1]), 0])
            ),
        })
    return {
        "schema": "lucid_closure_diagnostic_summary_v1",
        "source_result_schema": result["schema"],
        "source_positions_m": result["source_positions_m"],
        "n_events": result["n_events"],
        "n_pe": result["n_pe"],
        "objective_execution": result.get("objective_execution", "fused"),
        "initial_loss_per_pe": result["initial_loss_per_pe"],
        "final_loss_per_pe": result["final_loss_per_pe"],
        "stages": stages,
    }


def _plot_loss_page(pdf, result):
    """Plot train and MC-validation loss trajectories."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    offset = 0
    for stage in result["stages"]:
        losses = np.asarray(stage["loss_history"], dtype=np.float64)
        steps = offset + np.arange(len(losses))
        axes[0].plot(steps, losses, label=stage["name"])
        best = int(np.argmin(losses))
        axes[0].scatter(steps[best], losses[best], marker="*", s=70)
        validation = np.asarray(
            stage.get("mc_validation_history", ()), dtype=np.float64
        )
        if validation.size:
            axes[1].plot(
                offset + validation[:, 0],
                validation[:, 1],
                marker="o",
                markersize=2.5,
                label=stage["name"],
            )
        offset += len(losses) - 1
    axes[0].set_title("Train objective")
    axes[1].set_title("Independent-key MC validation")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("NLL / observed PE")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Synthetic closure loss trajectories")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def _plot_numerics_page(pdf, result):
    """Plot gradient norms and objective-evaluation wall time."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    offset = 0
    for stage in result["stages"]:
        gradients = np.asarray(stage["grad_norm_history"])
        seconds = np.asarray(stage["evaluation_seconds"])
        steps = offset + np.arange(len(gradients))
        axes[0].plot(steps, gradients, label=stage["name"])
        axes[1].plot(steps, seconds, label=stage["name"])
        offset += len(gradients) - 1
    axes[0].set_ylabel("Gradient L2 norm")
    axes[1].set_ylabel("Value-and-gradient time [s]")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Optimization numerical diagnostics")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def _plot_parameter_page(pdf, result):
    """Plot all physical-parameter trajectories with synthetic truth."""
    names = tuple(result["parameter_names"])
    truth = result["truth"]
    histories = []
    for stage_index, stage in enumerate(result["stages"]):
        values = np.asarray(stage["history"], dtype=np.float64)
        if stage_index:
            values = values[1:]
        histories.append(values)
    history = np.concatenate(histories, axis=0)
    figure, axes = plt.subplots(3, 2, figsize=(10.0, 9.0))
    for index, (axis, name) in enumerate(zip(axes.flat, names)):
        axis.plot(history[:, index], color="#1f77b4")
        axis.axhline(truth[name], color="#d62728", linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Physical parameter value")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(names):]:
        axis.axis("off")
    figure.suptitle("Parameter trajectories (red dashed: truth)")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def _plot_checkpoint_page(pdf, result):
    """Compare initial, selected, last and truth parameter states."""
    stage = result["stages"][-1]
    names = tuple(result["parameter_names"])
    states = {
        "Initial": np.asarray(result["stages"][0]["history"][0]),
        "Selected": np.asarray([result["fit"][name] for name in names]),
        "Last": np.asarray(stage["last_values"]),
        "Truth": np.asarray([result["truth"][name] for name in names]),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, indices, title in (
        (axes[0], (0, 1), "LS lengths"),
        (axes[1], (2, 3, 4), "LSPhase shape coordinates"),
    ):
        x = np.arange(len(indices), dtype=np.float64)
        width = 0.18
        for state_index, (label, values) in enumerate(states.items()):
            axis.bar(
                x + (state_index - 1.5) * width,
                values[list(indices)],
                width,
                label=label,
            )
        axis.set_xticks(x, [names[index] for index in indices])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Physical parameter value")
    axes[1].legend(fontsize=8)
    figure.suptitle("Checkpoint comparison")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def render(result, output):
    """Write the multi-page vector PDF for one closure result."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _plot_loss_page(pdf, result)
        _plot_numerics_page(pdf, result)
        _plot_parameter_page(pdf, result)
        _plot_checkpoint_page(pdf, result)


def main():
    """Parse paths and create PDF plus transparent JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = load_result(args.result)
    render(result, args.output)
    summary = summarize_result(result)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
