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


def load_fisher_result(path):
    """Load an optional numerical-basis Fisher result."""
    if path is None:
        return None
    with Path(path).open() as stream:
        result = json.load(stream)
    if result.get("schema") != "lucid_per_source_numerical_basis_fisher_v1":
        raise ValueError("unsupported Fisher result schema")
    return result


def calibration_rows(result, fisher=None):
    """Return truth/start/fit/error/CRB rows in physical coordinates."""
    names = tuple(result["parameter_names"])
    truth = result["truth"]
    start = result.get("start")
    if start is None:
        initial = result["stages"][0]["history"][0]
        start = dict(zip(names, initial))
    crb = None
    if fisher is not None:
        if tuple(fisher["parameter_names"]) != names:
            raise ValueError("Fisher and closure parameter names differ")
        fisher_positions = fisher.get("source_positions_m")
        if fisher_positions is None:
            fisher_positions = [
                item["position_m"] for item in fisher.get("sources", ())
            ]
        if fisher_positions != result.get("source_positions_m"):
            raise ValueError("Fisher and closure source positions differ")
        if int(fisher["n_events"]) != int(result["n_events"]):
            raise ValueError("Fisher and closure event counts differ")
        crb = np.asarray(
            fisher["total"]["crb_standard_deviation"],
            dtype=np.float64,
        )
    rows = []
    for index, name in enumerate(names):
        truth_value = float(truth[name])
        fit_value = float(result["fit"][name])
        is_log_length = index < 2
        error = (
            fit_value / truth_value - 1.0
            if is_log_length else fit_value - truth_value
        )
        rows.append({
            "parameter": name,
            "truth": truth_value,
            "start": float(start[name]),
            "fit": fit_value,
            "absolute_error": abs(fit_value - truth_value),
            "error": error,
            "error_unit": "fraction" if is_log_length else "coordinate",
            "crb_sigma": None if crb is None else float(crb[index]),
            "crb_unit": "fraction" if is_log_length else "coordinate",
            "crb_sigma_physical": (
                None if crb is None else
                float(crb[index] * truth_value)
                if is_log_length else float(crb[index])
            ),
        })
    return rows


def summarize_result(result, fisher=None):
    """Return compact optimization diagnostics for machine inspection."""
    stages = []
    for stage in result["stages"]:
        losses = np.asarray(stage["loss_history"], dtype=np.float64)
        mc_history = np.asarray(
            stage.get("mc_validation_history", ()), dtype=np.float64
        )
        stages.append({
            "name": stage["name"],
            "selected_checkpoint": stage.get("selected_checkpoint"),
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
        "truth_train_loss_per_pe": result.get(
            "truth_train_loss_per_pe"
        ),
        "truth_mc_validation_loss_per_pe": result.get(
            "truth_mc_validation_loss_per_pe"
        ),
        "parameters": calibration_rows(result, fisher),
        "crb_definition": (
            None if fisher is None else fisher.get("definition")
        ),
        "stages": stages,
    }


def _selected_checkpoint(stage):
    """Return the checkpoint used for the next stage and final result."""
    selected = stage.get("selected_checkpoint")
    if selected is not None:
        return selected
    if stage.get("best_heldout_validation_step") is not None:
        return {
            "kind": "heldout-best",
            "step": int(stage["best_heldout_validation_step"]),
            "loss_per_pe": float(stage["best_heldout_validation_loss"]),
        }
    if stage.get("best_mc_validation_step") is not None:
        return {
            "kind": "mc-validation-best",
            "step": int(stage["best_mc_validation_step"]),
            "loss_per_pe": float(stage["best_mc_validation_loss"]),
        }
    losses = np.asarray(stage["loss_history"], dtype=np.float64)
    return {
        "kind": "train-best",
        "step": int(stage.get("best_step", np.argmin(losses))),
        "loss_per_pe": float(stage.get("best_loss", np.min(losses))),
    }


def _plot_train_loss_page(pdf, result):
    """Plot train loss and its truth reference on an independent page."""
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    offset = 0
    for stage_index, stage in enumerate(result["stages"]):
        losses = np.asarray(stage["loss_history"], dtype=np.float64)
        steps = offset + np.arange(len(losses))
        label = "Train loss" if stage_index == 0 else stage["name"]
        axis.plot(steps, losses, color="#1f77b4", label=label)
        selected = _selected_checkpoint(stage)
        train_best = int(np.argmin(losses))
        marker_step = train_best
        marker_loss = float(losses[train_best])
        marker_label = "Train minimum"
        if selected["kind"] == "train-best":
            marker_step = int(selected["step"])
            marker_loss = float(selected["loss_per_pe"])
            marker_label = "Selected checkpoint"
        axis.scatter(
            offset + marker_step,
            marker_loss,
            marker="*",
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.6,
            s=120,
            zorder=5,
            label=marker_label if stage_index == 0 else None,
        )
        axis.annotate(
            f"{marker_label.lower()}\nstep={marker_step}, "
            f"loss={marker_loss:.6g}",
            (offset + marker_step, marker_loss),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
        )
        offset += len(losses) - 1
    truth_loss = result.get("truth_train_loss_per_pe")
    if truth_loss is not None:
        axis.axhline(
            truth_loss,
            color="#d62728",
            linestyle="--",
            label=f"Truth train loss = {truth_loss:.6g}",
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("NLL / observed PE")
    axis.set_title("Train objective")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def _plot_validation_loss_page(pdf, result):
    """Plot independent-key MC-validation on an independent page."""
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    offset = 0
    plotted = False
    for stage_index, stage in enumerate(result["stages"]):
        losses = np.asarray(stage["loss_history"], dtype=np.float64)
        validation = np.asarray(
            stage.get("mc_validation_history", ()), dtype=np.float64
        )
        if validation.size:
            plotted = True
            steps = offset + validation[:, 0]
            label = (
                "Independent-key MC validation"
                if stage_index == 0 else stage["name"]
            )
            axis.plot(
                steps,
                validation[:, 1],
                color="#ff7f0e",
                marker="o",
                markersize=2.5,
                label=label,
            )
            selected = _selected_checkpoint(stage)
            best = int(np.argmin(validation[:, 1]))
            marker_step = int(validation[best, 0])
            marker_loss = float(validation[best, 1])
            marker_label = "Validation minimum"
            if selected["kind"] == "mc-validation-best":
                marker_step = int(selected["step"])
                marker_loss = float(selected["loss_per_pe"])
                marker_label = "Selected checkpoint"
            axis.scatter(
                offset + marker_step,
                marker_loss,
                marker="*",
                color="#ff7f0e",
                edgecolor="black",
                linewidth=0.6,
                s=120,
                zorder=5,
                label=marker_label if stage_index == 0 else None,
            )
            axis.annotate(
                f"{marker_label.lower()}\nstep={marker_step}, "
                f"loss={marker_loss:.6g}",
                (offset + marker_step, marker_loss),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=8,
            )
        offset += len(losses) - 1
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No MC-validation evaluations",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("NLL / observed PE")
    axis.set_title("Independent-key MC validation")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend()
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


def _plot_parameter_table_page(pdf, result, fisher):
    """Render truth/start/fit/error/CRB in one inspection table."""
    rows = calibration_rows(result, fisher)
    columns = ("Parameter", "Truth", "Start", "Fit", "Err", "CRB σ")
    cells = []
    for row in rows:
        error = (
            f"{100.0 * row['error']:+.3g}%"
            if row["error_unit"] == "fraction"
            else f"{row['error']:+.6g}"
        )
        crb = "N/A"
        if row["crb_sigma"] is not None:
            crb = (
                f"{100.0 * row['crb_sigma']:.3g}%"
                if row["crb_unit"] == "fraction"
                else f"{row['crb_sigma']:.6g}"
            )
        cells.append([
            row["parameter"],
            f"{row['truth']:.6g}",
            f"{row['start']:.6g}",
            f"{row['fit']:.6g}",
            error,
            crb,
        ])
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    subtitle = (
        "CRB unavailable: run the matching Fisher job"
        if fisher is None else
        "CRB: expected-Poisson numerical-time-basis Fisher at truth"
    )
    axis.set_title(
        "Calibration parameter summary\n"
        "Err: fractional for LS lengths, coordinate difference for shapes; "
        f"{subtitle}",
        pad=16,
    )
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def _plot_lsphase_page(pdf, result):
    """Compare truth, start and fitted LSPhase angular intensities."""
    curve = result.get("lsphase_theta_curve")
    if curve is None:
        return
    theta = np.asarray(curve["theta_degrees"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    styles = (
        ("truth", "Truth", "#000000", "-"),
        ("start", "Start", "#ff7f0e", "--"),
        ("fit", "Fit", "#1f77b4", "-"),
    )
    for key, label, color, linestyle in styles:
        axis.plot(
            theta,
            curve[key],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
    axis.set_xlabel(r"Polar angle $\theta$ [degree]")
    axis.set_ylabel(r"LSPhase intensity $I(\theta)$ [a.u.]")
    axis.set_xlim(0.0, 180.0)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Diffuser LSPhase angular-intensity recovery")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def render(result, output, fisher=None):
    """Write the multi-page vector PDF for one closure result."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _plot_train_loss_page(pdf, result)
        _plot_validation_loss_page(pdf, result)
        _plot_numerics_page(pdf, result)
        _plot_parameter_page(pdf, result)
        _plot_checkpoint_page(pdf, result)
        _plot_parameter_table_page(pdf, result, fisher)
        _plot_lsphase_page(pdf, result)


def main():
    """Parse paths and create PDF plus transparent JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fisher-result", type=Path)
    args = parser.parse_args()
    result = load_result(args.result)
    fisher = load_fisher_result(args.fisher_result)
    render(result, args.output, fisher)
    summary = summarize_result(result, fisher)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
