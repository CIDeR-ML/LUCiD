"""Tests for closure-result diagnostics and rendering."""

import json

from scripts.plot_juno_closure_result import (
    _plot_train_loss_page,
    _plot_validation_loss_page,
    calibration_rows,
    load_result,
    render,
    summarize_result,
)


def _result():
    """Return a minimal completed two-parameter-like closure mapping."""
    names = [
        "scat_ls",
        "abs_ls",
        "lsphase_shape_0",
        "lsphase_shape_1",
        "lsphase_shape_2",
    ]
    return {
        "schema": "lucid_full_geometry_eventwise_closure_v1",
        "completed": True,
        "source_positions_m": [[0.0, 0.0, 0.0]],
        "n_events": 2,
        "n_pe": [10],
        "objective_execution": "per-source",
        "initial_loss_per_pe": 2.0,
        "final_loss_per_pe": 1.5,
        "truth_train_loss_per_pe": 1.4,
        "truth_mc_validation_loss_per_pe": 1.7,
        "parameter_names": names,
        "truth": dict(zip(names, [87.0, 85.0, 0.0, 0.0, 0.0])),
        "start": dict(zip(names, [80.0, 80.0, 0.2, 0.1, -0.2])),
        "fit": dict(zip(names, [86.0, 84.0, 0.1, -0.1, 0.2])),
        "lsphase_theta_curve": {
            "theta_degrees": [0.0, 90.0, 180.0],
            "truth": [0.7, 0.5, 0.3],
            "start": [0.5, 0.5, 0.5],
            "fit": [0.68, 0.49, 0.31],
        },
        "stages": [{
            "name": "joint",
            "selected_checkpoint": {
                "kind": "mc-validation-best",
                "step": 1,
                "loss_per_pe": 1.6,
            },
            "history": [
                [80.0, 80.0, 0.2, 0.1, -0.2],
                [86.0, 84.0, 0.1, -0.1, 0.2],
            ],
            "loss_history": [2.0, 1.5],
            "grad_norm_history": [0.4, 0.2],
            "evaluation_seconds": [1.0, 0.5],
            "mc_validation_history": [[0, 2.1], [1, 1.6]],
            "best_loss": 1.5,
            "best_step": 1,
            "best_mc_validation_loss": 1.6,
            "best_mc_validation_step": 1,
            "last_values": [85.0, 83.0, 0.2, -0.2, 0.3],
        }],
    }


def test_summary_reports_train_and_validation_best_steps(tmp_path):
    """The JSON summary preserves the optimization selection evidence."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result()))
    summary = summarize_result(load_result(path))
    stage = summary["stages"][0]
    assert stage["best_train_step"] == 1
    assert stage["best_mc_validation_step"] == 1
    assert stage["n_updates"] == 1
    assert summary["truth_train_loss_per_pe"] == 1.4
    assert summary["truth_mc_validation_loss_per_pe"] == 1.7


def test_loss_curves_use_separate_pages_and_truth_only_on_train():
    """Train and validation keep independent scales and truth semantics."""
    class RecordingPdf:
        """Capture figures passed to one PDF page writer."""

        def __init__(self):
            self.figures = []

        def savefig(self, figure):
            self.figures.append(figure)

    train_pdf = RecordingPdf()
    validation_pdf = RecordingPdf()
    _plot_train_loss_page(train_pdf, _result())
    _plot_validation_loss_page(validation_pdf, _result())
    assert len(train_pdf.figures) == 1
    assert len(validation_pdf.figures) == 1
    train_labels = [
        line.get_label() for line in train_pdf.figures[0].axes[0].lines
    ]
    validation_labels = [
        line.get_label()
        for line in validation_pdf.figures[0].axes[0].lines
    ]
    assert any("Truth train loss" in label for label in train_labels)
    assert not any("Truth" in label for label in validation_labels)


def test_render_writes_nonempty_vector_pdf(tmp_path):
    """The diagnostic renderer writes a readable multi-page PDF."""
    output = tmp_path / "diagnostics.pdf"
    render(_result(), output)
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1000


def test_parameter_rows_include_error_and_physical_crb():
    """Fisher coordinate errors are reported with physical parameters."""
    fisher = {
        "schema": "lucid_per_source_numerical_basis_fisher_v1",
        "parameter_names": _result()["parameter_names"],
        "sources": [{"position_m": [0.0, 0.0, 0.0]}],
        "n_events": 2,
        "total": {
            "crb_standard_deviation": [0.01, 0.02, 0.03, 0.04, 0.05]
        },
    }
    rows = calibration_rows(_result(), fisher)
    assert rows[0]["absolute_error"] == 1.0
    assert rows[0]["error"] == 86.0 / 87.0 - 1.0
    assert rows[0]["crb_sigma"] == 0.01
    assert rows[0]["crb_sigma_physical"] == 0.87
    assert rows[1]["crb_sigma_physical"] == 1.7
    assert rows[2]["crb_sigma"] == 0.03
