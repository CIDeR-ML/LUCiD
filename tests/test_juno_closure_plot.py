"""Tests for minimal closure-result diagnostics and rendering."""

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
    """Return a completed two-parameter minimal closure mapping."""
    names = ["scat_ls", "abs_ls"]
    return {
        "schema": "lucid_minimal_closure_v1",
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
        "truth": {"scat_ls": 87.0, "abs_ls": 85.0},
        "start": {"scat_ls": 80.0, "abs_ls": 80.0},
        "fit": {"scat_ls": 86.0, "abs_ls": 84.0},
        "stages": [{
            "name": "joint",
            "selected_checkpoint": {
                "kind": "mc-validation-best",
                "step": 1,
                "loss_per_pe": 1.6,
            },
            "history": [[80.0, 80.0], [86.0, 84.0]],
            "loss_history": [2.0, 1.5],
            "grad_norm_history": [0.4, 0.2],
            "evaluation_seconds": [1.0, 0.5],
            "mc_validation_history": [[0, 2.1], [1, 1.6]],
            "best_loss": 1.5,
            "best_step": 1,
            "best_mc_validation_loss": 1.6,
            "best_mc_validation_step": 1,
            "last_values": [85.0, 83.0],
        }],
    }


def test_summary_reports_train_and_validation_best_steps(tmp_path):
    """The JSON summary preserves checkpoint-selection evidence."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result()))
    summary = summarize_result(load_result(path))
    stage = summary["stages"][0]
    assert stage["best_train_step"] == 1
    assert stage["best_mc_validation_step"] == 1
    assert stage["n_updates"] == 1
    assert summary["truth_train_loss_per_pe"] == 1.4


def test_loss_curves_use_separate_pages_and_truth_only_on_train():
    """Train and MC-validation retain independent plot semantics."""
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


def test_parameter_rows_report_fractional_length_error():
    """Both fitted quantities are positive LS lengths."""
    rows = calibration_rows(_result())
    assert rows[0]["absolute_error"] == 1.0
    assert rows[0]["error"] == 86.0 / 87.0 - 1.0
    assert rows[1]["error"] == 84.0 / 85.0 - 1.0
