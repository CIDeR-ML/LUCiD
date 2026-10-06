"""Tests for closure-result diagnostics and rendering."""

import json

from scripts.plot_juno_closure_result import (
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
        "parameter_names": names,
        "truth": dict(zip(names, [87.0, 85.0, 0.0, 0.0, 0.0])),
        "fit": dict(zip(names, [86.0, 84.0, 0.1, -0.1, 0.2])),
        "stages": [{
            "name": "joint",
            "history": [
                [80.0, 80.0, 0.2, 0.1, -0.2],
                [86.0, 84.0, 0.1, -0.1, 0.2],
            ],
            "loss_history": [2.0, 1.5],
            "grad_norm_history": [0.4, 0.2],
            "evaluation_seconds": [1.0, 0.5],
            "mc_validation_history": [[0, 2.1], [1, 1.6]],
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


def test_render_writes_nonempty_vector_pdf(tmp_path):
    """The diagnostic renderer writes a readable multi-page PDF."""
    output = tmp_path / "diagnostics.pdf"
    render(_result(), output)
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1000
