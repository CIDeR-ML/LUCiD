"""Tests for the numerical time-basis convergence contract."""

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(
    str(ROOT / "scripts" / "summarize_juno_time_grid_convergence.py")
)
SUMMARIZE = MODULE["summarize_results"]


def _result(width, shift):
    """Return one comparable synthetic closure summary."""
    return {
        "config": "config.json",
        "source_positions_m": [[0.0, 0.0, 0.0]],
        "n_events": 10,
        "n_photons": 1000,
        "k": 64,
        "intensity": 1000.0,
        "event_exposure": 1.0,
        "event_time_offset_ns": 0.0,
        "parameter_names": [
            "scat_ls", "abs_ls", "lsphase_shape_0",
            "lsphase_shape_1", "lsphase_shape_2",
        ],
        "time_bin_ns": width,
        "fit": {
            "scat_ls": 80.0 + shift,
            "abs_ls": 85.0 + shift,
            "lsphase_shape_0": 0.5 + 0.001 * shift,
            "lsphase_shape_1": -0.5 - 0.001 * shift,
            "lsphase_shape_2": 0.25 + 0.001 * shift,
        },
        "final_gradient": {
            "scat_ls": 1.0 + shift,
            "abs_ls": 2.0 + shift,
            "lsphase_shape_0": 3.0 + shift,
            "lsphase_shape_1": 4.0 + shift,
            "lsphase_shape_2": 5.0 + shift,
        },
        "final_loss_per_pe": 2.0 + 0.001 * shift,
    }


def test_grid_summary_uses_finest_grid_and_finest_pair():
    """Convergence is judged between 1 ns and 0.5 ns, not 2 and 0.5 ns."""
    records = (
        (2.0, _result(2.0, 1.0)),
        (1.0, _result(1.0, 0.01)),
        (0.5, _result(0.5, 0.0)),
    )
    summary = SUMMARIZE(
        records,
        parameter_relative_tolerance=0.01,
        loss_absolute_tolerance=1.0e-3,
        gradient_relative_tolerance=0.1,
    )
    assert summary["reference_time_bin_ns"] == 0.5
    assert summary["finest_pair"]["coarser_time_bin_ns"] == 1.0
    assert summary["converged"]


def test_grid_summary_rejects_changed_non_grid_configuration():
    """A grid scan cannot silently compare different photon statistics."""
    coarse = _result(2.0, 0.0)
    fine = _result(1.0, 0.0)
    fine["n_photons"] = 2000
    with pytest.raises(ValueError, match="n_photons"):
        SUMMARIZE(
            ((2.0, coarse), (1.0, fine)),
            parameter_relative_tolerance=0.01,
            loss_absolute_tolerance=1.0e-3,
            gradient_relative_tolerance=0.1,
        )
