"""Deterministic test for the PPP zero-support counterexample report."""

import json

import numpy as np

from lucid.juno_laser.eventwise import EventwisePeObservation
from lucid.juno_laser.time_grid import TimeGrid
from lucid.juno_laser.zero_support import render_zero_support_example


def test_zero_support_report_preserves_exact_zero(tmp_path):
    """The selected PE must have truth support and exactly zero fit rate."""
    grid = TimeGrid(0.0, 4.0, 1.0)
    observation = EventwisePeObservation(
        event_ids=np.asarray([0], dtype=np.int32),
        pmt_ids=np.asarray([0], dtype=np.int32),
        pe_times_ns=np.asarray([0.5], dtype=np.float32),
        n_events=1,
        n_sensors=1,
        time_grid=grid,
    )
    prefix = tmp_path / "zero_support"
    summary = render_zero_support_example(
        observation,
        np.asarray([[2.0, 0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0, 3.0]]),
        np.asarray([0.0]),
        sigma_ns=0.1,
        truncate_sigma=5.0,
        output_prefix=prefix,
        label="test",
    )
    assert summary["truth_rate_at_pe_per_ns"] > 0.0
    assert summary["fit_rate_at_pe_per_ns"] == 0.0
    assert prefix.with_suffix(".pdf").is_file()
    saved = json.loads(prefix.with_suffix(".json").read_text())
    assert saved["event_id"] == 0
    assert saved["pmt_id"] == 0
