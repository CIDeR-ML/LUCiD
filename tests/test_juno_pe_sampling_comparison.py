"""Tests for the paired direct/binned JUNO PE sampling diagnostic."""

import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from lucid.juno_laser.eventwise import sample_gaussian_mixture_events
from lucid.juno_laser.histogram import weighted_time_histogram
from lucid.juno_laser.sampling_comparison import (
    histogram_distance,
    sample_direct_weighted_hits,
    write_sampling_comparison,
)
from lucid.juno_laser.time_grid import TimeGrid
from scripts.plot_juno_pe_sampling_comparison import render


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    """Return one compact frozen weighted-hit basis and its time grid."""
    grid = TimeGrid(0.0, 10.0, 1.0)
    weights = np.asarray((2.0, 1.0, 3.0), dtype=np.float32)
    sensors = np.asarray((0, 1, 0), dtype=np.int32)
    times = np.asarray((1.25, 4.5, 7.75), dtype=np.float32)
    masses = np.asarray(weighted_time_histogram(
        weights,
        sensors,
        times,
        n_sensors=2,
        time_grid=grid,
    ))
    return grid, weights, sensors, times, masses


def test_direct_and_binned_sampling_preserve_expected_total():
    """Both constructions reproduce the same frozen expected PE mass."""
    grid, weights, sensors, times, masses = _fixture()
    direct = sample_direct_weighted_hits(
        weights,
        sensors,
        times,
        n_events=20000,
        n_sensors=2,
        time_grid=grid,
        sigma_ns=0.25,
        seed=11,
    )
    binned = sample_gaussian_mixture_events(
        masses,
        n_events=20000,
        time_grid=grid,
        sigma_ns=0.25,
        seed=13,
        background_rate_per_ns=0.0,
    )
    expected = float(np.sum(weights))
    np.testing.assert_allclose(
        direct.n_pe / direct.n_events, expected, rtol=0.02
    )
    np.testing.assert_allclose(
        binned.n_pe / binned.n_events, expected, rtol=0.02
    )
    np.testing.assert_allclose(np.sum(masses), expected, rtol=1.0e-6)


def test_direct_sampler_is_reproducible():
    """The legacy direct sampler is deterministic for a fixed seed."""
    grid, weights, sensors, times, _ = _fixture()
    arguments = dict(
        n_events=5,
        n_sensors=2,
        time_grid=grid,
        sigma_ns=0.25,
        seed=17,
    )
    first = sample_direct_weighted_hits(
        weights, sensors, times, **arguments
    )
    second = sample_direct_weighted_hits(
        weights, sensors, times, **arguments
    )
    np.testing.assert_array_equal(first.event_ids, second.event_ids)
    np.testing.assert_array_equal(first.pmt_ids, second.pmt_ids)
    np.testing.assert_array_equal(first.pe_times_ns, second.pe_times_ns)


def test_comparison_artifacts_are_readable(tmp_path):
    """HDF5, JSON and PDF outputs retain the diagnostic contract."""
    grid, weights, sensors, times, masses = _fixture()
    direct = sample_direct_weighted_hits(
        weights,
        sensors,
        times,
        n_events=20,
        n_sensors=2,
        time_grid=grid,
        sigma_ns=0.25,
        seed=19,
    )
    binned = sample_gaussian_mixture_events(
        masses,
        n_events=20,
        time_grid=grid,
        sigma_ns=0.25,
        seed=23,
    )
    h5_path = tmp_path / "comparison.h5"
    metadata = {
        "n_events": 20,
        "n_sensors": 2,
        "time_min_ns": 0.0,
        "time_max_ns": 10.0,
        "time_bin_ns": 1.0,
        "jitter_sigma_ns": 0.25,
        "irf_enabled": False,
        "background_rate_hz": 0.0,
    }
    write_sampling_comparison(
        h5_path,
        raw_weights=weights,
        raw_pmt_ids=sensors,
        raw_times_ns=times,
        bin_masses=masses,
        direct=direct,
        binned=binned,
        metadata=metadata,
    )
    with h5py.File(h5_path) as handle:
        assert handle.attrs["schema"] == (
            "lucid_pe_sampling_comparison_v1"
        )
        assert handle["bin_masses"].shape == (2, 10)
    pdf_path = tmp_path / "comparison.pdf"
    json_path = tmp_path / "comparison.json"
    render(
        h5_path,
        pdf_path,
        json_path,
        0.25,
        x_min_ns=0.0,
        x_max_ns=8.0,
        log_y=True,
        spectrum_mode="counts",
    )
    summary = json.loads(json_path.read_text())
    assert summary["metadata"]["irf_enabled"] is False
    assert summary["metadata"]["background_rate_hz"] == 0.0
    assert summary["display_window_ns"] == [0.0, 8.0]
    assert summary["spectrum_y_scale"] == "log"
    assert summary["spectrum_normalization"] == "none_raw_pe_counts"
    assert pdf_path.stat().st_size > 0
    assert histogram_distance(
        direct.pe_times_ns,
        binned.pe_times_ns,
        np.linspace(0.0, 10.0, 41),
    )["total_variation"] >= 0.0


def test_submit_uses_whitelisted_sbatch_and_wrapped_environment():
    """Keep sudoers-compatible sbatch and set variables inside the job."""
    completed = subprocess.run(
        [
            "make",
            "-n",
            "-C",
            str(PROJECT_ROOT / "pe_sampling_comparison"),
            "submit",
            "OUTDIR=/junofs/C/test/pe-sampling",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command = completed.stdout.replace("\\\n", " ")
    scheduler = command.index("sudo -u#35905 sbatch")
    wrapped_environment = command.index('--wrap="env OUTDIR=')
    assert scheduler < wrapped_environment
    assert "sudo -u#35905 env" not in command
