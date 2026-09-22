"""Tests for the dark-to-simulation flip and source-intensity scale."""

import csv

import numpy as np
import pytest

from compare import compare_sim_vs_real as comparison
from compare.laser_compare_csv import write_comparison_stats_csv


def _inputs():
    """One-channel FSMP and two simulation sources with known counts."""
    edges = np.arange(-4.0, 5.0)
    fsmp = {
        "channels": np.array([5]),
        "pmt_type": np.array([1]),
        "tts": np.array([0.0]),
        "z": np.array([1.0]),
        "theta": np.array([2.0]),
        "fsmp_ch": np.full(6, 5),
        "fsmp_eid": np.array([10, 11, 10, 10, 11, 11]),
        "fsmp_time": np.array([-3.5, -2.5, 1.2, 1.2, 1.2, 1.2]),
        "fsmp_pe_count": np.ones(6, dtype=np.int64),
        "fsmp_physical_events": np.array(2),
        "fsmp_event_samples": np.array(2),
        "fsmp_target_pe": np.asarray(-1),
    }
    simulation = {
        "channels": np.array([5]),
        "edges": edges,
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "simulation_raw_density": np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        ),
        "simulation_entries": np.array([4]),
        "simulation_total_pe": np.array([4]),
        "simulation_events": np.array([4]),
        "label": np.asarray("LUCiD"),
        "kind": np.asarray("lucid"),
    }
    junosw = dict(simulation)
    junosw["label"] = np.asarray("JUNOsw")
    junosw["kind"] = np.asarray("junosw")
    return fsmp, simulation, junosw


def test_dark_is_kept_in_fsmp_and_added_to_simulations():
    """The FSMP spectrum stays untouched; MC gains the dark estimate."""
    fsmp, simulation, junosw = _inputs()
    off = comparison.build_channel_histograms_multi(
        fsmp, [simulation, junosw],
        alignment="peak", use_tts=False,
        add_dark_to_simulation=False, dark_window=1.0,
    )
    on = comparison.build_channel_histograms_multi(
        fsmp, [simulation, junosw],
        alignment="peak", use_tts=False,
        add_dark_to_simulation=True, dark_window=1.0,
    )
    dark = on["run_dark_rate_pe_per_event_per_ns"][0, 0]

    assert on["run_dark_prepeak_pe"][0, 0] == 2
    assert dark > 0.0
    np.testing.assert_allclose(
        on["run_pe_per_event_density"], off["run_pe_per_event_density"]
    )
    np.testing.assert_allclose(
        on["simulation_pe_per_event_density"]
        - off["simulation_pe_per_event_density"],
        dark,
    )
    processing = str(np.asarray(on["processing"]).item())
    assert "dark_to_sim=1" in processing
    assert "fsmp_dark=kept" in processing
    assert "dark_to_sim=0" in str(np.asarray(off["processing"]).item())


def test_dark_addition_dilutes_simulation_shape():
    """The unit-area MC shape is rebuilt from the dark-added amplitude."""
    fsmp, simulation, _ = _inputs()
    off = comparison.build_channel_histograms_multi(
        fsmp, [simulation], alignment="peak", use_tts=False,
        add_dark_to_simulation=False, dark_window=1.0,
    )
    on = comparison.build_channel_histograms_multi(
        fsmp, [simulation], alignment="peak", use_tts=False,
        add_dark_to_simulation=True, dark_window=1.0,
    )
    width = float(np.diff(on["edges"])[0])
    np.testing.assert_allclose(
        np.sum(on["simulation_density"][0, 0]) * width, 1.0
    )
    assert np.max(on["simulation_density"][0, 0]) < np.max(
        off["simulation_density"][0, 0]
    )


def test_source_intensity_scale_scales_signal_amplitude():
    """A per-source scale multiplies the MC signal amplitude only."""
    fsmp, simulation, junosw = _inputs()
    plain = comparison.build_channel_histograms_multi(
        fsmp, [simulation, junosw], alignment="peak", use_tts=False,
        add_dark_to_simulation=False,
    )
    scaled = comparison.build_channel_histograms_multi(
        fsmp, [simulation, junosw], alignment="peak", use_tts=False,
        add_dark_to_simulation=False,
        simulation_source_scale=[2.0, 1.0],
    )
    np.testing.assert_allclose(
        scaled["simulation_mean_pe_per_event"][0, 0],
        2.0 * plain["simulation_mean_pe_per_event"][0, 0],
    )
    np.testing.assert_allclose(
        scaled["simulation_mean_pe_per_event"][1, 0],
        plain["simulation_mean_pe_per_event"][1, 0],
    )
    np.testing.assert_allclose(
        np.asarray(scaled["simulation_applied_source_scale"]), [2.0, 1.0]
    )
    assert "source_scale=2:1" in str(np.asarray(scaled["processing"]).item())


def test_source_intensity_scale_validates_values():
    """Scales must be positive and match the number of sources."""
    fsmp, simulation, junosw = _inputs()
    with pytest.raises(ValueError, match="positive per source"):
        comparison.build_channel_histograms_multi(
            fsmp, [simulation, junosw], simulation_source_scale=[1.0]
        )
    with pytest.raises(ValueError, match="positive per source"):
        comparison.build_channel_histograms_multi(
            fsmp, [simulation, junosw],
            simulation_source_scale=[1.0, 0.0],
        )


def test_stats_csv_reports_dark_addition_and_scale(tmp_path):
    """The stats CSV carries the dark rate and the applied scale."""
    fsmp, simulation, junosw = _inputs()
    data = comparison.build_channel_histograms_multi(
        fsmp, [simulation, junosw], alignment="peak", use_tts=False,
        add_dark_to_simulation=True, dark_window=1.0,
        simulation_source_scale=[2.0, 1.0],
    )
    output = tmp_path / "stats.csv"
    write_comparison_stats_csv(output, data, 15040, 0.0)
    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    dark = data["run_dark_rate_pe_per_event_per_ns"][0, 0]
    assert float(row["fsmp_dark_rate_pe_per_event_per_ns"]) == dark
    assert "FSMP dark kept; MC dark addition=on" in row["processing"]
    scales = {
        key: float(value)
        for key, value in row.items()
        if key.endswith("_applied_source_scale")
    }
    assert sorted(scales.values()) == [1.0, 2.0]


def test_cli_dark_and_scale_flags():
    """The CLI exposes the dark flip and per-source intensity scales."""
    args = comparison.parse_args(["-z", "0", "--sim-dir", "/tmp"])
    assert args.add_dark_to_simulation is True
    assert args.source_intensity_scale is None
    args = comparison.parse_args(
        [
            "-z", "0", "--sim-dir", "/tmp",
            "--no-add-dark-to-simulation",
            "--source-intensity-scale", "2",
        ]
    )
    assert args.add_dark_to_simulation is False
    assert args.source_intensity_scale == [2.0]
    with pytest.raises(SystemExit):
        comparison.parse_args(
            [
                "-z", "0", "--sim-dir", "/tmp",
                "--source-intensity-scale", "2", "1",
            ]
        )
    with pytest.raises(SystemExit):
        comparison.parse_args(
            ["-z", "0", "--sim-dir", "/tmp", "--dark-subtract"]
        )

def test_cli_labels_override_cached_labels():
    """Reused caches must carry the labels of the current invocation."""
    sources = [
        {"kind": np.asarray("lucid"), "label": np.asarray("LUCiD (AllOn)")},
        {"kind": np.asarray("junosw"), "label": np.asarray("JUNOsw")},
    ]
    comparison._apply_cli_labels(
        sources, ["LUCiD (basis)"], "JUNOsw (J26.3.2)"
    )
    assert str(np.asarray(sources[0]["label"]).item()) == "LUCiD (basis)"
    assert str(np.asarray(sources[1]["label"]).item()) == "JUNOsw (J26.3.2)"

def test_auto_source_intensity_scale_corrects_to_junosw():
    """Auto scale divides each LUCiD source by its LUCiD:JUNOsw ratio."""
    fsmp, simulation, junosw = _inputs()
    simulation = dict(simulation)
    simulation["simulation_events"] = np.array([1])
    junosw = dict(junosw)
    junosw["simulation_total_pe"] = np.array([1])
    junosw["simulation_entries"] = np.array([1])
    junosw["simulation_events"] = np.array([1])
    sources = [simulation, junosw]

    scales = comparison.auto_source_intensity_scale(sources)

    np.testing.assert_allclose(scales, [0.25, 1.0])
    data = comparison.build_channel_histograms_multi(
        fsmp, sources, alignment="peak", use_tts=False,
        add_dark_to_simulation=False, simulation_source_scale=scales,
    )
    np.testing.assert_allclose(
        data["simulation_pe_per_event_density"][0],
        data["simulation_pe_per_event_density"][1],
    )
    assert "source_scale=0.25:1" in str(np.asarray(data["processing"]).item())


def test_auto_source_intensity_scale_requires_junosw():
    """Auto mode needs a JUNOsw reference source."""
    _, simulation, _ = _inputs()
    with pytest.raises(ValueError, match="JUNOsw"):
        comparison.auto_source_intensity_scale([simulation])

def test_spectrum_pages_render_to_real_pdfs(tmp_path):
    """Real PdfPages rendering catches errors the capture mocks miss."""
    fsmp, simulation, junosw = _inputs()
    data = comparison.build_channel_histograms_multi(
        fsmp, [simulation, junosw], alignment="peak", use_tts=False
    )
    shape_pdf = tmp_path / "shape.pdf"
    summary_pdf = tmp_path / "summary.pdf"
    comparison.plot_data(data, shape_pdf, (0.0, 4.0), 15040)
    comparison.plot_summary_data(data, summary_pdf, (0.0, 4.0), 15040, 1.0)
    for path in (shape_pdf, summary_pdf):
        blob = path.read_bytes()
        assert blob.startswith(b"%PDF")
        assert b"%%EOF" in blob[-1024:]
        assert len(blob) > 1000