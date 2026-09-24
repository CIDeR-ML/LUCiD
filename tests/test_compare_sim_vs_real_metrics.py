"""Deterministic tests for three-way laser comparison metrics."""

import csv

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from compare import compare_sim_vs_real as comparison
from compare import laser_compare_metrics as metrics
from compare import laser_compare_plotting as plotting
from compare.pe_scale_io import (
    fsmp_dark_counts,
    raw_pe_scale_inputs,
    valid_raw_pe_scale_cache,
)
from lucid.juno_laser.data import SIMULATION_SCHEMA, simulation_table


def _synthetic_inputs():
    """Return one-channel FSMP and simulation inputs with known rates."""
    edges = np.arange(5.0)
    fsmp = {
        "channels": np.array([5]),
        "pmt_type": np.array([1]),
        "tts": np.array([0.0]),
        "z": np.array([1.0]),
        "theta": np.array([2.0]),
        "fsmp_ch": np.array([5, 5, 5, 5]),
        "fsmp_eid": np.array([10, 10, 11, 11]),
        "fsmp_time": np.array([0.2, 1.2, 2.2, 3.2]),
        "fsmp_pe_count": np.ones(4, dtype=np.int64),
        "fsmp_physical_events": np.array(2),
        "fsmp_event_samples": np.array(2),
        "fsmp_target_pe": np.asarray(-1),
    }
    simulation = {
        "channels": np.array([5]),
        "edges": edges,
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "simulation_raw_density": np.full((1, 4), 0.25),
        "simulation_entries": np.array([4]),
        "simulation_total_pe": np.array([4]),
        "simulation_events": np.array([4]),
        "label": np.asarray("LUCiD"),
        "kind": np.asarray("lucid"),
    }
    return fsmp, simulation


def test_amplitude_data_has_plotting_compatibility_fields():
    """The cache-only amplitude adapter supplies all helper input fields."""
    data = {
        "simulation_labels": np.array(["LUCiD"]),
        "simulation_mean_pe_per_event": np.array([[1.0]]),
        "simulation_pe_per_event_density": np.array([[[1.0, 0.0]]]),
        "simulation_qe_factor": np.array([1.0]),
        "simulation_prompt_pe_per_event": np.array([[np.nan]]),
        "simulation_late_pe_per_event": np.array([[np.nan]]),
    }

    arrays = plotting.amplitude_plot_arrays(data)

    np.testing.assert_array_equal(
        arrays["simulation_density"], data["simulation_pe_per_event_density"]
    )
    np.testing.assert_array_equal(
        arrays["simulation_mean"], data["simulation_mean_pe_per_event"]
    )


def test_amplitude_density_and_gate_metrics():
    """Amplitude densities integrate to the requested PE/event metrics."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=False,
        prompt_window=(-1.0, 1.0),
        late_window=(1.0, 4.0),
    )
    width = float(np.diff(result["edges"])[0])

    np.testing.assert_allclose(
        np.sum(result["run_density"][0, 0]) * width,
        1.0,
    )
    np.testing.assert_allclose(
        np.sum(result["simulation_density"][0, 0]) * width,
        1.0,
    )
    np.testing.assert_allclose(
        result["run_pe_per_event_density"][0, 0],
        np.full(4, 0.5),
    )
    np.testing.assert_allclose(
        result["simulation_pe_per_event_density"][0, 0],
        np.full(4, 0.25),
    )
    assert result["run_mean_pe_per_event"][0, 0] == 2.0
    assert result["simulation_mean_pe_per_event"][0, 0] == 1.0
    assert result["run_shape_c_event_per_pe"][0, 0] == 0.5
    assert result["simulation_shape_c_event_per_pe"][0, 0] == 1.0
    assert result["simulation_applied_source_scale"][0] == 1.0
    assert result["run_prompt_pe_per_event"][0, 0] == 0.5
    assert result["simulation_prompt_pe_per_event"][0, 0] == 0.25
    assert result["run_late_pe_per_event"][0, 0] == 1.5
    assert result["simulation_late_pe_per_event"][0, 0] == 0.75
    assert result["run_prompt_late_ratio"][0, 0] == 1.0 / 3.0
    assert result["simulation_prompt_late_ratio"][0, 0] == 1.0 / 3.0


def test_dark_is_kept_in_fsmp_and_added_only_to_simulations(tmp_path):
    """Measured dark noise stays in FSMP and is added to every MC source."""
    fsmp, simulation = _synthetic_inputs()
    edges = np.arange(-4.0, 5.0)
    fsmp["fsmp_ch"] = np.full(6, 5)
    fsmp["fsmp_eid"] = np.array([10, 11, 10, 10, 11, 11])
    fsmp["fsmp_time"] = np.array([-3.5, -2.5, 1.2, 1.2, 1.2, 1.2])
    fsmp["fsmp_pe_count"] = np.ones(6, dtype=np.int64)
    simulation["edges"] = edges
    simulation["centers"] = (edges[:-1] + edges[1:]) / 2.0
    simulation["simulation_raw_density"] = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    )
    junosw = dict(simulation)
    junosw["label"] = np.asarray("JUNOsw (J26.3.2)")
    junosw["kind"] = np.asarray("junosw")

    without_dark = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation, junosw],
        alignment="peak",
        use_tts=False,
        add_dark_to_simulation=False,
        dark_window=1.0,
    )
    with_dark = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation, junosw],
        alignment="peak",
        use_tts=False,
        add_dark_to_simulation=True,
        dark_window=1.0,
    )
    dark = with_dark["run_dark_rate_pe_per_event_per_ns"][0, 0]

    assert dark > 0.0
    assert with_dark["run_dark_prepeak_pe"][0, 0] == 2
    np.testing.assert_allclose(
        without_dark["simulation_mean_pe_per_event"][:, 0],
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        with_dark["run_pe_per_event_density"],
        without_dark["run_pe_per_event_density"],
    )
    np.testing.assert_allclose(
        with_dark["simulation_pe_per_event_density"]
        - without_dark["simulation_pe_per_event_density"],
        dark,
    )
    title = plotting._amplitude_channel_title(
        5, "Hamamatsu", dark, ["LUCiD"], [1.0], ""
    )
    assert f"Dark added to MC={dark:.5g} PE/event/ns" in title
    output = tmp_path / "dark.csv"
    comparison.write_comparison_stats_csv(output, with_dark, 15040, 0.0)
    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["fsmp_dark_rate_pe_per_event_per_ns"]) == dark
    assert "FSMP dark kept; MC dark addition=on" in row["processing"]


def test_consensus_alignment_is_invariant_to_source_order():
    """Equal-weight L1 alignment must not privilege the first source."""
    edges = np.arange(9.0)
    times = np.array([1.2, 1.2, 2.2, 2.2, 2.2])
    first = comparison.density_hist(
        np.array([2.2, 2.2, 3.2, 3.2]), edges
    )
    second = comparison.density_hist(
        np.array([3.2, 3.2, 4.2, 4.2]), edges
    )

    forward = comparison.consensus_l1_alignment(
        times, [first, second], edges, 1.5
    )
    reverse = comparison.consensus_l1_alignment(
        times, [second, first], edges, 1.5
    )

    assert forward[0] == reverse[0]
    np.testing.assert_array_equal(forward[1], reverse[1])
    assert forward[2] == reverse[2]


def test_alignment_metadata_uses_comparable_peak_coordinates():
    """The cache exposes shift, aligned FSMP peak and peak differences."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=False,
    )

    aligned_peak = comparison.estimate_peak_from_density(
        result["centers"], result["run_density"][0, 0]
    )
    assert result["alignment_method"] == "global_median_consensus_l1"
    assert result["aligned_fsmp_peak_ns"][0, 0] == aligned_peak
    assert result["fsmp_peak_unaligned_ns"][0, 0] == result["run_peak"][0, 0]
    np.testing.assert_allclose(
        result["simulation_minus_fsmp_aligned_peak_ns"],
        result["simulation_peak"] - result["aligned_fsmp_peak_ns"],
    )


def test_all_channels_share_one_alignment_shift():
    """A detector comparison must preserve channel-to-channel timing."""
    fsmp, simulation = _synthetic_inputs()
    fsmp.update(
        {
            "channels": np.array([5, 6]),
            "pmt_type": np.array([1, 2]),
            "tts": np.zeros(2),
            "z": np.array([1.0, 2.0]),
            "theta": np.array([2.0, 3.0]),
            "fsmp_ch": np.array([5, 5, 6, 6]),
            "fsmp_eid": np.array([10, 11, 10, 11]),
            "fsmp_time": np.array([0.2, 1.2, 2.2, 3.2]),
            "fsmp_pe_count": np.ones(4, dtype=np.int64),
        }
    )
    simulation.update(
        {
            "channels": np.array([5, 6]),
            "simulation_raw_density": np.array(
                [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]
            ),
            "simulation_entries": np.array([2, 2]),
            "simulation_total_pe": np.array([2, 2]),
            "simulation_events": np.array([2, 2]),
        }
    )

    result = comparison.build_channel_histograms_multi(
        fsmp, [simulation], alignment="l1", use_tts=False
    )

    assert np.asarray(result["alignment_shift_ns"]).ndim == 0
    assert result["channel_alignment_candidate_shift_ns"].shape == (2,)


def test_shape_plot_displays_amplitude_normalization_c(
    tmp_path, monkeypatch
):
    """The shape page reports each curve's c=1/mu outside the legend."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=False,
    )

    class CapturePdf:
        """Capture figures passed to the PDF writer."""

        figures = []

        def __init__(self, _):
            self.figures = []
            CapturePdf.figures = self.figures

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def savefig(self, figure):
            self.figures.append(figure)

    monkeypatch.setattr(plotting, "PdfPages", CapturePdf)
    monkeypatch.setattr(plotting, "plot_metric_distribution", lambda *x: None)
    comparison.plot_data(result, tmp_path / "shape.pdf", (0.0, 4.0), 15040)

    axis = CapturePdf.figures[0].axes[0]
    legend = [item.get_text() for item in axis.get_legend().get_texts()]
    assert legend == ["FSMP RUN 15040", "LUCiD"]
    notes = [item.get_text() for item in axis.texts]
    scale_note = next(item for item in notes if "c=1/" in item)
    assert "FSMP: 0.5" in scale_note
    assert "LUCiD: 1" in scale_note
    assert "N=" not in scale_note
    assert "kappa" not in scale_note


def test_pmt_summary_pools_pe_before_normalizing():
    """PMT summaries weight channels by their PE contribution."""
    pe_density = np.array([[1.0, 0.0], [0.0, 9.0]])
    pooled = comparison.pooled_group_shape(
        pe_density, np.array([True, True]), 1.0
    )
    np.testing.assert_allclose(pooled, np.array([0.1, 0.9]))


def test_pmt_summary_displays_pooled_normalization_c(tmp_path, monkeypatch):
    """A pooled shape page reports its actual group normalization factor."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp, [simulation], alignment="l1", use_tts=False
    )

    class CapturePdf:
        """Capture figures passed to the PDF writer."""

        figures = []

        def __init__(self, _):
            self.figures = []
            CapturePdf.figures = self.figures

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def savefig(self, figure):
            self.figures.append(figure)

    monkeypatch.setattr(plotting, "PdfPages", CapturePdf)
    comparison.plot_summary_data(
        result,
        tmp_path / "summary.pdf",
        (0.0, 4.0),
        15040,
        1.0,
    )

    axis = CapturePdf.figures[0].axes[0]
    notes = [item.get_text() for item in axis.texts]
    scale_note = next(item for item in notes if "c=1/" in item)
    assert "N PMTs=1" in scale_note
    assert "FSMP: 0.5" in scale_note
    assert "LUCiD: 1" in scale_note


def test_lucid_event_count_uses_parquet_metadata(tmp_path):
    """Zero-hit events remain in the LUCiD PE/event denominator."""
    table = simulation_table(
        0,
        np.array([0]),
        np.array([0.5]),
        n_channels=2,
    )
    metadata = dict(SIMULATION_SCHEMA.metadata or {})
    metadata[b"n_events"] = b"3"
    path = tmp_path / "simulation.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), path)

    _, _, _, events = comparison.read_lucid_sim_histograms(
        [path],
        np.array([0.0, 1.0]),
        chunk_size=10,
        channels=np.array([0, 1]),
    )
    np.testing.assert_array_equal(events, np.array([3, 3]))


def test_fsmp_exposure_counts_event_step_pairs(tmp_path, monkeypatch):
    """FSMP PE/event uses posterior event-step states as its exposure."""
    schema = pa.schema(
        [
            pa.field("eid", pa.int64()),
            pa.field("ch", pa.int64()),
            pa.field("offset", pa.uint16()),
            pa.field("step", pa.uint16()),
            pa.field("PEt", pa.list_(pa.float64())),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([1, 1, 1, 2]),
            pa.array([5, 6, 5, 6]),
            pa.array([0, 0, 0, 0], type=pa.uint16()),
            pa.array([10, 10, 20, 10], type=pa.uint16()),
            pa.array([[0.1, 0.15], [0.2], [0.3], [0.4]]),
        ],
        schema=schema,
    )
    fsmp_root = tmp_path / "fsmp"
    fsmp_file = fsmp_root / "001" / "one.pq"
    fsmp_file.parent.mkdir(parents=True)
    pq.write_table(table, fsmp_file)
    monkeypatch.setattr(
        comparison,
        "_monitor_arrays",
        lambda _: (np.array([1, 2]), np.array([0.0, 0.0])),
    )

    records = comparison.read_fsmp_times_by_channel(
        [fsmp_file],
        fsmp_root,
        tmp_path / "rtraw",
        channels=np.array([5]),
        run=1,
    )
    assert int(records["physical_events"]) == 2
    assert int(records["event_samples"]) == 3
    np.testing.assert_array_equal(
        records["time"], np.array([0.1, 0.15, 0.3])
    )


def test_junosw_event_count_uses_header(tmp_path):
    """JUNOsw Header preserves zero-hit events in the denominator."""
    path = tmp_path / "junosw.h5"
    with h5py.File(path, "w") as stream:
        group = stream.create_group("Sim")
        group.create_dataset("Header", data=np.zeros(7, dtype=[("id", "i8")]))
    assert comparison._junosw_event_count([path], "Sim") == 7


def test_stats_csv_contains_amplitude_and_gate_metrics(tmp_path):
    """The machine-readable summary contains rate and shape audit fields."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=False,
        prompt_window=(-1.0, 1.0),
        late_window=(1.0, 4.0),
    )
    output = tmp_path / "comparison.csv"
    comparison.write_comparison_stats_csv(output, result, 15040, 0.0)

    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["fsmp_mean_pe_per_event"]) == 2.0
    assert float(row["fsmp_prompt_pe_per_event"]) == 0.5
    assert float(row["fsmp_late_pe_per_event"]) == 1.5
    assert float(row["lucid_mean_pe_per_event"]) == 1.0
    assert float(row["lucid_prompt_pe_per_event"]) == 0.25
    assert float(row["lucid_late_pe_per_event"]) == 0.75
    assert int(row["fsmp_pe"]) == 4
    assert int(row["fsmp_shape_normalization_pe"]) == 4
    assert (
        float(row["fsmp_shape_normalization_coefficient_per_pe_ns"])
        == 0.125
    )
    assert int(row["lucid_pe"]) == 4
    assert int(row["lucid_shape_normalization_pe"]) == 4
    assert (
        float(row["lucid_shape_normalization_coefficient_per_pe_ns"])
        == 0.25
    )
    assert float(row["lucid_applied_source_scale"]) == 1.0
    assert "fsmp_histogram_N_pe" not in row
    assert "lucid_shape_normalization_factor_event_per_pe" not in row
    assert row["alignment_method"] == "global_median_consensus_l1"
    assert "alignment_shift_ns" in row
    assert "fsmp_peak_unaligned_ns" in row
    assert "fsmp_peak_aligned_ns" in row
    np.testing.assert_allclose(
        float(row["lucid_minus_fsmp_aligned_peak_ns"]),
        float(row["lucid_peak_ns"])
        - float(row["fsmp_peak_aligned_ns"]),
    )


def test_amplitude_plot_is_vector_pdf(tmp_path):
    """Amplitude spectra and channel summaries are written to one PDF."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=False,
    )
    output = tmp_path / "amplitude.pdf"
    comparison.plot_amplitude_data(result, output, (0.0, 4.0), 15040)

    assert output.read_bytes().startswith(b"%PDF")


def test_amplitude_page_shows_mean_without_gate_metrics(
    tmp_path, monkeypatch
):
    """Amplitude pages show mean only; gate metrics remain CSV-only."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp, [simulation], alignment="l1", use_tts=False
    )

    class CapturePdf:
        """Capture figures passed to the PDF writer."""

        figures = []

        def __init__(self, _):
            self.figures = []
            CapturePdf.figures = self.figures

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def savefig(self, figure):
            self.figures.append(figure)

    monkeypatch.setattr(plotting, "PdfPages", CapturePdf)
    comparison.plot_amplitude_data(
        result, tmp_path / "amplitude.pdf", (0.0, 4.0), 15040
    )

    axis = CapturePdf.figures[0].axes[0]
    title = axis.get_title()
    legend = [item.get_text() for item in axis.get_legend().get_texts()]
    assert "Effective gates:" not in title
    assert "prompt=" not in title
    assert "late=" not in title
    assert "Dark added to MC=0 PE/event/ns" in title
    assert "Source intensity scale: LUCiD=1" in title
    assert "relative to" not in title
    assert "all-PE amplitude valid" not in title
    assert all("mean=" in item for item in legend)
    assert not any("prompt=" in item for item in legend)
    assert not any("late=" in item for item in legend)
    # two spectra plus the three unique gate-boundary vlines
    assert len(axis.lines) == 5


def test_lucid_qe_component_scales_only_lucid_sources():
    """The optional 532 nm factor must leave JUNOsw unchanged."""
    fsmp, lucid = _synthetic_inputs()
    junosw = dict(lucid)
    junosw["label"] = np.asarray("JUNOsw (J26.3.2)")
    junosw["kind"] = np.asarray("junosw")
    factors = metrics.simulation_qe_factors(
        [lucid, junosw], apply_lucid_qe=True, lucid_qe_eff=0.097
    )
    np.testing.assert_allclose(factors, [0.097, 1.0])

    result = comparison.build_channel_histograms_multi(
        fsmp,
        [lucid, junosw],
        alignment="l1",
        use_tts=False,
        add_dark_to_simulation=False,
    )
    scaled = plotting.amplitude_plot_arrays(
        result, apply_lucid_qe=True, lucid_qe_eff=0.097
    )

    np.testing.assert_allclose(scaled["source_factors"], [0.097, 1.0])
    np.testing.assert_allclose(
        scaled["simulation_mean"][:, 0], [0.097, 1.0]
    )
    np.testing.assert_allclose(
        scaled["simulation_density"][:, 0],
        np.array([np.full(4, 0.02425), np.full(4, 0.25)]),
    )

    disabled = plotting.amplitude_plot_arrays(result, apply_lucid_qe=False)
    np.testing.assert_allclose(disabled["source_factors"], [1.0, 1.0])


def test_source_intensity_scale_scales_signal_before_dark():
    """A configured source scale changes simulated signal before dark."""
    fsmp, simulation = _synthetic_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=False,
        add_dark_to_simulation=False,
        simulation_source_scale=np.array([2.5]),
    )

    assert result["simulation_applied_source_scale"][0] == 2.5
    assert result["simulation_mean_pe_per_event"][0, 0] == 2.5
    assert result["simulation_shape_c_event_per_pe"][0, 0] == 0.4


def test_cli_enables_common_lucid_qe(monkeypatch):
    """CLI defaults enable QE scaling and preserve the amplitude filename."""
    monkeypatch.setattr(comparison, "resolve_run", lambda *_: 15040)
    args = comparison.parse_args(["--sim-dir", "simulation"])

    assert args.apply_lucid_qe is True
    assert args.lucid_qe_effective == 0.097
    assert args.add_dark_to_simulation is True
    assert args.dark_window == 30.0
    assert args.source_intensity_scale is None
    assert args.amplitude_opt.name == "compare_sim_1000_amplitude.pdf"
    disabled = comparison.parse_args(
        [
            "--sim-dir",
            "simulation",
            "--no-lucid-qe-correction",
            "--no-add-dark-to-simulation",
        ]
    )
    assert disabled.apply_lucid_qe is False
    assert disabled.add_dark_to_simulation is False

    scaled = comparison.parse_args(
        [
            "--sim-dir",
            "simulation",
            "--source-intensity-scale",
            "0.5",
        ]
    )
    assert scaled.source_intensity_scale == [0.5]


def test_raw_pe_scale_uses_all_cached_pe_counts():
    """The independent tool preserves the raw-PDF all-PE definition."""
    fsmp = {
        "channels": np.array([4, 5]),
        "pmt_type": np.array([1, 2]),
        "theta": np.array([20.0, 30.0]),
        "total_pe": np.array([80.0, 0.0]),
        "events": 10.0,
    }
    junosw = {
        "label": "JUNOsw",
        "kind": "junosw",
        "total_pe": np.array([20.0, 10.0]),
        "events": np.array([5.0, 5.0]),
    }
    lucid = {
        "label": "LUCiD",
        "kind": "lucid",
        "total_pe": np.array([800.0, 100.0]),
        "events": np.array([5.0, 5.0]),
    }

    result = raw_pe_scale_inputs(fsmp, [junosw, lucid])

    np.testing.assert_allclose(result["pe_scale"][:, 0], [4.0, 80.0 / 77.6])
    assert np.isnan(result["pe_scale"][:, 1]).all()


def test_raw_pe_scale_dark_subtracts_sideband_counts():
    """The dark-subtracted ratio uses the corrected FSMP numerator."""
    edges = np.arange(11.0)
    histogram = np.full((3, 10), 2)
    histogram[0, 7] = 1002
    histogram[1, 7] = 50
    histogram[2] = 3
    total_pe = histogram.sum(axis=1).astype(np.float64)
    dark_pe, info = fsmp_dark_counts(total_pe, histogram, edges,
                                     margin_ns=3.0)

    np.testing.assert_allclose(dark_pe, [20.0, 20.0, 30.0])
    assert info["global_peak_ns"] == 7.5
    assert info["low_count_channels"] == 2
    assert info["no_window_channels"] == 0

    fsmp = {
        "channels": np.array([4, 5, 6]),
        "pmt_type": np.array([1, 2, 1]),
        "theta": np.array([20.0, 30.0, 40.0]),
        "total_pe": total_pe,
        "events": 10.0,
    }
    junosw = {
        "label": "JUNOsw",
        "kind": "junosw",
        "total_pe": np.array([250.0, 10.0, 5.0]),
        "events": np.array([5.0, 5.0, 5.0]),
    }
    lucid = {
        "label": "LUCiD",
        "kind": "lucid",
        "total_pe": np.array([2500.0, 100.0, 50.0]),
        "events": np.array([5.0, 5.0, 5.0]),
    }

    result = raw_pe_scale_inputs(fsmp, [junosw, lucid], dark_pe=dark_pe)

    np.testing.assert_allclose(result["pe_scale"][0], [4.08, 6.8, 6.0])
    np.testing.assert_allclose(result["pe_scale_dark"][0], [4.0, 4.8, np.nan])
    np.testing.assert_allclose(
        result["pe_scale_dark"][1],
        [1000.0 / 242.5, 48.0 / 9.7, np.nan],
    )
    assert result["pe_scale_dark_source_good_mask"][:, 2].sum() == 0


def test_raw_pe_scale_dark_respects_min_pe_per_event():
    """The dark ratio applies the per-event FSMP exposure filter."""
    fsmp = {
        "channels": np.array([4, 5]),
        "pmt_type": np.array([1, 2]),
        "theta": np.array([20.0, 30.0]),
        "total_pe": np.array([120.0, 15.0]),
        "events": 10.0,
    }
    junosw = {
        "label": "JUNOsw",
        "kind": "junosw",
        "total_pe": np.array([100.0, 10.0]),
        "events": np.array([5.0, 5.0]),
    }
    dark_pe = np.array([20.0, 5.0])

    result = raw_pe_scale_inputs(
        fsmp, [junosw], dark_pe=dark_pe, min_pe_per_event=2.0
    )

    np.testing.assert_allclose(result["pe_scale_dark"][0], [1.0, np.nan])
    np.testing.assert_allclose(result["pe_scale"][0], [1.2, 1.5])


def test_raw_pe_scale_per_event_averages_by_exposure():
    """Per-event mode divides each side by its own event exposure."""
    fsmp = {
        "channels": np.array([4, 5]),
        "pmt_type": np.array([1, 2]),
        "theta": np.array([20.0, 30.0]),
        "total_pe": np.array([120.0, 15.0]),
        "events": 10.0,
    }
    junosw = {
        "label": "JUNOsw",
        "kind": "junosw",
        "total_pe": np.array([100.0, 10.0]),
        "events": np.array([5.0, 5.0]),
    }
    dark_pe = np.array([20.0, 5.0])

    result = raw_pe_scale_inputs(
        fsmp, [junosw], dark_pe=dark_pe, min_pe_per_event=2.0,
        per_event=True,
    )

    np.testing.assert_allclose(result["pe_scale"][0], [0.6, 0.75])
    np.testing.assert_allclose(result["pe_scale_dark"][0], [0.5, np.nan])
    assert bool(result["pe_per_event"])


def test_raw_cache_validation_reuses_only_matching_channels(tmp_path):
    """Raw-cache reuse rejects a file from another channel selection."""
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        channels=np.array([1, 2]),
        fsmp_total_pe=np.array([3, 4]),
        kind=np.asarray("lucid"),
    )

    assert valid_raw_pe_scale_cache(
        path, np.array([1, 2]), {"fsmp_total_pe"}
    )
    assert not valid_raw_pe_scale_cache(
        path, np.array([1, 3]), {"fsmp_total_pe"}
    )
    assert not valid_raw_pe_scale_cache(path, np.array([1, 2]), {"pmt_type"})
    assert valid_raw_pe_scale_cache(
        path, np.array([1, 2]), {"kind"}, kind="lucid"
    )
    assert not valid_raw_pe_scale_cache(
        path, np.array([1, 2]), {"kind"}, kind="junosw"
    )


def test_raw_cache_validation_respects_first_pe_mode(tmp_path):
    """A first-PE cache must not be reused for the all-PE reference mode."""
    path = tmp_path / "first_pe.npz"
    np.savez_compressed(
        path,
        channels=np.array([1, 2]),
        fsmp_total_pe=np.array([3, 4]),
        fsmp_first_pe_only=np.asarray(True),
    )

    fields = {"fsmp_total_pe"}
    assert not valid_raw_pe_scale_cache(path, np.array([1, 2]), fields)
    assert valid_raw_pe_scale_cache(
        path, np.array([1, 2]), fields, first_pe_only=True
    )
