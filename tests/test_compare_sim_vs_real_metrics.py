"""Deterministic tests for three-way laser comparison metrics."""

import csv

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from compare import compare_sim_vs_real as comparison
from compare import laser_compare_plotting as plotting
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
    assert result["run_prompt_pe_per_event"][0, 0] == 0.5
    assert result["simulation_prompt_pe_per_event"][0, 0] == 0.25
    assert result["run_late_pe_per_event"][0, 0] == 1.5
    assert result["simulation_late_pe_per_event"][0, 0] == 0.75
    assert result["run_prompt_late_ratio"][0, 0] == 1.0 / 3.0
    assert result["simulation_prompt_late_ratio"][0, 0] == 1.0 / 3.0


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


def test_shape_plot_separates_legend_and_normalization(tmp_path, monkeypatch):
    """The shape page keeps only the source legend and physics metrics."""
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
    assert not any("Nnorm" in item.get_text() for item in axis.texts)
    assert not any(
        "Shape normalization" in item.get_text() for item in axis.texts
    )


def test_pmt_summary_pools_pe_before_normalizing():
    """PMT summaries weight channels by their PE contribution."""
    pe_density = np.array([[1.0, 0.0], [0.0, 9.0]])
    pooled = comparison.pooled_group_shape(
        pe_density, np.array([True, True]), 1.0
    )
    np.testing.assert_allclose(pooled, np.array([0.1, 0.9]))


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
    assert float(row["lucid_mean_pe_ratio_to_fsmp"]) == 0.5
    assert float(row["lucid_prompt_pe_per_event"]) == 0.25
    assert float(row["lucid_late_pe_per_event"]) == 0.75
    assert int(row["fsmp_shape_normalization_pe"]) == 4
    assert int(row["lucid_shape_normalization_pe"]) == 4
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


def test_amplitude_title_shows_effective_gates_only(tmp_path, monkeypatch):
    """Amplitude titles use actual gates without implementation boilerplate."""
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

    title = CapturePdf.figures[0].axes[0].get_title()
    assert "Effective gates:" in title
    assert "relative to" not in title
    assert "all-PE amplitude valid" not in title
