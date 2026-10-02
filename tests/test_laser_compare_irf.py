"""Deterministic tests for the measured per-channel IRF kernels."""

from pathlib import Path

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from compare_lucid_sim import compare_sim_vs_real as comparison
from compare_lucid_sim import laser_compare_irf as irf_mod
from compare_lucid_sim import laser_compare_metrics as metrics


def _row(sn, load, fwhm, peak_bin, n_bins=40):
    """One synthetic IRF struct row with a right-tailed peak."""
    mass = np.zeros(n_bins)
    mass[peak_bin] = 0.6
    mass[peak_bin + 1] = 0.3
    mass[peak_bin + 2] = 0.1
    return (sn, load, fwhm, -120.0, 0.5, mass)


def _write_irf_h5(path, extra=None):
    """Write a synthetic IRF file with three characterized tubes.

    Input: output path; extra appends raw entry tuples (sn, load, fwhm,
    peak_bin).  Output: none (h5 file written).
    """
    entries = [
        (b"EA0001", 41, 1.5, 10),
        (b"EA0002", 41, 2.5, 12),
        (b"PA0001", 41, 7.0, 20),
    ]
    entries.extend(extra or [])
    n_bins = 40
    dtype = [
        ("sn", "S12"),
        ("load", "<i2"),
        ("fwhm", "<f8"),
        ("t0", "<f8"),
        ("dt", "<f8"),
        ("mass", "<f8", (n_bins,)),
    ]
    rows = np.zeros(len(entries), dtype=dtype)
    for i, (sn, load, fwhm, peak_bin) in enumerate(entries):
        rows[i] = _row(sn, load, fwhm, peak_bin, n_bins)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("hist", data=rows)


def _write_pmt_map(path):
    """Write a synthetic PMT map linking serial numbers to channels.

    Input: output path.  Output: none (parquet file written).
    """
    pd.DataFrame(
        {
            "copy_number": [100, 200, 300],
            "pmt_type": [1, 1, 2],
            "SN": ["EA0001", "EA0002", "PA0001"],
        }
    ).to_parquet(path, index=False)


@pytest.fixture()
def irf_files(tmp_path):
    """Return (irf_h5, pmt_map) paths of the synthetic fixtures."""
    h5_path = tmp_path / "irf.h5"
    map_path = tmp_path / "pmt_map.pq"
    _write_irf_h5(h5_path)
    _write_pmt_map(map_path)
    return h5_path, map_path


def test_read_irf_hist_roundtrip(irf_files):
    """read_irf_hist returns the struct rows with all fields."""
    rows = irf_mod.read_irf_hist(irf_files[0])
    assert rows.dtype.names == ("sn", "load", "fwhm", "t0", "dt", "mass")
    assert len(rows) == 3
    assert rows["t0"][0] == -120.0
    assert rows["dt"][0] == 0.5


def test_build_channel_kernels_per_channel(irf_files):
    """Each characterized channel gets its own unit-sum centered kernel."""
    kernels, meta, orphans = irf_mod.build_channel_kernels(
        *irf_files, bin_width=1.0
    )
    assert set(kernels) == {100, 200, 300}
    assert orphans == []
    for kernel in kernels.values():
        assert len(kernel) % 2 == 1
        assert np.argmax(kernel) == (len(kernel) - 1) // 2
        assert kernel.sum() == pytest.approx(1.0, abs=1e-12)
        # every fixture tube peaks at pair mass 0.6 + 0.3 = 0.9
        assert kernel[(len(kernel) - 1) // 2] == pytest.approx(0.9)
    assert meta[100] == {"sn": "EA0001", "load": 41, "fwhm": 1.5}
    assert meta[300]["sn"] == "PA0001"


def test_multi_load_tube_uses_largest_load(irf_files, tmp_path):
    """A tube measured in several loads contributes the largest load."""
    h5_path = tmp_path / "irf_multiload.h5"
    _write_irf_h5(h5_path, extra=[(b"EA0001", 23, 9.9, 30)])
    kernels, meta, _ = irf_mod.build_channel_kernels(
        h5_path, irf_files[1], bin_width=1.0
    )
    assert set(kernels) == {100, 200, 300}
    assert meta[100]["load"] == 41
    assert meta[100]["fwhm"] == 1.5


def test_unmapped_serials_are_orphaned(irf_files, tmp_path):
    """IRF serials absent from the PMT map are listed as orphans."""
    h5_path = tmp_path / "irf_extra.h5"
    _write_irf_h5(h5_path, extra=[(b"ZZ9999", 41, 9.9, 30)])
    kernels, _, orphans = irf_mod.build_channel_kernels(
        h5_path, irf_files[1], bin_width=1.0
    )
    assert set(kernels) == {100, 200, 300}
    assert orphans == ["ZZ9999"]


def test_no_mapped_rows_raise(tmp_path):
    """An IRF file with no serial in the PMT map raises ValueError."""
    h5_path = tmp_path / "irf_none.h5"
    _write_irf_h5(h5_path)
    empty_map = tmp_path / "empty_map.pq"
    pd.DataFrame(
        {"copy_number": [1], "pmt_type": [1], "SN": ["ZZ0000"]}
    ).to_parquet(empty_map, index=False)
    with pytest.raises(ValueError, match="no IRF rows"):
        irf_mod.build_channel_kernels(h5_path, empty_map, bin_width=1.0)


def test_interpolated_bin_width(irf_files):
    """Non-integer width ratios use density interpolation, still unit-sum."""
    kernels, _, _ = irf_mod.build_channel_kernels(
        *irf_files, bin_width=0.6
    )
    for kernel in kernels.values():
        assert len(kernel) % 2 == 1
        assert kernel.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.argmax(kernel) == (len(kernel) - 1) // 2


def test_characterized_channels(irf_files):
    """characterized_channels lists the mapped copy numbers, sorted."""
    assert irf_mod.characterized_channels(*irf_files) == [100, 200, 300]


def test_tts_irf_autofills_channel_selection_from_irf_h5(
    irf_files, tmp_path
):
    """TTS-vs-IRF derives irf125.csv from the IRF h5 and PMT map."""
    from compare_lucid_sim import plot_tts_vs_irf as tts_irf

    args = SimpleNamespace(
        channel_selection=None,
        irf_h5=Path(irf_files[0]),
        pmt_map=Path(irf_files[1]),
        output_root=tmp_path / "out",
    )
    tts_irf.resolve_channel_selection(args)
    assert args.channel_selection == tmp_path / "out" / "irf125.csv"
    channels = np.loadtxt(args.channel_selection, skiprows=1, dtype=int)
    np.testing.assert_array_equal(channels, [100, 200, 300])


def test_tts_irf_keeps_explicit_channel_selection(tmp_path):
    """An explicit --channel-selection path is left untouched."""
    from compare_lucid_sim import plot_tts_vs_irf as tts_irf

    explicit = tmp_path / "custom.csv"
    args = SimpleNamespace(
        channel_selection=explicit,
        irf_h5=None,
        pmt_map=None,
        output_root=tmp_path / "out",
    )
    tts_irf.resolve_channel_selection(args)
    assert args.channel_selection == explicit
    assert not (tmp_path / "out").exists()


def test_out_of_cd_range_duplicate_serial_is_excluded(irf_files, tmp_path):
    """A non-CD map row sharing a characterized serial is ignored."""
    dup_map = tmp_path / "dup_map.pq"
    pd.DataFrame(
        {
            "copy_number": [100, 200, 300, 50639],
            "pmt_type": [1, 1, 2, 2],
            "SN": ["EA0001", "EA0002", "PA0001", "PA0001"],
        }
    ).to_parquet(dup_map, index=False)
    kernels, meta, _ = irf_mod.build_channel_kernels(
        irf_files[0], dup_map, bin_width=1.0
    )
    assert set(kernels) == {100, 200, 300}
    assert irf_mod.characterized_channels(irf_files[0], dup_map) == [
        100,
        200,
        300,
    ]


def test_irf_descriptor_counts_channels(irf_files):
    """The cache-key descriptor embeds the kernel channel count."""
    kernels, _, _ = irf_mod.build_channel_kernels(
        *irf_files, bin_width=1.0
    )
    descriptor = irf_mod.irf_descriptor(irf_files[0], kernels)
    assert descriptor == "file:irf.h5:ch3"
    assert irf_mod.irf_descriptor(irf_files[0], {100: kernels[100]}) == (
        "file:irf.h5:ch1"
    )


def _mini_inputs():
    """One-channel FSMP and simulation inputs with a flat sim density."""
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


def test_irf_mode_disables_tts_smearing():
    """With kernels given, the TTS convolution is skipped entirely."""
    fsmp, simulation = _mini_inputs()
    fsmp["tts"] = np.array([5.0])
    kernel = np.array([0.25, 0.5, 0.25])
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=True,
        irf_kernels={5: kernel},
        irf_fwhm_by_channel={5: 1.5},
        prompt_window=(-1.0, 1.0),
        late_window=(1.0, 4.0),
    )
    expected = metrics.convolve_kernel(
        simulation["simulation_raw_density"][0], kernel
    )
    np.testing.assert_allclose(result["simulation_density"][0, 0], expected)
    # TTS with sigma=5 ns would widen the array (5-sigma kernel radius),
    # so a matching shape already proves the TTS path was skipped.
    tts_only = metrics.convolve_density(
        simulation["simulation_raw_density"][0], 1.0, 5.0
    )
    assert result["simulation_density"][0, 0].shape != tts_only.shape
    assert result["irf_fwhm_by_channel"][0] == pytest.approx(1.5)
    assert "irf_sigma_by_channel" not in result


def test_irf_mode_without_kernel_leaves_density_raw():
    """A channel without a measured kernel is left unsmoothed."""
    fsmp, simulation = _mini_inputs()
    result = comparison.build_channel_histograms_multi(
        fsmp,
        [simulation],
        alignment="l1",
        use_tts=True,
        irf_kernels={999: np.array([0.25, 0.5, 0.25])},
        irf_fwhm_by_channel={999: 1.5},
        prompt_window=(-1.0, 1.0),
        late_window=(1.0, 4.0),
    )
    np.testing.assert_allclose(
        result["simulation_density"][0, 0],
        simulation["simulation_raw_density"][0],
    )
    assert result["irf_fwhm_by_channel"][0] == 0.0


def test_processing_string_carries_irf_descriptor():
    """The aligned-cache key serializes the measured-IRF descriptor."""
    text = metrics.processing_string(
        False, "file:irf.h5:ch3", True, alignment="l1"
    )
    assert "metrics=8" in text
    assert "tts=0" in text
    assert "irf=file:irf.h5:ch3" in text
    legacy = metrics.processing_string(True, "0", True, alignment="l1")
    assert "tts=1" in legacy
    assert "irf=0" in legacy


def _mini_two_source_inputs():
    """One-channel FSMP plus LUCiD and JUNOsw simulation inputs."""
    fsmp, lucid = _mini_inputs()
    junosw = dict(lucid)
    junosw["label"] = np.asarray("JUNOsw")
    junosw["kind"] = np.asarray("junosw")
    return fsmp, [lucid, junosw]


def _mini_two_channel_inputs():
    """Two-channel FSMP plus LUCiD/JUNOsw inputs, one channel per type.

    The PMT-type axis then has length two, exactly like the source axis,
    so the merge regression covers the axis-collision case.
    """
    edges = np.arange(5.0)
    fsmp = {
        "channels": np.array([5, 6]),
        "pmt_type": np.array([1, 2]),
        "tts": np.array([0.0, 0.0]),
        "z": np.array([1.0, 1.0]),
        "theta": np.array([2.0, 2.0]),
        "fsmp_ch": np.array([5, 5, 5, 5, 6, 6, 6, 6]),
        "fsmp_eid": np.array([10, 10, 11, 11, 10, 10, 11, 11]),
        "fsmp_time": np.array([0.2, 1.2, 2.2, 3.2, 0.2, 1.2, 2.2, 3.2]),
        "fsmp_pe_count": np.ones(8, dtype=np.int64),
        "fsmp_physical_events": np.array(2),
        "fsmp_event_samples": np.array(2),
        "fsmp_target_pe": np.asarray(-1),
    }
    simulation = {
        "channels": np.array([5, 6]),
        "edges": edges,
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "simulation_raw_density": np.full((2, 4), 0.25),
        "simulation_entries": np.array([4, 4]),
        "simulation_total_pe": np.array([4, 4]),
        "simulation_events": np.array([4, 4]),
        "label": np.asarray("LUCiD"),
        "kind": np.asarray("lucid"),
    }
    junosw = dict(simulation)
    junosw["label"] = np.asarray("JUNOsw")
    junosw["kind"] = np.asarray("junosw")
    return fsmp, [simulation, junosw]


def test_merge_smearing_variants_interleaves_sources():
    """The merged dictionary carries TTS then IRF per original source."""
    fsmp, sources = _mini_two_channel_inputs()
    kernel = np.array([0.25, 0.5, 0.25])
    common = {
        "alignment": "l1",
        "prompt_window": (-1.0, 1.0),
        "late_window": (1.0, 4.0),
    }
    tts_data = comparison.build_channel_histograms_multi(
        fsmp, sources, use_tts=True, **common
    )
    irf_data = comparison.build_channel_histograms_multi(
        fsmp,
        sources,
        use_tts=False,
        irf_kernels={5: kernel, 6: kernel},
        irf_fwhm_by_channel={5: 1.5, 6: 1.5},
        **common,
    )
    merged = metrics.merge_smearing_variants(tts_data, irf_data, "PROC")
    labels = [str(value) for value in merged["simulation_labels"]]
    assert labels == [
        "LUCiD (TTS)",
        "LUCiD (IRF)",
        "JUNOsw (TTS)",
        "JUNOsw (IRF)",
    ]
    assert merged["simulation_density"].shape == (4, 2, 4)
    np.testing.assert_allclose(
        merged["simulation_density"][0, 0], np.full(4, 0.25)
    )
    np.testing.assert_allclose(
        merged["simulation_density"][1, 0],
        metrics.convolve_kernel(np.full(4, 0.25), kernel),
    )
    np.testing.assert_allclose(
        merged["run_density"][0], tts_data["run_density"][0]
    )
    assert str(np.asarray(merged["processing"]).item()) == "PROC"
    assert merged["simulation_applied_source_scale"].tolist() == [1] * 4
    assert merged["simulation_kinds"].tolist() == [
        "lucid", "lucid", "junosw", "junosw",
    ]
    # every source-major array must gain the doubled variant axis
    for key in (
        "l1_distance",
        "wasserstein_distance",
        "run_density",
        "gate_reference_peak_ns",
        "fsmp_peak_unaligned_ns",
        "aligned_fsmp_peak_ns",
        "prompt_window_effective_low_ns",
    ):
        assert merged[key].shape[0] == 4, key
    # PMT-type/channel axes keep their length despite the collision
    assert merged["alignment_pmt_types"].tolist() == [1, 2]
    assert merged["alignment_pmt_type_candidate_median_ns"].shape == (2,)
    assert merged["pmt_type"].tolist() == [1, 2]
    assert merged["channels"].tolist() == [5, 6]


def test_merge_smearing_variants_rejects_unknown_source_axis():
    """A new source-major key without a merge rule fails loudly."""
    fsmp, sources = _mini_two_source_inputs()
    common = {
        "alignment": "l1",
        "prompt_window": (-1.0, 1.0),
        "late_window": (1.0, 4.0),
    }
    tts_data = comparison.build_channel_histograms_multi(
        fsmp, sources, use_tts=True, **common
    )
    irf_data = comparison.build_channel_histograms_multi(
        fsmp, sources, use_tts=True, **common
    )
    tts_data["mystery"] = np.zeros((2, 1))
    irf_data["mystery"] = np.zeros((2, 1))
    with pytest.raises(ValueError, match="mystery"):
        metrics.merge_smearing_variants(tts_data, irf_data)


def test_merge_smearing_variants_rejects_source_mismatch():
    """Mismatched source sets raise instead of silently merging."""
    fsmp, sources = _mini_two_source_inputs()
    common = {
        "alignment": "l1",
        "prompt_window": (-1.0, 1.0),
        "late_window": (1.0, 4.0),
    }
    tts_data = comparison.build_channel_histograms_multi(
        fsmp, sources[:1], use_tts=True, **common
    )
    irf_data = comparison.build_channel_histograms_multi(
        fsmp, sources[1:], use_tts=True, **common
    )
    with pytest.raises(ValueError, match="different sources"):
        metrics.merge_smearing_variants(tts_data, irf_data)


def test_cli_accepts_irf_h5():
    """--irf-h5 parses into a Path."""
    argv = ["--lucid-sim-dir", "/x", "--irf-h5", "kernels.h5"]
    args = comparison.parse_args(argv)
    assert args.irf_h5 == Path("kernels.h5")


def test_cli_rejects_removed_gaussian_irf_flags():
    """The removed Gaussian IRF placeholder flags are rejected."""
    argv = ["--lucid-sim-dir", "/x", "--irf-sigma-nnvt", "1.0"]
    with pytest.raises(SystemExit):
        comparison.parse_args(argv)


def test_main_cli_rejects_compare_tts_irf_flag():
    """The production CLI carries no TTS/IRF switch; it moved out."""
    argv = [
        "--lucid-sim-dir", "/x",
        "--compare-tts-irf", "--irf-h5", "kernels.h5",
    ]
    with pytest.raises(SystemExit):
        comparison.parse_args(argv)


def test_channel_selection_csv_overrides_random_choice(tmp_path):
    """--channel-selection fixes the plotted channels to the CSV list."""
    csv_path = tmp_path / "channels.csv"
    csv_path.write_text("channel\n7\n3\n5\n")
    args = comparison.parse_args(
        ["--lucid-sim-dir", "/x", "--channel-selection", str(csv_path)]
    )
    channels, _ = comparison.selected_channels(
        args, require_files=False, files=[]
    )
    assert channels.tolist() == [3, 5, 7]


def test_channel_selection_default_is_none():
    """Without --channel-selection the random -n/--seed path is kept."""
    args = comparison.parse_args(["--lucid-sim-dir", "/x"])
    assert args.channel_selection is None


def test_channel_selection_stem_names_outputs(tmp_path):
    """The output file suffix follows the channel-selection CSV stem."""
    csv_path = tmp_path / "channel_selection_irf125.csv"
    csv_path.write_text("channel\n3\n")
    args = comparison.parse_args(
        ["--lucid-sim-dir", "/x", "--channel-selection", str(csv_path)]
    )
    assert args.aligned_data.name == (
        "aligned_data_channel_selection_irf125.npz"
    )