"""Tests for compare/plot_source_scale_vs_z.py."""

import csv

import numpy as np
import pytest

from compare_lucid_sim import plot_source_scale_vs_z as scan


def test_lucid_z_dir_names_encode_centimeter_units():
    """z<N> names map to meters; suffixed variants are rejected."""
    assert scan.z_m_from_lucid_dir("z0") == 0.0
    assert scan.z_m_from_lucid_dir("z300") == 3.0
    assert scan.z_m_from_lucid_dir("z1610") == 16.1
    assert scan.z_m_from_lucid_dir("z1610_AcrylicBowloff") is None
    assert scan.z_m_from_lucid_dir("compare_sim-restored") is None


def test_junosw_z_dir_names_encode_millimeter_units():
    """Plain numeric JUNOsw names map from millimeters to meters."""
    assert scan.z_m_from_junosw_dir("0") == 0.0
    assert scan.z_m_from_junosw_dir("3000") == 3.0
    assert scan.z_m_from_junosw_dir("16100") == 16.1
    assert scan.z_m_from_junosw_dir("1.h5") is None


def test_paired_z_directories_keep_only_common_plain_dirs(tmp_path):
    """Only z points present in both frameworks are scanned."""
    lucid = tmp_path / "lucid"
    junosw = tmp_path / "junosw"
    for name in ("z0", "z300", "z600", "z1610_AcrylicBowloff"):
        (lucid / name).mkdir(parents=True)
    for name in ("0", "3000", "9000"):
        (junosw / name).mkdir(parents=True)

    pairs = scan.paired_z_directories(lucid, junosw)

    assert [z for z, _, _ in pairs] == [0.0, 3.0]
    assert pairs[0][1] == lucid / "z0"
    assert pairs[0][2] == junosw / "0"
    assert pairs[1][1] == lucid / "z300"
    assert pairs[1][2] == junosw / "3000"


def test_amplitude_cache_round_trip_and_channel_guard(tmp_path):
    """Caches are reused only when the channel selection matches."""
    cache = tmp_path / "c.npz"
    channels = np.array([3, 5])

    def compute():
        return np.array([7, 8]), np.array([2, 2])

    first = scan._load_or_compute_amplitude(
        cache, compute, channels, force=False
    )
    second = scan._load_or_compute_amplitude(
        cache, None, channels, force=False
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    with pytest.raises(ValueError, match="Channel mismatch"):
        scan._load_or_compute_amplitude(
            cache, None, np.array([3, 6]), force=False
        )


def _row(z_m, lucid, junosw, scale):
    """One synthetic scale-scan row."""
    return {
        "z_m": z_m,
        "lucid_amplitude_pe_per_event": lucid,
        "junosw_amplitude_pe_per_event": junosw,
        "lucid_over_junosw_ratio": lucid / junosw,
        "source_intensity_scale": scale,
        "channel_count": 100,
    }


def test_csv_and_pdf_outputs(tmp_path):
    """The scan writes a readable CSV table and a vector PDF."""
    rows = [
        _row(0.0, 36.5, 0.5533, 0.1563),
        _row(3.0, 40.0, 0.5800, 0.1495),
    ]
    csv_path = tmp_path / "source_scale_vs_z.csv"
    pdf_path = tmp_path / "source_scale_vs_z.pdf"

    scan.write_scale_csv(csv_path, rows)
    scan.plot_scale_vs_z(rows, pdf_path, 0.097)

    with csv_path.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert [float(row["z_m"]) for row in table] == [0.0, 3.0]
    assert float(table[0]["source_intensity_scale"]) == 0.1563
    assert int(table[0]["channel_count"]) == 100
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_read_scale_csv_round_trip(tmp_path):
    """read_scale_csv restores float z and scale values."""
    rows = [
        _row(0.0, 36.5, 0.5533, 0.1563),
        _row(3.0, 40.0, 0.5800, 0.1495),
    ]
    csv_path = tmp_path / "source_scale_vs_z.csv"
    scan.write_scale_csv(csv_path, rows)
    loaded = scan.read_scale_csv(csv_path)
    assert [row["z_m"] for row in loaded] == [0.0, 3.0]
    assert loaded[0]["source_intensity_scale"] == pytest.approx(0.1563)


def test_parse_compare_csv_specs():
    """PATH=LABEL specs split on the first equals sign only."""
    specs = scan.parse_compare_csv_specs(
        ["/a/b.csv=iso model", "relative.csv=diffuser"]
    )
    assert specs[0][1] == "iso model"
    assert specs[1][1] == "diffuser"
    with pytest.raises(ValueError):
        scan.parse_compare_csv_specs(["no-label.csv"])


def test_plot_scale_comparison_overlays_series(tmp_path):
    """The comparison PDF carries one legend entry per series."""
    rows_a = [
        {"z_m": 0.0, "source_intensity_scale": 0.156},
        {"z_m": 16.1, "source_intensity_scale": 0.126},
    ]
    rows_b = [
        {"z_m": 0.0, "source_intensity_scale": 0.157},
        {"z_m": 16.1, "source_intensity_scale": 0.130},
    ]
    output = tmp_path / "source_scale_compare.pdf"
    scan.plot_scale_comparison(
        [("isotropic point source", rows_a), ("diffuser ball", rows_b)],
        output,
    )
    assert output.read_bytes().startswith(b"%PDF")


def test_main_compare_csv_mode_skips_the_scan(tmp_path):
    """--compare-csv overlays existing CSVs without simulation input."""
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    scan.write_scale_csv(csv_a, [_row(0.0, 36.5, 0.5533, 0.156)])
    scan.write_scale_csv(csv_b, [_row(0.0, 36.5, 0.5533, 0.157)])
    out_dir = tmp_path / "out"
    scan.main(
        [
            "--compare-csv", f"{csv_a}=isotropic point source",
            "--compare-csv", f"{csv_b}=diffuser ball",
            "--output-dir", str(out_dir),
        ]
    )
    merged = out_dir / "source_scale_compare.csv"
    assert (out_dir / "source_scale_compare.pdf").exists()
    with merged.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert [row["series"] for row in table] == [
        "isotropic point source",
        "diffuser ball",
    ]
    assert all(row["z_m"] == "0.0" for row in table)
