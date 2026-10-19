"""Tests for measured Super-K geometry loaded from a ConnectionTable."""

import json

import numpy as np
import numpy.testing as npt
import pytest

uproot = pytest.importorskip("uproot")

from lucid.geometry import SuperK, generate_detector


def _write_connection_table(path):
    """Write four deliberately unordered PMTs in ROOT's centimetre units."""
    with uproot.recreate(path) as root_file:
        root_file["ConnectionTable"] = {
            # Input order: top, barrel, bottom, barrel.
            "cableid": np.array([30, 10, 40, 20], dtype=np.int32),
            "pmtx": np.array([0.0, 100.0, 0.0, 0.0]),
            "pmty": np.array([0.0, 0.0, 0.0, 100.0]),
            "pmtz": np.array([100.0, 0.0, -100.0, 0.0]),
            "pmtflag": np.array([3, 1, 4, 2], dtype=np.int32),
        }


def test_connection_table_units_ordering_axes_and_metadata(tmp_path):
    root_path = tmp_path / "connection_table.root"
    _write_connection_table(root_path)

    detector = SuperK(
        connection_table_path=root_path,
        radius=1.0,
        height=2.0,
        n_sensors=4,
        sensor_radius=0.254,
        z_boundary=0.9,
        snap_to_wall=False,
    )

    # ROOT coordinates are cm; LUCiD coordinates are m. The loader also
    # establishes Cylinder's canonical barrel, top, bottom ordering.
    npt.assert_allclose(
        detector.all_points,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    )
    npt.assert_array_equal(detector.surfaces,
                           ["barrel", "barrel", "top", "bottom"])
    npt.assert_array_equal(detector.pmt_id, [10, 20, 30, 40])
    npt.assert_array_equal(detector.pmtflag, [1, 2, 3, 4])

    # The PMT optical axes point inward toward the water volume.
    npt.assert_allclose(
        detector.pmt_directions,
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
    )
    npt.assert_allclose(np.linalg.norm(detector.pmt_directions, axis=1), 1.0)
    assert detector.cableid_to_ID == {10: 0, 20: 1, 30: 2, 40: 3}


def test_generate_detector_resolves_connection_table_relative_to_config(tmp_path):
    root_path = tmp_path / "connection_table.root"
    _write_connection_table(root_path)
    config_path = tmp_path / "sk_test.json"
    config_path.write_text(json.dumps({
        "detector_type": "superk",
        "material": "water",
        "geometry_definitions": {
            "connection_table_path": root_path.name,
            "radius": 1.0,
            "height": 2.0,
            "n_sensors": 4,
            "sensor_radius": 0.254,
            "z_boundary": 0.9,
            "snap_to_wall": False,
        },
    }))

    detector = generate_detector(config_path)

    assert isinstance(detector, SuperK)
    assert detector.geometry_type == "cylinder"
    assert detector.connection_table_path == str(root_path)
    npt.assert_array_equal(detector.pmt_id, [10, 20, 30, 40])


def test_duplicate_cable_ids_are_rejected(tmp_path):
    root_path = tmp_path / "duplicate.root"
    with uproot.recreate(root_path) as root_file:
        root_file["ConnectionTable"] = {
            "cableid": np.array([7, 7], dtype=np.int32),
            "pmtx": np.array([100.0, -100.0]),
            "pmty": np.array([0.0, 0.0]),
            "pmtz": np.array([0.0, 0.0]),
        }

    with pytest.raises(ValueError, match="cableid values must be unique"):
        SuperK(root_path, 1.0, 2.0, 2, 0.254, snap_to_wall=False)
