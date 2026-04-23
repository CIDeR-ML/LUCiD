"""End-to-end roundtrip test for v3 writers + readers.

Builds a synthetic event, writes it through all four v3 save functions,
reads it back via the v3 readers, and verifies every documented dataset
and attr is present with the right dtype / shape.
"""
import os
import tempfile

import h5py
import numpy as np
import pytest

from lucid.sources.event_io import (
    write_sensor_config_v3, write_inst_config_v3,
    write_seg_config_v3, write_labl_config_v3,
    save_sensor_event_v3, save_inst_event_v3,
    save_seg_event_v3, save_labl_event_v3,
    read_sensor_event_v3, read_inst_event_v3,
    read_seg_event_v3, read_labl_event_v3,
    list_events_v3,
)

from tests._v3_event_fixture import build_synthetic_event


@pytest.fixture
def v3_batch(tmp_path):
    cfg, ev, sensor_positions = build_synthetic_event()
    paths = {
        'sensor': tmp_path / 'wc_sensor_0000.h5',
        'inst':   tmp_path / 'wc_inst_0000.h5',
        'seg':    tmp_path / 'wc_seg_0000.h5',
        'labl':   tmp_path / 'wc_labl_0000.h5',
    }
    src_idx = np.array([ev['source_event_idx']], dtype=np.uint32)
    with h5py.File(paths['sensor'], 'w') as fs, \
         h5py.File(paths['inst'],   'w') as fi, \
         h5py.File(paths['seg'],    'w') as fg, \
         h5py.File(paths['labl'],   'w') as fl:
        write_sensor_config_v3(fs, cfg, src_idx, sensor_positions)
        write_inst_config_v3(fi, cfg, src_idx, sensor_positions)
        write_seg_config_v3(fg, cfg, src_idx)
        write_labl_config_v3(fl, cfg, src_idx)
        save_sensor_event_v3(fs, ev, seq_idx=0)
        save_inst_event_v3(fi, ev, seq_idx=0)
        save_seg_event_v3(fg, ev, seq_idx=0)
        save_labl_event_v3(fl, ev, seq_idx=0)
    return paths, cfg, ev, sensor_positions


def test_config_present_in_all_files(v3_batch):
    paths, cfg, ev, _ = v3_batch
    for name, path in paths.items():
        with h5py.File(path, 'r') as f:
            assert 'config' in f, f"{name} missing config group"
            assert f['config'].attrs['format_version'] == 4
            assert f['config'].attrs['run_id'].decode() == cfg['run_id'] \
                if isinstance(f['config'].attrs['run_id'], bytes) \
                else f['config'].attrs['run_id'] == cfg['run_id']
            assert np.array_equal(
                f['config/source_event_idx'][:],
                np.array([ev['source_event_idx']], dtype=np.uint32))


def test_sensor_event_roundtrip(v3_batch):
    paths, cfg, ev, sensor_positions = v3_batch
    sensor = read_sensor_event_v3(str(paths['sensor']), 0)
    # Sensors 0..4 have PE; others don't
    assert sensor['n_hits'] == 5
    # Stored T equals input T: save_* no longer shifts — the caller is
    # expected to apply t0 in absolute detector frame before saving.
    expected_pe = np.array([3.0, 2.0, 1.0, 5.0, 2.5], dtype=np.float32)
    expected_t = np.array([50.0, 52.0, 55.0, 60.0, 61.0], dtype=np.float32)
    assert np.allclose(sensor['PE'], expected_pe)
    assert np.allclose(sensor['T'], expected_t)
    with h5py.File(paths['sensor'], 'r') as f:
        assert f['config/sensor_positions'].shape == sensor_positions.shape


def test_inst_event_roundtrip(v3_batch):
    paths, cfg, ev, _ = v3_batch
    inst = read_inst_event_v3(str(paths['inst']), 0)
    assert inst['n_particles'] == 2
    # Particle 0 has 3 hits, particle 1 has 2
    assert inst['n_particle_hits'] == 5
    # FK column ordering matches the build
    np.testing.assert_array_equal(inst['particle_idx'], np.array([0, 0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(inst['sensor_idx'], np.array([0, 1, 2, 3, 4], dtype=np.uint16))
    # No save-time shift — stored T matches input T.
    expected_t = np.array([50.0, 52.0, 55.0, 60.0, 61.0], dtype=np.float32)
    assert np.allclose(inst['T'], expected_t)


def test_seg_event_roundtrip(v3_batch):
    paths, cfg, ev, _ = v3_batch
    seg = read_seg_event_v3(str(paths['seg']), 0)
    assert seg['n_tracks'] == 3
    assert seg['n_segments'] == 4
    # First two segments belong to track 0 (local idx), next two to track 2
    np.testing.assert_array_equal(seg['track_idx'], np.array([0, 0, 2, 2], dtype=np.int32))
    # beta_start and n_cherenkov pass through untouched (aside from dtype)
    np.testing.assert_allclose(seg['beta_start'], ev['segments']['beta_start'])
    np.testing.assert_array_equal(seg['n_cherenkov'], ev['segments']['n_cherenkov'])
    # No save-time shift — stored time matches input time.
    np.testing.assert_allclose(seg['time'], ev['segments']['time'])


def test_labl_event_roundtrip(v3_batch):
    paths, cfg, ev, _ = v3_batch
    labl = read_labl_event_v3(str(paths['labl']), 0)
    assert labl['n_particles'] == 2
    assert labl['n_tracks'] == 3
    # per_event (t0 has moved to per_interaction/)
    assert 't0' not in labl['per_event']
    assert float(labl['per_event']['overall_containment']) == pytest.approx(0.92)
    # per_interaction — 1 row for this single-interaction fixture
    pi = labl['per_interaction']
    assert len(pi['t0']) == 1
    assert float(pi['t0'][0]) == pytest.approx(ev['t0'])
    assert int(pi['source_type'][0]) == ev['source_type']
    np.testing.assert_allclose(
        [pi['vertex_x'][0], pi['vertex_y'][0], pi['vertex_z'][0]],
        ev['vertex_xyz'], atol=1e-6)
    # Only one primary in this fixture (track 100), so the ancestor is 100.
    assert int(pi['ancestor_track_id'][0]) == 100
    # per_particle
    pp = labl['per_particle']
    np.testing.assert_array_equal(pp['category'], np.array([0, 1], dtype=np.uint8))
    np.testing.assert_allclose(pp['containment'], [0.95, 0.85], atol=1e-6)
    # interaction_idx: both particles trace back to primary 100 → rank 0.
    np.testing.assert_array_equal(pp['interaction_idx'], np.array([0, 0], dtype=np.int32))
    # per_track
    pt = labl['per_track']
    np.testing.assert_array_equal(pt['track_id'], np.array([100, 150, 200], dtype=np.int32))
    np.testing.assert_array_equal(pt['parent_id'], np.array([0, 100, 100], dtype=np.int32))
    np.testing.assert_array_equal(pt['pdg'], np.array([13, 22, 11], dtype=np.int16))
    # ancestor/interaction derived from parent chain:
    # track 100 (parent 0) is a primary — its own ancestor; 150/200 descend from it.
    # With a single primary in the event, interaction_id == 0 for all rows.
    np.testing.assert_array_equal(pt['ancestor'], np.array([100, 100, 100], dtype=np.int32))
    np.testing.assert_array_equal(pt['interaction'], np.array([0, 0, 0], dtype=np.int32))


def test_list_events_v3(v3_batch):
    paths, _, ev, _ = v3_batch
    for path in paths.values():
        idx = list_events_v3(str(path))
        np.testing.assert_array_equal(idx, [ev['source_event_idx']])


def test_source_event_idx_matches_across_files(v3_batch):
    paths, _, ev, _ = v3_batch
    for path in paths.values():
        with h5py.File(path, 'r') as f:
            assert f['event_000'].attrs['source_event_idx'] == ev['source_event_idx']
