"""Roundtrip tests for the v4 pile-up schema additions.

Uses `build_synthetic_pileup_event` (2 vertices) to exercise:
  * per_interaction/ with n_interactions > 1, distinct t0/vertex/source_type
  * per_particle/interaction_idx correctly FK's into per_interaction/
  * per_track/interaction ranks primaries by track_id so vertex-0 tracks
    come first
"""
import h5py
import numpy as np
import pytest

from lucid.sources.event_io import (
    write_sensor_config_v3, write_inst_config_v3,
    write_seg_config_v3, write_labl_config_v3,
    save_sensor_event_v3, save_inst_event_v3,
    save_seg_event_v3, save_labl_event_v3,
    read_labl_event_v3, read_seg_event_v3,
)

from tests._v3_event_fixture import build_synthetic_pileup_event


@pytest.fixture
def pileup_batch(tmp_path):
    cfg, ev, sp = build_synthetic_pileup_event(t0_a=-17.0, t0_b=123.4)
    src = np.array([ev['source_event_idx']], dtype=np.uint32)
    paths = {
        'sensor': tmp_path / 'wc_sensor_0000.h5',
        'inst':   tmp_path / 'wc_inst_0000.h5',
        'seg':    tmp_path / 'wc_seg_0000.h5',
        'labl':   tmp_path / 'wc_labl_0000.h5',
    }
    with h5py.File(paths['sensor'], 'w') as fs, \
         h5py.File(paths['inst'],   'w') as fi, \
         h5py.File(paths['seg'],    'w') as fg, \
         h5py.File(paths['labl'],   'w') as fl:
        write_sensor_config_v3(fs, cfg, src, sp)
        write_inst_config_v3(fi, cfg, src, sp)
        write_seg_config_v3(fg, cfg, src)
        write_labl_config_v3(fl, cfg, src)
        save_sensor_event_v3(fs, ev, seq_idx=0)
        save_inst_event_v3(fi, ev, seq_idx=0)
        save_seg_event_v3(fg, ev, seq_idx=0)
        save_labl_event_v3(fl, ev, seq_idx=0)
    return paths, cfg, ev


def test_per_interaction_has_two_rows(pileup_batch):
    paths, cfg, ev = pileup_batch
    labl = read_labl_event_v3(str(paths['labl']), 0)
    pi = labl['per_interaction']
    assert len(pi['t0']) == 2
    # Rank 0 (mu- primary, track 100) carries vertex A's t0
    # Rank 1 (pi+ primary, track 300) carries vertex B's t0
    np.testing.assert_allclose(pi['t0'], [-17.0, 123.4], atol=1e-5)
    np.testing.assert_array_equal(pi['source_type'], [0, 1])
    np.testing.assert_allclose(pi['vertex_x'], [0.1, -1.0], atol=1e-5)
    np.testing.assert_allclose(pi['vertex_y'], [-0.2, 2.0], atol=1e-5)
    np.testing.assert_allclose(pi['vertex_z'], [0.3, -3.0], atol=1e-5)


def test_ancestor_and_n_particles_per_interaction(pileup_batch):
    paths, cfg, ev = pileup_batch
    labl = read_labl_event_v3(str(paths['labl']), 0)
    pi = labl['per_interaction']
    # Vertex A primary = track 100; vertex B primary = track 300.
    np.testing.assert_array_equal(pi['ancestor_track_id'], [100, 300])
    # Vertex A has 2 particles (mu- + decay e-); vertex B has 1 (pi+).
    np.testing.assert_array_equal(pi['n_particles'], [2, 1])


def test_per_particle_interaction_idx(pileup_batch):
    paths, cfg, ev = pileup_batch
    labl = read_labl_event_v3(str(paths['labl']), 0)
    pp = labl['per_particle']
    # Particle 0 (mu-, gen=[100]) and 1 (e-, gen=[100,150]) trace back to
    # primary 100 → interaction 0. Particle 2 (pi+, gen=[300]) →
    # interaction 1.
    np.testing.assert_array_equal(pp['interaction_idx'], [0, 0, 1])


def test_per_event_t0_is_min_of_per_interaction(pileup_batch):
    """per_event/t0 is the earliest interaction time in the event — a
    single-scalar convenience for downstream tools. For this fixture,
    t0_a = -17.0 and t0_b = 123.4, so per_event/t0 = -17.0."""
    paths, cfg, ev = pileup_batch
    labl = read_labl_event_v3(str(paths['labl']), 0)
    t0_min = float(min(labl['per_interaction']['t0']))
    assert float(labl['per_event']['t0']) == pytest.approx(t0_min)
    assert float(labl['per_event']['t0']) == pytest.approx(-17.0)


def test_per_track_interaction_ranks_by_track_id(pileup_batch):
    paths, cfg, ev = pileup_batch
    labl = read_labl_event_v3(str(paths['labl']), 0)
    pt = labl['per_track']
    # Tracks are inserted in the event_dict in the order: 100, 150, 300.
    np.testing.assert_array_equal(pt['track_id'], [100, 150, 300])
    # Ancestor walk: 100 → 100 (itself, primary); 150 → 100; 300 → 300.
    np.testing.assert_array_equal(pt['ancestor'],   [100, 100, 300])
    # Interaction ranks: primaries sorted by track_id → [100, 300].
    # Tracks with ancestor 100 → rank 0; ancestor 300 → rank 1.
    np.testing.assert_array_equal(pt['interaction'], [0, 0, 1])


def test_seg_time_preserved_with_per_vertex_shift(pileup_batch):
    """The fixture builds segment times already in absolute detector
    frame (+t0 applied per vertex). The writer must not shift further;
    the time array on read equals the input."""
    paths, cfg, ev = pileup_batch
    seg = read_seg_event_v3(str(paths['seg']), 0)
    np.testing.assert_allclose(seg['time'], ev['segments']['time'], atol=1e-5)
