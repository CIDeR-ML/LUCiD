"""Unit tests for derive_particle_idx_per_track (genealogy parent-chain walk)."""
import numpy as np

from lucid.sources.event_io import derive_particle_idx_per_track

from tests._v3_event_fixture import build_synthetic_event


def test_direct_and_indirect_match():
    """Track matching directly, track matching via parent, and a track that
    fails to reach any categorized particle.
    """
    _, ev, _ = build_synthetic_event()
    ev['meaningful_tracks'][300] = {
        'track_id': 300, 'parent_id': 999, 'pdg': 22,  # 999 not in tracks
        'initial_energy': 1.0, 'n_cherenkov': 0, 'n_segments': 0,
    }
    out = derive_particle_idx_per_track(ev)
    # track 100 -> particle 0 (direct), 150 -> particle 0 (via 100), 200 -> particle 1, 300 -> orphan
    np.testing.assert_array_equal(out, [0, 0, 1, -1])


def test_empty_inputs():
    """Zero tracks, zero particles."""
    out = derive_particle_idx_per_track({'meaningful_tracks': {}, 'particles': []})
    assert out.shape == (0,)
    assert out.dtype == np.int32


def test_cycle_protection():
    """Cyclic parent_id values shouldn't loop forever."""
    ev = {
        'particles': [{'genealogy': [9], 'track_info': {'category': 0}}],
        'meaningful_tracks': {
            1: {'track_id': 1, 'parent_id': 2, 'pdg': 13,
                'initial_energy': 1.0, 'n_cherenkov': 0, 'n_segments': 0},
            2: {'track_id': 2, 'parent_id': 1, 'pdg': 13,
                'initial_energy': 1.0, 'n_cherenkov': 0, 'n_segments': 0},
        },
    }
    out = derive_particle_idx_per_track(ev)
    # Cycle never reaches a categorized particle → -1
    np.testing.assert_array_equal(out, [-1, -1])


def test_parent_zero_terminates():
    """parent_id == 0 (primary) terminates the walk cleanly."""
    ev = {
        'particles': [{'genealogy': [1], 'track_info': {'category': 0}}],
        'meaningful_tracks': {
            5: {'track_id': 5, 'parent_id': 0, 'pdg': 13,  # no match, parent is 0
                'initial_energy': 1.0, 'n_cherenkov': 0, 'n_segments': 0},
        },
    }
    out = derive_particle_idx_per_track(ev)
    np.testing.assert_array_equal(out, [-1])
