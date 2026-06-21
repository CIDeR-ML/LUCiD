"""Invariant: segments whose start point is outside the detector volume must
not appear in ``seg.h5/event_NNN/sensor_hits/``.

The propagation in ``simulator._common_propagation`` zeroes out continuing
factors for photons that step outside the detector (line ~350 in
``simulator.py``), so a segment that emits photons exclusively from outside
the active volume should produce zero detected weight per sensor — and thus
no row in the sparsified ``sensor_hits`` arrays.

This file checks that invariant two ways:

1. The validator helper :func:`find_outside_start_violations` correctly
   classifies a synthetic event's hits.
2. The writer ``save_step_event_v3`` round-trips the ``sensor_hits`` subgroup
   in a way the validator can re-check on disk.

The same validator can be pointed at any real ``wc_step_NNNN.h5`` produced
by ``lucid-run-job`` to confirm the invariant holds end-to-end after a real
PhotonSim+LUCiD pass.
"""
import h5py
import numpy as np
import pytest

from lucid.sources.v3_writer import save_step_event_v3
from tests.io._v3_event_fixture import build_synthetic_event


def _segment_starts_inside(start_xyz, detector_bounds):
    """Vectorised inside-detector check on segment start points.

    Mirrors the start-point branch of ``_compute_contained`` (event_io.py
    around line 2848) but operates on raw ``(N, 3)`` arrays so the test
    does not need to reconstruct a full ``event_dict``.
    """
    sx, sy, sz = start_xyz[:, 0], start_xyz[:, 1], start_xyz[:, 2]
    kind = detector_bounds['type']
    if kind == 'cylinder':
        R = float(detector_bounds['radius'])
        HZ = float(detector_bounds['height']) / 2.0
        return (np.sqrt(sx * sx + sy * sy) <= R) & (np.abs(sz) <= HZ)
    if kind == 'sphere':
        R = float(detector_bounds['radius'])
        return np.sqrt(sx * sx + sy * sy + sz * sz) <= R
    if kind == 'box':
        HL = float(detector_bounds['length']) / 2.0
        HW = float(detector_bounds['width']) / 2.0
        HH = float(detector_bounds['height']) / 2.0
        return (np.abs(sx) <= HL) & (np.abs(sy) <= HW) & (np.abs(sz) <= HH)
    raise ValueError(f"Unknown detector_bounds type: {kind!r}")


def find_outside_start_violations(seg_h5_path, detector_bounds, event_idx=0):
    """Return segment indices that have at least one ``sensor_hits`` entry
    yet whose start point lies outside ``detector_bounds``.

    Empty result == invariant holds for ``event_idx``. Designed to be called
    on real datasets too: the user can point this at a generated
    ``wc_step_NNNN.h5`` after running ``lucid-run-job`` elsewhere.
    """
    with h5py.File(seg_h5_path, 'r') as f:
        grp = f[f'event_{event_idx:03d}']
        if 'sensor_hits' not in grp:
            return np.array([], dtype=np.int32)  # nothing to check; no hits stored
        n_segments = int(grp.attrs['n_segments'])
        if n_segments == 0:
            assert grp['sensor_hits/segment_idx'].shape == (0,)
            return np.array([], dtype=np.int32)
        start_xyz = np.column_stack([
            grp['start_x'][:], grp['start_y'][:], grp['start_z'][:]
        ]).astype(np.float64)
        inside = _segment_starts_inside(start_xyz, detector_bounds)
        seg_with_hit = np.unique(grp['sensor_hits/segment_idx'][:])
    # Violators: segments that are referenced by some hit but whose start is outside.
    violators = seg_with_hit[~inside[seg_with_hit]]
    return violators.astype(np.int32)


# Picked so that the synthetic fixture's segment starts at (0,0,0), (1,0,0)
# are clearly inside while a relocated segment at (0,0,12) is clearly outside.
_TEST_BOUNDS = {'type': 'cylinder', 'radius': 5.0, 'height': 10.0}


def _write_event(tmp_path, event_dict):
    seg_path = tmp_path / 'wc_step_0000.h5'
    with h5py.File(seg_path, 'w') as f:
        save_step_event_v3(f, event_dict, seq_idx=0)
    return seg_path


def _add_segment_sensor_hits(event_dict, segment_idx, sensor_idx, PE, T):
    event_dict['segment_sensor_hits'] = {
        'segment_idx': np.asarray(segment_idx, dtype=np.int32),
        'sensor_idx':  np.asarray(sensor_idx, dtype=np.uint16),
        'PE':          np.asarray(PE, dtype=np.float32),
        'T':           np.asarray(T, dtype=np.float32),
    }


def test_outside_start_segments_have_no_hits(tmp_path):
    """Positive: hits only on inside-start segments → no violation."""
    _, ev, _ = build_synthetic_event()
    # The fixture's 4 segments start at (0,0,0), (1,0,0), (5,0,0), (6,0,0).
    # Push segments 2 and 3 outside by translating their starts (and ends)
    # to z=+12 m, well outside the test cylinder (HZ = 5 m).
    seg = ev['segments']
    seg['start_z'] = seg['start_z'].copy()
    seg['end_z'] = seg['end_z'].copy()
    seg['start_z'][2:] = 12.0
    seg['end_z'][2:] = 12.0
    # Inside-start segments are 0, 1. Build hits referencing only those.
    _add_segment_sensor_hits(
        ev,
        segment_idx=[0, 0, 1],
        sensor_idx=[5, 6, 7],
        PE=[1.0, 2.0, 3.0],
        T=[10.0, 11.0, 12.0],
    )
    seg_path = _write_event(tmp_path, ev)
    violators = find_outside_start_violations(seg_path, _TEST_BOUNDS)
    assert violators.size == 0, (
        f"Inside-start hits should never violate the invariant; got {violators}")


def test_validator_flags_outside_start_with_hit(tmp_path):
    """Negative: hits on an outside-start segment → validator surfaces it.

    Guards against silent regressions: if a future propagation change ever
    let outside-start photons leak into ``sensor_hits``, this test would
    have flagged the validator unable to detect the leak.
    """
    _, ev, _ = build_synthetic_event()
    seg = ev['segments']
    seg['start_z'] = seg['start_z'].copy()
    seg['end_z'] = seg['end_z'].copy()
    seg['start_z'][2:] = 12.0  # segments 2 and 3 are outside-start
    seg['end_z'][2:] = 12.0
    # Inject a forbidden hit on segment 2 alongside legitimate inside-start hits.
    _add_segment_sensor_hits(
        ev,
        segment_idx=[0, 1, 2],
        sensor_idx=[5, 6, 9],
        PE=[1.0, 2.0, 0.5],
        T=[10.0, 11.0, 14.0],
    )
    seg_path = _write_event(tmp_path, ev)
    violators = find_outside_start_violations(seg_path, _TEST_BOUNDS)
    assert violators.tolist() == [2], (
        f"Validator should flag segment 2; got {violators.tolist()}")


def test_no_sensor_hits_subgroup_is_passthrough(tmp_path):
    """Default-off path: no ``sensor_hits`` subgroup → validator is a no-op.

    Confirms the opt-in design: existing datasets without the new subgroup
    don't trip the validator.
    """
    _, ev, _ = build_synthetic_event()
    # Don't add segment_sensor_hits — writer should skip the subgroup.
    seg_path = _write_event(tmp_path, ev)
    with h5py.File(seg_path, 'r') as f:
        assert 'sensor_hits' not in f['event_000']
        assert 'has_segment_sensor_map' not in f['event_000'].attrs
    violators = find_outside_start_violations(seg_path, _TEST_BOUNDS)
    assert violators.size == 0
