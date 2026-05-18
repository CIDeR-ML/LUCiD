"""Stage 3 byte-identity gate: hits file is a downstream view of
edep/event_NNN/sensor_hits/ plus the segment→track→particle map.

Two checks:

1. ``test_aggregator_matches_oracle``: unit-tests
   ``aggregate_hits_from_segments`` against a hand-rolled NumPy oracle
   on a synthetic mini-event (3 segments → 2 particles). Verifies the
   composition rules exactly:

       PE per particle = Σ over particle's segments of pe_per_seg
       T  per particle = min over particle's segments of t_per_seg
                         (preserving the "0 = no hit" sentinel)

2. ``test_hits_aggregates_from_edep_sensor_hits`` (skip-if-no-data):
   on an already-saved data-mode dataset (env var
   ``LUCID_HITS_FROM_EDEP_DATASET``), reads the hits file, the edep file,
   ``labl.h5`` and asserts that aggregating
   ``edep/event_NNN/sensor_hits/`` over the segment→track (from
   ``edep/event_NNN/track_idx``) and track→particle (from
   ``labl/event_NNN/per_track/particle_idx``) maps reproduces every
   column of the hits file event group (``PE``, ``T``, ``particle_idx``,
   ``sensor_idx``) bit-identically. Drive end-to-end via:

       LUCID_HITS_FROM_EDEP_DATASET=/path/to/dataset \\
           pytest tests/io/test_hits_byte_identity.py

   where ``/path/to/dataset`` contains
   ``{sensor,hits,edep,labl}/wc_*_NNNN.h5`` produced by ``lucid-run-job``.

Times in saved files are post-t0-shift, so the "no hit" sentinel is
``T == 0`` (NOT ``T > 0``); hits can be negative for early arrivals.
"""
from __future__ import annotations

import os
from glob import glob

import h5py
import numpy as np
import pytest

from lucid.sources.event_io import (
    aggregate_hits_from_segments,
    _aggregate_from_photon_records,
)


def test_aggregator_matches_oracle():
    """Synthetic 3-segments-into-2-particles aggregation."""
    n_sensors = 4
    # 3 segments, 4 sensors. Hand-pick PE/T values.
    pe_per_seg = np.array([
        [1.0, 0.0, 2.0, 0.0],   # seg 0 hits sensors 0,2
        [0.0, 3.0, 1.0, 0.0],   # seg 1 hits sensors 1,2
        [0.0, 0.0, 0.0, 4.0],   # seg 2 hits sensor 3
    ], dtype=np.float32)
    # T = first arrival; 0 = no hit. Use values that exercise min.
    t_per_seg = np.array([
        [10.0, 0.0, 20.0, 0.0],
        [0.0, 30.0, 5.0, 0.0],   # earlier on sensor 2 than seg 0
        [0.0, 0.0, 0.0, 40.0],
    ], dtype=np.float32)

    # Two tracks: track 0 owns segments 0, 1; track 1 owns segment 2.
    track_idx_per_segment = np.array([0, 0, 1], dtype=np.int32)
    # Two particles: track 0 → particle 0; track 1 → particle 1.
    particle_idx_per_track = np.array([0, 1], dtype=np.int32)

    PE_pp, T_pp = aggregate_hits_from_segments(
        pe_per_seg, t_per_seg,
        track_idx_per_segment, particle_idx_per_track,
        n_particles=2, n_sensors=n_sensors)

    expected_PE = np.array([
        [1.0, 3.0, 3.0, 0.0],   # seg 0 + seg 1
        [0.0, 0.0, 0.0, 4.0],   # seg 2
    ], dtype=np.float32)
    expected_T = np.array([
        [10.0, 30.0, 5.0, 0.0],   # min(10, .) = 10; min(., 30) = 30; min(20, 5) = 5; 0
        [0.0, 0.0, 0.0, 40.0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(PE_pp, expected_PE)
    np.testing.assert_array_equal(T_pp, expected_T)


def test_aggregator_drops_orphan_tracks():
    """Segments belonging to a track with particle_idx == -1 are dropped."""
    pe_per_seg = np.array([
        [1.0, 0.0],
        [2.0, 3.0],   # belongs to orphan track
    ], dtype=np.float32)
    t_per_seg = np.array([
        [5.0, 0.0],
        [10.0, 7.0],
    ], dtype=np.float32)
    track_idx_per_segment = np.array([0, 1], dtype=np.int32)
    particle_idx_per_track = np.array([0, -1], dtype=np.int32)  # track 1 orphan

    PE_pp, T_pp = aggregate_hits_from_segments(
        pe_per_seg, t_per_seg,
        track_idx_per_segment, particle_idx_per_track,
        n_particles=1, n_sensors=2)

    np.testing.assert_array_equal(PE_pp, np.array([[1.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(T_pp, np.array([[5.0, 0.0]], dtype=np.float32))


def test_aggregator_handles_empty_inputs():
    """No segments → all-zero outputs; no particles → empty outputs."""
    n_sensors = 3
    PE_pp, T_pp = aggregate_hits_from_segments(
        np.zeros((0, n_sensors), dtype=np.float32),
        np.zeros((0, n_sensors), dtype=np.float32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        n_particles=2, n_sensors=n_sensors)
    assert PE_pp.shape == (2, n_sensors)
    np.testing.assert_array_equal(PE_pp, 0.0)
    np.testing.assert_array_equal(T_pp, 0.0)

    PE_pp, T_pp = aggregate_hits_from_segments(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        n_particles=0, n_sensors=n_sensors)
    assert PE_pp.shape == (0, n_sensors)


def test_aggregator_from_photon_records_matches_oracle():
    """Synthetic per-photon aggregation: 5 photons → 2 segments → 2 particles.

    Verifies that ``_aggregate_from_photon_records`` produces the same
    hits PE/T tensors as the dense oracle and emits the expected
    edep sparse triplets.
    """
    n_sensors = 2
    n_particles = 2
    # 5 photons.
    # P0: seg 0, sensor 0, weight 1.0, time 10  (QE-pass) → particle 0
    # P1: seg 0, sensor 1, weight 2.0, time  5  (QE-pass) → particle 0
    # P2: seg 1, sensor 0, weight 3.0, time 20  (QE-pass) → particle 1
    # P3: seg 1, sensor 1, weight 0.0, time inf (QE-fail) → particle 1 (drops)
    # P4: orphan (seg=-1, particle=-1), weight 0.5 (QE-pass) → drops from both
    photon_qe_weight        = np.array([1.0, 2.0, 3.0, 0.0, 0.5], dtype=np.float32)
    photon_qe_time          = np.array([10.0, 5.0, 20.0, np.inf, 15.0], dtype=np.float32)
    photon_sensor_idx       = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    photon_seg_idx_filtered = np.array([0, 0, 1, 1, -1], dtype=np.int32)
    photon_particle_idx     = np.array([0, 0, 1, 1, -1], dtype=np.int32)

    out = _aggregate_from_photon_records(
        photon_qe_weight, photon_qe_time, photon_sensor_idx,
        photon_seg_idx_filtered, photon_particle_idx,
        n_particles=n_particles, n_sensors=n_sensors)

    expected_PE = np.array([
        [1.0, 2.0],   # particle 0: P0 + P1
        [3.0, 0.0],   # particle 1: P2; P3 failed QE
    ], dtype=np.float32)
    expected_T = np.array([
        [10.0, 5.0],
        [20.0, 0.0],   # 0 = no hit
    ], dtype=np.float32)
    np.testing.assert_array_equal(out['PE_per_particle'], expected_PE)
    np.testing.assert_array_equal(out['T_per_particle'], expected_T)

    sh = out['segment_sensor_hits']
    # Expected triplet rows in (seg, sensor) lex order.
    np.testing.assert_array_equal(sh['segment_idx'], np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(sh['sensor_idx'],  np.array([0, 1, 0], dtype=np.uint16))
    np.testing.assert_array_equal(sh['PE'],          np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(sh['T'],           np.array([10.0, 5.0, 20.0], dtype=np.float32))


def test_aggregator_from_photon_records_handles_empty():
    out = _aggregate_from_photon_records(
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        n_particles=3, n_sensors=4)
    assert out['PE_per_particle'].shape == (3, 4)
    np.testing.assert_array_equal(out['PE_per_particle'], 0.0)
    np.testing.assert_array_equal(out['T_per_particle'],  0.0)
    assert out['segment_sensor_hits']['segment_idx'].size == 0


def test_aggregator_from_photon_records_sums_within_group():
    """Two photons hitting the same (segment, sensor) sum their PE; T = min."""
    out = _aggregate_from_photon_records(
        np.array([1.5, 2.5], dtype=np.float32),         # weights
        np.array([20.0, 8.0], dtype=np.float32),        # times — second photon arrives earlier
        np.array([0, 0], dtype=np.int32),               # both sensor 0
        np.array([0, 0], dtype=np.int32),               # both segment 0
        np.array([0, 0], dtype=np.int32),               # both particle 0
        n_particles=1, n_sensors=1)
    np.testing.assert_array_equal(out['PE_per_particle'], np.array([[4.0]], dtype=np.float32))
    np.testing.assert_array_equal(out['T_per_particle'],  np.array([[8.0]], dtype=np.float32))
    sh = out['segment_sensor_hits']
    np.testing.assert_array_equal(sh['PE'], np.array([4.0], dtype=np.float32))
    np.testing.assert_array_equal(sh['T'],  np.array([8.0], dtype=np.float32))


_DATASET_ENV = 'LUCID_HITS_FROM_EDEP_DATASET'


@pytest.mark.skipif(
    _DATASET_ENV not in os.environ,
    reason=f'set {_DATASET_ENV} to a lucid-run-job output dir to run',
)
def test_hits_aggregates_from_edep_sensor_hits():
    """On a saved dataset, every inst.h5 column equals the aggregator's
    output over seg/sensor_hits + segment→track→particle map."""
    root = os.environ[_DATASET_ENV]
    sensor_files = sorted(glob(os.path.join(root, 'sensor', 'wc_sensor_*.h5')))
    inst_files = sorted(glob(os.path.join(root, 'hits', 'wc_hits_*.h5')))
    seg_files = sorted(glob(os.path.join(root, 'edep', 'wc_edep_*.h5')))
    labl_files = sorted(glob(os.path.join(root, 'labl', 'wc_labl_*.h5')))
    assert sensor_files, f'no sensor files under {root}/sensor/'
    assert len(sensor_files) == len(inst_files) == len(seg_files) == len(labl_files)

    for sen_p, inst_p, seg_p, labl_p in zip(sensor_files, inst_files, seg_files, labl_files):
        with h5py.File(sen_p, 'r') as fsen, h5py.File(inst_p, 'r') as finst, \
             h5py.File(seg_p, 'r') as fseg, h5py.File(labl_p, 'r') as flabl:
            n_sensors = int(fsen['config'].attrs['n_sensors'])
            ev_names = sorted(k for k in fseg if k.startswith('event_'))
            assert ev_names, f'no events in {seg_p}'
            for ev in ev_names:
                seg, inst, labl = fseg[ev], finst[ev], flabl[ev]
                n_particles = int(inst.attrs['n_particles'])
                n_segments = int(seg.attrs['n_segments'])
                if n_segments == 0:
                    continue
                assert 'sensor_hits' in seg, (
                    f'{ev}: seg/event missing sensor_hits subgroup '
                    f'(Stage 2 should have made this mandatory)')
                seg_track_local = np.asarray(seg['track_idx'])
                track_to_particle = np.asarray(labl['per_track']['particle_idx'])
                sh = seg['sensor_hits']
                sh_seg = np.asarray(sh['segment_idx'])
                sh_sensor = np.asarray(sh['sensor_idx'])
                sh_pe = np.asarray(sh['PE'])
                sh_t = np.asarray(sh['T'])

                pe_per_seg = np.zeros((n_segments, n_sensors), dtype=np.float32)
                t_per_seg = np.zeros((n_segments, n_sensors), dtype=np.float32)
                pe_per_seg[sh_seg, sh_sensor] = sh_pe
                t_per_seg[sh_seg, sh_sensor] = sh_t

                # Disk T values are post-t0-shift: 0 = no hit; hits can be
                # negative. The kernel-side aggregator runs pre-shift (0 = no
                # hit, hits > 0). Inline the post-shift variant here.
                PE_pp = np.zeros((n_particles, n_sensors), dtype=np.float32)
                T_pp_inf = np.full((n_particles, n_sensors), np.inf, dtype=np.float32)
                pidx_per_seg = track_to_particle[seg_track_local]
                valid_seg = pidx_per_seg >= 0
                valid_pidx = pidx_per_seg[valid_seg].astype(np.int64)
                np.add.at(PE_pp, valid_pidx, pe_per_seg[valid_seg])
                t_inf = np.where(t_per_seg != 0, t_per_seg, np.inf).astype(np.float32)
                np.minimum.at(T_pp_inf, valid_pidx, t_inf[valid_seg])
                T_pp = np.where(np.isfinite(T_pp_inf), T_pp_inf, np.float32(0.0))

                PE_inst = np.zeros((n_particles, n_sensors), dtype=np.float32)
                T_inst = np.zeros((n_particles, n_sensors), dtype=np.float32)
                inst_p_arr = np.asarray(inst['particle_idx'])
                inst_s_arr = np.asarray(inst['sensor_idx'])
                PE_inst[inst_p_arr, inst_s_arr] = np.asarray(inst['PE'])
                T_inst[inst_p_arr, inst_s_arr] = np.asarray(inst['T'])

                assert PE_pp.tobytes() == PE_inst.tobytes(), (
                    f'{seg_p}/{ev}: PE bit-mismatch '
                    f'(max|Δ|={float(np.abs(PE_pp - PE_inst).max())})')
                assert T_pp.tobytes() == T_inst.tobytes(), (
                    f'{seg_p}/{ev}: T bit-mismatch '
                    f'(max|Δ|={float(np.abs(T_pp - T_inst).max())})')
