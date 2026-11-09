"""End-to-end roundtrip test for writers + readers.

Builds a synthetic event, writes it through all four save functions,
reads it back via the readers, and verifies every documented dataset
and attr is present with the right dtype / shape.
"""

import h5py
import numpy as np
import pytest

from lucid.dataset.labels import compute_containment
from lucid.dataset.writer import write_step_config, save_step_event
from lucid.dataset.reader import (
    read_sensor_event, read_hits_event,
    read_step_event, read_labl_event,
    list_events,
)

from tests.helpers.event_fixture import build_synthetic_event, write_batch


SRC_IDX = 17          # != seq_idx 0, so writing one in place of the other is visible


@pytest.fixture
def batch(tmp_path):
    cfg, ev, sensor_positions = build_synthetic_event(source_event_idx=SRC_IDX)
    cfg = dict(cfg, n_events_requested=3, digitizer_model='ski', trigger='sliding_window',
               trigger_window_ns=200.0, trigger_n_thr=25, trigger_pad_before_ns=30.0,
               trigger_pad_after_ns=40.0, selection='trigger')
    paths = write_batch(tmp_path, cfg, ev, sensor_positions, complete=True)
    return paths, cfg, ev, sensor_positions


def test_config_complete_flag_is_false_until_marked(tmp_path):
    cfg, ev, sp = build_synthetic_event()
    paths = write_batch(tmp_path, cfg, ev, sp, complete=False)
    for path in paths.values():
        with h5py.File(path, 'r') as f:
            assert not f['config'].attrs['complete']


def test_config_present_in_all_files(batch):
    paths, cfg, ev, _ = batch
    for name, path in paths.items():
        with h5py.File(path, 'r') as f:
            assert 'config' in f, f"{name} missing config group"
            assert f['config'].attrs['format_version'] == 6
            assert f['config'].attrs['run_id'].decode() == cfg['run_id'] \
                if isinstance(f['config'].attrs['run_id'], bytes) \
                else f['config'].attrs['run_id'] == cfg['run_id']
            assert np.array_equal(
                f['config/source_event_idx'][:],
                np.array([ev['source_event_idx']], dtype=np.uint32))
            a = f['config'].attrs
            assert a['complete']
            assert (a['n_events'], a['n_events_requested']) == (1, 3)
            assert (a['trigger'], a['trigger_n_thr']) == ('sliding_window', 25)
            assert (a['trigger_window_ns'], a['trigger_pad_before_ns'],
                    a['trigger_pad_after_ns']) == (200.0, 30.0, 40.0)
            assert a['selection'] == 'trigger'
    with h5py.File(paths['sensor'], 'r') as f:
        assert f['config'].attrs['digitizer_model'] == 'ski'


def test_sensor_event_roundtrip(batch):
    paths, cfg, ev, sensor_positions = batch
    sensor = read_sensor_event(str(paths['sensor']), 0)
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


def test_hits_event_roundtrip(batch):
    paths, cfg, ev, _ = batch
    inst = read_hits_event(str(paths['hits']), 0)
    assert inst['n_particles'] == 2
    # Particle 0 has 3 hits, particle 1 has 2
    assert inst['n_particle_hits'] == 5
    # FK column ordering matches the build
    np.testing.assert_array_equal(inst['particle_idx'], np.array([0, 0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(inst['sensor_idx'], np.array([0, 1, 2, 3, 4], dtype=np.uint16))
    # No save-time shift — stored T matches input T.
    expected_t = np.array([50.0, 52.0, 55.0, 60.0, 61.0], dtype=np.float32)
    assert np.allclose(inst['T'], expected_t)
    np.testing.assert_allclose(inst['PE'], [3.0, 2.0, 1.0, 5.0, 2.5])


def test_hits_event_roundtrip_keeps_t_reco(tmp_path):
    """T_reco is written next to T, value for value."""
    cfg, ev, sp = build_synthetic_event()
    ev['hits_sparse'] = dict(ev['hits_sparse'], T_reco=ev['hits_sparse']['T'] + 0.5)
    paths = write_batch(tmp_path, cfg, ev, sp)
    inst = read_hits_event(str(paths['hits']), 0)
    np.testing.assert_allclose(inst['T_reco'], np.asarray(inst['T']) + 0.5)


def test_hits_keep_float64_time(tmp_path):
    """At a supernova-scale absolute time (1e9 ns) float32 has a 64 ns step."""
    cfg, ev, sp = build_synthetic_event()
    ev['hits_sparse'] = dict(ev['hits_sparse'], T=ev['hits_sparse']['T'] + 1e9 + 0.3)
    paths = write_batch(tmp_path, cfg, ev, sp)
    inst = read_hits_event(str(paths['hits']), 0)
    assert inst['T'].dtype == np.float64
    np.testing.assert_allclose(inst['T'], [1e9 + 50.3, 1e9 + 52.3, 1e9 + 55.3,
                                           1e9 + 60.3, 1e9 + 61.3], rtol=0, atol=1e-6)


def test_step_event_roundtrip(batch):
    paths, cfg, ev, _ = batch
    seg = read_step_event(str(paths['step']), 0)
    assert seg['n_tracks'] == 3
    assert seg['n_segments'] == 4
    # First two segments belong to track 0 (local idx), next two to track 2
    np.testing.assert_array_equal(seg['track_idx'], np.array([0, 0, 2, 2], dtype=np.int32))
    # beta_start and n_cherenkov pass through untouched (aside from dtype)
    np.testing.assert_allclose(seg['beta_start'], ev['segments']['beta_start'])
    np.testing.assert_array_equal(seg['n_cherenkov'], ev['segments']['n_cherenkov'])
    # No save-time shift — stored time matches input time.
    np.testing.assert_allclose(seg['time'], ev['segments']['time'])
    # group_id is always written: the fixture omits the key, so the
    # writer falls back to a contiguous identity range. Either way it
    # must be present and an int32 of length n_segments.
    assert 'group_id' in seg
    assert seg['group_id'].dtype == np.int32
    assert len(seg['group_id']) == 4
    # geometry, energy and containment columns, value by value
    s = ev['segments']
    for k in ('start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z', 'edep'):
        np.testing.assert_allclose(seg[k], s[k], err_msg=k)
    for k in ('dir_x', 'dir_y', 'dir_z'):
        np.testing.assert_allclose(seg[k], s[k], atol=1e-3, err_msg=k)
    np.testing.assert_array_equal(seg['contained'], ev['contained_per_segment'])
    with h5py.File(paths['step'], 'r') as f:
        g = f['event_000']
        assert g['beta_start'].dtype == np.float32 and g['n_cherenkov'].dtype == np.int32
        assert g['time'].dtype == np.float64 and g['dir_x'].dtype == np.float16


def test_empty_and_dark_event_roundtrip(tmp_path):
    """An event with no tracks or segments (only dark light) still writes every table.

    step gets zero-length columns, labl a one-row per_interaction table with
    empty primary CSR lists and contained = False.
    """
    cfg, ev, sp = build_synthetic_event()
    ev = dict(ev, n_particles=0, particles=[], meaningful_tracks={},
              segments={'n_segments': 0},
              sensor_digits={k: v[:0] for k, v in ev['sensor_digits'].items()},
              hits_sparse={k: v[:0] for k, v in ev['hits_sparse'].items()},
              interaction_metadata=[dict(ev['interaction_metadata'][0], primary_track_ids=[],
                                         primary_pdgs=[], primary_energies=[])],
              primary_to_interaction={}, contained=np.bool_(False),
              contained_per_interaction=np.array([False]),
              contained_per_particle=np.zeros(0, bool),
              contained_per_segment=np.zeros(0, bool))
    paths = write_batch(tmp_path, cfg, ev, sp)
    seg = read_step_event(str(paths['step']), 0)
    assert seg['n_segments'] == 0 and seg['n_tracks'] == 0
    for k in ('start_x', 'time', 'track_idx', 'contained'):
        assert np.asarray(seg[k]).shape == (0,), k
    labl = read_labl_event(str(paths['labl']), 0)
    assert labl['n_particles'] == 0 and labl['n_tracks'] == 0
    pi = labl['per_interaction']
    np.testing.assert_array_equal(pi['primary_track_ids_offsets'], [0, 0])
    np.testing.assert_array_equal(pi['n_particles'], [0])
    assert not bool(labl['per_event']['contained'])
    assert float(labl['per_event']['t0']) == pytest.approx(ev['interaction_metadata'][0]['t0'])
    assert np.asarray(labl['per_particle']['genealogy_offsets']).tolist() == [0]
    hits = read_hits_event(str(paths['hits']), 0)
    assert hits['n_particle_hits'] == 0


def test_step_event_group_id_roundtrip(tmp_path):
    """Explicit group_id values written by the caller round-trip unchanged."""
    cfg, ev, _ = build_synthetic_event()
    # Two coarser groups: segments 0+1 → group 0, segments 2+3 → group 1.
    ev['segments']['group_id'] = np.array([0, 0, 1, 1], dtype=np.int32)
    seg_path = tmp_path / 'wc_step_0000.h5'
    src_idx = np.array([ev['source_event_idx']], dtype=np.uint32)
    with h5py.File(seg_path, 'w') as fg:
        write_step_config(fg, cfg, src_idx)
        save_step_event(fg, ev, seq_idx=0)
    seg = read_step_event(str(seg_path), 0)
    np.testing.assert_array_equal(
        seg['group_id'], np.array([0, 0, 1, 1], dtype=np.int32))


def test_labl_event_roundtrip(batch):
    paths, cfg, ev, _ = batch
    labl = read_labl_event(str(paths['labl']), 0)
    assert labl['n_particles'] == 2
    assert labl['n_tracks'] == 3
    # per_event: contained scalar + t0 = min(per_interaction/t0)
    assert bool(labl['per_event']['contained']) is True
    assert float(labl['per_event']['t0']) == pytest.approx(ev['t0'])
    # per_interaction — 1 row for this single-interaction fixture
    pi = labl['per_interaction']
    assert len(pi['t0']) == 1
    meta = ev['interaction_metadata'][0]
    assert float(pi['t0'][0]) == pytest.approx(meta['t0'])
    assert int(pi['source_type'][0]) == meta['source_type']
    np.testing.assert_allclose(
        [pi['vertex_x'][0], pi['vertex_y'][0], pi['vertex_z'][0]],
        meta['vertex_xyz'], atol=1e-6)
    # v5 per_interaction scalars
    assert int(pi['n_primaries'][0]) == 1            # only mu- is a primary
    assert int(pi['n_particles'][0]) == 2            # mu- + decay-e
    assert int(pi['neutrino_pdg'][0]) == 0           # particle-gun
    assert float(pi['neutrino_energy_MeV'][0]) == 0.0
    # v5 CSR: primary list for interaction 0 = [13 @ 1000 MeV, track_id 100]
    np.testing.assert_array_equal(pi['primary_track_ids_offsets'], [0, 1])
    np.testing.assert_array_equal(pi['primary_track_ids_data'], [100])
    np.testing.assert_array_equal(pi['primary_pdgs_offsets'], [0, 1])
    np.testing.assert_array_equal(pi['primary_pdgs_data'], [13])
    np.testing.assert_array_equal(pi['primary_energies_offsets'], [0, 1])
    np.testing.assert_allclose(pi['primary_energies_data'], [1000.0])
    # per_interaction: contained column round-trips (one row here)
    np.testing.assert_array_equal(np.asarray(pi['contained'], dtype=bool),
                                  np.array([True], dtype=bool))
    # per_particle
    pp = labl['per_particle']
    np.testing.assert_array_equal(pp['category'], np.array([0, 1], dtype=np.uint8))
    np.testing.assert_array_equal(np.asarray(pp['contained'], dtype=bool),
                                  np.array([True, False], dtype=bool))
    # interaction_idx: both particles belong to the single interaction 0.
    np.testing.assert_array_equal(pp['interaction_idx'], np.array([0, 0], dtype=np.int32))
    # per_track
    pt = labl['per_track']
    np.testing.assert_array_equal(pt['track_id'], np.array([100, 150, 200], dtype=np.int32))
    np.testing.assert_array_equal(pt['parent_id'], np.array([0, 100, 100], dtype=np.int32))
    np.testing.assert_array_equal(pt['pdg'], np.array([13, 22, 11], dtype=np.int16))
    # ancestor (parent-chain walk) unchanged across schema versions.
    # interaction column now indexes per_interaction rows (all → 0 since
    # there's only one interaction in this fixture).
    np.testing.assert_array_equal(pt['ancestor'], np.array([100, 100, 100], dtype=np.int32))
    np.testing.assert_array_equal(pt['interaction'], np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_allclose(pt['initial_energy'], [1000.0, 5.0, 100.0])
    np.testing.assert_array_equal(pt['n_cherenkov'], [3, 0, 2])
    # particle_idx: 100 -> 0 (its own genealogy), 150 -> 0 via parent 100, 200 -> 1
    np.testing.assert_array_equal(pt['particle_idx'], [0, 0, 1])
    # genealogy CSR: [100] and [100, 200]; extended: [100] and [100, 150, 200]
    np.testing.assert_array_equal(pp['genealogy_offsets'], [0, 1, 3])
    np.testing.assert_array_equal(pp['genealogy_data'], [100, 100, 200])
    np.testing.assert_array_equal(pp['ext_genealogy_offsets'], [0, 1, 4])
    np.testing.assert_array_equal(pp['ext_genealogy_data'], [100, 100, 150, 200])
    # non-SN interaction: channel sentinel -1
    np.testing.assert_array_equal(pi['interaction_channel'], [-1])


def test_list_events(batch):
    paths, _, ev, _ = batch
    for path in paths.values():
        idx = list_events(str(path))
        np.testing.assert_array_equal(idx, [ev['source_event_idx']])


def test_source_event_idx_matches_across_files(batch):
    paths, _, ev, _ = batch
    for path in paths.values():
        with h5py.File(path, 'r') as f:
            assert f['event_000'].attrs['source_event_idx'] == ev['source_event_idx']


def test_per_track_initial_direction(tmp_path):
    """per_track dir is the direction of the track's EARLIEST-time segment.

    Segment order within a track is not guaranteed to be time order, so the
    fixture stores track 100's segments with the later one first and gives
    every segment a different direction: first-stored would report +x, the
    earliest segment points along +y. Track 150 has no segments -> (0, 0, 0).
    """
    cfg, ev, sp = build_synthetic_event()
    seg = dict(ev['segments'])
    seg['time'] = np.array([101.0, 100.0, 201.0, 200.0], np.float32)
    seg['dir_x'] = np.array([1, 0, 0, 0], np.float16)
    seg['dir_y'] = np.array([0, 1, 0, 0], np.float16)
    seg['dir_z'] = np.array([0, 0, 1, -1], np.float16)
    ev = dict(ev, segments=seg)
    paths = write_batch(tmp_path, cfg, ev, sp)
    with h5py.File(paths['labl'], 'r') as f:
        pt = f['event_000/per_track']
        got = np.stack([pt['dir_x'][:], pt['dir_y'][:], pt['dir_z'][:]], axis=1)
    np.testing.assert_allclose(got, [[0, 1, 0], [0, 0, 0], [0, 0, -1]], atol=1e-6)


# --------------------------------------------------------------------------
# compute_containment (the geometry the step/labl `contained` columns carry)
# --------------------------------------------------------------------------
def _contained_event(starts, ends):
    """One primary (track 1) owning every segment, plus a second particle whose
    track (2) is not among the meaningful tracks, so it owns no segment and its
    contained flag must be False."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    n = len(starts)
    return {
        'particles': [{'genealogy': [1]}, {'genealogy': [1, 2]}],
        'meaningful_tracks': {
            1: {'track_id': 1, 'parent_id': 0, 'n_segments': n}},
        'segments': {'start_x': starts[:, 0], 'start_y': starts[:, 1], 'start_z': starts[:, 2],
                     'end_x': ends[:, 0], 'end_y': ends[:, 1], 'end_z': ends[:, 2],
                     'n_segments': n},
        'interaction_metadata': [{'primary_track_ids': [1]}],
        'primary_to_interaction': {1: 0},
    }


@pytest.mark.parametrize('bounds,inside,outside', [
    ({'type': 'cylinder', 'radius': 2.0, 'height': 4.0}, (1.9, 0.0, 1.9), (2.1, 0.0, 0.0)),
    ({'type': 'cylinder', 'radius': 2.0, 'height': 4.0}, (0.0, 1.0, 1.9), (0.0, 1.0, 2.1)),
    ({'type': 'sphere', 'radius': 3.0}, (1.7, 1.7, 1.7), (1.8, 1.8, 1.8)),
    ({'type': 'box', 'length': 2.0, 'width': 4.0, 'height': 6.0}, (0.9, 1.9, 2.9),
     (0.9, 2.1, 2.9)),
])
def test_contained_needs_both_endpoints_inside(bounds, inside, outside):
    """Per segment: in->in True, in->out False, out->in False; ANDed upward.

    Points chosen by hand on either side of each surface (|r| = 2.94 vs 3.12
    for the sphere).
    """
    origin = (0.0, 0.0, 0.0)
    ev = _contained_event([origin, inside, outside], [inside, outside, origin])
    c = compute_containment(ev, bounds)
    np.testing.assert_array_equal(c['per_segment'], [True, False, False])
    np.testing.assert_array_equal(c['per_particle'], [False, False])
    np.testing.assert_array_equal(c['per_interaction'], [False])
    assert not c['overall']
    ev = _contained_event([origin, inside], [inside, origin])
    c = compute_containment(ev, bounds)
    np.testing.assert_array_equal(c['per_segment'], [True, True])
    # particle 1 owns no segment: empty subsets are False, never vacuously True
    np.testing.assert_array_equal(c['per_particle'], [True, False])
    assert c['per_interaction'].tolist() == [True] and bool(c['overall'])


def test_contained_without_bounds_is_all_false():
    ev = _contained_event([(0, 0, 0)], [(0.1, 0, 0)])
    c = compute_containment(ev, None)
    assert c['per_segment'].tolist() == [False]
    assert c['per_particle'].tolist() == [False, False]
    assert c['per_interaction'].tolist() == [False] and not c['overall']


@pytest.mark.parametrize("model, smearing, labels", [
    ("basic", True, ("SK_like", "none")),
    ("ski", True, ("spe", "sk_gauss")),
    ("hk", True, ("spe", "hk_emg")),
    ("hk", False, ("none", "none")),
])
def test_dataset_meta_records_the_digitizers_smearing_model(model, smearing, labels):
    """Every preset was recorded as 'SK_like' charge and time smearing; ski/hk sample the SPE
    spectrum with their own PMT jitter, and basic applies no time smearing."""
    import types
    from lucid.dataset.writer import dataset_meta
    from lucid.simulation.readout import Readout
    sim = types.SimpleNamespace(
        det_geom=types.SimpleNamespace(detector_type="cylinder",
                                       medium=types.SimpleNamespace(material="water")),
        reflection_model="scalar_mix")
    meta = dataset_meta(sim, Readout.from_blocks(model), dataset_name="t", run_id="r",
                        git_commit="g", source_files=["a.root"], master_seed=1,
                        n_events_requested=1, apply_smearing=smearing, detector_bounds=None)
    assert (meta["smearing_charge_function"], meta["smearing_time_function"]) == labels
    assert meta["digitizer_model"] == model
