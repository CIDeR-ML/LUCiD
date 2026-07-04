"""Shared synthetic event_dict fixture for writer/reader tests.

Builds a small event with 2 particles, 3 tracks, 4 segments and a handful of
sensor hits. Structured to match what ``generate_events_from_photonsim_particles``
produces just before calling the save functions.
"""
import numpy as np


def build_synthetic_event(source_event_idx=0, t0=7.5, n_sensors=20):
    """Return (config_meta, event_dict, sensor_positions) for tests."""
    # Two categorized particles
    particles = [
        {
            'genealogy': [100],
            'extended_genealogy': [100],
            'track_info': {'category': 0, 'energy': 1000.0},  # Primary mu- @ 1 GeV KE
        },
        {
            'genealogy': [100, 200],
            'extended_genealogy': [100, 150, 200],
            'track_info': {'category': 1, 'energy': 100.0},  # DecayElectron @ 100 MeV KE
        },
    ]

    # Three meaningful tracks:
    # - track 100 -> particle 0 (matches genealogy[-1] = 100)
    # - track 150 -> no direct match, but its parent chain 150 -> 100 reaches particle 0
    # - track 200 -> particle 1 (matches genealogy[-1] = 200)
    meaningful_tracks = {
        100: {'track_id': 100, 'parent_id': 0,   'pdg': 13,  'initial_energy': 1000.0, 'n_cherenkov': 3, 'n_segments': 2, 'segment_offset': 0},
        150: {'track_id': 150, 'parent_id': 100, 'pdg': 22,  'initial_energy':   5.0, 'n_cherenkov': 0, 'n_segments': 0, 'segment_offset': 2},
        200: {'track_id': 200, 'parent_id': 100, 'pdg': 11,  'initial_energy': 100.0, 'n_cherenkov': 2, 'n_segments': 2, 'segment_offset': 2},
    }

    # Four segments total: 2 for track 100, 0 for 150, 2 for 200
    segments = {
        'start_x': np.array([0.0, 1.0,  5.0, 6.0], dtype=np.float32),
        'start_y': np.array([0.0, 0.0,  0.0, 0.0], dtype=np.float32),
        'start_z': np.array([0.0, 0.0,  0.0, 0.0], dtype=np.float32),
        'end_x':   np.array([1.0, 2.0,  6.0, 7.0], dtype=np.float32),
        'end_y':   np.zeros(4, dtype=np.float32),
        'end_z':   np.zeros(4, dtype=np.float32),
        'dir_x':   np.array([1.0, 1.0,  1.0, 1.0], dtype=np.float16),
        'dir_y':   np.zeros(4, dtype=np.float16),
        'dir_z':   np.zeros(4, dtype=np.float16),
        'edep':    np.array([0.5, 0.4, 0.2, 0.1], dtype=np.float32),
        # G4-absolute times — writer must subtract t0
        'time':    np.array([100.0, 101.0, 200.0, 201.0], dtype=np.float32),
        'beta_start':   np.array([0.98, 0.96, 0.80, 0.70], dtype=np.float32),
        'n_cherenkov':  np.array([2, 1, 1, 1], dtype=np.int32),
        'n_segments': 4,
    }

    # Dense per-particle PE/T (pre-smearing) — two particles, 20 sensors
    # Particle 0 lights sensors 0..2; particle 1 lights sensors 3..4
    PE_per_particle = np.zeros((2, n_sensors), dtype=np.float32)
    T_per_particle = np.zeros((2, n_sensors), dtype=np.float32)
    PE_per_particle[0, 0] = 3.0
    PE_per_particle[0, 1] = 2.0
    PE_per_particle[0, 2] = 1.0
    PE_per_particle[1, 3] = 5.0
    PE_per_particle[1, 4] = 2.5
    # Per-particle times are G4-absolute (writer t0-shifts them)
    T_per_particle[0, 0] = 50.0
    T_per_particle[0, 1] = 52.0
    T_per_particle[0, 2] = 55.0
    T_per_particle[1, 3] = 60.0
    T_per_particle[1, 4] = 61.0

    # Sensor-level aggregates (also G4-absolute pre-shift; typically smeared)
    PE_reco = PE_per_particle.sum(axis=0).astype(np.float32)
    T_reco = np.zeros(n_sensors, dtype=np.float32)
    T_reco[:5] = [50.0, 52.0, 55.0, 60.0, 61.0]

    # v5: exactly one interaction (one G4 event fired the gun with a single
    # primary — track 100 is the only parent_id==0 track). The decay-e
    # (track 200, parent_id=100) descends from it and shares the same
    # interaction.
    event_dict = {
        'source_event_idx': source_event_idx,
        'n_particles': 2,
        'particles': particles,
        'meaningful_tracks': meaningful_tracks,
        'segments': segments,
        't0': t0,
        'primary_to_interaction': {100: 0},
        'interaction_metadata': [{
            't0': float(t0),
            'vertex_xyz': np.array([0.1, -0.2, 0.3], dtype=np.float32),
            'source_type': 0,  # SOURCE_TYPE_PARTICLES
            'neutrino_pdg': 0,
            'neutrino_energy_MeV': 0.0,
            'primary_track_ids': [100],
            'primary_pdgs':      [13],       # mu-
            'primary_energies':  [1000.0],   # MeV
        }],
        'PE_per_particle': PE_per_particle,
        'T_per_particle': T_per_particle,
        'PE_reco': PE_reco,
        'T_reco': T_reco,
        # Synthetic containment flags: writer round-trips verbatim. Picked
        # to exercise both True and False at every level so the column
        # types and shapes are checked end-to-end.
        'contained': np.bool_(True),
        'contained_per_interaction': np.array([True], dtype=bool),
        'contained_per_particle': np.array([True, False], dtype=bool),
        'contained_per_segment': np.array([True, True, False, False], dtype=bool),
    }

    sensor_positions = np.random.default_rng(0).normal(size=(n_sensors, 3)).astype(np.float32)

    config_meta = {
        'n_events': 1,
        'git_commit': 'test_commit',
        'run_id': 'test_run',
        'dataset_name': 'test_dataset',
        'file_index': 0,
        'source_file': '/tmp/fake.root',
        'lucid_master_seed': 42,
        'photonsim_seed': -1,
        'n_sensors': n_sensors,
        'detector_type': 'Cylinder',
        'material': 'water',
        'smearing_applied': False,
        'smearing_charge_function': 'none',
        'smearing_time_function': 'none',
        'label_names': ['category'],
    }

    return config_meta, event_dict, sensor_positions


def build_synthetic_pileup_event(source_event_idx=0, n_sensors=20,
                                 t0_a=-17.0, t0_b=123.4):
    """Return a 2-vertex synthetic pile-up event_dict.

    Vertex A: one primary (mu-, track_id=100) with one decay daughter
              (e-, track_id=150). Source type = particles.
    Vertex B: one primary (pi+, track_id=300). Source type = GENIE.

    t0 and vertex_xyz are stored as (n_primaries,) / (n_primaries, 3)
    arrays so the labl writer's per_interaction/ table broadcasts them
    row-per-primary. Since this fixture has one primary per vertex, the
    table has 2 rows, one per vertex, carrying the (t0_a, t0_b) values
    the caller supplies.

    Times in T_per_particle and segments['time'] are already in
    absolute detector frame (+t0 applied per primary).
    """
    # Vertex A primaries
    particles = [
        {'genealogy': [100],      'extended_genealogy': [100],
         'track_info': {'category': 0, 'energy': 1000.0}},   # mu- primary
        {'genealogy': [100, 150], 'extended_genealogy': [100, 150],
         'track_info': {'category': 1, 'energy': 100.0}},    # decay e-
        {'genealogy': [300],      'extended_genealogy': [300],
         'track_info': {'category': 0, 'energy': 500.0}},    # pi+ primary (vertex B)
    ]

    # Tracks — spanning both vertices
    meaningful_tracks = {
        100: {'track_id': 100, 'parent_id': 0,   'pdg': 13,  'initial_energy': 1000.0, 'n_cherenkov': 3, 'n_segments': 2, 'segment_offset': 0},
        150: {'track_id': 150, 'parent_id': 100, 'pdg': 11,  'initial_energy':  100.0, 'n_cherenkov': 1, 'n_segments': 1, 'segment_offset': 2},
        300: {'track_id': 300, 'parent_id': 0,   'pdg': 211, 'initial_energy':  500.0, 'n_cherenkov': 2, 'n_segments': 1, 'segment_offset': 3},
    }

    # Segments, already in absolute detector frame (+t0 applied per stream).
    # Vertex A segs (tracks 100, 150) carry t0_a; vertex B seg (track 300) carries t0_b.
    seg_times_a = np.array([100.0, 101.0, 110.0], dtype=np.float32) + np.float32(t0_a)
    seg_time_b  = np.array([200.0], dtype=np.float32) + np.float32(t0_b)
    segments = {
        'start_x': np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32),
        'start_y': np.zeros(4, dtype=np.float32),
        'start_z': np.zeros(4, dtype=np.float32),
        'end_x':   np.array([1.0, 2.0, 3.0, 6.0], dtype=np.float32),
        'end_y':   np.zeros(4, dtype=np.float32),
        'end_z':   np.zeros(4, dtype=np.float32),
        'dir_x':   np.ones(4, dtype=np.float16),
        'dir_y':   np.zeros(4, dtype=np.float16),
        'dir_z':   np.zeros(4, dtype=np.float16),
        'edep':    np.array([0.5, 0.4, 0.2, 0.3], dtype=np.float32),
        'time':    np.concatenate([seg_times_a, seg_time_b]),
        'beta_start':   np.array([0.98, 0.96, 0.70, 0.85], dtype=np.float32),
        'n_cherenkov':  np.array([2, 1, 0, 1], dtype=np.int32),
        'n_segments': 4,
    }

    # Per-particle PE/T (absolute frame — times include per-vertex t0).
    # Particle 0 (mu-, vertex A) and 1 (e-, vertex A) → + t0_a
    # Particle 2 (pi+, vertex B) → + t0_b
    PE_per_particle = np.zeros((3, n_sensors), dtype=np.float32)
    T_per_particle  = np.zeros((3, n_sensors), dtype=np.float32)
    PE_per_particle[0, 0] = 3.0; PE_per_particle[0, 1] = 2.0
    PE_per_particle[1, 2] = 1.5
    PE_per_particle[2, 3] = 4.0; PE_per_particle[2, 4] = 2.0
    T_per_particle[0, 0] = 50.0 + t0_a; T_per_particle[0, 1] = 52.0 + t0_a
    T_per_particle[1, 2] = 58.0 + t0_a
    T_per_particle[2, 3] = 60.0 + t0_b; T_per_particle[2, 4] = 61.0 + t0_b

    PE_reco = PE_per_particle.sum(axis=0).astype(np.float32)
    T_reco = np.zeros(n_sensors, dtype=np.float32)
    # Aggregated T_true = earliest per sensor (absolute frame)
    T_reco[0] = 50.0 + t0_a
    T_reco[1] = 52.0 + t0_a
    T_reco[2] = 58.0 + t0_a
    T_reco[3] = 60.0 + t0_b
    T_reco[4] = 61.0 + t0_b

    # v5 per_interaction: one row per source vertex (= per G4 event).
    # Vertex A fired one primary (mu-, track 100); its decay-e (track 150)
    # descends from it and shares interaction 0. Vertex B fired one
    # primary (pi+, track 300) alone in interaction 1. Vertex B is a
    # synthetic "GENIE" vertex for the fixture — neutrino metadata is
    # populated only for that row.
    interaction_metadata = [
        {'t0': float(t0_a),
         'vertex_xyz': np.array([0.1, -0.2, 0.3], dtype=np.float32),
         'source_type': 0,
         'neutrino_pdg': 0,
         'neutrino_energy_MeV': 0.0,
         'primary_track_ids': [100],
         'primary_pdgs':      [13],
         'primary_energies':  [1000.0]},
        {'t0': float(t0_b),
         'vertex_xyz': np.array([-1.0, 2.0, -3.0], dtype=np.float32),
         'source_type': 1,
         'neutrino_pdg': 14,
         'neutrino_energy_MeV': 1234.5,
         'primary_track_ids': [300],
         'primary_pdgs':      [211],     # pi+
         'primary_energies':  [500.0]},
    ]
    primary_to_interaction = {100: 0, 300: 1}

    event_dict = {
        'source_event_idx': source_event_idx,
        'n_particles': 3,
        'particles': particles,
        'meaningful_tracks': meaningful_tracks,
        'segments': segments,
        'interaction_metadata': interaction_metadata,
        'primary_to_interaction': primary_to_interaction,
        'PE_per_particle': PE_per_particle,
        'T_per_particle':  T_per_particle,
        'PE_reco': PE_reco,
        'T_reco':  T_reco,
        'contained': np.bool_(False),
        'contained_per_interaction': np.array([True, False], dtype=bool),
        'contained_per_particle': np.array([True, True, False], dtype=bool),
        'contained_per_segment': np.array([True, True, True, False], dtype=bool),
    }

    sensor_positions = np.random.default_rng(0).normal(size=(n_sensors, 3)).astype(np.float32)

    config_meta = {
        'n_events': 1,
        'git_commit': 'test_commit',
        'run_id': 'test_pileup',
        'dataset_name': 'test_pileup_dataset',
        'file_index': 0,
        'source_file': '/tmp/fake_pileup.root',
        'lucid_master_seed': 42,
        'photonsim_seed': -1,
        'n_sensors': n_sensors,
        'detector_type': 'Cylinder',
        'material': 'water',
        'smearing_applied': False,
        'smearing_charge_function': 'none',
        'smearing_time_function': 'none',
        'label_names': ['category'],
    }
    return config_meta, event_dict, sensor_positions
