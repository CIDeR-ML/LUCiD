"""Shared synthetic event_dict fixture for v3 writer/reader tests.

Builds a small event with 2 particles, 3 tracks, 4 segments and a handful of
sensor hits. Structured to match what ``generate_events_from_photonsim_particles``
produces just before calling the v3 save functions.
"""
import numpy as np


def build_synthetic_event(source_event_idx=0, t0=7.5, n_sensors=20):
    """Return (config_meta, event_dict, sensor_positions) for tests."""
    # Two categorized particles
    particles = [
        {
            'genealogy': [100],
            'extended_genealogy': [100],
            'track_info': {'category': 0},  # Primary
        },
        {
            'genealogy': [100, 200],
            'extended_genealogy': [100, 150, 200],
            'track_info': {'category': 1},  # DecayElectron
        },
    ]

    # Three meaningful tracks:
    # - track 100 -> particle 0 (matches genealogy[-1] = 100)
    # - track 150 -> no direct match, but its parent chain 150 -> 100 reaches particle 0
    # - track 200 -> particle 1 (matches genealogy[-1] = 200)
    meaningful_tracks = {
        100: {'track_id': 100, 'parent_id': 0,   'pdg': 13,  'initial_energy': 1000.0, 'n_cherenkov': 3, 'n_segments': 2},
        150: {'track_id': 150, 'parent_id': 100, 'pdg': 22,  'initial_energy':   5.0, 'n_cherenkov': 0, 'n_segments': 0},
        200: {'track_id': 200, 'parent_id': 100, 'pdg': 11,  'initial_energy': 100.0, 'n_cherenkov': 2, 'n_segments': 2},
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

    event_dict = {
        'source_event_idx': source_event_idx,
        'n_particles': 2,
        'particles': particles,
        'meaningful_tracks': meaningful_tracks,
        'segments': segments,
        't0': t0,
        'PE_per_particle': PE_per_particle,
        'T_per_particle': T_per_particle,
        'PE_reco': PE_reco,
        'T_reco': T_reco,
        'overall_light_containment': 0.92,
        'light_containment_by_particle': np.array([0.95, 0.85], dtype=np.float32),
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
