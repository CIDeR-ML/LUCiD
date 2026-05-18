"""event_io.py -- backward-compatibility shim.

All logic has been moved to:
    lucid.sources.seed_utils
    lucid.sources.root_reader
    lucid.sources.event_builder
    lucid.sources.event_generation
    lucid.sources.v3_writer
    lucid.sources.v3_reader
    lucid.sources.particle_physics
    lucid.sources.legacy_io

This module re-exports every public name so existing imports are unaffected.
"""
from __future__ import annotations

# --- seed_utils ---
from lucid.sources.seed_utils import (          # noqa: F401
    derive_event_keys,
    derive_subprocess_seeds,
    T0_HALF_WINDOW_NS,
)

# --- root_reader ---
from lucid.sources.root_reader import (         # noqa: F401
    get_max_photons_per_particle,
    read_photon_data_from_photonsim,
    read_particle_data_from_photonsim,
)

# --- event_builder ---
from lucid.sources.event_builder import (       # noqa: F401
    aggregate_hits_from_segments,
    derive_particle_idx_per_track,
    derive_track_ancestor_and_interaction,
    _aggregate_from_photon_records,
    _trace_event_bucketed,
    _derive_views_from_segments,
    _DEFAULT_PAD_SIZE_BUCKETS,
    _normalize_buckets,
    _warmup_buckets,
)

# --- event_generation ---
from lucid.sources.event_generation import (    # noqa: F401
    generate_events_from_photonsim_particles,
    generate_events_from_photonsim_pileup,
)

# --- v3_writer ---
from lucid.sources.v3_writer import (           # noqa: F401
    SOURCE_TYPE_PARTICLES,
    SOURCE_TYPE_GENIE,
    _source_type_code,
    _compute_contained,
    build_interaction_metadata,
    sample_translation_vector,
    write_sensor_config_v3,
    write_hits_config_v3,
    write_edep_config_v3,
    write_labl_config_v3,
    save_sensor_event_v3,
    save_hits_event_v3,
    save_edep_event_v3,
    save_labl_event_v3,
)

# --- v3_reader ---
from lucid.sources.v3_reader import (           # noqa: F401
    list_events_v3,
    read_sensor_event_v3,
    read_hits_event_v3,
    read_edep_event_v3,
    read_labl_event_v3,
)

# --- particle_physics ---
from lucid.sources.particle_physics import (    # noqa: F401
    PARTICLE_MASSES,
    get_pdg_code,
    get_particle_mass,
    extract_particle_properties,
    analyze_loaded_particle,
    analyze_event_directory,
    momentum_to_angles_and_energy,
    analyze_event_kinematics,
    print_event_kinematics,
    derive_particle_interaction_idx,
)

# --- legacy_io ---
from lucid.sources.legacy_io import (           # noqa: F401
    full_to_sparse,
    sparse_to_full,
    save_single_event,
    load_single_event,
    get_random_root_entry_index,
    read_photon_data_from_root,
    generate_events_from_root,
)
