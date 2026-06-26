"""Backwards-compatibility shim -- re-exports from lucid.sources.*

All functions that previously lived here have been moved to
``lucid.sources.siren_rays``, ``lucid.sources.calibration_sources``,
and ``lucid.sources.event_io`` as part of the Phase 2.2 refactor.

New code should import from those modules directly.
"""

# --- siren_rays ---
from lucid.sources.siren_rays import (          # noqa: F401
    generate_random_cone_vectors,
    denormalize_log_predictions,
    normalize_inputs_jit,
    predict_t0,
    predict_t0_wrapper,
)

# --- calibration_sources ---
from lucid.sources.calibration_sources import (  # noqa: F401
    get_isotropic_rays,
    get_isotropic_rays_random,
    generate_laser_photons,
    setup_calibration_generator,
    generate_random_direction,
    generate_random_vertex,
    IsotropicSource,
    LaserSource,
    isotropic_source,
    laser_source,
)

# --- event_io ---
from lucid.sources.event_io import (             # noqa: F401
    get_max_photons_per_particle,
    generate_events_from_root,
    generate_multi_folder_events,
    read_photon_data_from_photonsim,
    read_event_data_from_photonsim,
    read_particle_data_from_photonsim,
    generate_events_from_photonsim,
    generate_events_from_photonsim_particles,
    # Moved from utils in Phase 2.5
    save_single_event,
    load_single_event,
    get_random_root_entry_index,
    read_photon_data_from_root,
    get_pdg_code,
    get_particle_mass,
    save_single_event_with_extended_info,
    save_single_event_with_particle_info,
    merge_event_files,
    read_multi_folder_events,
    read_event_file,
    extract_particle_properties,
    analyze_loaded_particle,
    analyze_event_directory,
    PARTICLE_MASSES,
    momentum_to_angles_and_energy,
    analyze_event_kinematics,
    print_event_kinematics,
    full_to_sparse,
    sparse_to_full,
)

# --- utils (shared math) ---
from lucid.utils import (                        # noqa: F401
    jax_rotate_vector_local,
    normalize,
    generate_orthonormal_basis,
)
