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
    photonsim_differentiable_get_rays,
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
    read_particle_data_from_photonsim,
    generate_events_from_photonsim,
    generate_events_from_photonsim_particles,
)

# --- utils (shared math) ---
from lucid.utils import (                        # noqa: F401
    jax_rotate_vector_local,
    normalize,
    generate_orthonormal_basis,
)
