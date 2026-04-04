"""Simulation package — split from monolithic simulation.py."""
from lucid.simulation.simulator import setup_event_simulator
from lucid.simulation.optics import (
    normalize, jax_normalize, compute_reflection_direction,
    create_local_frame, sample_scatter_distance, solve_rayleigh_inverse_cdf,
    compute_scatter_direction, sample_cosine_hemisphere, jax_rotate_vector,
)
from lucid.simulation.photon_step import (
    photon_iteration_sample, photon_iteration_update_factors,
    photon_iteration_update_factors_safe,
)
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood,
)
