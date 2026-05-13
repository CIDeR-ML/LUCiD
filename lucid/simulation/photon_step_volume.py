"""Volume photon step — per-DOM survival for string-telescope detectors.

Replaces the single-surface photon_step for volume detectors where DOMs
float in the medium at different distances along the photon ray. Each DOM
gets its own reach probability based on its distance from the photon origin.

No reflection — photons either scatter in the medium, are absorbed, or
are detected by a DOM. The detection weight per DOM is:

    charge_j = overlap(d_j) × exp(-dist_j / λ_scat) × intensity × survival

The continuing factor is:

    continuing = (1 - Σ_j overlap_j × reach_j) × exp(-scatter_dist / λ_abs)

This is fully differentiable — no STE branching needed since there is no
wall/surface to "reach or scatter before."
"""

import jax
import jax.numpy as jnp
from functools import partial

from lucid.simulation.optics import (
    normalize,
    sample_scatter_distance,
    compute_scatter_direction,
)


def photon_step_volume(
    position,
    direction,
    time,
    per_dom_distances,
    per_dom_overlaps,
    scatter_length,
    absorption_length,
    segment_length,
    rng_key,
    speed_of_light,
):
    """Single volume photon step with per-DOM survival.

    Parameters
    ----------
    position : (3,)           current photon position
    direction : (3,)          current photon direction
    time : scalar             current photon time (ns)
    per_dom_distances : (max_dom,)  distance along ray to each candidate DOM (meters)
    per_dom_overlaps : (max_dom,)   overlap weight for each candidate DOM
    scatter_length : scalar   λ_scat for this photon (meters)
    absorption_length : scalar  λ_abs for this photon (meters)
    segment_length : scalar   max propagation distance this step (envelope exit)
    rng_key : PRNGKey
    speed_of_light : scalar   m/ns in medium

    Returns
    -------
    new_position : (3,)
    new_direction : (3,)
    new_time : scalar
    per_dom_charges : (max_dom,)   per-DOM detection weights (NOT multiplied by intensity)
    continuing_factor : scalar     photon weight surviving to next K step
    """
    k1, k2 = jax.random.split(rng_key)

    # Per-DOM reach probability: exp(-dist_j / λ_scat)
    # Clamp negative distances to 0 (DOM behind the photon)
    safe_distances = jnp.maximum(per_dom_distances, 0.0)
    per_dom_reach = jnp.exp(-safe_distances / scatter_length)

    # Per-DOM detection weight: overlap × reach
    per_dom_charges = per_dom_overlaps * per_dom_reach

    # Total detection probability (summed over all DOMs this step)
    total_detected = jnp.sum(per_dom_charges)
    total_detected = jnp.minimum(total_detected, 1.0)

    # Scatter: sample how far the photon goes before scattering
    scatter_distance = sample_scatter_distance(segment_length, scatter_length, k1)
    scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)

    # Continuing factor: not detected AND survived absorption
    continuing_factor = (1.0 - total_detected) * scatter_attenuation

    # New position and direction (always scatter — no wall to hit)
    new_position = position + scatter_distance * normalize(direction)
    new_direction = compute_scatter_direction(direction, k2)
    new_time = time + scatter_distance / speed_of_light

    return new_position, new_direction, new_time, per_dom_charges, continuing_factor
