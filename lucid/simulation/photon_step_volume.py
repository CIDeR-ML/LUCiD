"""Volume photon step — per-DOM survival for string-telescope detectors.

Replaces the single-surface photon_step for volume detectors where DOMs
float in the medium at different distances along the photon ray. Each DOM
gets its own reach probability based on its distance from the photon origin.

No reflection — photons either scatter in the medium, are absorbed, or
are detected by a DOM. The detection weight per DOM is:

    charge_j = overlap(d_perp_j) × exp(-d_j / λ_scat) × exp(-d_j / λ_abs)

where exp(-d/λ_scat) is the probability of remaining on the ray (STE soft
weight, not a hard gate on scatter_dist), and exp(-d/λ_abs) is the weight
attenuation from absorption. These are kept separate for clarity and to
support independent wavelength dependence.

The continuing factor is:

    continuing = (1 - Σ_j charge_j) × exp(-scatter_dist / λ_abs)

Weight budget: deposited + continuing ≤ 1 (difference is absorbed).

Scattering is a two-channel Rayleigh+Mie mixture (forward-peaked Mie is
physically required for ice). The two channels combine by RATE:

    1/λ_scat_eff = 1/λ_R + 1/λ_M ,   p_mie = (1/λ_M) / (1/λ_R + 1/λ_M)

λ_scat_eff drives the reach + free path (so λ_R and λ_M both carry an exact
reparam gradient through the deposit), and the scattered direction is drawn
from the Rayleigh phase function with probability (1-p_mie) and the
Henyey-Greenstein (Mie) phase function — reparametrised in the asymmetry g,
so g carries an exact pathwise gradient — with probability p_mie. This
mirrors the validated water ``photon_step`` mixture.
"""

import jax
import jax.numpy as jnp
from functools import partial

from lucid.utils import normalize
from lucid.simulation.optics import (
    sample_scatter_distance,
    compute_scatter_direction,
)
from lucid.wavelength.scattering import compute_mie_scatter_direction


def photon_step_volume(
    position,
    direction,
    time,
    per_dom_distances,
    per_dom_overlaps,
    scatter_length,
    mie_scatter_length,
    absorption_length,
    segment_length,
    rng_key,
    speed_of_light,
    g,
):
    """Single volume photon step with per-DOM survival (Rayleigh+Mie scatter).

    Parameters
    ----------
    position : (3,)           current photon position
    direction : (3,)          current photon direction
    time : scalar             current photon time (ns)
    per_dom_distances : (max_dom,)  distance along ray to each candidate DOM (meters)
    per_dom_overlaps : (max_dom,)   overlap weight for each candidate DOM
    scatter_length : scalar   Rayleigh λ_R for this photon (meters)
    mie_scatter_length : scalar  Mie λ_M for this photon (meters)
    absorption_length : scalar  λ_abs for this photon (meters)
    segment_length : scalar   max propagation distance this step (envelope exit)
    rng_key : PRNGKey
    speed_of_light : scalar   m/ns in medium
    g : scalar                Henyey-Greenstein asymmetry for the Mie channel [0,1]

    Returns
    -------
    new_position : (3,)
    new_direction : (3,)
    new_time : scalar
    per_dom_charges : (max_dom,)   per-DOM detection weights (NOT multiplied by intensity)
    continuing_factor : scalar     photon weight surviving to next K step
    """
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Two-channel scatter: combine Rayleigh + Mie by RATE (NOT min); the effective
    # free path drives both the per-DOM reach and the sampled free path, so λ_R and
    # λ_M each carry an exact reparam gradient through the deposit.
    mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
    inv_total = 1.0 / scatter_length + 1.0 / mie_safe
    scatter_length_eff = 1.0 / inv_total
    p_mie = (1.0 / mie_safe) / inv_total              # P(a scatter is Mie)

    safe_distances = jnp.maximum(per_dom_distances, 0.0)

    # Scatter reach: probability photon is still on this ray at distance d
    scatter_reach = jnp.exp(-safe_distances / scatter_length_eff)
    # Absorption: weight attenuation to distance d (reduces intensity, not direction)
    absorb_weight = jnp.exp(-safe_distances / absorption_length)

    # Per-DOM charge: detection probability × surviving weight
    per_dom_charges = per_dom_overlaps * scatter_reach * absorb_weight

    total_detected = jnp.sum(per_dom_charges)

    # Scatter: sample how far the photon goes before scattering. eps>0 floors the
    # truncated-exponential log so a long envelope segment (segment_length ≫ λ_scat,
    # routine for a sparse telescope) keeps the distance + its optical-param gradient
    # finite (caps the sampled distance at ~16·λ_scat, whose weight exp(-16) is ~0).
    scatter_distance = sample_scatter_distance(segment_length, scatter_length_eff, k1, eps=1e-7)
    scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)

    # Continuing factor: not detected AND survived absorption
    continuing_factor = (1.0 - total_detected) * scatter_attenuation

    # New position and direction (always scatter — no wall to hit). The scattered
    # direction is a Rayleigh/Mie mixture: g enters only the Mie (HG) sampler, which
    # is reparametrised, so g's gradient is exact; the Rayleigh/Mie choice is a hard
    # Bernoulli on p_mie (same as the validated water step).
    rayleigh_dir = compute_scatter_direction(direction, k2)
    mie_dir = compute_mie_scatter_direction(direction, k2, g)
    is_mie = jax.random.uniform(k3) < p_mie
    new_direction = jnp.where(is_mie, mie_dir, rayleigh_dir)

    new_position = position + scatter_distance * normalize(direction)
    new_time = time + scatter_distance / speed_of_light

    return new_position, new_direction, new_time, per_dom_charges, continuing_factor
