"""Volume photon step — per-DOM survival for string-telescope detectors.

Replaces the single-surface photon_step for volume detectors where DOMs
float in the medium at different distances along the photon ray. Each DOM
gets its own reach probability based on its distance from the photon origin.

No reflection — photons either scatter in the medium, are absorbed, or
are detected by a DOM. The detection weight per DOM is the implicit-capture
(Rao-Blackwellised) expected charge:

    charge_j = overlap(d_perp_j) × exp(-d_j · mu_tot) × exp(-d_j / λ_abs)

where exp(-d·mu_tot) is the probability of remaining on the ray to distance d
(``mu_tot`` = combined Rayleigh+Mie scatter rate) and exp(-d/λ_abs) is the
absorption survival. These are kept separate to support independent wavelength
dependence.

Gradient design — this is a DiCE-forward citizen, mirroring the water
``photon_step``:
  * TRACK params flow PATHWISE — geometry (per-DOM distances, direction,
    position) is kept LIVE, so the deposit and the carried position/time carry
    the track gradient.
  * OPTICAL scatter-rate/asymmetry params (λ_R, λ_M, g) flow through the DiCE
    SCORE (``logp_increment`` = free-path log-prob + Mie/Rayleigh angle log-prob),
    with the sampled free path ``d`` and the cos-angles DETACHED inside the score
    so it carries optical gradients only (via the live ``mu_tot``/``p_mie``/``g``
    coefficients). The increment is accumulated into ``PhotonState.log_p`` and the
    scan body folds ``dice_dep = exp(log_p − sg(log_p))`` into each future deposit.
    The sampled free path is detached for the trajectory (``new_position`` uses
    ``sg(d)``), so the optical gradient does NOT take the high-variance pathwise
    route through the discrete DOM-candidate selection — it rides the score.
  * λ_abs flows PATHWISE through the deposit / continuing weight (a deterministic
    attenuation, not a sampled decision — exact and low-variance).

Two-channel scatter combines λ_R and λ_M by RATE: mu_tot = 1/λ_R + 1/λ_M,
p_mie = (1/λ_M)/mu_tot.

The step returns ``logp_increment``; the simulator's volume branch applies the
pre-step ``dice_dep`` to the deposit and accumulates ``log_p``.
"""

import jax
import jax.numpy as jnp
from functools import partial

from lucid.utils import normalize
from lucid.simulation.optics import (
    sample_scatter_distance,
    create_local_frame,
    solve_rayleigh_inverse_cdf,
)
from lucid.wavelength.scattering import hg_sample_cos_theta, hg_logpdf, rayleigh_logpdf


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
    """Single volume photon step with per-DOM survival (DiCE Rayleigh+Mie scatter).

    Returns
    -------
    new_position, new_direction, new_time : photon state for the next step
    per_dom_charges : (max_dom,)   implicit-capture per-DOM deposit (× intensity later)
    continuing_factor : scalar     photon weight surviving to next K step
    logp_increment : scalar        DiCE score increment (free-path + angle), accumulated
                                   into PhotonState.log_p for future deposits
    """
    sg = jax.lax.stop_gradient
    k = jax.random.split(rng_key, 5)

    # Two-channel scatter rate (LIVE → direct optical gradient into the deposit).
    mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
    mu_tot = 1.0 / scatter_length + 1.0 / mie_safe
    p_mie = (1.0 / mie_safe) / mu_tot                  # P(a scatter is Mie)

    safe_distances = jnp.maximum(per_dom_distances, 0.0)

    # Implicit-capture deposit: prob of reaching DOM j (mu_tot LIVE) × absorption survival.
    scatter_reach = jnp.exp(-safe_distances * mu_tot)
    absorb_weight = jnp.exp(-safe_distances / absorption_length)
    per_dom_charges = per_dom_overlaps * scatter_reach * absorb_weight
    total_detected = jnp.sum(per_dom_charges)

    # Free path: sampled from the truncated exponential on [0, segment_length]; LIVE in
    # mu_tot (reparam) for the time, DETACHED for the trajectory + decision + score.
    d_live = sample_scatter_distance(segment_length, 1.0 / mu_tot, k[0], eps=1e-7)
    d = sg(d_live)
    scatter_attenuation = jnp.exp(-d / absorption_length)   # absorption survival (λ_abs grad direct)
    continuing_factor = (1.0 - total_detected) * scatter_attenuation

    # Scattered direction: Mie/Rayleigh mixture; g DETACHED in the sampler (scored below).
    is_mie = jax.random.uniform(k[1]) < sg(p_mie)
    ua = jax.random.uniform(k[2])
    phi = jax.random.uniform(k[3]) * 2.0 * jnp.pi
    cmie = hg_sample_cos_theta(ua, sg(g))
    cray = jnp.clip(solve_rayleigh_inverse_cdf(ua), -1.0, 1.0)
    cth = jnp.where(is_mie, cmie, cray)
    sth = jnp.sqrt(jnp.clip(1.0 - cth ** 2, 0.0, 1.0))
    frame = create_local_frame(direction)
    local = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
    new_direction = normalize(frame.T @ local)

    # Trajectory: detached free path → optical gradient rides the score, not the pathwise
    # route through the discrete DOM-candidate selection. direction LIVE → track gradient.
    new_position = position + d * normalize(direction)
    new_time = time + d_live / speed_of_light            # Option C: live free path → optical time grad

    # DiCE score (optical only): truncated-exponential free-path log-pdf + Mie/Rayleigh angle
    # log-pdf. d, segment_length, cmie, cray DETACHED so the score carries optical gradients
    # via the live mu_tot / p_mie / g coefficients (never a track gradient).
    lf = jnp.log(mu_tot) - mu_tot * d - jnp.log1p(-jnp.exp(-mu_tot * sg(segment_length)))
    la = jnp.where(is_mie,
                   jnp.log(p_mie) + hg_logpdf(sg(cmie), g),
                   jnp.log1p(-p_mie) + rayleigh_logpdf(sg(cray)))
    logp_increment = lf + la

    return new_position, new_direction, new_time, per_dom_charges, continuing_factor, logp_increment
