"""Photon iteration functions (sample, update_factors, custom VJP)."""
import jax
import jax.numpy as jnp
from lucid.simulation.optics import (
    normalize, compute_reflection_direction, sample_cosine_hemisphere,
    sample_scatter_distance, compute_scatter_direction,
    create_local_frame, solve_rayleigh_inverse_cdf,
)
from lucid.wavelength.scattering import (
    compute_mie_scatter_direction, hg_sample_cos_theta, hg_logpdf, rayleigh_logpdf,
)
from lucid.simulation.reflection import scalar_reflection, fresnel_rr

# Photon iteration functions (12-arg signatures: dual reflection, no tau_gs)
# ===================================================================


def _interface_refract_reflect(direction_n, radial_normal, medium_id,
                               n_inner, n_outer, u_sample):
    """Snell refraction + Fresnel/TIR at a two-medium interface (single photon).

    The geometry normal is the OUTWARD radial unit vector. ``medium_id`` is the
    photon's CURRENT medium (0 = inner, 1 = outer); ``n_inner``/``n_outer`` the two
    refractive indices. A transmit/reflect outcome is SAMPLED ~ Bernoulli(T) using
    ``u_sample`` (the only single-ray scheme that handles TIR, where T→0).

    Returns
    -------
    new_dir : (3,)            refracted (transmit) or specular (reflect/TIR) direction
    new_medium_id : int       flipped on transmit, unchanged on reflect
    score : float             DiCE score log(p_chosen) with the probability DETACHED
    transmit : bool           whether the sampled outcome was transmission
    """
    sg = jax.lax.stop_gradient
    n_from = jnp.where(medium_id == 0, n_inner, n_outer)
    n_to = jnp.where(medium_id == 0, n_outer, n_inner)
    cos_inc = jnp.clip(jnp.abs(jnp.sum(direction_n * radial_normal)), 0.0, 1.0)
    R, cos_t = fresnel_rr(cos_inc, n_from, n_to)          # TIR → R=1
    T = 1.0 - R

    transmit = u_sample < sg(T)                          # Bernoulli(T), T detached for the score

    # Snell refraction. m = normal pointing back toward the incident side.
    s = jnp.sign(jnp.sum(direction_n * radial_normal))   # +1 outward-going, -1 inward-going
    m = -s * radial_normal
    eta = n_from / n_to
    refr_dir = normalize(eta * direction_n + (eta * cos_inc - cos_t) * m)
    refl_dir = compute_reflection_direction(direction_n, radial_normal)

    new_dir = jnp.where(transmit, refr_dir, refl_dir)
    new_medium_id = jnp.where(transmit, 1 - medium_id, medium_id)
    score = jnp.where(transmit, jnp.log(sg(T) + 1e-12), jnp.log(sg(R) + 1e-12))
    return new_dir, new_medium_id, score, transmit

def photon_iteration_sample(
        position, direction, time, surface_distance,
        normal, scatter_length, mie_scatter_length, g, refl_params,
        absorption_length,
        hit_sensor, lam, rng_key, speed_of_light):
    """
    Sampling version of photon iteration that makes binary decisions.

    Performs Monte Carlo sampling where photons make discrete choices
    (detect/reflect/scatter) rather than computing expected values.
    Walls use diffuse reflection, sensors use specular reflection.

    Parameters
    ----------
    position : jnp.ndarray
        Current 3D position of the photon
    direction : jnp.ndarray
        Current normalized direction vector of the photon
    time : float
        Current time of the photon
    surface_distance : float
        Distance to the nearest surface intersection
    normal : jnp.ndarray
        Surface normal at the intersection point
    scatter_length : float
        Mean free path for scattering in the medium
    wall_reflection_rate : float
        Probability of reflection when hitting a wall
    sensor_reflection_rate : float
        Probability of reflection when hitting a sensor
    absorption_length : float
        Mean free path for absorption in the medium
    hit_sensor : bool
        Whether the photon hit a sensor (True) or wall (False)
    rng_key : jax.random.PRNGKey
        Random key for sampling
    speed_of_light : float
        Speed of light in the medium (m/ns)

    Returns
    -------
    new_pos : jnp.ndarray
        Updated photon position
    new_dir : jnp.ndarray
        Updated photon direction
    new_time : float
        Updated photon time
    detect_prob : float
        1.0 if photon is detected, 0.0 otherwise
    reflection_attenuation : float
        Attenuation factor due to absorption
    continuing_factor : float
        Factor for continuing photons (0.0 if detected, attenuation if continues)
    """

    k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

    # MC path keeps inline scalar reflection (unpacks the packed refl_params); ``lam`` unused.
    reflection_rate = jnp.where(hit_sensor, refl_params.sensor_rate, refl_params.wall_rate)
    # Combine Rayleigh + Mie by RATES (NOT min): total scatter coeff = 1/L_R + 1/L_M.
    mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
    inv_total = 1.0 / scatter_length + 1.0 / mie_safe
    scatter_length = 1.0 / inv_total                 # effective combined scatter length
    p_mie = (1.0 / mie_safe) / inv_total             # P(a scatter is Mie) = (1/L_M)/(1/L_R+1/L_M)
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)

    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob

    u2 = jax.random.uniform(k2)
    reflects = reaches_surface & (u2 < reflection_rate)
    detects = reaches_surface & (u2 >= reflection_rate)
    scatters = ~reaches_surface

    # The fraction of 'errors' for a given epsilon change with the detector size.
    # This is because the numerical error in direction over a large distance translates
    # into a relatively larger deviation. For HK-size epsilon 1e-4 translates into a few
    # tens of rays going out of the detector per each million after several steps.
    # The rule of thumb is that epsilon needs to go down/up proportionally to the detector size.
    epsilon = 1e-4
    inward_normal = -normal  # geometry normals point outward; negate to get into-medium direction
    new_pos = jnp.where(
        scatters,
        position + scatter_distance * normalize(direction),
        position + surface_distance * normalize(direction) + epsilon * normalize(inward_normal),
    )

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(inward_normal, k4)
    reflection_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    rayleigh_dir = compute_scatter_direction(direction, k3)
    mie_dir = compute_mie_scatter_direction(direction, k3, g)
    is_mie = jax.random.uniform(k5) < p_mie          # choose Mie vs Rayleigh per scatter (~5% Mie at physical L)
    chosen_scatter_dir = jnp.where(is_mie, mie_dir, rayleigh_dir)

    new_dir = jnp.where(
        reflects,
        reflection_dir,
        jnp.where(scatters, chosen_scatter_dir, direction),
    )

    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + distance_traveled / speed_of_light

    # Binary absorption sampling (Bernoulli) — DEDICATED key k6 (was k3, which is also used
    # for the scatter direction → absorption was correlated with scatter angle; PORT_PLAN §4.3).
    survival_prob = jnp.exp(-distance_traveled / absorption_length)
    u_absorption = jax.random.uniform(k6)
    survives_absorption = u_absorption < survival_prob
    attenuation = survives_absorption.astype(jnp.float32)

    detect_prob = detects.astype(jnp.float32)
    reflection_attenuation = attenuation
    continuing_factor = jnp.where(detects, 0.0, attenuation)

    # 7th return: DiCE score increment. The sampling path is the truth/data generator and
    # carries no score -> 0.0 (keeps the shared scan-body step signature consistent).
    logp_increment = jnp.zeros_like(new_time)

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor, logp_increment


def photon_iteration_update_factors(
        position, direction, time, surface_distance,
        normal, scatter_length, mie_scatter_length, g, refl_params,
        absorption_length,
        hit_sensor, lam, rng_key, speed_of_light,
        reflection_fn=scalar_reflection):
    """
    Expected-value photon update with Straight-Through Estimator (STE).

    Computes expected values for detect/reflect/scatter outcomes for full
    differentiability. Walls use diffuse reflection, sensors use specular.

    Parameters
    ----------
    position, direction : jnp.ndarray
        (3,) arrays for the photon's current state.
    time : float
        Current photon time.
    surface_distance : float
        Distance to nearest surface (norm of hit_position - position).
    normal : jnp.ndarray
        (3,) surface normal at the hit position.
    scatter_length : float
        Mean free path for scattering.
    wall_reflection_rate : float
        Probability of reflection at walls.
    sensor_reflection_rate : float
        Probability of reflection at sensors.
    absorption_length : float
        Mean free path for absorption.
    hit_sensor : bool
        Whether the photon hit a sensor (True) or wall (False).
    rng_key : jax.random.PRNGKey
        JAX PRNG key.
    speed_of_light : float
        Speed of light in the medium (m/ns).

    Returns
    -------
    new_pos : jnp.ndarray
        (3,) updated photon position.
    new_dir : jnp.ndarray
        (3,) updated photon direction.
    new_time : float
        Updated photon time.
    detect_prob : float
        Probability of detection.
    reflection_attenuation : float
        Attenuation factor from absorption over surface distance.
    continuing_factor : float
        Factor for continuation (reflection or scatter).
    """

    # Ported from mie_hunter/implicit_engine_lik.py (per-photon body; the function is vmapped
    # over photons in the scan). Analog two-channel free path + Mie/Rayleigh Bernoulli + DiCE
    # score + implicit-capture deposit. Gradient design (see mie_hunter/PORT_PLAN.md):
    #   • TRACK params flow PATHWISE — geometry (surface_distance Dd, direction, per-sensor
    #     distances) kept LIVE → `reach`, positions and time carry the track gradient.
    #   • OPTICAL scatter-rate/angle params flow through the DiCE score `lf`/`la`; `d` and `Dd`
    #     are stop_gradient'd INSIDE the score so it carries optical gradients only (no double
    #     count with the pathwise `reach`). The per-step increment `lf+la` is returned and
    #     accumulated into PhotonState.log_p; the scan body folds dice_dep=exp(logp−sg(logp)).
    #   • OPTICAL gradient into the arrival TIME flows via the reparameterised LIVE free path
    #     `d_live` (Option C, low variance); decision/trajectory use the detached `d`.
    k = jax.random.split(rng_key, 8)
    sg = jax.lax.stop_gradient

    mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
    mu_tot = 1.0 / scatter_length + 1.0 / mie_safe                 # full two-channel scatter rate
    p_mie = scatter_length / (scatter_length + mie_safe)           # P(a scatter is Mie)

    # Analog free path. d_live LIVE in mu_tot (reparam → optical gradient into TIME);
    # d detached for the hard decision, trajectory and scores (charge ≡ validated engine).
    u0 = jax.random.uniform(k[0])
    d_live = -jnp.log1p(-sg(u0)) / mu_tot
    d = sg(d_live)
    Dd = surface_distance                                         # LIVE → pathwise track gradient
    is_scat = d < Dd

    dist = jnp.where(is_scat, d, Dd)
    atten = jnp.exp(-dist / absorption_length)                    # absorption survival to event

    # Mie/Rayleigh Bernoulli + the matching angle samplers
    is_mie = jax.random.uniform(k[2]) < sg(p_mie)
    ua = jax.random.uniform(k[3])
    phi = jax.random.uniform(k[4]) * 2.0 * jnp.pi
    cmie = hg_sample_cos_theta(ua, sg(g))
    cray = jnp.clip(solve_rayleigh_inverse_cdf(ua), -1.0, 1.0)
    cth = jnp.where(is_mie, cmie, cray)
    sth = jnp.sqrt(jnp.clip(1.0 - cth**2, 0.0, 1.0))
    frame = create_local_frame(direction)
    local = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
    scat_dir = normalize(frame @ local)

    # Reflection — delegated to the pluggable model (default scalar_reflection, byte-identical).
    # The model returns (refl_prob, reflection_dir, lr_score): the angle/λ-dependent MAGNITUDE,
    # the post-reflection DIRECTION, and a DiCE score for any DISCRETE reflection branch (0 for
    # scalar; the spec/diff-mix log-prob for angular models). The reflected direction detaches the
    # normal internally (Igehy 1999 curvature term compounds ~1/r_sensor across K bounces).
    epsilon = 1e-4
    inward_normal = -sg(normal)
    refl_prob, reflection_dir, lr = reflection_fn(
        direction, normal, hit_sensor, refl_params, lam, k[5])
    new_dir = jnp.where(is_scat, scat_dir, reflection_dir)

    # Positions (hard, detached free path d): scatter point vs surface point.
    dir_n = normalize(direction)
    scatter_pos = position + d * dir_n
    surface_pos = position + Dd * dir_n + epsilon * normalize(inward_normal)
    new_pos = jnp.where(is_scat, scatter_pos, surface_pos)

    # DiCE scores — d, Dd, cmie, cray, p_mie, g-sample DETACHED so the score carries OPTICAL
    # gradients only (via live mu_tot, p_mie, g coefficients), never a track gradient. lr carries
    # the reflection spec/diff-mix gradient for angular models (0 for scalar).
    lf = jnp.where(is_scat, jnp.log(mu_tot) - mu_tot * sg(d), -mu_tot * sg(Dd))
    la = jnp.where(is_scat,
                   jnp.where(is_mie,
                             jnp.log(p_mie) + hg_logpdf(sg(cmie), g),
                             jnp.log1p(-p_mie) + rayleigh_logpdf(sg(cray))),
                   0.0)
    logp_increment = lf + la + lr

    # Implicit-capture deposit factor (expected detected charge, Rao-Blackwellised over the
    # free-path decision). Dd LIVE → pathwise track gradient through `reach`/`atten_surf`.
    # No qe (applied in make_hits); no dice_dep (the scan body multiplies it from PRE-step log_p).
    reach = jnp.exp(-mu_tot * Dd)
    atten_surf = jnp.exp(-Dd / absorption_length)
    detect_prob = reach * (1.0 - refl_prob) * atten_surf
    reflection_attenuation = jnp.ones_like(detect_prob)           # folded into detect_prob
    continuing_factor = jnp.where(is_scat, atten, refl_prob * atten)

    # Arrival-time path uses the LIVE free path d_live (Option C reparam → optical L_M
    # gradient into time at low variance); Dd LIVE → pathwise track-time gradient.
    distance_for_time = jnp.where(is_scat, d_live, Dd)
    new_time = time + distance_for_time / speed_of_light

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor, logp_increment


# ===================================================================
# Custom VJP wrapper — NaN gradient sanitisation
# ===================================================================
#
# Factory: the reflection model is captured STATICALLY in the closure, so the
# custom_vjp signature stays fixed (refl_params is a single packed pytree arg) —
# a new reflection model never reshapes _fwd/_bwd residuals or cotangents.

def make_photon_iteration_update_factors_safe(reflection_fn=scalar_reflection):
    """Build the expected-value step closed over ``reflection_fn`` (a static model choice).

    Historically this wrapped the step in a ``jax.custom_vjp`` whose backward scrubbed
    NaN/Inf cotangents — a backstop for a sqrt-at-zero 0/0 in the surface-distance norm
    (a reflected photon sitting exactly on the surface it just hit → |Δ|=0 → 0·∞). That
    0/0 is now removed AT THE SOURCE by the eps-inside-sqrt floors (``simulator.py``
    surface_distances ``+1e-12``; the cylinder discriminant ``maximum(·,1e-6)``), so the
    backstop is unnecessary and is dropped. The reason to drop it: ``custom_vjp`` blocks
    forward-mode (``jvp``/``jacfwd``), which the calibration/recon Jacobians need (P inputs
    → N_sensors outputs → forward-mode is the efficient, low-variance mode). Forward is
    byte-identical; the gradient/Hessian are now the true unbiased AD values. Verified
    NaN-free without the backstop across ~3·10⁴ grad/Hessian evals over random + on-surface
    stress geometries (hessian_probe/recon_nan_thorough.py).
    """

    def _step(position, direction, time, surface_distance,
              normal, scatter_length, mie_scatter_length, g, refl_params,
              absorption_length, hit_sensor, lam, rng_key, speed_of_light):
        return photon_iteration_update_factors(
            position, direction, time, surface_distance,
            normal, scatter_length, mie_scatter_length, g, refl_params,
            absorption_length, hit_sensor, lam, rng_key, speed_of_light,
            reflection_fn=reflection_fn)

    return _step


# Module-level default (scalar reflection) — the byte-identical drop-in.
photon_iteration_update_factors_safe = make_photon_iteration_update_factors_safe()


# ===================================================================
# Two-medium (nested) photon steps — interface refraction/Fresnel/TIR
# ===================================================================
#
# These mirror the single-medium steps above and ADD a third surface outcome: when the
# nearest surface is the inner optical interface (``hit_interface`` True), the photon
# refracts (Snell) / reflects (Fresnel, incl. TIR) instead of detecting/reflecting at a
# wall. The transmit/reflect choice is SAMPLED (Bernoulli(T)) — the only single-ray scheme
# that handles TIR — and carries a DiCE score so the gradient stays unbiased. The
# scatter-in-medium branch is byte-identical to the single-medium step; only the
# reached-surface outcome gains the interface case. Each returns an 8-tuple: the usual
# 7-tuple plus the photon's updated ``medium_id``.
#
# Kept as DEDICATED functions (not a flag on the single-medium step) so the single-medium
# RNG stream and arithmetic are provably untouched → the no-interface path is byte-identical.


def photon_iteration_update_factors_nested(
        position, direction, time, surface_distance,
        normal, scatter_length, mie_scatter_length, g, refl_params,
        absorption_length, hit_sensor, lam, rng_key, speed_of_light,
        hit_interface, medium_id, n_inner, n_outer,
        reflection_fn=scalar_reflection):
    """Expected-value (STE/DiCE) step with a two-medium interface outcome.

    Identical to :func:`photon_iteration_update_factors` for the scatter and
    outer-wall/sensor outcomes; when ``hit_interface`` the reached-surface outcome is the
    sampled Snell/Fresnel/TIR interface (see :func:`_interface_refract_reflect`). Returns
    the 7-tuple plus ``new_medium_id``.
    """
    k = jax.random.split(rng_key, 9)
    sg = jax.lax.stop_gradient

    mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
    mu_tot = 1.0 / scatter_length + 1.0 / mie_safe
    p_mie = scatter_length / (scatter_length + mie_safe)

    u0 = jax.random.uniform(k[0])
    d_live = -jnp.log1p(-sg(u0)) / mu_tot
    d = sg(d_live)
    Dd = surface_distance
    is_scat = d < Dd

    dist = jnp.where(is_scat, d, Dd)
    atten = jnp.exp(-dist / absorption_length)

    is_mie = jax.random.uniform(k[2]) < sg(p_mie)
    ua = jax.random.uniform(k[3])
    phi = jax.random.uniform(k[4]) * 2.0 * jnp.pi
    cmie = hg_sample_cos_theta(ua, sg(g))
    cray = jnp.clip(solve_rayleigh_inverse_cdf(ua), -1.0, 1.0)
    cth = jnp.where(is_mie, cmie, cray)
    sth = jnp.sqrt(jnp.clip(1.0 - cth**2, 0.0, 1.0))
    frame = create_local_frame(direction)
    local = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
    scat_dir = normalize(frame @ local)

    epsilon = 1e-4
    inward_normal = -sg(normal)
    refl_prob, reflection_dir, lr = reflection_fn(
        direction, normal, hit_sensor, refl_params, lam, k[5])

    dir_n = normalize(direction)
    scatter_pos = position + d * dir_n
    surface_pos = position + Dd * dir_n + epsilon * normalize(inward_normal)

    # --- interface outcome (Snell/Fresnel/TIR, sampled) ---
    radial = sg(normal)
    iface_dir, iface_medium, iface_score, _transmit = _interface_refract_reflect(
        dir_n, radial, medium_id, n_inner, n_outer, jax.random.uniform(k[8]))
    iface_pos = position + Dd * dir_n + epsilon * iface_dir
    at_interface = hit_interface & (~is_scat)

    # Direction / position: scatter | (interface | wall-or-sensor)
    surf_dir = jnp.where(hit_interface, iface_dir, reflection_dir)
    surf_pos = jnp.where(hit_interface, iface_pos, surface_pos)
    new_dir = jnp.where(is_scat, scat_dir, surf_dir)
    new_pos = jnp.where(is_scat, scatter_pos, surf_pos)
    new_medium_id = jnp.where(at_interface, iface_medium, medium_id)

    # DiCE score: free-path lf + scatter-angle la; at the interface the reached-surface
    # score is the transmit/reflect log-prob (replaces the wall reflection score lr).
    lf = jnp.where(is_scat, jnp.log(mu_tot) - mu_tot * sg(d), -mu_tot * sg(Dd))
    la = jnp.where(is_scat,
                   jnp.where(is_mie,
                             jnp.log(p_mie) + hg_logpdf(sg(cmie), g),
                             jnp.log1p(-p_mie) + rayleigh_logpdf(sg(cray))),
                   0.0)
    surf_score = jnp.where(hit_interface, iface_score, lr)
    logp_increment = lf + la + jnp.where(is_scat, 0.0, surf_score)

    # Charge deposit: interface deposits nothing (no sensor); wall/sensor as before.
    reach = jnp.exp(-mu_tot * Dd)
    atten_surf = jnp.exp(-Dd / absorption_length)
    detect_prob = jnp.where(hit_interface, 0.0,
                            reach * (1.0 - refl_prob) * atten_surf)
    reflection_attenuation = jnp.ones_like(detect_prob)
    # Continuation weight: scatter → atten; interface → atten (T/R realised by sampling,
    # full weight either way); wall → refl_prob·atten.
    cont_surface = jnp.where(hit_interface, atten, refl_prob * atten)
    continuing_factor = jnp.where(is_scat, atten, cont_surface)

    distance_for_time = jnp.where(is_scat, d_live, Dd)
    new_time = time + distance_for_time / speed_of_light

    return (new_pos, new_dir, new_time, detect_prob, reflection_attenuation,
            continuing_factor, logp_increment, new_medium_id)


def make_photon_iteration_update_factors_nested_safe(reflection_fn=scalar_reflection):
    """Build the nested expected-value step closed over ``reflection_fn``."""
    def _step(position, direction, time, surface_distance,
              normal, scatter_length, mie_scatter_length, g, refl_params,
              absorption_length, hit_sensor, lam, rng_key, speed_of_light,
              hit_interface, medium_id, n_inner, n_outer):
        return photon_iteration_update_factors_nested(
            position, direction, time, surface_distance,
            normal, scatter_length, mie_scatter_length, g, refl_params,
            absorption_length, hit_sensor, lam, rng_key, speed_of_light,
            hit_interface, medium_id, n_inner, n_outer,
            reflection_fn=reflection_fn)
    return _step


def photon_iteration_sample_nested(
        position, direction, time, surface_distance,
        normal, scatter_length, mie_scatter_length, g, refl_params,
        absorption_length, hit_sensor, lam, rng_key, speed_of_light,
        hit_interface, medium_id, n_inner, n_outer):
    """MC-sampling step (truth generator) with a two-medium interface outcome.

    Mirrors :func:`photon_iteration_sample`; when ``hit_interface`` the reached-surface
    outcome is the sampled Snell/Fresnel/TIR interface. Returns the 7-tuple plus
    ``new_medium_id``. No DiCE score (sampling path → logp 0).
    """
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng_key, 7)

    reflection_rate = jnp.where(hit_sensor, refl_params.sensor_rate, refl_params.wall_rate)
    mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
    inv_total = 1.0 / scatter_length + 1.0 / mie_safe
    scatter_length = 1.0 / inv_total
    p_mie = (1.0 / mie_safe) / inv_total
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob
    scatters = ~reaches_surface

    # At the surface: interface (refract/reflect) vs wall/sensor (reflect/detect).
    u2 = jax.random.uniform(k2)
    reflects_wall = reaches_surface & (~hit_interface) & (u2 < reflection_rate)
    detects = reaches_surface & (~hit_interface) & (u2 >= reflection_rate)

    epsilon = 1e-4
    inward_normal = -normal
    dir_n = normalize(direction)

    radial = normal
    iface_dir, iface_medium, _score, _transmit = _interface_refract_reflect(
        dir_n, radial, medium_id, n_inner, n_outer, jax.random.uniform(k7))

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(inward_normal, k4)
    wall_refl_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    rayleigh_dir = compute_scatter_direction(direction, k3)
    mie_dir = compute_mie_scatter_direction(direction, k3, g)
    is_mie = jax.random.uniform(k5) < p_mie
    chosen_scatter_dir = jnp.where(is_mie, mie_dir, rayleigh_dir)

    at_interface = reaches_surface & hit_interface
    # surface direction: interface | wall
    surf_dir = jnp.where(hit_interface, iface_dir, wall_refl_dir)
    new_dir = jnp.where(scatters, chosen_scatter_dir, surf_dir)

    surf_pos = jnp.where(
        hit_interface,
        position + surface_distance * dir_n + epsilon * iface_dir,
        position + surface_distance * dir_n + epsilon * normalize(inward_normal))
    new_pos = jnp.where(scatters, position + scatter_distance * dir_n, surf_pos)
    new_medium_id = jnp.where(at_interface, iface_medium, medium_id)

    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + distance_traveled / speed_of_light

    survival_prob = jnp.exp(-distance_traveled / absorption_length)
    survives_absorption = jax.random.uniform(k6) < survival_prob
    attenuation = survives_absorption.astype(jnp.float32)

    detect_prob = detects.astype(jnp.float32)
    reflection_attenuation = attenuation
    # interface photons always continue (if they survive absorption); detected photons stop.
    continuing_factor = jnp.where(detects, 0.0, attenuation)
    logp_increment = jnp.zeros_like(new_time)

    return (new_pos, new_dir, new_time, detect_prob, reflection_attenuation,
            continuing_factor, logp_increment, new_medium_id)
