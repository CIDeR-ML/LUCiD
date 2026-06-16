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
from lucid.simulation.reflection import scalar_reflection

# Photon iteration functions (12-arg signatures: dual reflection, no tau_gs)
# ===================================================================

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


def _field_optical_factor(field_fn, field_params, position, direction, dist, n_eval):
    """Line-integrated spatial-absorption multiplier ``mean_i field(x_i)`` over a step.

    The absorption optical depth becomes ``(dist / L_abs) * mean_i field(x_i)``. Sample
    geometry is gradient-DETACHED (``sg``), so the field contributes only its magnitude and
    its own ``field_params`` gradient — the existing track/optical gradients keep their
    original structure (and ``field_fn is None`` skips this entirely → byte-identical).
    """
    sg = jax.lax.stop_gradient
    dir_n = normalize(sg(direction))
    base = sg(position)
    ts = jnp.linspace(0.0, 1.0, n_eval + 2)[1:-1]                  # (n_eval,) interior midpoints
    pts = base[None, :] + ts[:, None] * (sg(dist) * dir_n)[None, :]  # (n_eval, 3)
    return jnp.mean(field_fn(field_params, pts))


def photon_iteration_update_factors(
        position, direction, time, surface_distance,
        normal, scatter_length, mie_scatter_length, g, refl_params,
        absorption_length,
        hit_sensor, lam, rng_key, speed_of_light,
        reflection_fn=scalar_reflection, field_fn=None, field_params=None,
        n_field_eval=1):
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
    # Spatial-absorption multiplier: ONE field evaluation per step (piecewise-constant field
    # over the step), SHARED by the event-survival (atten) and surface-deposit (atten_surf)
    # hooks below. Sampled over the surface ray [0, Dd] (the step's full spatial extent;
    # dist ≤ Dd). This is the irreducible 1 (×n_field_eval) eval/photon/step — half the work
    # of evaluating each hook's segment separately. field_fn is None ⇒ phi unused and the
    # homogeneous path stays byte-identical.
    if field_fn is None:
        atten = jnp.exp(-dist / absorption_length)                # absorption survival to event
    else:
        # NB: named field_mult (NOT phi — `phi` is reused below for the scatter azimuth angle).
        field_mult = _field_optical_factor(field_fn, field_params, position, direction, Dd, n_field_eval)
        atten = jnp.exp(-dist / absorption_length * field_mult)

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
    if field_fn is None:
        atten_surf = jnp.exp(-Dd / absorption_length)
    else:
        atten_surf = jnp.exp(-Dd / absorption_length * field_mult)  # reuse the single per-step factor
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

def make_photon_iteration_update_factors_safe(reflection_fn=scalar_reflection,
                                              field_fn=None, n_field_eval=1):
    """Build the expected-value step closed over ``reflection_fn`` (a static model choice).

    ``field_fn`` is the optional static spatial-absorption map (from
    ``simulation.fields.make_field``). When ``None`` the step is byte-identical to the
    homogeneous engine and ``field_params`` is never referenced; when set, the per-step
    ``field_params`` pytree leaf is passed at call time (vmapped, broadcast) and the
    absorption hooks line-integrate the map over the step. ``n_field_eval`` (static) is the
    number of interior sample points per step.

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
              absorption_length, hit_sensor, lam, rng_key, speed_of_light,
              field_params=None):
        return photon_iteration_update_factors(
            position, direction, time, surface_distance,
            normal, scatter_length, mie_scatter_length, g, refl_params,
            absorption_length, hit_sensor, lam, rng_key, speed_of_light,
            reflection_fn=reflection_fn, field_fn=field_fn, field_params=field_params,
            n_field_eval=n_field_eval)

    return _step


# Module-level default (scalar reflection) — the byte-identical drop-in.
photon_iteration_update_factors_safe = make_photon_iteration_update_factors_safe()
