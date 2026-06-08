"""SIREN-based photon ray generation functions.

Moved from lucid/generate.py during Phase 2.2 refactor.
"""

import os
import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from lucid.siren.core import SIREN
from lucid.utils import normalize, generate_orthonormal_basis, jax_rotate_vector_local

# Emission weighting: the photons are importance-sampled ∝density (inverse-CDF), so to make the physical
# emission ∝density the per-photon weight must be UNIFORM. Weighting by density again (legacy) double-counts
# -> emission ∝density², which over-sharpens the Cherenkov ring (angular std 1.8° vs the true ~11°,
# collapsing the ring onto ~3x too few PMTs). Set SIREN_DENSITY_SQUARED=1 to restore the legacy behavior.
_SIREN_DENSITY_SQUARED = os.environ.get('SIREN_DENSITY_SQUARED', '0') == '1'
# Optional cap on emission distance along the track (mm); 0 = no cap (use grid max). Test knob for whether
# the longitudinal tail to the 10m grid edge (past the ~4.7m muon range) drives the residual direction bias.
_SIREN_DIST_CAP_MM = float(os.environ.get('SIREN_DIST_CAP_MM', '0'))
# CONTINUOUS inverse-CDF emission sampler. When set, the (angle,distance) for each systematic sample u
# is LINEARLY INTERPOLATED between the bracketing CDF bins (fraction (u-cdf[lo])/(cdf[hi]-cdf[lo])),
# instead of the piecewise-constant searchsorted lookup angle=mesh[idx]. The interpolation fraction is a
# smooth function of cdf(E) -> p_emit(E) -> energy, so the sampled angle/distance carry the energy
# gradient PATHWISE (and the 2nd-order resample shift), with no detached searchsorted. With the continuous
# sampler the DiCE score is unnecessary (weights are uniform = mean_topk).
_SIREN_CONTINUOUS = os.environ.get('SIREN_CONTINUOUS', '0') == '1'

# ---------------------------------------------------------------------------------------------------
# THREE EXPERIMENTAL EMISSION SAMPLERS (each behind its own env flag, default OFF -> default path is
# byte-identical). Goal: a sampler that is correct for COVERAGE, 1st-order GRADIENTS, AND the energy
# 2nd-order (d2/dE2) DIAGONAL of the Hessian.  See lucid/sources/siren_rays.py docstring of each branch.
#
#   SIREN_IMPORTANCE  (A)  fixed-proposal importance reweight: resample bins ONCE at E_CAL (energy-
#                          independent sample, no step function in E); weight = mean_topk * ratio
#                          p_emit(E)[idx]/p_emit(E_CAL)[idx]. Energy enters ONLY via the smooth
#                          density-ratio -> pathwise exact 1st AND 2nd order. (ratio==1 at E=E_CAL.)
#   SIREN_SMOOTH      (B)  Gaussian-kernel-smoothed 2-D (angle,distance) density (separable conv,
#                          sigma SIREN_SMOOTH_SIGMA bins) BEFORE the CDF, then continuous inverse-CDF.
#   SIREN_MARGINAL    (C)  factored continuous: 1-D marginal angle inverse-CDF, then a per-angle-bin
#                          conditional 1-D distance inverse-CDF (each axis far denser than the 2-D).
# ---------------------------------------------------------------------------------------------------
_SIREN_IMPORTANCE = os.environ.get('SIREN_IMPORTANCE', '0') == '1'
_SIREN_SMOOTH = os.environ.get('SIREN_SMOOTH', '0') == '1'
_SIREN_SMOOTH_SIGMA = float(os.environ.get('SIREN_SMOOTH_SIGMA', '1.5'))  # Gaussian kernel width, bins
_SIREN_MARGINAL = os.environ.get('SIREN_MARGINAL', '0') == '1'


def _gaussian_kernel(sigma):
    """1-D normalized Gaussian kernel, radius = ceil(3 sigma) (static length for jit)."""
    r = int(max(1, round(3.0 * sigma)))
    x = jnp.arange(-r, r + 1, dtype=jnp.float32)
    k = jnp.exp(-0.5 * (x / sigma) ** 2)
    return k / jnp.sum(k)


def _smooth2d_separable(grid2d, sigma):
    """Separable Gaussian blur of a 2-D (n_angle, n_dist) density with reflect padding."""
    k = _gaussian_kernel(sigma)
    r = (k.shape[0] - 1) // 2
    # blur along axis 0 (angle)
    g = jnp.pad(grid2d, ((r, r), (0, 0)), mode='edge')
    g = jax.vmap(lambda col: jnp.convolve(col, k, mode='valid'), in_axes=1, out_axes=1)(g)
    # blur along axis 1 (distance)
    g = jnp.pad(g, ((0, 0), (r, r)), mode='edge')
    g = jax.vmap(lambda row: jnp.convolve(row, k, mode='valid'), in_axes=0, out_axes=0)(g)
    return g


def _continuous_invcdf_1d(cdf, us, lo_edge, hi_edge):
    """Continuous inverse-CDF on a 1-D bin CDF (cdf[i] = cumulative through bin i, monotone in [0,1]).
    Returns a value in [lo_edge, hi_edge] that is a SMOOTH (piecewise-linear-in-cdf) function of the
    bin probabilities -> carries the energy gradient pathwise (no detached searchsorted lookup)."""
    n = cdf.shape[0]
    idx = jnp.clip(jnp.searchsorted(cdf, us), 0, n - 1)
    cdf_hi = cdf[idx]
    cdf_lo = jnp.where(idx <= 0, 0.0, cdf[jnp.clip(idx - 1, 0, n - 1)])
    frac = jnp.clip((us - cdf_lo) / (cdf_hi - cdf_lo + 1e-12), 0.0, 1.0)  # within-bin fraction in [0,1]
    # bin i spans [lo_edge + i*dx, lo_edge + (i+1)*dx]; sampled value at (i + frac)*dx
    dx = (hi_edge - lo_edge) / n
    return lo_edge + (idx + frac) * dx


@partial(jax.jit, static_argnums=(2,))
def generate_random_cone_vectors(R, theta, num_vectors, key):
    """Generate random vectors uniformly distributed on a cone surface.

    Parameters
    ----------
    R : jnp.ndarray
        Direction vector of the cone axis
    theta : float
        Opening angle of the cone in radians
    num_vectors : int
        Number of random vectors to generate
    key : jax.random.PRNGKey
        Random number generator key

    Returns
    -------
    jnp.ndarray
        Array of shape (num_vectors, 3) containing random unit vectors on cone surface
    """
    R = normalize(R)
    theta = jnp.clip(theta, 1e-6, jnp.pi - 1e-6)

    key1, key2 = random.split(key)
    phi = random.uniform(key1, (num_vectors,), minval=0, maxval=2 * jnp.pi)

    # Convert from polar to cartesian coordinates on cone surface
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    x = jnp.cos(phi) * sin_theta
    y = jnp.sin(phi) * sin_theta
    z = cos_theta * jnp.ones_like(x)

    basis = generate_orthonormal_basis(R)
    vectors = jnp.column_stack((x, y, z))
    rotated_vectors = jnp.einsum('ij,kj->ki', basis, vectors)

    return rotated_vectors

@jax.jit
def denormalize_log_predictions(predictions, log_max, log_min):
    log_predictions = predictions * (log_max - log_min) + log_min
    return 10 ** log_predictions - 1e-10

@jax.jit
def normalize_inputs_jit(inputs, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max):
    """
    Normalize inputs to the range [-1, 1] in all dimensions.

    Args:
        inputs: Array of shape (..., 3) containing [energy, angle, distance] values.
        energy_min, energy_max: The minimum and maximum energy values.
        angle_min, angle_max: The minimum and maximum angle values.
        distance_min, distance_max: The minimum and maximum distance values.

    Returns:
        Array of shape (..., 3) with normalized values in the range [-1, 1].
    """
    # Extract the individual components
    energy = inputs[:, 0]
    angle = inputs[:, 1]
    distance = inputs[:, 2]

    # Normalize each component to [-1, 1]
    normalized_energy = 2.0 * (energy - energy_min) / (energy_max - energy_min) - 1.0
    normalized_angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
    normalized_distance = 2.0 * (distance - distance_min) / (distance_max - distance_min) - 1.0

    # Stack the normalized components back together
    normalized_inputs = jnp.stack([normalized_energy, normalized_angle, normalized_distance], axis=1)

    return normalized_inputs

@partial(jax.jit, static_argnums=(3,))
def photonsim_differentiable_get_rays(track_origin, track_direction, energy, Nphot,
                                         table_data, model_params, key, num_seeds_a=0.035882, num_seeds_b=1.417106, num_seeds_c=2.75):
    """
    Generate photon rays using SIREN model for photon generation.

    Parameters
    ----------
    track_origin : jnp.ndarray
        3D position of the track origin
    track_direction : jnp.ndarray
        3D direction vector of the track
    energy : float
        Energy of the particle in MeV
    Nphot : int
        Number of photons to generate
    table_data : tuple
        Grid data for SIREN model evaluation
    model_params : dict
        SIREN model parameters
    key : jax.random.PRNGKey
        Random key for sampling
    num_seeds_a : float, optional
        Parameter 'a' for power law num_seeds calculation, default 0.035882
    num_seeds_b : float, optional
        Parameter 'b' for power law num_seeds calculation, default 1.417106
    num_seeds_c : float, optional
        Parameter 'c' for power law num_seeds calculation, default 2.75

    Returns
    -------
    ray_vectors : jnp.ndarray
        Array of photon direction vectors
    ray_origins : jnp.ndarray
        Array of photon origin positions
    photon_weights : jnp.ndarray
        Array of photon weights
    """
    key, subkey = random.split(key)

    n_bins, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max, angle_bins, distance_bins, angle_dist_grid, angle_mesh, distance_mesh, log_min, log_max = table_data

    # ============================================================================
    # FIRST EVALUATION: Full grid to get photon weights for sampling
    # ============================================================================
    evaluation_grid = jnp.stack([
        jnp.full_like(angle_mesh, energy).ravel(),  # Energy (MeV)
        angle_mesh.ravel(),                         # Angle (radians)
        distance_mesh.ravel(),                      # Distance (mm)
    ], axis=1)

    normalized_grid = normalize_inputs_jit(evaluation_grid, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max)

    # Initialize SIREN model
    model = SIREN(
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
    )

    # SIREN emission density over the (angle, distance) grid (single call at this energy).
    log_pred, _ = model.apply(model_params, normalized_grid)
    density = jnp.clip(denormalize_log_predictions(jnp.squeeze(log_pred), log_max, log_min), 1e-12, None)
    angle_dist_mesh = jnp.array(angle_dist_grid)
    n_grid = angle_dist_mesh.shape[0]
    p_emit = density / jnp.sum(density)

    # --- DiCE-score emission sampling -------------------------------------------------------
    # Sample bins PROPORTIONAL to the SIREN density via a low-discrepancy SYSTEMATIC resample
    # (selection detached), and attach a DiCE magic box dice_score = exp(logp - sg logp).
    # Forward = density (score==1) so the density-weighted emission profile (and occupancy) is
    # preserved; the gradient picks up d/dE[density·dice_score] = d(density)/dE + density·d(logp)/dE
    # — the EXACT gradient of the density-weighted forward, so energy now flows through the
    # emission GEOMETRY (which bins are populated), not just the per-photon weight. The old
    # argsort top-k selection was non-differentiable; ∝density sampling makes p(angle,dist|E)
    # move smoothly with E. (See mie_hunter sampling-method study; replaces the 2nd SIREN call.)
    key, sampling_key = random.split(key)
    cdf = jnp.cumsum(p_emit)
    u0 = random.uniform(sampling_key, ()) / Nphot
    us = u0 + jnp.arange(Nphot) / Nphot
    idx = jnp.clip(jnp.searchsorted(cdf, us), 0, n_grid - 1)
    logp = jnp.log(p_emit[idx])
    dice_score = jnp.exp(logp - jax.lax.stop_gradient(logp))

    # CONTINUOUS inverse-CDF reparameterization: interpolate (angle,distance) between the bracketing CDF
    # bins so the sampled values are a SMOOTH function of p_emit (hence energy). frac depends on cdf(E),
    # so d(sampled)/dE is exact pathwise (1st AND 2nd order); no detached searchsorted gradient.
    if _SIREN_CONTINUOUS:
        hi = jnp.clip(idx, 0, n_grid - 1)
        lo = jnp.clip(idx - 1, 0, n_grid - 1)
        cdf_lo = jnp.where(idx <= 0, 0.0, cdf[lo])      # CDF below the bracket (0 for the first bin)
        cdf_hi = cdf[hi]
        frac = jnp.clip((us - cdf_lo) / (cdf_hi - cdf_lo + 1e-12), 0.0, 1.0)  # in [0,1], smooth in p_emit
        mesh_lo = angle_dist_mesh[lo]
        mesh_hi = angle_dist_mesh[hi]
        selected_angle_dist = (1.0 - frac[:, None]) * mesh_lo + frac[:, None] * mesh_hi
        sampled_angle = selected_angle_dist[:, 0]
        sampled_dist = selected_angle_dist[:, 1]
        # Continuous sampling already carries the energy gradient pathwise; the DiCE score is not needed.
        dice_score = jnp.ones_like(dice_score)

    # Frozen occupancy-match scale (DETACHED, fixed reference energy E_CAL so it injects no
    # spurious dc/dE): ∝density sampling has a larger per-photon weight-sum (Σd²/Σd) than the
    # OLD top-k-uniform scheme (mean of the top num_seeds densities); this constant rescales it
    # back to the validated forward occupancy without touching the gradient direction.
    E_CAL = 1050.0
    cal_grid = jnp.stack([jnp.full((n_grid,), E_CAL), angle_dist_mesh[:, 0], angle_dist_mesh[:, 1]], axis=1)
    cal_norm = normalize_inputs_jit(cal_grid, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max)
    cal_log, _ = model.apply(model_params, cal_norm)
    cal_dens = jax.lax.stop_gradient(jnp.clip(denormalize_log_predictions(jnp.squeeze(cal_log), log_max, log_min), 1e-12, None))
    ns_cal = jnp.int32(num_seeds_a * jnp.power(E_CAL, num_seeds_b) + num_seeds_c)
    ds_sorted = jnp.sort(cal_dens)[::-1]
    mean_topk = jnp.sum(jnp.where(jnp.arange(n_grid) < ns_cal, ds_sorted, 0.0)) / ns_cal
    occ_scale = jax.lax.stop_gradient(mean_topk / (jnp.sum(cal_dens * cal_dens) / jnp.sum(cal_dens)))

    if not _SIREN_CONTINUOUS:                         # discrete (piecewise-constant) inverse-CDF lookup
        selected_angle_dist = angle_dist_mesh[idx]
        sampled_angle = selected_angle_dist[:, 0]
        sampled_dist  = selected_angle_dist[:, 1]

    # Per-photon multiplicative weight factor applied to mean_topk in the final weight (1 = uniform).
    # Variations override this and/or (sampled_angle, sampled_dist, dice_score) below.
    weight_factor = jnp.ones((Nphot,))
    # static side length of the (angle, distance) grid (n_grid = nb_static**2); n_bins from table_data is
    # a TRACED value under jit, so derive the reshape dimension from the concrete grid shape instead.
    nb_static = int(round(n_grid ** 0.5))

    # --- VARIATION A: FIXED-PROPOSAL IMPORTANCE REWEIGHT -----------------------------------------
    # Resample the bins ONCE at the fixed reference energy E_CAL (cal_dens), so the SAMPLE idx is
    # ENERGY-INDEPENDENT (no step function in E -> no delta-spike 2nd derivative). Energy then enters
    # ONLY through the smooth per-photon density ratio p_emit(E)[idx]/p_emit(E_CAL)[idx], which is
    # pathwise -> exact 1st AND 2nd order. At E=E_CAL the ratio is 1 (identical forward to current).
    # No DiCE score is needed (the selection is detached and energy-independent).
    if _SIREN_IMPORTANCE:
        p_cal = cal_dens / jnp.sum(cal_dens)
        cdf_cal = jnp.cumsum(p_cal)
        idx_A = jnp.clip(jnp.searchsorted(cdf_cal, us), 0, n_grid - 1)
        selected = angle_dist_mesh[idx_A]
        sampled_angle = selected[:, 0]
        sampled_dist = selected[:, 1]
        # smooth density ratio (energy in numerator only; denominator is the detached E_CAL proposal)
        weight_factor = p_emit[idx_A] / jax.lax.stop_gradient(p_cal[idx_A])
        dice_score = jnp.ones((Nphot,))

    # --- VARIATION B: GAUSSIAN-SMOOTHED 2-D DENSITY + CONTINUOUS INVERSE-CDF ----------------------
    # Blur the 2-D (angle,distance) density with a small separable Gaussian (sigma _SIREN_SMOOTH_SIGMA
    # bins) BEFORE forming the flattened CDF, so per-bin occupancy is C^2-smooth in energy; then the
    # continuous (linear-interp) inverse-CDF over the SMOOTHED flattened CDF. Smoothing removes the
    # sparse-CDF kinks (1/(cdf_hi-cdf_lo), clip(frac)) that made the raw SIREN_CONTINUOUS path diverge.
    if _SIREN_SMOOTH:
        dens2d = density.reshape(nb_static, nb_static)                  # [angle, distance]
        sm2d = _smooth2d_separable(dens2d, _SIREN_SMOOTH_SIGMA)
        dens_s = jnp.clip(sm2d.reshape(-1), 1e-12, None)
        p_s = dens_s / jnp.sum(dens_s)
        cdf_s = jnp.cumsum(p_s)
        idx_s = jnp.clip(jnp.searchsorted(cdf_s, us), 0, n_grid - 1)
        hi = idx_s
        lo = jnp.clip(idx_s - 1, 0, n_grid - 1)
        cdf_lo = jnp.where(idx_s <= 0, 0.0, cdf_s[lo])
        cdf_hi = cdf_s[hi]
        frac = jnp.clip((us - cdf_lo) / (cdf_hi - cdf_lo + 1e-12), 0.0, 1.0)
        sel = (1.0 - frac[:, None]) * angle_dist_mesh[lo] + frac[:, None] * angle_dist_mesh[hi]
        sampled_angle = sel[:, 0]
        sampled_dist = sel[:, 1]
        dice_score = jnp.ones((Nphot,))                          # continuous reparam carries the gradient

    # --- VARIATION C: MARGINAL FACTORED CONTINUOUS -----------------------------------------------
    # Factor the 2-D density into a 1-D marginal angle density (sum over distance) and a per-angle-bin
    # conditional distance density. Each 1-D axis (n_bins=250) is far denser than the flattened 2-D, so
    # a continuous inverse-CDF per axis is well-behaved. Sample the angle from the marginal continuous
    # inverse-CDF; pick the conditional distance row at the (detached) angle bin and sample distance from
    # that row's continuous inverse-CDF. Energy flows pathwise through both 1-D probabilities.
    if _SIREN_MARGINAL:
        dens2d = density.reshape(nb_static, nb_static)                  # [angle, distance]
        marg_a = jnp.sum(dens2d, axis=1)                         # marginal angle density (n_bins,)
        p_a = marg_a / jnp.sum(marg_a)
        cdf_a = jnp.cumsum(p_a)
        # continuous angle sample over the angle support [angle_min, angle_max]
        us_a = us                                               # systematic stratification (one axis)
        sampled_angle = _continuous_invcdf_1d(cdf_a, us_a, angle_min, angle_max)
        # detached angle bin for the conditional-distance row lookup (selection only, no gradient)
        a_bin = jnp.clip(jnp.searchsorted(cdf_a, us_a), 0, nb_static - 1)
        # second stratified stream for distance (independent permutation -> decorrelate the two axes)
        key, subkey_cdist = random.split(key)
        u0d = random.uniform(subkey_cdist, ()) / Nphot
        us_d = u0d + random.permutation(subkey_cdist, Nphot) / Nphot
        rows = dens2d[a_bin]                                     # (Nphot, n_bins) conditional dist rows
        p_d = rows / jnp.sum(rows, axis=1, keepdims=True)
        cdf_d = jnp.cumsum(p_d, axis=1)
        n = nb_static
        idx_d = jnp.clip(jax.vmap(lambda c, u: jnp.searchsorted(c, u))(cdf_d, us_d), 0, n - 1)
        ar = jnp.arange(Nphot)
        cdf_d_hi = cdf_d[ar, idx_d]
        cdf_d_lo = jnp.where(idx_d <= 0, 0.0, cdf_d[ar, jnp.clip(idx_d - 1, 0, n - 1)])
        frac_d = jnp.clip((us_d - cdf_d_lo) / (cdf_d_hi - cdf_d_lo + 1e-12), 0.0, 1.0)
        dxd = (distance_max - distance_min) / n
        sampled_dist = distance_min + (idx_d + frac_d) * dxd
        dice_score = jnp.ones((Nphot,))

    # ============================================================================
    # STRATIFIED SAMPLING: Better coverage than pure MC
    # ============================================================================
    bin_width_angle = (angle_max - angle_min) / n_bins
    bin_width_dist = (distance_max - distance_min) / n_bins

    # Create stratified samples: divide [0, 1] into Nphot strata
    # Then shuffle to avoid systematic bias
    key, subkey_angle = random.split(key)
    key, subkey_dist = random.split(key)
    key, subkey_jitter_angle = random.split(key)
    key, subkey_jitter_dist = random.split(key)

    # Permute stratum indices for random assignment
    strata_indices_angle = random.permutation(subkey_angle, Nphot)
    strata_indices_dist = random.permutation(subkey_dist, Nphot)

    # Sample within each stratum: (stratum_index + uniform[0,1]) / Nphot
    jitter_angle = random.uniform(subkey_jitter_angle, (Nphot,))
    jitter_dist = random.uniform(subkey_jitter_dist, (Nphot,))

    strata_angle = (strata_indices_angle + jitter_angle) / Nphot
    strata_dist = (strata_indices_dist + jitter_dist) / Nphot

    # Map from [0, 1] to [-bin_width/2, bin_width/2]
    stratified_angle = (strata_angle - 0.5) * bin_width_angle
    stratified_dist = (strata_dist - 0.5) * bin_width_dist

    smeared_angle = sampled_angle + stratified_angle
    smeared_dist = sampled_dist + stratified_dist

    photon_thetas = smeared_angle

    # Generate ray vectors and origins
    subkey, subkey2 = random.split(subkey)
    ray_vectors = generate_random_cone_vectors(track_direction, photon_thetas, Nphot, subkey)

    # Convert ranges to meters and compute ray origins
    ranges = smeared_dist / 1000
    ray_origins = jnp.ones((Nphot, 3)) * track_origin[None, :] + ranges[:, None] * normalize(track_direction[None, :])

    # Per-photon weight. Photons are already importance-sampled ∝density (the inverse-CDF resample above),
    # so a UNIFORM forward weight makes the physical emission ∝density (correct ring width). mean_topk keeps
    # the total normalization (Σ = Nphot·mean_topk, same as the legacy density²·occ_scale total). The DiCE
    # score carries the energy gradient (d log p_emit/dE). Legacy density² path kept behind the flag.
    if _SIREN_DENSITY_SQUARED:
        weights = density[idx] * dice_score * occ_scale            # legacy: emission ∝ density² (over-sharp)
    else:
        # mean_topk keeps the validated total normalization; dice_score carries the energy gradient for
        # the default/discrete path (==1 for the continuous/A/B/C reparameterized paths); weight_factor is
        # the importance ratio for Variation A (==1 for all others, so default path is byte-identical).
        weights = mean_topk * dice_score * weight_factor           # FIX: emission ∝ density (correct width)
    weights = jnp.where(smeared_angle < angle_min, 0.0, weights)
    weights = jnp.where(smeared_angle > angle_max, 0.0, weights)
    weights = jnp.where(smeared_dist < distance_min, 0.0, weights)
    weights = jnp.where(smeared_dist > distance_max, 0.0, weights)
    if _SIREN_DIST_CAP_MM > 0:                                     # optional physical-range cap (test)
        weights = jnp.where(smeared_dist > _SIREN_DIST_CAP_MM, 0.0, weights)

    return ray_vectors, ray_origins, weights


@jit
def predict_t0(distance, energy, baseline_slope, baseline_intercept,
                   A_slope, A_intercept, B_slope, B_intercept, offset):
    """
    JAX JIT-compatible version of predict_t0.
    Parameters are passed as individual arrays/scalars instead of nested dict.
    """
    # Baseline from 1000 MeV linear fit
    baseline = baseline_slope * distance + baseline_intercept

    # Calculate delta timing
    log10_A = A_slope * energy + A_intercept
    B = B_slope * energy + B_intercept
    delta = 10**log10_A * jnp.power(distance, B) + offset

    return baseline + delta

# Helper function to unpack your existing params dict
def predict_t0_wrapper(distance, energy, params):
    """Wrapper to use your existing params dict structure"""
    return predict_t0(
        distance, energy,
        params['baseline']['slope'],
        params['baseline']['intercept'],
        params['delta_parameterization']['A_slope'],
        params['delta_parameterization']['A_intercept'],
        params['delta_parameterization']['B_slope'],
        params['delta_parameterization']['B_intercept'],
        params['delta_parameterization']['offset']
    )
