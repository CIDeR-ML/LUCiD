"""SIREN-based ray generation for Cherenkov and scintillation processes.

Track-mode photon generation for LUCiD. For a charged-particle track of a
given energy, a SIREN surrogate predicts the source density over a 2D phase
space; this module turns that into a set of weighted photon rays.

Two factories — same two-pass importance-sampling structure, different
domain axes and different absolute-scale normalization:

* :func:`make_cherenkov_surrogate_fn` — Cherenkov path. 2nd axis is the
  opening angle; absolute scale comes from ``ctx.n_photons_fn(E)``. Returns
  ``(ray_vectors, ray_origins, intensities)``.
* :func:`make_scintillation_surrogate_fn` — scintillation path. 2nd axis is
  ``dE/dx``; absolute scale comes from the medium's light-yield parameters
  (S, kB, C) via Chou. Returns ``(ray_vectors, ray_origins, intensities,
  t_delays, wavelengths)`` — scintillation owns its own emission time
  (biexponential mixture) and wavelength (Moyal) samplers, so the simulator
  doesn't have to special-case them.

Approach — two-pass importance sampling:
  1. Evaluate the SIREN on a ``grid_bins x grid_bins`` grid over the
     ``(angle, s/s_max)`` domain (first pass).
  2. Seed the ``Nphot`` rays only in grid bins whose weight clears
     ``threshold * max(grid)`` — drawn uniformly within the *area* of those
     bins (a categorical bin pick + uniform jitter) — then evaluate the SIREN
     a second time at the jittered seed points.

Concentrating the rays in the bright part of the phase space (the Cherenkov
pattern is sharply peaked) means a few rays describe it well — area-uniform
sampling over the whole domain wastes most rays on near-zero weight. The seed
region is computed on the fly from the SIREN values + a threshold knob (from
``siren_params.json``); there is no parametrised seed-count function.

Per-ray intensity is ``(w / Σw) * N_photons(E)``: SIREN supplies the *shape*
(a PMF), a stored power law supplies the *scale*, so ``sum(intensity)`` equals
``N_photons(E)`` exactly. No extra normalisation is needed — the discarded
sub-threshold tail is simply folded into the kept rays.

The dimensionless ``s/s_max`` coordinate is converted to a physical distance
``s = (s/s_max)·s_max(E)`` along the track.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial

from lucid.utils import normalize, generate_orthonormal_basis


@partial(jax.jit, static_argnums=(2,))
def generate_random_cone_vectors(R, theta, num_vectors, key):
    """Generate random vectors uniformly distributed on a cone surface.

    Parameters
    ----------
    R : jnp.ndarray
        Direction vector of the cone axis
    theta : float or jnp.ndarray
        Opening angle(s) of the cone in radians (scalar or (num_vectors,))
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
def normalize_inputs_jit(inputs, energy_min, energy_max, angle_min, angle_max,
                         distance_min, distance_max):
    """
    Normalize inputs to the range [-1, 1] in all dimensions.

    Args:
        inputs: Array of shape (..., 3) containing [energy, angle, distance] values.
                The 3rd column is the dimensionless s/s_max coordinate.
        energy_min, energy_max: The minimum and maximum energy values.
        angle_min, angle_max: The minimum and maximum angle values.
        distance_min, distance_max: The minimum and maximum s/s_max values.

    Returns:
        Array of shape (..., 3) with normalized values in the range [-1, 1].
    """
    energy = inputs[:, 0]
    angle = inputs[:, 1]
    distance = inputs[:, 2]

    normalized_energy = 2.0 * (energy - energy_min) / (energy_max - energy_min) - 1.0
    normalized_angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
    normalized_distance = 2.0 * (distance - distance_min) / (distance_max - distance_min) - 1.0

    return jnp.stack([normalized_energy, normalized_angle, normalized_distance], axis=1)


def _siren_weights(ctx, model_params, energy, axis2, s_over_smax):
    """Evaluate the SIREN at the given (axis2, s/s_max) points for ``energy``.

    ``axis2`` is the opening angle (radians) for Cherenkov contexts or dE/dx
    (keV/mm) for dE/dx contexts; the SIREN doesn't care which — both are just
    the 2nd input axis of the surrogate.

    Returns denormalised, non-negative weights. Shared by the grid pass, the
    seed pass, and the LHS diagnostic helper.
    """
    grid = jnp.stack([
        jnp.full_like(axis2, energy),   # energy (MeV)
        axis2,                          # opening angle (Cherenkov) or dE/dx (dedx)
        s_over_smax,                    # s / s_max  ∈ [0, 1]
    ], axis=1)
    normalized = normalize_inputs_jit(
        grid, ctx.energy_min, ctx.energy_max,
        ctx.axis2_min, ctx.axis2_max,
        ctx.smax_dist_min, ctx.smax_dist_max,
    )
    raw, _ = ctx.model.apply(model_params, normalized)
    w = denormalize_log_predictions(jnp.squeeze(raw), ctx.log_max, ctx.log_min)
    return jnp.maximum(w, 0.0)


def latin_hypercube_2d(key, nphot, x_lo, x_hi, y_lo, y_hi):
    """Area-uniform Latin-Hypercube sample of ``nphot`` points over the 2D box
    ``[x_lo, x_hi] x [y_lo, y_hi]``.

    Each axis is split into ``nphot`` equal strata with exactly one sample per
    stratum (``(perm + uniform_jitter) / nphot``); two independent permutations
    pair the axes so the strata are decorrelated.
    """
    k_px, k_py, k_jx, k_jy = random.split(key, 4)
    perm_x = random.permutation(k_px, nphot)
    perm_y = random.permutation(k_py, nphot)
    jit_x = random.uniform(k_jx, (nphot,))
    jit_y = random.uniform(k_jy, (nphot,))
    u_x = (perm_x + jit_x) / nphot
    u_y = (perm_y + jit_y) / nphot
    x = x_lo + u_x * (x_hi - x_lo)
    y = y_lo + u_y * (y_hi - y_lo)
    return x, y


def evaluate_siren_lhs(ctx, model_params, energy, nphot, key):
    """Draw ``nphot`` area-uniform LHS points over the whole (axis2, s/s_max)
    domain and evaluate the SIREN once.

    Returns ``(weights, axis2, s_over_smax)`` with raw (pre-PMF) weights. Used
    by the ``validate.py`` integral diagnostic, which sums the raw weights over
    an area-uniform draw as a Monte-Carlo estimate of the phase-space integral.
    """
    axis2, s_over_smax = latin_hypercube_2d(
        key, nphot,
        ctx.axis2_min, ctx.axis2_max,
        ctx.smax_dist_min, ctx.smax_dist_max,
    )
    w = _siren_weights(ctx, model_params, energy, axis2, s_over_smax)
    return w, axis2, s_over_smax


def make_cherenkov_surrogate_fn(ctx):
    """Build the jitted Cherenkov ray generator for a Cherenkov SIREN context.

    ``ctx`` is a :class:`lucid.siren.core.SirenContext` (SIREN model, domain
    ranges, log-normalisation range, the ``s_max(E)`` / ``N_photons(E)``
    closures, and the ``grid_bins`` / ``threshold`` ray-sampling knobs).
    Constructed from :func:`build_cherenkov_context`. The simulator builds
    this once at setup.

    Returns ``cherenkov_get_rays(track_origin, track_direction, energy, Nphot,
    model_params, key)`` -> ``(ray_vectors, ray_origins, intensities)`` using
    the two-pass importance sampling described in the module docstring.
    """
    g = ctx.grid_bins
    # Bin-center grid over (angle, s/s_max) — constant, closed over.
    a_edges = jnp.linspace(ctx.axis2_min, ctx.axis2_max, g + 1)
    s_edges = jnp.linspace(ctx.smax_dist_min, ctx.smax_dist_max, g + 1)
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    a_bin_w = (float(a_centers[1] - a_centers[0]) if g > 1
               else float(ctx.axis2_max - ctx.axis2_min))
    s_bin_w = (float(s_centers[1] - s_centers[0]) if g > 1
               else float(ctx.smax_dist_max - ctx.smax_dist_min))
    AA, SS = jnp.meshgrid(a_centers, s_centers, indexing='ij')
    grid_angle = AA.ravel()      # (g*g,) bin-center angles
    grid_s = SS.ravel()          # (g*g,) bin-center s/s_max
    threshold = ctx.threshold

    @partial(jax.jit, static_argnums=(3,))
    def cherenkov_get_rays(track_origin, track_direction,
                           energy, Nphot, model_params, key):
        pick_key, jit_key, cone_key = random.split(key, 3)

        # --- pass 1: SIREN on the bin-center grid ---
        grid_w = _siren_weights(ctx, model_params, energy, grid_angle, grid_s)

        # --- seed region: bins at/above threshold * per-energy grid max.
        # `>=` keeps the peak bin (threshold < 1) and degrades gracefully to
        # the whole domain if the grid is all-zero (sub-threshold energy).
        thresh = threshold * jnp.max(grid_w)
        above = grid_w >= thresh                          # (g*g,) bool mask

        # Uniformly pick Nphot of the M above-threshold bins. Drawing a rank in
        # [0, M) and mapping it to a bin via searchsorted on the inclusive
        # cumulative count keeps everything (Nphot,)- and (g*g,)-sized — unlike
        # random.categorical, which would materialise a (Nphot, g*g) array.
        csum = jnp.cumsum(above.astype(jnp.int32))        # (g*g,), runs 0..M
        n_seed_bins = csum[-1]                            # M (traced scalar)
        rank = random.randint(pick_key, (Nphot,), 0, n_seed_bins)
        bin_idx = jnp.searchsorted(csum, rank + 1)        # r-th above-thresh bin

        # --- uniform jitter within the chosen bins (uniform over their area) ---
        jitter = random.uniform(jit_key, (2, Nphot)) - 0.5
        angle = grid_angle[bin_idx] + jitter[0] * a_bin_w
        s_over_smax = grid_s[bin_idx] + jitter[1] * s_bin_w

        # --- pass 2: SIREN at the jittered seed points ---
        w = _siren_weights(ctx, model_params, energy, angle, s_over_smax)

        # s/s_max -> physical distance along the track (mm -> m for origins).
        physical_dist_mm = s_over_smax * ctx.s_max_fn(energy)
        ranges_m = physical_dist_mm / 1000.0
        ray_origins = (track_origin[None, :]
                       + ranges_m[:, None] * normalize(track_direction)[None, :])

        # Photons emitted on a cone at the SIREN-predicted opening angle.
        ray_vectors = generate_random_cone_vectors(
            track_direction, angle, Nphot, cone_key)

        # Shape (SIREN, normalised to a PMF) x absolute scale (N_photons(E)).
        # sum(intensities) == N_photons(E) by construction.
        pmf = w / jnp.maximum(jnp.sum(w), 1e-30)
        intensities = pmf * ctx.n_photons_fn(energy)

        return ray_vectors, ray_origins, intensities

    return cherenkov_get_rays


# ---------------------------------------------------------------------------
# Scintillation surrogate
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(1,))
def _sample_isotropic(key, n):
    """Sample ``n`` uniform unit vectors on the sphere."""
    k1, k2 = random.split(key)
    cos_theta = 2.0 * random.uniform(k1, (n,)) - 1.0
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta ** 2, 0.0))
    phi = 2.0 * jnp.pi * random.uniform(k2, (n,))
    return jnp.stack([sin_theta * jnp.cos(phi),
                      sin_theta * jnp.sin(phi),
                      cos_theta], axis=1)


def _biexp_pdf(t, tau_r, tau_1, tau_2, R_1):
    """Mixture-biexponential emission-time PDF.

    ``g(t; tau_r, tau_d) = (exp(-t/tau_d) - exp(-t/tau_r)) / (tau_d - tau_r)``;
    ``p(t) = R_1 g(t; tau_r, tau_1) + (1-R_1) g(t; tau_r, tau_2)``.

    Both ``tau_d - tau_r`` denominators get a tiny epsilon to stay
    differentiable at the (unphysical) degenerate point ``tau_d == tau_r``.
    """
    def g(tau_d):
        denom = (tau_d - tau_r) + 1e-9
        return (jnp.exp(-t / tau_d) - jnp.exp(-t / tau_r)) / denom
    return R_1 * g(tau_1) + (1.0 - R_1) * g(tau_2)


def _moyal_pdf(x, loc, scale):
    """Moyal PDF — used as the WbLS emission-spectrum shape.

    ``f(x; mu, sigma) = (1/(sigma·sqrt(2π))) · exp(-0.5·(z + exp(-z)))``
    where ``z = (x - mu)/sigma``. The medium-level ``moyal_amp`` then scales
    this into the per-photon wavelength weight.
    """
    z = (x - loc) / scale
    return (1.0 / (scale * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
        -0.5 * (z + jnp.exp(-z)))


# Maximum emission delay sampled from the biexponential. 10× tau_2 of 10%
# WbLS (27 ns) covers >99.94% of the integrated PDF — additional photons
# past this are accepted as a negligible truncation.
_SCINTILLATION_T_MAX_NS = 200.0


def make_scintillation_surrogate_fn(dedx_ctx, scint_lambda_min, scint_lambda_max,
                                    T_max_ns: float = _SCINTILLATION_T_MAX_NS):
    """Build the jitted scintillation ray generator for a dE/dx SIREN context.

    Two-pass importance sampling on the dE/dx SIREN:
      1. Evaluate the dE/dx SIREN on a ``grid_bins x grid_bins`` grid over
         ``(dE/dx, s/s_max)`` and threshold on the energy density
         ``w · dE/dx`` (rare-but-very-ionizing bins are kept).
      2. Pick ``N_deposits = Nphot // 5`` bins from the seed region, jitter,
         re-evaluate the SIREN — these are the energy deposits along the
         track.

    Each deposit emits 5 rays. Direction, emission delay (biexp mixture) and
    wavelength (Moyal) are independently sampled per ray. The per-ray
    intensity is::

        (photons_per_deposit / 5)
            · p_biexp(t_j; tau_r, tau_1, tau_2, R_1) · T_max
            · moyal_amp · moyal_pdf(λ_j; loc, scale) · (λ_max - λ_min)

    with ``photons_per_deposit = S · dE_i / (1 + kB·d_i + C·d_i²)`` (Chou)
    and ``dE_i = E · (w_i·d_i) / Σ(w·d)`` (PMF — sums to E across deposits).

    All 10 medium-level scintillation scalars (S, kB, C, tau_r, tau_1, tau_2,
    R_1, moyal_amp/loc/scale) are arguments of the returned function so
    gradients flow through to them.

    Returns ``scintillation_get_rays(track_origin, track_direction, energy,
    Nphot, model_params, key, S, kB, C, tau_r, tau_1, tau_2, R_1,
    moyal_amp, moyal_loc, moyal_scale)`` ->
    ``(ray_vectors, ray_origins, intensities, t_delays, wavelengths)``.
    ``t_delays`` are biexp samples relative to the deposit time; the caller
    is expected to add the Cherenkov ``predict_t0(s, E)`` baseline (the time
    the track itself reaches each deposit).
    """
    g = dedx_ctx.grid_bins
    # Bin-center grid over (dE/dx, s/s_max) — constant, closed over.
    d_edges = jnp.linspace(dedx_ctx.axis2_min, dedx_ctx.axis2_max, g + 1)
    s_edges = jnp.linspace(dedx_ctx.smax_dist_min, dedx_ctx.smax_dist_max, g + 1)
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    d_bin_w = (float(d_centers[1] - d_centers[0]) if g > 1
               else float(dedx_ctx.axis2_max - dedx_ctx.axis2_min))
    s_bin_w = (float(s_centers[1] - s_centers[0]) if g > 1
               else float(dedx_ctx.smax_dist_max - dedx_ctx.smax_dist_min))
    DD, SS = jnp.meshgrid(d_centers, s_centers, indexing='ij')
    grid_dedx = DD.ravel()       # (g*g,) bin-center dE/dx (keV/mm)
    grid_s = SS.ravel()          # (g*g,) bin-center s/s_max
    threshold = dedx_ctx.threshold
    lambda_window = float(scint_lambda_max - scint_lambda_min)
    lambda_min_f = float(scint_lambda_min)
    lambda_max_f = float(scint_lambda_max)

    @partial(jax.jit, static_argnums=(3,))
    def scintillation_get_rays(track_origin, track_direction, energy, Nphot,
                               model_params, key,
                               S, kB, C,
                               tau_r, tau_1, tau_2, R_1,
                               moyal_amp, moyal_loc, moyal_scale):
        # Each deposit emits 5 rays varying in time/direction/wavelength.
        n_deposits = Nphot // 5
        pick_key, jit_key, dir_key, t_key, lam_key = random.split(key, 5)

        # --- pass 1: dE/dx SIREN on the bin-center grid ---
        grid_w = _siren_weights(dedx_ctx, model_params, energy, grid_dedx, grid_s)

        # --- seed region: threshold on the energy density w·dE/dx so
        # rare-but-very-ionizing bins (low w, high dE/dx) are kept.
        energy_density = grid_w * grid_dedx
        thresh = threshold * jnp.max(energy_density)
        above = energy_density >= thresh

        # Uniformly pick n_deposits of the M above-threshold bins (same
        # randint + searchsorted trick as the Cherenkov path).
        csum = jnp.cumsum(above.astype(jnp.int32))
        n_seed_bins = csum[-1]
        rank = random.randint(pick_key, (n_deposits,), 0, n_seed_bins)
        bin_idx = jnp.searchsorted(csum, rank + 1)

        # --- uniform jitter within the chosen bins ---
        jitter = random.uniform(jit_key, (2, n_deposits)) - 0.5
        dedx_i = grid_dedx[bin_idx] + jitter[0] * d_bin_w
        s_i = grid_s[bin_idx] + jitter[1] * s_bin_w

        # --- pass 2: dE/dx SIREN at jittered deposit points ---
        w_i = _siren_weights(dedx_ctx, model_params, energy, dedx_i, s_i)

        # --- energy PMF: dE_i = E · (w·d) / Σ(w·d), sums to E ---
        wd = w_i * dedx_i
        dE = energy * wd / jnp.maximum(jnp.sum(wd), 1e-30)

        # --- Birks/Chou per-deposit photon count ---
        photons_per_deposit = S * dE / (1.0 + kB * dedx_i + C * dedx_i ** 2)

        # s/s_max -> physical distance along the track (mm -> m for origins).
        physical_dist_mm = s_i * dedx_ctx.s_max_fn(energy)
        ranges_m = physical_dist_mm / 1000.0
        deposit_origins = (track_origin[None, :]
                           + ranges_m[:, None] * normalize(track_direction)[None, :])

        # --- replicate each deposit 5 times: origin + photon count are
        # shared across the 5; direction / time / wavelength vary per ray.
        ray_origins = jnp.repeat(deposit_origins, 5, axis=0)
        photons_per_ray = jnp.repeat(photons_per_deposit, 5) / 5.0

        # --- per-ray isotropic directions ---
        ray_vectors = _sample_isotropic(dir_key, Nphot)

        # --- per-ray biexp time sample (uniform proposal + reweight) ---
        t_uniform = random.uniform(t_key, (Nphot,), minval=0.0, maxval=T_max_ns)
        time_weight = _biexp_pdf(t_uniform, tau_r, tau_1, tau_2, R_1) * T_max_ns

        # --- per-ray Moyal wavelength sample (uniform proposal + reweight)
        # Sampling inside [λ_min, λ_max] only — the user-supplied window
        # already enforces the "0 photons outside" cutoff.
        wavelengths = random.uniform(lam_key, (Nphot,),
                                     minval=lambda_min_f, maxval=lambda_max_f)
        lambda_weight = (moyal_amp * _moyal_pdf(wavelengths, moyal_loc, moyal_scale)
                         * lambda_window)

        intensities = photons_per_ray * time_weight * lambda_weight

        return ray_vectors, ray_origins, intensities, t_uniform, wavelengths

    return scintillation_get_rays


# ---------------------------------------------------------------------------
# t0 (Cherenkov vertex-time baseline — shared with scintillation as the
# deposit-time anchor; scintillation adds its biexp delay on top)
# ---------------------------------------------------------------------------


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
