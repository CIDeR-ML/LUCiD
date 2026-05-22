"""SIREN-based photon ray generation.

Track-mode photon generation for LUCiD. For a charged-particle track of a
given energy, the SIREN model predicts the photon density over the
``(opening-angle, s/s_max)`` phase space; this module turns that into a set of
weighted photon rays.

Approach (post s/s_max refactor):
  * Latin-Hypercube sample ``Nphot`` points over the ``(angle, s/s_max)``
    domain — area-uniform, so no region is left undersampled.
  * One SIREN evaluation at those points.
  * The SIREN-predicted weights are normalised to a PMF and multiplied by the
    physical photon count ``N_photons(E)`` — so ``sum(intensity) == N_photons(E)``
    exactly. SIREN supplies the *shape*, the stored power law supplies the *scale*.
  * The dimensionless ``s/s_max`` coordinate is converted to a physical
    distance ``s = (s/s_max)·s_max(E)`` along the track.

There is no fixed grid and no importance sampling — the compact s/s_max phase
space makes both unnecessary.
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


def latin_hypercube_2d(key, nphot, x_lo, x_hi, y_lo, y_hi):
    """Area-uniform Latin-Hypercube sample of ``nphot`` points over the 2D box
    ``[x_lo, x_hi] x [y_lo, y_hi]``.

    Each axis is split into ``nphot`` equal strata with exactly one sample per
    stratum (``(perm + uniform_jitter) / nphot``); two independent permutations
    pair the axes so the strata are decorrelated. The result is area-uniform —
    no importance weighting — so no region can be left severely undersampled.

    ``nphot`` is a Python int (it sizes the permutations / jitter arrays).
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
    """Draw ``nphot`` LHS points over the (angle, s/s_max) domain, evaluate the
    SIREN once, and return ``(weights, angle, s_over_smax)``.

    ``weights`` are the denormalised photon densities, **before** PMF
    normalisation — the raw SIREN prediction. Shared by the ray function and
    the ``validate.py`` integral diagnostic (which sums the raw weights as a
    Monte-Carlo estimate of the phase-space integral).
    """
    angle, s_over_smax = latin_hypercube_2d(
        key, nphot,
        ctx.angle_min, ctx.angle_max,
        ctx.smax_dist_min, ctx.smax_dist_max,
    )
    grid = jnp.stack([
        jnp.full_like(angle, energy),   # energy (MeV)
        angle,                          # opening angle (radians)
        s_over_smax,                    # s / s_max  ∈ [0, 1]
    ], axis=1)
    normalized = normalize_inputs_jit(
        grid, ctx.energy_min, ctx.energy_max,
        ctx.angle_min, ctx.angle_max,
        ctx.smax_dist_min, ctx.smax_dist_max,
    )
    raw, _ = ctx.model.apply(model_params, normalized)
    w = denormalize_log_predictions(jnp.squeeze(raw), ctx.log_max, ctx.log_min)
    return jnp.maximum(w, 0.0), angle, s_over_smax


def make_photonsim_ray_fn(ctx):
    """Build the jitted track-mode ray generator for a given model context.

    ``ctx`` is a :class:`lucid.siren.core.PhotonSimContext` (SIREN model,
    domain ranges, log-normalisation range, and the ``s_max(E)`` / ``N_photons(E)``
    closures). The simulator builds this once at setup.

    Returns ``photonsim_differentiable_get_rays(track_origin, track_direction,
    energy, Nphot, model_params, key)`` -> ``(ray_vectors, ray_origins,
    photon_intensities)``:

      * one SIREN evaluation at ``Nphot`` LHS points;
      * ``s/s_max`` -> physical mm via ``ctx.s_max_fn(E)``;
      * ``intensity = (w / Σw) · N_photons(E)`` — so the per-ray intensities
        sum to the physical photon count for that energy.
    """

    @partial(jax.jit, static_argnums=(3,))
    def photonsim_differentiable_get_rays(track_origin, track_direction,
                                          energy, Nphot, model_params, key):
        eval_key, cone_key = random.split(key)

        # One SIREN evaluation over an area-uniform LHS draw of the phase space.
        w, angle, s_over_smax = evaluate_siren_lhs(
            ctx, model_params, energy, Nphot, eval_key)

        # s/s_max -> physical distance along the track (mm -> m for origins).
        physical_dist_mm = s_over_smax * ctx.s_max_fn(energy)
        ranges_m = physical_dist_mm / 1000.0
        ray_origins = (track_origin[None, :]
                       + ranges_m[:, None] * normalize(track_direction)[None, :])

        # Photons emitted on a cone at the SIREN-predicted opening angle.
        ray_vectors = generate_random_cone_vectors(
            track_direction, angle, Nphot, cone_key)

        # Shape (SIREN, normalised to a PMF) x absolute scale (N_photons(E)).
        # sum(intensity) == N_photons(E) by construction.
        pmf = w / jnp.maximum(jnp.sum(w), 1e-30)
        intensity = pmf * ctx.n_photons_fn(energy)

        return ray_vectors, ray_origins, intensity

    return photonsim_differentiable_get_rays


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
