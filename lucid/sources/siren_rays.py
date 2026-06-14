"""SIREN-based photon ray generation functions.

Moved from lucid/generate.py during Phase 2.2 refactor.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from typing import NamedTuple
from lucid.siren.core import SIREN
from lucid.utils import normalize, generate_orthonormal_basis, jax_rotate_vector_local

# Photon emission uses a fixed-proposal IMPORTANCE reweight: resample the (angle,distance) bins ONCE at a
# fixed reference energy E_CAL (so the bin selection is ENERGY-INDEPENDENT — no step function in E, hence no
# delta-spike in d²/dE²), then reweight each photon by the smooth density ratio p_emit(E)/p_emit(E_CAL).
# Energy enters ONLY through that pathwise ratio (exact 1st AND 2nd order; ratio==1 at E=E_CAL). The forward
# emission is ∝ density with the correct Cherenkov ring width (~11°).


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

class SirenContext(NamedTuple):
    """Inference-time SIREN context for the Cherenkov surrogate (refactor-v2 form). A plain
    Python container (Flax module + float fields + closures) — never traced; ``model_params``
    is passed separately as the traced argument to the ray factory."""
    model: object
    energy_min: float
    energy_max: float
    axis2_min: float
    axis2_max: float
    smax_dist_min: float
    smax_dist_max: float
    log_min: float
    log_max: float
    s_max_fn: object        # s_max(E_mev) -> mm
    n_photons_fn: object    # N_photons(E_mev) -> total photons/event
    grid_bins: int
    threshold: float


def _siren_weights(ctx, model_params, energy, axis2, s_over_smax):
    """Evaluate the SIREN at (axis2, s/s_max) for ``energy``; returns non-negative weights."""
    grid = jnp.stack([jnp.full_like(axis2, energy), axis2, s_over_smax], axis=1)
    normalized = normalize_inputs_jit(grid, ctx.energy_min, ctx.energy_max,
                                      ctx.axis2_min, ctx.axis2_max,
                                      ctx.smax_dist_min, ctx.smax_dist_max)
    raw, _ = ctx.model.apply(model_params, normalized)
    w = denormalize_log_predictions(jnp.squeeze(raw), ctx.log_max, ctx.log_min)
    return jnp.maximum(w, 0.0)


def latin_hypercube_2d(key, nphot, x_lo, x_hi, y_lo, y_hi):
    """Area-uniform Latin-Hypercube sample of ``nphot`` points over [x_lo,x_hi]x[y_lo,y_hi]."""
    k_px, k_py, k_jx, k_jy = random.split(key, 4)
    perm_x = random.permutation(k_px, nphot)
    perm_y = random.permutation(k_py, nphot)
    jit_x = random.uniform(k_jx, (nphot,))
    jit_y = random.uniform(k_jy, (nphot,))
    u_x = (perm_x + jit_x) / nphot
    u_y = (perm_y + jit_y) / nphot
    return x_lo + u_x * (x_hi - x_lo), y_lo + u_y * (y_hi - y_lo)


def make_cherenkov_surrogate_fn(ctx):
    """Build the jitted Cherenkov ray generator (refactor-v2 two-pass importance sampling).

    Returns ``cherenkov_get_rays(track_origin, track_direction, energy, Nphot, model_params, key)``
    -> ``(ray_vectors, ray_origins, intensities)``; ``sum(intensities) == n_photons_fn(E)``.
    """
    g = ctx.grid_bins
    a_edges = jnp.linspace(ctx.axis2_min, ctx.axis2_max, g + 1)
    s_edges = jnp.linspace(ctx.smax_dist_min, ctx.smax_dist_max, g + 1)
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    a_bin_w = float(a_centers[1] - a_centers[0]) if g > 1 else float(ctx.axis2_max - ctx.axis2_min)
    s_bin_w = float(s_centers[1] - s_centers[0]) if g > 1 else float(ctx.smax_dist_max - ctx.smax_dist_min)
    AA, SS = jnp.meshgrid(a_centers, s_centers, indexing='ij')
    grid_angle = AA.ravel()
    grid_s = SS.ravel()
    threshold = ctx.threshold

    @partial(jax.jit, static_argnums=(3,))
    def cherenkov_get_rays(track_origin, track_direction, energy, Nphot, model_params, key):
        pick_key, jit_key, cone_key = random.split(key, 3)
        grid_w = _siren_weights(ctx, model_params, energy, grid_angle, grid_s)  # pass 1: grid
        thresh = threshold * jnp.max(grid_w)
        above = grid_w >= thresh
        csum = jnp.cumsum(above.astype(jnp.int32))
        rank = random.randint(pick_key, (Nphot,), 0, csum[-1])
        bin_idx = jnp.searchsorted(csum, rank + 1)                              # r-th above-thresh bin
        jit_a, jit_s = latin_hypercube_2d(jit_key, Nphot, -0.5, 0.5, -0.5, 0.5)
        angle = grid_angle[bin_idx] + jit_a * a_bin_w
        s_over_smax = grid_s[bin_idx] + jit_s * s_bin_w
        w = _siren_weights(ctx, model_params, energy, angle, s_over_smax)       # pass 2: seeds
        physical_dist_mm = s_over_smax * ctx.s_max_fn(energy)
        ranges_m = physical_dist_mm / 1000.0
        ray_origins = track_origin[None, :] + ranges_m[:, None] * normalize(track_direction)[None, :]
        ray_vectors = generate_random_cone_vectors(track_direction, angle, Nphot, cone_key)
        pmf = w / jnp.maximum(jnp.sum(w), 1e-30)                                # SIREN shape -> PMF
        intensities = pmf * ctx.n_photons_fn(energy)                           # x absolute count
        return ray_vectors, ray_origins, intensities

    return cherenkov_get_rays


def build_cherenkov_context(predictor, n_photons_fn, ray_sampling=None):
    """SirenContext from OUR trained-model metadata + a count closure.

    Adapted for unification's PHYSICAL-distance net: the SIREN 3rd axis is physical distance
    (``dataset_info['distance_range']`` mm), so smax_dist range = that physical range and
    ``s_max_fn = 1.0`` — i.e. s_over_smax IS the physical distance in mm, queried as the net
    was trained. The absolute count comes from ``n_photons_fn`` (= tot_n_photons_normalization;
    refactor-v2 reads it from the net's 'nphot' metadata block, ours passes it in).
    """
    rs = {"grid_bins": 250, "threshold": 0.05}
    if ray_sampling:
        rs.update(ray_sampling)
    meta = predictor.metadata
    di = meta['dataset_info']
    tn = meta['target_normalization']
    emin, emax = di['energy_range']
    amin, amax = di['angle_range']
    dmin, dmax = di['distance_range']
    return SirenContext(
        model=predictor.model,
        energy_min=float(emin), energy_max=float(emax),
        axis2_min=float(amin), axis2_max=float(amax),
        smax_dist_min=float(dmin), smax_dist_max=float(dmax),
        log_min=float(tn['log_min']), log_max=float(tn['log_max']),
        s_max_fn=lambda E: 1.0,
        n_photons_fn=n_photons_fn,
        grid_bins=int(rs['grid_bins']), threshold=float(rs['threshold']),
    )


_C_MM_PER_NS = 299.792  # vacuum c in PhotonSim units (mm/ns)


def predict_t0(distance, energy, a_coeffs, l_coeffs, b_coeffs):
    """Photon emission-time baseline ``t(d,E)`` (refactor-v2 stretched_exp_delay form).

    ``t(d,E) = d/c + A(E)·(exp((d/λ(E))^β(E)) − 1)`` with d in mm, E in MeV, t in ns.
    ``log10 A``, ``log10 λ`` and ``β`` are each a CUBIC in ``log10 E``; a/l/b_coeffs are
    length-4 ascending ``[c0,c1,c2,c3]`` (loaded from the ``t0.json`` cubic schema). Fit over
    150–100000 MeV. Differentiable in E and d; used detached (stop_gradient) for emission-time.
    """
    x = jnp.log10(energy)

    def _cubic(c):
        return c[0] + c[1] * x + c[2] * x * x + c[3] * x * x * x

    A = 10.0 ** _cubic(a_coeffs)
    lam = 10.0 ** _cubic(l_coeffs)
    beta = _cubic(b_coeffs)
    arg = jnp.power(jnp.clip(distance / lam, 1e-12, None), beta)
    delay = A * (jnp.exp(arg) - 1.0)
    return distance / _C_MM_PER_NS + delay


def predict_t0_wrapper(distance, energy, params):
    """Evaluate :func:`predict_t0` from a nested-dict ``t0.json`` (the cubic schema)."""
    return predict_t0(
        distance, energy,
        params['A']['log10_poly_logE'],
        params['lambda']['log10_poly_logE'],
        params['beta']['poly_logE'],
    )
