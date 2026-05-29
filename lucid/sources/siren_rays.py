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

        # --- LHS-stratified jitter within the chosen bins ---
        # Same role as the previous independent-uniform jitter, but the
        # N jitter pairs are Latin-Hypercube-stratified across [-0.5, 0.5]²
        # — rays in the same bin land in different jitter strata; sub-bin
        # coverage of the Nphot rays is uniform rather than randomly
        # clustered.
        jit_a, jit_s = latin_hypercube_2d(
            jit_key, Nphot, -0.5, 0.5, -0.5, 0.5)
        angle = grid_angle[bin_idx] + jit_a * a_bin_w
        s_over_smax = grid_s[bin_idx] + jit_s * s_bin_w

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
    """Sample ``n`` unit vectors approximately uniformly on the sphere via
    Latin-Hypercube draws in ``(cos θ, φ)``.

    Independent uniform draws leave random "thin spots" on the sphere that
    show up as patchiness on the detector display; LHS guarantees every
    cos θ stratum and every φ stratum is hit exactly once across the n
    samples, dropping sphere-coverage error from O(1/√n) to O(1/n).
    """
    u_cos, u_phi = latin_hypercube_2d(key, n, 0.0, 1.0, 0.0, 1.0)
    cos_theta = 2.0 * u_cos - 1.0
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta ** 2, 0.0))
    phi = 2.0 * jnp.pi * u_phi
    return jnp.stack([sin_theta * jnp.cos(phi),
                      sin_theta * jnp.sin(phi),
                      cos_theta], axis=1)


def _biexp_pdf(t, tau_rise, tau_fall):
    """Rise+fall emission-time PDF.

    ``g(t; τ_rise, τ_fall) = (exp(-t/τ_fall) - exp(-t/τ_rise)) / (τ_fall - τ_rise)``,
    the density of ``T = T_rise + T_fall`` with ``T_rise ~ Exp(τ_rise)``,
    ``T_fall ~ Exp(τ_fall)`` (hypoexponential sum). The simulator samples ``T``
    directly via the reparametrization in :func:`_sample_hypoexp`; this PDF is
    kept so it can be plotted / compared against an empirical histogram.

    The ``τ_fall - τ_rise`` denominator gets a tiny epsilon to stay
    differentiable at the (unphysical) degenerate point ``τ_fall == τ_rise``.
    """
    denom = (tau_fall - tau_rise) + 1e-9
    return (jnp.exp(-t / tau_fall) - jnp.exp(-t / tau_rise)) / denom


def _sample_hypoexp(key, n, tau_rise, tau_fall):
    """Sample ``n`` times from the rise+fall biexp PDF via the hypoexp sum.

    ``T = -τ_rise · log(U₁) + -τ_fall · log(U₂)`` with ``U₁, U₂ ~ U(0, 1)``
    independent. Differentiable wrt both τ via reparametrization — they appear
    only as scalar multipliers on the uniform-derived `log(U)`s.

    The ``minval=jnp.finfo(jnp.float32).tiny`` is to keep `log(U)` finite when
    U lands exactly at 0.
    """
    k1, k2 = random.split(key)
    tiny = jnp.finfo(jnp.float32).tiny
    u1 = random.uniform(k1, (n,), minval=tiny, maxval=1.0)
    u2 = random.uniform(k2, (n,), minval=tiny, maxval=1.0)
    return -tau_rise * jnp.log(u1) + -tau_fall * jnp.log(u2)


def _moyal_pdf(x, loc, scale):
    """Moyal PDF — used as the WbLS emission-spectrum shape.

    ``f(x; mu, sigma) = (1/(sigma·sqrt(2π))) · exp(-0.5·(z + exp(-z)))``
    where ``z = (x - mu)/sigma``. The surrogate samples directly from the
    truncated Moyal via an inverse-CDF lookup built once at factory time —
    see :func:`_build_moyal_inverse_cdf`.
    """
    z = (x - loc) / scale
    return (1.0 / (scale * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
        -0.5 * (z + jnp.exp(-z)))


def _build_moyal_inverse_cdf(lambda_min, lambda_max, moyal_loc, moyal_scale,
                              n_knots: int = 512):
    """Precompute the inverse CDF of the Moyal distribution truncated to
    ``[lambda_min, lambda_max]``.

    Returns ``(cdf_knots, lambda_knots)`` ready for ``jnp.interp(u, ...)``.
    Built once at factory time (Python eval — the result is closed over as
    a JAX constant inside the jit'd ray fn). The truncation re-normalizes
    the CDF to span [0, 1] over the window — outside the window the
    surrogate emits zero rays by construction.
    """
    lambda_grid = jnp.linspace(lambda_min, lambda_max, n_knots)
    pdf = _moyal_pdf(lambda_grid, moyal_loc, moyal_scale)
    # Cumulative trapezoidal integral
    bw = lambda_grid[1:] - lambda_grid[:-1]
    cdf = jnp.concatenate([jnp.array([0.0]),
                           jnp.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * bw)])
    cdf = cdf / cdf[-1]
    return cdf, lambda_grid


def make_scintillation_surrogate_fn(dedx_ctx, scint_lambda_min, scint_lambda_max,
                                    moyal_loc: float, moyal_scale: float):
    """Build the jitted scintillation ray generator for a dE/dx SIREN context.

    Same two-pass importance-sampling structure as the Cherenkov surrogate —
    one ray per SIREN evaluation point — plus a Rao-Blackwell collapse over
    the dE/dx axis to kill the per-ray d fluctuation that the original
    formula carried through both the energy PMF and the Birks denominator.

    Assumes an **edep-weighted** dE/dx SIREN: the underlying PhotonSim
    histogram is filled with ``weight = step_edep / MeV`` (not unit counts),
    so ``grid_w`` is already an *energy density* over ``(dE/dx, s/s_max)``.
    The threshold and per-ray PMF therefore use ``w`` directly; no extra
    multiplication by ``d`` is needed. Equivalently, ``<d>_s`` collapses
    against an edep-weighted distribution of d, which equals
    ``<d²>_count / <d>_count`` — the "typical d where the energy is", not
    the "typical d where the steps are". Either is a defensible meaning of
    "typical"; edep-weighted is the physically right one for Birks math
    because Birks quenches *energy*, not steps.

    For ``Nphot`` rays:

    1. Evaluate the dE/dx SIREN on a ``grid_bins x grid_bins`` grid over
       ``(dE/dx, s/s_max)``.
    2. Compute the conditional mean ``<d>_s = Σ_d w(d, s)·d / Σ_d w(d, s)``
       for every s-bin from the full grid (no threshold mask).
    3. Threshold the grid on ``w`` directly (an edep density); pick ``Nphot``
       bins from the seed region, LHS-jitter, re-evaluate the SIREN to get
       the per-ray edep weight ``w_i``.
    4. 3-point parabolic Lagrange interpolation of ``<d>_s`` at the jittered
       s gives the per-ray "typical dE/dx" ``d_typ_i``.

    Per-ray intensity is the Chou-quenched photon count with ``d_i``
    replaced by ``d_typ_i`` in the Birks denominator::

        dE_i        = E · w_i / Σ_j w_j
        intensity_i = S · dE_i / (1 + kB · d_typ_i + C · d_typ_i²)

    The per-ray sample's d_i is still used to evaluate ``w_i`` (so the
    per-ray SIREN energy gradient is preserved), but no longer leaks
    bin-to-bin d-axis variance into the intensity. ``<d>_s`` depends on
    ``grid_w`` and hence on the input energy, adding a second SIREN-side
    gradient pathway through the Rao-Blackwell collapse.

    Direction is isotropic per ray (LHS on cos θ, φ); emission delay is
    sampled directly from the rise+fall biexp PDF via the hypoexp sum
    (``-τ_rise·log(U₁) + -τ_fall·log(U₂)``); wavelength is sampled
    directly from the Moyal CDF inverse (precomputed once at factory time
    on the ``[lambda_min, lambda_max]`` window).

    Differentiable wrt the 5 runtime args (S, kB, C, τ_rise, τ_fall). The
    Moyal sampling parameters (``moyal_loc``, ``moyal_scale``) are factory
    args — closed over as the inverse-CDF lookup; not gradient targets.

    Returns ``scintillation_get_rays(track_origin, track_direction, energy,
    Nphot, model_params, key, S, kB, C, tau_rise, tau_fall)`` ->
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

    # Moyal inverse-CDF lookup, truncated to [lambda_min, lambda_max].
    # Built once here in Python; closed over as a JAX constant in the ray fn.
    _moyal_cdf, _moyal_lam = _build_moyal_inverse_cdf(
        scint_lambda_min, scint_lambda_max, moyal_loc, moyal_scale)

    @partial(jax.jit, static_argnums=(3,))
    def scintillation_get_rays(track_origin, track_direction, energy, Nphot,
                               model_params, key,
                               S, kB, C, tau_rise, tau_fall):
        pick_key, jit_key, dir_key, t_key, lam_key = random.split(key, 5)

        # --- pass 1: dE/dx SIREN on the bin-center grid ---
        grid_w = _siren_weights(dedx_ctx, model_params, energy, grid_dedx, grid_s)

        # --- conditional mean dE/dx given s (Rao-Blackwell over the d-axis) ---
        # The per-ray sample's d_i is a noisy draw from p(d | s_i); collapsing
        # the d-axis analytically using the SIREN's full distribution gives a
        # smooth-in-s "typical dE/dx" that we use in place of d_i in both the
        # energy PMF and the Birks denominator. Uses all d-bins (no threshold
        # mask) — the threshold decides *where to sample*, not what the
        # physical conditional distribution of dE/dx looks like at that s.
        # `grid_w` is laid out row-major from meshgrid(indexing='ij'), so
        # reshape(g, g)[i, j] = SIREN at (d_centers[i], s_centers[j]).
        grid_w_2d      = grid_w.reshape(g, g)                            # (g_d, g_s)
        weighted_d_sum = jnp.sum(grid_w_2d * d_centers[:, None], axis=0) # (g_s,)
        total_w_at_s   = jnp.sum(grid_w_2d, axis=0)                       # (g_s,)
        expected_d     = weighted_d_sum / jnp.maximum(total_w_at_s, 1e-30)

        # --- seed region: threshold directly on grid_w. The SIREN is trained
        # on an edep-weighted PhotonSim histogram, so grid_w is already an
        # energy density (no extra dE/dx multiplication needed). High-dE/dx
        # bins are kept by the weighting itself.
        thresh = threshold * jnp.max(grid_w)
        above = grid_w >= thresh

        # Uniformly pick Nphot of the M above-threshold bins (same
        # randint + searchsorted trick as the Cherenkov path).
        csum = jnp.cumsum(above.astype(jnp.int32))
        n_seed_bins = csum[-1]
        rank = random.randint(pick_key, (Nphot,), 0, n_seed_bins)
        bin_idx = jnp.searchsorted(csum, rank + 1)

        # --- LHS-stratified jitter within the chosen bins ---
        jit_d, jit_s = latin_hypercube_2d(
            jit_key, Nphot, -0.5, 0.5, -0.5, 0.5)
        dedx_i = grid_dedx[bin_idx] + jit_d * d_bin_w
        s_i = grid_s[bin_idx] + jit_s * s_bin_w

        # --- pass 2: dE/dx SIREN at jittered deposit points (per-ray w_i,
        # still used as the per-ray frequency weight in the energy PMF) ---
        w_i = _siren_weights(dedx_ctx, model_params, energy, dedx_i, s_i)

        # --- 3-point Lagrange interpolation of <d>_s at the jittered s ---
        # bin_idx = d_idx · g + s_idx (row-major flat index from meshgrid +
        # ravel above), so s_idx = bin_idx % g and the t in [-0.5, 0.5] of
        # the parabolic Lagrange is jit_s itself. Boundary s-bins clamp the
        # missing neighbour to the home bin — the resulting overshoot is
        # bounded by ~0.125 · (neighbour gap) at the very edge of the grid.
        s_idx = bin_idx % g
        k_lo = jnp.clip(s_idx - 1, 0, g - 1)
        k_hi = jnp.clip(s_idx + 1, 0, g - 1)
        y_lo = expected_d[k_lo]
        y_ct = expected_d[s_idx]
        y_hi = expected_d[k_hi]
        t = jit_s
        d_typ = ((t * (t - 1.0) / 2.0) * y_lo
                 + (1.0 - t * t)        * y_ct
                 + (t * (t + 1.0) / 2.0) * y_hi)

        # --- energy PMF + Birks/Chou, evaluated at the typical dE/dx ---
        # w_i is already an edep weight (PhotonSim fills the dE/dx hist
        # weighted by step edep), so the per-ray energy share is
        # w_i / Σw_j directly. Per-ray d-axis fluctuation is gone; per-ray
        # frequency variation still comes from w_i (preserves the energy
        # gradient through the per-ray SIREN call).
        # sum(intensities) ≈ S·E·<1/(1+kB·<d>_s+...)>_w.
        dE = energy * w_i / jnp.maximum(jnp.sum(w_i), 1e-30)
        intensities = S * dE / (1.0 + kB * d_typ + C * d_typ ** 2)

        # s/s_max -> physical distance along the track (mm -> m for origins).
        physical_dist_mm = s_i * dedx_ctx.s_max_fn(energy)
        ranges_m = physical_dist_mm / 1000.0
        ray_origins = (track_origin[None, :]
                       + ranges_m[:, None] * normalize(track_direction)[None, :])

        # --- per-ray isotropic directions ---
        ray_vectors = _sample_isotropic(dir_key, Nphot)

        # --- per-ray emission delay (hypoexp sum — direct sample) ---
        t_delays = _sample_hypoexp(t_key, Nphot, tau_rise, tau_fall)

        # --- per-ray wavelength (Moyal inverse-CDF lookup) ---
        u = random.uniform(lam_key, (Nphot,))
        wavelengths = jnp.interp(u, _moyal_cdf, _moyal_lam)

        return ray_vectors, ray_origins, intensities, t_delays, wavelengths

    return scintillation_get_rays


# ---------------------------------------------------------------------------
# t0 (Cherenkov vertex-time baseline — shared with scintillation as the
# deposit-time anchor; scintillation adds its biexp delay on top)
#
# Schema: stretched_exp_delay
#   t(d, E) = d / c  +  A(E) * (exp((d/λ(E))^β(E)) − 1)
#   log10 A(E), log10 λ(E) and β(E) are each a cubic in log10 E:
#       value = c0 + c1·logE + c2·logE² + c3·logE³   (coeffs ascending)
#
# Three length-4 coefficient vectors total. The d/c term uses vacuum c
# because the underlying PhotonSim hist stores delay relative to vacuum-c
# (muon track time at β ≈ 1). Distance is in mm, energy in MeV, time in ns.
# ---------------------------------------------------------------------------

_C_MM_PER_NS = 299.792  # vacuum c in PhotonSim units


@jit
def predict_t0(distance, energy, a_coeffs, l_coeffs, b_coeffs):
    """Photon emission time t(d, E) for the Cherenkov vertex-time baseline.

    log10 A, log10 λ and β are each a cubic in log10 E; a_coeffs, l_coeffs,
    b_coeffs are length-4, ascending (c0 + c1·x + c2·x² + c3·x³). Returns
    d/c + delay, the stretched-exp parametrisation fit from
    PhotonHist_TimeDistanceNorm. JAX-jitted; vmaps cleanly over distance.
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
    """Evaluate predict_t0 from a nested-dict params (the t0.json structure)."""
    return predict_t0(
        distance, energy,
        params['A']['log10_poly_logE'],
        params['lambda']['log10_poly_logE'],
        params['beta']['poly_logE'],
    )
