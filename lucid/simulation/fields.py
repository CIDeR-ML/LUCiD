"""
fields.py
---------
Pluggable 3D-varying optical-property maps (``spatial fields``).

A *field* is a dimensionless multiplicative correction ``f(x) ≥ 0`` on an optical
coefficient, evaluated at a position ``x`` in the detector frame (meters). It lets a
scalar global property (e.g. ``absorption_length``) acquire spatial structure while
composing cleanly with the existing wavelength axis::

    mu_eff(x, lam) = (1 / absorption_length) * abs_dev(lam) * field(x)

Design (see also the reflection-model precedent in ``reflection.py``):

* The **representation** (poly / siren / ...), its shape hyperparameters, the input
  encoding, and the detector geometry ``(R, H)`` are all **static**. They are fixed at
  setup time by ``make_field(...)`` and captured in the simulator's step closure — they
  are never JAX-traced and never live in ``DetectorParams``. Changing any of them is a
  structural change that triggers a (deliberate, rare) recompile.

* The **parameters** (poly coefficients, siren weights) are the only traced, optimizable
  quantity. They live as a single pytree leaf ``DetectorParams.absorption.field_params``
  and flow through grad / optax / normalize like any other calibration parameter.

``make_field`` returns the matched pair ``(apply_fn, init_params)`` so the static closure
and the traced leaf are always built together with consistent pytree structure (treedef).

Gauge: every non-uniform field is built **without a constant mode**, so ``field(x)``
averages to ~1 over the volume and carries only *spatial deviation*. The overall
normalization stays with the global ``absorption_length`` — the field cannot silently
absorb (or fight) it.

Conventions
-----------
``apply_fn(field_params, xyz) -> corrections``
    ``xyz``  : ``(..., 3)`` points in the detector frame (meters)
    returns  : ``(...,)`` multiplicative corrections (``1.0`` ≡ no spatial variation)

``init_params``
    the identity-correction parameters (``None`` for ``uniform``; all-zero coeffs for
    ``poly``), so a freshly built field is byte-identical to the homogeneous detector.
"""

from typing import Any, Callable, Tuple

import numpy as np
import jax
import jax.numpy as jnp

# The SIREN reps reuse the shared network. ``square_output=False`` gives the LINEAR SIREN a
# spatial field needs (a signed deviation, field = 1 +/- dev); the default squared output is
# only for the non-negative Cherenkov/dedx density surrogate. The shared SIREN's symmetric
# init (fixed alongside square_output) is what makes from-scratch JAX training work.
from lucid.siren.core import SIREN


def _field_siren(hidden_features, hidden_layers, w0):
    return SIREN(hidden_features=hidden_features, hidden_layers=hidden_layers,
                 out_features=1, outermost_linear=True, w0=w0, square_output=False)


# ---------------------------------------------------------------------------
# Input encoding (geometry-aware — no hardcoded detector size)
# ---------------------------------------------------------------------------

def encode_cylindrical(xyz: jnp.ndarray, R: float, H: float) -> jnp.ndarray:
    """Cartesian ``(..., 3)`` → normalized cylindrical features ``(..., 4)``.

        u_r   = sqrt(x^2 + y^2) / R     ∈ [0, 1]
        cos_t = x / r                   ∈ [-1, 1]
        sin_t = y / r                   ∈ [-1, 1]
        u_z   = 2 z / H                 ∈ [-1, 1]
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    # eps INSIDE the sqrt: r is never exactly 0, so its gradient (x/r) stays finite even on
    # the cylinder axis — the same 0/0 sqrt-gradient floor the rest of the engine relies on
    # (see photon_step's dropped-custom_vjp note). Without it, an on-axis sample point poisons
    # the backward pass with NaN even though the sample geometry is detached.
    r = jnp.sqrt(x * x + y * y + 1e-12)
    # Clamp the encoded coords to the field's physical domain. A no-op for valid in-detector
    # photons (dist ≤ surface_distance ⇒ samples stay inside); only saturates pathological
    # masked photons, keeping the Chebyshev basis bounded (no inf → no masked-NaN in backward).
    u_r = jnp.clip(r / R, 0.0, 1.0)
    u_z = jnp.clip(2.0 * z / H, -1.0, 1.0)
    return jnp.stack([u_r, x / r, y / r, u_z], axis=-1)


# ---------------------------------------------------------------------------
# Chebyshev basis (static degree → built at make_field time, no tracing)
# ---------------------------------------------------------------------------

def _cheb(u: jnp.ndarray, deg: int) -> jnp.ndarray:
    """Chebyshev-T basis ``[T_0(u), ..., T_deg(u)]`` along a new last axis."""
    cols = [jnp.ones_like(u)]
    if deg >= 1:
        cols.append(u)
    for _ in range(2, deg + 1):
        cols.append(2.0 * u * cols[-1] - cols[-2])
    return jnp.stack(cols, axis=-1)            # (..., deg+1)


def _azimuth(cos_t: jnp.ndarray, sin_t: jnp.ndarray, n_harm: int) -> jnp.ndarray:
    """Azimuthal harmonics ``[1, cos t, sin t, cos 2t, sin 2t, ...]``."""
    cols = [jnp.ones_like(cos_t)]
    ck, sk = cos_t, sin_t                       # cos(kt), sin(kt) via recurrence
    for k in range(1, n_harm + 1):
        if k > 1:
            ck, sk = ck * cos_t - sk * sin_t, sk * cos_t + ck * sin_t
        cols.append(ck)
        cols.append(sk)
    return jnp.stack(cols, axis=-1)             # (..., 1 + 2*n_harm)


# ---------------------------------------------------------------------------
# Field builders — each returns (apply_fn, init_params) with matched treedef
# ---------------------------------------------------------------------------

def _make_uniform() -> Tuple[Callable, None]:
    """No-op field: ``f(x) ≡ 1``. ``init_params=None`` ⇒ byte-identical homogeneous run."""
    def apply(field_params, xyz):
        return jnp.ones(xyz.shape[:-1], xyz.dtype)
    return apply, None


def _make_poly(R: float, H: float,
               deg_r: int = 3, deg_z: int = 3, n_azim: int = 2) -> Tuple[Callable, jnp.ndarray]:
    """Separable Chebyshev(r) × Chebyshev(z) × azimuthal-harmonic polynomial.

    ``field(x) = 1 + Σ_{i,j,m}≠const  c[i,j,m] · T_i(u_r) T_j(u_z) A_m(θ)``

    The pure-constant mode ``(i=j=0, m=0)`` is excluded (gauge fix), so ``c = 0`` gives
    exactly ``1`` and the field never competes with the global ``absorption_length``.

    Evaluated by **mode-by-mode (separable) contraction**, never forming the full
    ``(..., n_coeff)`` outer-product basis. The naive `Tr[ii]*Tz[jj]*Az[mm]` gather
    materializes an ``N×n_coeff`` tensor (≈5 GB at N=16e6, n_coeff=79) and is
    memory-bound — ~16× slower than the staged einsum below for an identical result
    (the basis functions themselves cost only ~2 ms; the gather/contract was the wall).
    """
    n_r, n_z, n_a = deg_r + 1, deg_z + 1, 1 + 2 * n_azim
    # Flat indices into a dense (n_r, n_z, n_a) coeff tensor, dropping the constant (0,0,0)
    # so the fittable vector stays length n_coeff and the constant mode is pinned to 0.
    flat = np.array([i * n_z * n_a + j * n_a + m
                     for i in range(n_r) for j in range(n_z) for m in range(n_a)
                     if not (i == 0 and j == 0 and m == 0)])
    n_coeff = len(flat)

    def apply(field_params, xyz):
        # Scatter the gauge-fixed coeff vector into the dense tensor ONCE (per call, not per
        # point); the constant mode (0,0,0) stays 0. Cheap — operates on n_coeff, not N.
        C = jnp.zeros(n_r * n_z * n_a).at[flat].set(field_params).reshape(n_r, n_z, n_a)
        enc = encode_cylindrical(xyz, R, H)
        u_r, cos_t, sin_t, u_z = enc[..., 0], enc[..., 1], enc[..., 2], enc[..., 3]
        Tr = _cheb(2.0 * u_r - 1.0, deg_r)      # map [0,1]→[-1,1] for conditioning
        Tz = _cheb(u_z, deg_z)
        Az = _azimuth(cos_t, sin_t, n_azim)
        # Staged contraction: azimuth → z → r. Largest per-point intermediate is (n_r, n_z),
        # not n_coeff, and there is no gather. XLA fuses the chain.
        t1 = jnp.einsum('...m,ijm->...ij', Az, C)        # (..., n_r, n_z)
        t2 = jnp.einsum('...j,...ij->...i', Tz, t1)      # (..., n_r)
        return 1.0 + jnp.einsum('...i,...i->...', Tr, t2)

    init_params = jnp.zeros(n_coeff)
    return apply, init_params


def _make_siren(R: float, H: float,
                hidden_features: int = 128, hidden_layers: int = 2, w0: float = 10.0,
                out_scale: float = 5.0, seed: int = 0) -> Tuple[Callable, Any]:
    """SIREN field — reuses the shared lucid.siren.core.SIREN (square_output=False).

    ``field(x) = 1 + SIREN(encode(x)) / out_scale``. The Flax weight dict is the traced
    ``field_params`` leaf; the ``model`` (architecture) is static, captured in the closure.
    Not gauge-fixed to exactly 1 at init (SIREN has no zero-able constant mode) — keep
    ``out_scale`` large so the init correction stays near 1.
    """
    model = _field_siren(hidden_features, hidden_layers, w0)

    def apply(field_params, xyz):
        enc = encode_cylindrical(xyz, R, H)
        out, _ = model.apply({"params": field_params}, enc)
        return 1.0 + out.squeeze(-1) / out_scale

    init_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 4)))["params"]
    return apply, init_params


def _trilinear(grid, fr, ft, fz, n_r, n_theta, n_z):
    """Trilinear interpolation of ``grid`` (n_r, n_theta, n_z) at continuous indices.

    r, z are clamped; theta is PERIODIC (wraps). Differentiable in the grid values
    (gather + linear blend) — the gradient flows to every touched voxel.
    """
    r0 = jnp.clip(jnp.floor(fr).astype(jnp.int32), 0, n_r - 2); wr = fr - r0
    z0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, n_z - 2); wz = fz - z0
    t0 = jnp.mod(jnp.floor(ft).astype(jnp.int32), n_theta);    wt = ft - jnp.floor(ft)
    r1, z1, t1 = r0 + 1, z0 + 1, jnp.mod(t0 + 1, n_theta)

    def g(ri, ti, zi):
        return grid[ri, ti, zi]

    def lerp(a, b, w):
        return a + (b - a) * w

    # blend z, then theta, then r
    c00 = lerp(g(r0, t0, z0), g(r0, t0, z1), wz); c01 = lerp(g(r0, t1, z0), g(r0, t1, z1), wz)
    c10 = lerp(g(r1, t0, z0), g(r1, t0, z1), wz); c11 = lerp(g(r1, t1, z0), g(r1, t1, z1), wz)
    return lerp(lerp(c00, c01, wt), lerp(c10, c11, wt), wr)


def _cyl_indices(xyz, R, H, n_r, n_theta, n_z):
    """Encode points to continuous (fr, ft, fz) grid indices on the cylindrical grid."""
    enc = encode_cylindrical(xyz, R, H)
    u_r, cos_t, sin_t, u_z = enc[..., 0], enc[..., 1], enc[..., 2], enc[..., 3]
    theta = jnp.arctan2(sin_t, cos_t)
    fr = u_r * (n_r - 1)
    fz = (u_z + 1.0) * 0.5 * (n_z - 1)
    ft = (theta + jnp.pi) / (2.0 * jnp.pi) * n_theta            # periodic, [0, n_theta]
    return fr, ft, fz


def _make_grid(R: float, H: float,
               n_r: int = 8, n_theta: int = 8, n_z: int = 8) -> Tuple[Callable, jnp.ndarray]:
    """Voxel-grid field: trilinear-interpolated cylindrical grid, voxels ARE the params.

    ``field(x) = 1 + trilinear(grid, x)``. Cheap (a gather + blend per point, no MLP, no
    basis), smooth (C0), and local — captures sharp structure a low-order poly cannot, at
    the cost of O(n_r·n_theta·n_z) params. Init zeros ⇒ field ≡ 1 (identity). Gauge: the
    grid *mean* is degenerate with the global ``absorption_length`` (a uniform offset); pin
    ``mean(grid)=0`` during calibration, as with the poly constant mode.
    """
    def apply(field_params, xyz):
        fr, ft, fz = _cyl_indices(xyz, R, H, n_r, n_theta, n_z)
        return 1.0 + _trilinear(field_params, fr, ft, fz, n_r, n_theta, n_z)

    return apply, jnp.zeros((n_r, n_theta, n_z))


def _make_siren_grid(R: float, H: float,
                     n_r: int = 12, n_theta: int = 12, n_z: int = 12,
                     hidden_features: int = 128, hidden_layers: int = 2, w0: float = 10.0,
                     out_scale: float = 5.0, seed: int = 0) -> Tuple[Callable, Any]:
    """SIREN amortized onto a grid: params are SIREN weights, but each call BAKES the SIREN
    onto the fixed (n_r, n_theta, n_z) node lattice ONCE, then trilinear-interpolates.

    The bake depends only on the (broadcast) SIREN weights, not on the per-photon points, so
    it is vmap-invariant — XLA computes it once per step instead of once per photon, turning
    O(n_rays·K) MLP evals into O(grid)·K. Differentiable: grad flows SIREN→grid→interp. Grid
    resolution caps spatial frequency (fine for a smooth absorption field).
    """
    model = _field_siren(hidden_features, hidden_layers, w0)
    # Static encoded node lattice (u_r, cosθ, sinθ, u_z), matching _cyl_indices' convention.
    rr = np.linspace(0.0, 1.0, n_r); zz = np.linspace(-1.0, 1.0, n_z)
    tt = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    nodes = np.array([[r, np.cos(t), np.sin(t), z] for r in rr for t in tt for z in zz])
    nodes_j = jnp.asarray(nodes, dtype=jnp.float32)

    def apply(siren_params, xyz):
        out, _ = model.apply({"params": siren_params}, nodes_j)          # (G,1) — vmap-invariant bake
        grid = (out.squeeze(-1) / out_scale).reshape(n_r, n_theta, n_z)  # deviation grid
        fr, ft, fz = _cyl_indices(xyz, R, H, n_r, n_theta, n_z)
        return 1.0 + _trilinear(grid, fr, ft, fz, n_r, n_theta, n_z)

    init_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 4)))["params"]
    return apply, init_params


# ---------------------------------------------------------------------------
# Gauge fix — remove the field↔global-L_abs degeneracy
# ---------------------------------------------------------------------------

def _gauge_fix(apply: Callable, R: float, H: float, n_ref: int = 2048,
               seed: int = 12345) -> Callable:
    """Wrap a field apply so its spatial-deviation mean is pinned to 0.

    The absorption optical depth is ``(1/L_abs)·∫field ds``, so a uniform shift of the field
    is exactly degenerate with the global ``absorption_length`` (the field's mean ≡ a change
    in L_abs). For a JOINT global+field fit this makes the Hessian singular. The wrapper
    subtracts the field's mean (over a FIXED uniform-in-cylinder reference set) each call, so
    the field carries only spatial *deviation* and L_abs owns the normalization. The mean
    depends only on ``field_params`` (not the query points) ⇒ vmap-invariant, hoisted to
    once per step. Zero/identity params ⇒ mean already 0 ⇒ byte-identical (still ≡ 1).
    """
    rng = np.random.default_rng(seed)
    rr = R * np.sqrt(rng.uniform(size=n_ref))
    th = rng.uniform(-np.pi, np.pi, size=n_ref)
    zz = rng.uniform(-H / 2.0, H / 2.0, size=n_ref)
    ref = jnp.asarray(np.stack([rr * np.cos(th), rr * np.sin(th), zz], axis=-1), jnp.float32)

    def gauged(field_params, xyz):
        mean_dev = jnp.mean(apply(field_params, ref)) - 1.0   # mean of (field − 1) over ref
        return apply(field_params, xyz) - mean_dev            # → mean deviation 0

    return gauged


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_field(kind: str, *, R: float, H: float, gauge: bool = False,
               **hparams) -> Tuple[Callable, Any]:
    """Build a spatial field: returns ``(apply_fn, init_params)``.

    Called **once, at setup, outside JIT.** ``apply_fn`` closes over the static
    representation + geometry; ``init_params`` is the identity-correction leaf to seed
    ``DetectorParams.absorption.field_params``.

    Parameters
    ----------
    kind   : ``'uniform'`` | ``'poly'`` | ``'siren'`` | ``'grid'`` | ``'siren_grid'``
    R, H   : detector radius / height (m), from ``det_geom.detector`` — drives the encoding
    gauge  : if True, pin the field's spatial-deviation mean to 0 (removes the field↔L_abs
             degeneracy — REQUIRED for a joint global+field calibration). Identity-preserving.
    hparams: representation hyperparameters (e.g. poly ``deg_r``, ``deg_z``, ``n_azim``)
    """
    if kind == 'uniform':
        return _make_uniform()
    elif kind == 'poly':
        apply, init = _make_poly(R, H, **hparams)
    elif kind == 'siren':
        apply, init = _make_siren(R, H, **hparams)
    elif kind == 'grid':
        apply, init = _make_grid(R, H, **hparams)
    elif kind == 'siren_grid':
        apply, init = _make_siren_grid(R, H, **hparams)
    else:
        raise ValueError(
            f"unknown field kind {kind!r} (have: 'uniform', 'poly', 'siren', 'grid', 'siren_grid')")
    if gauge:
        apply = _gauge_fix(apply, R, H)
    return apply, init
