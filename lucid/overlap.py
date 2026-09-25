import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Optional
from jax import vmap, jit
from functools import partial
import os
import json
import tempfile
import numpy as np
from lucid.utils import base_dir_path

def _natural_cubic_moments(x, y):
    """Second derivatives M_i for the natural cubic spline through (x, y), with M_0 = M_{n-1} = 0.
    Solved once at build time (Thomas algorithm); supports non-uniform knots. Used by the 'cubic'
    overlap mode to give a value-exact-at-knots AND C2 (continuous-curvature) overlap, so the
    autodiff Hessian wrt the photon->sensor distance is correct (unlike the C0 jnp.interp lookup)."""
    import numpy as _np
    x = _np.asarray(x, dtype=float); y = _np.asarray(y, dtype=float); n = x.shape[0]
    if n < 3:
        return _np.zeros(n)
    h = _np.diff(x)
    a = _np.zeros(n); b = _np.zeros(n); c = _np.zeros(n); rhs = _np.zeros(n)
    b[0] = 1.0; b[n - 1] = 1.0                              # natural BC: M_0 = M_{n-1} = 0
    for i in range(1, n - 1):
        a[i] = h[i - 1]; b[i] = 2.0 * (h[i - 1] + h[i]); c[i] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    for i in range(1, n):                                  # forward elimination
        w = a[i] / b[i - 1]; b[i] -= w * c[i - 1]; rhs[i] -= w * rhs[i - 1]
    M = _np.zeros(n); M[n - 1] = rhs[n - 1] / b[n - 1]      # back substitution
    for i in range(n - 2, -1, -1):
        M[i] = (rhs[i] - c[i] * M[i + 1]) / b[i]
    return M


# TABLE GEOMETRY CONSTANTS.
#
# _KSIG is the half-width of the lookup table's dense node region, in units of sigma. It was 3,
# which placed the handoff to sparse spacing where the kernel is NOT yet negligible, so the
# interpolant returned spurious overlap just past the dense region -- growing as the width narrows.
# An interpolation artefact, not a quadrature one: `mode='cubic'` returned NEGATIVE overlap over
# the same span, and a quadrature error would be common to both. Measured below, in
# `precompute_lookup`.
_KSIG = 8.0

# Dense nodes across [r - K*sigma, r + K*sigma]. Scaled WITH _KSIG so the node spacing at the sensor
# edge stays what it was at K = 3 with 150 nodes (~0.04 sigma). Widening K alone spreads the same
# nodes over 8/3 the span, and the piecewise-linear interpolant -- whose slope is what autodiff
# reads -- degrades exactly where the overlap changes fastest. Measured across the edge
# [r - 3s, r + 3s] against a float64 reference on a 4x finer grid, at temperature 0.2 and n = 1000
# (a comparison between layouts, whose interpolation error does not depend on n):
#       layout        edge spacing   max slope error   max value error
#       K=3, 150        0.040 s         1.17e-02          4.9e-05
#       K=8, 150        0.087 s         2.54e-02          2.3e-04
#       K=8, 400        0.033 s         9.63e-03          3.2e-05
# (at temperature 0.1: 1.16e-02, 3.09e-02, 1.07e-02). 400 keeps the halo fix without paying for it at
# the edge; the price is table build time, once per (r, sigma), since the table is cached.
_NUM_DENSE = 400

# create_overlap_prob's remaining layout defaults, named once so the signature and the
# default-layout cache check below cannot drift apart.
_N_QUAD = 2000            # n_theta and n_rho
_NUM_SPARSE = 50
_D_MAX_FACTOR = 10.0

# The narrowest sigma the convolution is built for, as a fraction of r; below this the call falls
# through to the straight-through hard step.
#
# THIS THRESHOLD IS A SILENT TRAP. A width sweep down to temperature 0.01 returned results
# BIT-IDENTICAL to temperature=None, because 0.01*r lands below the floor and takes the hard-step
# branch. A narrow-convolution arm that is secretly the hard step looks like a measurement and is
# not one. Going below 0.02 also needs `precompute_lookup` checked for convergence first: it
# integrates on a FIXED grid (n_rho=1000 across [0, r]) while sigma shrinks, so at 0.005*r the
# Gaussian gets ~5 radial points.
_ST_MIN_SIGMA = 0.02

# `np.trapezoid` is NumPy >= 2.0; `np.trapz` is its pre-2.0 name, deprecated in 2.0. pyproject.toml
# allows numpy>=1.24 and the container installs numpy unpinned beside jax=0.4, and the image warms
# this cache at BUILD time -- so on a 1.x resolve an unguarded call would fail `docker build`, and
# every CI job that needs the image with it.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def gaussian_kernel(rho: float, theta: float, d: float, r: float, sigma: float) -> float:
    """2D Gaussian distribution centered at the origin with std sigma.

    Parameters
    ----------
    rho : float
        Radial coordinate for integration
    theta : float
        Angular coordinate for integration
    d : float
        Distance from center
    r : float
        Circle radius
    sigma : float
        Standard deviation of the Gaussian

    Returns
    -------
    float
        Value of the Gaussian kernel at the given point
    """
    dist_sq = d ** 2 + rho ** 2 - 2 * d * rho * jnp.cos(theta)
    return (rho / (2 * jnp.pi * sigma ** 2)) * jnp.exp(-dist_sq / (2 * sigma ** 2))


def lorentz_kernel(rho: float, theta: float, d: float, r: float, gamma: float) -> float:
    """2D Lorentzian distribution centered at the origin.

    Parameters
    ----------
    rho : float
        Radial coordinate for integration
    theta : float
        Angular coordinate for integration
    d : float
        Distance from center
    r : float
        Circle radius
    gamma : float
        Width parameter of the Lorentzian

    Returns
    -------
    float
        Value of the Lorentzian kernel at the given point
    """
    dist_sq = d ** 2 + rho ** 2 - 2 * d * rho * jnp.cos(theta)
    return (gamma / (2 * jnp.pi)) * (rho / (gamma ** 2 + dist_sq) ** (3 / 2))


def get_cache_filename(r: float, sigma: float) -> str:
    """Generate a unique filename for caching results.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Width parameter

    Returns
    -------
    str
        Cache filename
    """
    # THE NODE LAYOUT IS PART OF THE TABLE'S IDENTITY, so it is part of the key. `_KSIG` sets
    # where the dense region hands off to the sparse one, i.e. the d_values themselves, and the
    # integration dtype sets their accuracy, and _NUM_DENSE sets the edge spacing -- so a table built
    # with K = 8 but the old 150 nodes must not be mistaken for this one. Keyed on (r, sigma) alone,
    # a run would silently LOAD
    # a table built with the old narrow region and single precision -- including from the shared
    # warm cache production points at -- and report it as the new result. The suffix keeps
    # pre-existing caches from ever being mistaken for tables built the current way.
    return (f"gaussian_overlap_r{r:.6f}_sigma{sigma:.6f}"
            f"_ksig{_KSIG:g}_nd{_NUM_DENSE}_f64.json")


_CACHE_SUBDIR = 'spatial_overlap_integrals'


_CACHE_DIR_OVERRIDE = None
_USER_CACHE_ROOT = None


def set_cache_dir(path: Optional[str]) -> None:
    """Point the overlap cache at an explicit directory (None restores default).

    The forward path reads no environment itself — see the B6 ratchet in
    tests/test_unification_pins.py — so a site that wants one warm shared cache
    sets it from the production/orchestration layer, which owns infra config.
    """
    global _CACHE_DIR_OVERRIDE
    _CACHE_DIR_OVERRIDE = path


def set_user_cache_root(path: Optional[str]) -> None:
    """Override the fallback root used when the install dir is not writable.

    Same reason as ``set_cache_dir``: $XDG_CACHE_HOME is read by the caller,
    not here. None restores ``~/.cache``.
    """
    global _USER_CACHE_ROOT
    _USER_CACHE_ROOT = path


def _cache_dirs() -> list:
    """Directories to search for cached overlap lookups, most-preferred first.

    The install dir is the historical location and stays first, so a writable
    checkout (or a shipped, pre-populated cache) behaves exactly as before. A
    read-only install — a container image, a root-owned site-packages — falls
    back to the user cache.
    """
    if _CACHE_DIR_OVERRIDE:
        return [os.path.join(_CACHE_DIR_OVERRIDE, _CACHE_SUBDIR)]
    root = _USER_CACHE_ROOT or os.path.join(os.path.expanduser('~'), '.cache')
    return [os.path.join(base_dir_path(), _CACHE_SUBDIR),
            os.path.join(root, 'lucid', _CACHE_SUBDIR)]


def _writable_cache_dir() -> Optional[str]:
    """First cache dir we can actually create/write, or None if none can be."""
    for d in _cache_dirs():
        try:
            os.makedirs(d, exist_ok=True)
            if os.access(d, os.W_OK):
                return d
        except OSError:
            continue
    return None


def save_overlap_values(r: float, sigma: float, d_values: jnp.ndarray, f_values: jnp.ndarray) -> None:
    """Save overlap values to a cache file.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Width parameter
    d_values : jnp.ndarray
        Array of distance values
    f_values : jnp.ndarray
        Array of overlap probabilities
    """
    # The cache is a pure function of (r, sigma), so failing to persist it costs
    # recomputation, never correctness — a read-only install must not be fatal.
    cache_dir = _writable_cache_dir()
    if cache_dir is None:
        return

    cache_data = {
        'r': float(r),
        'sigma': float(sigma),
        'd_values': d_values.tolist(),
        'f_values': f_values.tolist()
    }

    # Written to a temporary file and moved into place, so a reader never sees a half-written table:
    # parallel jobs sharing a cold cache all build the same table and all write it.
    filename = os.path.join(cache_dir, get_cache_filename(r, sigma))
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=cache_dir, suffix='.tmp', delete=False) as f:
            tmp = f.name
            json.dump(cache_data, f)
        os.replace(tmp, filename)
    except OSError:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)


def load_overlap_values(r: float, sigma: float) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Load overlap values from cache if they exist.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Width parameter

    Returns
    -------
    Optional[Tuple[jnp.ndarray, jnp.ndarray]]
        Cached values if they exist, None otherwise
    """
    name = get_cache_filename(r, sigma)
    filename = next((c for c in (os.path.join(d, name) for d in _cache_dirs())
                     if os.path.exists(c)), None)

    if filename is None:
        return None
    try:
        with open(filename, 'r') as f:
            cache_data = json.load(f)
        return jnp.array(cache_data['d_values']), jnp.array(cache_data['f_values'])
    except (OSError, ValueError, KeyError):
        return None        # unreadable or partial: rebuild it, as if it were not cached


@partial(jax.jit, device=jax.devices('cpu')[0])
def integral_f_of_d(d: float, r: float, sigma: float,
                    theta_vals: jnp.ndarray, rho_vals: jnp.ndarray) -> float:
    """Computes the double integral of the Gaussian kernel over polar coordinates.

    Parameters
    ----------
    d : float
        Distance from center
    r : float
        Circle radius
    sigma : float
        Standard deviation of the Gaussian
    theta_vals : jnp.ndarray
        Array of theta values for angular integration
    rho_vals : jnp.ndarray
        Array of rho values for radial integration

    Returns
    -------
    float
        Value of the double integral
    """

    def integrand_theta(theta):
        def integrand_rho(rho_):
            return gaussian_kernel(rho_, theta, d, r, sigma)

        return vmap(integrand_rho)(rho_vals)

    integrand = vmap(integrand_theta)(theta_vals)
    integral_theta = jnp.trapezoid(integrand, theta_vals, axis=0)
    integral_rho = jnp.trapezoid(integral_theta, rho_vals)
    return integral_rho


def precompute_lookup(r: float,
                      sigma: float,
                      n_theta: int = 1000,
                      n_rho: int = 1000,
                      num_dense: int = _NUM_DENSE,
                      num_sparse: int = 50,
                      d_max_factor: float = 10.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precomputes lookup tables for overlap probability calculation.

    Parameters
    ----------
    r : float
        Circle radius
    sigma : float
        Standard deviation of the Gaussian
    n_theta : int, optional
        Number of points for angular integration, by default 1000
    n_rho : int, optional
        Number of points for radial integration, by default 1000
    num_dense : int, optional
        Number of points in transition region, by default _NUM_DENSE (400)
    num_sparse : int, optional
        Number of points outside transition region, by default 50
    d_max_factor : float, optional
        Maximum distance as multiple of radius, by default 10.0

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Arrays of distances and overlap probabilities
    """
    # A Python float, so the float64 build below cannot be silently demoted: a JAX scalar sigma
    # would turn `rr / (2 pi sigma^2)` into a float32 array whenever x64 is off.
    sigma = float(sigma)
    if d_max_factor <= 1.0:
        # The table must reach past the disk edge. At d_max_factor <= 1 the dense region can start
        # beyond d_max and the node grid runs backwards, which jnp.interp does not detect.
        raise ValueError(f"d_max_factor must exceed 1, got {d_max_factor}")
    theta_vals = jnp.linspace(0, 2 * jnp.pi, n_theta)
    rho_vals = jnp.linspace(0, r, n_rho)

    # WHERE THE DENSE REGION HANDS OFF TO THE SPARSE ONE.
    #
    # The dense nodes span [r - K*sigma, r + K*sigma]; beyond that 24 sparse nodes stretch to
    # d_max and the default lookup interpolates between them LINEARLY. At the historical K = 3 the
    # handoff happens where f is still 1.3e-3, not zero, so the table draws a straight line from
    # 1.3e-3 down to the next sparse node -- while the truth falls off like a Gaussian and is
    # negligible within one more sigma. The result is spurious overlap weight across the halo.
    #
    # MEASURED: overlap weight the table places beyond r + 6*sigma, where the true overlap is below
    # 1e-9 -- the integral of f(d) * 2*pi*d over [r + 6s, 10r], as a fraction of the disk area:
    #       temperature    0.2       0.1       0.05      0.02
    #       K = 3        4.6e-08   2.0e-05   2.2e-04   4.2e-04
    #       K = 8        9.3e-11   4.0e-11   1.8e-11   6.7e-12      (400 dense nodes; n = 1000)
    # The fix is the layout, not the precision: K = 8 gives the same numbers in float32.
    transition_start = max(0, r - _KSIG * sigma)  # Ensure we don't go below 0
    # CLAMPED TO d_max. Unclamped, a wide kernel puts the end of the dense region BEYOND d_max and
    # the sparse tail below is built from linspace(transition_end, d_max) -- running BACKWARDS, so
    # d_values stops being monotonic and jnp.interp, which assumes increasing knots, returns
    # garbage without complaint. That needs sigma > (d_max_factor - 1) * r / _KSIG: above 3r at the
    # old K = 3, but above 1.125r at K = 8 with the default d_max_factor = 10. Every shipped width
    # is <= 0.2r, so this changes nothing in production; it keeps the wide-kernel regime, which the
    # overlap tests exercise at 3r, 6r and 30r, well-formed.
    transition_end = min(r + _KSIG * sigma, d_max_factor * r)

    # Dense spacing in transition region
    d_dense = jnp.linspace(transition_start, transition_end, num_dense)

    # Sparse spacing before and after transition region
    if transition_start > 0:
        d_sparse_before = jnp.linspace(0, transition_start, num_sparse // 2)[:-1]
    else:
        d_sparse_before = jnp.array([])

    if transition_end < d_max_factor * r:
        d_sparse_after = jnp.linspace(transition_end, d_max_factor * r, num_sparse // 2)[1:]
    else:
        d_sparse_after = jnp.array([])

    # Combine all regions
    d_values = jnp.concatenate((d_sparse_before, d_dense, d_sparse_after))

    # FLOAT64 PRECOMPUTE.
    #
    # The integrator is a trapezoid over n_theta * n_rho terms -- 4e6 at create_overlap_prob's
    # default n = 2000 -- and JAX runs it in float32 by default, where accumulation over that many
    # terms, not the grid spacing, limits it. Measured at n = 2000, on identical nodes:
    #       temperature     float32 vs float64     radial grid error at d = 0
    #           0.2              6.0e-07                  5.1e-07
    #           0.02             5.5e-05                  5.2e-05
    # So at the resolution production uses, removing the float32 error helps only as much as the
    # O(h^2) grid error it sits beside, which float64 does not touch (it falls 16x per 4x in n_rho).
    # Narrowing the width is limited by n_rho as much as by precision.
    #
    # THE COST IS BUILD TIME. This NumPy build takes ~54 s per table at temperature 0.2 and ~86 s at
    # 0.02 on one CPU core, against ~0.4 s for the float32 JAX build it replaces. It is paid once per
    # (r, sigma) because the table is cached -- but it is paid by the container build, which warms
    # the cache, by a cold-cache first job, and by tests that build with use_cache=False.
    #
    # NumPy rather than jax.config: this is a one-off, non-differentiated, cached computation, and
    # flipping the global x64 flag mid-session would silently change dtypes everywhere else.
    #
    # BUILD THE GRIDS IN FLOAT64, do not convert them. `theta_vals` and `rho_vals` above are float32
    # jnp arrays; converting them promotes coordinates that are ALREADY rounded, which alone held
    # the error at 2.9e-06 where re-deriving the grids gives ~1e-14. They are kept above only for
    # `integral_f_of_d`, which the overlap tests still use as an independent reference.
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, dtype=np.float64)
    rh = np.linspace(0.0, float(r), n_rho, dtype=np.float64)
    dv = np.asarray(d_values, dtype=np.float64)
    rr, tt = np.meshgrid(rh, th, indexing='ij')          # (n_rho, n_theta)
    cos_t = np.cos(tt)
    out = np.empty(dv.shape[0], dtype=np.float64)
    # CHUNKED over d: unchunked, the (n_d, n_rho, n_theta) array at the defaults is 448 x 2000 x 2000
    # float64 = 14 GB. `step` keeps each chunk at ~4e7 elements (~320 MB per array). The integral is
    # independent per d, so chunking changes nothing numerically.
    step = max(1, int(4e7 // max(1, rr.size)))
    for i0 in range(0, dv.shape[0], step):
        dd = dv[i0:i0 + step][:, None, None]
        # Same kernel as `gaussian_kernel`: squared distance from the disk point (rho, theta) to a
        # source offset by d along theta = 0, times the polar area element rho.
        dist2 = rr ** 2 + dd ** 2 - 2.0 * rr * dd * cos_t
        integ = (rr / (2.0 * np.pi * sigma ** 2)) * np.exp(-dist2 / (2.0 * sigma ** 2))
        out[i0:i0 + step] = _trapezoid(_trapezoid(integ, th, axis=2), rh, axis=1)
    return d_values, jnp.asarray(out)


def create_overlap_prob(sigma: Optional[float],
                        r: float,
                        n_theta: int = _N_QUAD,
                        n_rho: int = _N_QUAD,
                        num_dense: int = _NUM_DENSE,
                        num_sparse: int = _NUM_SPARSE,
                        d_max_factor: float = _D_MAX_FACTOR,
                        use_cache: bool = True,
                        st_width_frac: float = 0.35,
                        renorm: float = 1.0,
                        mode: str = 'interp') -> Callable[[float], float]:
    """Creates a function that calculates overlap probability between sensor and photon.

    Parameters
    ----------
    sigma : Optional[float]
        Width parameter (if None or < 0.02*r, uses step function)
    r : float
        Radius (must be >0)
    n_theta : int, optional
        Number of points for angular integration, by default 2000
    n_rho : int, optional
        Number of points for radial integration, by default 2000
    num_dense : int, optional
        Number of points in transition region, by default _NUM_DENSE (400)
    num_sparse : int, optional
        Number of points outside transition region, by default 50
    d_max_factor : float, optional
        Maximum distance as multiple of radius, by default 10.0
    use_cache : bool, optional
        Whether to use cached values if available, by default True
    st_width_frac : float, optional
        Straight-through surrogate width (fraction of r) for the hard-step
        overlap (sigma None / tiny). Backward gradient only; forward stays the
        hard step. ``<= 0`` -> pure hard step, no surrogate. Default 0.35.
    renorm : float, optional
        Global soft-overlap renormalization constant C = hard_total/soft_total
        (restores the total/energy lost to inter-sensor gaps without changing
        the gradient direction). Default 1.0 = OFF (byte-identical).
    mode : str, optional
        Lookup interpolation for the soft overlap: ``'interp'`` (default,
        piecewise-linear jnp.interp) or ``'cubic'`` (natural cubic spline, C2 —
        correct curvature for the autodiff Hessian wrt the photon->sensor distance).

    Returns
    -------
    Callable[[float], float]
        Function that takes distance d and returns overlap probability

    Raises
    ------
    ValueError
        If r is not positive
    """
    if r <= 0:
        raise ValueError("r must be positive.")

    # Use step function if sigma is None or very small.
    # STRAIGHT-THROUGH: the forward value is the hard step (occupancy/forward unchanged), but
    # the BACKWARD pass uses a smooth sigmoid surrogate so the deposit retains a gradient wrt
    # the photon->sensor distance (hence wrt the track position/direction). Without this, a hard
    # step has zero gradient a.e. and charge cannot constrain position. Forward is byte-identical
    # to the step, so this is NOT a soft-temperature (the model output is unchanged).
    if sigma is None or sigma < _ST_MIN_SIGMA * r:
        # surrogate width for the backward gradient only (fwd stays hard). Narrower -> sharper
        # (larger-magnitude) spatial gradient, closer to the true local slope; wider -> smoother.
        # st_width_frac<=0 -> PURE HARD step (no backward surrogate): overlap contributes zero
        # gradient; the track/position gradient then flows ONLY through DiCE score + smooth physics.
        st_frac = st_width_frac

        if st_frac <= 0.0:
            def overlap_prob(d: float) -> float:
                return jnp.where(d < r, 1.0, 0.0)            # hard, grad 0 a.e. (no surrogate)
            return overlap_prob

        st_width = st_frac * r

        def overlap_prob(d: float) -> float:
            hard = jnp.where(d < r, 1.0, 0.0)
            soft = jax.nn.sigmoid((r - d) / st_width)
            return jax.lax.stop_gradient(hard - soft) + soft   # fwd = hard; grad = d(soft)/dd

        return overlap_prob

    # SOFT-OVERLAP RENORMALIZATION (default 1.0 = OFF, byte-identical).
    # The soft overlap f(d) = mass of a sigma-Gaussian (centered at the photon) inside the sensor disk.
    # Convolution conserves the integral ONLY if all the spread is captured; the fraction landing in the
    # GAPS between sensors (no sensor to catch it) is LOST -> the soft total under-counts the hard top-hat
    # by ~1% (worse in sparse/dim regions). A GLOBAL constant C = hard_total/soft_total restores the total
    # (and hence the energy) without changing the gradient DIRECTION (it is a pure scale). Calibrate C once
    # per detector (ratio of temp=None to temp=0.1 total charge). See MISMATCH_PLAN.md.
    _RENORM = renorm

    # mode == 'cubic' falls through: load the EXACT lookup below, then interpolate it with a
    # natural cubic spline (C2) instead of jnp.interp (C0) -- value-exact at the knots AND correct
    # curvature, so it carries value, gradient, AND Hessian wrt the photon->sensor distance.

    # Try to load from cache first
    # The cache key names only (r, sigma) and the DEFAULT layout (_KSIG, _NUM_DENSE, float64). A
    # table built with any other layout would be written under that key and then served to every
    # default caller. So the cache is used for the default layout only; anything else is built
    # fresh. No caller in the repo passes a non-default layout with the cache on today.
    _default_layout = (n_theta == _N_QUAD and n_rho == _N_QUAD and num_dense == _NUM_DENSE
                       and num_sparse == _NUM_SPARSE and d_max_factor == _D_MAX_FACTOR)
    if use_cache and _default_layout:
        cached_values = load_overlap_values(r, sigma)
        if cached_values is not None:
            d_values, f_values = cached_values
        else:
            d_values, f_values = precompute_lookup(
                r, sigma, n_theta, n_rho, num_dense, num_sparse, d_max_factor
            )
            save_overlap_values(r, sigma, d_values, f_values)
    else:
        d_values, f_values = precompute_lookup(
            r, sigma, n_theta, n_rho, num_dense, num_sparse, d_max_factor
        )

    if mode == 'cubic':
        # Natural cubic spline through the precomputed knots: passes through the exact overlap value at
        # every knot (so value+gradient match the lookup) and is C2 (continuous 2nd derivative), so the
        # autodiff Hessian wrt d is correct -- unlike jnp.interp (C0: 2nd deriv 0 a.e. + delta spikes).
        _M = _natural_cubic_moments(d_values, f_values)
        _xk = jnp.asarray(d_values); _yk = jnp.asarray(f_values); _Mk = jnp.asarray(_M)
        _hk = _xk[1:] - _xk[:-1]
        _nseg = _xk.shape[0] - 1
        def overlap_prob(d: float) -> float:
            d_c = jnp.clip(d, _xk[0], _xk[-1])
            i = jnp.clip(jnp.searchsorted(_xk, d_c) - 1, 0, _nseg - 1)
            x0 = _xk[i]; x1 = _xk[i + 1]; y0 = _yk[i]; y1 = _yk[i + 1]
            m0 = _Mk[i]; m1 = _Mk[i + 1]; h = _hk[i]
            A = (x1 - d_c) / h; B = (d_c - x0) / h
            return _RENORM * (A * y0 + B * y1 + ((A ** 3 - A) * m0 + (B ** 3 - B) * m1) * (h * h) / 6.0)
        return overlap_prob

    def overlap_prob(d: float) -> float:
        d_clamped = jnp.clip(d, d_values[0], d_values[-1])
        return _RENORM * jnp.interp(d_clamped, d_values, f_values)

    return overlap_prob


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set up parameters
    r = 0.04  # radius
    gamma_values = [0.01 * r, 0.02 * r, 0.05 * r, 0.1 * r, 0.2 * r, 0.4 * r, 0.5 * r, 0.8 * r, 1.0 * r]
    gamma_labels = [f'{g / r:.2f}r' for g in gamma_values]

    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for different lines
    colors = plt.cm.viridis(jnp.linspace(0, 1, len(gamma_values)))

    # Generate d values from 0 to 8r
    d_values = jnp.linspace(0, 3 * r, 200)

    # Store values for table
    table_data = []
    table_d_values = [0, 2 * r, 3 * r, 4 * r]

    # Calculate and plot for each gamma
    for gamma, label, color in zip(gamma_values, gamma_labels, colors):
        overlap_func = create_overlap_prob(gamma, r)
        overlap_vmap = vmap(overlap_func)
        overlaps = overlap_vmap(d_values)

        # Plot with custom styling
        ax.plot(d_values / r, overlaps, '-', linewidth=2, label=f'σ = {label}', color=color)

        # Collect values for table
        table_values = [overlap_func(d) for d in table_d_values]
        table_data.append([label] + [f'{v:.2e}' for v in table_values])

    # Customize plot
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Distance (d/r)', fontsize=12)
    ax.set_ylabel('Overlap Probability', fontsize=12)
    ax.set_title('Overlap Probability vs Distance for Different σ Values (r = 0.04)', fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Add reference lines
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.3)

    # Set reasonable axis limits
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.1, 1.1)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig('output/overlap_probability.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

    # Print table without tabulate
    print("\nOverlap Probability Values at Selected Distances:")
    print("-" * 80)
    print(f"{'σ':>10} {'d = 0':>15} {'d = 2r':>15} {'d = 3r':>15} {'d = 4r':>15}")
    print("-" * 80)
    for row in table_data:
        print(f"{row[0]:>10} {row[1]:>15} {row[2]:>15} {row[3]:>15} {row[4]:>15}")
    print("-" * 80)