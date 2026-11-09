"""Per-PMT t0 + global TTS timing calibration (mie_hunter/timingcal.py, unified).

The per-PMT first-arrival over M flashes, with a per-PE transit-time spread σ_TTS, is

    first_arrival[p]  =  t_geo[p]  +  t0[p]  +  σ_TTS · min(z_1..z_{N_p}),   z ~ N(0,1)

with N_p ~ Poisson(μ_p) the detected count (hit only when N≥1). Over flashes this gives

    E[first - t_geo][p] = t0[p] + σ_TTS · E[min | N≥1](μ_p)        (MEAN  → t0)
    Var[first][p]       =          σ_TTS² · Var[min | N≥1](μ_p)    (SPREAD → TTS)

Key identifiability point: the MEAN early-bias is nearly degenerate with t0 (a per-PMT
t0 absorbs it), so TTS is recovered from the **variance** (t0-independent), and t0 then
from the mean with the bias removed. The global clock offset (degenerate with the source
emission time) is fixed by the gauge ``mean(t0)=0`` — the timing analog of ``mean(log k)=0``.

This is a CLOSED-FORM estimator on sample-mode truth (real M-flash shot noise), so — unlike
the charge CRB on the implicit engine — there is no √12 honesty factor; the M flashes ARE
the real statistics. ``t_geo`` and ``μ`` are known inputs (geometry + the charge stage).
"""

from functools import cache

import numpy as np
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# TTS occupancy order-statistic, Poisson-conditioned: the per-PMT first-arrival.
#
# A per-PE transit-time spread sigma makes a PMT with N detected PEs report
#   first_arrival = t_geo + sigma · min(z_1..z_N),  z_i ~ N(0,1).
# N is Poisson(μ) and the hit is only recorded when N≥1, so over flashes
#   E[first - t_geo] = sigma · E[min | N~Poisson(μ), N≥1]      (the EARLY BIAS → t0/mean)
#   Var[first]       = sigma² · Var[min | N~Poisson(μ), N≥1]   (the SPREAD → TTS)
# Both are Poisson-MIXED, N≥1-CONDITIONED order statistics — NOT the "μ fixed draws"
# version, which fails badly at μ≲1, exactly the low-occupancy regime where the timing
# levers live. The MEAN bias is nearly degenerate with a per-PMT t0 (t0 absorbs it), so
# TTS is identified from the VARIANCE (the spread is t0-independent) — hence both lookups.
#
# Built by exact quadrature of the order-statistic density
#   f_n(x) = n φ(x) S(x)^(n-1),  S = 1-Φ,  giving m_n=E[min_n], v_n=Var[min_n],
# then Poisson-conditioned over μ. Deterministic; evaluated by jnp.interp.
#
# Built on FIRST USE, not at import: it is an 8000-point quadrature over 600 orders followed by an
# 801-point Poisson mixture, and only calibrate_timing reads it, so at import every caller of
# lucid.fitting would pay for it and almost none use it (G6, J11; the before/after import cost is
# probe B5). functools.cache builds it once per process, and the jnp
# conversion happens at the use site so no jnp array -- hence no JAX backend -- exists at import.
# ---------------------------------------------------------------------------
_NMAX_PE = 600


@cache
def _order_stat_moments():
    """Exact E[min of n N(0,1)] and Var[min of n] for n=1..NMAX via quadrature.

    Returns
    -------
    tuple of (NMAX_PE,) float64 numpy arrays
        ``(m, v)``: ``m[i]`` is E[min of i+1 standard normals] and ``v[i]`` its variance, both
        dimensionless. Cached; the caller must not mutate them.
    """
    from scipy.special import log_ndtr as _log_ndtr

    x = np.linspace(-12.0, 8.0, 8000)
    dx = x[1] - x[0]
    logphi = -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)
    logS = _log_ndtr(-x)                                  # log(1-Φ(x)), stable
    n = np.arange(1, _NMAX_PE + 1)
    m = np.zeros(_NMAX_PE); v = np.zeros(_NMAX_PE)
    for i, nn in enumerate(n):
        f = np.exp(np.log(nn) + logphi + (nn - 1) * logS)
        norm = f.sum() * dx
        mi = (x * f).sum() * dx / norm
        qi = (x * x * f).sum() * dx / norm
        m[i] = mi; v[i] = max(qi - mi * mi, 1e-9)
    return m, v


@cache
def _build_occ_tables():
    """Poisson-conditioned first-arrival bias and variance on a mu grid.

    Returns
    -------
    tuple of (801,) float64 numpy arrays
        ``(mu_grid, bias, var)``: ``mu_grid`` is the mean detected-PE count (dimensionless),
        ``bias`` is E[min | N~Poisson(mu), N>=1] in units of the per-PE TTS sigma, and ``var`` is
        its variance in units of sigma squared. Cached; the caller must not mutate them.
    """
    from scipy.special import gammaln as _gammaln

    _M_N, _V_N = _order_stat_moments()
    mu_grid = np.concatenate([[1e-8], np.geomspace(1e-3, 400.0, 800)])
    bias = np.zeros_like(mu_grid); var = np.zeros_like(mu_grid)
    n = np.arange(1, _NMAX_PE + 1)
    for i, mu in enumerate(mu_grid):
        if mu < 1e-6:
            bias[i] = _M_N[0]; var[i] = _V_N[0]          # only N=1 (m_1≈0, v_1≈1)
            continue
        logp = -mu + n * np.log(mu) - _gammaln(n + 1.0)
        w = np.exp(logp) / max(1.0 - np.exp(-mu), 1e-300)   # P(N=n | N≥1)
        em = float(np.sum(w * _M_N))
        e2 = float(np.sum(w * (_V_N + _M_N ** 2)))           # E[min² | N≥1]
        bias[i] = em; var[i] = max(e2 - em * em, 1e-9)
    return mu_grid, bias, var


def _occ_tables_jnp():
    """:func:`_build_occ_tables` as jnp arrays, converted at each use site.

    Not cached: a first call inside ``jit`` would cache tracers, and the next trace or eager
    call would raise ``UnexpectedTracerError``. The numpy tables underneath are cached.

    Returns
    -------
    tuple of (801,) float32 jax arrays
        ``(mu_grid, bias, var)``, as in :func:`_build_occ_tables`.
    """
    mu_grid, bias, var = _build_occ_tables()
    return jnp.asarray(mu_grid), jnp.asarray(bias), jnp.asarray(var)


def _occ_bias_mean(mu):
    """Poisson-conditioned TTS occupancy early-bias E[min | N~Poisson(μ), N≥1] (≤0).
    →0 as μ→0, → −√(2 ln μ) for large μ. The per-PMT first-arrival MEAN shift is
    ``tts · _occ_bias_mean(μ)``. Differentiable in μ via ``jnp.interp``."""
    mu_grid, bias, _ = _occ_tables_jnp()
    return jnp.interp(mu, mu_grid, bias)


def _occ_bias_var(mu):
    """Poisson-conditioned Var[min | N~Poisson(μ), N≥1] (∈(0,1]). The per-PMT
    first-arrival VARIANCE is ``tts² · _occ_bias_var(μ)`` — the t0-independent signal
    that identifies TTS. →1 as μ→0 (a single N(0,1)), shrinks as occupancy rises."""
    mu_grid, _, var = _occ_tables_jnp()
    return jnp.interp(mu, mu_grid, var)



def calibrate_timing(fa_mean_obs, fa_var_obs, t_geo, mu, n_flashes, *,
                     weights=None, t0_true=None, tts_true=None):
    """Recover per-PMT ``t0`` and global ``TTS`` from per-PMT first-arrival moments.

    Parameters
    ----------
    fa_mean_obs, fa_var_obs : (NS,) arrays
        Observed per-PMT mean and variance of the first-arrival over ``n_flashes`` flashes.
    t_geo : (NS,)  known geometric first-arrival (ns).
    mu : (NS,)     known occupancy (expected detected PEs).
    n_flashes : int
        Number of flashes the moments were accumulated over (sets the noise floor).
    weights : (NS,) or None
        Per-PMT weight / lit mask (defaults: lit where mu>0).
    t0_true, tts_true : optional truth for reporting recovery.

    Returns
    -------
    dict: ``t0_hat`` (gauged), ``tts_hat``, ``t0_floor`` (per-PMT CRB ≈ TTS·√Var/√M),
    ``tts_floor`` (relative), and recovery metrics (``t0_rms``, ``t0_corr``,
    ``tts_frac_err``) when truth is supplied.
    """
    mu = np.asarray(mu, float)
    fa_mean_obs = np.asarray(fa_mean_obs, float)
    fa_var_obs = np.asarray(fa_var_obs, float)
    t_geo = np.asarray(t_geo, float)
    NS = mu.shape[0]
    W = (mu > 0).astype(float) if weights is None else np.asarray(weights, float)
    lit = W > 0
    n_lit = int(lit.sum())

    b_mean = np.asarray(_occ_bias_mean(jnp.asarray(mu)))     # E[min | N≥1]
    b_var = np.asarray(_occ_bias_var(jnp.asarray(mu)))       # Var[min | N≥1]

    # 1) TTS from the variance (t0-independent): Var[first] = TTS²·Var[min|N≥1].
    ratio = fa_var_obs[lit] / np.clip(b_var[lit], 1e-6, None)
    tts_hat = float(np.sqrt(max(np.average(ratio, weights=W[lit]), 0.0)))

    # 2) t0 from the mean with the TTS early-bias removed; gauge mean(t0)=0.
    t0_hat = fa_mean_obs - t_geo - tts_hat * b_mean
    t0_hat = t0_hat - np.average(t0_hat[lit], weights=W[lit])
    t0_hat = np.where(lit, t0_hat, 0.0)

    # Noise floors. Per-PMT t0: the M-flash mean first-arrival has std TTS·√Var/√M.
    t0_floor = tts_hat * np.sqrt(np.clip(b_var, 0, None)) / np.sqrt(n_flashes)
    # Global TTS: variance estimator over M flashes, pooled over n_lit PMTs.
    tts_floor = 1.0 / np.sqrt(2.0 * n_flashes * n_lit)

    out = dict(t0_hat=t0_hat, tts_hat=tts_hat, t0_floor=t0_floor, tts_floor=tts_floor,
               lit=lit, b_mean=b_mean, b_var=b_var, n_lit=n_lit)

    if t0_true is not None:
        t0t = np.asarray(t0_true, float)
        t0t = np.where(lit, t0t - np.average(t0t[lit], weights=W[lit]), 0.0)   # same gauge
        out['t0_rms'] = float(np.sqrt(np.average((t0_hat[lit] - t0t[lit]) ** 2, weights=W[lit])))
        out['t0_corr'] = float(np.corrcoef(t0_hat[lit], t0t[lit])[0, 1])
        out['t0_floor_median'] = float(np.median(t0_floor[lit]))
    if tts_true is not None:
        out['tts_frac_err'] = abs(tts_hat / tts_true - 1.0)

    return out
