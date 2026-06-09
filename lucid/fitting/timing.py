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

import numpy as np
import jax.numpy as jnp

from lucid.simulation.sensor_response import _occ_bias_mean, _occ_bias_var


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
    if n_lit == 0:
        return dict(t0_hat=np.zeros(NS), tts_hat=float('nan'), t0_floor=np.zeros(NS),
                    tts_floor=float('inf'), lit=lit, n_lit=0,
                    b_mean=np.zeros(NS), b_var=np.zeros(NS))

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
    tts_floor = 1.0 / np.sqrt(2.0 * n_flashes * max(n_lit, 1)) if n_lit else np.inf

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
