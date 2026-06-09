"""Tests for the per-PMT t0 + global TTS timing calibrator (lucid.fitting.timing).

Uses a synthetic per-PMT first-arrival generator (Poisson count → min-of-N N(0,1) →
t_geo + t0 + TTS·z_min over M flashes) — the same data process as mie_hunter/timingcal.py,
no photon sim — so the estimator is tested directly and fast.
"""
import numpy as np
import jax.numpy as jnp

from lucid.fitting import calibrate_timing
from lucid.simulation.sensor_response import _occ_bias_mean, _occ_bias_var


def _synthetic_moments(mu, t_geo, t0_true, tts_true, n_flashes, seed=0):
    """Generate per-PMT (mean, var) of the first-arrival over n_flashes flashes."""
    rng = np.random.default_rng(seed)
    NS = len(mu)
    means = np.zeros(NS); vars = np.zeros(NS); lit = np.zeros(NS, bool)
    for p in range(NS):
        N = rng.poisson(mu[p], size=n_flashes)
        N = N[N >= 1]                                  # hit only when ≥1 PE
        if len(N) < 2:
            continue
        zmin = np.array([rng.standard_normal(n).min() for n in N])
        fa = t_geo[p] + t0_true[p] + tts_true * zmin
        means[p] = fa.mean(); vars[p] = fa.var(); lit[p] = True
    return means, vars, lit


def _setup(NS=400, seed=1):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.uniform(np.log(0.5), np.log(40.0), NS))   # spread of occupancies
    t_geo = rng.uniform(20.0, 60.0, NS)
    t0_true = 3.0 * rng.standard_normal(NS); t0_true -= t0_true.mean()  # gauge mean=0
    tts_true = 2.0
    return mu, t_geo, t0_true, tts_true


class TestExactRecovery:
    def test_exact_moments_recover_truth(self):
        """Fed the EXACT model moments, the estimator recovers t0 and TTS exactly."""
        mu, t_geo, t0_true, tts = _setup()
        b_mean = np.asarray(_occ_bias_mean(jnp.asarray(mu)))
        b_var = np.asarray(_occ_bias_var(jnp.asarray(mu)))
        fa_mean = t_geo + t0_true + tts * b_mean
        fa_var = tts ** 2 * b_var
        r = calibrate_timing(fa_mean, fa_var, t_geo, mu, n_flashes=10_000,
                             t0_true=t0_true, tts_true=tts)
        assert r['t0_rms'] < 1e-6
        assert r['tts_frac_err'] < 1e-6


class TestFlashNoiseRecovery:
    def test_recovers_within_noise_floor(self):
        """With real M-flash sampling, t0 RMS sits near the floor and TTS within a few %."""
        mu, t_geo, t0_true, tts = _setup()
        M = 2000
        fa_mean, fa_var, lit = _synthetic_moments(mu, t_geo, t0_true, tts, M, seed=3)
        r = calibrate_timing(fa_mean, fa_var, t_geo, mu, n_flashes=M,
                             weights=lit.astype(float), t0_true=t0_true, tts_true=tts)
        # TTS recovered to a few percent
        assert r['tts_frac_err'] < 0.05, r['tts_frac_err']
        # t0 RMS recovery is comparable to the per-PMT noise floor (within ~2×)
        assert r['t0_rms'] < 2.0 * r['t0_floor_median'] + 0.05
        # and t0 is well-correlated with truth
        assert r['t0_corr'] > 0.95

    def test_more_flashes_tightens_t0(self):
        """t0 RMS should fall ~1/√M with more flashes."""
        mu, t_geo, t0_true, tts = _setup(NS=300, seed=5)
        rms = []
        for M in (500, 4000):
            fa_mean, fa_var, lit = _synthetic_moments(mu, t_geo, t0_true, tts, M, seed=7)
            r = calibrate_timing(fa_mean, fa_var, t_geo, mu, n_flashes=M,
                                 weights=lit.astype(float), t0_true=t0_true)
            rms.append(r['t0_rms'])
        assert rms[1] < rms[0]            # more flashes → smaller t0 RMS
