"""Tests for the per-PMT t0 + global TTS timing calibrator (lucid.fitting.timing).

Uses a synthetic per-PMT first-arrival generator (Poisson count → min-of-N N(0,1) →
t_geo + t0 + TTS·z_min over M flashes) — the same data process as mie_hunter/timingcal.py,
no photon sim — so the estimator is tested directly and fast.
"""
import numpy as np
import jax.numpy as jnp

from lucid.fitting import calibrate_timing
from lucid.fitting.timing import _occ_bias_mean, _occ_bias_var


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
        # Measured: TTS error 8.2e-4 against a statistical floor of 7.9e-4, t0 RMS 1.18x the
        # median per-PMT floor. The bounds (5x the TTS floor + 1e-3, 1.5x the t0 floor) fail a
        # 1 % TTS scale bias, which the previous 5 % bound let through.
        assert r['tts_frac_err'] < 5.0 * r['tts_floor'] + 1e-3, (r['tts_frac_err'], r['tts_floor'])
        assert r['t0_rms'] < 1.5 * r['t0_floor_median'], (r['t0_rms'], r['t0_floor_median'])
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
        # 8x the flashes: 1/sqrt(M) predicts a ratio of sqrt(8) = 2.83; measured 3.05. A floor
        # that does not scale with M (ratio ~1) or scales as 1/M (ratio 8) fails.
        assert 2.2 < rms[0] / rms[1] < 3.6, rms
