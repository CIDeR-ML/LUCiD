"""Joint charge + first-arrival-time GN (gap 3) — mechanism test on a toy forward.

Verifies the timing-as-a-residual extension: the two independent per-PMT Schur blocks
(k multiplicative on charge, t0 additive on time) and a global block (incl a tts-like param)
fed by BOTH residuals are recovered jointly. Uses a cheap analytic forward; two sources
break the per-PMT charge degeneracy.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from lucid.fitting import fit_charge_time, ChargeTimeModel

NS = 16
_GEO = {'a': np.linspace(10., 200., NS), 'b': np.linspace(200., 10., NS)}
_OCC = np.linspace(0.5, 2.0, NS)            # per-sensor occupancy-like factor for the tts term


def _forward_factory(src):
    geom = jnp.asarray(_GEO[src]); occ = jnp.asarray(_OCC)

    def forward(theta, ek, pk):
        # theta = [log L_R, log qe_absn_amp, log tts]; k=1, t0=0 baseline
        s = jnp.exp(theta[0]); amp = jnp.exp(theta[1]); tts = jnp.exp(theta[2])
        charge = amp * (s / (s + geom))                       # shape carries L_R
        time = 0.05 * geom + tts * occ                        # tts enters the time
        return charge, time
    return forward


def test_joint_charge_time_recovers_globals_k_t0():
    rng = np.random.default_rng(0)
    th_true = np.log(np.array([70.0, 4.2, 2.0]))              # L_R, amp(=qe·absn), tts
    k_true = np.exp(0.15 * rng.standard_normal(NS)); k_true /= np.exp(np.mean(np.log(k_true)))
    t0_true = rng.standard_normal(NS); t0_true -= t0_true.mean()

    srcs = [ChargeTimeModel(_forward_factory(s)) for s in ['a', 'b']]
    # truth observables: charge = k·c, time = T + t0
    truth_c, truth_t = [], []
    for m in srcs:
        c, t = m.forward(th_true, None, None)
        truth_c.append(np.asarray(c) * k_true)
        truth_t.append(np.asarray(t) + t0_true)

    start = th_true + np.array([0.15, -0.15, 0.1])            # perturb globals
    res = fit_charge_time(srcs, truth_c, truth_t, start, NS,
                          steps=400, refresh=10, w_time=1.0, step_max=0.1,
                          kstep_max=0.3, t0step_max=1.0, nb_h=1)

    # globals recovered (incl tts, which lives in the TIME residual)
    np.testing.assert_allclose(res['theta'], np.exp(th_true), rtol=0.05)
    # per-PMT k (charge Schur) and t0 (time Schur) recovered
    assert np.corrcoef(res['k'], k_true)[0, 1] > 0.98
    np.testing.assert_allclose(res['k'], k_true, atol=0.03)
    assert np.corrcoef(res['t0'], t0_true)[0, 1] > 0.98
    np.testing.assert_allclose(res['t0'], t0_true, atol=0.05)


def test_tts_needs_the_time_residual():
    """tts is invisible to charge — with w_time=0 it must NOT be recovered, with w_time>0 it is."""
    th_true = np.log(np.array([70.0, 4.2, 2.0]))
    srcs = [ChargeTimeModel(_forward_factory(s)) for s in ['a', 'b']]
    truth_c, truth_t = [], []
    for m in srcs:
        c, t = m.forward(th_true, None, None)
        truth_c.append(np.asarray(c)); truth_t.append(np.asarray(t))
    start = th_true + np.array([0.1, -0.1, 0.3])              # tts off by exp(0.3)

    off = fit_charge_time(srcs, truth_c, truth_t, start, NS, steps=200, refresh=10,
                          w_time=0.0, nb_h=1)
    on = fit_charge_time(srcs, truth_c, truth_t, start, NS, steps=200, refresh=10,
                         w_time=1.0, nb_h=1)
    tts_off, tts_on = off['theta'][2], on['theta'][2]
    # charge-only leaves tts ~at its (wrong) start; the time residual pulls it to truth 2.0
    assert abs(tts_on - 2.0) < abs(tts_off - 2.0)
    assert tts_on == pytest.approx(2.0, rel=0.05)
