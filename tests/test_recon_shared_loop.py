"""fit_track through the shared Gauss-Newton loop.

⚠️ READ THIS BEFORE TRUSTING THE PARAMETRISED TEST BELOW. It was written while `fit_track` still
had its own loop, and it proved the shared one reproduced it. `fit_track` now DELEGATES to
`gauss_newton`, so `test_shared_loop_matches_fit_track` compares the shared loop against itself
and can no longer fail. It is kept only as executable documentation of the call mapping — which
`fit_track` argument becomes which loop argument — and must not be counted as coverage.

The gate on that delegation was a same-node comparison made when it landed: fit_track's final
estimate, trajectory and gradient norms before and after, bit-identical in every array. That
reference is not shipped as a test, because an exact golden does not survive a change of CPU.
`test_the_refresh_switch_is_actually_exercised` at the bottom of this file is still discriminating.

Uses the analytic model in `tests/_recon_analytic_model.py` — no simulator, no detector, no SIREN — so it
runs in milliseconds and can live in CI. The physics is gated elsewhere; what is at stake here is
the loop's arithmetic and bookkeeping: metric-refresh cadence with a mid-run switch, the lr anneal,
the trust clip, the non-finite reject, and all three readouts.
"""
import numpy as np
import pytest

from tests._recon_analytic_model import AnalyticModel, _THETA_STAR, _U_START, _scale9

NITERS = 14
LAM, RIDGE_I, LR, LR_FINAL = 0.01, 0.1, 1.0, 0.4
REFRESH, REFRESH_FINAL, REFRESH_SWITCH = 3, 1, 0.5
NKEYS, POLYAK_W = 3, 5


def _setup(trust):
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    model = AnalyticModel()
    start = _THETA_STAR + _scale9() * _U_START
    oc = np.zeros(12); ot = np.zeros(12)
    return model, start, oc, ot, trust


def _via_fit_track(readout, trust):
    from lucid.fitting import fit_track
    model, start, oc, ot, trust = _setup(trust)
    return fit_track(model, oc, ot, start, nkeys=NKEYS, niters=NITERS, lr=LR, lr_final=LR_FINAL,
                     ridge_i=RIDGE_I, lam=LAM, refresh=REFRESH, refresh_final=REFRESH_FINAL,
                     refresh_switch=REFRESH_SWITCH, seed=0, readout=readout, polyak_w=POLYAK_W,
                     trust=trust, hist=True, fisher_mode='ad', verbose=False)


def _via_shared(readout, trust):
    import jax
    from lucid.fitting.recon import ReconProblem, SCALE9
    from lucid.fitting.gn import gauss_newton
    model, start, oc, ot, trust = _setup(trust)
    keys = [jax.random.PRNGKey(0 + s) for s in range(NKEYS)]
    prob = ReconProblem(model, oc, ot, keys, 0.4 * SCALE9, fisher_mode='ad')
    return gauss_newton(prob, np.asarray(start, float), NITERS,
                        lam=LAM, mu=RIDGE_I, jitter=1e-9, lr=LR, lr_final=LR_FINAL,
                        scale=SCALE9, max_step=trust, refresh=REFRESH,
                        refresh_final=REFRESH_FINAL, refresh_switch=REFRESH_SWITCH,
                        readout=readout, polyak=POLYAK_W, reject_nonfinite=True)


@pytest.mark.parametrize('readout', ['final', 'polyak', 'ming'])
@pytest.mark.parametrize('trust', [None, 0.5])
def test_shared_loop_matches_fit_track(readout, trust):
    """Identical readout, trajectory and gradient-norm history, for every readout and clip."""
    out_ref, info = _via_fit_track(readout, trust)
    res = _via_shared(readout, trust)
    np.testing.assert_allclose(res['theta'], np.asarray(out_ref, float), rtol=0, atol=0,
                               err_msg=f'{readout}/{trust}: readout differs')
    np.testing.assert_allclose(res['history'], np.asarray(info['traj'], float), rtol=0, atol=0,
                               err_msg=f'{readout}/{trust}: trajectory differs')
    np.testing.assert_allclose(res['gnorm'], np.asarray(info['gnorm'], float), rtol=0, atol=0,
                               err_msg=f'{readout}/{trust}: gradient-norm history differs')


def test_the_refresh_switch_is_actually_exercised():
    """Guard on the guard: the mid-run cadence change must alter the result.

    If refresh_final stopped taking effect, the test above would still pass while silently
    comparing two runs of the same simpler schedule.
    """
    a = _via_shared('final', None)
    import jax
    from lucid.fitting.recon import ReconProblem, SCALE9
    from lucid.fitting.gn import gauss_newton
    model, start, oc, ot, _ = _setup(None)
    keys = [jax.random.PRNGKey(s) for s in range(NKEYS)]
    prob = ReconProblem(model, oc, ot, keys, 0.4 * SCALE9)
    b = gauss_newton(prob, np.asarray(start, float), NITERS, lam=LAM, mu=RIDGE_I, jitter=1e-9,
                     lr=LR, lr_final=LR_FINAL, scale=SCALE9, refresh=REFRESH,
                     refresh_final=None, readout='final', reject_nonfinite=True)
    assert not np.allclose(a['theta'], b['theta']), \
        'refresh_final has no effect — the cadence switch is not being exercised'
