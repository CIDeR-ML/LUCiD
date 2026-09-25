"""The shared Gauss-Newton loop, driven the way fit_track drives it.

fit_track itself is compared against its original loop in `tests/test_fit_track_matches_main_loop.py`.
This file keeps the one check that needs the loop called directly: that the mid-run switch of the
metric refresh cadence actually changes the result. Uses the analytic model in
`tests/_recon_analytic_model.py`, so it needs no simulator, detector or SIREN weights.
"""
import numpy as np

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


def test_the_refresh_switch_is_actually_exercised():
    """Guard on the guard: the mid-run cadence change must alter the result.

    If refresh_final stopped taking effect, a comparison of two runs would silently compare the
    same simpler schedule twice.
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
