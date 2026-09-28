"""fit_track against a transcription of its original standalone loop.

`fit_track` delegates to `lucid.fitting.gn.gauss_newton`, so comparing the two cannot fail
(`tests/test_recon_shared_loop.py`). The reference here is the original loop ("main" in the
assertion messages), transcribed statement for statement: the float64 damped solve, the refresh
cadence, the anneal, the trust clip and the non-finite refusal. It reads out with the shared
loop's convention, which differs at the edges (a Polyak window of 0 or longer than the run, a
'ming' minimum at the start).

Two differences are by design and not tested: the shared solve is float32, so trajectories agree
to the float32 floor (TOL); and the shared Levenberg base is the median of the diagonal entries
above a relative cutoff rather than of every entry floored at 1e-12, which coincides here because
no diagonal entry is near zero.

Cases: a clean fit for every readout and Polyak window (the start is never averaged; a window of
0 means no averaging); refused steps at several refresh cadences, because a cached metric must be
the kept iterate's, not the refused proposal's, and refresh=1 cannot tell the two apart; and a
learning-rate schedule in `damped_gauss_newton`, which a refusal must not rewind.
"""
import numpy as np
import pytest

from tests._recon_analytic_model import AnalyticModel, _THETA_STAR, _U_START, _scale9

TOL = 1e-4          # SCALE9 units, on the whole trajectory; the float32 solve gives ~1e-6


def _main_fit_track(model, start, *, nkeys, niters, lr, lr_final, ridge_i, lam, refresh,
                    refresh_final, refresh_switch, seed, readout, polyak_w, trust):
    """The original fit_track loop, transcribed, with the shared loop's readout; returns
    (out, traj, gnorms, n_rejected)."""
    import jax
    from lucid.fitting import SCALE9
    oc = np.zeros(12); ot = np.zeros(12)
    keys = [jax.random.PRNGKey(seed + s) for s in range(nkeys)]
    fdh = 0.4 * SCALE9
    S = SCALE9

    def G(th):
        return np.mean([np.asarray(model.grad(th, oc, ot, k)) for k in keys], 0)

    th = np.asarray(start, float); F = None
    g = G(th); traj = [th.copy()]; gnorms = [float(np.linalg.norm(g * S))]
    sw = int(refresh_switch * niters); since = 0; n_rej = 0
    for it in range(niters):
        r_it = refresh if (refresh_final is None or it < sw) else refresh_final
        if F is None or since >= r_it or (refresh_final is not None and it == sw):
            F = model.fisher_ad(th, oc, ot, keys, fdh); since = 0
        since += 1
        Fs = S[:, None] * F * S[None, :]; gs = S * g
        marq = np.diag(lam * np.diag(Fs))
        rI = ridge_i * np.median(np.clip(np.diag(Fs), 1e-12, None)) * np.eye(9)
        lr_it = lr if lr_final is None else lr + (lr_final - lr) * (it / max(1, niters - 1))
        du = -lr_it * np.linalg.solve(Fs + marq + rI + 1e-9 * np.eye(9), gs)
        if trust is not None:
            du = np.clip(du, -trust, trust)
        th_new = th + S * du; g_new = G(th_new)
        if np.isfinite(th_new).all() and np.isfinite(g_new).all():
            th, g = th_new, g_new
        else:
            n_rej += 1
        gn = float(np.linalg.norm(g * S))
        traj.append(th.copy()); gnorms.append(gn)
    if readout == 'polyak' and polyak_w:
        out = np.mean(np.array(traj[1:])[-polyak_w:], axis=0)
    elif readout == 'ming':
        out = traj[int(np.argmin(gnorms))]
    else:
        out = th
    return out, np.array(traj), np.array(gnorms), n_rej


class CliffModel:
    """A well-conditioned problem with a cliff: gradient and metric are NaN past a set energy.

    ``r = u − u*`` on the scaled offset ``u = (θ−θ*)/SCALE9``, with the target ``u*`` a small
    energy offset, so the metric is exactly ``diag(1/SCALE9²)`` and every step is predictable. The
    published lr=4 then overshoots the target past the cliff, the step is refused, and the fit
    comes back only once the anneal has shrunk the step enough -- the situation the refusal guard
    exists for. Deterministic, so a disagreement cannot be Monte-Carlo noise or chaos.
    (The analytic model is no use here: at lr=4 its exact metric makes the reference loop diverge.)
    """
    CLIFF = 1.5                     # SCALE9 units of energy above truth
    energy_from_scale = False

    def __init__(self):
        self.S = _scale9()
        self.u_star = np.zeros(9); self.u_star[0] = 0.9

    def _u(self, th):
        return (np.asarray(th, float) - _THETA_STAR) / self.S

    def grad(self, th, oc, ot, key):
        u = self._u(th)
        g = (u - self.u_star) / self.S
        return g * np.nan if u[0] > self.CLIFF else g

    def fisher_ad(self, th, oc, ot, keys, fdh):
        F = np.diag(1.0 / self.S ** 2)
        return F * np.nan if self._u(th)[0] > self.CLIFF else F


def _cliff_start():
    return _THETA_STAR.copy()


def _start():
    return _THETA_STAR + _scale9() * _U_START


def _both(model_cls, **kw):
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    from lucid.fitting import fit_track
    start = _cliff_start() if model_cls is CliffModel else _start()
    out, info = fit_track(model_cls(), np.zeros(12), np.zeros(12), start, hist=True,
                          fisher_mode='ad', verbose=False, **kw)
    ref = _main_fit_track(model_cls(), start, **kw)
    return (np.asarray(out, float), info), ref


def _assert_same(new, ref, what):
    S = _scale9()
    (out, info), (r_out, r_traj, r_gn, r_rej) = new, ref
    assert info['traj'].shape == r_traj.shape, what
    dev = np.max(np.abs(info['traj'] - r_traj) / S)
    assert dev < TOL, f'{what}: trajectory departs from main\'s loop by {dev:.2e} SCALE9 units'
    assert np.max(np.abs(out - r_out) / S) < TOL, f'{what}: readout differs from main\'s'
    assert info['n_rejected'] == r_rej, f'{what}: refused {info["n_rejected"]} vs main {r_rej}'


CLEAN = dict(nkeys=3, niters=14, lr=1.0, lr_final=0.4, ridge_i=0.1, lam=0.01,
             refresh=3, refresh_final=1, refresh_switch=0.5, seed=0)


@pytest.mark.parametrize('readout,polyak_w', [('final', 5), ('ming', 5), ('polyak', 5),
                                              ('polyak', 0), ('polyak', 40)])
@pytest.mark.parametrize('trust', [None, 0.5])
def test_a_clean_fit_matches_main(readout, polyak_w, trust):
    new, ref = _both(AnalyticModel, readout=readout, polyak_w=polyak_w, trust=trust, **CLEAN)
    _assert_same(new, ref, f'{readout}/w={polyak_w}/trust={trust}')
    assert new[1]['n_rejected'] == 0 and new[1]['rejected_steps'] == ()


def test_the_polyak_edge_cases_are_not_the_plain_window():
    """Guard on the guard: the three windows must give three different answers, or the
    parametrised cases above would pass while testing one path three times."""
    a = _both(AnalyticModel, readout='polyak', polyak_w=5, trust=None, **CLEAN)[0][0]
    b = _both(AnalyticModel, readout='polyak', polyak_w=0, trust=None, **CLEAN)[0][0]
    c = _both(AnalyticModel, readout='polyak', polyak_w=40, trust=None, **CLEAN)[0][0]
    assert not np.allclose(a, b) and not np.allclose(b, c) and not np.allclose(a, c)


@pytest.mark.parametrize('refresh', [1, 2, 4, 8])
@pytest.mark.parametrize('niters', [40, 150])
def test_refused_steps_match_main(refresh, niters):
    kw = dict(nkeys=2, niters=niters, lr=4.0, lr_final=1.5, ridge_i=0.1, lam=0.01,
              refresh=refresh, refresh_final=None, refresh_switch=0.5, seed=0,
              readout='final', polyak_w=5, trust=None)
    new, ref = _both(CliffModel, **kw)
    assert 0 < ref[3] < niters - 3, f'main refused {ref[3]}/{niters}: no refusal, or no recovery'
    _assert_same(new, ref, f'refresh={refresh} niters={niters}')
    info = new[1]
    assert len(info['rejected_steps']) == info['n_rejected']
    assert all(isinstance(i, int) for i in info['rejected_steps'])


def test_a_scheduled_learning_rate_is_not_rewound_by_a_refusal():
    """`damped_gauss_newton(learning_rate=schedule)` must anneal on the loop's iteration, as
    `gauss_newton(lr, lr_final)` does. optax's own counter is restored with the state on a refused
    step, which would spend that step's learning rate twice."""
    import jax
    from lucid.fitting import SCALE9
    from lucid.fitting.gn import gauss_newton
    from lucid.fitting.minimize import minimize
    from lucid.fitting.recon import ReconProblem
    from lucid.fitting.transforms import damped_gauss_newton, annealed_learning_rate

    def prob():
        keys = [jax.random.PRNGKey(s) for s in range(2)]
        return ReconProblem(CliffModel(), np.zeros(12), np.zeros(12), keys, 0.4 * SCALE9)

    n = 40
    common = dict(scale=SCALE9, refresh=2, readout='final', reject_nonfinite=True)
    a = gauss_newton(prob(), _cliff_start(), n, lam=0.01, mu=0.1, jitter=1e-9, lr=4.0, lr_final=1.5,
                     **common)
    b = minimize(prob(), _cliff_start(), n,
                 damped_gauss_newton(0.01, 0.1, jitter=1e-9,
                                     learning_rate=annealed_learning_rate(4.0, 1.5, n)),
                 **common)
    assert a['n_rejected'] > 0, 'no refusal -- this case cannot see a rewind'
    assert a['n_rejected'] == b['n_rejected']
    np.testing.assert_allclose(b['history'], a['history'], rtol=0, atol=0)
