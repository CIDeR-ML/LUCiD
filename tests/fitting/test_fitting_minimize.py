"""`minimize` must BE `gauss_newton` when handed the Gauss-Newton transformation.

If that holds, the optimizer is an argument to the driver and there is one loop rather than two
to keep in step. If it does not, this is a second optimizer wearing the same name.

The comparison is on the whole TRAJECTORY, not the endpoint. Damping moves the path and not the
minimiser, so a converged endpoint comparison passes even when the step rule is wrong.

Beyond equivalence, this file gates the hazard the driver is written around: optimizer-state
contamination on a rejected step.
"""
import numpy as np
import optax
import pytest

from lucid.fitting import gauss_newton
from lucid.fitting.minimize import minimize
from lucid.fitting.transforms import damped_gauss_newton

LAM, MU, LR = 0.01, 0.1, 4.0
P, N = 9, 30
SCALE = np.array([50., .2, .2, .2, .02, .02, .02, .02, .2])
TRUTH = np.array([1000., .3, -.2, .1, .5, .86, .7, .71, 2.0])
# The float32 solve moves the step by ~2.8e-07 relative at the published conditioning
# (tests/fitting/test_float32_is_adequate.py). 1e-4 on an ACCUMULATED 40-step trajectory allows that
# to compound; `test_the_equivalence_can_fail` holds it honest.
TOL = 1e-4


class Quad:
    """Deterministic nonlinear least squares. Deterministic so the two drivers are comparable."""

    def __init__(self):
        self.A = np.random.default_rng(7).standard_normal((N, P)) / SCALE[None, :]
        self.seen = []                       # (step, refresh) as the driver requested them

    def grad_metric_loss(self, theta, step, refresh=True):
        self.seen.append((step, bool(refresh)))
        d = np.asarray(theta, float) - TRUTH
        Ad = self.A @ d
        r = Ad + 0.1 * Ad ** 2
        J = self.A + 2 * 0.1 * (Ad[:, None] * self.A)
        return J.T @ r, J.T @ J, float(r @ r)


START = TRUTH + SCALE * 0.4
COMMON = dict(scale=SCALE, max_step=3.0, refresh=4, readout='final')


def _gn(steps=40):
    return gauss_newton(Quad(), START.copy(), steps, lam=LAM, mu=MU, lr=LR, **COMMON)


def _mz(steps=40, tx=None, **kw):
    return minimize(Quad(), START.copy(), steps,
                    tx if tx is not None else damped_gauss_newton(LAM, MU, learning_rate=LR),
                    **{**COMMON, **kw})


def test_it_reproduces_the_numpy_loops_whole_trajectory():
    a, b = _gn(), _mz()
    assert a['history'].shape == b['history'].shape
    rel = np.abs(a['history'] - b['history']) / SCALE[None, :]
    assert rel.max() < TOL, (
        f'minimize+GN diverges from gauss_newton by {rel.max():.2e} in SCALE units — it is a '
        f'different optimizer, not the same one behind an argument')


def test_the_equivalence_can_fail():
    """Control. A 10% damping error is a real convention change and must be rejected."""
    a = _gn()
    b = _mz(tx=damped_gauss_newton(LAM * 1.1, MU, learning_rate=LR))
    rel = np.abs(a['history'] - b['history']) / SCALE[None, :]
    assert rel.max() > TOL, 'a 10% lam error slipped under the tolerance — the gate is inert'


def test_the_metric_refresh_schedule_is_requested_identically():
    """The step INDEX is part of calibration's random-number stream (`forward_key(step)`).

    Any restructuring that changes which index each gradient is evaluated at moves every
    calibration number, and no gate on the reconstruction side can see it -- `fit_track`'s pin
    uses a model that ignores `step` entirely. So the request sequence is compared directly.
    """
    pa, pb = Quad(), Quad()
    gauss_newton(pa, START.copy(), 12, lam=LAM, mu=MU, lr=LR, **COMMON)
    minimize(pb, START.copy(), 12, damped_gauss_newton(LAM, MU, learning_rate=LR), **COMMON)
    assert pa.seen == pb.seen, (
        f'grad_metric_loss was called on a different (step, refresh) sequence\n'
        f'  gauss_newton: {pa.seen}\n  minimize    : {pb.seen}')


T, F_ = True, False


@pytest.mark.parametrize('kw,expected', [
    # refresh=4: the metric is built at the start and then every 4th step (0, 4, 8). Each step k
    # evaluates its look-ahead at index k+1; the last step (11) makes no look-ahead.
    (dict(refresh=4),
     [(0, T), (1, F_), (2, F_), (3, F_), (4, T), (5, F_), (6, F_), (7, F_),
      (8, T), (9, F_), (10, F_), (11, F_)]),
    # refresh_final=1 from refresh_switch*steps = 6: a forced refresh at 6, then every step.
    (dict(refresh=4, refresh_final=1, refresh_switch=0.5),
     [(0, T), (1, F_), (2, F_), (3, F_), (4, T), (5, F_), (6, T), (7, T),
      (8, T), (9, T), (10, T), (11, T)]),
])
def test_the_refresh_schedule_is_the_written_contract(kw, expected):
    """The (step, refresh) request sequence against a hand-written one, not against another loop.

    Both drivers share `_due`, so comparing them cannot see an off-by-one in it. The contract
    written out above can.
    """
    p = Quad()
    gauss_newton(p, START.copy(), 12, lam=LAM, mu=MU, lr=LR,
                 **{**COMMON, **kw})
    assert p.seen == expected


@pytest.mark.parametrize('readout,polyak', [('final', 0), ('polyak', 10), ('ming', 0)])
def test_every_readout_matches(readout, polyak):
    a = gauss_newton(Quad(), START.copy(), 30, lam=LAM, mu=MU, lr=LR,
                     **{**COMMON, 'readout': readout, 'polyak': polyak})
    b = minimize(Quad(), START.copy(), 30, damped_gauss_newton(LAM, MU, learning_rate=LR),
                 **{**COMMON, 'readout': readout, 'polyak': polyak})
    rel = np.abs(a['theta'] - b['theta']) / SCALE
    assert rel.max() < TOL, f'readout={readout} diverged by {rel.max():.2e}'


# ------------------------------------------------------------------ first-order backends

@pytest.mark.parametrize('tx', [optax.sgd(1e-2), optax.adam(3e-2),
                                optax.chain(optax.clip(10.0), optax.sgd(1e-2))])
def test_first_order_rules_run_and_descend(tx):
    """First-order rules need no curvature: each must run with needs_metric=False and descend."""
    p = Quad()
    res = minimize(p, START.copy(), 60, tx, needs_metric=False, **COMMON)
    assert np.isfinite(res['history']).all()
    assert res['gnorm'][-1] < res['gnorm'][0], (
        f'no descent: {res["gnorm"][0]:.3e} -> {res["gnorm"][-1]:.3e}')


def test_a_rejected_step_does_not_advance_the_optimizer_state():
    """With a stateful rule the update has already folded the gradient in by rejection time.

    The contract: a refused step leaves (theta, g, optimizer state) exactly as they were, so the
    next step re-proposes the same update and the run continues as the clean run delayed by one
    step. Checked against a clean adam run on the same deterministic problem. Without the state
    restore, adam's moments absorb the kept gradient twice and the trajectory departs from the
    clean one (an earlier version of this test only checked that the history was finite, which
    the unrestored loop also passes).
    """
    class Blows(Quad):
        def __init__(self):
            super().__init__()
            self.n = 0

        def grad_metric_loss(self, theta, step, refresh=True):
            g, H, loss = super().grad_metric_loss(theta, step, refresh)
            self.n += 1
            if self.n == 4:                        # step 2's look-ahead is poisoned
                g = g * np.nan
            return g, H, loss

    kw = dict(needs_metric=False, reject_nonfinite=True, **COMMON)
    res = minimize(Blows(), START.copy(), 12, optax.adam(1e-2), **kw)
    clean = minimize(Quad(), START.copy(), 12, optax.adam(1e-2), **kw)
    assert res['rejected_steps'] == (2,)
    h, c = res['history'], clean['history']
    np.testing.assert_array_equal(h[3], h[2])                  # the refused step kept the iterate
    np.testing.assert_allclose(h[:3], c[:3], rtol=0, atol=0)
    np.testing.assert_allclose(h[4:], c[3:-1], rtol=1e-6, atol=0)   # then clean, one step late


def test_fix_freezes_exactly_and_in_both_places():
    p = Quad()
    res = minimize(p, START.copy(), 20, damped_gauss_newton(LAM, MU, learning_rate=LR),
                   fix=(0, 4), **COMMON)
    start = np.float32(START)                          # the iterate is float32
    np.testing.assert_allclose(res['history'][:, 0], start[0], rtol=0, atol=0)
    np.testing.assert_allclose(res['history'][:, 4], start[4], rtol=0, atol=0)
    assert not np.allclose(res['history'][-1, 1], START[1]), 'nothing moved — fix proves nothing'


def test_a_bad_readout_is_refused():
    with pytest.raises(ValueError, match='readout must be'):
        _mz(readout='best')


def test_the_published_anneal_is_expressible_as_an_optax_schedule():
    """There is no `lr` argument, so the published lr 4.0 -> 1.5 must live in the transformation.

    `transition_steps=steps-1` is not incidental: it matches gn.py's own denominator,
    `lr + (lr_final-lr)*step/max(1, steps-1)`. Getting it wrong by one shifts the whole anneal.

    optax computes the schedule in float32 where gn.py uses float64 numpy, so this is close but
    not bit-equal -- which is exactly why the GN path keeps its own float64 anneal and this is
    offered as the equivalent, not as a replacement for the pinned one.
    """
    N = 40
    a = gauss_newton(Quad(), START.copy(), N, lam=LAM, mu=MU, lr=4.0, lr_final=1.5, **COMMON)
    sched = optax.linear_schedule(4.0, 1.5, transition_steps=N - 1)
    b = minimize(Quad(), START.copy(), N, damped_gauss_newton(LAM, MU, learning_rate=sched),
                 **COMMON)
    rel = np.abs(a['history'] - b['history']) / SCALE[None, :]
    assert rel.max() < TOL, f'the optax schedule does not reproduce the anneal ({rel.max():.2e})'


def test_microbatched_accumulation_equals_one_full_batch_step():
    """optax.MultiSteps is the memory lever, and it is only a lever if it is EXACT.

    Accumulating k microbatch gradients must give the step one full-batch gradient would have
    given. If it merely approximates, it trades memory for an unquantified bias rather than for
    time.

    MultiSteps averages the accumulated GRADIENTS and knows nothing about extra_args, so the
    metric is whatever arrived last; here the metric is held fixed to isolate the accumulation.
    Microbatching by N_PH is invalid for this simulator for a reason unrelated to optax -- see
    lucid/fitting/transforms.py.
    """
    p = Quad()
    g, H, _ = p.grad_metric_loss(START.copy(), 0)
    Hs = SCALE[:, None] * np.asarray(H) * SCALE[None, :]
    gs = SCALE * np.asarray(g)

    tx = damped_gauss_newton(LAM, MU, learning_rate=LR)
    full, _ = tx.update(gs, tx.init(gs), None, metric=Hs)

    # DISTINCT microbatches whose mean is `gs`: identical ones would make "last metric" and
    # "mean metric" the same array and the batches noiseless, hiding the property real use needs.
    k = 4
    rng = np.random.default_rng(3)
    parts = rng.standard_normal((k, gs.size)) * np.abs(gs).mean()
    parts -= parts.mean(0) - gs                      # mean(parts) == gs exactly
    ms = optax.MultiSteps(tx, every_k_schedule=k)
    st = ms.init(gs)
    for i in range(k):
        acc, st = ms.update(parts[i], st, None, metric=Hs)

    rel = np.linalg.norm(np.asarray(acc) - np.asarray(full)) / np.linalg.norm(np.asarray(full))
    assert rel < 1e-5, (
        f'accumulated step differs from the full-batch step by {rel:.2e} — MultiSteps is not '
        f'exact here, so it cannot be used to trade memory for time')
