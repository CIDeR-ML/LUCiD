"""`minimize` against `gauss_newton` on the REAL CalibrationProblem, not a toy.

The toy in `test_fitting_minimize.py` hands back float64 numpy; `CalibrationProblem` returns `jnp`
arrays, profiles the per-PMT gains inside `grad_metric_loss`, and draws its forward at a key that
depends on the STEP INDEX. Each is a place the drivers could differ while the toy agrees:

* `jnp` gradients and metrics, where the toy's are numpy;
* `forward_key(step)`: different step indices fit different photon draws, which no comparison of
  the ANSWER would explain.

The forward is an analytic stub so this runs on CPU; everything between forward and iterate is real.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lucid.fitting import gauss_newton, FieldParams
from lucid.fitting.calib import CalibrationForward, CalibrationProblem
from lucid.fitting.minimize import minimize
from lucid.fitting.transforms import damped_gauss_newton

from tests.helpers import calib_toy as toy
from tests.helpers.calib_toy import FIELDS

NS = 32
LAM, MU, LR = 0.01, 0.1, 1.0


def _dp():
    return toy.dp_true(NS)


def _sim(source, dp, key):
    """Analytic stand-in for the photon forward: smooth, positive, theta-dependent."""
    return toy.charge(source, dp), jnp.zeros(NS)


def _sources():
    return toy.sources(NS, seed=11, b_spread=0.4, b_scale=2.1)


def _problem(record=None):
    params = FieldParams(_dp(), FIELDS, n_sensors=NS)
    th = jnp.asarray(params.theta_from_physical(), dtype=jnp.float32)
    gains = jnp.ones(NS)
    data = CalibrationForward(_sim, _sources(), params, NS)(th, jax.random.PRNGKey(0), gains)
    fwd = CalibrationForward(_sim, _sources(), params, NS)
    prob = CalibrationProblem(fwd, data, q_floor_frac=1e-6, n_forward_draws=1, jacobian_draws=1)
    if record is not None:
        inner = prob.grad_metric_loss

        def spy(theta, step, refresh=True):
            record.append((step, bool(refresh)))
            return inner(theta, step, refresh)
        prob.grad_metric_loss = spy
    start = np.asarray(params.theta_from_physical(), dtype=np.float32) + 0.25
    return prob, jnp.asarray(start, dtype=jnp.float32), th


KW = dict(max_step=0.5, refresh=2, readout='final')


def test_the_two_drivers_agree_on_the_real_calibration_problem():
    pa, sa, _ = _problem()
    pb, sb, _ = _problem()
    a = gauss_newton(pa, sa, 12, lam=LAM, mu=MU, lr=LR, **KW)
    b = minimize(pb, sb, 12, damped_gauss_newton(LAM, MU, learning_rate=LR), **KW)
    ha, hb = np.asarray(a['history'], float), np.asarray(b['history'], float)
    assert ha.shape == hb.shape
    rel = np.abs(ha - hb).max()
    # theta is log-space here, so an absolute difference IS a relative one on the physical value.
    assert rel < 1e-4, (
        f'the drivers diverge by {rel:.2e} in log-parameter units on the real problem — the toy '
        f'agreement did not survive jnp arrays and a float32 iterate')


def test_the_real_problem_is_asked_for_the_same_steps():
    """`forward_key(step)` means a different index is a different photon draw."""
    ra, rb = [], []
    pa, sa, _ = _problem(record=ra)
    pb, sb, _ = _problem(record=rb)
    gauss_newton(pa, sa, 10, lam=LAM, mu=MU, lr=LR, **KW)
    minimize(pb, sb, 10, damped_gauss_newton(LAM, MU, learning_rate=LR), **KW)
    assert ra == rb, f'different (step, refresh) sequences\n  gn: {ra}\n  mz: {rb}'


def test_a_first_order_rule_takes_exactly_the_sgd_steps_on_the_real_problem():
    """With ``optax.sgd(lr)`` the trajectory is theta_{k+1} = theta_k - clip(lr g_k, max_step),
    with g_k the problem's gradient at theta_k, and the loss falls.

    The reference recurrence is written out here against a fresh problem, with the driver's
    metric cadence (``refresh=2``: the Jacobian, and so the gradient J^T r, is rebuilt at even
    steps). ``gauss_newton`` is ``minimize`` configured, so the two-driver agreement tests above
    cannot see a fault in the shared loop; this reference can. A driver that applied the step
    with the wrong sign or scale, skipped the clip, or took the gradient at another iterate
    departs from it. Not that SGD is a good idea on calibration: the point is that a
    first-order rule goes through the same plumbing.
    """
    import optax
    lr, steps = 3e-3, 20
    pb, sb, _ = _problem()
    res = minimize(pb, sb, steps, optax.sgd(lr), needs_metric=False, **KW)

    pr, theta, _ = _problem()
    want = [np.asarray(theta, np.float64)]
    for k in range(steps):
        g = np.asarray(pr.grad_metric_loss(theta, k, refresh=(k % 2 == 0))[0], np.float64)
        du = np.clip(-lr * g, -KW['max_step'], KW['max_step'])
        theta = theta + jnp.asarray(du, jnp.float32)
        want.append(np.asarray(theta, np.float64))
    hist = np.asarray(res['history'], np.float64)
    assert hist.shape == (steps + 1, len(want[0]))
    np.testing.assert_allclose(hist, np.stack(want), rtol=0, atol=2e-6)
    assert np.abs(hist[-1] - hist[0]).max() > 1e-3, 'the rule must actually move theta'

    loss = np.asarray(res['loss'], float)
    assert loss[-1] < loss[0], f'20 SGD steps should reduce the loss: {loss[0]:.6g} -> {loss[-1]:.6g}'


# --------------------------------------------------------- the public entry point, calibrate()

def test_calibrate_is_minimize_on_the_default_chain():
    """``calibrate(problem, theta0)`` must take exactly the steps ``minimize`` takes with
    ``gauss_newton_tx(0.01, 0.1)`` on the same problem: the default is that chain, not a lookalike."""
    from lucid.fitting import calibrate, gauss_newton_tx
    pa, sa, _ = _problem()
    pb, sb, _ = _problem()
    a = calibrate(pa, sa, steps=12, refresh=2, max_step=0.5, polyak=0)
    b = minimize(pb, sb, 12, gauss_newton_tx(LAM, MU), max_step=0.5, refresh=2, readout='final')
    np.testing.assert_array_equal(np.asarray(a['history']), np.asarray(b['history']))


def test_calibrate_runs_a_first_order_rule_on_the_same_problem():
    """``tx=optax.adam(...)`` plugs in without a flag: the rule ignores the metric it is handed.
    The reference is the plain optax loop over ``grad_metric_loss`` on a fresh problem, with the
    driver's refresh cadence and step clip."""
    import optax
    from lucid.fitting import calibrate
    lr, steps = 1e-2, 15
    pa, sa, _ = _problem()
    res = calibrate(pa, sa, tx=optax.adam(lr), steps=steps, refresh=2, max_step=0.5, polyak=0)

    pr, theta, _ = _problem()
    opt = optax.adam(lr)
    state = opt.init(np.asarray(theta))
    for k in range(steps):
        g = np.asarray(pr.grad_metric_loss(theta, k, refresh=(k % 2 == 0))[0])
        du, state = opt.update(g, state, np.asarray(theta))
        theta = theta + jnp.asarray(np.clip(np.asarray(du), -0.5, 0.5), jnp.float32)
    np.testing.assert_allclose(np.asarray(res['history'][-1]), np.asarray(theta),
                               rtol=0, atol=1e-6)
    assert res['loss'][-1] < res['loss'][0]
