"""`minimize` against `gauss_newton` on the REAL CalibrationProblem, not a toy.

The toy in `test_fitting_minimize.py` hands back float64 numpy; `CalibrationProblem` returns `jnp`
arrays, keeps a float32 iterate, profiles the per-PMT gains inside `grad_metric_loss`, and draws
its forward at a key that depends on the STEP INDEX. Each is a place the drivers could differ
while the toy agrees:

* a jax float32 array reaching `accumulate` downcasts and retypes a numpy iterate absorbingly;
* a float32 iterate, whose solve floor is not obviously negligible;
* `forward_key(step)`: different step indices fit different photon draws, which no comparison of
  the ANSWER would explain.

The forward is an analytic stub so this runs on CPU; everything between forward and iterate is real.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.fitting import gauss_newton, FieldParams
from lucid.fitting.calib import CalibrationForward, CalibrationJacobian, CalibrationProblem
from lucid.fitting.minimize import minimize
from lucid.fitting.transforms import damped_gauss_newton

from tests import _calib_toy as toy
from tests._calib_toy import FIELDS

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
    data = CalibrationForward(_sim, _sources(), params, NS)(th, 5000, gains)
    fwd = CalibrationForward(_sim, _sources(), params, NS)
    jac = CalibrationJacobian(_sim, _sources(), params, NS, key0=9_000_000)
    prob = CalibrationProblem(fwd, jac, params, data, 1e-6,
                              n_forward_draws=1, jacobian_draws=1)
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


def test_the_float32_iterate_survives_the_new_driver():
    """CalibrationProblem.accumulate re-wraps with jnp.asarray; it must stay float32, not promote."""
    seen = []
    pb, sb, _ = _problem()
    inner = pb.accumulate

    def watch(theta, dtheta):
        out = inner(theta, dtheta)
        seen.append((jnp.asarray(out).dtype, type(dtheta).__module__))
        return out
    pb.accumulate = watch
    minimize(pb, sb, 6, damped_gauss_newton(LAM, MU, learning_rate=LR), **KW)
    assert seen, 'accumulate never ran'
    for dt, mod in seen:
        assert dt == jnp.float32, f'the float32 calibration iterate became {dt}'
        assert not mod.startswith('jax'), f'a {mod} array reached accumulate'


def test_a_first_order_rule_also_runs_on_the_real_problem():
    """Not that it is a good idea on calibration — only that the plumbing holds."""
    import optax
    pb, sb, _ = _problem()
    res = minimize(pb, sb, 20, optax.sgd(1e-3), needs_metric=False, **KW)
    assert np.isfinite(np.asarray(res['history'], float)).all()


# --------------------------------------------------------- the public entry point, calibrate()

def _calibrate_kw():
    params = FieldParams(_dp(), FIELDS, n_sensors=NS)
    th = jnp.asarray(params.theta_from_physical(), dtype=jnp.float32)
    data = CalibrationForward(_sim, _sources(), params, NS)(th, 5000, jnp.ones(NS))
    start = np.asarray(params.theta_from_physical(), dtype=np.float32) + 0.25
    return dict(sim=_sim, sources=_sources(), params=params, data=data, theta0=start)


def test_calibrate_reaches_the_same_answer_through_an_optax_transformation():
    """`tx=` must route the SAME problem, not a lookalike, through the same estimator."""
    from lucid.fitting import calibrate
    kw = _calibrate_kw()
    common = dict(steps=12, max_step=0.5, refresh=2, readout='final',
                  n_forward_draws=1, jacobian_draws=1, q_floor=1e-6)
    a = calibrate(kw['sim'], kw['sources'], kw['params'], kw['data'], kw['theta0'],
                  lam=LAM, mu=MU, lr=LR, **common)
    b = calibrate(kw['sim'], kw['sources'], kw['params'], kw['data'], kw['theta0'],
                  tx=damped_gauss_newton(LAM, MU, learning_rate=LR), **common)
    d = np.abs(np.asarray(a['theta'], float) - np.asarray(b['theta'], float)).max()
    assert d < 1e-4, f'calibrate(tx=...) diverged from the default path by {d:.2e}'


def test_calibrate_refuses_step_knobs_alongside_a_transformation():
    """Silently ignoring lam next to an explicit tx is the `_RETIRED` failure mode."""
    from lucid.fitting import calibrate
    kw = _calibrate_kw()
    with pytest.raises(TypeError, match='lam'):
        calibrate(kw['sim'], kw['sources'], kw['params'], kw['data'], kw['theta0'],
                  steps=4, lam=0.5, tx=damped_gauss_newton(LAM, MU))
