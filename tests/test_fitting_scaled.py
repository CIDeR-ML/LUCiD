"""`ScaledProblem` must be the SAME optimisation, not merely a similar one.

The claim it rests on is exact: moving the driver's coordinate preconditioner into the
parameterisation changes where a multiply happens, not what is computed. So

    ScaledProblem(inner, S)  driven with scale=None
    inner                    driven with scale=S

must agree to float64 round-off on a deterministic problem. That is a far stronger statement than
"the answers are statistically compatible", and it is what makes the reparameterisation safe to
adopt before anything moves to JAX -- it is the last change the existing float64 machinery can
police to round-off.

The payoff is separate and is measured elsewhere (`tests/test_float32_is_adequate.py`): once
the iterate is O(1), float32 carries it as well as float64 carries the raw one. Here the float32
arm is asserted only to be no worse than raw float32, because a toy cannot settle production.
"""
import numpy as np
import pytest

from lucid.fitting import gauss_newton
from lucid.fitting.scaled import ScaledProblem

SCALE = np.array([50., .2, .2, .2, .02, .02, .02, .02, .2])
LAM, MU = 0.01, 0.1
TRUTH = np.array([1000., .3, -.2, .1, .5, .86, .7, .71, 2.0])


class Raw:
    """Deterministic nonlinear least squares in RAW physical coordinates.

    Deterministic on purpose: with noise the two arms would only be statistically comparable and
    the round-off claim could not be tested at all.
    """

    def __init__(self, dtype=np.float64):
        self.A = np.random.default_rng(7).standard_normal((40, 9)) / SCALE[None, :]
        self.dtype = dtype

    def _res_jac(self, theta):
        d = np.asarray(theta, float) - TRUTH
        r = self.A @ d + 0.1 * (self.A @ d) ** 2
        J = self.A + 2 * 0.1 * ((self.A @ d)[:, None] * self.A)
        return r, J

    def grad_metric_loss(self, theta, step, refresh=True):
        r, J = self._res_jac(theta)
        return J.T @ r, J.T @ J, float(r @ r)

    def accumulate(self, theta, dtheta):
        return np.asarray(np.asarray(theta) + np.asarray(dtheta), dtype=self.dtype)


START = TRUTH + SCALE * 0.4
KW = dict(lam=LAM, mu=MU, max_step=3.0, lr=1.0, refresh=1, readout='final')


# Compare TRAJECTORIES, not endpoints. Damping changes the path, not the fixed point, so after
# enough steps both arms land on the same minimiser and an endpoint comparison passes even when
# the scaling is wrong -- measured: a deliberate 5% scale error gave a difference of EXACTLY zero.
# The trajectory is what the reparameterisation actually has to reproduce.
def _raw_arm(steps=40, scale=SCALE):
    res = gauss_newton(Raw(), START.copy(), steps, scale=scale, **KW)
    return np.asarray(res['theta']), np.asarray(res['history'])


def _scaled_arm(steps=40, dtype=None, origin=None, scale=SCALE):
    org = START.copy() if origin is None else origin
    p = ScaledProblem(Raw(), scale, origin=org, dtype=dtype)
    u0 = p.to_scaled(START)
    res = gauss_newton(p, np.asarray(u0, dtype=dtype or np.float64), steps, scale=None, **KW)
    hist = np.stack([p.to_raw(u) for u in np.asarray(res['history'])])
    return p.to_raw(res['theta']), hist


def test_scaled_coordinates_are_the_same_optimisation():
    """The core claim, on the whole TRAJECTORY, at float64 round-off."""
    (_, ha), (_, hb) = _raw_arm(), _scaled_arm()
    assert ha.shape == hb.shape
    rel = np.abs(ha - hb) / SCALE[None, :]
    assert rel.max() < 1e-10, (
        f'scaled and raw trajectories diverge by {rel.max():.2e} in SCALE units — the chain rule '
        f'in ScaledProblem does not reproduce the driver\'s scale=')


def test_that_equivalence_test_can_fail():
    """Control: a WRONG scale must be rejected, or the test above proves nothing.

    Two near-misses are folded into this control, and both had to be measured rather than
    reasoned about:

    * On the ENDPOINT a scale error is invisible (measured: exactly 0.0). Damping moves the path,
      not the minimiser, and 40 steps is enough for both arms to arrive. Hence the trajectory.
    * A UNIFORM rescale S -> cS is invisible even on the trajectory, because damped Gauss-Newton
      is genuinely invariant to it: in `D A^-1 D` the c^2 from `median(diag S'HS')` cancels the
      c^-2 from `D^-2`, leaving `[M + mu·b·diag(1/S^2)]^-1 g`. So the perturbation must be
      NON-UNIFORM to be an error at all.
    """
    bad = SCALE.copy()
    bad[0] *= 1.05                      # one component only -- see above
    (_, ha) = _raw_arm()
    (_, hb) = _scaled_arm(scale=bad)
    rel = np.abs(ha - hb) / SCALE[None, :]
    assert rel.max() > 1e-10, 'a 5% scale error was invisible — the equivalence gate is inert'


def test_the_origin_puts_the_iterate_at_zero():
    """Starting at exactly 0 is the representation float32 handles best."""
    p = ScaledProblem(Raw(), SCALE, origin=START.copy())
    np.testing.assert_array_equal(p.to_scaled(START), np.zeros(9))
    np.testing.assert_allclose(p.to_raw(np.zeros(9)), START, rtol=0, atol=0)


def test_the_round_trip_is_exact_at_the_origin():
    p = ScaledProblem(Raw(), SCALE, origin=START.copy())
    for v in (START, TRUTH, START + SCALE):
        np.testing.assert_allclose(p.to_raw(p.to_scaled(v)), v, rtol=1e-12, atol=0)


def test_it_does_not_call_the_inner_accumulate():
    """Delegating would double-apply ProjectedReconProblem's own `S *` in `accumulate`."""
    class Trap(Raw):
        def accumulate(self, theta, dtheta):        # pragma: no cover - must not run
            raise AssertionError('inner.accumulate must not be called by ScaledProblem')

    p = ScaledProblem(Trap(), SCALE, origin=START.copy())
    gauss_newton(p, p.to_scaled(START), 3, scale=None, **KW)


def test_a_zero_scale_is_refused():
    bad = SCALE.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError, match='could never move'):
        ScaledProblem(Raw(), bad)


def test_float32_in_scaled_coordinates_beats_float32_in_raw():
    """The point of the exercise, on the toy: raw-f32 loses resolution, scaled-f32 does not.

    Asserted as an ORDERING, not a threshold. The production claim belongs to
    tests/test_float32_is_adequate.py and ultimately to the statistical gate; a toy can only
    show the mechanism is real and pointed the right way.
    """
    ref, _ = _raw_arm()
    raw32 = gauss_newton(Raw(np.float32), np.asarray(START, np.float32), 40,
                         scale=SCALE, **KW)['theta']
    sc32, _ = _scaled_arm(dtype=np.float32)
    e_raw = (np.abs(np.asarray(raw32, float) - ref) / SCALE).max()
    e_sc = (np.abs(sc32 - ref) / SCALE).max()
    assert e_sc < e_raw, (
        f'scaled float32 ({e_sc:.2e}) is not better than raw float32 ({e_raw:.2e}) — the '
        f'reparameterisation is not buying the resolution it exists for')
