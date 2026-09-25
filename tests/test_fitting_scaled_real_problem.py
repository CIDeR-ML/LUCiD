"""`ScaledProblem` against the REAL `ReconProblem`, not a toy.

`tests/test_fitting_scaled.py` establishes the equivalence on a problem written for the purpose.
That leaves the claim the module exists for untested where it matters: `ReconProblem` is the thing
that carries a float64 numpy iterate, and it does so because energy is raw MeV at ~1000, where
float32 spacing is 6.1e-5 and 200 steps of 1e-6 leave the value EXACTLY unchanged. Whether scaled
coordinates make float32 adequate THERE is what decides if the module is useful or merely correct.

Three claims, in order of what they would cost to be wrong:

  EQUIVALENCE  ScaledProblem(inner, S) with scale=None is the same optimisation as `inner` with
               scale=S -- on the real problem class, at float64 round-off.
  RESOLUTION   in scaled coordinates float32 tracks float64; in raw coordinates it does not.
  PLUMBING     it composes with the shared driver and with an optax transformation, since
               `gauss_newton` now delegates to `minimize` and everything goes through it.

The model is analytic. `ReconProblem` needs exactly `grad`, `fisher_ad` and the iterate policy
from it, so supplying those in closed form exercises the REAL problem wrapper -- its key handling,
its metric caching, its `accumulate` -- with no simulator, no SIREN weights and no GPU. What is
under test is the coordinate change, not the physics.
"""
import numpy as np
import pytest

from lucid.fitting import gauss_newton
from lucid.fitting.recon import ReconProblem, SCALE9
from lucid.fitting.scaled import ScaledProblem

P, N = 9, 30
# A track-like truth: energy in raw MeV is the component that makes float64 load-bearing.
TRUTH = np.array([1000., .3, -.2, .1, .5, .86, .7, .71, 2.0])
START = TRUTH + SCALE9 * 0.35
LAM, MU = 0.01, 0.1


class Analytic:
    """Nonlinear least squares wearing `ReconModel`'s interface.

    `grad` is the gradient of a scalar; `fisher_ad` builds a PSD metric separately -- which is the
    real problem's structure too, and the reason a first-order rule could ever be cheaper on
    reconstruction where it cannot be on calibration.
    """

    energy_from_scale = True

    def __init__(self, noise=0.0):
        self.A = np.random.default_rng(11).standard_normal((N, P)) / SCALE9[None, :]
        self.noise = noise

    def _r_J(self, theta, key):
        d = np.asarray(theta, float) - TRUTH
        Ad = self.A @ d
        r = Ad + 0.08 * Ad ** 2
        if self.noise:
            r = r + self.noise * np.random.default_rng(int(key) % 2 ** 31).standard_normal(N)
        return r, self.A + 0.16 * (Ad[:, None] * self.A)

    def grad(self, theta, oc, ot, key):
        r, J = self._r_J(theta, key)
        return J.T @ r

    def fisher_ad(self, theta, oc, ot, keys, fdh=None):
        _, J = self._r_J(theta, keys[0])
        return J.T @ J


KEYS = [0, 1, 2]
KW = dict(lam=LAM, mu=MU, max_step=3.0, lr=1.0, refresh=2, readout='final')


def _raw(dtype=np.float64, steps=30, noise=0.0):
    """The status quo: raw iterate, driver-side scale=SCALE9."""
    p = ReconProblem(Analytic(noise), None, None, KEYS, None)
    p.accumulate = lambda t, dt: np.asarray(np.asarray(t) + np.asarray(dt), dtype=dtype)
    res = gauss_newton(p, np.asarray(START, dtype=dtype), steps, scale=SCALE9, **KW)
    return np.asarray(res['theta'], float), np.asarray(res['history'], float)


def _scaled(dtype=np.float64, steps=30, noise=0.0):
    """The proposal: iterate in scaled coordinates, no driver-side scale."""
    inner = ReconProblem(Analytic(noise), None, None, KEYS, None)
    p = ScaledProblem(inner, SCALE9, origin=START.copy(), dtype=dtype)
    res = gauss_newton(p, np.asarray(p.to_scaled(START), dtype=dtype), steps, scale=None, **KW)
    hist = np.stack([p.to_raw(u) for u in np.asarray(res['history'])])
    return p.to_raw(res['theta']), hist


def test_it_is_the_same_optimisation_on_the_real_problem():
    """Trajectory, not endpoint: damping moves the path, so a converged endpoint hides errors."""
    (_, ha), (_, hb) = _raw(), _scaled()
    assert ha.shape == hb.shape
    rel = np.abs(ha - hb) / SCALE9[None, :]
    assert rel.max() < 1e-10, (
        f'scaled and raw trajectories diverge by {rel.max():.2e} in SCALE9 units on the real '
        f'ReconProblem — the wrapper does not reproduce the driver\'s scale= here')


def test_that_equivalence_can_fail():
    """Control. NON-uniform, because damped GN is genuinely invariant to a uniform rescale."""
    bad = SCALE9.copy()
    bad[0] *= 1.05
    inner = ReconProblem(Analytic(), None, None, KEYS, None)
    p = ScaledProblem(inner, bad, origin=START.copy())
    res = gauss_newton(p, p.to_scaled(START), 30, scale=None, **KW)
    hb = np.stack([p.to_raw(u) for u in np.asarray(res['history'])])
    rel = np.abs(_raw()[1] - hb) / SCALE9[None, :]
    assert rel.max() > 1e-10, 'a 5% error on one component was invisible — the gate is inert'


def test_scaled_float32_tracks_float64_where_raw_float32_does_not():
    """The claim the module exists for, on the problem that carries the float64 iterate.

    Asserted as an ORDERING plus a bound: scaled-f32 must be closer to the f64 reference than
    raw-f32 is, AND close in absolute terms. The ordering alone could be satisfied by both being
    terrible; the bound alone could be satisfied by a test too easy to separate them.
    """
    ref, _ = _raw(np.float64, steps=180)
    raw32, _ = _raw(np.float32, steps=180)
    sc32, _ = _scaled(np.float32, steps=180)
    e_raw = float((np.abs(raw32 - ref) / SCALE9).max())
    e_sc = float((np.abs(sc32 - ref) / SCALE9).max())
    assert e_sc < e_raw, (f'scaled float32 ({e_sc:.2e}) is not better than raw float32 '
                          f'({e_raw:.2e}) — the reparameterisation buys nothing here')
    assert e_sc < 1e-5, f'scaled float32 is {e_sc:.2e} from the float64 answer — not adequate'


def test_where_this_file_CANNOT_speak_for_the_float32_claim():
    """An honest boundary, asserted rather than left implicit.

    The mechanism the module exists for is energy carried at ~1000 MeV, where float32 spacing is
    6.1e-5 and a small late step vanishes. This analytic problem cannot demonstrate it, and the
    reason is worth pinning down rather than tuning around: it is NOISELESS and its minimiser is
    TRUTH[0] = 1000.0 exactly, which float32 represents exactly. Both precisions converge onto the
    same representable value, so raw-float32 loses nothing on energy however long the fit runs --
    at 30 steps and at 180.

    That was found by a vacuity check that FAILED, which is the only reason the ordering test
    above is not quietly passing on an unrelated component.

    The claim itself is measured elsewhere, on an instrument built for it:
    `tests/test_float32_is_adequate.py` site 3, where raw-float32 stalls at 1.9e-5 against
    raw-float64's 3.3e-14 and scaled-float32 is exact. What THIS file establishes is the
    equivalence and the plumbing on the real `ReconProblem`; the resolution claim it inherits.
    """
    ref, _ = _raw(np.float64, steps=180)
    raw32, _ = _raw(np.float32, steps=180)
    assert abs(raw32[0] - ref[0]) == 0.0, (
        'raw float32 now loses something on energy — this file CAN exercise the mechanism after '
        'all, so replace this boundary marker with a real assertion on it')
    assert np.float32(TRUTH[0]) == TRUTH[0], 'the truth energy is no longer float32-exact'


def test_it_runs_through_the_shared_driver_with_an_optax_transformation():
    """`gauss_newton` delegates to `minimize`, so ScaledProblem must work with a transformation."""
    from lucid.fitting.minimize import minimize
    from lucid.fitting.transforms import damped_gauss_newton
    inner = ReconProblem(Analytic(), None, None, KEYS, None)
    p = ScaledProblem(inner, SCALE9, origin=START.copy())
    res = minimize(p, p.to_scaled(START), 20, damped_gauss_newton(LAM, MU, learning_rate=1.0),
                   max_step=3.0, refresh=2, readout='final')
    out = p.to_raw(res['theta'])
    assert np.isfinite(out).all()
    assert (np.abs(out - TRUTH) / SCALE9).max() < (np.abs(START - TRUTH) / SCALE9).max(), \
        'the fit did not move toward truth'


@pytest.mark.parametrize('noise', [0.0, 1e-3])
def test_it_holds_with_a_redrawn_forward(noise):
    """Both problems here wander on a noise floor; the equivalence must not depend on being clean."""
    (_, ha), (_, hb) = _raw(noise=noise), _scaled(noise=noise)
    rel = np.abs(ha - hb) / SCALE9[None, :]
    assert rel.max() < 1e-10, f'equivalence broke at noise={noise}: {rel.max():.2e}'
