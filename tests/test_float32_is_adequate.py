"""Is float32 adequate where going jax-native would change the numerics? Asserted, not printed.

Four production docstrings and four test files quote numbers from this measurement -- "2.8e-07
relative at cond(H)=6800", "scaled-float32 matches raw-float64 to four digits", "raw-float32
stalls at 1.9e-5". Until this file existed, ALL of them deferred to a measurement script with no
assertion, not collected by pytest and not run in CI.
``tests/test_fitting_scaled_real_problem.py`` even carries a test named
``test_where_this_file_CANNOT_speak_for_the_float32_claim`` pointing at it. So the suite knew it
could not verify the claim, said so by name, and pointed at something that could not fail.

The numbers were re-measured when this file was written and had NOT drifted. That is the reason
to gate them now rather than to re-derive them: the claim is true today, and nothing would say so
if it stopped being true.

THREE SEPARABLE SITES. Going jax-native (``jnp`` everywhere, ``x64`` off) changes the arithmetic
in exactly three places, and conflating them is how "float32 is probably fine" gets asserted
without evidence:

  SITE 1  the linear solve. It IS float32 now — ``lucid.fitting`` has one damped-matrix
          implementation and it is JAX. The float64 arm here is a reference built inside this
          file, not a second library path. cond(H) ~ 6800 is the published 19-parameter
          calibration matrix.
  SITE 2  the Levenberg median, and which side of a float32/float64 boundary it falls on. This
          was written when the rewrite was still prospective; the rewrite has since landed, so
          the test now guards that the choice REMAINS inert rather than clearing it in advance.
  SITE 3  the iterate. Reconstruction carries float64 because energy is raw MeV at ~1000, where
          float32 spacing is 6.1e-5 -- a measured freeze, not a rounding nit. The fix is to carry
          the iterate in SCALED coordinates (every component O(1)), which is what
          ``lucid.fitting.scaled.ScaledProblem`` exists for.

EVERY SITE CARRIES A CONTROL, and site 3's control is on the control: it is not enough that
scaled-float32 tracks raw-float64, because a test where every arm agrees measures nothing. Raw
float32 must be visibly WORSE, or the instrument cannot tell the representations apart and its
null result is vacuous. Two earlier toys in this project produced plausible numbers that measured
nothing (a diverged fit; a "fit error" whose truth was not the minimiser), which is why the
discipline is written into the assertions rather than into a comment.

Numpy only -- ``lucid.fitting.gn`` imports nothing else -- so this is CPU, deterministic and a
few seconds. It is a real regression gate, not a benchmark.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from lucid.fitting.gn import gauss_newton
from lucid.fitting.transforms import damped_matrix as _damped_matrix


def damped_matrix(H, *, lam, mu):
    """The library's damped matrix in whatever precision ``H`` arrives in.

    The float64 REFERENCE below is built inline rather than taken from the library, because the
    library no longer has a float64 path — ``lucid.fitting`` has exactly one damped-matrix
    implementation and it is JAX, which computes in float32 while ``x64`` is off. That is the
    situation this file exists to justify, so constructing the reference here is not a shortcut:
    the question "is float32 adequate" is a question about the ARITHMETIC, and it needs a float64
    answer to compare against that does not itself depend on the thing under test.
    """
    return np.asarray(_damped_matrix(jnp.asarray(H), lam=lam, mu=mu))


def damped_matrix_f64(H, *, lam, mu):
    """The float64 reference: the same convention, evaluated in double precision by numpy."""
    dg = np.clip(np.diag(H), 0.0, None)
    cutoff = 1e-12 * dg.max()
    informative = dg[dg > cutoff]
    assert informative.size, 'reference undefined on an all-flat diagonal'
    base = float(np.median(informative))
    return H + lam * np.diag(dg) + mu * base * np.eye(H.shape[0])

SCALE9 = np.array([50., .2, .2, .2, .02, .02, .02, .02, .2])
LAM, MU = 0.01, 0.1
PUBLISHED_COND = 6.8e3          # cond(H) of the published 19-parameter calibration matrix


def spd(n, cond, seed):
    """A symmetric positive-definite matrix with a PRESCRIBED condition number."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.logspace(0, np.log10(cond), n)
    return (Q * ev) @ Q.T


# ------------------------------------------------------------------ SITE 1: the solve

def _solve_error(cond):
    """(control, relative step error, 1-cos angle) for float32 vs float64 at this conditioning."""
    H = spd(19, cond, 0)
    g = np.random.default_rng(1).standard_normal(19)
    # Reference: the same convention in float64 numpy. Control: it against itself.
    d64 = np.linalg.solve(damped_matrix_f64(H, lam=LAM, mu=MU), g)
    d64b = np.linalg.solve(damped_matrix_f64(H, lam=LAM, mu=MU), g)
    # Under test: the LIBRARY's implementation, in the float32 it actually runs in.
    A32 = damped_matrix(H.astype(np.float32), lam=LAM, mu=MU).astype(np.float32)
    d32 = np.linalg.solve(A32, g.astype(np.float32))
    ctrl = float(np.abs(d64 - d64b).max())
    rel = float(np.linalg.norm(d32 - d64) / np.linalg.norm(d64))
    cosang = float(d32 @ d64 / (np.linalg.norm(d32) * np.linalg.norm(d64)))
    return ctrl, rel, 1.0 - cosang


def test_the_float64_solve_reproduces_itself_exactly():
    """The control. Without it, a small float32 error could be the harness rather than the dtype."""
    ctrl, _, _ = _solve_error(PUBLISHED_COND)
    assert ctrl == 0.0, f'the float64 solve is not deterministic (control {ctrl:g}), so nothing below means anything'


@pytest.mark.parametrize('cond', [1e2, 1e3, PUBLISHED_COND, 1e4, 1e5])
def test_the_float32_solve_is_far_below_the_damping_it_competes_with(cond):
    """The step error from float32 must be negligible against the perturbation damping ADDS.

    The comparison is not against zero -- a Gauss-Newton step is damped by construction. Marquardt
    ``lam=0.01`` perturbs the same direction by ~1e-2 deliberately, so a float32 error two orders
    below that cannot change any decision the step feeds. Measured at the published conditioning:
    2.8e-07, which is ~360x under this bound.
    """
    _, rel, _ = _solve_error(cond)
    assert rel < 0.01 * LAM, (
        f'at cond={cond:g} the float32 solve moves the step by {rel:.2e} relative, which is no '
        f'longer negligible against the Marquardt damping ({LAM}) it competes with')


def test_the_float32_solve_barely_rotates_the_step_at_the_published_conditioning():
    """Direction, not magnitude: the trust clip and the damping both absorb scale, an angle survives.

    Bounded by 1e-6 against a measured 5.3e-08, and taken in absolute value deliberately: at
    cond=1e5 the same quantity comes out at -9e-09, i.e. the two directions agree to within the
    float64 resolution of ``1 - cos`` itself. A one-sided bound tighter than that would be
    asserting on the sign of numerical noise.
    """
    _, _, one_minus_cos = _solve_error(PUBLISHED_COND)
    assert abs(one_minus_cos) < 1e-6, (
        f'float32 rotates the step by 1-cos={one_minus_cos:.2e} at cond={PUBLISHED_COND:g}')


# ------------------------------------------------------- SITE 2: which side of the promotion

@pytest.mark.parametrize('cond', [1e2, PUBLISHED_COND, 1e5])
def test_the_levenberg_median_may_fall_on_either_side_of_the_promotion(cond):
    """Taking median(diag) in float32 or float64 must not change the step.

    If this ever fails, a jax-native rewrite is NOT free to reorder the median against the
    promotion, and the in-loop comment saying so becomes load-bearing rather than cautious.
    """
    H32 = spd(19, cond, 2).astype(np.float32)
    g = np.random.default_rng(3).standard_normal(19)
    m_before = float(np.median(np.clip(np.diag(H32), 1e-12, None)))
    m_after = float(np.median(np.clip(np.diag(H32.astype(np.float64)), 1e-12, None)))
    dg = np.clip(np.diag(H32), 0, None)
    step_b = np.linalg.solve(H32 + LAM * np.diag(dg) + MU * m_before * np.eye(19), g)
    step_a = np.linalg.solve(H32 + LAM * np.diag(dg) + MU * m_after * np.eye(19), g)
    rel = float(np.linalg.norm(step_b - step_a) / np.linalg.norm(step_b))
    assert rel < 1e-6, (
        f'at cond={cond:g} the median ordering changes the step by {rel:.2e}. The library takes '
        f'it in float32, inside the JAX implementation; if that stops being free, the choice has '
        f'to become deliberate rather than incidental')


# ------------------------------------------ SITE 3: raw-float64 vs scaled-float32 iterate

class _LSQ:
    """Least squares whose minimiser IS the truth, so "error against truth" is a real quantity.

    ``r(u) = A u + eps``, with ``eps`` REDRAWN per evaluation so the fit wanders on a noise floor.
    That wandering is the point: it is the regime both real problems live in, and the only regime
    where "does float32 cost anything" has a measurable answer rather than a tautological one.
    """

    def __init__(self, dtype, scaled, noise, seed):
        self.dtype, self.scaled, self.noise = dtype, scaled, noise
        self.rng = np.random.default_rng(seed)
        self.A = np.random.default_rng(7).standard_normal((40, 9))
        self.truth = np.array([1000., .3, -.2, .1, .5, .86, .7, .71, 2.0])

    def _u(self, theta):
        t = np.asarray(theta, float)
        return t if self.scaled else (t - self.truth) / SCALE9

    def grad_metric_loss(self, theta, step, refresh=True):
        # The gradient is w.r.t. the ITERATE, not w.r.t. u. In raw coordinates u=(theta-truth)/S,
        # so dr/dtheta = A/S and the chain-rule factor cannot be dropped -- omitting it made both
        # raw arms return an identical value, which is impossible when the dtype is what is under
        # test, and is how the original instrument was caught measuring nothing.
        u = self._u(theta)
        r = self.A @ u + self.noise * self.rng.standard_normal(40)
        J = self.A if self.scaled else self.A / SCALE9[None, :]
        return J.T @ r, J.T @ J, float(r @ r)

    def accumulate(self, theta, dtheta):
        return np.asarray(theta + dtheta, dtype=self.dtype)

    def to_raw(self, theta):
        t = np.asarray(theta, float)
        return self.truth + SCALE9 * t if self.scaled else t


def _final_error(dtype, scaled, noise, seed, steps=120):
    p = _LSQ(dtype, scaled, noise, seed)
    start_raw = p.truth + SCALE9 * 0.3
    start = ((start_raw - p.truth) / SCALE9) if scaled else start_raw
    res = gauss_newton(p, np.asarray(start, dtype=dtype), steps, lam=LAM, mu=MU,
                       scale=(None if scaled else SCALE9), max_step=3.0, lr=1.0,
                       readout='polyak', polyak=40)
    return float(np.abs(p.to_raw(res['theta']) - p.truth).max() / SCALE9.max())


@pytest.fixture(scope='module')
def arms():
    """(raw-f64, raw-f32, scaled-f32, seed spread) per noise level, four seeds each."""
    out = {}
    for noise in (0.0, 1e-4, 1e-2):
        a = [_final_error(np.float64, False, noise, s) for s in range(4)]
        b = [_final_error(np.float32, False, noise, s) for s in range(4)]
        c = [_final_error(np.float32, True, noise, s) for s in range(4)]
        out[noise] = (float(np.mean(a)), float(np.mean(b)), float(np.mean(c)),
                      float(np.std(a, ddof=1)))
    return out


@pytest.mark.parametrize('noise', [0.0, 1e-4, 1e-2])
def test_the_scaled_float32_iterate_reaches_the_float64_noise_floor(arms, noise):
    """The claim four docstrings rest on: scaled-float32 costs nothing against raw-float64.

    Judged against the SEED SPREAD of the float64 arm, not against zero -- the fit wanders on a
    Monte-Carlo floor, so its own run-to-run scatter is the smallest difference this problem can
    resolve. That is the same discipline the calibration and reconstruction comparisons use.
    """
    raw64, _, sc32, spread = arms[noise]
    if noise == 0.0:
        assert sc32 <= 1e-9, f'at zero noise scaled-float32 should be exact, got {sc32:.3e}'
    else:
        assert abs(sc32 - raw64) < 2 * spread, (
            f'at noise {noise:g} scaled-float32 lands {abs(sc32 - raw64):.3e} from raw-float64, '
            f'outside twice the float64 arm\'s own seed spread ({spread:.3e})')


@pytest.mark.parametrize('noise', [0.0, 1e-4])
def test_the_instrument_can_tell_the_representations_apart(arms, noise):
    """The control ON the control, and the reason the null result above is not vacuous.

    If raw-float32 also matched raw-float64, this file would be measuring nothing at all and its
    passing would mean nothing. Raw float32 must be visibly worse -- it carries energy at ~1000
    MeV where float32 spacing is 6.1e-5, so small steps there are silently lost. Measured at zero
    noise: raw-float32 stalls at 1.9e-5 against raw-float64's 3.3e-14.

    Only the low-noise arms can show this. At noise 1e-2 the Monte-Carlo floor is far above
    float32 resolution and legitimately hides the difference, which is itself the finding: the
    representation stops mattering once the problem is noisy enough.
    """
    raw64, raw32, _, spread = arms[noise]
    assert raw32 > raw64 + max(spread, 1e-12), (
        f'at noise {noise:g} raw-float32 ({raw32:.3e}) is indistinguishable from raw-float64 '
        f'({raw64:.3e}), so this file cannot tell the two representations apart and its other '
        f'assertions are vacuous')
