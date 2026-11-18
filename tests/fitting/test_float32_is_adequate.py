"""Is float32 adequate where going jax-native (``jnp`` everywhere, ``x64`` off) changes the numerics?

Production docstrings quote the float32 numbers asserted here; this file is what fails if they stop
being true. The arithmetic changes at two separable sites, each tested with its own control:

  SITE 1  the linear solve. ``lucid.fitting`` has one damped Gauss-Newton direction,
          ``scale_by_damped_gauss_newton`` (damped matrix + ``jnp.linalg.solve``), in JAX, so it
          runs in float32; the float64 arm is a reference built in this file. cond(H) ~ 6800 is
          the published 19-parameter calibration matrix.

(The Levenberg median is taken inside the float32 ``damped_matrix``; that the float32 direction
matches the float64 reference covers it. A pure-numpy comparison of the two median orderings
called no library code and is gone.)

The iterate is float32 too, in raw units. Near 1000 MeV float32 resolves 6.1e-5 MeV, while a
reconstruction step late in the fit still moves energy by ~0.5 MeV, so no step is lost.

Everything here runs on CPU and is deterministic: a regression gate, not a benchmark.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from lucid.fitting.transforms import scale_by_damped_gauss_newton


def damped_matrix_f64(H, *, lam, mu):
    """The float64 reference: the documented convention (lucid/fitting/transforms.py damped_matrix),
    evaluated in double precision by numpy. Built here because the library's only implementation
    is JAX, which computes in float32 while ``x64`` is off."""
    dg = np.clip(np.diag(H), 0.0, None)
    cutoff = 1e-12 * dg.max()
    informative = dg[dg > cutoff]
    assert informative.size, 'reference undefined on an all-flat diagonal'
    base = float(np.median(informative))
    return H + lam * np.diag(dg) + mu * base * np.eye(H.shape[0])

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
    # Under test: the LIBRARY's direction (damped matrix and solve), in the float32 it runs in.
    opt = scale_by_damped_gauss_newton(LAM, MU)
    d32, _ = opt.update(jnp.asarray(g, jnp.float32), opt.init(None),
                        metric=jnp.asarray(H, jnp.float32))
    assert d32.dtype == jnp.float32
    d32 = np.asarray(d32, np.float64)
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
    below that cannot change any decision the step feeds (measured 2.8e-07 at the published
    conditioning).
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
