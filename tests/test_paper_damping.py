"""The 2-D loss-geometry figure's damped step, kept equal to the library's.

``analysis/paper/utils/damping.py`` holds one definition shared by the figure's two halves (the
streamline field and the trajectory overlaid on it), so the GN arm takes the same step the joint
fit takes by construction.

It does not call :func:`lucid.fitting.transforms.damped_matrix` because importing it runs
``lucid/fitting/__init__.py``, which pulls jax and the detector geometry into what is otherwise a
numpy plotting dependency. Both use the same convention (the Levenberg median is filtered to the
entries carrying curvature), and these tests keep the numpy transcription equal to the library.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from analysis.paper.utils.damping import damped_step
from lucid.fitting.transforms import damped_matrix as _damped_matrix_jax

LAM, MU = 0.01, 0.1

# The library is JAX and this repo never enables x64, so it computes in float32 where `damping.py`
# is float64: comparisons are at float32 tolerance, not exact. A convention change is a factor of
# two on the Levenberg base while float32 noise is ~1e-7;
# `test_the_tolerance_can_still_catch_a_convention_change` checks the bound still separates them.
RTOL = 2e-6


def damped_matrix(H, *, lam, mu):
    """The library's damped matrix, as float64 numpy so it can be compared with `damped_step`."""
    return np.asarray(_damped_matrix_jax(jnp.asarray(H, dtype=jnp.float32), lam=lam, mu=mu),
                      dtype=float)


def _psd(rng, n):
    a = rng.standard_normal((n, n))
    return a.T @ a


def test_agrees_with_the_library_wherever_the_diagonal_is_positive():
    """The two conventions coincide on every matrix either figure has actually seen."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (2, 3, 5, 19):
        for _ in range(200):
            f, g = _psd(rng, n), rng.standard_normal(n)
            assert np.diag(f).min() > 1e-12          # the condition under which they coincide
            lib = np.linalg.solve(damped_matrix(f, lam=LAM, mu=MU), g)
            fig = damped_step(f, g, LAM, MU)
            # Norm-relative: RTOL is a fraction, so it must be compared with a relative error.
            worst = max(worst, float(np.linalg.norm(fig - lib) / np.linalg.norm(lib)))
    assert worst < RTOL, f"conventions disagree on a positive diagonal by {worst:.3e} relative"


def test_they_now_AGREE_on_a_null_direction():
    """Both exclude a direction with no curvature from the Levenberg base.

    This is the case that distinguishes the conventions: a floored median would halve the base on
    this 2x2 and double the step along the null direction.
    """
    f = np.array([[1.0, 0.0], [0.0, 0.0]])
    g = np.array([1.0, 1.0])
    fig = damped_step(f, g, LAM, MU)
    lib = np.linalg.solve(damped_matrix(f, lam=LAM, mu=MU), g)
    np.testing.assert_allclose(lib, fig, rtol=RTOL, atol=0)


def test_a_fully_degenerate_metric_still_solves():
    """base falls back to 1.0, so the Levenberg term is the only thing keeping the solve finite."""
    out = damped_step(np.zeros((3, 3)), np.ones(3), LAM, MU)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, np.ones(3) / MU, rtol=1e-9)


# A real Fisher diagonal captured from a muon event at the published nrays working point (250k
# rays, SK geometry), at the seed and at truth. Order: E, x, y, z, sin/cos(theta), sin/cos(phi), t0.
#
# Captured rather than generated: the fits run in SCALE9 coordinates, so the diagonal carries
# different units per parameter and spans 2.8e-03 to 2.6e+06 (a factor of 9e8), a range the
# Gaussian `a.T @ a` matrices above never reach. Regenerating it needs a GPU and a PhotonSim ROOT
# file outside the repo, which is why the numbers are stored here.
REAL_FISHER_DIAG = {
    'seed': np.array([2.793e-03, 1.084e+04, 8.324e+03, 1.182e+04, 2.631e+06,
                      1.953e+05, 9.316e+03, 1.460e+05, 1.087e+03]),
    'truth': np.array([3.414e-03, 3.607e+03, 3.466e+03, 4.776e+03, 1.248e+06,
                       1.224e+05, 8.145e+02, 1.150e+05, 2.981e+02]),
}


@pytest.mark.parametrize('where', ['seed', 'truth'])
def test_the_two_conventions_agree_on_a_REAL_fisher_spectrum(where):
    """The two agree on a real, badly scaled spectrum that the synthetic matrices above cannot reach."""
    d = REAL_FISHER_DIAG[where]
    H = np.diag(d)
    lib = np.linalg.solve(damped_matrix(H, lam=LAM, mu=MU), np.ones(9))
    fig = damped_step(H, np.ones(9), LAM, MU)
    np.testing.assert_allclose(lib, fig, rtol=RTOL, atol=0)


@pytest.mark.parametrize('where', ['seed', 'truth'])
def test_the_real_problem_stays_far_from_the_regime_where_they_diverge(where):
    """Margin, not just agreement — agreement alone would not say how close the edge is.

    Two things must hold for the convention to be inert: no entry near the cutoff, and (the part
    that actually drives divergence) fewer than half the entries below it, since the median is
    what collapses. Asserted separately because the first can hold while the second fails on a
    problem with many soft directions.
    """
    import inspect
    # Read the LIBRARY's cutoff rather than restating 1e-12. Hardcoding it would make this test
    # blind to the one change most likely to break the claim: someone raising `rel_cutoff` to a
    # value that starts excluding real, informative directions.
    default_cutoff = inspect.signature(_damped_matrix_jax).parameters["rel_cutoff"].default

    d = REAL_FISHER_DIAG[where]
    cutoff = default_cutoff * d.max()
    n_below = int((d <= cutoff).sum())
    assert n_below == 0, (
        f'{n_below} of {len(d)} directions fall at or below the library cutoff '
        f'({cutoff:.3e}); they no longer contribute to the Levenberg base')
    assert n_below < len(d) // 2, (
        'more than half the diagonal sits at the cutoff — the median IS the cutoff and the '
        'isotropic damping has collapsed, which is the failure the filter convention prevents')
    # 100x, against a measured 1062x (seed) and 2736x (truth). This is the margin against the
    # RELATIVE cutoff (1e-12*max(diag), ~2.6e-06 here), not against an absolute 1e-12 floor; the
    # two differ by max(diag), six orders of magnitude.
    assert d.min() / cutoff > 100, (
        f'smallest curvature {d.min():.3e} is only {d.min() / cutoff:.0f}x the relative cutoff '
        f'{cutoff:.3e}; the conventions are no longer comfortably interchangeable here')


def test_the_real_spectrum_is_something_the_random_generator_cannot_produce():
    """The control on the fixture, and the reason it is captured rather than generated.

    If a Gaussian `a.T @ a` could produce this dynamic range, the tests above would already cover
    the case and this fixture would be redundant. It cannot: 200 draws stay within a factor of
    ~1e3, against the real problem's 9e8. Failing here would mean the fixture is no longer
    exercising anything the cheap generator misses.
    """
    rng = np.random.default_rng(0)
    worst = max(
        (lambda dg: dg.max() / dg.min())(np.diag(a.T @ a))
        for a in (rng.standard_normal((9, 9)) for _ in range(200)))
    real = REAL_FISHER_DIAG['seed']
    real_range = real.max() / real.min()
    assert real_range > 1e3 * worst, (
        f'the random ensemble reaches a dynamic range of {worst:.1e}, close to the real '
        f'{real_range:.1e} — the captured fixture no longer adds coverage')


def test_the_tolerance_can_still_catch_a_convention_change():
    """RTOL still rejects a convention change, not just float32 noise.

    The floored-median convention on a null direction must miss the bound by orders of magnitude,
    otherwise the tolerance used above could hide a real divergence.
    """
    f = np.array([[1.0, 0.0], [0.0, 0.0]])
    g = np.array([1.0, 1.0])
    # The floored convention: floor the diagonal at 1e-12 and take the median over ALL entries,
    # so the null direction halves the Levenberg base instead of being excluded from it.
    dg = np.clip(np.diag(f), 0.0, None)
    floored_base = float(np.median(np.clip(np.diag(f), 1e-12, None)))
    a_floor = f + LAM * np.diag(dg) + MU * floored_base * np.eye(2)
    old = np.linalg.solve(a_floor, g)
    new = damped_step(f, g, LAM, MU)

    rel = np.abs(old / new - 1.0).max()
    assert rel > 1e3 * RTOL, (
        f'the old floored convention differs from the current one by only {rel:.2e}, which is '
        f'not comfortably above the {RTOL:g} float32 tolerance — the comparisons above can no '
        f'longer distinguish a convention change from arithmetic noise')
