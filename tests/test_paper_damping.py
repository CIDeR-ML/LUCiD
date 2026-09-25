"""The 2-D loss-geometry figure's damped step, and why it is not the library's.

``analysis/paper/utils/damping.py`` holds one definition shared by the figure's two halves — the
streamline field and the trajectory overlaid on it — which is what makes "the GN arm takes the
same step the joint fit takes" true by construction rather than by two files restating the same
arithmetic.

It does not call :func:`lucid.fitting.transforms.damped_matrix`, and the reason is now purely a packaging
one: importing it runs ``lucid/fitting/__init__.py``, which pulls jax and the detector geometry
into what is otherwise a numpy plotting dependency.

The CONVENTIONS no longer differ, and this file used to say they did. The library floored the
Levenberg median at 1e-12 where the figure filtered to the entries carrying curvature, which made
them disagree by exactly a factor of two on a 2x2 with a null direction. The library has since
adopted the filter — a floor is not invariant to adding an unconstrained parameter, and collapses
once half the diagonal sits at it — so the tests below pin the two as EQUAL, and one of them says
in its own name that it used to assert the opposite.

What the tests here are for, then, is keeping a transcription in step with the thing it
transcribes. That is a weaker guarantee than having one implementation, and it is worth being
honest that it exists because of an import boundary rather than because two conventions are
genuinely wanted.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from analysis.paper.utils.damping import damped_step
from lucid.fitting.transforms import damped_matrix as _damped_matrix_jax

LAM, MU = 0.01, 0.1

# The library implementation is JAX and this repo never enables x64, so it computes in float32
# where `damping.py` is float64. The comparison is therefore at float32 tolerance, NOT exact --
# and that is a deliberate loosening from the `rtol=0` this file used to assert against the numpy
# implementation that has since been deleted.
#
# It costs nothing the test was actually for. A CONVENTION divergence is a factor of two on the
# Levenberg base (which is what the null-direction case measured before the library adopted the
# filter); float32 noise is ~1e-7. Six orders separate the thing being caught from the thing being
# tolerated, and `test_the_tolerance_can_still_catch_a_convention_change` holds that claim rather
# than leaving it as an assertion in prose.
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
            # NORM-relative. An absolute difference here compares a float32 rounding error against
            # a tolerance expressed as a fraction, which is a units error -- it read 8.1e-06 on
            # steps of order one and looked like a failure when it was 1e-6 relative.
            worst = max(worst, float(np.linalg.norm(fig - lib) / np.linalg.norm(lib)))
    assert worst < RTOL, f"conventions disagree on a positive diagonal by {worst:.3e} relative"


def test_they_now_AGREE_on_a_null_direction():
    """This test used to assert the opposite, and the change is the point.

    The figure filtered the Levenberg median to the entries carrying curvature while the library
    floored it at 1e-12, so a direction with no curvature made the library's base the mean of the
    surviving entry and the floor — half the figure's — and the step along the null direction
    differed by exactly two. The library has since adopted the filter, for reasons recorded in
    :func:`lucid.fitting.transforms.damped_matrix`, so the divergence is gone.

    ``damping.py`` remains a separate module, but no longer because it disagrees: importing
    ``lucid.fitting.gn`` runs ``lucid/fitting/__init__.py``, which pulls jax and the detector
    geometry into what is otherwise a numpy plotting dependency. This test is now the guard that
    the two stay in step.
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


# A CAPTURED REAL Fisher diagonal, not a synthetic one. Order: E, x, y, z, sin/cos(theta),
# sin/cos(phi), t0. Measured on a real muon
# event at the published nrays working point (250k rays, SK geometry, 11,096 PMTs), at the seed
# and again at truth.
#
# WHY CAPTURED RATHER THAN GENERATED. The tests above build matrices as `a.T @ a` from Gaussian
# entries, whose diagonal spans a ratio of roughly 0.006-0.3 — a well-conditioned ensemble that
# never approaches the degenerate regime the Levenberg term exists for, and that cannot reproduce
# what makes the real problem hard: these fits run in SCALE9 coordinates, so the diagonal carries
# DIFFERENT UNITS per parameter. The real spectrum spans 2.8e-03 to 2.6e+06 — a factor of 9e8,
# with the energy direction sitting 4e6 below the median. No random PSD generator produces that,
# which is why an audit found this case genuinely uncovered by the tests above.
#
# Regenerating: run the tool and paste the printed diagonals. It needs a GPU and a PhotonSim ROOT
# file outside the repo, which is exactly why the NUMBERS are captured here and the RUN is not.
REAL_FISHER_DIAG = {
    'seed': np.array([2.793e-03, 1.084e+04, 8.324e+03, 1.182e+04, 2.631e+06,
                      1.953e+05, 9.316e+03, 1.460e+05, 1.087e+03]),
    'truth': np.array([3.414e-03, 3.607e+03, 3.466e+03, 4.776e+03, 1.248e+06,
                       1.224e+05, 8.145e+02, 1.150e+05, 2.981e+02]),
}


@pytest.mark.parametrize('where', ['seed', 'truth'])
def test_the_two_conventions_agree_on_a_REAL_fisher_spectrum(where):
    """The gap the synthetic matrices above cannot reach.

    Floor and filter diverge only once a large fraction of the diagonal sits at or below the
    cutoff. On this problem nothing comes close — the smallest entry is ~1e9x the old 1e-12 floor
    — so the two forms return the same ``base`` and the change of convention was inert. That
    inertness is the claim `lucid/fitting/gn.py` makes, and this is what holds it.
    """
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
    # 100x, against a measured 1062x (seed) and 2736x (truth). NOTE these are margins against the
    # RELATIVE cutoff, which is 1e-12*max(diag) = ~2.6e-06 here — NOT the 2.8e9x quoted in
    # `gn.py`, which is the margin against the old ABSOLUTE 1e-12 floor. The two differ by
    # max(diag), six orders of magnitude, and conflating them is easy: the first draft of this
    # assertion demanded 1e6x and failed, because it was asserting the absolute-floor margin
    # against the relative cutoff.
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
    """The control on RTOL, promised where it is defined.

    Comparing a float64 transcription against a float32 implementation costs exactness, so the
    comparisons above loosened from `rtol=0` to 2e-6. That is only acceptable if the loosened
    bound still rejects the thing it exists to reject — a change in the CONVENTION, not in the
    arithmetic. Here that is the old floored median on a null direction, which is exactly the
    divergence this file used to assert and now asserts the absence of; it must fail the bound by
    orders of magnitude, not squeak past it.
    """
    f = np.array([[1.0, 0.0], [0.0, 0.0]])
    g = np.array([1.0, 1.0])
    # The pre-change convention: floor the diagonal at 1e-12 and take the median over ALL entries,
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
