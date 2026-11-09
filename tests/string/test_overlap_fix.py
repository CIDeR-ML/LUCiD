"""The soft overlap table across the width range it is built for, against its closed form.

The overlap of a disk of radius r with a 2-D Gaussian of width sigma centred a distance d away is
P(|X|^2 <= r^2) for X ~ N((d, 0), sigma^2 I). |X|^2 / sigma^2 is noncentral chi-square with 2
degrees of freedom and noncentrality (d/sigma)^2, so

    f(d) = ncx2.cdf((r/sigma)^2, 2, (d/sigma)^2),      f(0) = 1 - exp(-r^2 / (2 sigma^2)),
    f'(d) = -(d / sigma^2) * (F_2 - F_4)  (the same CDF at 2 and 4 degrees of freedom),

(1 - f is the Marcum Q-function Q_1(d/sigma, r/sigma); M. K. Simon, *Probability Distributions
Involving Gaussian Random Variables*, Springer 2002, ch. 2). The derivative follows from
dF_k(x; lam)/dlam = -(F_k - F_{k+2})/2, term by term in the Poisson mixture. The library never
evaluates this: it integrates the kernel numerically into a lookup table. And independently of
both, the integral of f(d) * 2 pi d over the plane is the disk area pi r^2.

``lucid/overlap.py`` has **one** soft code path: above ``_ST_MIN_SIGMA * r`` every width goes
through ``precompute_lookup`` and the same interpolant, and below it the call falls through to
the hard step. What is special about sigma = 3r is only that the default ``d_max_factor=10``
stops reaching far enough past the disk for a kernel that wide, which is why the wide-kernel
tests pass an explicit ``d_max_factor``.

QUADRATURE. Except for the production-layout test, the tables here are built with 400 x 400
quadrature nodes instead of the default 2000 x 2000: 2 s per table instead of ~55 s, and the
same error against the closed form (measured, sigma/r in 0.2..3: max abs error 1.9e-5 vs 1.6e-5;
the error is the interpolant's, not the quadrature's).

The branch map itself --- which width takes which branch, and the hard and straight-through
arms this file never touches --- is ``tests/propagation/test_overlap_branches.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import ncx2

from lucid.overlap import create_overlap_prob

r = 0.165  # IceCube DOM radius
N_QUAD = 400


def exact(d, sigma):
    return ncx2.cdf((r / sigma) ** 2, 2, (np.asarray(d, np.float64) / sigma) ** 2)


def exact_slope(d, sigma):
    x, lam = (r / sigma) ** 2, (np.asarray(d, np.float64) / sigma) ** 2
    return -(np.asarray(d) / sigma ** 2) * (ncx2.cdf(x, 2, lam) - ncx2.cdf(x, 4, lam))


def table(ratio, d_max_factor=10.0):
    return create_overlap_prob(ratio * r, r, n_theta=N_QUAD, n_rho=N_QUAD, use_cache=False,
                               d_max_factor=d_max_factor)


def values(fn, d):
    return np.asarray(jax.vmap(fn)(jnp.asarray(d, jnp.float32)), np.float64)


def cross_section(overlap_fn, d_max, n=5000):
    """∫ overlap(d) × 2πd dd — should equal πr²."""
    d_vals = jnp.linspace(0, d_max, n)
    integrand = jax.vmap(overlap_fn)(d_vals) * 2 * jnp.pi * d_vals
    return float(jnp.trapezoid(integrand, d_vals))


# ── The production table: default layout and the shipped temperature ──

def test_default_table_matches_closed_form():
    """temperature 0.2 with the default layout, through the cache like every production caller.

    Max abs error over [0, 3r] measured 3.1e-5 (float32 interpolation between float64 nodes);
    1e-4 fails a sigma off by 1% (which moves f by up to 2e-2 at the edge)."""
    sigma = 0.2 * r
    fn = create_overlap_prob(sigma, r, use_cache=True)
    d = np.linspace(0, 3 * r, 301)
    assert np.max(np.abs(values(fn, d) - exact(d, sigma))) < 1e-4
    rel = abs(cross_section(fn, 10 * r) - np.pi * r ** 2) / (np.pi * r ** 2)
    assert rel < 1e-4, rel                                   # measured 6.4e-06


# ── Every width the table is built for ──

# (sigma / r, d_max_factor): the default reach up to 2.9 r, an explicit one beyond.
WIDTHS = [(1.0, 10.0), (2.0, 10.0), (2.9, 10.0), (3.0, 15.0), (6.0, 30.0), (30.0, 150.0)]


@pytest.mark.parametrize("ratio,d_max_factor", WIDTHS)
def test_table_matches_closed_form(ratio, d_max_factor):
    """f(d) over the table's reach, and f(0) = 1 - exp(-r^2 / 2 sigma^2).

    Measured max abs error <= 1e-5 at these widths (f(0) is 0.39 at sigma = r, 5.6e-4 at 30 r).
    The bound 1e-4 abs plus 1e-3 relative fails the kernel exponent missing its factor 2
    (f(0) becomes (1 - exp(-r^2/sigma^2)) / 2) by orders of magnitude."""
    sigma = ratio * r
    fn = table(ratio, d_max_factor)
    d = np.linspace(0, min(d_max_factor * r, r + 5 * sigma), 201)
    got, want = values(fn, d), exact(d, sigma)
    assert np.max(np.abs(got - want)) < 1e-4
    np.testing.assert_allclose(got[0], 1 - np.exp(-r ** 2 / (2 * sigma ** 2)), rtol=1e-3)
    big = want > 1e-2 * want[0]            # relative error where f is not in its far tail
    assert np.max(np.abs(got - want)[big] / want[big]) < 1e-3


@pytest.mark.parametrize("ratio,d_max_factor", [w for w in WIDTHS if w[0] != 2.9])
def test_cross_section_is_the_disk_area(ratio, d_max_factor):
    """Wherever the table reaches past r + 5 sigma the integral is pi r^2 (measured rel error
    <= 3.3e-5). At sigma = 2.9r the default reach of 10 r = 3.4 sigma cuts the tail short (2.2%
    missing), so that width is checked against the closed form above instead."""
    sigma = ratio * r
    fn = table(ratio, d_max_factor)
    cs = cross_section(fn, min(d_max_factor * r, r + 5 * sigma))
    rel = abs(cs - np.pi * r ** 2) / (np.pi * r ** 2)
    assert rel < 1e-3, f"σ/r={ratio}: cross-section error {rel:.2e}"


# ── Test: the table is continuous in σ across 3r ──

def test_table_is_continuous_in_sigma_across_3r():
    """Two widths 0.67% apart change the overlap by what the closed form says, 1.3% at most.

    3r is where the tests start passing an explicit ``d_max_factor``, so a
    reader could take it for a code boundary. It is not one: both calls take
    the same ``precompute_lookup`` path, and this asserts they agree.
    """
    d = np.array([0.0, 0.5 * r, r, 2 * r, 3 * r])
    below, above = values(table(2.99), d), values(table(3.01), d)
    want = exact(d, 2.99 * r) - exact(d, 3.01 * r)
    assert np.max(np.abs(below - above) / below) < 0.03
    np.testing.assert_allclose(below - above, want, atol=2e-5)


# ── Test: gradient works in all regimes ──

@pytest.mark.parametrize("ratio", [0.2, 1.5, 6.0])
def test_gradient_matches_closed_form_at_every_width(ratio):
    """d(overlap)/d(d) at the disk edge d = r, narrow to wide, against f'(d).

    Autodiff reads the slope of one interpolation segment; against the true slope that is
    within 1e-2 at the edge (lucid/overlap.py, _NUM_DENSE comment; measured 2e-4 at sigma = 0.2 r,
    9e-3 at 2.9 r). rtol 3e-2 fails a factor 2 or a sign."""
    sigma = ratio * r
    fn = table(ratio, max(10.0, 5.0 * ratio))
    for d0 in (r, 0.5 * r) if ratio >= 1 else (r,):
        g = float(jax.grad(fn)(jnp.float32(d0)))
        want = float(exact_slope(d0, sigma))
        assert want < 0
        np.testing.assert_allclose(g, want, rtol=3e-2, err_msg=f"σ/r={ratio}, d={d0 / r}r")
