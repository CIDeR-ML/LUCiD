"""The whole branch map of the softness knob, including the published point.

``overlap_sigma_frac`` --- the dimensionless width the call sites spell
``temperature`` --- is the most consequential dial in the forward model, and
before this file no test exercised the hard-step or the straight-through branch
at all: ``tests/string/test_overlap_fix.py`` uses sigma/r in {0.2, 1.0 ... 30},
so the surrogate that is live on two paper figures and in every
``temperature=0.0`` notebook had no coverage, and nothing pinned sigma/r = 0.02.

0.02 is the published point and it sits **exactly** on ``_ST_MIN_SIGMA``: the
branch test is ``frac * r < _ST_MIN_SIGMA * r``, both sides the same IEEE
product, so only the strict ``<`` keeps the published figure on the soft table.
One ULP below it is the hard step --- which is the same bit-for-bit as
``temperature=None``, so a width sweep that crosses the boundary returns a
narrow-convolution arm that is secretly the hard step.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.overlap import _ST_MIN_SIGMA, create_overlap_prob

# The three shipped sensor radii: SK/HK (0.25 m, dyadic), IceCube DOM
# (0.165 m, not dyadic) and the small-PMT value the overlap demo uses (0.04 m).
RADII = (0.25, 0.165, 0.04)

# The fracs the tree actually sets, plus the two that bracket the boundary.
FRACS = (None, 0.0, 1e-9, math.nextafter(0.02, 0.0), 0.02, 0.05, 0.1, 0.2)

# Which branch each frac must take, at every radius. Only 0.02 and above are
# soft; the published 0.02 is soft because the comparison is strict.
EXPECT_SOFT = {None: False, 0.0: False, 1e-9: False, math.nextafter(0.02, 0.0): False,
               0.02: True, 0.05: True, 0.1: True, 0.2: True}

# Reduced quadrature: this file asserts WHICH branch is taken and the exact
# arithmetic of the hard one, not the accuracy of the soft table (that is
# tests/string/test_overlap_fix.py). The reduced layout also keeps every build
# out of the shared on-disk cache, which is keyed on the default layout only.
SMALL = dict(n_theta=200, n_rho=200, num_dense=60, num_sparse=10, use_cache=False)


def _overlap(frac, r, **kw):
    """The closure the call sites build: sigma = frac * r, or None (shared.py)."""
    return create_overlap_prob(None if frac is None else frac * r, r, **{**SMALL, **kw})


def _is_soft(fn, r):
    """A soft table is strictly between 0 and 1 at d = r; both hard forms are 0."""
    at_r = float(fn(r))
    if at_r == 0.0:
        return False
    assert 0.0 < at_r < 1.0, f'neither hard nor a probability at d=r: {at_r}'
    return True


@pytest.mark.parametrize('r', RADII)
@pytest.mark.parametrize('frac', FRACS)
def test_branch_map(frac, r):
    """Each (frac, r) takes the branch the published table says it takes."""
    assert _is_soft(_overlap(frac, r), r) is EXPECT_SOFT[frac]


@pytest.mark.parametrize('r', RADII)
def test_published_point_is_soft_and_one_ulp_below_it_is_hard(r):
    """0.02 is soft; nextafter(0.02, 0) is hard. The published figure rests on this.

    Asserted separately from the branch map because it is the one pair whose
    answer depends on the comparison being ``<`` and not ``<=``, and on both
    sides being the identical float product ``frac * r``.
    """
    below = math.nextafter(0.02, 0.0)
    assert below * r < _ST_MIN_SIGMA * r, 'one ULP below no longer rounds below the bar'
    assert not _is_soft(_overlap(below, r), r), f'{below!r} * {r} took the soft branch'
    assert _is_soft(_overlap(0.02, r), r), f'the published 0.02 * {r} took the hard branch'


@pytest.mark.parametrize('r', RADII)
def test_none_and_zero_give_bit_identical_closures(r):
    """``temperature=None`` and ``temperature=0.0`` are the same function, exactly.

    Both the straight-through default and the pure hard step, because the two
    differ in their backward pass and a caller may pick either.
    """
    d = jnp.linspace(0.0, 3.0 * r, 257)
    for st_width_frac in (0.35, 0.0):
        none_v = np.asarray(jax.vmap(_overlap(None, r, st_width_frac=st_width_frac))(d))
        zero_v = np.asarray(jax.vmap(_overlap(0.0, r, st_width_frac=st_width_frac))(d))
        assert np.array_equal(none_v, zero_v), (
            f'None and 0.0 differ at st_width_frac={st_width_frac}')


@pytest.mark.parametrize('r', RADII)
@pytest.mark.parametrize('frac', [f for f in FRACS if not EXPECT_SOFT[f]])
def test_hard_forward_is_exactly_the_step(frac, r):
    """Every hard-branch forward equals ``where(d < r, 1, 0)`` bit for bit.

    Including at d = r itself, where the strict ``<`` puts the sensor edge
    outside: the straight-through surrogate changes the gradient only, never
    the value.
    """
    d = jnp.concatenate([jnp.linspace(0.0, 3.0 * r, 256), jnp.asarray([r])])
    expected = np.asarray(jnp.where(d < r, 1.0, 0.0))
    for st_width_frac in (0.35, 0.0):
        got = np.asarray(jax.vmap(_overlap(frac, r, st_width_frac=st_width_frac))(d))
        assert np.array_equal(got, expected), (
            f'frac={frac!r} r={r} st_width_frac={st_width_frac} is not the hard step')


@pytest.mark.parametrize('r', RADII)
def test_straight_through_has_a_gradient_where_the_hard_step_has_none(r):
    """The surrogate's gradient is finite and negative exactly where the step's is 0.

    This is the whole reason the straight-through branch exists: a hard step has
    zero gradient almost everywhere, so charge cannot constrain the track
    position without it.
    """
    pure = _overlap(None, r, st_width_frac=0.0)
    surrogate = _overlap(None, r)
    for d in (0.1 * r, 0.5 * r, 0.9 * r, 1.1 * r, 2.0 * r):
        g_hard = float(jax.grad(pure)(d))
        g_soft = float(jax.grad(surrogate)(d))
        assert g_hard == 0.0, f'the pure hard step has a gradient at d={d}: {g_hard}'
        assert math.isfinite(g_soft) and g_soft < 0.0, (
            f'the straight-through gradient at d={d} is {g_soft}, want finite and negative')
        # The documented surrogate (create_overlap_prob, st_width_frac=0.35): the logistic
        # s = 1/(1 + exp(-(r - d)/w)), w = 0.35 r, whose slope is -s(1 - s)/w. A wrong width or
        # a factor 2 moves it by far more than float32 rounding (rtol 1e-5).
        w = 0.35 * r
        s_ = 1.0 / (1.0 + math.exp(-(r - d) / w))
        assert g_soft == pytest.approx(-s_ * (1.0 - s_) / w, rel=1e-5), (d, g_soft)


@pytest.mark.parametrize('r', RADII)
def test_soft_branch_falls_across_the_sensor_edge_and_is_differentiable(r):
    """At the published 0.02 the overlap falls from ~1 to ~0 across the sensor edge.

    Coarse-grained on purpose. The reduced quadrature this file uses overshoots
    1 by ~7e-5 and wobbles at the 1e-6 level, so point-by-point monotonicity is
    a property of SMALL and not of the shipped table: the table's accuracy is
    ``tests/string/test_overlap_fix.py``'s remit, at the full quadrature. What
    is asserted here is that the soft branch is the soft branch --- a smooth
    fall through a half at the edge, with a gradient --- and not the hard step.
    """
    fn = _overlap(0.02, r)
    sigma = 0.02 * r
    assert abs(float(fn(0.0)) - 1.0) < 0.01, 'a centred photon is not (nearly) fully inside'
    # Half the Gaussian mass is inside the disk when the photon sits on the edge.
    assert abs(float(fn(r)) - 0.5) < 0.05, f'overlap at d = r is {float(fn(r))}, want ~0.5'
    fall = float(fn(r - 4.0 * sigma)) - float(fn(r + 4.0 * sigma))
    assert fall > 0.9, f'the overlap falls by only {fall} across +-4 sigma of the edge'
    assert float(fn(2.0 * r)) < 1e-6, 'overlap survives two radii out'
    g = float(jax.grad(fn)(0.98 * r))
    assert math.isfinite(g) and g < 0.0, f'soft gradient at 0.98r is {g}'
