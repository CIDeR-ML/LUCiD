"""Calibration accepts an ARBITRARY source layout, not one privileged shape.

Source diversity is the calibration lever `docs/guides/calibration.md` ranks first, so sources of
different pytree structure (a `LaserSource` has 5 fields, an `IsotropicSource` 3) must coexist.
Sources are grouped by pytree structure, mapped within each group, and scattered back into
source order. Grouping by STRUCTURE rather than by class is the point: two sources can be mapped
together precisely when they can be stacked, which is a property of their pytree.

What these tests hold:

  ADMISSIBILITY   every layout constructs -- mixed, reversed, interleaved, singleton, uniform.
  PARTITION       the grouping is by structure, contiguous runs are not assumed, and the
                  published layout resolves to one solo call plus one stacked group.
  ORDER           row `g = wl*n_sources + s` still belongs to source `s`. This is the one that
                  matters: a grouping that scattered results back wrongly would produce a
                  perfectly shaped, entirely misattributed forward.
  KEYS            a source's random stream depends on its ABSOLUTE index, not on how the layout
                  happened to be partitioned.

Bit-exactness of the published path is held elsewhere, by `tests/reconciliation/`.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.fitting import CalibrationForward, CalibrationJacobian, CalibrationParams
from lucid.sources import isotropic_source, laser_source

W = 2
P = CalibrationParams(n_wavelengths=W, basis='rlogit')


def L(x):
    # x is DISTINCT per source and is what the stubs below report, so a swap WITHIN a
    # group is visible; two sources at the same x would hide an intra-group swap.
    return laser_source(position=[x, 0, 15], direction=[0, 0, -1], intensity=1e7)


def I(x):
    return isotropic_source(position=[x, 0, 0], intensity=1e7)


LAYOUTS = {
    'published_1_laser_then_iso': [L(15), I(0), I(5), I(-5)],
    'mixed_two_lasers_then_iso': [L(15), L(-15), I(0)],
    'iso_first_lasers_after': [I(0), L(15), L(-15)],
    'interleaved': [L(1), I(2), L(3), I(4), I(5)],
    'single_source': [L(15)],
    'all_one_kind': [I(0), I(5), I(-5)],
}


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_every_layout_constructs(name):
    srcs = LAYOUTS[name]
    fwd = CalibrationForward(None, srcs, P, 32)
    assert fwd.S == W * len(srcs)


def test_the_published_layout_still_resolves_to_one_solo_plus_one_group():
    """The published layout must take the solo-plus-stacked path that `tests/reconciliation/` holds bit-exact."""
    fwd = CalibrationForward(None, LAYOUTS['published_1_laser_then_iso'], P, 32)
    idx = [ix for ix, _ in fwd._groups]
    assert idx == [[0], [1, 2, 3]], idx
    assert fwd._group_stacks[0] is None          # the laser is called singly
    assert fwd._group_stacks[1] is not None      # the isotropics are stacked


def test_grouping_is_by_structure_not_by_contiguity():
    """Interleaved sources must group with their own kind, not with their neighbours."""
    fwd = CalibrationForward(None, LAYOUTS['interleaved'], P, 32)
    assert [ix for ix, _ in fwd._groups] == [[0, 2], [1, 3, 4]]


def test_every_source_lands_in_exactly_one_group():
    """A partition, not a cover: a duplicated index would double-count a source silently."""
    for name, srcs in LAYOUTS.items():
        fwd = CalibrationForward(None, srcs, P, 32)
        seen = sorted(i for ix, _ in fwd._groups for i in ix)
        assert seen == list(range(len(srcs))), f'{name}: {seen}'


def test_rows_are_attributed_to_the_RIGHT_source():
    """The failure a shape check cannot see, on the route that could actually commit it.

    Row `g = wl*n_sources + s` must be source `s`. A partition that mapped correctly but scattered
    results back in GROUP order rather than SOURCE order would give a forward of exactly the right
    shape with every row belonging to the wrong source, and every downstream residual, gain and
    Jacobian column would inherit the misattribution silently.

    This must run the TRACED route: `predict=` returns from `__init__` before the partition is
    built (`calib.py`: the per-source branch), so its ordering is trivially [0..n-1]. A `sim` stub
    reporting the source's own x keeps it simulator-free while staying on the real path.
    """
    srcs = LAYOUTS['interleaved']
    NS = 8

    def sim(src, dp, key):                     # -> (charge,) ; reports WHICH source it got
        return (jnp.full((NS,), jnp.asarray(src.position)[0], dtype=jnp.float32),)

    fwd = CalibrationForward(sim, srcs, P, NS)
    assert [ix for ix, _ in fwd._groups] == [[0, 2], [1, 3, 4]]      # the route under test
    M = np.asarray(fwd(jnp.zeros(P.P), 0, jnp.ones(NS)))
    for wl in range(W):
        for s, src in enumerate(srcs):
            want = float(np.asarray(src.position)[0])
            assert np.allclose(M[wl * len(srcs) + s], want), \
                f'row (wl={wl}, s={s}) carries {M[wl * len(srcs) + s][0]}, expected source {s} ({want})'


def test_the_JACOBIAN_scatters_to_the_same_rows():
    """The Jacobian's rows are assembled in source order to match the forward's.

    A different partition, or the same partition scattered differently, would pair row `g` with
    another source's columns -- and because both objects group independently, nothing but a test
    forces them to agree.

    The stub is theta-DEPENDENT so the derivative is non-zero and still identifies its source.
    """
    srcs = LAYOUTS['interleaved']
    NS = 8

    def sim(src, dp, key):
        # The absorption length carries theta, so d(charge)/d(theta) is non-zero; x tags the
        # source. Any theta-dependent leaf would do -- this one is a scalar per wavelength.
        lam = jnp.asarray(dp.absorption.absorption_length, dtype=jnp.float32).ravel()[0]
        return (jnp.full((NS,), jnp.asarray(src.position)[0], dtype=jnp.float32) * lam,)

    jac = CalibrationJacobian(sim, srcs, P, NS, key0=3)
    assert [ix for ix, _ in jac._groups] == [[0, 2], [1, 3, 4]]
    theta = jnp.zeros(P.P)
    data = jnp.ones((W * len(srcs), NS))
    J = np.asarray(jac(theta, jnp.zeros(NS), 0, data, 1e-3, draws=(0,)))
    assert J.shape == (W * len(srcs), NS, P.P)
    # Each row's derivative magnitude must scale with ITS source's x, so a swap shows up.
    for wl in range(W):
        mags = [np.abs(J[wl * len(srcs) + s]).max() for s in range(len(srcs))]
        order = np.argsort(mags)
        want = np.argsort([abs(float(np.asarray(sc.position)[0])) for sc in srcs])
        assert list(order) == list(want), f'wl={wl}: rows ordered {order}, sources {want}'


def test_a_sources_keys_do_not_depend_on_the_partition():
    """Source `s` must draw `kb + 1000*s + wl` whatever else is in the layout.

    Otherwise adding a source would silently change the noise realisation of every source after
    it, and two layouts sharing a source would not share its draw.
    """
    a = CalibrationForward(None, LAYOUTS['interleaved'], P, 32)
    b = CalibrationForward(None, LAYOUTS['mixed_two_lasers_then_iso'], P, 32)
    ka_s, ka_g = a.keys(1000)
    kb_s, kb_g = b.keys(1000)
    assert np.array_equal(np.asarray(ka_s), np.asarray(kb_s))          # source 0
    for s in (1, 2):                                                   # sources shared by both
        assert np.array_equal(np.asarray(ka_g[s - 1]), np.asarray(kb_g[s - 1])), s


def test_a_layout_that_used_to_raise():
    """Mixed pytree arities (laser 5 fields, isotropic 3) must construct, not fail inside `tree_map`."""
    fwd = CalibrationForward(None, [L(15), L(-15), I(0)], P, 32)
    assert fwd.S == W * 3
