"""Structural calibration setup: partition / combine / classify (no simulator)."""
import numpy as np
import jax.numpy as jnp
import pytest

from lucid.detector_params import DetectorParams, _flatten_detector_params
from lucid.fitting.partition import partition, combine, classify, trained_leaves
from lucid.wavelength.optical_model import N_CONTROL

NS = 16


def _dp():
    rng = np.random.default_rng(0)
    return DetectorParams.from_flat(
        scatter_length=70., mie_scatter_length=3000., g=0.9,
        absorption_length=60., wall_reflection_rate=0.2, qe=0.07,
        abs_dev=jnp.asarray(1.0 + 0.1 * rng.standard_normal(N_CONTROL)),
        qe_corrections=jnp.asarray(1.0 + 0.05 * rng.standard_normal(NS)),
        t0=jnp.asarray(rng.standard_normal(NS)))


def test_combine_inverts_partition_roundtrip():
    """combine(partition(dp)) == dp for an arbitrary leaf selection."""
    dp = _dp()
    train = ['scatter_length', 'abs_dev', 'qe_corrections', 't0']
    theta, fixed = partition(dp, train)
    back = combine(theta, fixed)
    fa, fb = _flatten_detector_params(dp), _flatten_detector_params(back)
    for k in fa:
        np.testing.assert_allclose(np.asarray(fa[k]), np.asarray(fb[k]), err_msg=k)


def test_partition_splits_trained_and_frozen():
    dp = _dp()
    theta, fixed = partition(dp, ['scatter_length', 'qe_corrections'])
    # trained leaves live in theta, are None in fixed (and vice-versa)
    assert theta.scattering.scatter_length is not None
    assert fixed.scattering.scatter_length is None
    assert theta.per_pmt.qe_corrections is not None
    assert fixed.per_pmt.qe_corrections is None
    assert theta.absorption.absorption_length is None
    assert fixed.absorption.absorption_length is not None
    names = [n for n, _ in trained_leaves(theta)]
    assert set(names) == {'scatter_length', 'qe_corrections'}


def test_accepts_dotted_paths_and_bare_names():
    dp = _dp()
    a, _ = partition(dp, ['scattering.scatter_length'])
    b, _ = partition(dp, ['scatter_length'])
    assert a.scattering.scatter_length is not None
    assert b.scattering.scatter_length is not None


def test_unknown_leaf_raises():
    with pytest.raises(KeyError):
        partition(_dp(), ['not_a_leaf'])


def test_classify_routes_by_shape():
    """scalar→global/log, curve→global/geomean-gauge, (NS,)→per-PMT, t0→linear/mean."""
    dp = _dp()
    theta, _ = partition(dp, ['scatter_length', 'abs_dev', 'qe_corrections', 't0'])
    spec = classify(theta, NS)
    assert spec['n_sensors'] == NS
    g = {n: (sh, sp) for n, sh, sp in spec['globals']}
    p = {n: (sh, sp) for n, sh, sp in spec['per_pmt']}
    # scalar length → global, log space, no gauge
    assert g['scatter_length'][1] == 'log'
    assert 'scatter_length' not in spec['gauge']
    # curve → global, geomean gauge (multiplicative scale degeneracy)
    assert g['abs_dev'][0] == (N_CONTROL,)
    assert spec['gauge']['abs_dev'] == 'geomean'
    # per-PMT qe → Schur, geomean gauge (positive)
    assert p['qe_corrections'][1] == 'log'
    assert spec['gauge']['qe_corrections'] == 'geomean'
    # per-PMT t0 → Schur, linear, mean gauge (signed)
    assert p['t0'][1] == 'linear'
    assert spec['gauge']['t0'] == 'mean'


def test_classify_infers_n_sensors():
    dp = _dp()
    theta, _ = partition(dp, ['qe_corrections'])
    assert classify(theta)['n_sensors'] == NS
