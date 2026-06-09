"""Default optax fitter + loss terms — mechanism tests on a cheap analytic forward.

Uses a differentiable stand-in for the photon forward so grad flows to every leaf without
building a detector. Two sources with different per-sensor geometry are used wherever
globals must be identifiable: with a single source the 12 free per-PMT factors absorb any
per-sensor pattern (the real "source diversity breaks degeneracy" effect), so the globals
are genuinely unidentifiable — that's physics, not a fitter bug.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lucid.detector_params import DetectorParams
from lucid.fitting import (make_loss, fit, charge, charge_var, first_arrival,
                           gauge_mean_log, gauge_mean)

NS = 12
_GEOMS = {'a': jnp.linspace(10., 200., NS), 'b': jnp.linspace(200., 10., NS)}


def _dp(scatter=70., absn=60., qe=0.07, k=None):
    return DetectorParams.from_flat(
        scatter_length=scatter, absorption_length=absn, qe=qe, g=0.9,
        mie_scatter_length=3000., wall_reflection_rate=0.2,
        qe_corrections=jnp.ones(NS) if k is None else jnp.asarray(k))


def _sim(src, dp, key):
    """(mean, var, time) like moments mode. scatter_length floored (as the real propagator
    floors lengths) and enters a strongly-varying across-sensor shape so it's identifiable."""
    geom = _GEOMS[src]
    s = jnp.maximum(dp.scattering.scatter_length, 1.0)
    shape = s / (s + geom)
    mean = dp.response.qe * dp.absorption.absorption_length * shape * dp.per_pmt.qe_corrections
    return mean, mean * (1.0 + 0.35 ** 2), geom * 0.01 + dp.per_pmt.t0


def test_loss_terms_zero_at_truth():
    pred = _sim('a', _dp(), None)
    assert float(charge(pred, pred)) == pytest.approx(0.0, abs=1e-6)
    assert float(charge_var(pred, pred)) == pytest.approx(0.0, abs=1e-6)
    assert float(first_arrival(pred, pred)) == pytest.approx(0.0, abs=1e-6)


def test_fit_recovers_globals():
    """Two sources break the per-PMT-k degeneracy → the global optics are recovered."""
    srcs = ['a', 'b']
    dp_true = _dp(scatter=70., absn=60., qe=0.07)
    obs = [_sim(s, dp_true, None) for s in srcs]
    loss = make_loss(_sim, srcs, obs, terms=[charge])

    dp_hat = fit(loss, _dp(50., 90., 0.12), steps=6000, lr=1e-2, project=gauge_mean_log)

    # qe·absorption is the identifiable amplitude; scatter is the across-sensor shape
    assert float(dp_hat.response.qe * dp_hat.absorption.absorption_length) == pytest.approx(4.2, rel=0.05)
    assert float(dp_hat.scattering.scatter_length) == pytest.approx(70., rel=0.06)


def test_fit_recovers_per_pmt_k():
    srcs = ['a', 'b']
    rng = np.random.default_rng(0)
    k_true = 1.0 + 0.15 * rng.standard_normal(NS)
    k_true = k_true / np.exp(np.mean(np.log(k_true)))          # gauge truth the same way
    obs = [_sim(s, _dp(k=k_true), None) for s in srcs]
    loss = make_loss(_sim, srcs, obs, terms=[charge])

    dp_hat = fit(loss, _dp(), steps=4000, lr=5e-3, project=gauge_mean_log)
    k_hat = np.asarray(dp_hat.per_pmt.qe_corrections)
    assert np.corrcoef(k_hat, k_true)[0, 1] > 0.98
    np.testing.assert_allclose(k_hat, k_true, atol=0.03)


def test_gauges():
    rng = np.random.default_rng(1)
    dp = _dp(k=1.0 + 0.3 * rng.standard_normal(NS))
    dp = dp._replace(per_pmt=dp.per_pmt._replace(t0=jnp.asarray(rng.standard_normal(NS))))
    g = gauge_mean_log(dp)
    assert float(jnp.mean(jnp.log(g.per_pmt.qe_corrections))) == pytest.approx(0.0, abs=1e-6)
    g2 = gauge_mean(dp, 't0')
    assert float(jnp.mean(g2.per_pmt.t0)) == pytest.approx(0.0, abs=1e-6)


def test_multi_transform_freezes_subset():
    """optax.multi_transform + set_to_zero freezes a subset — the staging mechanism."""
    srcs = ['a', 'b']
    obs = [_sim(s, _dp(scatter=70.), None) for s in srcs]
    loss = make_loss(_sim, srcs, obs, terms=[charge])

    # label every leaf 'freeze' except scatter_length → 'train'
    labels = jax.tree.map(lambda _: 'freeze', _dp())
    labels = labels._replace(scattering=labels.scattering._replace(scatter_length='train'))
    opt = optax.multi_transform(
        {'train': optax.adam(5e-2), 'freeze': optax.set_to_zero()}, labels)

    # freeze absn/qe at TRUTH and start scatter off → scatter must recover, frozen stay put
    dp0 = _dp(scatter=50., absn=60., qe=0.07)
    dp_hat = fit(loss, dp0, steps=2000, optimizer=opt)   # lr lives in opt; fit's lr ignored
    assert float(dp_hat.absorption.absorption_length) == pytest.approx(60., abs=1e-5)  # frozen
    assert float(dp_hat.response.qe) == pytest.approx(0.07, abs=1e-6)                   # frozen
    assert float(dp_hat.scattering.scatter_length) == pytest.approx(70., rel=0.05)      # trained
