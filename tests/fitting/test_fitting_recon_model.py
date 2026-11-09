"""ReconModel on a differentiable per-photon stub: the Fisher metric, the energy routing, the t0 sign.

The real predictor is a SIREN-driven simulator, too slow for the default lane. The stub below has
the same contract (``pred(track, key) -> (log_w, flat_times, flat_indices, total_charge)``): a
fixed ring of sensors, a Cherenkov-like angular profile around the track direction, a 1/r^2
falloff, arrival times from the distance at c/n, and a small key-dependent jitter so that the
``nkeys`` average is not trivial. Nothing in it reads ReconModel's code.

Three independent truths are checked:

* ``fisher_ad`` is the textbook metric ``J_mu^T diag(1/mu) J_mu + J_l^T J_l`` of the key-averaged
  model, with both Jacobians taken by central finite differences of ``perpmt`` instead of AD.
* Energy is routed through the total charge only. For ``mu_i = A(E, g) p_i(g)`` the Poisson
  charge term gives ``dL/dE = (d log A / dE) (sum mu - sum Q)`` exactly, and the time term gives
  nothing. ``d log A / dE`` comes from finite differences of the stub's own total.
* The time term reads ``t_obs - t0``, through the predicted times (which start at the track's
  t0): shifting the observed times and t0 together leaves it unchanged.
"""
import types

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.fitting import ReconModel, SCALE9, track_from_vec9, vec9_from_track
from lucid.simulation.simulator import Simulator

ND, P = 24, 6                                   # sensors, photons per sensor
_rng = np.random.default_rng(11)
_v = _rng.standard_normal((ND, 3))
POS = jnp.asarray(3.0 * _v / np.linalg.norm(_v, axis=1, keepdims=True))   # sensors on r = 3 m
SPREAD = jnp.asarray(np.linspace(-1.5, 1.5, P))                             # ns, intra-sensor
IDX = jnp.repeat(jnp.arange(ND), P)
TRUTH = vec9_from_track(800.0, [0.3, -0.2, 0.4], [0.3, 0.5, 0.81], t0=1.0)


def _direction(track):
    st, ct = jnp.sin(track.theta), jnp.cos(track.theta)
    return jnp.stack([st * jnp.cos(track.phi), st * jnp.sin(track.phi), ct])


def make_pred(shape_depends_on_energy):
    """A stub per-photon predictor. With ``shape_depends_on_energy`` the cone opens with energy
    and the arrival times move with it, so a model that leaks energy gradient through the shape
    is visibly different from one that routes it through the total."""

    def pred(track, key):
        E = track.energy
        r = POS - track.position
        dist = jnp.linalg.norm(r, axis=1)
        cosang = (r / dist[:, None]) @ _direction(track)
        c0 = 0.72 + (0.08 * (E / 1000.0) if shape_depends_on_energy else 0.0)
        prof = jnp.exp(-(cosang - c0) ** 2 / 0.08) + 0.05
        amp = 40.0 * (E / 1000.0) ** 1.3                      # nonlinear in E on purpose
        mean = amp * prof / dist ** 2                         # (ND,) expected charge
        k1, k2 = jax.random.split(key)
        jw = 1.0 + 0.03 * jax.random.normal(k1, (ND * P,))
        jt = 0.05 * jax.random.normal(k2, (ND * P,))
        w = (mean / P)[IDX] * jw
        tof = dist / 0.2225                                   # ns, c/n with n = 1.348
        if shape_depends_on_energy:
            tof = tof + 0.3 * (E / 1000.0)
        ft = track.t0 + tof[IDX] + SPREAD[jnp.arange(ND * P) % P] + jt
        tot = jax.ops.segment_sum(w, IDX, num_segments=ND)
        return jnp.log(w), ft, IDX, tot

    return Simulator(pred, types.SimpleNamespace(num_sensors=ND), None, None, None)


def _observed(model, key=jax.random.PRNGKey(99)):
    """Noise-free observables at TRUTH: counts = the model mean, times = the earliest photon."""
    lw, ft, fi, tot = model.pred(track_from_vec9(jnp.asarray(TRUTH)), key)
    return jnp.round(tot), jax.ops.segment_min(ft, fi, num_segments=ND)


KEYS = [jax.random.PRNGKey(s) for s in range(4)]


def test_fisher_ad_is_the_poisson_plus_time_metric_by_finite_differences():
    """fisher_ad against central FD of the key-averaged (mu, tnll), h = 0.01 SCALE9.

    The energy-independent shape is required here: with it, routing energy through the total
    changes no derivative, so FD of the forward IS the Jacobian the metric should use. Measured
    agreement 4e-5 of the largest entry (float32 FD, h = 0.01 SCALE9); tolerance 2e-3 of it. The
    charge block is 83 % of the largest entry and the time block 38 %, so not averaging mu over the
    keys (a factor nkeys = 4 on the charge block) or a missing block fails by two orders.
    """
    m = ReconModel(make_pred(False), sigma=2.1, nphot_fn=None)
    oc, ot = _observed(m)
    t9 = jnp.asarray(TRUTH + np.array([20., .05, -.03, .04, .01, -.01, .01, .01, .1]))

    def avg(t):
        mu = np.mean([np.asarray(m.perpmt(t, oc, ot, k)[0], float) for k in KEYS], 0)
        tl = np.mean([np.asarray(m.perpmt(t, oc, ot, k)[1], float) for k in KEYS], 0)
        return mu, tl

    Jm = np.zeros((ND, 9)); Jl = np.zeros((ND, 9))
    for j in range(9):
        h = 0.01 * SCALE9[j]
        up, dn = avg(t9.at[j].add(h)), avg(t9.at[j].add(-h))
        Jm[:, j] = (up[0] - dn[0]) / (2 * h)
        Jl[:, j] = (up[1] - dn[1]) / (2 * h)
    mu0 = avg(t9)[0]
    want = (Jm / mu0[:, None]).T @ Jm + Jl.T @ Jl
    got = m.fisher_ad(t9, oc, ot, KEYS)
    # compare in SCALE9 units so every entry is O(1)-comparable
    S = np.outer(SCALE9, SCALE9)
    np.testing.assert_allclose(got * S, want * S, rtol=2e-3, atol=2e-3 * np.abs(want * S).max())


@pytest.mark.parametrize('mode', ['simtotal', 'nphot'])
def test_the_energy_gradient_flows_through_the_total_charge_only(mode):
    """dL/dE = (d log A / dE) (sum mu - sum Q), from the charge term alone.

    A is the stub's total in 'simtotal' mode and the nphot curve in 'nphot' mode. The stub's
    shape and times DO depend on energy here, so a gradient leaking through the shape (or through
    the time term) breaks the identity. Data are the truth's counts scaled by 0.9 and with the
    shape permuted, so sum mu - sum Q is not zero and the shapes disagree.
    """
    nphot = lambda e: 2e4 * (e / 1000.0) ** 1.1 + 1e3
    m = ReconModel(make_pred(True), sigma=2.1, nphot_fn=nphot if mode == 'nphot' else None)
    oc0, ot = _observed(m)
    oc = jnp.round(0.9 * oc0[np.random.default_rng(5).permutation(ND)])
    t9 = jnp.asarray(TRUTH)
    key = KEYS[1]

    def logA(E):
        if mode == 'nphot':
            return np.log(nphot(E))
        return np.log(float(jnp.sum(m.pred(track_from_vec9(t9.at[0].set(E)), key)[3])))

    h = 1.0                                             # MeV; logA is smooth on this scale
    dlogA = (logA(TRUTH[0] + h) - logA(TRUTH[0] - h)) / (2 * h)
    mu = np.asarray(m.perpmt(t9, oc, ot, key)[0], float)
    want = dlogA * (mu.sum() - float(np.sum(oc)))
    got = float(m.grad(t9, oc, ot, key)[0])
    # measured agreement 3e-6 (simtotal) and 7e-7 (nphot) relative. The un-routed gradient, which
    # also sees the shape's energy dependence, differs by 3.5 % (simtotal) and 22 % (nphot).
    assert abs(want) > 1e-3
    np.testing.assert_allclose(got, want, rtol=1e-3)


def test_matched_total_charge_gives_zero_energy_gradient_whatever_the_shape():
    """The flip side: when sum Q = sum mu the energy gradient vanishes even if the shape is wrong.

    That is the property the routing exists for (a shape misspecification must not pull energy).
    """
    m = ReconModel(make_pred(True), sigma=2.1, nphot_fn=None)
    t9 = jnp.asarray(TRUTH)
    key = KEYS[2]
    mu = np.asarray(m.perpmt(t9, jnp.ones(ND), jnp.zeros(ND), key)[0], float)
    perm = np.random.default_rng(6).permutation(ND)
    oc = jnp.asarray(mu[perm])                          # same total, wrong shape (not integer;
    _, ot = _observed(m)                                # the Poisson term does not need it)
    g = np.asarray(m.grad(t9, oc, ot, key), float)
    # in SCALE9 units: measured |g_E| / max|g_geometry| = 1.3e-8. The un-routed gradient gives
    # 1.4e-2 on this data.
    gs = g * SCALE9
    assert abs(gs[0]) < 1e-4 * np.abs(gs[1:8]).max(), gs


def test_the_time_term_depends_on_observed_minus_t0():
    """Shifting the observed times and t0 by the same delta leaves every per-PMT term unchanged,
    and shifting t0 alone does change the time term (so the identity is not vacuous)."""
    m = ReconModel(make_pred(False), sigma=2.1, nphot_fn=None)
    oc, ot = _observed(m)
    t9 = jnp.asarray(TRUTH)
    d = 1.7
    mu0, tl0 = m.perpmt(t9, oc, ot, KEYS[0])
    mu1, tl1 = m.perpmt(t9.at[8].add(d), oc, ot + d, KEYS[0])
    np.testing.assert_allclose(np.asarray(mu1), np.asarray(mu0), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(tl1), np.asarray(tl0), rtol=1e-4, atol=1e-4)
    _, tl2 = m.perpmt(t9.at[8].add(d), oc, ot, KEYS[0])
    assert float(jnp.sum(tl2)) - float(jnp.sum(tl0)) > 0.5        # measured 1.69 nats
