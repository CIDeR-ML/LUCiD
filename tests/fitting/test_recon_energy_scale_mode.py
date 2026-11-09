"""A model recipe's energy scale is one of two names, anything else is refused by
``ReconModel.build`` before it builds anything, and each scale routes the energy derivative
through itself: the emitter's ``nphot_fn``, or (``nphot_fn=None``) the simulator's own total.

Before, any string other than 'simtotal' ran the 'nphot' branch, so a typo in a recipe silently
changed the estimator (or failed at trace time with an unrelated error).
"""
import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.fitting import MODEL_RECIPE, ReconModel
from lucid.simulation.simulator import Simulator


@pytest.mark.parametrize('mode', ['nphto', 'NPHOT', 'sim_total', ''])
def test_an_unknown_mode_is_refused(mode):
    recipe = dataclasses.replace(MODEL_RECIPE, energy_scale=mode)
    with pytest.raises(ValueError, match='energy_scale'):
        ReconModel.build('no-geometry.json', 'no-physics.json', 'muon', sigma=2.1, recipe=recipe,
                         grid={})


# A simulator-free per-photon predictor: one photon per sensor, charge ``A(E) * shape_i(x)`` with
# ``A = 50 (E/1 GeV)^P_SIM``, a Gaussian falloff with distance, and arrival time distance / v.
ND = 10
P_SIM = 0.8                                  # the "simulated" total scales as E^0.8 ...
P_NPHOT = 1.1                                # ... the analytic nphot curve as E^1.1
_PMT = np.stack([4.0 * np.cos(2 * np.pi * np.arange(ND) / ND),
                 4.0 * np.sin(2 * np.pi * np.arange(ND) / ND),
                 np.linspace(-1.0, 1.0, ND)], axis=1)


def _toy_total(energy, position):
    d2 = jnp.sum((jnp.asarray(_PMT) - position) ** 2, axis=1)
    return 50.0 * (energy / 1000.0) ** P_SIM * jnp.exp(-d2 / 20.0)


def _toy_pred(track, key):
    tot = _toy_total(track.energy, track.position)
    dist = jnp.linalg.norm(jnp.asarray(_PMT) - track.position, axis=1)
    return jnp.log(tot), dist / 0.2, jnp.arange(ND), tot


_TOY = Simulator(_toy_pred, types.SimpleNamespace(num_sensors=ND), None, None, None)


def _nphot(e):
    return 30.0 * e ** P_NPHOT


T9 = jnp.array([1000.0, 0.3, -0.2, 0.1, 0.6, 0.8, 0.0, 1.0, 0.0])


@pytest.mark.parametrize('mode, dlogA_dE', [('simtotal', P_SIM / 1000.0),
                                            ('nphot', P_NPHOT / 1000.0)])
def test_each_mode_routes_the_energy_gradient_through_its_own_scale(mode, dlogA_dE):
    """mu = A(E) * p_i(geometry), with the energy derivative taken only through the scale A.

    'simtotal' takes A from the simulator's own total, so d mu_i / dE = mu_i * d log(sum tot)/dE
    = mu_i * 0.8 / E for this predictor. 'nphot' takes it from nphot_fn, so d mu_i / dE =
    mu_i * nphot'(E) / nphot(E) = mu_i * 1.1 / E. In both modes the value is the simulator's
    total (the scale ratio is 1), the geometry derivative is the shape's own, and the time NLL
    carries no energy derivative. A mode that ran the other branch, or a scale that lost its
    energy gradient, fails the energy column.
    """
    model = ReconModel(_TOY, sigma=2.1, nphot_fn=_nphot if mode == 'nphot' else None)
    key = jax.random.PRNGKey(0)
    tot = np.asarray(_toy_total(T9[0], T9[1:4]))
    ot = jnp.asarray(np.linalg.norm(_PMT - np.asarray(T9[1:4]), axis=1) / 0.2 + 0.3)
    oc = jnp.asarray(np.round(tot) + 1.0)
    mu, _ = model.perpmt(T9, oc, ot, key)
    np.testing.assert_allclose(np.asarray(mu), tot, rtol=1e-6)

    jm, jl = jax.jacfwd(lambda t: model.perpmt(t, oc, ot, key))(T9)
    np.testing.assert_allclose(np.asarray(jm[:, 0]), tot * dlogA_dE, rtol=1e-4)
    dtot_dx = np.asarray(jax.jacfwd(lambda x: _toy_total(T9[0], x))(T9[1:4]))
    np.testing.assert_allclose(np.asarray(jm[:, 1:4]), dtot_dx, rtol=1e-4, atol=1e-9)
    np.testing.assert_array_equal(np.asarray(jl[:, 0]), 0.0)
    assert np.abs(np.asarray(jl[:, 1:4])).max() > 0       # the time term does see the vertex
