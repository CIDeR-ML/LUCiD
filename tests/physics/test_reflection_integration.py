"""Slow end-to-end test: the angular reflection model through setup_event_simulator.

Confirms reflection_model='angular' (Schlick blacksheet wall + multilayer-Fresnel
cathode sensor) runs a full calibration forward and is differentiable wrt the
angular reflection DetectorParams leaves.
"""
import os

import jax
import jax.numpy as jnp
import pytest
from tests.helpers.paths import REPO

pytestmark = pytest.mark.slow

GEOM = os.path.join(REPO, 'config', 'SK_like_geom_config.json')
GRID_KW = dict(n_cap=150, n_angular=250, n_height=150)


@pytest.fixture(scope="module")
def angular_setup():
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams
    det = generate_detector(GEOM)
    n = len(det.all_points)
    dp = DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.065, qe_corrections=jnp.ones(n))
    sim = setup_event_simulator(
        GEOM, 50_000, source='light_source', temperature=None, K=6,
        reflection_model='angular', reflection_wavelength=405.0, **GRID_KW)
    return det, dp, sim


def test_angular_forward_nonzero_finite(angular_setup):
    from lucid.sources import isotropic_source
    det, dp, sim = angular_setup
    source = isotropic_source(position=[0.0, 0.0, 0.0], intensity=50_000_000)
    charges, _ = sim(source, dp, jax.random.PRNGKey(1))
    assert float(jnp.sum(charges)) > 0
    assert int(jnp.sum(charges > 0)) > 100
    assert jnp.all(jnp.isfinite(charges))


def test_angular_gradient_wrt_reflection_params(angular_setup):
    from lucid.sources import isotropic_source
    from lucid.losses import WC_smooth_loss
    det, dp, sim = angular_setup
    source = isotropic_source(position=[0.0, 0.0, 0.0], intensity=50_000_000)
    sp = jnp.array(det.all_points)
    key = jax.random.PRNGKey(1)
    truth = jax.lax.stop_gradient(sim(source, dp, key))

    def loss(refl):
        dp2 = dp._replace(reflection=refl)
        pred = sim(source, dp2, key)
        return WC_smooth_loss(sp, *truth, *pred, lambda_poisson=1.0, lambda_time=0.0, tau=2.0)

    g = jax.grad(loss)(dp.reflection)
    # the angular magnitude leaves (wall_R0, cathode_nr) must carry finite gradient
    assert jnp.all(jnp.isfinite(jnp.array([g.wall_R0, g.wall_p, g.cathode_nr, g.cathode_nk])))

    # Off truth, the gradient must point back to it. With common random numbers the loss is
    # minimised at the truth's wall_R0, so d loss/d R0 is positive above it and negative below.
    r0 = float(dp.reflection.wall_R0)
    for delta in (+0.5 * r0, -0.5 * r0):
        g_off = jax.grad(loss)(dp.reflection._replace(wall_R0=jnp.asarray(r0 + delta)))
        assert float(g_off.wall_R0) * delta > 0.0, (delta, float(g_off.wall_R0))
