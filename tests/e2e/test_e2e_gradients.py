"""End-to-end test for gradient flow through the simulation."""
import os
import sys

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from tests.e2e.conftest import (
    WCTE_GEOM,
    NPHOT, K_VAL, KEY, report,
)


# ===================================================================
# 9. Gradient flow
# ===================================================================
def test_9_gradient_flow():
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams
    from lucid.sources import laser_source
    from lucid.losses import WC_smooth_loss
    from lucid.geometry import generate_detector

    det = generate_detector(WCTE_GEOM)
    N = len(det.all_points)
    sp = jnp.array(det.all_points)

    dp = DetectorParams(
        scatter_length=jnp.array(50.0),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        absorption_length=jnp.array(150.0),
        qe=jnp.array(0.2),
        qe_corrections=jnp.ones(N),
    )

    # Build a reference simulator with baked-in params
    sim_ref = setup_event_simulator(
        WCTE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True,
        default_detector_params=dp,
        wavelength_mode=True,
    )

    # Build a trainable simulator (no baked-in params)
    sim = setup_event_simulator(
        WCTE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True,
        wavelength_mode=True,
    )

    src = laser_source(position=[0., 0., det.H / 2 - 0.1], intensity=1e8,
                       wavelength=405.0)
    true_data = jax.lax.stop_gradient(sim_ref(src, KEY))

    @jax.jit
    def loss_fn(dp_in):
        pred = sim(src, dp_in, KEY)
        return WC_smooth_loss(sp, *true_data, *pred,
                              lambda_poisson=1.0, lambda_time=0.0, tau=2.0)

    try:
        loss, grads = jax.value_and_grad(loss_fn)(dp)

        wall_grad_finite = bool(jnp.isfinite(grads.wall_reflection_rate))
        sensor_grad_finite = bool(jnp.isfinite(grads.sensor_reflection_rate))
        loss_finite = bool(jnp.isfinite(loss))
        all_grads_finite = all(bool(jnp.all(jnp.isfinite(g)))
                               for g in jax.tree.leaves(grads))

        report("9a_gradient_loss_finite",
               loss_finite,
               f"loss={float(loss):.6f}")
        report("9b_gradient_wall_reflection_finite",
               wall_grad_finite,
               f"grad_wall_reflection_rate={float(grads.wall_reflection_rate):.6e}")
        report("9c_gradient_sensor_reflection_finite",
               sensor_grad_finite,
               f"grad_sensor_reflection_rate={float(grads.sensor_reflection_rate):.6e}")
        report("9d_all_gradients_finite",
               all_grads_finite,
               f"grad fields: {[f for f in DetectorParams._fields]}")
    except Exception as e:
        report("9a_gradient_loss_finite", False, f"Exception: {e}")
        report("9b_gradient_wall_reflection_finite", False, "Skipped")
        report("9c_gradient_sensor_reflection_finite", False, "Skipped")
        report("9d_all_gradients_finite", False, "Skipped")
