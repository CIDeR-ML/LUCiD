"""End-to-end test for SuperK simulation mode."""
import os
import sys

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from tests.e2e.conftest import (
    SK_GEOM, SK_PHYS,
    NPHOT, K_VAL, KEY, report,
)


# ===================================================================
# 8. SuperK mode
# ===================================================================
def test_8_superk():
    try:
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source
        from lucid.geometry import generate_detector

        det = generate_detector(SK_GEOM)
        n_sensors = len(det.all_points)

        sim = setup_event_simulator(
            SK_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
            is_calibration=True,
            detector_type='Cylinder',
            physics_config=SK_PHYS,
            default_detector_params=True,
            wavelength_mode=True,
        )

        src = laser_source(position=[0., 0., 10.0], intensity=1e8, wavelength=405.0)
        charges, _ = sim(src, KEY)

        ok_sensors = (n_sensors == 11096)
        ok_charges = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)

        report("8_superk_mode",
               ok_sensors and ok_charges,
               f"n_sensors={n_sensors} (expect 11096), "
               f"total_charge={float(jnp.sum(charges)):.2f}, "
               f"charges_shape={charges.shape}")
    except Exception as e:
        report("8_superk_mode", False, f"Exception: {e}")
