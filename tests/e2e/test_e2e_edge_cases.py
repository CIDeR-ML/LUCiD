"""End-to-end tests for edge cases in wavelength simulation."""
import os
import sys

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from tests.e2e.conftest import (
    CONFIG, WCTE_GEOM, WCTE_PHYS, JUNO_PHYS,
    NPHOT, K_VAL, KEY, report,
)


# ===================================================================
# 10. Edge cases
# ===================================================================
def test_10_edge_cases():
    # 10a: wavelength_mode=True but physics_config has no qe_curve (JUNO)
    # JUNO has medium_model but no qe_curve field in its physics config
    from lucid.simulation import setup_event_simulator
    from lucid.sources import laser_source

    JUNO_GEOM = os.path.join(CONFIG, "JUNO_geom_config.json")
    try:
        sim = setup_event_simulator(
            JUNO_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
            is_calibration=True,
            detector_type='Sphere',
            physics_config=JUNO_PHYS,
            default_detector_params=True,
            wavelength_mode=True,
        )
        src = laser_source(position=[0., 0., 0.], intensity=1e8, wavelength=405.0)
        charges, _ = sim(src, KEY)
        ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
        report("10a_no_qe_curve_JUNO",
               ok,
               f"total_charge={float(jnp.sum(charges)):.2f} "
               f"(has medium_model but no qe_curve)")
    except Exception as e:
        report("10a_no_qe_curve_JUNO", False, f"Exception: {e}")

    # 10b: wavelength_mode=True but physics_config has no medium_model (WCTE)
    try:
        sim2 = setup_event_simulator(
            WCTE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
            is_calibration=True,
            physics_config=WCTE_PHYS,
            default_detector_params=True,
            wavelength_mode=True,
        )
        src2 = laser_source(position=[0., 0., 1.5], intensity=1e8, wavelength=405.0)
        charges2, _ = sim2(src2, KEY)
        ok2 = bool(jnp.all(jnp.isfinite(charges2)) and jnp.sum(charges2) > 0)
        report("10b_no_medium_model_WCTE",
               ok2,
               f"total_charge={float(jnp.sum(charges2)):.2f} "
               f"(falls back to legacy water.json)")
    except Exception as e:
        report("10b_no_medium_model_WCTE", False, f"Exception: {e}")

    # 10c: Extreme wavelengths (200nm and 800nm)
    from lucid.detector_params import DetectorParams
    from lucid.geometry import generate_detector

    det = generate_detector(WCTE_GEOM)
    N = len(det.all_points)
    dp = DetectorParams(
        scatter_length=jnp.array(50.0),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        absorption_length=jnp.array(150.0),
        qe=jnp.array(0.2),
        qe_corrections=jnp.ones(N),
    )

    try:
        sim3 = setup_event_simulator(
            WCTE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
            is_calibration=True,
            default_detector_params=dp,
            wavelength_mode=True,
        )

        src_200 = laser_source(position=[0., 0., 1.5], intensity=1e8, wavelength=200.0)
        charges_200, _ = sim3(src_200, KEY)
        ok_200 = bool(jnp.all(jnp.isfinite(charges_200)))

        src_800 = laser_source(position=[0., 0., 1.5], intensity=1e8, wavelength=800.0)
        charges_800, _ = sim3(src_800, KEY)
        ok_800 = bool(jnp.all(jnp.isfinite(charges_800)))

        report("10c_extreme_wavelength_200nm",
               ok_200,
               f"total_charge={float(jnp.sum(charges_200)):.4f}, all_finite={ok_200}")
        report("10d_extreme_wavelength_800nm",
               ok_800,
               f"total_charge={float(jnp.sum(charges_800)):.4f}, all_finite={ok_800}")
    except Exception as e:
        report("10c_extreme_wavelength_200nm", False, f"Exception: {e}")
        report("10d_extreme_wavelength_800nm", False, f"Exception: {e}")
