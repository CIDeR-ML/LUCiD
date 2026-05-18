"""End-to-end tests for calibration simulation modes.

Tests cover: calibration with scalar physics, wavelength physics,
manual DetectorParams, and forced scalar mode.
"""
import os
import sys

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from tests.e2e.conftest import (
    WCTE_GEOM, WCTE_PHYS, SK_LIKE_GEOM, SK_LIKE_PHYS,
    NPHOT, K_VAL, KEY, report,
)


# ===================================================================
# 1. Calibration mode, scalar physics (WCTE)
# ===================================================================
def test_1_calibration_scalar():
    from lucid.simulation import setup_event_simulator
    from lucid.sources import laser_source, isotropic_source
    from lucid.geometry import generate_detector

    det = generate_detector(WCTE_GEOM)
    n_sensors = len(det.all_points)
    H = det.H

    sim = setup_event_simulator(
        WCTE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True,
        physics_config=WCTE_PHYS,
        default_detector_params=True,
        wavelength_mode=False,
    )

    # Laser source
    src_laser = laser_source(position=[0., 0., H / 2 - 0.1], intensity=1e8)
    charges_l, times_l = sim(src_laser, KEY)

    ok_l = bool(jnp.all(jnp.isfinite(charges_l)) and jnp.sum(charges_l) > 0)
    report("1a_calibration_scalar_laser",
           ok_l,
           f"total_charge={float(jnp.sum(charges_l)):.2f}, n_sensors={n_sensors}")

    # Isotropic source
    src_iso = isotropic_source(position=[0., 0., 0.], intensity=1e8)
    charges_i, times_i = sim(src_iso, KEY)

    ok_i = bool(jnp.all(jnp.isfinite(charges_i)) and jnp.sum(charges_i) > 0)
    report("1b_calibration_scalar_isotropic",
           ok_i,
           f"total_charge={float(jnp.sum(charges_i)):.2f}")


# ===================================================================
# 2. Calibration mode, wavelength physics (SK_like)
# ===================================================================
def test_2_calibration_wavelength():
    from lucid.simulation import setup_event_simulator
    from lucid.sources import laser_source

    sim = setup_event_simulator(
        SK_LIKE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True,
        physics_config=SK_LIKE_PHYS,
        default_detector_params=True,
        wavelength_mode=True,
    )

    # Laser with explicit wavelength=405 (monochromatic)
    src_wl = laser_source(position=[0., 0., 10.0], intensity=1e8, wavelength=405.0)
    charges_wl, _ = sim(src_wl, KEY)

    ok_wl = bool(jnp.all(jnp.isfinite(charges_wl)) and jnp.sum(charges_wl) > 0)
    report("2a_calibration_wavelength_laser_405nm",
           ok_wl,
           f"total_charge={float(jnp.sum(charges_wl)):.2f}")

    # Laser without wavelength (Cherenkov sampling)
    src_no_wl = laser_source(position=[0., 0., 10.0], intensity=1e8)
    charges_chk, _ = sim(src_no_wl, KEY)

    ok_chk = bool(jnp.all(jnp.isfinite(charges_chk)) and jnp.sum(charges_chk) > 0)
    report("2b_calibration_wavelength_laser_cherenkov",
           ok_chk,
           f"total_charge={float(jnp.sum(charges_chk)):.2f}")

    # They should differ (mono vs Cherenkov spectrum)
    diff = float(jnp.sum(jnp.abs(charges_wl - charges_chk)))
    report("2c_wavelength_vs_cherenkov_differ",
           diff > 0,
           f"abs_diff={diff:.4f}")


# ===================================================================
# 3. Calibration mode, no physics config (manual DetectorParams)
# ===================================================================
def test_3_calibration_manual_dp():
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams
    from lucid.geometry import generate_detector
    from lucid.sources import laser_source

    det = generate_detector(SK_LIKE_GEOM)
    N = len(det.all_points)

    dp = DetectorParams(
        scatter_length=jnp.array(50.0),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        absorption_length=jnp.array(150.0),
        qe=jnp.array(0.2),
        qe_corrections=jnp.ones(N),
    )

    sim = setup_event_simulator(
        SK_LIKE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True,
        wavelength_mode=True,
        default_detector_params=dp,
    )

    src = laser_source(position=[0., 0., 10.0], intensity=1e8, wavelength=405.0)
    charges, _ = sim(src, KEY)

    ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
    report("3_calibration_manual_dp_wavelength",
           ok,
           f"total_charge={float(jnp.sum(charges)):.2f} (falls back to legacy water.json, no QE curve)")


# ===================================================================
# 4. Calibration mode, wavelength_mode=False
# ===================================================================
def test_4_calibration_scalar_forced():
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams
    from lucid.geometry import generate_detector
    from lucid.sources import laser_source

    det = generate_detector(SK_LIKE_GEOM)
    N = len(det.all_points)

    dp = DetectorParams(
        scatter_length=jnp.array(50.0),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        absorption_length=jnp.array(150.0),
        qe=jnp.array(0.2),
        qe_corrections=jnp.ones(N),
    )

    sim = setup_event_simulator(
        SK_LIKE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True,
        wavelength_mode=False,
        default_detector_params=dp,
    )

    src = laser_source(position=[0., 0., 10.0], intensity=1e8, wavelength=405.0)
    charges, _ = sim(src, KEY)

    ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
    report("4_calibration_scalar_forced",
           ok,
           f"total_charge={float(jnp.sum(charges)):.2f} (scalar scatter_length=50 used)")
