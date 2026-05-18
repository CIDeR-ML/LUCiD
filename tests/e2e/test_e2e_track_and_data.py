"""End-to-end tests for track (SIREN) and data simulation modes.

Tests cover: track mode with SIREN, data mode with old ROOT file
(no wavelengths), data mode with new ROOT file (with wavelengths).
"""
import os
import sys

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from tests.e2e.conftest import (
    SK_LIKE_GEOM, SK_LIKE_PHYS, OLD_ROOT, NEW_ROOT,
    NPHOT, K_VAL, KEY, report,
)


# ===================================================================
# 5. Track mode (SIREN) with wavelength
# ===================================================================
def test_5_track_siren():
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import ParticleParams

    try:
        sim = setup_event_simulator(
            SK_LIKE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
            is_calibration=False, is_data=False,
            physics_config=SK_LIKE_PHYS,
            default_detector_params=True,
            wavelength_mode=True,
        )

        pp = ParticleParams(
            energy=jnp.array(500.0),
            position=jnp.zeros(3),
            theta=jnp.array(jnp.pi / 2),
            phi=jnp.array(0.0),
            t0=jnp.array(0.0),
        )

        # Track mode uses make_hits_likelihood which returns 4 values:
        # (log_w, flat_times, flat_indices, total_charge)
        result = sim(pp, KEY)
        n_outputs = len(result)

        if n_outputs == 4:
            log_w, flat_times, flat_indices, total_charge = result
            ok = bool(jnp.all(jnp.isfinite(total_charge)) and jnp.sum(total_charge) > 0)
            report("5_track_siren_wavelength",
                   ok,
                   f"total_charge_sum={float(jnp.sum(total_charge)):.2f}, "
                   f"log_w_shape={log_w.shape}, total_charge_shape={total_charge.shape}, "
                   f"n_outputs={n_outputs} (likelihood mode: log_w, flat_times, flat_indices, total_charge)")
        elif n_outputs == 2:
            charges, times = result
            ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
            report("5_track_siren_wavelength",
                   ok,
                   f"total_charge={float(jnp.sum(charges)):.2f}, "
                   f"charges_shape={charges.shape}, times_shape={times.shape}")
        else:
            report("5_track_siren_wavelength", False,
                   f"Unexpected number of outputs: {n_outputs}")
    except Exception as e:
        report("5_track_siren_wavelength", False, f"Exception: {e}")


# ===================================================================
# 6. Data mode with old ROOT file (no wavelengths)
# ===================================================================
def test_6_data_old_root():
    if not os.path.exists(OLD_ROOT):
        report("6a_old_root_no_wavelengths", False, f"File not found: {OLD_ROOT}")
        report("6b_old_root_simulation", False, "Skipped (no file)")
        return

    # Old ROOT file uses OpticalPhotons tree (PhotonSim format), not v_photon
    from lucid.sources.event_io import read_photon_data_from_photonsim

    photon_data = read_photon_data_from_photonsim(OLD_ROOT, 0)
    has_wl = 'wavelengths' in photon_data
    report("6a_old_root_no_wavelengths",
           not has_wl,
           f"keys={list(photon_data.keys())}")

    # Now try running through simulator in data mode
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams, ParticleParams
    from lucid.geometry import generate_detector

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

    try:
        n_actual = photon_data['photon_origins'].shape[0]
        sim = setup_event_simulator(
            SK_LIKE_GEOM, n_photons=n_actual, temperature=None, K=K_VAL,
            is_data=True, is_calibration=False,
            wavelength_mode=True,
            default_detector_params=dp,
        )

        pp = ParticleParams.from_cartesian(
            energy=photon_data['energy'],
            position=[0., 0., 0.],
            direction=[0., 0., 1.],
        )

        # photon_times key is present from read_photon_data_from_photonsim
        ptimes = photon_data.get('photon_times', jnp.zeros(n_actual))
        sim_data = {
            'photon_origins': photon_data['photon_origins'],
            'photon_directions': photon_data['photon_directions'],
            'photon_times': ptimes,
            'N': n_actual,
            'apply_rotation': False,
            'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
            'rotation_angle': 0.0,
        }

        charges, times_true, times_reco = sim(pp, KEY, sim_data)
        ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
        report("6b_old_root_simulation",
               ok,
               f"total_charge={float(jnp.sum(charges)):.2f}, n_photons={n_actual} (falls back to Cherenkov sampling)")
    except Exception as e:
        report("6b_old_root_simulation", False, f"Exception: {e}")


# ===================================================================
# 7. Data mode with new ROOT file (with wavelengths)
# ===================================================================
def test_7_data_new_root():
    if not os.path.exists(NEW_ROOT):
        report("7a_new_root_has_wavelengths", False, f"File not found: {NEW_ROOT}")
        report("7b_new_root_simulation", False, "Skipped (no file)")
        return

    from lucid.sources.event_io import read_photon_data_from_photonsim

    photon_data = read_photon_data_from_photonsim(NEW_ROOT, 0)
    has_wl = 'wavelengths' in photon_data
    report("7a_new_root_has_wavelengths",
           has_wl,
           f"keys={list(photon_data.keys())}" +
           (f", wavelengths range=[{float(photon_data['wavelengths'].min()):.1f}, {float(photon_data['wavelengths'].max()):.1f}]"
            if has_wl else ""))

    # Run through simulator
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams, ParticleParams
    from lucid.geometry import generate_detector

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

    try:
        n_actual = photon_data['photon_origins'].shape[0]
        sim = setup_event_simulator(
            SK_LIKE_GEOM, n_photons=n_actual, temperature=None, K=K_VAL,
            is_data=True, is_calibration=False,
            physics_config=SK_LIKE_PHYS,
            wavelength_mode=True,
            default_detector_params=dp,
        )

        pp = ParticleParams.from_cartesian(
            energy=photon_data['energy'],
            position=[0., 0., 0.],
            direction=[0., 0., 1.],
        )

        sim_data = {
            'photon_origins': photon_data['photon_origins'],
            'photon_directions': photon_data['photon_directions'],
            'photon_times': photon_data['photon_times'],
            'N': n_actual,
            'apply_rotation': False,
            'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
            'rotation_angle': 0.0,
        }
        if has_wl:
            sim_data['wavelengths'] = photon_data['wavelengths']

        charges, times_true, times_reco = sim(pp, KEY, sim_data)
        ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
        report("7b_new_root_simulation",
               ok,
               f"total_charge={float(jnp.sum(charges)):.2f}, n_photons={n_actual} (wavelengths used)")
    except Exception as e:
        report("7b_new_root_simulation", False, f"Exception: {e}")
