"""End-to-end tests for all simulation modes with wavelength integration.

Tests cover: calibration (scalar, wavelength, manual, scalar-forced),
track (SIREN), data (old ROOT, new ROOT), SuperK, gradients, edge cases.
"""
import os
import sys
import traceback
import time

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

# ---- Paths ---------------------------------------------------------------
CONFIG = os.path.join(BASE, "config")

WCTE_GEOM = os.path.join(CONFIG, "WCTE_geom_config.json")
WCTE_PHYS = os.path.join(CONFIG, "WCTE_physics_config.json")
SK_LIKE_GEOM = os.path.join(CONFIG, "SK_like_geom_config.json")
SK_LIKE_PHYS = os.path.join(CONFIG, "SK_like_physics_config.json")
SK_GEOM = os.path.join(CONFIG, "SK_geom_config.json")
SK_PHYS = os.path.join(CONFIG, "SK_physics_config.json")
JUNO_PHYS = os.path.join(CONFIG, "JUNO_physics_config.json")

OLD_ROOT = os.path.join(BASE, "data", "water", "muon",
                        "muon_gun_1050_MeV_100_events_fixed_energy.root")
NEW_ROOT = os.path.join(BASE, "..", "PhotonSim", "build",
                        "test_wavelength_5_events.root")

NPHOT = 5_000          # small for speed
K_VAL = 3              # few bounces for speed
KEY = jax.random.PRNGKey(42)

results = {}


def report(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    results[name] = (passed, detail)
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))


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

        charges, times = sim(pp, KEY, sim_data)
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

        charges, times = sim(pp, KEY, sim_data)
        ok = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)
        report("7b_new_root_simulation",
               ok,
               f"total_charge={float(jnp.sum(charges)):.2f}, n_photons={n_actual} (wavelengths used)")
    except Exception as e:
        report("7b_new_root_simulation", False, f"Exception: {e}")


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
            detector_type='superk',
            physics_config=SK_PHYS,
            default_detector_params=True,
            wavelength_mode=True,
        )

        src = laser_source(position=[0., 0., 10.0], intensity=1e8, wavelength=405.0)
        charges, _ = sim(src, KEY)

        ok_sensors = (n_sensors == 11146)
        ok_charges = bool(jnp.all(jnp.isfinite(charges)) and jnp.sum(charges) > 0)

        report("8_superk_mode",
               ok_sensors and ok_charges,
               f"n_sensors={n_sensors} (expect 11146), "
               f"total_charge={float(jnp.sum(charges)):.2f}, "
               f"charges_shape={charges.shape}")
    except Exception as e:
        report("8_superk_mode", False, f"Exception: {e}")


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


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("LUCiD End-to-End Wavelength Integration Tests")
    print("=" * 70)
    print()

    all_tests = [
        ("1. Calibration scalar (WCTE)", test_1_calibration_scalar),
        ("2. Calibration wavelength (SK_like)", test_2_calibration_wavelength),
        ("3. Calibration manual DetectorParams", test_3_calibration_manual_dp),
        ("4. Calibration scalar forced", test_4_calibration_scalar_forced),
        ("5. Track mode (SIREN)", test_5_track_siren),
        ("6. Data mode old ROOT (no wavelengths)", test_6_data_old_root),
        ("7. Data mode new ROOT (with wavelengths)", test_7_data_new_root),
        ("8. SuperK mode", test_8_superk),
        ("9. Gradient flow", test_9_gradient_flow),
        ("10. Edge cases", test_10_edge_cases),
    ]

    for label, fn in all_tests:
        print(f"\n--- {label} ---")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            report(label + "_CRASH", False, f"Unhandled exception: {e}\n{traceback.format_exc()}")
        elapsed = time.time() - t0
        print(f"    ({elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for v in results.values() if v[0])
    n_fail = sum(1 for v in results.values() if not v[0])
    for name, (passed, detail) in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}")
    print(f"\n{n_pass} passed, {n_fail} failed out of {len(results)} tests")
    sys.exit(0 if n_fail == 0 else 1)
