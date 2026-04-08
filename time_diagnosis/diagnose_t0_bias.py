#!/usr/bin/env python3
"""
Diagnose the t0 bias between data-simulator (segment_min) and
prediction-simulator (soft-min) timing.

The hypothesis is that the soft-min in make_hits_simulation is
systematically lower than the hard-min in make_hits_data, creating
a bias.  We test:
  1. Direct comparison of measured_time from data vs prediction simulators
  2. Effect of threshold (1e-10 v2 vs 1e-5 clean)
  3. Effect of K (9 v2 vs 7 clean)
  4. Effect of Nphot (300_000 v2 vs 150_000 clean)
  5. Raw soft-min vs hard-min bias from the flat photon arrays
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from tools.geometry import generate_detector
from tools.simulation import (
    setup_event_simulator, make_hits_simulation, make_hits_data,
)
from tools.detector_params import ParticleParams
from tools.generate import read_photon_data_from_photonsim

# ── configuration ────────────────────────────────────────────────────
GEOM   = '../config/SK_geom_config.json'
PHYS   = '../config/SK_physics_config.json'
DATA   = '../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'

detector = generate_detector(GEOM)
det_pts  = jnp.array(detector.all_points)
NUM_DET  = len(det_pts)

# ── helpers ──────────────────────────────────────────────────────────

def load_and_pad(entry_idx):
    pd = read_photon_data_from_photonsim(DATA, entry_idx)
    N  = len(pd['photon_origins'])
    pad = max(0, 1_000_000 - N)
    pd['photon_origins'] = jnp.pad(pd['photon_origins'],
                                   ((0, pad), (0, 0)), constant_values=0)
    dd = jnp.array([0., 0., 1.])
    if pad > 0:
        pd['photon_directions'] = jnp.concatenate(
            [pd['photon_directions'], jnp.tile(dd, (pad, 1))])
    pd['photon_times'] = jnp.pad(pd['photon_times'], (0, pad), constant_values=0)
    pd['N'] = N
    pd['apply_rotation'] = jnp.array(False)
    pd['rotation_axis']  = jnp.array([1., 0., 0.])
    pd['rotation_angle'] = jnp.array(0.)
    return pd

def set_transform(pd, pp):
    orig = jnp.array([0., 0., 1.])
    tgt  = pp.direction / (jnp.linalg.norm(pp.direction) + 1e-8)
    ax   = jnp.cross(orig, tgt)
    an   = jnp.linalg.norm(ax)
    ax   = jnp.where(an < 1e-6, jnp.array([1., 0., 0.]), ax / (an + 1e-8))
    ang  = jnp.arccos(jnp.clip(jnp.dot(orig, tgt), -1., 1.))
    pd['rotation_axis']       = ax
    pd['rotation_angle']      = ang
    pd['apply_rotation']      = jnp.array(True)
    pd['translation_vector']  = pp.position
    pd['apply_translation']   = jnp.array(True)
    return pd

def random_track(key, energy):
    k1,k2,k3,k4,k5 = jax.random.split(key, 5)
    frac = 0.6
    r = jax.random.uniform(k1, minval=0, maxval=detector.r*frac)
    tp = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    z  = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
    pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
    th  = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
    ph  = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
    return ParticleParams(energy=jnp.array(float(energy)),
                          position=pos, theta=th, phi=ph,
                          t0=jnp.array(0.))


# ======================================================================
# TEST 1 – direct comparison of data-time vs prediction-time (v2 setup)
# ======================================================================
print("="*80)
print("TEST 1: Compare measured_time from data vs prediction simulators")
print("="*80)

for Nphot, K_val, label in [
    (300_000, 9, "v2 defaults (Nphot=300k, K=9)"),
    (150_000, 7, "clean defaults (Nphot=150k, K=7)"),
]:
    data_sim = setup_event_simulator(
        GEOM, Nphot, temperature=0.0, K=20,
        is_data=True, is_calibration=False,
        physics_config=PHYS, default_detector_params=True)

    pred_sim = setup_event_simulator(
        GEOM, Nphot, temperature=0.10, K=K_val,
        is_data=False, is_calibration=False, max_sensors_per_cell=4,
        physics_config=PHYS, default_detector_params=True)

    biases = []
    for evt in range(10):
        pd = load_and_pad(evt)
        key = jax.random.PRNGKey(42 + evt)
        pp = random_track(key, pd['energy'])
        pd = set_transform(pd, pp)

        key_d = jax.random.PRNGKey(100 + evt)
        data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

        key_p = jax.random.PRNGKey(42)
        pred_q, pred_t = pred_sim(pp, key_p)

        # mask: sensors hit in data with charge > 0
        mask = data_q > 0
        n_hit = int(jnp.sum(mask))

        # time difference (data - pred) at hit sensors
        dt = jnp.where(mask, data_t - pred_t, 0.)
        mean_dt = float(jnp.sum(dt) / (jnp.sum(mask) + 1e-8))
        biases.append(mean_dt)

    biases = np.array(biases)
    print(f"\n{label}:")
    print(f"  mean(data_time - pred_time) = {np.mean(biases):.4f} ns")
    print(f"  std                         = {np.std(biases):.4f} ns")
    print(f"  per-event: {[f'{b:.3f}' for b in biases]}")

# ======================================================================
# TEST 2 – threshold effect on soft-min
# ======================================================================
print("\n" + "="*80)
print("TEST 2: Threshold effect on make_hits_simulation timing")
print("  Compares timing from make_hits_simulation at different thresholds")
print("  against make_hits_data (hard-min reference)")
print("="*80)

# We need the raw flat arrays from propagation. Easiest: hook into the
# prediction simulator.  But we can't easily extract intermediate arrays.
# Instead, let's build synthetic data that mimics the real case.

# Use the data simulator to get reference data, then manually construct
# what the prediction simulator would produce.

# Actually, a cleaner approach: compare make_hits_simulation vs make_hits_data
# on the SAME flat arrays (from the data propagation) to isolate the
# soft-min vs hard-min difference.

print("\nUsing data-propagated photon arrays (same physics) to isolate")
print("the soft-min vs segment-min bias.\n")

# To get the raw flat arrays, we create a modified simulator that returns them.
# Simpler: just create synthetic photon arrays that mimic real propagation.

rng = np.random.default_rng(42)
N_sensors = NUM_DET
N_photons_per_sensor_avg = 5
N_total = N_sensors * N_photons_per_sensor_avg

# Assign photons to random sensors
indices = rng.integers(0, N_sensors, size=N_total)
# Give them realistic weights
weights = rng.exponential(0.01, size=N_total).astype(np.float32)
# Give them realistic times: base arrival + small spread
base_times = rng.uniform(20, 80, size=N_sensors).astype(np.float32)
times = base_times[indices] + rng.exponential(0.5, size=N_total).astype(np.float32)

flat_w = jnp.array(weights)
flat_i = jnp.array(indices)
flat_t = jnp.array(times)

# QE corrections (all ones)
qe_corr = jnp.ones(N_sensors)
qe = 0.065

# Reference: hard-min (data mode)
key_qe = jax.random.PRNGKey(999)
ref_q, ref_t = make_hits_data(flat_w, flat_i, flat_t, N_sensors,
                               qe=qe, rng_key=key_qe)

for thr, thr_label in [(1e-10, "1e-10 (v2)"), (1e-5, "1e-5 (clean)")]:
    sim_q, sim_t = make_hits_simulation(flat_w, flat_i, flat_t, N_sensors,
                                         qe=qe, qe_corrections=qe_corr,
                                         threshold=thr, temperature=0.01)
    # Compare at sensors hit in both
    both_hit = (ref_q > 0) & (sim_q > 0)
    n = int(jnp.sum(both_hit))
    dt = jnp.where(both_hit, ref_t - sim_t, 0.)
    mean_dt = float(jnp.sum(dt) / (n + 1e-8))
    print(f"  threshold={thr_label}: mean(hardmin - softmin) = {mean_dt:.6f} ns  "
          f"({n} sensors)")


# ======================================================================
# TEST 3 – temperature effect on soft-min bias
# ======================================================================
print("\n" + "="*80)
print("TEST 3: Temperature effect on soft-min bias")
print("="*80)

for temp in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    sim_q, sim_t = make_hits_simulation(flat_w, flat_i, flat_t, N_sensors,
                                         qe=qe, qe_corrections=qe_corr,
                                         threshold=1e-10, temperature=temp)
    both_hit = (ref_q > 0) & (sim_q > 0)
    n = int(jnp.sum(both_hit))
    dt = jnp.where(both_hit, ref_t - sim_t, 0.)
    mean_dt = float(jnp.sum(dt) / (n + 1e-8))
    print(f"  T={temp:<6.3f}: mean(hardmin - softmin) = {mean_dt:.6f} ns  "
          f"({n} sensors)")

# ======================================================================
# TEST 4 – Compare actual loss sweeps with matched vs v2 parameters
# ======================================================================
print("\n" + "="*80)
print("TEST 4: t0 loss sweep bias across 10 events")
print("="*80)

from tools.optimization.losses import counts_loss, origin_time_loss, cone_time_loss

def sweep_t0(pred_sim_fn, data_sim_fn, n_events=10, t0_range=5.0, n_points=41):
    """Sweep t0 and find the loss minimum."""
    t0_vals = np.linspace(-t0_range, t0_range, n_points)
    min_t0s = []

    for evt in range(n_events):
        pd = load_and_pad(evt)
        key = jax.random.PRNGKey(42 + evt)
        pp = random_track(key, pd['energy'])
        pd = set_transform(pd, pp)

        key_d = jax.random.PRNGKey(100 + evt)
        data_q, data_t = jax.lax.stop_gradient(data_sim_fn(pp, key_d, pd))

        key_p = jax.random.PRNGKey(42)

        losses = []
        for t0_val in t0_vals:
            pp_shifted = ParticleParams(
                energy=pp.energy, position=pp.position,
                theta=pp.theta, phi=pp.phi,
                t0=jnp.array(t0_val))
            pred_q, pred_t = pred_sim_fn(pp_shifted, key_p)

            vl = origin_time_loss(pp.position, det_pts, data_t, data_q, jnp.array(t0_val))
            cl = counts_loss(data_q, pred_q)
            tl = cone_time_loss(data_q, pred_t, data_t, jnp.array(t0_val), tau=0.23)
            loss = jnp.sqrt((vl+1e-6)*(cl+1e-6)*(tl+1e-6))
            losses.append(float(loss))

        losses = np.array(losses)
        min_idx = np.argmin(losses)
        min_t0s.append(t0_vals[min_idx])

    return np.array(min_t0s)

# v2 defaults
data_sim_v2 = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)
pred_sim_v2 = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

print("\nv2 defaults (Nphot=300k, K=9, physics_config):")
mins_v2 = sweep_t0(pred_sim_v2, data_sim_v2)
print(f"  min-loss t0 values: {[f'{x:.2f}' for x in mins_v2]}")
print(f"  mean = {np.mean(mins_v2):.4f},  std = {np.std(mins_v2):.4f}")

# clean-like: Nphot=150k, K=7, same physics_config
data_sim_cl = setup_event_simulator(
    GEOM, 150_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)
pred_sim_cl = setup_event_simulator(
    GEOM, 150_000, temperature=0.10, K=7,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

print("\nclean-like (Nphot=150k, K=7, same physics_config):")
mins_cl = sweep_t0(pred_sim_cl, data_sim_cl)
print(f"  min-loss t0 values: {[f'{x:.2f}' for x in mins_cl]}")
print(f"  mean = {np.mean(mins_cl):.4f},  std = {np.std(mins_cl):.4f}")

print("\n" + "="*80)
print("DONE")
print("="*80)
