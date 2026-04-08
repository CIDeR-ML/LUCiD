#!/usr/bin/env python3
"""
TEST 21: Check if NaN photons occur during v2 propagation and if they corrupt results.

Hypothesis: v2 lacks clean's NaN protection. When photons get NaN positions/directions:
- Clean replaces NaN states with previous values and kills the ray (survival=0)
- V2 lets NaN propagate through scan → 0 * NaN = NaN → corrupts make_hits

This test:
1. Counts NaN photons in v2 propagation
2. Patches v2 with clean-style NaN protection
3. Compares bias with and without NaN protection
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import tools.simulation as sim_module
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.detector_params import ParticleParams
from tools.generate import read_photon_data_from_photonsim
from tools.optimization.losses import cone_time_loss

BASE = os.path.join(os.path.dirname(__file__), '..')
GEOM = os.path.join(BASE, 'config/SK_geom_config.json')
PHYS = os.path.join(BASE, 'config/SK_physics_config.json')
DATA = os.path.join(BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

detector = generate_detector(GEOM)

def load_and_pad(entry_idx):
    pd = read_photon_data_from_photonsim(DATA, entry_idx)
    N = len(pd['photon_origins'])
    pad = max(0, 1_000_000 - N)
    pd['photon_origins'] = jnp.pad(pd['photon_origins'], ((0,pad),(0,0)), constant_values=0)
    dd = jnp.array([0.,0.,1.])
    if pad > 0:
        pd['photon_directions'] = jnp.concatenate(
            [pd['photon_directions'], jnp.tile(dd, (pad,1))])
    pd['photon_times'] = jnp.pad(pd['photon_times'], (0,pad), constant_values=0)
    pd['N'] = N
    pd['apply_rotation'] = jnp.array(False)
    pd['rotation_axis'] = jnp.array([1.,0.,0.])
    pd['rotation_angle'] = jnp.array(0.)
    return pd

def set_transform(pd, pp):
    orig = jnp.array([0.,0.,1.])
    tgt = pp.direction / (jnp.linalg.norm(pp.direction)+1e-8)
    ax = jnp.cross(orig, tgt)
    an = jnp.linalg.norm(ax)
    ax = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax; pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = pp.position
    pd['apply_translation'] = jnp.array(True)
    return pd

def random_track(key, energy):
    k1,k2,k3,k4,k5 = jax.random.split(key, 5)
    frac = 0.6
    r = jax.random.uniform(k1, minval=0, maxval=detector.r*frac)
    tp = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    z = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
    pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
    th = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
    ph = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
    return ParticleParams(energy=jnp.array(float(energy)),
                          position=pos, theta=th, phi=ph, t0=jnp.array(0.))

# ── Step 1: Count NaN photons in v2 output ──────────────────
print("=== Step 1: Check for NaN in v2 output ===")
pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

pos = jnp.array([0., 0., 0.])
pp = ParticleParams(energy=jnp.array(1050.), position=pos,
                    theta=jnp.array(1.0), phi=jnp.array(0.5), t0=jnp.array(0.))

key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(pp, key_p)
nan_pred_q = int(jnp.sum(jnp.isnan(pred_q)))
nan_pred_t = int(jnp.sum(jnp.isnan(pred_t)))
print(f"  pred: nan_q={nan_pred_q} nan_t={nan_pred_t}")
print(f"  pred: total_q={float(jnp.nansum(pred_q)):.2f}  nhit={int(jnp.sum(pred_q > 0))}")

pd = load_and_pad(0); pd = set_transform(pd, pp)
key_d = jax.random.PRNGKey(100)
data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))
nan_data_q = int(jnp.sum(jnp.isnan(data_q)))
nan_data_t = int(jnp.sum(jnp.isnan(data_t)))
print(f"  data: nan_q={nan_data_q} nan_t={nan_data_t}")
print(f"  data: total_q={float(jnp.nansum(data_q)):.2f}  nhit={int(jnp.sum(data_q > 0))}")

# ── Step 2: Instrument propagation to count NaN photons per iteration ──
# We need to add debug prints inside the propagation loop.
# The easiest way is to monkey-patch the photon_update function to count NaNs.

print("\n=== Step 2: Check for NaN INSIDE propagation loop ===")
print("(Adding jax.debug.print to photon_update_fn)")

_orig_safe = sim_module.photon_iteration_update_factors_safe

def nan_counting_update(*args):
    """Wrapper that counts NaN outputs from the update function."""
    result = _orig_safe(*args)
    new_pos, new_dir, new_time, detect_prob, refl_atten, cont_factor = result
    has_nan = (jnp.any(jnp.isnan(new_pos)) | jnp.any(jnp.isnan(new_dir)) |
               jnp.isnan(new_time) | jnp.isnan(cont_factor))
    # Can't use debug.print inside vmap easily, so just flag NaN
    return result

# Instead, let's check the raw flat_weights/times for NaN
# We need to look at what make_hits receives
_orig_make_hits_sim = sim_module.make_hits_simulation

def debug_make_hits_sim(flat_weights, flat_indices, flat_times, num_sensors, *args, **kwargs):
    n_nan_w = int(jnp.sum(jnp.isnan(flat_weights)))
    n_nan_t = int(jnp.sum(jnp.isnan(flat_times)))
    n_total = len(flat_weights)
    n_nonzero_w = int(jnp.sum(flat_weights > 0))
    jax.debug.print("make_hits_sim: {}/{} NaN weights, {}/{} NaN times, {} nonzero weights",
                     n_nan_w, n_total, n_nan_t, n_total, n_nonzero_w)
    return _orig_make_hits_sim(flat_weights, flat_indices, flat_times, num_sensors, *args, **kwargs)

_orig_make_hits_data = sim_module.make_hits_data

def debug_make_hits_data(flat_weights, flat_indices, flat_times, num_sensors, *args, **kwargs):
    n_nan_w = int(jnp.sum(jnp.isnan(flat_weights)))
    n_nan_t = int(jnp.sum(jnp.isnan(flat_times)))
    n_total = len(flat_weights)
    n_nonzero_w = int(jnp.sum(flat_weights > 0))
    jax.debug.print("make_hits_data: {}/{} NaN weights, {}/{} NaN times, {} nonzero weights",
                     n_nan_w, n_total, n_nan_t, n_total, n_nonzero_w)
    return _orig_make_hits_data(flat_weights, flat_indices, flat_times, num_sensors, *args, **kwargs)

# Monkey-patch
sim_module.make_hits_simulation = debug_make_hits_sim
sim_module.make_hits_data = debug_make_hits_data

# Rebuild simulators with instrumented make_hits
pred_sim2 = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)
data_sim2 = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

# Restore originals
sim_module.make_hits_simulation = _orig_make_hits_sim
sim_module.make_hits_data = _orig_make_hits_data

print("\nPred sim (instrumented):")
pred_q2, pred_t2 = pred_sim2(pp, key_p)
print(f"  Result: nhit={int(jnp.sum(pred_q2>0))}")

print("\nData sim (instrumented):")
pd = load_and_pad(0); pd = set_transform(pd, pp)
data_q2, data_t2 = jax.lax.stop_gradient(data_sim2(pp, key_d, pd))
print(f"  Result: nhit={int(jnp.sum(data_q2>0))}")

# ── Step 3: Check if NaN flat_weights affect make_hits ──────
print("\n=== Step 3: Check how NaN affects make_hits_simulation ===")
# Create synthetic test with some NaN weights
n_test = 1000
test_w = jnp.ones(n_test) * 0.01
test_idx = jnp.zeros(n_test, dtype=jnp.int32)  # all map to sensor 0
test_t = jnp.ones(n_test) * 50.0
key_test = jax.random.PRNGKey(0)

# Normal case
from tools.simulation import make_hits_simulation
q_normal, t_normal = make_hits_simulation(test_w, test_idx, test_t, 10, qe=0.065, qe_corrections=jnp.ones(10))
print(f"  Normal: q[0]={float(q_normal[0]):.6f} t[0]={float(t_normal[0]):.4f}")

# With some NaN weights
test_w_nan = test_w.at[500:600].set(float('nan'))
q_nan, t_nan = make_hits_simulation(test_w_nan, test_idx, test_t, 10, qe=0.065, qe_corrections=jnp.ones(10))
print(f"  With NaN weights: q[0]={float(q_nan[0]):.6f} t[0]={float(t_nan[0]):.4f}")

# With NaN times too
test_t_nan = test_t.at[500:600].set(float('nan'))
q_nan2, t_nan2 = make_hits_simulation(test_w_nan, test_idx, test_t_nan, 10, qe=0.065, qe_corrections=jnp.ones(10))
print(f"  With NaN weights+times: q[0]={float(q_nan2[0]):.6f} t[0]={float(t_nan2[0]):.4f}")

print("\nDone.")
