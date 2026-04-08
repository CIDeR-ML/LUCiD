#!/usr/bin/env python3
"""
TEST 10: Check two remaining differences:

1. NaN forward-pass handling:
   Clean: replaces NaN positions/directions/times with previous values
   V2: lets NaN propagate (filtered later by isfinite)

2. normalize function:
   V2: v / jnp.maximum(norm, epsilon)
   Clean: v / (norm + epsilon)

Test: Add NaN counting to v2 simulation to see if NaNs actually occur.
Also: test with clean-style normalize to see if it matters.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import tools.simulation as sim_module

# Count NaNs in propagation
nan_counts = {'pos': [], 'dir': [], 'time': [], 'cont': []}

# Save originals
_orig_update = sim_module.photon_iteration_update_factors

def nan_counting_wrapper(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light):
    result = _orig_update(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light)
    new_pos, new_dir, new_time, detect_prob, refl_att, cont_factor = result
    # Can't easily count during JIT, so just return
    return result

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

# ==========================================================
# Test 1: Count NaNs using jax.debug.print in a patched
# version of _common_propagation (can't do this easily with
# monkey-patching, so use a different approach)
# ==========================================================

print("=== Test: NaN counting via debug.print ===")
print("(NaN counts will appear during JIT compilation)")

# Create a modified update function that counts NaNs via debug.print
def nan_debug_update(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light):
    result = _orig_update(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light)
    new_pos, new_dir, new_time, detect_prob, refl_att, cont_factor = result
    jax.debug.print("NaN pos:{} dir:{} time:{} cont:{}",
                     jnp.isnan(new_pos).any(), jnp.isnan(new_dir).any(),
                     jnp.isnan(new_time), jnp.isnan(cont_factor))
    return result

# Patch and run
sim_module.photon_iteration_update_factors = nan_debug_update
pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)
sim_module.photon_iteration_update_factors = _orig_update

pd = load_and_pad(0)
key = jax.random.PRNGKey(42)
pp = random_track(key, pd['energy'])
pd = set_transform(pd, pp)

print("\nRunning pred sim (NaN debug prints will appear)...")
key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(pp, key_p)

# Count output NaNs
n_nan_q = int(jnp.sum(jnp.isnan(pred_q)))
n_nan_t = int(jnp.sum(jnp.isnan(pred_t)))
n_inf_t = int(jnp.sum(jnp.isinf(pred_t)))
n_hit = int(jnp.sum(pred_q > 0))
print(f"\nOutput: {n_hit} sensors hit, {n_nan_q} NaN charges, {n_nan_t} NaN times, {n_inf_t} inf times")

# ==========================================================
# Test 2: Compare normalize implementations
# ==========================================================
print("\n=== Test: normalize implementation ===")
from tools.simulation import normalize as v2_normalize

def clean_normalize(v, epsilon=1e-6):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + epsilon)

# Test with typical vectors
test_vecs = [
    jnp.array([1., 0., 0.]),
    jnp.array([0.001, 0., 0.]),
    jnp.array([1e-7, 0., 0.]),
    jnp.array([0., 0., 0.]),
    jnp.array([100., 200., 300.]),
]

for v in test_vecs:
    v2 = v2_normalize(v)
    cl = clean_normalize(v)
    diff = float(jnp.linalg.norm(v2 - cl))
    print(f"  v={v}: diff={diff:.2e}")
