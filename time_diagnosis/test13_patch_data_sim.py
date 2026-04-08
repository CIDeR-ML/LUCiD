#!/usr/bin/env python3
"""
TEST 13: Patch the DATA simulator to match clean behavior.

V2 data sim differences from clean:
1. k3 reuse: scatter_dir uses k3, absorption uses k3 (should be k4)
2. Diffuse wall reflection (should be all specular)
3. Dual reflection rates (wall=0.2, sensor=0.1, should be single 0.2)

Test each fix individually to find which causes the bias.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

import tools.simulation as sim_module
from tools.simulation import (
    sample_scatter_distance, compute_reflection_direction,
    compute_scatter_direction, normalize, sample_cosine_hemisphere,
)

_orig_sample = sim_module.photon_iteration_sample
_orig_update = sim_module.photon_iteration_update_factors

# ── Fix A: Fix k3/k4 absorption key reuse ────────────────────────
def sample_fix_absorption_key(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light):
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    reflection_rate = jnp.where(hit_sensor, sensor_reflection_rate, wall_reflection_rate)
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)
    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob
    u2 = jax.random.uniform(k2)
    reflects = reaches_surface & (u2 < reflection_rate)
    detects = reaches_surface & (u2 >= reflection_rate)
    scatters = ~reaches_surface
    epsilon = 1e-4
    new_pos = jnp.where(scatters,
        position + scatter_distance * normalize(direction),
        position + surface_distance * normalize(direction) + epsilon * normalize(normal))
    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(normal, k4)
    reflection_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    scatter_dir = compute_scatter_direction(direction, k3)
    new_dir = jnp.where(reflects, reflection_dir,
                         jnp.where(scatters, scatter_dir, direction))
    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + distance_traveled / speed_of_light
    survival_prob = jnp.exp(-distance_traveled / absorption_length)
    # FIX: use k4 for absorption instead of k3
    k5 = jax.random.fold_in(rng_key, 99)  # independent key for absorption
    u_absorption = jax.random.uniform(k5)
    survives_absorption = u_absorption < survival_prob
    attenuation = survives_absorption.astype(jnp.float32)
    detect_prob = detects.astype(jnp.float32)
    reflection_attenuation = attenuation
    continuing_factor = jnp.where(detects, 0.0, attenuation)
    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


# ── Fix B: All specular reflection (like clean) ───────────────────
def sample_all_specular(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light):
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    reflection_rate = jnp.where(hit_sensor, sensor_reflection_rate, wall_reflection_rate)
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)
    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob
    u2 = jax.random.uniform(k2)
    reflects = reaches_surface & (u2 < reflection_rate)
    detects = reaches_surface & (u2 >= reflection_rate)
    scatters = ~reaches_surface
    epsilon = 1e-4
    new_pos = jnp.where(scatters,
        position + scatter_distance * normalize(direction),
        position + surface_distance * normalize(direction) + epsilon * normalize(normal))
    # FIX: all specular
    reflection_dir = compute_reflection_direction(direction, normal)
    scatter_dir = compute_scatter_direction(direction, k3)
    new_dir = jnp.where(reflects, reflection_dir,
                         jnp.where(scatters, scatter_dir, direction))
    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + distance_traveled / speed_of_light
    survival_prob = jnp.exp(-distance_traveled / absorption_length)
    u_absorption = jax.random.uniform(k3)
    survives_absorption = u_absorption < survival_prob
    attenuation = survives_absorption.astype(jnp.float32)
    detect_prob = detects.astype(jnp.float32)
    reflection_attenuation = attenuation
    continuing_factor = jnp.where(detects, 0.0, attenuation)
    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


# ── Fix C: All fixes combined (clean-like data sampling) ──────────
def sample_clean_like(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length, hit_sensor, rng_key, speed_of_light):
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    # Use wall_reflection_rate for all surfaces (like clean's single rate)
    reflection_rate = wall_reflection_rate
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)
    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob
    u2 = jax.random.uniform(k2)
    reflects = reaches_surface & (u2 < reflection_rate)
    detects = reaches_surface & (u2 >= reflection_rate)
    scatters = ~reaches_surface
    epsilon = 1e-4
    new_pos = jnp.where(scatters,
        position + scatter_distance * normalize(direction),
        position + surface_distance * normalize(direction) + epsilon * normalize(normal))
    # All specular
    reflection_dir = compute_reflection_direction(direction, normal)
    scatter_dir = compute_scatter_direction(direction, k3)
    new_dir = jnp.where(reflects, reflection_dir,
                         jnp.where(scatters, scatter_dir, direction))
    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + distance_traveled / speed_of_light
    survival_prob = jnp.exp(-distance_traveled / absorption_length)
    # k4 for absorption (independent)
    u_absorption = jax.random.uniform(k4)
    survives_absorption = u_absorption < survival_prob
    attenuation = survives_absorption.astype(jnp.float32)
    detect_prob = detects.astype(jnp.float32)
    reflection_attenuation = attenuation
    continuing_factor = jnp.where(detects, 0.0, attenuation)
    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


# ── Setup ─────────────────────────────────────────────────────────────
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

t0_vals = np.linspace(-1.0, 1.0, 21)

# Standard pred sim (unchanged)
print("Building pred simulator...")
pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

def run_data_test(label, sample_fn):
    # Patch data mode sampling function
    sim_module.photon_iteration_sample = sample_fn
    data_sim = setup_event_simulator(
        GEOM, 300_000, temperature=0.0, K=20,
        is_data=True, is_calibration=False,
        physics_config=PHYS, default_detector_params=True)
    # Keep patch active through first call (lazy JIT)

    mins = []
    for evt in range(3):
        pd = load_and_pad(evt)
        key = jax.random.PRNGKey(42 + evt)
        pp = random_track(key, pd['energy'])
        pd = set_transform(pd, pp)
        key_d = jax.random.PRNGKey(100 + evt)
        data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))
        key_p = jax.random.PRNGKey(42)
        pred_q, pred_t = pred_sim(pp, key_p)
        losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
                  for t in t0_vals]
        cm = t0_vals[np.argmin(losses)]
        mins.append(cm)

    sim_module.photon_iteration_sample = _orig_sample
    print(f"  {label}: cone_min = {[f'{m:+.2f}' for m in mins]}  avg={np.mean(mins):+.3f}")
    return mins

print("Running tests...\n")
run_data_test("A: v2 original data sim", _orig_sample)
run_data_test("B: fix absorption key (k4 instead of k3)", sample_fix_absorption_key)
run_data_test("C: all specular (no diffuse walls)", sample_all_specular)
run_data_test("D: clean-like (specular + single refl + k4 absorption)", sample_clean_like)
