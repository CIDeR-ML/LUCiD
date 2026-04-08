#!/usr/bin/env python3
"""
TEST 8: Monkey-patch the STE probability basis in v2 simulation and
see if the t0 bias disappears.

V2 uses: probs = [reach_surface_prob, scatter_prob]
Clean:   probs = [reflect_prob, scatter_prob]

We patch photon_iteration_update_factors to use reflect_prob
and compare cone_time_loss.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

# Import the module so we can patch it
import tools.simulation as sim_module

# Save original function
_orig_update = sim_module.photon_iteration_update_factors

def patched_update_factors(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length,
        hit_sensor, rng_key, speed_of_light):
    """Same as original but uses [reflect_prob, scatter_prob] for STE."""
    from tools.propagation.common import (
        sample_scatter_distance, compute_reflection_direction,
        compute_scatter_direction, normalize, sample_cosine_hemisphere,
    )

    k1, k2, k3 = jax.random.split(rng_key, 3)

    reflection_rate = jnp.where(hit_sensor, sensor_reflection_rate, wall_reflection_rate)
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    ratio = surface_distance / scatter_length
    reach_surface_prob = jnp.exp(-ratio)
    scatter_prob = -jnp.expm1(-ratio)

    reflect_prob = reach_surface_prob * reflection_rate
    detect_prob = reach_surface_prob * (1 - reflection_rate)

    reflection_attenuation = jnp.exp(-surface_distance / absorption_length)
    scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)

    # KEY CHANGE: use reflect_prob instead of reach_surface_prob
    probs = jnp.array([reflect_prob, scatter_prob])
    probs_normalized = probs / (jnp.sum(probs) + 1e-10)

    u = jax.random.uniform(k1)
    hard_choice = (u < probs_normalized[0]).astype(jnp.float32)
    hard_weights = jnp.array([hard_choice, 1.0 - hard_choice])
    action_weights = hard_weights - jax.lax.stop_gradient(probs_normalized) + probs_normalized

    surface_weight = action_weights[0]
    scatter_weight = action_weights[1]

    epsilon = 1e-4
    surface_pos = position + surface_distance * normalize(direction) + epsilon * normalize(normal)
    scatter_pos = position + scatter_distance * normalize(direction)

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(normal, k3)
    reflection_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    scatter_dir = compute_scatter_direction(direction, k3)

    new_pos = surface_weight * surface_pos + scatter_weight * scatter_pos
    new_dir = normalize(surface_weight * reflection_dir + scatter_weight * scatter_dir)

    continuing_factor = reflect_prob * reflection_attenuation + scatter_prob * scatter_attenuation

    distance_traveled = surface_weight * surface_distance + scatter_weight * scatter_distance
    new_time = time + distance_traveled / speed_of_light

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


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
det_pts = jnp.array(detector.all_points)

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

# ── RUN 1: original v2 STE (reach_surface_prob) ──────────────────────
print("=== Building ORIGINAL v2 simulators ===")
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)
pred_sim_orig = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

# Run on 3 events
orig_mins = []
for evt in range(3):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp = random_track(key, pd['energy'])
    pd = set_transform(pd, pp)

    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim_orig(pp, key_p)

    losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
              for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    orig_mins.append(cm)
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")
print(f"  >> mean = {np.mean(orig_mins):+.3f}")

# ── RUN 2: patched STE (reflect_prob) ────────────────────────────────
print("\n=== Building PATCHED v2 simulators (reflect_prob STE) ===")
# Monkey-patch before creating the simulator
sim_module.photon_iteration_update_factors = patched_update_factors

pred_sim_patched = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

# Restore original
sim_module.photon_iteration_update_factors = _orig_update

patched_mins = []
for evt in range(3):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp = random_track(key, pd['energy'])
    pd = set_transform(pd, pp)

    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim_patched(pp, key_p)

    losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
              for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    patched_mins.append(cm)
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")
print(f"  >> mean = {np.mean(patched_mins):+.3f}")

print(f"\nORIGINAL (reach_surface): mean cone_min = {np.mean(orig_mins):+.3f}")
print(f"PATCHED  (reflect_prob):  mean cone_min = {np.mean(patched_mins):+.3f}")
