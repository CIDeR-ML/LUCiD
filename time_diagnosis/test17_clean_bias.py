#!/usr/bin/env python3
"""
TEST 17: Run the CLEAN simulation with the same test methodology to measure bias.
"""
import sys, os
CLEAN_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'clean_run', 'LUCiD')
sys.path.insert(0, CLEAN_BASE)

import jax
import jax.numpy as jnp
import numpy as np

from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.generate import read_photon_data_from_photonsim
from tools.optimization.losses import cone_time_loss

GEOM = os.path.join(CLEAN_BASE, 'config/SK_geom_config.json')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

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

def spherical_to_cartesian(theta, phi):
    return jnp.array([jnp.sin(theta)*jnp.cos(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(theta)])

def set_transform(pd, pos, direction):
    orig = jnp.array([0.,0.,1.])
    tgt = direction / (jnp.linalg.norm(direction)+1e-8)
    ax = jnp.cross(orig, tgt)
    an = jnp.linalg.norm(ax)
    ax = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax; pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = pos
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
    direction = spherical_to_cartesian(th, ph)
    return pos, direction, th, ph, jnp.array(float(energy))

t0_vals = np.linspace(-1.0, 1.0, 21)

# Clean's detector params tuple: (scatter_length, reflection_rate, absorption_length, qe_as_tau_gs)
detector_params = (jnp.array(100.0), jnp.array(0.2), jnp.array(60.0), jnp.array(0.065))

print("Building CLEAN simulators...")
data_sim = setup_event_simulator(GEOM, 300_000, temperature=0.0, K=20, is_data=True, is_calibration=False)
pred_sim = setup_event_simulator(GEOM, 300_000, temperature=0.10, K=9, is_data=False, max_sensors_per_cell=4)

print("\n=== CLEAN: data_sim vs pred_sim (tau=0.23) ===")
mins = []
for evt in range(5):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pos, direction, th, ph, energy = random_track(key, pd['energy'])
    pd = set_transform(pd, pos, direction)
    particle_params = (energy, pos, jnp.array([th, ph]))

    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim(particle_params, detector_params, key_d, pd))
    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim(particle_params, detector_params, key_p)

    losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23)) for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    mins.append(cm)

    # Also show timing stats
    both = (data_q > 0) & (pred_q > 0)
    dp = data_t[both] - pred_t[both]
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}  23pct(data-pred)={float(jnp.percentile(dp, 23)):+.4f}  median={float(jnp.median(dp)):+.4f}")

print(f"  avg = {np.mean(mins):+.3f}")

print("\n=== CLEAN: same but tau=0.12 (default) ===")
mins12 = []
for evt in range(5):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pos, direction, th, ph, energy = random_track(key, pd['energy'])
    pd = set_transform(pd, pos, direction)
    particle_params = (energy, pos, jnp.array([th, ph]))
    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim(particle_params, detector_params, key_d, pd))
    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim(particle_params, detector_params, key_p)
    losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.12)) for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    mins12.append(cm)
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")
print(f"  avg = {np.mean(mins12):+.3f}")
