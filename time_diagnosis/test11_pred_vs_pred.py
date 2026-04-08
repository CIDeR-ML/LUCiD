#!/usr/bin/env python3
"""
TEST 11: Use pred_sim for BOTH data and prediction to isolate the bias source.

If cone_time_loss minimum is at t0=0 when both come from pred_sim,
then the bias is from data_sim vs pred_sim differences.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
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

# ── Simulators ──────────────────────────────────────────
print("Building simulators...")

# Standard data sim (sampling mode)
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

# Standard pred sim (expected-value mode)
pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

# Second pred sim to use as "data" (same settings as pred)
pred_as_data = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

print("\n=== A: Normal (data_sim vs pred_sim) ===")
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
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")

print("\n=== B: pred_sim vs pred_sim (different RNG keys) ===")
for evt in range(3):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp = random_track(key, pd['energy'])
    # Use pred_as_data with one key
    key_d = jax.random.PRNGKey(100 + evt)
    pad_q, pad_t = jax.lax.stop_gradient(pred_as_data(pp, key_d))
    # Use pred_sim with different key
    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim(pp, key_p)
    losses = [float(cone_time_loss(pad_q, pred_t, pad_t, jnp.array(t), tau=0.23))
              for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")

print("\n=== C: pred_sim vs pred_sim (same RNG key) ===")
for evt in range(3):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp = random_track(key, pd['energy'])
    key_same = jax.random.PRNGKey(42)
    pad_q, pad_t = jax.lax.stop_gradient(pred_as_data(pp, key_same))
    pred_q, pred_t = pred_sim(pp, key_same)
    losses = [float(cone_time_loss(pad_q, pred_t, pad_t, jnp.array(t), tau=0.23))
              for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")
