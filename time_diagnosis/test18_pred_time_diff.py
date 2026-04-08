#!/usr/bin/env python3
"""
TEST 18: Direct comparison of pred_t between clean and v2 for the SAME track.

Run both simulators on identical inputs and compare output times.
Since we can't import both in one script, save v2 results and load in clean.
This script runs V2.
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

# Use a fixed simple track for reproducibility
pos = jnp.array([0., 0., 0.])
th = jnp.array(1.0)
ph = jnp.array(0.5)
energy = jnp.array(1050.)
pp = ParticleParams(energy=energy, position=pos, theta=th, phi=ph, t0=jnp.array(0.))

print("Building V2 pred simulator...")
pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

print("Building V2 data simulator...")
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(pp, key_p)

pd = load_and_pad(0)
pd = set_transform(pd, pp)
key_d = jax.random.PRNGKey(100)
data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

# Save for comparison
np.savez(os.path.join(os.path.dirname(__file__), 'v2_results.npz'),
         pred_q=np.array(pred_q), pred_t=np.array(pred_t),
         data_q=np.array(data_q), data_t=np.array(data_t),
         pos=np.array(pos), th=float(th), ph=float(ph), energy=float(energy))

hit_pred = pred_q > 0
hit_data = data_q > 0
both = hit_pred & hit_data
n_both = int(jnp.sum(both))

print(f"\nV2 results:")
print(f"  pred: {int(jnp.sum(hit_pred))} sensors hit")
print(f"  data: {int(jnp.sum(hit_data))} sensors hit")
print(f"  both: {n_both} sensors")

print(f"\n  pred_t (hit sensors): mean={float(jnp.mean(pred_t[hit_pred])):.4f}  median={float(jnp.median(pred_t[hit_pred])):.4f}")
print(f"  data_t (hit sensors): mean={float(jnp.mean(data_t[hit_data])):.4f}  median={float(jnp.median(data_t[hit_data])):.4f}")

dp = data_t[both] - pred_t[both]
print(f"\n  data_t - pred_t (jointly hit):")
print(f"    mean={float(jnp.mean(dp)):.4f}  std={float(jnp.std(dp)):.4f}")
for p in [10, 23, 50, 77, 90]:
    print(f"    {p}th pct: {float(jnp.percentile(dp, p)):+.4f}")

# Fine t0 scan
fine_t0 = np.linspace(-1.0, 1.0, 201)
losses23 = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23)) for t in fine_t0]
losses12 = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.12)) for t in fine_t0]
print(f"\n  cone_min (tau=0.23): t0={fine_t0[np.argmin(losses23)]:+.3f}")
print(f"  cone_min (tau=0.12): t0={fine_t0[np.argmin(losses12)]:+.3f}")
print(f"\n  Saved to v2_results.npz")
