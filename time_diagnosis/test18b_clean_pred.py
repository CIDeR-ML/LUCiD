#!/usr/bin/env python3
"""
TEST 18b: Run clean simulation with same track as test18, compare pred_t.
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

# Same fixed track as test18
pos = jnp.array([0., 0., 0.])
th = jnp.array(1.0)
ph = jnp.array(0.5)
energy = jnp.array(1050.)
particle_params = (energy, pos, jnp.array([th, ph]))

detector_params = (jnp.array(100.0), jnp.array(0.2), jnp.array(60.0), jnp.array(0.065))

print("Building CLEAN pred simulator...")
pred_sim = setup_event_simulator(GEOM, 300_000, temperature=0.10, K=9, is_data=False, max_sensors_per_cell=4)

key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(particle_params, detector_params, key_p)

# Load v2 results
v2 = np.load(os.path.join(os.path.dirname(__file__), 'v2_results.npz'))
v2_pred_q = jnp.array(v2['pred_q'])
v2_pred_t = jnp.array(v2['pred_t'])
v2_data_q = jnp.array(v2['data_q'])
v2_data_t = jnp.array(v2['data_t'])

hit_clean = pred_q > 0
hit_v2 = v2_pred_q > 0
both_pred = hit_clean & hit_v2

n_clean = int(jnp.sum(hit_clean))
n_v2 = int(jnp.sum(hit_v2))
n_both = int(jnp.sum(both_pred))

print(f"\nClean pred: {n_clean} sensors hit")
print(f"V2 pred:    {n_v2} sensors hit")
print(f"Both pred:  {n_both} sensors jointly hit")

print(f"\nClean pred_t (hit): mean={float(jnp.mean(pred_t[hit_clean])):.4f}  median={float(jnp.median(pred_t[hit_clean])):.4f}")
print(f"V2    pred_t (hit): mean={float(jnp.mean(v2_pred_t[hit_v2])):.4f}  median={float(jnp.median(v2_pred_t[hit_v2])):.4f}")

# Direct comparison on jointly-hit sensors
diff = pred_t[both_pred] - v2_pred_t[both_pred]
print(f"\nclean_pred_t - v2_pred_t (jointly hit):")
print(f"  mean={float(jnp.mean(diff)):.4f}  std={float(jnp.std(diff)):.4f}")
for p in [10, 23, 50, 77, 90]:
    print(f"  {p}th pct: {float(jnp.percentile(diff, p)):+.4f}")

# Charges comparison
q_diff = pred_q[both_pred] - v2_pred_q[both_pred]
print(f"\nclean_pred_q - v2_pred_q (jointly hit):")
print(f"  mean={float(jnp.mean(q_diff)):.6f}  std={float(jnp.std(q_diff)):.6f}")

# Now use v2's DATA with clean's PRED for cone_time_loss
fine_t0 = np.linspace(-1.0, 1.0, 201)
losses_clean_pred = [float(cone_time_loss(v2_data_q, pred_t, v2_data_t, jnp.array(t), tau=0.23)) for t in fine_t0]
losses_v2_pred = [float(cone_time_loss(v2_data_q, v2_pred_t, v2_data_t, jnp.array(t), tau=0.23)) for t in fine_t0]

print(f"\nUsing same data (v2 data_sim), comparing pred sources:")
print(f"  cone_min with clean_pred (tau=0.23): t0={fine_t0[np.argmin(losses_clean_pred)]:+.3f}")
print(f"  cone_min with v2_pred    (tau=0.23): t0={fine_t0[np.argmin(losses_v2_pred)]:+.3f}")

# Also compare data_t between v2 data and clean data on the same track
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

def set_transform(pd, pos, direction):
    from tools.simulation import spherical_to_cartesian
    orig = jnp.array([0.,0.,1.])
    direction_vec = jnp.array([jnp.sin(th)*jnp.cos(ph), jnp.sin(th)*jnp.sin(ph), jnp.cos(th)])
    tgt = direction_vec / (jnp.linalg.norm(direction_vec)+1e-8)
    ax = jnp.cross(orig, tgt)
    an = jnp.linalg.norm(ax)
    ax = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax; pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = pos
    pd['apply_translation'] = jnp.array(True)
    return pd

print("\nBuilding CLEAN data simulator...")
data_sim = setup_event_simulator(GEOM, 300_000, temperature=0.0, K=20, is_data=True, is_calibration=False)

pd = load_and_pad(0)
pd = set_transform(pd, pos, None)
key_d = jax.random.PRNGKey(100)
clean_data_q, clean_data_t = jax.lax.stop_gradient(data_sim(particle_params, detector_params, key_d, pd))

both_data = (clean_data_q > 0) & (v2_data_q > 0)
n_both_data = int(jnp.sum(both_data))
data_diff = clean_data_t[both_data] - v2_data_t[both_data]
print(f"\nData comparison ({n_both_data} jointly-hit sensors):")
print(f"  clean_data_t - v2_data_t:")
print(f"    mean={float(jnp.mean(data_diff)):.4f}  std={float(jnp.std(data_diff)):.4f}")
for p in [10, 23, 50, 77, 90]:
    print(f"    {p}th pct: {float(jnp.percentile(data_diff, p)):+.4f}")
