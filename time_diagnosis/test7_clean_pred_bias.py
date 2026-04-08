#!/usr/bin/env python3
"""
TEST 7: Is the prediction time biased in CLEAN simulation?
Run origin_time_loss using PRED times to see if pred_t is early.
"""
import sys, os
CLEAN_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '..', '..', 'clean_run', 'LUCiD'))
sys.path.insert(0, CLEAN_BASE)

import jax
import jax.numpy as jnp
import numpy as np
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.utils import spherical_to_cartesian
from tools.generate import read_photon_data_from_photonsim
from tools.optimization.losses import origin_time_loss, cone_time_loss

GEOM = os.path.join(CLEAN_BASE, 'config/SK_geom_config.json')
DATA = os.path.join(CLEAN_BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

detector = generate_detector(GEOM)
det_pts = jnp.array(detector.all_points)
C_MEDIUM = 0.299792 / 1.33

detector_params = (jnp.array(50.), jnp.array(0.2), jnp.array(50.), jnp.array(0.001))

print("Building clean simulators...")
data_sim = setup_event_simulator(GEOM, 150_000, temperature=0.0, K=20,
                                  is_data=True, is_calibration=False)
pred_sim = setup_event_simulator(GEOM, 150_000, temperature=0.10, K=7,
                                  is_data=False, max_sensors_per_cell=4)

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
    return pd

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

pd = load_and_pad(0)
key = jax.random.PRNGKey(42)
k1,k2,k3,k4,k5 = jax.random.split(key, 5)
frac = 0.6
r = jax.random.uniform(k1, minval=0, maxval=detector.r*frac)
tp = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
z = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
theta = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
phi = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
direction = spherical_to_cartesian(theta, phi)

pd = set_transform(pd, pos, direction)
true_params = (jnp.array(float(pd['energy'])), pos, jnp.array([theta, phi]))

key_d = jax.random.PRNGKey(100)
data_q, data_t = jax.lax.stop_gradient(data_sim(true_params, detector_params, key_d, pd))

key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(true_params, detector_params, key_p)

both = (data_q > 0) & (pred_q > 0)
n_both = int(jnp.sum(both))

# Geometric expected time
d = jnp.linalg.norm(det_pts - pos[None,:], axis=1)
expected_t = (d - 0.25) / C_MEDIUM

data_res = data_t[both] - expected_t[both]
pred_res = pred_t[both] - expected_t[both]

print(f"\n=== CLEAN Event 0: {n_both} jointly-hit sensors ===")
print(f"data - geom: mean={float(jnp.mean(data_res)):.4f}  median={float(jnp.median(data_res)):.4f}")
print(f"pred - geom: mean={float(jnp.mean(pred_res)):.4f}  median={float(jnp.median(pred_res)):.4f}")

for pct in [10, 23, 50]:
    d_p = float(jnp.percentile(data_res, pct))
    p_p = float(jnp.percentile(pred_res, pct))
    print(f"  {pct}th pct: data-geom={d_p:+.4f}  pred-geom={p_p:+.4f}  diff={d_p-p_p:+.4f}")

print(f"\n--- origin_time_loss with DATA times ---")
for t in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    vl = float(origin_time_loss(pos, det_pts, data_t, data_q, jnp.array(t)))
    print(f"  t0={t:+.2f}: {vl:.6f}")

print(f"\n--- origin_time_loss with PRED times ---")
for t in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    vl = float(origin_time_loss(pos, det_pts, pred_t, pred_q, jnp.array(t)))
    print(f"  t0={t:+.2f}: {vl:.6f}")

print(f"\n--- cone_time_loss ---")
for t in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    tl = float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
    print(f"  t0={t:+.2f}: {tl:.6f}")
