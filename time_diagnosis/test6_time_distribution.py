#!/usr/bin/env python3
"""
TEST 6: Compare per-sensor time distributions from data vs prediction.

For sensors hit by both, compute:
- data_time (hard-min from ROOT photons, sample mode)
- pred_time (soft-min from SIREN photons, expected-value mode)
- The residual: data_time - pred_time

Then check: does origin_time_loss alone have a t0 bias?
(origin_time_loss only uses data_time + geometry, no prediction)
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
from tools.optimization.losses import origin_time_loss, cone_time_loss

BASE = os.path.join(os.path.dirname(__file__), '..')
GEOM = os.path.join(BASE, 'config/SK_geom_config.json')
PHYS = os.path.join(BASE, 'config/SK_physics_config.json')
DATA = os.path.join(BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

detector = generate_detector(GEOM)
det_pts = jnp.array(detector.all_points)
C_MEDIUM = 0.299792 / 1.33

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

print("Building simulators...")
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)
pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

t0_vals = np.linspace(-1.0, 1.0, 21)

# Single event, detailed analysis
pd = load_and_pad(0)
key = jax.random.PRNGKey(42)
pp = random_track(key, pd['energy'])
pd = set_transform(pd, pp)
pos = pp.position

key_d = jax.random.PRNGKey(100)
data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(pp, key_p)

both = (data_q > 0) & (pred_q > 0)
n_both = int(jnp.sum(both))

# 1. Geometric expected time (from origin_time_loss formula)
d = jnp.linalg.norm(det_pts - pos[None, :], axis=1)
expected_t = (d - 0.25) / C_MEDIUM  # photosensor_radius=0.25

print(f"\n=== Event 0: {n_both} jointly-hit sensors ===")

# Residuals at t0=0
data_residuals = data_t[both] - expected_t[both]    # data - geometry
pred_residuals = pred_t[both] - expected_t[both]     # pred - geometry
dp_residuals = data_t[both] - pred_t[both]           # data - pred

print(f"\ndata - expected_geom:")
print(f"  mean={float(jnp.mean(data_residuals)):.4f}  median={float(jnp.median(data_residuals)):.4f}  "
      f"std={float(jnp.std(data_residuals)):.4f}")

print(f"pred - expected_geom:")
print(f"  mean={float(jnp.mean(pred_residuals)):.4f}  median={float(jnp.median(pred_residuals)):.4f}  "
      f"std={float(jnp.std(pred_residuals)):.4f}")

print(f"data - pred:")
print(f"  mean={float(jnp.mean(dp_residuals)):.4f}  median={float(jnp.median(dp_residuals)):.4f}  "
      f"std={float(jnp.std(dp_residuals)):.4f}")

# 2. Percentiles of (data - expected) and (pred - expected)
for pct in [10, 23, 50, 77]:
    d_pct = float(jnp.percentile(data_residuals, pct))
    p_pct = float(jnp.percentile(pred_residuals, pct))
    print(f"  {pct}th pct:  data-geom={d_pct:+.4f}  pred-geom={p_pct:+.4f}  diff={d_pct-p_pct:+.4f}")

# 3. Sweep t0 for each loss component independently
print(f"\n=== t0 sweep: origin_time_loss only (uses data_t + geometry) ===")
for t in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    vl = float(origin_time_loss(pos, det_pts, data_t, data_q, jnp.array(t)))
    print(f"  t0={t:+.2f}: vertex_loss={vl:.6f}")

print(f"\n=== t0 sweep: cone_time_loss only (uses data_t + pred_t) ===")
for t in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    tl = float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
    print(f"  t0={t:+.2f}: cone_loss={tl:.6f}")

# 4. What if we use pred_t in origin_time_loss instead of data_t?
#    This tells us if pred timing is biased relative to geometry
print(f"\n=== t0 sweep: origin_time_loss with PRED times (pred_t + geometry) ===")
for t in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    vl = float(origin_time_loss(pos, det_pts, pred_t, pred_q, jnp.array(t)))
    print(f"  t0={t:+.2f}: vertex_loss={vl:.6f}")
