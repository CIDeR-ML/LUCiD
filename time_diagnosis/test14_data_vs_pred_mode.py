#!/usr/bin/env python3
"""
TEST 14: Is the bias from data_mode(sampling) vs pred_mode(expected-value),
or from ROOT photons vs SIREN photons?

Compare:
A. Normal: data_sim(ROOT, sampling) vs pred_sim(SIREN, expected-value)
B. Both expected-value mode: data as expected-value vs pred expected-value
   (create a second pred_sim to act as "data" with ROOT photons)
C. Both sampling mode: data_sim(ROOT, sampling) vs pred but with sampling

We can't easily swap the photon source, but we can test the PROCESSING mode.
The data_sim uses is_data=True → photon_iteration_sample
The pred_sim uses is_data=False → photon_iteration_update_factors_safe

What if we make the data_sim use expected-value mode too?
(Not possible directly, but we can hack it)

Instead: use pred_sim with TWO different K values to see if K difference matters.
Also: directly compare the aggregated times from data vs pred.
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

t0_vals = np.linspace(-1.0, 1.0, 21)

print("Building simulators...")
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

# Data sim but with expected-value mode (use_expected_value=True, is_data=False)
# but fed ROOT photons (this is a "prediction" sim fed ROOT data)
# Actually we can't easily do this because is_data controls BOTH the
# photon source AND the processing mode.

# Instead, let's look at the time differences directly
print("\n=== Detailed timing analysis ===")

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

# Geometric expected time
d = jnp.linalg.norm(det_pts - pos[None,:], axis=1)
expected_t = (d - 0.25) / C_MEDIUM

data_res = data_t[both] - expected_t[both]
pred_res = pred_t[both] - expected_t[both]
dp_diff = data_t[both] - pred_t[both]

print(f"\n{n_both} jointly-hit sensors")
print(f"\ndata_t - geometric_expected:")
print(f"  mean={float(jnp.mean(data_res)):.4f}  std={float(jnp.std(data_res)):.4f}")
for p in [10, 23, 50, 77]:
    print(f"  {p}th pct: {float(jnp.percentile(data_res, p)):+.4f}")

print(f"\npred_t - geometric_expected:")
print(f"  mean={float(jnp.mean(pred_res)):.4f}  std={float(jnp.std(pred_res)):.4f}")
for p in [10, 23, 50, 77]:
    print(f"  {p}th pct: {float(jnp.percentile(pred_res, p)):+.4f}")

print(f"\ndata_t - pred_t (direct residual):")
print(f"  mean={float(jnp.mean(dp_diff)):.4f}  std={float(jnp.std(dp_diff)):.4f}")
for p in [10, 23, 50, 77]:
    print(f"  {p}th pct: {float(jnp.percentile(dp_diff, p)):+.4f}")

# The smooth_pinball minimum is at r = sigma*ln((1-tau)/tau) where r = data - pred - t0
# For tau=0.23, sigma=0.25: r* = 0.25*ln(0.77/0.23) = 0.302
# So the cone_min is at t0 such that weighted_quantile(data - pred - t0) ≈ 0.302
# Meaning t0 ≈ weighted_quantile(data - pred) - 0.302
smooth_offset = 0.25 * np.log(0.77/0.23)
print(f"\nSmooth pinball offset: {smooth_offset:.4f}")

# Weight-aware quantile
w = jnp.where(data_q > 0., data_q, 0.)
# Simple: compute what t0 should be
# The loss minimizer finds t0 such that sigma*ln((1-tau)/tau) = quantile_tau(r)
# r = data - pred - t0, so quantile_tau(data - pred - t0) = offset
# For a fixed distribution, this gives t0 = quantile_tau(data - pred) - offset
# But it's weighted by observed_counts...

# Direct scan
print(f"\nFine-grained t0 sweep (0.01 resolution):")
fine_t0 = np.linspace(-0.5, 0.5, 101)
losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23)) for t in fine_t0]
min_idx = np.argmin(losses)
print(f"  cone_min t0 = {fine_t0[min_idx]:+.3f} (loss = {losses[min_idx]:.6f})")
print(f"  expected from median(data-pred) - smooth_offset: {float(jnp.median(dp_diff)) - smooth_offset:+.4f}")
print(f"  expected from 23pct(data-pred) - smooth_offset: {float(jnp.percentile(dp_diff, 23)) - smooth_offset:+.4f}")
