#!/usr/bin/env python3
"""
TEST 15: Swap SIREN model weights to test if they cause the t0 bias.

V2 uses:  photonsim_siren.npz       (MD5: 8672198e...)
Clean uses: photonsim_siren_weights.npz (MD5: 1017c327...)

Test A: Normal v2 (v2 SIREN weights) — baseline
Test B: v2 code + clean SIREN weights — if bias disappears, SIREN is the cause
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

# SIREN model paths
V2_SIREN = os.path.join(BASE, 'data/water/muon/siren_training/trained_model/photonsim_siren')
CLEAN_SIREN = os.path.join(os.path.dirname(__file__), '..', '..', 'clean_run/LUCiD/data/water/muon/siren_training/trained_model/photonsim_siren')

# Verify files exist
v2_weights = V2_SIREN + '.npz'
clean_weights = CLEAN_SIREN + '_weights.npz'
print(f"V2 weights:    {v2_weights}  exists={os.path.exists(v2_weights)}")
print(f"Clean weights: {clean_weights}  exists={os.path.exists(clean_weights)}")

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

# To swap the SIREN model, we need to monkey-patch how setup_event_simulator
# loads the model. The model path comes from photonsim_params['siren_model_path'].
# We'll patch load_photonsim_params to return a modified dict.

import tools.simulation as sim_module
_orig_load = sim_module.unpack_photonsim_params

def make_patched_loader(siren_path):
    def patched_load(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        result['siren_model_path'] = siren_path
        print(f"  [PATCHED] siren_model_path -> {siren_path}")
        return result
    return patched_load

# ── Test A: Normal v2 SIREN ──────────────────────────────
print("\n=== A: V2 SIREN weights (baseline) ===")
sim_module.unpack_photonsim_params = make_patched_loader(V2_SIREN)

data_sim_a = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

pred_sim_a = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

mins_a = []
for evt in range(3):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp = random_track(key, pd['energy'])
    pd = set_transform(pd, pp)
    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim_a(pp, key_d, pd))
    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim_a(pp, key_p)
    losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
              for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    mins_a.append(cm)
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")
print(f"  avg = {np.mean(mins_a):+.3f}")

# ── Test B: Clean SIREN weights ──────────────────────────
print("\n=== B: Clean SIREN weights (swap test) ===")
sim_module.unpack_photonsim_params = make_patched_loader(CLEAN_SIREN)

# Data sim uses ROOT photons, not SIREN — so only pred_sim needs the swap
# But setup_event_simulator always loads photonsim params, so we patch both
data_sim_b = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

pred_sim_b = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

sim_module.unpack_photonsim_params = _orig_load

mins_b = []
for evt in range(3):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp = random_track(key, pd['energy'])
    pd = set_transform(pd, pp)
    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim_b(pp, key_d, pd))
    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim_b(pp, key_p)
    losses = [float(cone_time_loss(data_q, pred_t, data_t, jnp.array(t), tau=0.23))
              for t in t0_vals]
    cm = t0_vals[np.argmin(losses)]
    mins_b.append(cm)
    print(f"  evt {evt}: cone_min@t0={cm:+.2f}")
print(f"  avg = {np.mean(mins_b):+.3f}")

# ── Compare timing directly ──────────────────────────────
print("\n=== C: Direct timing comparison (same track, different SIREN) ===")
pd = load_and_pad(0)
key = jax.random.PRNGKey(42)
pp = random_track(key, pd['energy'])

key_p = jax.random.PRNGKey(42)
pred_q_a, pred_t_a = pred_sim_a(pp, key_p)
pred_q_b, pred_t_b = pred_sim_b(pp, key_p)

both = (pred_q_a > 0) & (pred_q_b > 0)
n = int(jnp.sum(both))
diff = pred_t_a[both] - pred_t_b[both]
print(f"  {n} jointly-hit sensors")
print(f"  pred_t(v2_siren) - pred_t(clean_siren):")
print(f"    mean={float(jnp.mean(diff)):.4f}  std={float(jnp.std(diff)):.4f}")
for p in [10, 23, 50, 77, 90]:
    print(f"    {p}th pct: {float(jnp.percentile(diff, p)):+.4f}")
