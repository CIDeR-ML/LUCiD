#!/usr/bin/env python3
"""
TEST 2: Run v2 data + prediction simulators on same events.
Compare observed_times vs simulated_time directly, and see how
cone_time_loss and origin_time_loss each contribute to t0 bias.
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
from tools.optimization.losses import origin_time_loss, cone_time_loss, counts_loss

BASE = os.path.join(os.path.dirname(__file__), '..')
GEOM = os.path.join(BASE, 'config/SK_geom_config.json')
PHYS = os.path.join(BASE, 'config/SK_physics_config.json')
DATA = os.path.join(BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

detector = generate_detector(GEOM)
det_pts  = jnp.array(detector.all_points)
NUM_DET  = len(det_pts)

def load_and_pad(entry_idx):
    pd = read_photon_data_from_photonsim(DATA, entry_idx)
    N  = len(pd['photon_origins'])
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
    tgt  = pp.direction / (jnp.linalg.norm(pp.direction)+1e-8)
    ax   = jnp.cross(orig, tgt)
    an   = jnp.linalg.norm(ax)
    ax   = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang  = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax
    pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = pp.position
    pd['apply_translation'] = jnp.array(True)
    return pd

def random_track(key, energy):
    k1,k2,k3,k4,k5 = jax.random.split(key, 5)
    frac = 0.6
    r = jax.random.uniform(k1, minval=0, maxval=detector.r*frac)
    tp = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    z  = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
    pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
    th  = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
    ph  = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
    return ParticleParams(energy=jnp.array(float(energy)),
                          position=pos, theta=th, phi=ph, t0=jnp.array(0.))

# ── build simulators ─────────────────────────────────────────────────
print("Building simulators...")
data_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.0, K=20,
    is_data=True, is_calibration=False,
    physics_config=PHYS, default_detector_params=True)

pred_sim = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)
print("Done.\n")

# ── run on a few events ──────────────────────────────────────────────
N_EVENTS = 5
print(f"Running {N_EVENTS} events...\n")

for evt in range(N_EVENTS):
    pd = load_and_pad(evt)
    key = jax.random.PRNGKey(42 + evt)
    pp  = random_track(key, pd['energy'])
    pd  = set_transform(pd, pp)

    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim(pp, key_p)

    # Hit masks
    data_hit = data_q > 0
    pred_hit = pred_q > 0
    both_hit = data_hit & pred_hit

    n_data = int(jnp.sum(data_hit))
    n_pred = int(jnp.sum(pred_hit))
    n_both = int(jnp.sum(both_hit))

    # Time differences at jointly-hit sensors
    dt = data_t[both_hit] - pred_t[both_hit]

    print(f"--- Event {evt} ---")
    print(f"  data hit: {n_data}  pred hit: {n_pred}  both: {n_both}")
    print(f"  time diff (data-pred): mean={float(jnp.mean(dt)):.4f}  "
          f"median={float(jnp.median(dt)):.4f}  std={float(jnp.std(dt)):.4f} ns")

    # Now compute each loss component at t0=0 and sweep small t0 range
    # to find which loss component drives the bias
    pos = pp.position
    for t0_val in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        t0 = jnp.array(t0_val)
        vl = float(origin_time_loss(pos, det_pts, data_t, data_q, t0))
        cl = float(counts_loss(data_q, pred_q))
        tl = float(cone_time_loss(data_q, pred_t, data_t, t0, tau=0.23))
        combined = float(jnp.sqrt((vl+1e-6)*(cl+1e-6)*(tl+1e-6)))
        if t0_val == 0.0:
            print(f"  t0={t0_val:+.2f}: vertex={vl:.6f}  counts={cl:.6f}  "
                  f"cone={tl:.6f}  combined={combined:.6f}")
        else:
            print(f"  t0={t0_val:+.2f}: vertex={vl:.6f}  cone={tl:.6f}  combined={combined:.6f}")
    print()
