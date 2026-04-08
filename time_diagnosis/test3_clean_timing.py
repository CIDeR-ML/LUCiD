#!/usr/bin/env python3
"""
TEST 3: Same timing comparison but using the CLEAN simulation.
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
from tools.optimization.losses import origin_time_loss, cone_time_loss, counts_loss

GEOM = os.path.join(CLEAN_BASE, 'config/SK_geom_config.json')
DATA = os.path.join(CLEAN_BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

detector = generate_detector(GEOM)
det_pts  = jnp.array(detector.all_points)
NUM_DET  = len(det_pts)
Nphot = 150_000
K = 7

detector_params = (
    jnp.array(50.),    # scatter_length
    jnp.array(0.2),    # reflection_rate
    jnp.array(50.),    # absorption_length
    jnp.array(0.001),  # tau_gs (unused)
)

# Build simulators using the clean API (setup_event_simulator)
data_sim_fn = setup_event_simulator(
    GEOM, Nphot, temperature=0.0, K=20,
    is_data=True, is_calibration=False)

pred_sim_fn = setup_event_simulator(
    GEOM, Nphot, temperature=0.10, K=K,
    is_data=False, max_sensors_per_cell=4)

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
    return pd

def set_transform_clean(pd, position, direction):
    """Set rotation/translation for clean notebook API."""
    orig = jnp.array([0.,0.,1.])
    tgt  = direction / (jnp.linalg.norm(direction)+1e-8)
    ax   = jnp.cross(orig, tgt)
    an   = jnp.linalg.norm(ax)
    ax   = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang  = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax
    pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = position
    pd['apply_translation'] = jnp.array(True)
    return pd

print(f"Clean: {NUM_DET} sensors, Nphot={Nphot}, K={K}")
print(f"Running 5 events...\n")

for evt in range(5):
    pd = load_and_pad(evt)
    energy_val = float(pd['energy'])

    # Generate random track (same keys as v2 test)
    key = jax.random.PRNGKey(42 + evt)
    k1,k2,k3,k4,k5 = jax.random.split(key, 5)
    frac = 0.6
    r = jax.random.uniform(k1, minval=0, maxval=detector.r*frac)
    tp = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    z  = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
    pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
    theta = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
    phi   = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
    direction = spherical_to_cartesian(theta, phi)

    pd = set_transform_clean(pd, pos, direction)

    # Clean API: particle_params = (energy, position, direction_angles)
    true_params = (jnp.array(energy_val), pos, jnp.array([theta, phi]))

    key_d = jax.random.PRNGKey(100 + evt)
    data_q, data_t = jax.lax.stop_gradient(
        data_sim_fn(true_params, detector_params, key_d, pd))

    key_p = jax.random.PRNGKey(42)
    pred_q, pred_t = pred_sim_fn(true_params, detector_params, key_p)

    data_hit = data_q > 0
    pred_hit = pred_q > 0
    both_hit = data_hit & pred_hit

    n_data = int(jnp.sum(data_hit))
    n_pred = int(jnp.sum(pred_hit))
    n_both = int(jnp.sum(both_hit))

    dt = data_t[both_hit] - pred_t[both_hit]
    print(f"--- Event {evt} ---")
    print(f"  data hit: {n_data}  pred hit: {n_pred}  both: {n_both}")
    print(f"  time diff (data-pred): mean={float(jnp.mean(dt)):.4f}  "
          f"median={float(jnp.median(dt)):.4f}  std={float(jnp.std(dt)):.4f} ns")

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
