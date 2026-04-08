#!/usr/bin/env python3
"""
TEST 4: Isolate which physics parameter causes the t0 bias.

Creates modified physics configs and runs each with the v2 simulator.
Compares cone_time_loss minimum location for each configuration.

Configs tested:
  A) v2 default: wall_ref=0.2, sensor_ref=0.1
  B) matched reflection: wall_ref=0.2, sensor_ref=0.2 (like clean)
  C) no reflection difference test: wall_ref=0.1, sensor_ref=0.1
"""
import sys, os, json, tempfile
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
ORIG_PHYS = os.path.join(BASE, 'config/SK_physics_config.json')
DATA = os.path.join(BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')
DIAG = os.path.dirname(__file__)

detector = generate_detector(GEOM)
det_pts = jnp.array(detector.all_points)
NUM_DET = len(det_pts)

# Load original config as template
with open(ORIG_PHYS) as f:
    orig_cfg = json.load(f)

def write_config(wall_ref, sensor_ref, suffix):
    cfg = dict(orig_cfg)
    cfg['wall_reflection_rate'] = wall_ref
    cfg['sensor_reflection_rate'] = sensor_ref
    path = os.path.join(DIAG, f'phys_{suffix}.json')
    with open(path, 'w') as f:
        json.dump(cfg, f)
    return path

configs = {
    'A_v2_default':       write_config(0.2, 0.1, 'A'),
    'B_matched_0.2':      write_config(0.2, 0.2, 'B'),
    'C_both_0.1':         write_config(0.1, 0.1, 'C'),
}

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
    z = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
    pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
    th = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
    ph = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
    return ParticleParams(energy=jnp.array(float(energy)),
                          position=pos, theta=th, phi=ph, t0=jnp.array(0.))

N_EVENTS = 5
t0_vals = np.linspace(-1.0, 1.0, 21)

for label, phys_path in configs.items():
    with open(phys_path) as f:
        cfg = json.load(f)
    print(f"\n{'='*70}")
    print(f"Config: {label}")
    print(f"  wall_ref={cfg['wall_reflection_rate']}, sensor_ref={cfg['sensor_reflection_rate']}")
    print(f"{'='*70}")

    data_sim = setup_event_simulator(
        GEOM, 300_000, temperature=0.0, K=20,
        is_data=True, is_calibration=False,
        physics_config=phys_path, default_detector_params=True)

    pred_sim = setup_event_simulator(
        GEOM, 300_000, temperature=0.10, K=9,
        is_data=False, max_sensors_per_cell=4,
        physics_config=phys_path, default_detector_params=True)

    cone_min_t0s = []

    for evt in range(N_EVENTS):
        pd = load_and_pad(evt)
        key = jax.random.PRNGKey(42 + evt)
        pp = random_track(key, pd['energy'])
        pd = set_transform(pd, pp)

        key_d = jax.random.PRNGKey(100 + evt)
        data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))

        key_p = jax.random.PRNGKey(42)
        pred_q, pred_t = pred_sim(pp, key_p)

        # Sweep t0 — track cone_time_loss only
        cone_losses = []
        combined_losses = []
        for t0_val in t0_vals:
            t0 = jnp.array(t0_val)
            tl = float(cone_time_loss(data_q, pred_t, data_t, t0, tau=0.23))
            vl = float(origin_time_loss(pp.position, det_pts, data_t, data_q, t0))
            cl = float(counts_loss(data_q, pred_q))
            comb = float(jnp.sqrt((vl+1e-6)*(cl+1e-6)*(tl+1e-6)))
            cone_losses.append(tl)
            combined_losses.append(comb)

        cone_losses = np.array(cone_losses)
        combined_losses = np.array(combined_losses)
        cone_min = t0_vals[np.argmin(cone_losses)]
        comb_min = t0_vals[np.argmin(combined_losses)]
        cone_min_t0s.append(cone_min)

        print(f"  evt {evt}: cone_min@t0={cone_min:+.2f}  combined_min@t0={comb_min:+.2f}")

    cone_min_t0s = np.array(cone_min_t0s)
    print(f"  >> mean cone_min t0 = {np.mean(cone_min_t0s):+.3f}")
