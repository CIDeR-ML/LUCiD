#!/usr/bin/env python3
"""
TEST 16: Run the CLEAN simulation to see what data_t - pred_t looks like.

The v2 bias is +0.233. If clean has no bias, the difference must be in
the simulation code (not SIREN weights, which we already ruled out).

Run clean's data_sim and pred_sim with the same tracks and compare
the timing distribution.
"""
import sys, os
# Point to clean codebase
CLEAN_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'clean_run', 'LUCiD')
sys.path.insert(0, CLEAN_BASE)

import jax
import jax.numpy as jnp
import numpy as np
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.detector_params import ParticleParams
from tools.generate import read_photon_data_from_photonsim
from tools.optimization.losses import cone_time_loss

GEOM = os.path.join(CLEAN_BASE, 'config/SK_geom_config.json')
PHYS_exists = os.path.exists(os.path.join(CLEAN_BASE, 'config/SK_physics_config.json'))
print(f"Clean GEOM exists: {os.path.exists(GEOM)}")
print(f"Clean PHYS exists: {PHYS_exists}")

# Use v2's data file (ROOT) since clean may not have its own
DATA = os.path.join(os.path.dirname(__file__), '..', 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')
print(f"DATA exists: {os.path.exists(DATA)}")

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

# Check if clean uses physics_config or not
# Clean's setup_event_simulator signature may differ
import inspect
sig = inspect.signature(setup_event_simulator)
print(f"\nClean setup_event_simulator params: {list(sig.parameters.keys())}")
