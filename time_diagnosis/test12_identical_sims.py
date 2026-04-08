#!/usr/bin/env python3
"""
TEST 12: Verify that two pred_sim instances produce identical output.
If they don't, the cone_time_loss results from test11 are invalid.
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

BASE = os.path.join(os.path.dirname(__file__), '..')
GEOM = os.path.join(BASE, 'config/SK_geom_config.json')
PHYS = os.path.join(BASE, 'config/SK_physics_config.json')
DATA = os.path.join(BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

detector = generate_detector(GEOM)

def random_track(key, energy):
    k1,k2,k3,k4,k5 = jax.random.split(key, 5)
    frac = 0.6
    r = jax.random.uniform(k1, minval=0, maxval=detector.r*frac)
    tp = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    z = jax.random.uniform(k3, minval=-detector.H/2*frac, maxval=detector.H/2*frac)
    pos = jnp.array([r*jnp.cos(tp), r*jnp.sin(tp), z])
    th = jax.random.uniform(k4, minval=0.01, maxval=jnp.pi-0.01)
    ph = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)
    return ParticleParams(energy=jnp.array(1050.),
                          position=pos, theta=th, phi=ph, t0=jnp.array(0.))

print("Building two identical pred simulators...")
sim_a = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

sim_b = setup_event_simulator(
    GEOM, 300_000, temperature=0.10, K=9,
    is_data=False, max_sensors_per_cell=4,
    physics_config=PHYS, default_detector_params=True)

key = jax.random.PRNGKey(42)
pp = random_track(key, 1050.)

# Same key for both
k = jax.random.PRNGKey(42)
q_a, t_a = sim_a(pp, k)
q_b, t_b = sim_b(pp, k)

print(f"\nCharges:")
print(f"  max |q_a - q_b| = {float(jnp.max(jnp.abs(q_a - q_b))):.6e}")
print(f"  mean q_a = {float(jnp.mean(q_a)):.6f}  mean q_b = {float(jnp.mean(q_b)):.6f}")
print(f"  q_a sum = {float(jnp.sum(q_a)):.4f}  q_b sum = {float(jnp.sum(q_b)):.4f}")

both_hit = (q_a > 0) & (q_b > 0)
n_both = int(jnp.sum(both_hit))
print(f"\nTimes ({n_both} jointly-hit sensors):")
if n_both > 0:
    diff = t_a[both_hit] - t_b[both_hit]
    print(f"  max |t_a - t_b| = {float(jnp.max(jnp.abs(diff))):.6e}")
    print(f"  mean diff = {float(jnp.mean(diff)):.6e}")
    print(f"  mean t_a = {float(jnp.mean(t_a[both_hit])):.4f}  mean t_b = {float(jnp.mean(t_b[both_hit])):.4f}")
print(f"  q_a > 0: {int(jnp.sum(q_a > 0))}  q_b > 0: {int(jnp.sum(q_b > 0))}")

# Now test: calling the SAME sim twice with same key
print("\n=== Calling sim_a twice with same key ===")
q_a1, t_a1 = sim_a(pp, k)
q_a2, t_a2 = sim_a(pp, k)
both_aa = (q_a1 > 0) & (q_a2 > 0)
n_aa = int(jnp.sum(both_aa))
print(f"  max |q_diff| = {float(jnp.max(jnp.abs(q_a1 - q_a2))):.6e}")
if n_aa > 0:
    diff_aa = t_a1[both_aa] - t_a2[both_aa]
    print(f"  max |t_diff| = {float(jnp.max(jnp.abs(diff_aa))):.6e}")
