"""Confirm the backward-DOM bug: photons depositing on DOMs behind them."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.fast import create_fast_string_simulator

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
det = StringTelescope.from_npz(os.path.join(CONFIG_DIR, "icecube86_simple.npz"))

# Single photon at string 0, pointing AWAY from it
string0_xy = det.string_anchors[0, :2]
z_mid = (det.envelope_z_min + det.envelope_z_max) / 2

# Place photon 1m from string 0, pointing AWAY
origin = jnp.array([string0_xy[0] + 1.0, string0_xy[1], z_mid])
# Direction pointing away from string 0
direction = jnp.array([1.0, 0.0, 0.0])  # +x, away from string

origins = origin[None, :]
dirs = direction[None, :]
weights = jnp.ones(1)

sim = create_fast_string_simulator(
    det, det.S_radius, temperature=0.2,
    lambda_abs=100.0, lambda_scat=30.0,
    speed_of_light=0.2254, n_closest=4, n_dom_snap=2)

dom_q, _ = sim(origins, dirs, weights, 1, jax.random.PRNGKey(42))
charges = np.array(dom_q)
hit_mask = charges > 1e-10

print(f"Photon at {np.array(origin)}, dir={np.array(direction)}")
print(f"String 0 at xy={string0_xy}")
print(f"DOMs hit: {hit_mask.sum()}")
if hit_mask.sum() > 0:
    hit_ids = np.where(hit_mask)[0]
    hit_charges = charges[hit_mask]
    for dom_id, q in zip(hit_ids, hit_charges):
        dom_pos = det.all_points[dom_id]
        vec_to_dom = dom_pos - np.array(origin)
        dot_forward = np.dot(vec_to_dom, np.array(direction))
        print(f"  DOM {dom_id}: pos={dom_pos}, q={q:.6f}, "
              f"forward_dot={dot_forward:.1f}m {'BEHIND' if dot_forward < 0 else 'ahead'}")
