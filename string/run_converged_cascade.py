"""
Converged 100 TeV cascade: n_photons=200k, K=20, temperature=0.2.

Uses the fast string simulator. Runs twice with different seeds for
convergence check.

Run: python string/run_converged_cascade.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.fast import create_fast_string_simulator
from lucid.sources.cascade import generate_cascade_photons

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")


def run_cascade(sim, det, n_photons, K, seed):
    SPEED = 0.2254

    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    vertex = jnp.array([0.0, 0.0, z_mid])
    direction = jnp.array([0.0, 0.0, 1.0])
    energy_mev = 100e6

    key = jax.random.PRNGKey(seed)
    key, gen_key, sim_key = jax.random.split(key, 3)
    origins, directions, weights = generate_cascade_photons(
        vertex, direction, energy_mev, n_photons, gen_key, n_medium=1.33)

    t0 = time.perf_counter()
    dom_charges, _ = sim(origins, directions, weights, K, sim_key)
    jax.block_until_ready(dom_charges)
    elapsed = time.perf_counter() - t0

    charges = np.array(dom_charges)
    hit_mask = charges > 1e-6
    return {
        'charges': charges,
        'hit_mask': hit_mask,
        'n_doms_hit': int(hit_mask.sum()),
        'total_charge': float(charges.sum()),
        'elapsed': elapsed,
    }


if __name__ == "__main__":
    n_photons = 200_000
    K = 20
    temperature = 0.2

    print(f"{'='*70}")
    print(f"100 TeV cascade — converged run")
    print(f"n_photons={n_photons}, K={K}, temperature={temperature}")
    print(f"{'='*70}")

    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sim = create_fast_string_simulator(
        det, det.S_radius, temperature=temperature,
        lambda_abs=100.0, lambda_scat=30.0, speed_of_light=0.2254)

    print("\nRun 1 (seed=42)...")
    r1 = run_cascade(sim, det, n_photons, K, seed=42)
    print(f"  {r1['elapsed']:.1f}s, {r1['n_doms_hit']} DOMs hit, "
          f"total_Q={r1['total_charge']:.1f}")

    print("\nRun 2 (seed=123)...")
    r2 = run_cascade(sim, det, n_photons, K, seed=123)
    print(f"  {r2['elapsed']:.1f}s, {r2['n_doms_hit']} DOMs hit, "
          f"total_Q={r2['total_charge']:.1f}")

    c1, c2 = r1['charges'], r2['charges']
    both_hit = r1['hit_mask'] & r2['hit_mask']
    n_both = both_hit.sum()

    if n_both > 0:
        rel_diff = np.abs(c1[both_hit] - c2[both_hit]) / (0.5 * (c1[both_hit] + c2[both_hit]) + 1e-10)
        print(f"\nConvergence (DOMs hit in both runs: {n_both}):")
        print(f"  Relative difference: median={np.median(rel_diff):.2%}, "
              f"mean={np.mean(rel_diff):.2%}, p90={np.percentile(rel_diff, 90):.2%}")

    q_mean = 0.5 * (r1['total_charge'] + r2['total_charge'])
    q_diff = abs(r1['total_charge'] - r2['total_charge'])
    print(f"\n  Total charge: {r1['total_charge']:.0f} vs {r2['total_charge']:.0f} "
          f"(diff={q_diff/q_mean:.1%})")
    print(f"  DOMs hit: {r1['n_doms_hit']} vs {r2['n_doms_hit']}")
    print(f"{'='*70}")
