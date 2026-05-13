"""
Converged 100 TeV cascade: n_photons=200k, K=20, temperature=0.2.

Runs the volume photon step with per-DOM survival and checks convergence
by comparing two independent runs with different seeds.

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
from lucid.propagation.string.propagator import create_string_propagator
from lucid.sources.cascade import generate_cascade_photons
from lucid.simulation.photon_step_volume import photon_step_volume
from lucid.simulation.optics import normalize

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")


def run_cascade(n_photons, K, temperature, seed):
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=temperature)
    NUM_SENSORS = det.n_sensors
    SPEED = 0.2254

    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    vertex = jnp.array([0.0, 0.0, z_mid])
    direction = jnp.array([0.0, 0.0, 1.0])
    energy_mev = 100e6

    key = jax.random.PRNGKey(seed)
    key, gen_key = jax.random.split(key)
    origins, directions, weights = generate_cascade_photons(
        vertex, direction, energy_mev, n_photons, gen_key, n_medium=1.33)

    positions = origins
    dirs = directions
    times_state = jnp.zeros(n_photons)
    survival = jnp.ones(n_photons)
    intensities = weights
    scatter_lengths = jnp.full(n_photons, 30.0)
    absorption_lengths = jnp.full(n_photons, 100.0)
    dom_charges = jnp.zeros(NUM_SENSORS)
    per_k_charge = []

    t0 = time.perf_counter()

    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, n_photons)

        inside_flag = jnp.array(det.bounds_check(np.array(positions)))
        safe_pos = jnp.where(inside_flag[:, None], positions, jax.lax.stop_gradient(positions))
        safe_dir = jnp.where(inside_flag[:, None], dirs, jax.lax.stop_gradient(dirs))

        result = prop(safe_pos, safe_dir)
        depositions = result['sensor_weights']
        sensor_indices = result['sensor_indices']
        sensor_dists = result['sensor_distances'].squeeze(-1)

        hit_positions = result['positions']
        segment_lengths = jnp.linalg.norm(hit_positions - positions, axis=1)
        segment_lengths = jnp.maximum(segment_lengths, 1.0)

        (new_pos, new_dir, new_times, per_dom_charges, cont_factors) = jax.vmap(
            photon_step_volume,
            in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0, None)
        )(positions, dirs, times_state,
          sensor_dists, depositions,
          scatter_lengths, absorption_lengths, segment_lengths, rng_keys, SPEED)

        inside_det = jnp.array(det.bounds_check(np.array(new_pos)))
        safe_cont = jnp.where(inside_det, cont_factors, 0.0)

        physical_intensities = intensities * survival
        weighted_charges = per_dom_charges * physical_intensities[:, None]

        idx_T = sensor_indices.T
        valid = (idx_T >= 0) & (idx_T < NUM_SENSORS)
        k_charge = jnp.sum(jnp.where(valid, weighted_charges, 0.0))
        per_k_charge.append(float(k_charge))

        dom_charges = dom_charges.at[jnp.where(valid, idx_T, 0).ravel()].add(
            jnp.where(valid, weighted_charges, 0.0).ravel())

        survival = survival * safe_cont
        positions = new_pos
        dirs = new_dir
        times_state = new_times

    elapsed = time.perf_counter() - t0

    charges = np.array(dom_charges)
    hit_mask = charges > 1e-6
    return {
        'charges': charges,
        'hit_mask': hit_mask,
        'n_doms_hit': int(hit_mask.sum()),
        'total_charge': float(charges.sum()),
        'per_k_charge': per_k_charge,
        'survival_mean': float(survival.mean()),
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

    # Run twice with different seeds for convergence check
    print("\nRun 1 (seed=42)...")
    r1 = run_cascade(n_photons, K, temperature, seed=42)
    print(f"  {r1['elapsed']:.1f}s, {r1['n_doms_hit']} DOMs hit, "
          f"total_Q={r1['total_charge']:.1f}")

    print("\nRun 2 (seed=123)...")
    r2 = run_cascade(n_photons, K, temperature, seed=123)
    print(f"  {r2['elapsed']:.1f}s, {r2['n_doms_hit']} DOMs hit, "
          f"total_Q={r2['total_charge']:.1f}")

    # Convergence: compare per-DOM charges
    c1, c2 = r1['charges'], r2['charges']
    both_hit = r1['hit_mask'] & r2['hit_mask']
    n_both = both_hit.sum()

    if n_both > 0:
        rel_diff = np.abs(c1[both_hit] - c2[both_hit]) / (0.5 * (c1[both_hit] + c2[both_hit]) + 1e-10)
        print(f"\nConvergence (DOMs hit in both runs: {n_both}):")
        print(f"  Relative difference: median={np.median(rel_diff):.2%}, "
              f"mean={np.mean(rel_diff):.2%}, p90={np.percentile(rel_diff, 90):.2%}")

    # Total charge convergence
    q_mean = 0.5 * (r1['total_charge'] + r2['total_charge'])
    q_diff = abs(r1['total_charge'] - r2['total_charge'])
    print(f"\n  Total charge: {r1['total_charge']:.0f} vs {r2['total_charge']:.0f} "
          f"(diff={q_diff/q_mean:.1%})")
    print(f"  DOMs hit: {r1['n_doms_hit']} vs {r2['n_doms_hit']}")

    # Per-K charge profile
    print(f"\nPer-K charge profile (run 1):")
    cumulative = 0
    total = sum(r1['per_k_charge'])
    for k, q in enumerate(r1['per_k_charge']):
        cumulative += q
        frac = q / total if total > 0 else 0
        cum_frac = cumulative / total if total > 0 else 0
        bar = '#' * int(frac * 40)
        print(f"  K={k+1:2d}: {q:12.0f} ({frac:5.1%}) cum={cum_frac:5.1%} {bar}")

    # Hit distance distribution
    if n_both > 0:
        det = StringTelescope.from_npz(SIMPLE_NPZ)
        z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
        vertex_np = np.array([0.0, 0.0, z_mid])
        hit_positions = det.all_points[r1['hit_mask']]
        dists = np.linalg.norm(hit_positions - vertex_np, axis=1)
        print(f"\nHit DOM distances from vertex:")
        print(f"  min={dists.min():.0f}m, median={np.median(dists):.0f}m, "
              f"max={dists.max():.0f}m")

    print(f"\nSurvival after K={K}: mean={r1['survival_mean']:.6f}")
    print(f"{'='*70}")
