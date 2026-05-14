"""
Isolate physics differences between old and new simulators.

1. K=1 comparison: only K=0 charges matter (no scatter divergence)
2. Per-K charge profile comparison
3. Segment length comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.propagator import create_string_propagator
from lucid.propagation.string.fast import create_fast_string_simulator
from lucid.simulation.photon_step_volume import photon_step_volume

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")

N = 50_000
SPEED = 0.2254
TEMPERATURE = 0.2


def run_old_per_k(det, origins, dirs, weights, K, key):
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=TEMPERATURE)
    NUM_SENSORS = det.n_sensors

    positions = origins
    directions = dirs
    times_state = jnp.zeros(N)
    survival = jnp.ones(N)
    intensities = weights
    scatter_lengths = jnp.full(N, 30.0)
    absorption_lengths = jnp.full(N, 100.0)

    per_k_charges = []
    per_k_seg_len = []
    per_k_survival = []

    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, N)

        inside_flag = jnp.array(det.bounds_check(np.array(positions)))
        safe_pos = jnp.where(inside_flag[:, None], positions, jax.lax.stop_gradient(positions))
        safe_dir = jnp.where(inside_flag[:, None], directions, jax.lax.stop_gradient(directions))

        result = prop(safe_pos, safe_dir)
        depositions = result['sensor_weights']
        sensor_indices = result['sensor_indices']
        sensor_dists = result['sensor_distances'].squeeze(-1)

        hit_positions = result['positions']
        segment_lengths = jnp.linalg.norm(hit_positions - positions, axis=1)
        segment_lengths = jnp.maximum(segment_lengths, 1.0)

        per_k_seg_len.append(float(jnp.median(segment_lengths)))

        (new_pos, new_dir, new_times, per_dom_charges, cont_factors) = jax.vmap(
            photon_step_volume,
            in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0, None)
        )(positions, directions, times_state,
          sensor_dists, depositions,
          scatter_lengths, absorption_lengths, segment_lengths, rng_keys, SPEED)

        inside_det = jnp.array(det.bounds_check(np.array(new_pos)))
        safe_cont = jnp.where(inside_det, cont_factors, 0.0)

        physical_intensities = intensities * survival
        weighted_charges = per_dom_charges * physical_intensities[:, None]

        idx_T = sensor_indices.T
        valid = (idx_T >= 0) & (idx_T < NUM_SENSORS)
        k_charge = float(jnp.sum(jnp.where(valid, weighted_charges, 0.0)))
        per_k_charges.append(k_charge)
        per_k_survival.append(float(survival.mean()))

        survival = survival * safe_cont
        positions = new_pos
        directions = new_dir
        times_state = new_times

    return per_k_charges, per_k_seg_len, per_k_survival


def run_new_per_k(det, origins, dirs, weights, K_values, key):
    """Run new simulator at multiple K values to extract per-K charges."""
    sim = create_fast_string_simulator(
        det, det.S_radius, temperature=TEMPERATURE,
        lambda_abs=100.0, lambda_scat=30.0,
        speed_of_light=SPEED, n_closest=4, n_dom_snap=2)

    results = {}
    for K in K_values:
        dom_q, _ = sim(origins, dirs, weights, K, key)
        results[K] = float(jnp.sum(dom_q))

    per_k_charges = []
    prev = 0.0
    for K in sorted(results.keys()):
        per_k_charges.append(results[K] - prev)
        prev = results[K]

    return per_k_charges


def main():
    det = StringTelescope.from_npz(SIMPLE_NPZ)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    origins = jax.random.uniform(k1, (N, 3),
                                 minval=jnp.array([-100, -100, z_mid - 200]),
                                 maxval=jnp.array([100, 100, z_mid + 200]))
    dirs_raw = jax.random.normal(k2, (N, 3))
    dirs = dirs_raw / (jnp.linalg.norm(dirs_raw, axis=1, keepdims=True) + 1e-10)
    weights = jnp.ones(N)

    K = 10

    print(f"N={N}, K={K}")
    print(f"\n--- OLD per-K ---")
    old_key = jax.random.PRNGKey(99)
    old_per_k, old_seg, old_surv = run_old_per_k(det, origins, dirs, weights, K, old_key)

    print(f"\n--- NEW per-K (via subtraction) ---")
    new_key = jax.random.PRNGKey(99)
    K_vals = list(range(1, K + 1))
    new_per_k = run_new_per_k(det, origins, dirs, weights, K_vals, new_key)

    print(f"\n{'='*70}")
    print(f"{'K':>3} | {'OLD charge':>12} | {'NEW charge':>12} | {'ratio':>8} | {'OLD seg_len':>12} | {'OLD survival':>12}")
    print(f"{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}")
    old_total = 0
    new_total = 0
    for k in range(K):
        old_q = old_per_k[k]
        new_q = new_per_k[k] if k < len(new_per_k) else 0
        old_total += old_q
        new_total += new_q
        ratio = new_q / (old_q + 1e-30)
        seg = old_seg[k] if k < len(old_seg) else 0
        surv = old_surv[k] if k < len(old_surv) else 0
        print(f"{k:3d} | {old_q:12.4f} | {new_q:12.4f} | {ratio:8.2f} | {seg:12.1f} | {surv:12.6f}")

    print(f"\nTotal OLD: {old_total:.4f}")
    print(f"Total NEW: {new_total:.4f}")
    print(f"Ratio: {new_total / (old_total + 1e-30):.4f}")

    # K=0 comparison: should be similar regardless of random stream
    print(f"\nK=0 charge: OLD={old_per_k[0]:.4f}, NEW={new_per_k[0]:.4f}, "
          f"ratio={new_per_k[0]/(old_per_k[0]+1e-30):.4f}")


if __name__ == "__main__":
    main()
