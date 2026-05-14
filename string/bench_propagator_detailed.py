"""
Detailed propagation benchmark: isolate prop vs volume step timing.

Runs at K=1,2,3 with 1M photons to confirm propagation is the bottleneck,
then breaks down propagation into DDA+match vs ray-sphere intersection.

Run: python string/bench_propagator_detailed.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import numpy as np

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.propagator import create_string_propagator
from lucid.simulation.photon_step_volume import photon_step_volume
from lucid.simulation.optics import normalize

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")

N_PHOTONS = 1_000_000
SPEED = 0.2254


def bench_full_loop(det, prop, K_values):
    """Run full K-loop at various K, timing prop vs volume step."""
    sp = jnp.array(det.all_points)
    NUM_SENSORS = det.n_sensors

    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    key = jax.random.PRNGKey(42)

    for K in K_values:
        key_run = jax.random.PRNGKey(42)
        key_run, k1, k2 = jax.random.split(key_run, 3)

        origins = jax.random.uniform(k1, (N_PHOTONS, 3),
                                     minval=jnp.array([-200, -200, z_mid - 400]),
                                     maxval=jnp.array([200, 200, z_mid + 400]))
        dirs_raw = jax.random.normal(k2, (N_PHOTONS, 3))
        dirs = dirs_raw / (jnp.linalg.norm(dirs_raw, axis=1, keepdims=True) + 1e-10)

        positions = origins
        directions = dirs
        times_state = jnp.zeros(N_PHOTONS)
        survival = jnp.ones(N_PHOTONS)
        intensities = jnp.ones(N_PHOTONS)
        scatter_lengths = jnp.full(N_PHOTONS, 30.0)
        absorption_lengths = jnp.full(N_PHOTONS, 100.0)
        dom_charges = jnp.zeros(NUM_SENSORS)

        # Warmup JIT on first K
        if K == K_values[0]:
            key_run, subkey = jax.random.split(key_run)
            rng_keys = jax.random.split(subkey, N_PHOTONS)
            _ = prop(positions, directions)
            jax.block_until_ready(_['sensor_weights'])
            print("  JIT warmup done")

        t_prop_total = 0
        t_vol_total = 0
        t_other_total = 0

        key_run = jax.random.PRNGKey(42)
        positions = origins
        directions = dirs
        times_state = jnp.zeros(N_PHOTONS)
        survival = jnp.ones(N_PHOTONS)

        for k in range(K):
            key_run, subkey = jax.random.split(key_run)
            rng_keys = jax.random.split(subkey, N_PHOTONS)

            inside_flag = det.bounds_check(np.array(positions))
            inside_flag = jnp.array(inside_flag)
            safe_pos = jnp.where(inside_flag[:, None], positions, jax.lax.stop_gradient(positions))
            safe_dir = jnp.where(inside_flag[:, None], directions, jax.lax.stop_gradient(directions))

            # Time propagation
            t0 = time.perf_counter()
            result = prop(safe_pos, safe_dir)
            jax.block_until_ready(result['sensor_weights'])
            t_prop = time.perf_counter() - t0
            t_prop_total += t_prop

            depositions = result['sensor_weights']
            sensor_indices = result['sensor_indices']
            sensor_dists = result['sensor_distances'].squeeze(-1)
            hit_positions = result['positions']
            segment_lengths = jnp.linalg.norm(hit_positions - positions, axis=1)
            segment_lengths = jnp.maximum(segment_lengths, 1.0)

            # Time volume step
            t0 = time.perf_counter()
            (new_pos, new_dir, new_times, per_dom_charges, cont_factors) = jax.vmap(
                photon_step_volume,
                in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0, None)
            )(positions, directions, times_state,
              sensor_dists, depositions,
              scatter_lengths, absorption_lengths, segment_lengths, rng_keys, SPEED)
            jax.block_until_ready(new_pos)
            t_vol = time.perf_counter() - t0
            t_vol_total += t_vol

            # Time other (charge accumulation)
            t0 = time.perf_counter()
            inside_det = jnp.array(det.bounds_check(np.array(new_pos)))
            safe_cont = jnp.where(inside_det, cont_factors, 0.0)
            physical_intensities = intensities * survival
            weighted_charges = per_dom_charges * physical_intensities[:, None]
            idx_T = sensor_indices.T
            valid = (idx_T >= 0) & (idx_T < NUM_SENSORS)
            dom_charges = dom_charges.at[jnp.where(valid, idx_T, 0).ravel()].add(
                jnp.where(valid, weighted_charges, 0.0).ravel())
            jax.block_until_ready(dom_charges)
            t_other = time.perf_counter() - t0
            t_other_total += t_other

            survival = survival * safe_cont
            positions = new_pos
            directions = new_dir
            times_state = new_times

        total = t_prop_total + t_vol_total + t_other_total
        print(f"  K={K:2d}: total={total:.3f}s | prop={t_prop_total:.3f}s ({t_prop_total/total*100:.0f}%) "
              f"| vol={t_vol_total:.3f}s ({t_vol_total/total*100:.0f}%) "
              f"| other={t_other_total:.3f}s ({t_other_total/total*100:.0f}%)")
        print(f"         per-K: prop={t_prop_total/K:.4f}s, vol={t_vol_total/K:.4f}s")


def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"N_PHOTONS: {N_PHOTONS:,}")

    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=0.2)

    print(f"\nSizing: {prop.sizing}")
    print(f"  max_cells_per_segment = {prop.sizing.max_cells_per_segment}")
    print(f"  max_dom_per_segment = {prop.sizing.max_dom_per_segment}")
    print(f"  n_dom_snap = {prop.sizing.n_dom_snap}")
    print(f"  max_str_per_cell = {prop.sizing.max_str_per_cell}")

    print(f"\nTiming breakdown (post-JIT):")
    bench_full_loop(det, prop, [1, 2, 3, 5, 10, 20])


if __name__ == "__main__":
    main()
