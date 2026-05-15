"""
100 TeV cascade with high-sigma overlap for variance reduction.

Compares temperature=0.2 (narrow, current default) vs temperature=18
(σ=3m, wide kernel for variance reduction). Both should give the same
expected charge (mean preserved) but the high-sigma run should have
more DOMs with nonzero charge and lower per-DOM variance.

Run: python string/test_high_sigma_cascade.py
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
from lucid.simulation.photon_step import photon_iteration_update_factors_safe

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")


def run_cascade(temperature, n_photons=100_000, K=10, seed=42):
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)

    t0_build = time.perf_counter()
    prop = create_string_propagator(det, sp, det.S_radius, temperature=temperature)
    build_time = time.perf_counter() - t0_build

    NUM_SENSORS = det.n_sensors
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    vertex = jnp.array([0.0, 0.0, z_mid])
    direction = jnp.array([0.0, 0.0, 1.0])
    energy_mev = 100e6
    SPEED = 0.2254

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
    wall_rr = jnp.float32(0.0)
    sensor_rr = jnp.float32(0.0)
    dom_charges = jnp.zeros(NUM_SENSORS)

    t0_sim = time.perf_counter()
    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, n_photons)

        inside_flag = jnp.array(det.bounds_check(np.array(positions)))
        safe_pos = jnp.where(inside_flag[:, None], positions, jax.lax.stop_gradient(positions))
        safe_dir = jnp.where(inside_flag[:, None], dirs, jax.lax.stop_gradient(dirs))

        result = prop(safe_pos, safe_dir)
        inside_sensor = result['inside_sensor']
        hit_sensor = jnp.max(inside_sensor, axis=0)
        hit_positions = result['positions']
        normals = result['normals']
        depositions = result['sensor_weights']
        sensor_indices = result['sensor_indices']
        surface_distances = jnp.linalg.norm(hit_positions - positions, axis=1) - 1e-6

        (new_pos, new_dir, new_times, detect_probs, refl_att, cont_factors) = jax.vmap(
            photon_iteration_update_factors_safe,
            in_axes=(0,0,0,0,0,0,None,None,0,0,0,None)
        )(positions, dirs, times_state, surface_distances, normals,
          scatter_lengths, wall_rr, sensor_rr, absorption_lengths,
          hit_sensor, rng_keys, SPEED)

        inside_det = jnp.array(det.bounds_check(np.array(new_pos)))
        safe_cont = jnp.where(inside_det, cont_factors, 0.0)
        phys_int = intensities * survival
        det_factors = detect_probs * refl_att
        iter_w = depositions * phys_int[None, :] * det_factors[None, :]
        iter_idx = sensor_indices
        valid = (iter_idx >= 0) & (iter_idx < NUM_SENSORS)
        dom_charges = dom_charges.at[jnp.where(valid, iter_idx, 0).ravel()].add(
            jnp.where(valid, iter_w, 0.0).ravel())
        survival = survival * safe_cont
        positions = new_pos
        dirs = new_dir
        times_state = new_times

    sim_time = time.perf_counter() - t0_sim

    charges = np.array(dom_charges)
    hit_mask = charges > 1e-10
    return {
        'temperature': temperature,
        'sigma_m': temperature * det.S_radius,
        'n_photons': n_photons,
        'K': K,
        'build_time': build_time,
        'sim_time': sim_time,
        'n_doms_hit': int(hit_mask.sum()),
        'total_charge': float(charges.sum()),
        'charges': charges,
        'hit_mask': hit_mask,
        'max_charge': float(charges.max()),
        'median_hit_charge': float(np.median(charges[hit_mask])) if hit_mask.any() else 0,
    }


if __name__ == "__main__":
    temps = [0.2, 3.0, 10.0, 18.0, 30.0]
    n_photons = 100_000
    K = 10

    print(f"100 TeV cascade, {n_photons} photons, K={K}")
    print(f"{'temp':>6s}  {'σ (m)':>6s}  {'σ/r':>5s}  {'DOMs hit':>9s}  "
          f"{'total Q':>12s}  {'max Q':>12s}  {'med Q':>12s}  "
          f"{'build(s)':>8s}  {'sim(s)':>7s}")
    print("-" * 95)

    results = []
    for temp in temps:
        r = run_cascade(temp, n_photons=n_photons, K=K)
        results.append(r)
        print(f"{temp:6.1f}  {r['sigma_m']:6.3f}  {temp:5.1f}  {r['n_doms_hit']:9d}  "
              f"{r['total_charge']:12.1f}  {r['max_charge']:12.1f}  {r['median_hit_charge']:12.4f}  "
              f"{r['build_time']:8.1f}  {r['sim_time']:7.1f}")

    # Compare: total charge should be similar across temperatures (mean preserved)
    print(f"\n{'='*60}")
    charges_baseline = results[0]['total_charge']
    print(f"Mean preservation check (total charge relative to temperature=0.2):")
    for r in results:
        ratio = r['total_charge'] / charges_baseline if charges_baseline > 0 else 0
        print(f"  temp={r['temperature']:5.1f}: total_Q={r['total_charge']:12.1f}, "
              f"ratio={ratio:.4f}, DOMs_hit={r['n_doms_hit']}")
