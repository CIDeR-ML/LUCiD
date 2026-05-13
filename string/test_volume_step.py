"""
Test volume photon step with per-DOM survival.

Runs 100 TeV cascade through IceCube-86 at multiple temperatures using
the per-DOM survival model. Checks that total charge is approximately
preserved across temperatures (mean preservation).

Run: python string/test_volume_step.py
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


def run_cascade_volume(temperature, n_photons=50_000, K=8, seed=42):
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

    # Envelope exit distances for segment_length upper bound
    env_r_sq = det.envelope_radius ** 2
    env_z_min = det.envelope_z_min
    env_z_max = det.envelope_z_max

    t0 = time.perf_counter()

    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, n_photons)

        inside_flag = jnp.array(det.bounds_check(np.array(positions)))
        safe_pos = jnp.where(inside_flag[:, None], positions, jax.lax.stop_gradient(positions))
        safe_dir = jnp.where(inside_flag[:, None], dirs, jax.lax.stop_gradient(dirs))

        result = prop(safe_pos, safe_dir)
        depositions = result['sensor_weights']          # (max_dom, n_rays)
        sensor_indices = result['sensor_indices']       # (max_dom, n_rays)
        sensor_dists = result['sensor_distances'].squeeze(-1)  # (max_dom, n_rays)

        # Envelope exit as segment upper bound
        hit_positions = result['positions']
        segment_lengths = jnp.linalg.norm(hit_positions - positions, axis=1)
        segment_lengths = jnp.maximum(segment_lengths, 1.0)

        # Volume photon step: per-DOM survival
        def step_one_photon(pos, dir_, t, dom_dists, dom_overlaps, scat_l, abs_l, seg_l, rng):
            return photon_step_volume(
                pos, dir_, t, dom_dists, dom_overlaps,
                scat_l, abs_l, seg_l, rng, SPEED)

        (new_pos, new_dir, new_times, per_dom_charges, cont_factors) = jax.vmap(
            step_one_photon,
            in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0)
        )(positions, dirs, times_state,
          sensor_dists, depositions,
          scatter_lengths, absorption_lengths, segment_lengths, rng_keys)
        # per_dom_charges: (n_rays, max_dom)

        inside_det = jnp.array(det.bounds_check(np.array(new_pos)))
        safe_cont = jnp.where(inside_det, cont_factors, 0.0)

        # Accumulate per-DOM charges: weight by intensity and photon survival
        physical_intensities = intensities * survival  # (n_rays,)
        weighted_charges = per_dom_charges * physical_intensities[:, None]  # (n_rays, max_dom)

        # Transpose sensor_indices to (n_rays, max_dom) for accumulation
        idx_T = sensor_indices.T  # (n_rays, max_dom) — sensor_indices was (max_dom, n_rays)
        valid = (idx_T >= 0) & (idx_T < NUM_SENSORS)
        dom_charges = dom_charges.at[jnp.where(valid, idx_T, 0).ravel()].add(
            jnp.where(valid, weighted_charges, 0.0).ravel())

        survival = survival * safe_cont
        positions = new_pos
        dirs = new_dir
        times_state = new_times

    elapsed = time.perf_counter() - t0

    charges = np.array(dom_charges)
    hit_mask = charges > 1e-10
    return {
        'temperature': temperature,
        'sigma_m': temperature * det.S_radius,
        'n_doms_hit': int(hit_mask.sum()),
        'total_charge': float(charges.sum()),
        'max_charge': float(charges.max()),
        'elapsed': elapsed,
    }


if __name__ == "__main__":
    temps = [0.2, 1.0, 3.0, 10.0, 18.0, 30.0]
    n_photons = 50_000
    K = 8

    print(f"100 TeV cascade, {n_photons} photons, K={K}")
    print(f"Volume photon step with per-DOM survival (no reflection)")
    print(f"\n{'temp':>6s}  {'σ (m)':>6s}  {'σ/r':>5s}  {'DOMs hit':>9s}  "
          f"{'total Q':>14s}  {'max Q':>14s}  {'time(s)':>7s}")
    print("-" * 75)

    results = []
    for temp in temps:
        r = run_cascade_volume(temp, n_photons=n_photons, K=K)
        results.append(r)
        print(f"{temp:6.1f}  {r['sigma_m']:6.3f}  {temp:5.1f}  {r['n_doms_hit']:9d}  "
              f"{r['total_charge']:14.1f}  {r['max_charge']:14.1f}  {r['elapsed']:7.1f}")

    print(f"\n{'='*75}")
    baseline = results[0]['total_charge']
    print(f"Mean preservation (total charge relative to temperature=0.2):")
    for r in results:
        ratio = r['total_charge'] / baseline if baseline > 0 else 0
        print(f"  temp={r['temperature']:5.1f}: total_Q={r['total_charge']:14.1f}, "
              f"ratio={ratio:.4f}, DOMs_hit={r['n_doms_hit']}")
