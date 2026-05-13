"""
Batch SIREN track simulation: 20 events, 1M photons each, GPU.

Generates random track orientations through IceCube-86, simulates with
the SIREN F=30 hack, and saves per-event hit data as NPZ files.

Run: python string/batch_siren_tracks.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import jax
import jax.numpy as jnp
import numpy as np

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.propagator import create_string_propagator
from lucid.simulation.photon_step_volume import photon_step_volume
from lucid.simulation.optics import normalize
from lucid.siren.core import create_photonsim_siren_grid
from lucid.siren.training.inference import SIRENPredictor
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.utils import base_dir_path

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "siren_tracks")

F = 30
N_PHOTONS = 1_000_000
K = 15
TEMPERATURE = 0.2
N_EVENTS = 20
SIREN_ENERGY = 2000.0  # MeV
SPEED = 0.2254


def generate_siren_track_photons(track_origin, track_direction, n_photons, key,
                                  grid_data, model_params):
    ray_vectors, ray_origins, photon_weights = photonsim_differentiable_get_rays(
        track_origin, track_direction, SIREN_ENERGY, n_photons,
        grid_data, model_params, key,
    )
    offsets = ray_origins - track_origin[None, :]
    ray_origins_scaled = track_origin[None, :] + F * offsets
    photon_weights_scaled = photon_weights * F
    return ray_vectors, ray_origins_scaled, photon_weights_scaled


def simulate_one_event(det, prop, grid_data, model_params, event_idx, master_key):
    NUM_SENSORS = det.n_sensors

    key = jax.random.PRNGKey(event_idx * 1000 + int(master_key))

    # Random track: origin near center, random direction (upper hemisphere bias)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    origin_offset = jax.random.uniform(k1, (3,), minval=-50.0, maxval=50.0)
    origin_offset = origin_offset.at[2].set(origin_offset[2] * 5)  # larger z range
    track_origin = jnp.array([0.0, 0.0, z_mid]) + origin_offset

    dir_raw = jax.random.normal(k2, (3,))
    track_direction = dir_raw / (jnp.linalg.norm(dir_raw) + 1e-10)

    # Generate photons
    key, gen_key = jax.random.split(key)
    ray_vectors, ray_origins, photon_weights = generate_siren_track_photons(
        track_origin, track_direction, N_PHOTONS, gen_key, grid_data, model_params)

    # Propagate
    positions = ray_origins
    dirs = ray_vectors
    times_state = jnp.zeros(N_PHOTONS)
    survival = jnp.ones(N_PHOTONS)
    intensities = photon_weights
    scatter_lengths = jnp.full(N_PHOTONS, 30.0)
    absorption_lengths = jnp.full(N_PHOTONS, 100.0)
    dom_charges = jnp.zeros(NUM_SENSORS)
    dom_time_weighted = jnp.zeros(NUM_SENSORS)

    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, N_PHOTONS)

        inside_flag = det.bounds_check(np.array(positions))
        inside_flag = jnp.array(inside_flag)
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

        # Approximate hit times from sensor distances
        sensor_times_ns = sensor_dists / SPEED
        weighted_times = sensor_times_ns.T * jnp.where(
            per_dom_charges > 1e-10, 1.0, 0.0)  # only count actual hits

        # Per-DOM hit times: photon current time + distance to DOM / speed
        dom_hit_times = times_state[:, None] + sensor_dists.T / SPEED  # (n_photons, max_dom) in ns

        idx_T = sensor_indices.T
        valid = (idx_T >= 0) & (idx_T < NUM_SENSORS)
        dom_charges = dom_charges.at[jnp.where(valid, idx_T, 0).ravel()].add(
            jnp.where(valid, weighted_charges, 0.0).ravel())

        # Charge-weighted time accumulation
        weighted_times = dom_hit_times * weighted_charges
        dom_time_weighted = dom_time_weighted.at[jnp.where(valid, idx_T, 0).ravel()].add(
            jnp.where(valid, weighted_times, 0.0).ravel())

        survival = survival * safe_cont
        positions = new_pos
        dirs = new_dir
        times_state = new_times

    charges_np = np.array(dom_charges)
    time_weighted_np = np.array(dom_time_weighted)
    hit_mask = charges_np > 1e-6

    # Mean hit time per DOM = charge-weighted time / charge
    dom_times_np = np.where(hit_mask, time_weighted_np / (charges_np + 1e-30), 0.0)

    return {
        'event_idx': event_idx,
        'track_origin': np.array(track_origin),
        'track_direction': np.array(track_direction),
        'dom_charges': charges_np,
        'dom_times': dom_times_np,
        'hit_mask': hit_mask,
        'n_doms_hit': int(hit_mask.sum()),
        'total_charge': float(charges_np.sum()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading detector and SIREN...")
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=TEMPERATURE)

    data_dir = os.path.join(base_dir_path(), 'data', 'water', 'muon')
    model_path = os.path.join(data_dir, 'siren_training', 'trained_model', 'photonsim_siren')
    predictor = SIRENPredictor(model_path)
    grid_data = create_photonsim_siren_grid(predictor)
    model_params = predictor.params

    # Save detector geometry for the viewer
    det_info = {
        'string_anchors': det.string_anchors.tolist(),
        'string_tops': det.string_tops.tolist(),
        'dom_positions': det.all_points.tolist(),
        'n_str': det.n_str,
        'n_sensors': det.n_sensors,
        'n_dom_per_str': det.n_dom_per_str_np.tolist(),
        'envelope_radius': det.envelope_radius,
        'envelope_z_min': det.envelope_z_min,
        'envelope_z_max': det.envelope_z_max,
        'sensor_radius': det.S_radius,
    }
    with open(os.path.join(OUTPUT_DIR, 'detector.json'), 'w') as f:
        json.dump(det_info, f)

    print(f"\nSimulating {N_EVENTS} events: {N_PHOTONS:,} photons, K={K}, F={F}")
    print(f"Backend: {jax.default_backend()}")
    print(f"{'='*70}")

    all_events = []
    event_times = []

    for ev in range(N_EVENTS):
        t0 = time.perf_counter()
        result = simulate_one_event(det, prop, grid_data, model_params, ev, 42)
        jax.block_until_ready(result['dom_charges'])
        elapsed = time.perf_counter() - t0
        event_times.append(elapsed)

        # Save per-event data
        np.savez(os.path.join(OUTPUT_DIR, f'event_{ev:03d}.npz'),
                 track_origin=result['track_origin'],
                 track_direction=result['track_direction'],
                 dom_charges=result['dom_charges'],
                 dom_times=result['dom_times'],
                 hit_mask=result['hit_mask'])

        # Summary for viewer JSON
        hit_ids = np.where(result['hit_mask'])[0].tolist()
        hit_charges = result['dom_charges'][result['hit_mask']].tolist()
        hit_times = result['dom_times'][result['hit_mask']].tolist()
        all_events.append({
            'event_idx': ev,
            'track_origin': result['track_origin'].tolist(),
            'track_direction': result['track_direction'].tolist(),
            'n_doms_hit': result['n_doms_hit'],
            'total_charge': result['total_charge'],
            'hit_dom_ids': hit_ids,
            'hit_charges': hit_charges,
            'hit_times_ns': hit_times,
        })

        tag = " (JIT)" if ev == 0 else ""
        print(f"  Event {ev:2d}: {result['n_doms_hit']:4d} DOMs, "
              f"Q={result['total_charge']:10.0f}, {elapsed:.1f}s{tag}")

    # Save events summary for viewer
    with open(os.path.join(OUTPUT_DIR, 'events.json'), 'w') as f:
        json.dump(all_events, f)

    jit_time = event_times[0]
    run_times = event_times[1:]
    mean_time = np.mean(run_times) if run_times else jit_time
    std_time = np.std(run_times) if run_times else 0

    print(f"\n{'='*70}")
    print(f"Timing:")
    print(f"  Event 0 (includes JIT): {jit_time:.1f}s")
    print(f"  Events 1-{N_EVENTS-1} (post-JIT): {mean_time:.2f} ± {std_time:.2f} s/event")
    print(f"  Total wall time: {sum(event_times):.1f}s")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
