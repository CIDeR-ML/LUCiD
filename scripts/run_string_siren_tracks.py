"""
Batch SIREN track simulation: 20 events, 1M photons each, GPU.

Uses the fast string simulator (brute-force + lax.scan) for ~18× speedup
over the old DDA/hash pipeline.

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
from lucid.propagation.string.fast import create_fast_string_simulator
from lucid.siren.core import build_cherenkov_context
from lucid.siren.training.inference import SIRENPredictor
from lucid.sources.siren_rays import make_cherenkov_surrogate_fn
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
                                  ray_fn, model_params):
    ray_vectors, ray_origins, photon_intensities = ray_fn(
        track_origin, track_direction, SIREN_ENERGY, n_photons,
        model_params, key,
    )
    offsets = ray_origins - track_origin[None, :]
    ray_origins_scaled = track_origin[None, :] + F * offsets
    photon_intensities_scaled = photon_intensities * F
    return ray_vectors, ray_origins_scaled, photon_intensities_scaled


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading detector and SIREN...")
    det = StringTelescope.from_npz(SIMPLE_NPZ)

    sim = create_fast_string_simulator(
        det, det.S_radius, temperature=TEMPERATURE,
        lambda_abs=100.0, lambda_scat=30.0,
        speed_of_light=SPEED)

    data_dir = os.path.join(base_dir_path(), 'data', 'water', 'muon')
    model_path = os.path.join(data_dir, 'siren_training', 'trained_model', 'photonsim_siren')
    predictor = SIRENPredictor(model_path)
    ctx = build_cherenkov_context(predictor)
    ray_fn = make_cherenkov_surrogate_fn(ctx)
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
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2

    for ev in range(N_EVENTS):
        t0 = time.perf_counter()

        key = jax.random.PRNGKey(ev * 1000 + 42)
        key, k1, k2, gen_key, sim_key = jax.random.split(key, 5)

        origin_offset = jax.random.uniform(k1, (3,), minval=-50.0, maxval=50.0)
        origin_offset = origin_offset.at[2].set(origin_offset[2] * 5)
        track_origin = jnp.array([0.0, 0.0, z_mid]) + origin_offset

        dir_raw = jax.random.normal(k2, (3,))
        track_direction = dir_raw / (jnp.linalg.norm(dir_raw) + 1e-10)

        ray_vectors, ray_origins, photon_weights = generate_siren_track_photons(
            track_origin, track_direction, N_PHOTONS, gen_key, ray_fn, model_params)

        dom_charges, dom_time_weighted = sim(
            ray_origins, ray_vectors, photon_weights, K, sim_key)
        jax.block_until_ready(dom_charges)

        elapsed = time.perf_counter() - t0
        event_times.append(elapsed)

        charges_np = np.array(dom_charges)
        time_weighted_np = np.array(dom_time_weighted)
        hit_mask = charges_np > 1e-6
        dom_times_np = np.where(hit_mask, time_weighted_np / (charges_np + 1e-30), 0.0)

        np.savez(os.path.join(OUTPUT_DIR, f'event_{ev:03d}.npz'),
                 track_origin=np.array(track_origin),
                 track_direction=np.array(track_direction),
                 dom_charges=charges_np,
                 dom_times=dom_times_np,
                 hit_mask=hit_mask)

        hit_ids = np.where(hit_mask)[0].tolist()
        hit_charges = charges_np[hit_mask].tolist()
        hit_times = dom_times_np[hit_mask].tolist()
        all_events.append({
            'event_idx': ev,
            'track_origin': np.array(track_origin).tolist(),
            'track_direction': np.array(track_direction).tolist(),
            'n_doms_hit': int(hit_mask.sum()),
            'total_charge': float(charges_np.sum()),
            'hit_dom_ids': hit_ids,
            'hit_charges': hit_charges,
            'hit_times_ns': hit_times,
        })

        tag = " (JIT)" if ev == 0 else ""
        print(f"  Event {ev:2d}: {int(hit_mask.sum()):4d} DOMs, "
              f"Q={charges_np.sum():10.0f}, {elapsed:.1f}s{tag}")

    with open(os.path.join(OUTPUT_DIR, 'events.json'), 'w') as f:
        json.dump(all_events, f)

    jit_time = event_times[0]
    run_times = event_times[1:]
    mean_time = np.mean(run_times) if run_times else jit_time
    std_time = np.std(run_times) if run_times else 0

    print(f"\n{'='*70}")
    print(f"Timing:")
    print(f"  Event 0 (includes JIT): {jit_time:.1f}s")
    print(f"  Events 1-{N_EVENTS-1} (post-JIT): {mean_time:.2f} +/- {std_time:.2f} s/event")
    print(f"  Total wall time: {sum(event_times):.1f}s")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
