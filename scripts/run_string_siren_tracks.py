"""
Batch SIREN track simulation for IceCube-86 string detector.

Uses setup_event_simulator in track mode with the standard string propagator.
Generates muon and electron events at IceCube-relevant energies.

Run:
    python scripts/run_string_siren_tracks.py
    python scripts/run_string_siren_tracks.py --particle electron
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import ParticleParams
from lucid.siren.core import build_cherenkov_context
from lucid.siren.training.inference import SIRENPredictor
from lucid.sources.siren_rays import _siren_weights
from lucid.simulation.simulator import setup_event_simulator
from lucid.utils import unpack_siren_params

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
GEOM_CONFIG = os.path.join(CONFIG_DIR, "IceCube86_full_geom_config.json")
PHYS_CONFIG = os.path.join(CONFIG_DIR, "IceCube86_ice_physics_config.json")

N_PHOTONS = 1_000_000
K = 20
N_EVENTS = 20

ENERGIES_GEV = [5, 10, 15, 20, 30, 40, 50, 60, 70, 70,
                80, 80, 80, 90, 90, 90, 100, 100, 100, 100]


def compute_track_lengths(particle, energies_mev):
    """Compute p95 emission extent per energy by integrating the SIREN grid."""
    sp = unpack_siren_params(particle, 'ice')
    predictor = SIRENPredictor(sp['siren_model_path'])
    ctx = build_cherenkov_context(predictor, sp['ray_sampling'])

    n_a, n_s = 20, 60
    angles = jnp.linspace(0.1, 3.0, n_a)
    s_vals = jnp.linspace(0.01, 0.99, n_s)
    AA, SS = jnp.meshgrid(angles, s_vals, indexing='ij')
    a_flat, s_flat = AA.ravel(), SS.ravel()

    results = {}
    for e_mev in energies_mev:
        w = _siren_weights(ctx, predictor.params, float(e_mev), a_flat, s_flat)
        profile = w.reshape(n_a, n_s).sum(axis=0)
        cum = jnp.cumsum(profile) / profile.sum()
        p95_s = float(s_vals[int(jnp.searchsorted(cum, 0.95))])
        smax_m = float(ctx.s_max_fn(jnp.array(float(e_mev)))) / 1000
        results[e_mev] = max(p95_s * smax_m, 5.0)
    return results


def main():
    parser = argparse.ArgumentParser(description="IceCube-86 SIREN track simulation")
    parser.add_argument("--particle", default="muon", choices=["muon", "electron"])
    args = parser.parse_args()

    particle = args.particle
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", f"siren_tracks_{particle}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Building simulator: {particle}, {N_PHOTONS:,} photons, K={K}")
    sim = setup_event_simulator(
        GEOM_CONFIG,
        n_photons=N_PHOTONS,
        temperature=None,
        K=K,
        is_calibration=False,
        detector_type='string',
        physics_config=PHYS_CONFIG,
        default_detector_params=True,
        hit_mode='aggregated',
        wavelength_mode=False,
        particle=particle,
        use_expected_value=True,
    )

    det = sim.det_geom.detector
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2

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
    with open(os.path.join(output_dir, 'detector.json'), 'w') as f:
        json.dump(det_info, f)

    # Precompute track lengths per unique energy
    unique_mev = sorted(set(e * 1000.0 for e in ENERGIES_GEV))
    print("Computing track lengths...")
    lengths_mev = compute_track_lengths(particle, unique_mev)
    track_lengths = {int(e / 1000): l for e, l in lengths_mev.items()}
    for e_gev in sorted(track_lengths):
        print(f"  {e_gev:4d} GeV -> {track_lengths[e_gev]:.1f} m")

    print(f"\nSimulating {N_EVENTS} {particle} events")
    print(f"Backend: {jax.default_backend()}")
    print(f"{'='*70}")

    all_events = []
    event_times = []

    for ev in range(N_EVENTS):
        t0 = time.perf_counter()
        energy_gev = ENERGIES_GEV[ev]
        energy_mev = energy_gev * 1000.0

        key = jax.random.PRNGKey(ev * 1000 + 42)
        key, k1, k2 = jax.random.split(key, 3)

        offset = jax.random.uniform(k1, (3,), minval=-80.0, maxval=80.0)
        offset = offset.at[2].set(offset[2] * 4)
        origin = jnp.array([0.0, 0.0, z_mid]) + offset

        dir_raw = jax.random.normal(k2, (3,))
        direction = dir_raw / (jnp.linalg.norm(dir_raw) + 1e-10)

        pp = ParticleParams.from_cartesian(
            energy=energy_mev, position=origin.tolist(),
            direction=direction.tolist(), t0=0.0)

        charges, times = sim(pp, key)
        jax.block_until_ready(charges)

        elapsed = time.perf_counter() - t0
        event_times.append(elapsed)

        charges_np = np.array(charges)
        times_np = np.array(times)
        hit_mask = charges_np > 1e-6
        dom_times_np = np.where(hit_mask, times_np, 0.0)

        np.savez(os.path.join(output_dir, f'event_{ev:03d}.npz'),
                 track_origin=np.array(origin),
                 track_direction=np.array(direction),
                 dom_charges=charges_np,
                 dom_times=dom_times_np,
                 hit_mask=hit_mask)

        hit_ids = np.where(hit_mask)[0].tolist()
        all_events.append({
            'event_idx': ev,
            'track_origin': np.array(origin).tolist(),
            'track_direction': np.array(direction).tolist(),
            'energy_gev': energy_gev,
            'track_length_m': track_lengths[energy_gev],
            'n_doms_hit': int(hit_mask.sum()),
            'total_charge': float(charges_np.sum()),
            'hit_dom_ids': hit_ids,
            'hit_charges': charges_np[hit_mask].tolist(),
            'hit_times_ns': dom_times_np[hit_mask].tolist(),
        })

        tag = " (JIT)" if ev == 0 else ""
        print(f"  Event {ev:2d} [{energy_gev:4d} GeV]: {int(hit_mask.sum()):4d} DOMs, "
              f"Q={charges_np.sum():10.1f}, {elapsed:.1f}s{tag}")

    with open(os.path.join(output_dir, 'events.json'), 'w') as f:
        json.dump(all_events, f)

    jit_time = event_times[0]
    run_times = event_times[1:]
    mean_time = np.mean(run_times) if run_times else jit_time
    std_time = np.std(run_times) if run_times else 0

    print(f"\n{'='*70}")
    print(f"Timing: JIT={jit_time:.1f}s, post-JIT={mean_time:.2f}+/-{std_time:.2f} s/event")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
