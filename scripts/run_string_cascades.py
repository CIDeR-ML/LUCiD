"""
Batch cascade simulation using the full LUCiD simulator pipeline.

Uses setup_event_simulator with the string propagator + volume photon step.
Writes events.json + detector.json + per-event .npz under output/cascades_sim/.

Run: python scripts/run_string_cascades.py --n-events 20
(This is a heavy batch job: ~1M photons x K=20 per event.)
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
from lucid.simulation.simulator import setup_event_simulator
from lucid.sources.cascade import cascade_source

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
GEOM_CONFIG = os.path.join(CONFIG_DIR, "IceCube86_full_geom_config.json")
PHYS_CONFIG = os.path.join(CONFIG_DIR, "IceCube86_ice_physics_config.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "cascades_sim")

N_PHOTONS = 1_000_000
K = 20
N_EVENTS = 20
N_MEDIUM = 1.33

ENERGIES_GEV = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    det = StringTelescope.from_npz(os.path.join(CONFIG_DIR, "icecube86_simple.npz"))
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2

    # Save detector geometry for viewer
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

    # Build simulator ONCE — reused for all events
    print("Building simulator (setup_event_simulator)...")
    sim = setup_event_simulator(
        GEOM_CONFIG,
        n_photons=N_PHOTONS,
        temperature=0.2,
        K=K,
        is_calibration=True,
        detector_type='string',
        physics_config=PHYS_CONFIG,
        default_detector_params=True,
        hit_mode='aggregated',
        wavelength_mode=False,
    )

    print(f"\nSimulating {N_EVENTS} cascade events: {N_PHOTONS:,} photons, K={K}")
    print(f"Backend: {jax.default_backend()}")
    print(f"{'='*70}")

    all_events = []
    event_times = []

    for ev in range(N_EVENTS):
        t0 = time.perf_counter()

        key = jax.random.PRNGKey(ev * 1000 + 7)
        key, k1, k2 = jax.random.split(key, 3)

        energy_gev = ENERGIES_GEV[ev % len(ENERGIES_GEV)]
        energy_mev = energy_gev * 1000.0
        is_hadronic = (ev % 3 == 2)

        offset = jax.random.uniform(k1, (3,), minval=-80.0, maxval=80.0)
        offset = offset.at[2].set(offset[2] * 4)
        vertex = jnp.array([0.0, 0.0, z_mid]) + offset

        dir_raw = jax.random.normal(k2, (3,))
        direction = dir_raw / (jnp.linalg.norm(dir_raw) + 1e-10)

        src = cascade_source(
            position=vertex.tolist(),
            direction=direction.tolist(),
            energy_mev=float(energy_mev),
            n_medium=N_MEDIUM,
            is_hadronic=is_hadronic,
        )

        charges, times = sim(src, key)
        jax.block_until_ready(charges)

        elapsed = time.perf_counter() - t0
        event_times.append(elapsed)

        charges_np = np.array(charges)
        times_np = np.array(times)
        hit_mask = charges_np > 1e-6

        # For aggregated mode: charges and times are (n_sensors,)
        dom_times_np = np.where(hit_mask, times_np, 0.0)

        np.savez(os.path.join(OUTPUT_DIR, f'event_{ev:03d}.npz'),
                 vertex=np.array(vertex),
                 direction=np.array(direction),
                 dom_charges=charges_np,
                 dom_times=dom_times_np,
                 hit_mask=hit_mask)

        hit_ids = np.where(hit_mask)[0].tolist()
        hit_charges = charges_np[hit_mask].tolist()
        hit_times = dom_times_np[hit_mask].tolist()
        all_events.append({
            'event_idx': ev,
            'vertex': np.array(vertex).tolist(),
            'direction': np.array(direction).tolist(),
            'energy_gev': energy_gev,
            'is_hadronic': bool(is_hadronic),
            'n_doms_hit': int(hit_mask.sum()),
            'total_charge': float(charges_np.sum()),
            'hit_dom_ids': hit_ids,
            'hit_charges': hit_charges,
            'hit_times_ns': hit_times,
        })

        tag = " (JIT)" if ev == 0 else ""
        kind = "had" if is_hadronic else " em"
        print(f"  Event {ev:2d} [{kind} {energy_gev:6d} GeV]: {int(hit_mask.sum()):4d} DOMs, "
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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-events', type=int, default=N_EVENTS,
                    help='events per energy point (heavy: ~1M photons x K=20 each)')
    args = ap.parse_args()
    N_EVENTS = args.n_events
    main()
