"""
End-to-end test: 100 TeV cascade in IceCube-86 with string propagation.

Generates Cherenkov photons from a parametric cascade, propagates them
through K scattering iterations, and collects DOM hits. Verifies that
the full pipeline (cascade emission → string propagator → photon_step
→ hit accumulation) produces a physically sensible hit pattern.

Run: python string/test_cascade_e2e.py
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
from lucid.sources.cascade import generate_cascade_photons, cherenkov_angle
from lucid.simulation.photon_step import photon_iteration_update_factors_safe

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")


def run_cascade_simulation(
    energy_tev=100.0,
    n_photons=10000,
    K=10,
    vertex=None,
    direction=None,
    seed=42,
):
    """Run a full cascade simulation and return hit data.

    Parameters
    ----------
    energy_tev : float      cascade energy in TeV
    n_photons : int         number of photon samples
    K : int                 scatter iterations
    vertex : (3,) or None   cascade vertex (default: center of IceCube)
    direction : (3,) or None  cascade direction (default: upgoing)
    seed : int              random seed

    Returns
    -------
    dict with:
        'dom_charges': (N_sensors,) accumulated charge per DOM
        'dom_hit_count': (N_sensors,) number of hit photons per DOM
        'strings_hit': set of string indices with at least one hit
        'total_detected': float — total detected photon weight
        'n_photons': int
        'K': int
        'energy_tev': float
    """
    # Load detector
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=0.2)
    NUM_SENSORS = det.n_sensors

    # Default vertex: center of IceCube array
    if vertex is None:
        z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
        vertex = jnp.array([0.0, 0.0, z_mid])
    else:
        vertex = jnp.array(vertex)
    if direction is None:
        direction = jnp.array([0.0, 0.0, 1.0])  # upgoing
    else:
        direction = jnp.array(direction)

    energy_mev = energy_tev * 1e6
    SPEED = 0.2254  # m/ns in water (from DetectorGeometry)

    # Generate cascade photons
    key = jax.random.PRNGKey(seed)
    key, gen_key = jax.random.split(key)
    origins, directions, weights = generate_cascade_photons(
        vertex, direction, energy_mev, n_photons, gen_key,
        n_medium=1.33, is_hadronic=False,
    )

    print(f"  Cascade: E={energy_tev} TeV, n_photons={n_photons}, K={K}")
    print(f"  Vertex: {np.array(vertex)}")
    print(f"  Per-photon weight: {float(weights[0]):.0f}")
    print(f"  Cherenkov angle: {float(jnp.degrees(cherenkov_angle(1.33))):.1f} deg")

    # Initialize photon state
    positions = origins
    dirs = directions
    times_state = jnp.zeros(n_photons)
    survival = jnp.ones(n_photons)
    intensities = weights

    scatter_lengths = jnp.full(n_photons, 30.0)    # ~30m in water
    absorption_lengths = jnp.full(n_photons, 100.0)  # ~100m in water
    wall_reflection_rate = jnp.float32(0.0)
    sensor_reflection_rate = jnp.float32(0.0)

    # Accumulate hits
    dom_charges = jnp.zeros(NUM_SENSORS)

    t0 = time.perf_counter()

    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, n_photons)

        inside_flag = jnp.array(det.bounds_check(np.array(positions)))
        safe_positions = jnp.where(inside_flag[:, None], positions,
                                   jax.lax.stop_gradient(positions))
        safe_directions = jnp.where(inside_flag[:, None], dirs,
                                    jax.lax.stop_gradient(dirs))

        result = prop(safe_positions, safe_directions)

        depositions = result['sensor_weights']      # (max_dom, n_rays)
        sensor_indices = result['sensor_indices']    # (max_dom, n_rays)
        inside_sensor = result['inside_sensor']      # (max_dom, n_rays)
        hit_positions = result['positions']
        normals = result['normals']

        hit_sensor = jnp.max(inside_sensor, axis=0)
        surface_distances = jnp.linalg.norm(hit_positions - positions, axis=1) - 1e-6

        (new_positions, new_directions, new_times,
         detect_probs, reflection_attenuations,
         continuing_factors) = jax.vmap(
            photon_iteration_update_factors_safe,
            in_axes=(0, 0, 0, 0, 0, 0, None, None, 0, 0, 0, None)
        )(positions, dirs, times_state,
          surface_distances, normals,
          scatter_lengths, wall_reflection_rate, sensor_reflection_rate,
          absorption_lengths,
          hit_sensor, rng_keys, SPEED)

        inside_det = jnp.array(det.bounds_check(np.array(new_positions)))
        safe_continuing = jnp.where(inside_det, continuing_factors, 0.0)

        physical_intensities = intensities * survival
        detected_factors = detect_probs * reflection_attenuations
        iter_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
        iter_indices = sensor_indices

        valid_mask = (iter_indices >= 0) & (iter_indices < NUM_SENSORS)
        flat_weights = jnp.where(valid_mask, iter_weights, 0.0)
        flat_indices = jnp.where(valid_mask, iter_indices, 0)

        dom_charges = dom_charges.at[flat_indices.ravel()].add(flat_weights.ravel())

        survival = survival * safe_continuing
        positions = new_positions
        dirs = new_directions
        times_state = new_times

    elapsed = time.perf_counter() - t0

    # Compute hit statistics
    dom_charges_np = np.array(dom_charges)
    hit_mask = dom_charges_np > 0
    n_hit_doms = int(hit_mask.sum())
    total_detected = float(dom_charges_np.sum())

    # Which strings were hit
    strings_hit = set()
    for dom_id in np.where(hit_mask)[0]:
        for si in range(det.n_str):
            n = det.n_dom_per_str_np[si]
            gids = det.dom_global_ids[si, :n]
            if dom_id in gids:
                strings_hit.add(si)
                break

    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    DOMs hit: {n_hit_doms} / {NUM_SENSORS}")
    print(f"    Strings hit: {len(strings_hit)} / {det.n_str}")
    print(f"    Total detected weight: {total_detected:.1f}")
    print(f"    Survival (mean): {float(survival.mean()):.6f}")

    if n_hit_doms > 0:
        hit_charges = dom_charges_np[hit_mask]
        print(f"    Charge per hit DOM: min={hit_charges.min():.2f}, "
              f"max={hit_charges.max():.2f}, median={np.median(hit_charges):.2f}")

        # Distance from vertex to each hit DOM
        hit_dom_positions = np.array(det.all_points[hit_mask])
        vertex_np = np.array(vertex)
        dists = np.linalg.norm(hit_dom_positions - vertex_np, axis=1)
        print(f"    Hit DOM distance from vertex: "
              f"min={dists.min():.0f}m, max={dists.max():.0f}m, "
              f"median={np.median(dists):.0f}m")

    return {
        'dom_charges': dom_charges_np,
        'dom_hit_count': hit_mask.astype(int),
        'strings_hit': strings_hit,
        'total_detected': total_detected,
        'n_photons': n_photons,
        'K': K,
        'energy_tev': energy_tev,
        'elapsed_s': elapsed,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("100 TeV cascade in IceCube-86 — end-to-end test")
    print("=" * 60)

    result = run_cascade_simulation(
        energy_tev=100.0,
        n_photons=5000,
        K=8,
    )

    # Basic sanity checks
    ok = True
    if result['total_detected'] <= 0:
        print("\nFAIL: no photons detected at all")
        ok = False
    if len(result['strings_hit']) < 2:
        print(f"\nWARN: only {len(result['strings_hit'])} strings hit — "
              f"expected multiple for 100 TeV cascade at array center")
    if result['total_detected'] > 0:
        print("\nPASS: cascade produced detectable hits")

    print(f"\n{'=' * 60}")
    sys.exit(0 if ok else 1)
