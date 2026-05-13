"""
High-energy track in IceCube-86 using SIREN with F=30 distance scaling.

Uses the existing 2 GeV muon SIREN model but stretches the track to
~300m by scaling the emission distances by F=30. Photon weights are
scaled by F to preserve photons-per-meter.

Run: python string/run_siren_track.py
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
from lucid.simulation.photon_step_volume import photon_step_volume
from lucid.simulation.optics import normalize
from lucid.siren.core import create_photonsim_siren_grid
from lucid.siren.training.inference import SIRENPredictor
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.utils import base_dir_path

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")

F = 30  # distance scaling factor: 10m track -> 300m track


def generate_siren_track_photons(track_origin, track_direction, energy_mev, n_photons, key,
                                  grid_data, model_params, F=30):
    """Generate photons from SIREN with distance scaling.

    Uses the SIREN at the given energy (within its training range),
    then scales the emission positions along the track by factor F.
    Weights are scaled by F to preserve photons per meter.
    """
    ray_vectors, ray_origins, photon_weights = photonsim_differentiable_get_rays(
        track_origin, track_direction, energy_mev, n_photons,
        grid_data, model_params, key,
    )

    # Scale the emission positions: stretch along the track direction
    offsets = ray_origins - track_origin[None, :]
    ray_origins_scaled = track_origin[None, :] + F * offsets

    # Scale weights to preserve photons per meter
    photon_weights_scaled = photon_weights * F

    return ray_vectors, ray_origins_scaled, photon_weights_scaled


def run_siren_track(n_photons=100_000, K=15, temperature=0.2, seed=42):
    # Load detector
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=temperature)
    NUM_SENSORS = det.n_sensors
    SPEED = 0.2254

    # Load SIREN model
    data_dir = os.path.join(base_dir_path(), 'data', 'water', 'muon')
    model_path = os.path.join(data_dir, 'siren_training', 'trained_model', 'photonsim_siren')
    predictor = SIRENPredictor(model_path)
    grid_data = create_photonsim_siren_grid(predictor)
    model_params = predictor.params

    # Track: through the center of IceCube, slightly tilted
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    track_origin = jnp.array([0.0, 0.0, z_mid])
    track_direction = jnp.array([0.3, 0.1, 0.95])  # slightly off-vertical
    track_direction = track_direction / jnp.linalg.norm(track_direction)

    # Use SIREN at 2 GeV (max training energy, longest track ~10m)
    siren_energy = 2000.0  # MeV

    print(f"Track: origin={np.array(track_origin)}, dir={np.array(track_direction)}")
    print(f"SIREN energy: {siren_energy} MeV, F={F}, effective track: ~{F*10:.0f}m")
    print(f"n_photons={n_photons}, K={K}, temperature={temperature}")

    # Generate photons from SIREN with F scaling
    key = jax.random.PRNGKey(seed)
    key, gen_key = jax.random.split(key)

    t0_gen = time.perf_counter()
    ray_vectors, ray_origins, photon_weights = generate_siren_track_photons(
        track_origin, track_direction, siren_energy, n_photons, gen_key,
        grid_data, model_params, F=F,
    )
    gen_time = time.perf_counter() - t0_gen
    print(f"SIREN generation: {gen_time:.1f}s")

    # Check photon distribution
    offsets = ray_origins - track_origin[None, :]
    along_track = jnp.sum(offsets * track_direction[None, :], axis=1)
    print(f"Emission positions along track: "
          f"min={float(along_track.min()):.1f}m, max={float(along_track.max()):.1f}m")
    print(f"Weight range: min={float(photon_weights.min()):.1f}, "
          f"max={float(photon_weights.max()):.1f}, mean={float(photon_weights.mean()):.1f}")

    # Propagate with volume photon step
    positions = ray_origins
    dirs = ray_vectors
    times_state = jnp.zeros(n_photons)
    survival = jnp.ones(n_photons)
    intensities = photon_weights
    scatter_lengths = jnp.full(n_photons, 30.0)
    absorption_lengths = jnp.full(n_photons, 100.0)
    dom_charges = jnp.zeros(NUM_SENSORS)
    per_k_charge = []

    t0_prop = time.perf_counter()
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
        k_charge = float(jnp.sum(jnp.where(valid, weighted_charges, 0.0)))
        per_k_charge.append(k_charge)

        dom_charges = dom_charges.at[jnp.where(valid, idx_T, 0).ravel()].add(
            jnp.where(valid, weighted_charges, 0.0).ravel())

        survival = survival * safe_cont
        positions = new_pos
        dirs = new_dir
        times_state = new_times

        n_inside = int(inside_flag.sum())
        if k < 5 or k == K - 1:
            print(f"  K={k+1:2d}: charge={k_charge:12.0f}, inside={n_inside}, "
                  f"survival_mean={float(survival.mean()):.6f}")

    prop_time = time.perf_counter() - t0_prop

    charges = np.array(dom_charges)
    hit_mask = charges > 1e-6
    n_hit = int(hit_mask.sum())

    print(f"\n{'='*60}")
    print(f"Results ({prop_time:.1f}s propagation)")
    print(f"{'='*60}")
    print(f"  DOMs hit: {n_hit} / {NUM_SENSORS}")
    print(f"  Total charge: {charges.sum():.0f}")

    if n_hit > 0:
        hit_charges = charges[hit_mask]
        print(f"  Charge per hit DOM: min={hit_charges.min():.1f}, "
              f"median={np.median(hit_charges):.1f}, max={hit_charges.max():.1f}")

        hit_positions = det.all_points[hit_mask]
        vertex_np = np.array(track_origin)
        dists = np.linalg.norm(hit_positions - vertex_np, axis=1)
        print(f"  Hit DOM distances: min={dists.min():.0f}m, "
              f"median={np.median(dists):.0f}m, max={dists.max():.0f}m")

    # Strings hit
    strings_hit = set()
    for dom_id in np.where(hit_mask)[0]:
        for si in range(det.n_str):
            n = det.n_dom_per_str_np[si]
            if dom_id in det.dom_global_ids[si, :n]:
                strings_hit.add(si)
                break
    print(f"  Strings hit: {len(strings_hit)} / {det.n_str}")

    # Per-K profile
    print(f"\nPer-K charge:")
    total = sum(per_k_charge)
    cum = 0
    for k, q in enumerate(per_k_charge):
        cum += q
        frac = q / total if total > 0 else 0
        cum_frac = cum / total if total > 0 else 0
        bar = '#' * int(frac * 30)
        print(f"  K={k+1:2d}: {q:12.0f} ({frac:5.1%}) cum={cum_frac:5.1%} {bar}")

    print(f"{'='*60}")


if __name__ == "__main__":
    run_siren_track()
