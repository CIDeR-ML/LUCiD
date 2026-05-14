"""
High-energy track in IceCube-86 using SIREN with F=30 distance scaling.

Uses the fast string simulator for the propagation loop.

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
from lucid.propagation.string.fast import create_fast_string_simulator
from lucid.siren.core import create_photonsim_siren_grid
from lucid.siren.training.inference import SIRENPredictor
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.utils import base_dir_path

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")

F = 30


def generate_siren_track_photons(track_origin, track_direction, energy_mev, n_photons, key,
                                  grid_data, model_params, F=30):
    ray_vectors, ray_origins, photon_weights = photonsim_differentiable_get_rays(
        track_origin, track_direction, energy_mev, n_photons,
        grid_data, model_params, key,
    )
    offsets = ray_origins - track_origin[None, :]
    ray_origins_scaled = track_origin[None, :] + F * offsets
    photon_weights_scaled = photon_weights * F
    return ray_vectors, ray_origins_scaled, photon_weights_scaled


def run_siren_track(n_photons=100_000, K=15, temperature=0.2, seed=42):
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    SPEED = 0.2254

    sim = create_fast_string_simulator(
        det, det.S_radius, temperature=temperature,
        lambda_abs=100.0, lambda_scat=30.0, speed_of_light=SPEED)

    data_dir = os.path.join(base_dir_path(), 'data', 'water', 'muon')
    model_path = os.path.join(data_dir, 'siren_training', 'trained_model', 'photonsim_siren')
    predictor = SIRENPredictor(model_path)
    grid_data = create_photonsim_siren_grid(predictor)
    model_params = predictor.params

    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    track_origin = jnp.array([0.0, 0.0, z_mid])
    track_direction = jnp.array([0.3, 0.1, 0.95])
    track_direction = track_direction / jnp.linalg.norm(track_direction)

    siren_energy = 2000.0

    print(f"Track: origin={np.array(track_origin)}, dir={np.array(track_direction)}")
    print(f"SIREN energy: {siren_energy} MeV, F={F}, effective track: ~{F*10:.0f}m")
    print(f"n_photons={n_photons}, K={K}, temperature={temperature}")

    key = jax.random.PRNGKey(seed)
    key, gen_key, sim_key = jax.random.split(key, 3)

    t0_gen = time.perf_counter()
    ray_vectors, ray_origins, photon_weights = generate_siren_track_photons(
        track_origin, track_direction, siren_energy, n_photons, gen_key,
        grid_data, model_params, F=F,
    )
    gen_time = time.perf_counter() - t0_gen
    print(f"SIREN generation: {gen_time:.1f}s")

    offsets = ray_origins - track_origin[None, :]
    along_track = jnp.sum(offsets * track_direction[None, :], axis=1)
    print(f"Emission positions along track: "
          f"min={float(along_track.min()):.1f}m, max={float(along_track.max()):.1f}m")
    print(f"Weight range: min={float(photon_weights.min()):.1f}, "
          f"max={float(photon_weights.max()):.1f}, mean={float(photon_weights.mean()):.1f}")

    t0_prop = time.perf_counter()
    dom_charges, dom_time_weighted = sim(
        ray_origins, ray_vectors, photon_weights, K, sim_key)
    jax.block_until_ready(dom_charges)
    prop_time = time.perf_counter() - t0_prop

    charges = np.array(dom_charges)
    hit_mask = charges > 1e-6
    n_hit = int(hit_mask.sum())

    print(f"\n{'='*60}")
    print(f"Results ({prop_time:.1f}s propagation)")
    print(f"{'='*60}")
    print(f"  DOMs hit: {n_hit} / {det.n_sensors}")
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

    strings_hit = set()
    for dom_id in np.where(hit_mask)[0]:
        for si in range(det.n_str):
            n = det.n_dom_per_str_np[si]
            if dom_id in det.dom_global_ids[si, :n]:
                strings_hit.add(si)
                break
    print(f"  Strings hit: {len(strings_hit)} / {det.n_str}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_siren_track()
