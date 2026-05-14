"""
End-to-end SIREN track benchmark using the fast string simulator.

Runs 5 events with 1M photons each and compares timing.

Run: python string/bench_siren_fast.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import numpy as np

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.fast import create_fast_string_simulator
from lucid.siren.core import create_photonsim_siren_grid
from lucid.siren.training.inference import SIRENPredictor
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.utils import base_dir_path

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")

F = 30
N_PHOTONS = 1_000_000
K = 20
SPEED = 0.2254
SIREN_ENERGY = 2000.0


def main():
    print(f"Backend: {jax.default_backend()}")
    det = StringTelescope.from_npz(SIMPLE_NPZ)

    sim = create_fast_string_simulator(
        det, det.S_radius, temperature=0.2,
        lambda_abs=100.0, lambda_scat=30.0,
        speed_of_light=SPEED, n_closest=2)

    data_dir = os.path.join(base_dir_path(), 'data', 'water', 'muon')
    model_path = os.path.join(data_dir, 'siren_training', 'trained_model', 'photonsim_siren')
    predictor = SIRENPredictor(model_path)
    grid_data = create_photonsim_siren_grid(predictor)
    model_params = predictor.params

    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    N_EVENTS = 5

    print(f"\n{N_EVENTS} events, {N_PHOTONS:,} photons, K={K}, F={F}")
    print(f"{'='*60}")

    event_times = []
    gen_times = []
    sim_times = []

    for ev in range(N_EVENTS):
        key = jax.random.PRNGKey(ev * 1000 + 42)
        key, k1, k2, gen_key = jax.random.split(key, 4)

        origin_offset = jax.random.uniform(k1, (3,), minval=-50.0, maxval=50.0)
        origin_offset = origin_offset.at[2].set(origin_offset[2] * 5)
        track_origin = jnp.array([0.0, 0.0, z_mid]) + origin_offset
        dir_raw = jax.random.normal(k2, (3,))
        track_direction = dir_raw / (jnp.linalg.norm(dir_raw) + 1e-10)

        # Generate SIREN photons
        t0 = time.perf_counter()
        ray_vectors, ray_origins, photon_weights = photonsim_differentiable_get_rays(
            track_origin, track_direction, SIREN_ENERGY, N_PHOTONS,
            grid_data, model_params, gen_key)
        offsets = ray_origins - track_origin[None, :]
        ray_origins_scaled = track_origin[None, :] + F * offsets
        photon_weights_scaled = photon_weights * F
        jax.block_until_ready(photon_weights_scaled)
        t_gen = time.perf_counter() - t0
        gen_times.append(t_gen)

        # Simulate
        t0 = time.perf_counter()
        dom_q, dom_tw = sim(ray_origins_scaled, ray_vectors, photon_weights_scaled, K, key)
        jax.block_until_ready(dom_q)
        t_sim = time.perf_counter() - t0
        sim_times.append(t_sim)

        total = t_gen + t_sim
        event_times.append(total)

        charges = np.array(dom_q)
        n_hit = int((charges > 1e-6).sum())
        tag = " (JIT)" if ev == 0 else ""
        print(f"  Event {ev}: {n_hit:4d} DOMs, Q={charges.sum():10.0f}, "
              f"gen={t_gen:.2f}s sim={t_sim:.3f}s total={total:.2f}s{tag}")

    if len(event_times) > 1:
        post_jit_sim = sim_times[1:]
        post_jit_total = event_times[1:]
        print(f"\nPost-JIT timing:")
        print(f"  SIREN gen: {np.mean(gen_times[1:]):.3f} +/- {np.std(gen_times[1:]):.3f} s")
        print(f"  Simulation: {np.mean(post_jit_sim):.3f} +/- {np.std(post_jit_sim):.3f} s")
        print(f"  Total: {np.mean(post_jit_total):.3f} +/- {np.std(post_jit_total):.3f} s")
        print(f"\n  Old batch timing was: 4.71 +/- 0.09 s/event")


if __name__ == "__main__":
    main()
