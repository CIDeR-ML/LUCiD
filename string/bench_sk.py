"""
Benchmark SK cylinder propagation for comparison with string telescope.

Run: python string/bench_sk.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.simulator import setup_event_simulator
from lucid.detector_params import ParticleParams

N_PHOTONS = 1_000_000


def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"N_PHOTONS: {N_PHOTONS:,}")

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    geom_json = os.path.join(config_dir, "SK_geom_config.json")
    phys_json = os.path.join(config_dir, "SK_like_physics_config.json")

    for K in [7, 20]:
        print(f"\n{'='*60}")
        print(f"SK cylinder, K={K}, n_photons={N_PHOTONS:,}")
        print(f"{'='*60}")

        sim_fn = setup_event_simulator(
            geom_json,
            n_photons=N_PHOTONS,
            temperature=0.2,
            K=K,
            detector_type='Cylinder',
            use_expected_value=True,
            physics_config=phys_json,
            default_detector_params=True,
            wavelength_mode=False,
        )

        pp = ParticleParams(
            energy=jnp.array(500.0),
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=jnp.array(1.0),
            phi=jnp.array(0.5),
            t0=jnp.array(0.0),
        )

        print("  JIT compiling...")
        key = jax.random.PRNGKey(42)
        t0 = time.perf_counter()
        result = sim_fn(pp, key)
        charges = result[0]
        jax.block_until_ready(charges)
        jit_time = time.perf_counter() - t0
        print(f"  JIT compile: {jit_time:.1f}s")

        run_times = []
        for i in range(5):
            key = jax.random.PRNGKey(100 + i)
            t0 = time.perf_counter()
            result = sim_fn(pp, key)
            charges = result[0]
            jax.block_until_ready(charges)
            elapsed = time.perf_counter() - t0
            run_times.append(elapsed)
            n_hit = int((np.array(charges) > 0).sum())
            print(f"  Run {i}: {elapsed:.4f}s, {n_hit} sensors hit")

        mean_t = np.mean(run_times)
        std_t = np.std(run_times)
        print(f"\n  Mean: {mean_t:.4f} +/- {std_t:.4f} s")
        print(f"  Per-K: {mean_t/K:.4f} s")


if __name__ == "__main__":
    main()
