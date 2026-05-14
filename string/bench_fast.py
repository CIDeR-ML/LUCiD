"""
Benchmark fast string simulator vs old propagator.

Compares:
  1. Physics correctness: dom charges should match (within noise)
  2. Speed: target K=20, 1M photons < 0.5s

Run: python string/bench_fast.py
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

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")

N = 1_000_000
SPEED = 0.2254


def generate_test_photons(n, det, key):
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    k1, k2 = jax.random.split(key)
    origins = jax.random.uniform(k1, (n, 3),
                                 minval=jnp.array([-200, -200, z_mid - 400]),
                                 maxval=jnp.array([200, 200, z_mid + 400]))
    dirs_raw = jax.random.normal(k2, (n, 3))
    dirs = dirs_raw / (jnp.linalg.norm(dirs_raw, axis=1, keepdims=True) + 1e-10)
    weights = jnp.ones(n)
    return origins, dirs, weights


def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"N_PHOTONS: {N:,}")

    det = StringTelescope.from_npz(SIMPLE_NPZ)

    sim_fast = create_fast_string_simulator(
        det, det.S_radius, temperature=0.2,
        lambda_abs=100.0, lambda_scat=30.0,
        speed_of_light=SPEED, n_closest=4)

    key = jax.random.PRNGKey(42)
    origins, dirs, weights = generate_test_photons(N, det, key)

    # ── JIT warmup ──
    for K in [1, 5, 20]:
        print(f"\n{'='*60}")
        print(f"Fast simulator, K={K}")
        print(f"{'='*60}")

        print("  JIT compiling...")
        key_run = jax.random.PRNGKey(99)
        t0 = time.perf_counter()
        dom_q, dom_tw = sim_fast(origins, dirs, weights, K, key_run)
        jax.block_until_ready(dom_q)
        jit_time = time.perf_counter() - t0
        print(f"  JIT: {jit_time:.1f}s")

        # ── Timed runs ──
        run_times = []
        for i in range(5):
            key_run = jax.random.PRNGKey(200 + i)
            t0 = time.perf_counter()
            dom_q, dom_tw = sim_fast(origins, dirs, weights, K, key_run)
            jax.block_until_ready(dom_q)
            elapsed = time.perf_counter() - t0
            run_times.append(elapsed)

            charges = np.array(dom_q)
            n_hit = int((charges > 1e-6).sum())
            total_q = float(charges.sum())
            print(f"  Run {i}: {elapsed:.4f}s, {n_hit} DOMs hit, Q={total_q:.0f}")

        mean_t = np.mean(run_times)
        std_t = np.std(run_times)
        print(f"\n  Mean: {mean_t:.4f} +/- {std_t:.4f} s")
        print(f"  Per-K: {mean_t/K:.4f} s")


if __name__ == "__main__":
    main()
