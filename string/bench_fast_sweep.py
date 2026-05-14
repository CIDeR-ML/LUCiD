"""
Sweep n_closest values and verify physics.

Run: python string/bench_fast_sweep.py
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
K = 20
SPEED = 0.2254


def main():
    print(f"Backend: {jax.default_backend()}")
    det = StringTelescope.from_npz(SIMPLE_NPZ)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
    origins = jax.random.uniform(k1, (N, 3),
                                 minval=jnp.array([-200, -200, z_mid - 400]),
                                 maxval=jnp.array([200, 200, z_mid + 400]))
    dirs_raw = jax.random.normal(k2, (N, 3))
    dirs = dirs_raw / (jnp.linalg.norm(dirs_raw, axis=1, keepdims=True) + 1e-10)
    weights = jnp.ones(N)

    results = {}
    for nc in [1, 2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"n_closest={nc} ({nc*2} DOM candidates/ray)")
        print(f"{'='*50}")

        sim = create_fast_string_simulator(
            det, det.S_radius, temperature=0.2,
            lambda_abs=100.0, lambda_scat=30.0,
            speed_of_light=SPEED, n_closest=nc)

        # JIT warmup
        key_run = jax.random.PRNGKey(99)
        dom_q, _ = sim(origins, dirs, weights, K, key_run)
        jax.block_until_ready(dom_q)

        # Timed runs
        run_times = []
        for i in range(5):
            key_run = jax.random.PRNGKey(300 + i)
            t0 = time.perf_counter()
            dom_q, _ = sim(origins, dirs, weights, K, key_run)
            jax.block_until_ready(dom_q)
            elapsed = time.perf_counter() - t0
            run_times.append(elapsed)

        mean_t = np.mean(run_times)
        charges = np.array(dom_q)
        n_hit = int((charges > 1e-6).sum())
        total_q = float(charges.sum())

        print(f"  Time: {mean_t:.4f}s ({mean_t/K*1000:.1f}ms/K)")
        print(f"  DOMs hit: {n_hit}, total Q: {total_q:.1f}")
        results[nc] = (mean_t, n_hit, total_q, charges.copy())

    # Compare physics across n_closest values
    print(f"\n{'='*50}")
    print(f"Physics comparison (K={K}, N={N:,})")
    print(f"{'='*50}")
    ref_charges = results[4][3]
    for nc in [1, 2, 3, 4]:
        t, n_hit, total_q, charges = results[nc]
        if nc == 4:
            diff_str = "(reference)"
        else:
            both_hit = (charges > 1e-6) & (ref_charges > 1e-6)
            if both_hit.sum() > 0:
                rel_diff = np.abs(charges[both_hit] - ref_charges[both_hit]) / (ref_charges[both_hit] + 1e-10)
                diff_str = f"median rel diff vs nc=4: {np.median(rel_diff):.2%}"
            else:
                diff_str = "no overlap"
        print(f"  nc={nc}: {t:.4f}s, {n_hit} DOMs, Q={total_q:.1f} — {diff_str}")


if __name__ == "__main__":
    main()
