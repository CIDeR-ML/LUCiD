"""
Benchmark: string propagator on IceCube-86 geometry.

Measures per-ray propagation time and candidate counts for the string
propagator. Compares to the expected SK baseline numbers.

Run: python string/bench_propagator.py
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

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")


def bench(n_rays=1000, n_warmup=2, n_trials=5):
    print(f"Loading IceCube-86 simple ({n_rays} rays)...")
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=0.2)

    # Random rays: origins inside envelope, random directions
    rng = np.random.RandomState(42)
    origins_np = rng.uniform(-200, 200, (n_rays, 3))
    origins_np[:, 2] = rng.uniform(-2400, -1500, n_rays)
    dirs_np = rng.randn(n_rays, 3)
    dirs_np /= np.linalg.norm(dirs_np, axis=1, keepdims=True)

    origins = jnp.array(origins_np)
    directions = jnp.array(dirs_np)

    # Warmup (JIT compilation)
    print("Warming up JIT...")
    for _ in range(n_warmup):
        result = prop(origins[:10], directions[:10])
        jax.block_until_ready(result['sensor_weights'])

    # Benchmark
    print(f"Benchmarking ({n_trials} trials)...")
    times = []
    for trial in range(n_trials):
        t0 = time.perf_counter()
        result = prop(origins, directions)
        jax.block_until_ready(result['sensor_weights'])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    us_per_ray = times.mean() * 1e6 / n_rays

    # Count candidate stats
    weights = np.array(result['sensor_weights'])
    inside = np.array(result['inside_sensor'])
    n_nonzero_weights = np.sum(weights > 0)
    n_inside = np.sum(inside)
    max_dom_per_seg = weights.shape[0]

    # Rays with at least one hit
    rays_with_hit = np.sum(np.any(inside, axis=0))

    print(f"\n{'='*60}")
    print(f"IceCube-86 string propagator benchmark")
    print(f"{'='*60}")
    print(f"  Strings:              {det.n_str}")
    print(f"  Total DOMs:           {det.n_sensors}")
    print(f"  Sensor radius:        {det.S_radius} m")
    print(f"  max_dom_per_segment:  {max_dom_per_seg}")
    print(f"  K_min:                {prop.sizing.K_min}")
    print(f"  N rays:               {n_rays}")
    print(f"")
    print(f"  Time (mean ± std):    {times.mean()*1e3:.1f} ± {times.std()*1e3:.1f} ms")
    print(f"  Per-ray:              {us_per_ray:.1f} µs")
    print(f"")
    print(f"  Non-zero weights:     {n_nonzero_weights} / {max_dom_per_seg * n_rays}")
    print(f"  Inside-sensor hits:   {n_inside} / {max_dom_per_seg * n_rays}")
    print(f"  Rays with any hit:    {rays_with_hit} / {n_rays} ({100*rays_with_hit/n_rays:.1f}%)")
    print(f"")
    print(f"  Candidate slots per ray (max): {max_dom_per_seg}")
    print(f"  Expected SK baseline: 4 candidates/ray")
    print(f"  Ratio: {max_dom_per_seg/4:.1f}x more candidate slots than SK")
    print(f"  (But SK checks 4 per K step; string checks {max_dom_per_seg} per segment)")
    print(f"{'='*60}")


if __name__ == "__main__":
    bench()
