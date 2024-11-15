import sys
import os

# Add the parent directory of 'tools' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
import time

from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector

import argparse


def generate_random_photons(n_photons, cylinder_radius, cylinder_height):
    """Generate random photon origins and directions inside the cylinder."""
    key = jax.random.PRNGKey(4)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    r = jax.random.uniform(subkey1, (n_photons,)) * cylinder_radius
    theta = jax.random.uniform(subkey1, (n_photons,)) * 2 * jnp.pi
    z = jax.random.uniform(subkey1, (n_photons,)) * cylinder_height - cylinder_height / 2

    origins = jnp.column_stack((r * jnp.cos(theta), r * jnp.sin(theta), z))

    directions = jax.random.normal(subkey2, (n_photons, 3))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    return origins, directions


def run_simulation(n_photons=1000, cylinder_radius=4.0, cylinder_height=6.0, n_cap=39, n_angular=168, n_height=28,
                   weight_threshold=0.5, temperature=100, json_filename='config/cyl_geom_config.json'):
    # Generate detector positions
    detector = generate_detector(json_filename)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius

    # Generate random photons
    photon_origins, photon_directions = generate_random_photons(n_photons, cylinder_radius, cylinder_height)

    propagate_photons = create_photon_propagator(detector_points, detector_radius)

    # Propagate photons
    results = propagate_photons(photon_origins, photon_directions, temperature=temperature)

    # Unpack results with weights
    detector_weights = results['detector_weights']  # Shape: (max_detectors_per_cell, n_photons)
    detector_indices = results['detector_indices']  # Shape: (max_detectors_per_cell, n_photons)
    hit_positions = results['positions']  # Shape: (max_detectors_per_cell, n_photons, 3)
    hit_times = results['times']  # Shape: (max_detectors_per_cell, n_photons, 1)

    # Calculate hits based on weight threshold
    significant_hits = detector_weights > weight_threshold
    any_significant_hit = jnp.any(significant_hits, axis=0)

    # Get statistics
    n_detected = jnp.sum(any_significant_hit)
    n_missed = n_photons - n_detected

    # For photons with significant hits, get their primary detector (highest weight)
    max_weight_indices = jnp.argmax(detector_weights, axis=0)
    primary_detector_indices = jnp.where(
        any_significant_hit,
        jnp.take_along_axis(detector_indices, max_weight_indices[None, :], axis=0)[0],
        -1
    )

    # Get hit distribution statistics
    unique_detectors = jnp.unique(primary_detector_indices[primary_detector_indices >= 0])
    n_unique_detectors_hit = len(unique_detectors)

    # Calculate average weights for detected photons
    avg_weight = jnp.where(
        any_significant_hit,
        jnp.max(detector_weights, axis=0),
        0.0
    ).mean()

    # Print statistics
    print(f"\nSimulation Statistics:")
    print(f"Number of photons: {n_photons}")
    print(f"Number of detectors: {detector_points.shape[0]}")
    print(f"Temperature parameter: {temperature}")
    print(f"Weight threshold: {weight_threshold}")
    print(f"\nDetection Results:")
    print(f"Photons detected (above threshold): {n_detected}")
    print(f"Photons missed: {n_missed}")
    print(f"Detection rate: {n_detected / n_photons * 100:.2f}%")
    print(f"Unique detectors hit: {n_unique_detectors_hit}")
    print(f"Average maximum weight for detected photons: {avg_weight:.3f}")

    # Return comprehensive results dictionary
    return {
        'detector_weights': detector_weights,
        'detector_indices': detector_indices,
        'hit_positions': hit_positions,
        'hit_times': hit_times,
        'primary_detector_indices': primary_detector_indices,
        'any_significant_hit': any_significant_hit,
        'photon_origins': photon_origins,
        'photon_directions': photon_directions,
        'detector_points': detector_points,
        'stats': {
            'n_photons': n_photons,
            'n_detected': n_detected,
            'n_missed': n_missed,
            'n_unique_detectors': n_unique_detectors_hit,
            'avg_weight': float(avg_weight),
            'detection_rate': float(n_detected / n_photons)
        }
    }


def generate_random_photons(n_photons, cylinder_radius, cylinder_height):
    """Generate random photon origins and directions inside the cylinder."""
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    r = jax.random.uniform(subkey1, (n_photons,)) * cylinder_radius
    theta = jax.random.uniform(subkey1, (n_photons,)) * 2 * jnp.pi
    z = jax.random.uniform(subkey1, (n_photons,)) * cylinder_height - cylinder_height / 2

    origins = jnp.column_stack((r * jnp.cos(theta), r * jnp.sin(theta), z))

    directions = jax.random.normal(subkey2, (n_photons, 3))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    return origins, directions


def simulate_core(key, propagate_photons, n_photons=1000, cylinder_radius=4.0, cylinder_height=6.0):
    """Core simulation function."""
    key1, key2 = jax.random.split(key)

    # Generate random photon positions and directions
    r = jax.random.uniform(key1, (n_photons,)) * cylinder_radius
    theta = jax.random.uniform(key1, (n_photons,)) * 2 * jnp.pi
    z = jax.random.uniform(key1, (n_photons,)) * cylinder_height - cylinder_height / 2

    origins = jnp.column_stack((r * jnp.cos(theta), r * jnp.sin(theta), z))

    directions = jax.random.normal(key2, (n_photons, 3))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    return propagate_photons(origins, directions)


def benchmark_simulation(photon_counts, n_runs=10, temperature=100):
    """
    Benchmark with fixed timing issues.
    """
    compile_times = []
    run_times = []

    # Setup once
    detector = generate_detector('config/cyl_geom_config.json')
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius
    propagate_photons = create_photon_propagator(
        detector_points,
        detector_radius,
        r=4.0,
        h=6.0,
        temperature=temperature
    )

    device_type = jax.devices()[0].device_kind
    print(f"Running on {device_type}")

    for n_photons in photon_counts:
        print(f"\nBenchmarking with {n_photons:,} photons...")

        # Compile
        start_compile = time.time()
        simulate_partial = partial(
            simulate_core,
            propagate_photons=propagate_photons,
            n_photons=n_photons,
            cylinder_radius=4.0,
            cylinder_height=6.0
        )
        jitted_simulate = jax.jit(simulate_partial)

        # Warmup run and compile timing
        key = jax.random.PRNGKey(0)
        _ = jitted_simulate(key)
        jax.block_until_ready(_)
        compile_time = time.time() - start_compile
        compile_times.append(compile_time)

        # Actual benchmark runs
        run_times_for_this_n = []

        for run in range(n_runs):
            key = jax.random.PRNGKey(run)
            jax.block_until_ready(key)

            # Run and time
            start = time.time()
            results = jitted_simulate(key)
            jax.block_until_ready(results)
            end = time.time()

            run_time = end - start
            run_times_for_this_n.append(run_time)

            # Force sync between runs
            _ = jax.device_get(jnp.sum(results['detector_weights']))

        avg_run_time = sum(run_times_for_this_n) / n_runs
        run_times.append(avg_run_time)

        print(f"  Compile time: {compile_time:.4f} seconds")
        print(f"  Run times (ms):", end=" ")
        print(" ".join(f"{t * 1000:.1f}" for t in run_times_for_this_n))
        print(f"  Average run time: {avg_run_time * 1000:.2f} ms")

        if isinstance(results, dict):
            weights_sum = float(jnp.sum(results['detector_weights']))

    return compile_times, run_times


def plot_benchmark_results(photon_counts, compile_times, run_times, backend):
    plt.figure(figsize=(10, 6))
    plt.loglog(photon_counts, compile_times, 'bo-', label='Compile Time')
    plt.loglog(photon_counts, run_times, 'ro-', label='Average Run Time')
    plt.xlabel('Number of Photons')
    plt.ylabel('Time (seconds)')
    plt.title(f'JAX Photon Simulation Benchmark on {backend}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"propagate_runtime_{backend}.png")
    print(f"saved to propagate_runtime_{backend}.png")


def setup_backend(force_cpu):
    if force_cpu:
        jax.config.update('jax_platform_name', 'cpu')
        print("Running on CPU mode")
        return "cpu"

    if jax.default_backend() != "gpu":
        print("No GPU available, running on CPU mode")
        jax.config.update('jax_platform_name', 'cpu')
        return "cpu"

    print("Running on GPU mode")
    return "gpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CPU', action='store_true', default=False,
                       help='Run in CPU mode')
    return parser.parse_args()

def main():
    args = parse_args()
    backend = setup_backend(args.CPU)

    photon_counts_gpu = [1000, 10000, 100000, 1000000, 2000000, 5000000, 10000000]
    photon_counts_cpu = [1000, 10000, 100000, 1000000, 2000000]

    photon_counts = photon_counts_cpu if backend == "cpu" else photon_counts_gpu

    compile_times, run_times = benchmark_simulation(photon_counts)
    plot_benchmark_results(photon_counts, compile_times, run_times, backend)

if __name__ == "__main__":
    main()