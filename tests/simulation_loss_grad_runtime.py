import sys
import os
import time
import argparse
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.geometry import generate_detector
from tools.utils import generate_random_params
from tools.losses import compute_loss
from tools.simulation import setup_event_simulator


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


def do_throwaway_runs(simulate_event, loss_fn, loss_and_grad_fn, n_throwaway, n_photons):
    """Do multiple throwaway runs to ensure stable performance."""
    print(f"\nDoing {n_throwaway} throwaway runs for {n_photons} photons...")

    for i in range(n_throwaway):
        key = jax.random.PRNGKey(i)
        keys = jax.random.split(key, 6)

        params1 = generate_random_params(keys[0])
        params2 = generate_random_params(keys[1])
        params3 = generate_random_params(keys[2])

        true_data1 = jax.lax.stop_gradient(simulate_event(generate_random_params(keys[3]), keys[3]))
        true_data2 = jax.lax.stop_gradient(simulate_event(generate_random_params(keys[4]), keys[4]))

        _ = simulate_event(params1, keys[0])
        _ = loss_fn(params2, keys[1], true_data1)
        _ = loss_and_grad_fn(params3, keys[2], true_data2)

        jax.block_until_ready(_)

    print(f"Warmup complete for {n_photons} photons.")


def benchmark_operations(photon_counts, n_runs=20, n_throwaway=5, temperature=100.0,
                         json_filename='config/cyl_geom_config.json'):
    detector = generate_detector(json_filename)
    detector_points = jnp.array(detector.all_points)

    device_type = jax.devices()[0].device_kind
    print(f"Running on {device_type}")

    times = {
        'simulate': [],
        'simulate_loss': [],
        'simulate_loss_grad': []
    }

    for n_photons in photon_counts:
        print(f"\nBenchmarking with {n_photons:,} photons...")

        simulate_event = setup_event_simulator(json_filename, n_photons, temperature)

        def loss_fn(params, key, true_data):
            simulated_data = simulate_event(params, key)
            return compute_loss(*true_data, *simulated_data)

        loss_and_grad_fn = jax.value_and_grad(loss_fn)

        jitted_simulate = jax.jit(simulate_event)
        jitted_loss_fn = jax.jit(loss_fn)
        jitted_loss_and_grad = jax.jit(loss_and_grad_fn)

        do_throwaway_runs(jitted_simulate, jitted_loss_fn, jitted_loss_and_grad, n_throwaway, n_photons)

        times_for_this_n = {
            'simulate': [],
            'simulate_loss': [],
            'simulate_loss_grad': []
        }

        for run in range(n_runs):
            base_key = jax.random.PRNGKey(run + n_throwaway + n_photons)
            keys = jax.random.split(base_key, 8)

            params_sim = generate_random_params(keys[0])
            params_loss = generate_random_params(keys[1])
            params_grad = generate_random_params(keys[2])

            true_data_loss = jax.lax.stop_gradient(simulate_event(generate_random_params(keys[3]), keys[3]))
            true_data_grad = jax.lax.stop_gradient(simulate_event(generate_random_params(keys[4]), keys[4]))

            start = time.time()
            result = jitted_simulate(params_sim, keys[5])
            jax.block_until_ready(result)
            times_for_this_n['simulate'].append(time.time() - start)

            start = time.time()
            result = jitted_loss_fn(params_loss, keys[6], true_data_loss)
            jax.block_until_ready(result)
            times_for_this_n['simulate_loss'].append(time.time() - start)

            start = time.time()
            result = jitted_loss_and_grad(params_grad, keys[7], true_data_grad)
            jax.block_until_ready(result)
            times_for_this_n['simulate_loss_grad'].append(time.time() - start)

        for op in times:
            op_times = sorted(times_for_this_n[op])[1:-1]
            avg_time = sum(op_times) / len(op_times)
            times[op].append(avg_time)
            print(f"  Average {op} time (excluding outliers): {avg_time * 1000:.2f} ms")
            print(f"  All times: {[f'{t * 1000:.2f}' for t in times_for_this_n[op]]} ms")

    return times


def plot_benchmark_results(photon_counts, times, backend):
    plt.figure(figsize=(12, 8))

    plt.loglog(photon_counts, times['simulate'], 'bo-', label='Simulate Only')
    plt.loglog(photon_counts, times['simulate_loss'], 'ro-', label='Simulate + Loss')
    plt.loglog(photon_counts, times['simulate_loss_grad'], 'go-', label='Simulate + Loss + Grad')

    plt.xlabel('Number of Photons')
    plt.ylabel('Time (seconds)')
    plt.title(f'JAX Photon Simulation Operations Benchmark on {backend}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"operation_runtimes_{backend}.png")
    print(f"Plot saved to operation_runtimes_{backend}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CPU', action='store_true', default=False,
                        help='Run in CPU mode')
    return parser.parse_args()


def main():
    args = parse_args()
    backend = setup_backend(args.CPU)

    photon_counts_gpu = [1000, 10000, 100000, 500000, 1000000, 2000000, 5000000]
    photon_counts_cpu = [1000, 10000, 100000, 1000000]

    photon_counts = photon_counts_cpu if backend == "cpu" else photon_counts_gpu

    # Use CPU loss function if in CPU mode
    times = benchmark_operations(photon_counts)
    plot_benchmark_results(photon_counts, times, backend)


if __name__ == "__main__":
    main()