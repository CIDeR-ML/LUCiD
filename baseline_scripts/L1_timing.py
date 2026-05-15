"""Level 1 Runtime Comparison — Time key operations on both codebases.

Usage:
  LUCID_BACKEND=lucid JAX_PLATFORM_NAME=cpu python baseline_scripts/L1_timing.py
  cd /tmp/lucid-baseline && LUCID_BACKEND=tools JAX_PLATFORM_NAME=cpu python /path/to/L1_timing.py
"""
import os
import sys
import time
import json

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

BACKEND = os.environ.get("LUCID_BACKEND", "lucid")

if BACKEND == "tools":
    sys.path.insert(0, os.getcwd())
    from tools.simulation import (
        normalize, photon_iteration_update_factors,
        photon_iteration_update_factors_safe,
        make_hits_simulation, make_hits_likelihood,
    )
    from tools.losses import poisson_nll
    from tools.optimization.losses import counts_loss, origin_time_loss
    from tools.detector_params import DetectorParams, ParticleParams
    from tools.geometry import generate_detector
    from tools.propagation.cylinder import create_photon_propagator
    GEOM_CONFIG = "config/WCTE_geom_config.json"
else:
    sys.path.insert(0, os.getcwd())
    from lucid.simulation.optics import normalize
    from lucid.simulation.photon_step import (
        photon_iteration_update_factors,
        photon_iteration_update_factors_safe,
    )
    from lucid.simulation.sensor_response import make_hits_simulation, make_hits_likelihood
    from lucid.losses import poisson_nll, counts_loss, origin_time_loss
    from lucid.detector_params import DetectorParams, ParticleParams
    from lucid.geometry import generate_detector
    from lucid.propagation.cylinder import create_photon_propagator
    GEOM_CONFIG = "config/WCTE_geom_config.json"

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit

WARMUP = 3
RUNS = 20

timings = {}


def bench(name, fn, warmup=WARMUP, runs=RUNS):
    """Benchmark a function: warmup, then time multiple runs."""
    for _ in range(warmup):
        result = fn()
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
        times.append(time.perf_counter() - start)

    mean_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t - mean_ms/1000)**2 for t in times) / len(times))**0.5 * 1000
    timings[name] = {"mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3)}
    print(f"  {name}: {mean_ms:.3f} ± {std_ms:.3f} ms")


print(f"Backend: {BACKEND}")
print(f"Warmup: {WARMUP}, Runs: {RUNS}")
print()

# ── Setup ────────────────────────────────────────────────────────────

key = jax.random.PRNGKey(42)

# Photon step args
pi_args = dict(
    position=jnp.array([0.5, 0.5, 0.5]),
    direction=jnp.array([0.0, 0.0, 1.0]),
    time=0.0,
    surface_distance=1.5,
    normal=jnp.array([0.0, 0.0, 1.0]),
    scatter_length=50.0,
    wall_reflection_rate=0.5,
    sensor_reflection_rate=0.3,
    absorption_length=100.0,
    hit_sensor=False,
    rng_key=key,
    speed_of_light=0.2253,
)

# Sensor response args
flat_w = jnp.array([0.5, 0.3, 0.8, 0.1, 0.6])
flat_i = jnp.array([0, 2, 5, 5, 10])
flat_t = jnp.array([10.0, 15.0, 12.0, 20.0, 8.0])
qe_corr = jnp.ones(20)

# Loss args
true_q = jnp.array([10.0, 5.0, 0.0, 3.0, 8.0])
pred_q = jnp.array([9.0, 6.0, 0.1, 2.5, 7.5])


# ── Benchmarks ───────────────────────────────────────────────────────

print("Photon step functions:")
bench("photon_iteration_update_factors",
      lambda: photon_iteration_update_factors(**pi_args))

bench("photon_iteration_update_factors_safe",
      lambda: photon_iteration_update_factors_safe(**pi_args))

bench("photon_step_gradient",
      lambda: jax.grad(lambda p: photon_iteration_update_factors_safe(
          **{**pi_args, 'position': p})[3])(jnp.array([0.5, 0.5, 0.5])))

print("\nSensor response:")
bench("make_hits_simulation",
      lambda: make_hits_simulation(flat_w, flat_i, flat_t, 20,
                                    qe=0.2, qe_corrections=qe_corr))

bench("make_hits_likelihood",
      lambda: make_hits_likelihood(flat_w, flat_i, flat_t, 20,
                                    qe=0.2, qe_corrections=qe_corr))

print("\nLoss functions:")
bench("poisson_nll", lambda: poisson_nll(true_q, pred_q))
bench("counts_loss", lambda: counts_loss(true_q, pred_q))

bench("counts_loss_gradient",
      lambda: jax.grad(lambda p: counts_loss(true_q, p))(pred_q))

print("\nPropagator:")
if os.path.exists(GEOM_CONFIG):
    det = generate_detector(GEOM_CONFIG)
    sp = jnp.array(det.all_points)

    # Time propagator CREATION (includes grid building)
    start = time.perf_counter()
    prop = create_photon_propagator(sp, det.S_radius, r=det.r, h=det.H,
                                    temperature=0.2, max_sensors_per_cell=4)
    create_time = time.perf_counter() - start
    timings["propagator_creation"] = {"mean_ms": round(create_time * 1000, 1), "std_ms": 0}
    print(f"  propagator_creation: {create_time*1000:.1f} ms (single run)")

    origins = jnp.zeros((10, 3))
    dirs = jax.random.normal(key, (10, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    bench("propagator_10_rays", lambda: prop(origins, dirs))

    # Propagator gradient
    def prop_loss(origin):
        return jnp.sum(prop(origin[None, :], jnp.array([[1., 0., 0.]]))[
            'sensor_weights'])

    bench("propagator_gradient_1_ray",
          lambda: jax.grad(prop_loss)(jnp.array([0., 0., 0.])))


# ── Save results ─────────────────────────────────────────────────────

output_file = f"baseline_scripts/L1_timing_{BACKEND}.json"
os.makedirs("baseline_scripts", exist_ok=True)
with open(output_file, "w") as f:
    json.dump(timings, f, indent=2)

print(f"\nSaved {len(timings)} timings → {output_file}")
