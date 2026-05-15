"""L2 Track Forward Pass — Capture forward pass + gradient at fixed params.

Tests the core track simulation pipeline (SIREN → propagation → sensor response)
and gradient computation. Uses WCTE config for speed.

This is the foundation for all L2 track tests — if the forward pass and
gradients match, optimization convergence will match too.
"""
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

BACKEND = os.environ.get("LUCID_BACKEND", "lucid")
if BACKEND == "tools":
    sys.path.insert(0, os.getcwd())
    from tools.simulation import setup_event_simulator
    from tools.detector_params import ParticleParams
    from tools.losses import poisson_nll
    from tools.optimization.losses import counts_loss, origin_time_loss, cone_time_loss
    GEOM = "config/WCTE_geom_config.json"
else:
    sys.path.insert(0, os.getcwd())
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import ParticleParams
    from lucid.losses import poisson_nll, counts_loss, origin_time_loss, cone_time_loss
    GEOM = "config/WCTE_geom_config.json"

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad

print(f"L2 Track Forward Pass | Backend: {BACKEND}")
t_start = time.perf_counter()

# ── Setup ────────────────────────────────────────────────────────────
NPHOT = 1000
K = 2
TEMPERATURE = 0.1

# Need detector geometry to build default DetectorParams
if BACKEND == "tools":
    from tools.geometry import generate_detector
    from tools.detector_params import DetectorParams
else:
    from lucid.geometry import generate_detector
    from lucid.detector_params import DetectorParams

det = generate_detector(GEOM)
NUM_SENSORS = len(det.all_points)

default_dp = DetectorParams(
    scatter_length=50.0,
    wall_reflection_rate=0.2,
    sensor_reflection_rate=0.2,
    absorption_length=50.0,
    qe=0.2,
    qe_corrections=jnp.ones(NUM_SENSORS),
)

# Use explicit grid params matching old defaults for cylinder
sim = setup_event_simulator(
    GEOM, NPHOT, TEMPERATURE, K=K,
    is_data=False, max_sensors_per_cell=4,
    default_detector_params=default_dp,
    n_cap=150, n_angular=250, n_height=150,
    wavelength_mode=False)

print(f"  Simulator created (Nphot={NPHOT}, K={K})")

# ── Fixed particle params ────────────────────────────────────────────
track = ParticleParams(
    energy=jnp.array(500.0),
    position=jnp.array([0.0, 0.0, 0.0]),
    theta=jnp.array(0.5),
    phi=jnp.array(1.0),
    t0=jnp.array(0.0),
)

# ── Forward pass ─────────────────────────────────────────────────────
key = jax.random.PRNGKey(42)
output = sim(track, key)

results = {}

if isinstance(output, tuple) and len(output) == 2:
    charges, times = output
    results["mode"] = "simulation"
    results["charges_sum"] = float(jnp.sum(charges))
    results["charges_nonzero"] = int(jnp.sum(charges > 0))
    results["times_min_nonzero"] = float(jnp.min(jnp.where(charges > 0, times, 1e6)))
    results["charges_shape"] = list(charges.shape)
    print(f"  Forward pass: charge_sum={results['charges_sum']:.4f}, "
          f"nonzero={results['charges_nonzero']}")

    # ── Loss at true params ──────────────────────────────────────────
    # Create "observed" data from the same params with different key
    key2 = jax.random.PRNGKey(99)
    obs_charges, obs_times = sim(track, key2)

    loss_counts = float(counts_loss(obs_charges, charges))
    results["loss_counts"] = loss_counts
    print(f"  counts_loss = {loss_counts:.6f}")

    # ── Gradient ─────────────────────────────────────────────────────
    @jit
    def loss_fn(params_arr):
        t = ParticleParams(
            energy=params_arr[0],
            position=params_arr[1:4],
            theta=params_arr[4],
            phi=params_arr[5],
            t0=jnp.array(0.0))
        pred_q, pred_t = sim(t, key)
        return counts_loss(obs_charges, pred_q)

    params = jnp.array([500.0, 0.0, 0.0, 0.0, 0.5, 1.0])
    loss_val, grad_val = value_and_grad(loss_fn)(params)
    results["grad_loss_value"] = float(loss_val)
    results["grad_vector"] = grad_val.tolist()
    results["grad_finite"] = bool(jnp.all(jnp.isfinite(grad_val)))
    print(f"  Gradient: loss={float(loss_val):.6f}, finite={results['grad_finite']}")
    print(f"  grad = {grad_val}")

elif isinstance(output, tuple) and len(output) == 4:
    log_w, flat_times, flat_indices, total_charge = output
    results["mode"] = "likelihood"
    results["total_charge_sum"] = float(jnp.sum(total_charge))
    results["total_charge_nonzero"] = int(jnp.sum(total_charge > 0))
    results["log_w_shape"] = list(log_w.shape)
    results["flat_times_shape"] = list(flat_times.shape)
    print(f"  Forward pass (likelihood): charge_sum={results['total_charge_sum']:.4f}, "
          f"nonzero={results['total_charge_nonzero']}")

    # Gradient through poisson_nll
    key2 = jax.random.PRNGKey(99)
    _, _, _, obs_charge = sim(track, key2)

    @jit
    def loss_fn(params_arr):
        t = ParticleParams(
            energy=params_arr[0],
            position=params_arr[1:4],
            theta=params_arr[4],
            phi=params_arr[5],
            t0=jnp.array(0.0))
        _, _, _, pred_q = sim(t, key)
        return poisson_nll(obs_charge, pred_q)

    params = jnp.array([500.0, 0.0, 0.0, 0.0, 0.5, 1.0])
    loss_val, grad_val = value_and_grad(loss_fn)(params)
    results["grad_loss_value"] = float(loss_val)
    results["grad_vector"] = grad_val.tolist()
    results["grad_finite"] = bool(jnp.all(jnp.isfinite(grad_val)))
    print(f"  Gradient: loss={float(loss_val):.6f}, finite={results['grad_finite']}")
    print(f"  grad = {grad_val}")

# ── Save ─────────────────────────────────────────────────────────────
output_file = f"baseline_scripts/L2_track_forward_{BACKEND}.json"
os.makedirs("baseline_scripts", exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.perf_counter() - t_start
print(f"\nDone in {elapsed:.1f}s → {output_file}")
