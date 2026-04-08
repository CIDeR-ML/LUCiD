"""L2-4: Calibration convergence — 4 scalar parameters, QE frozen.

Mirrors: grad_param_calibration_multi_init_no_qe.ipynb
Reduced config: Nphot=5000, K=3, 50 Adam iterations, 1 initial guess.
"""
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

BACKEND = os.environ.get("LUCID_BACKEND", "lucid")
if BACKEND == "tools":
    sys.path.insert(0, os.getcwd())
    from tools.geometry import generate_detector
    from tools.simulation import setup_event_simulator
    from tools.detector_params import (
        DetectorParams, laser_source, load_detector_params,
        normalize_params, denormalize_params, default_bounds,
        make_optimization_mask,
    )
    from tools.losses import WC_smooth_loss
    GEOM = "config/WCTE_geom_config.json"
else:
    sys.path.insert(0, os.getcwd())
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import (
        DetectorParams, laser_source, load_detector_params,
        normalize_params, denormalize_params, default_bounds,
        make_optimization_mask,
    )
    from lucid.losses import WC_smooth_loss
    GEOM = "config/WCTE_geom_config.json"

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
import optax

print(f"L2-4 Calibration 4-param | Backend: {BACKEND}")
t_start = time.perf_counter()

# ── Setup ────────────────────────────────────────────────────────────
NPHOT = 5000
K = 3
NPHOT_TRUE = 5000
K_TRUE = 4
ADAM_LR = 0.05
ADAM_ITERS = 50
WARMUP_FRAC = 0.4

det = generate_detector(GEOM)
sensor_points = jnp.array(det.all_points)
NUM_SENSORS = len(sensor_points)

TRUE_PARAMS = DetectorParams(
    scatter_length=50.0,
    wall_reflection_rate=0.2,
    sensor_reflection_rate=0.2,
    absorption_length=50.0,
    qe=0.2,
    qe_corrections=jnp.ones(NUM_SENSORS),
)

source = laser_source(
    position=[0.0, 0.0, det.H / 2 - 0.1],
    intensity=1_000_000,
)

bounds_min, bounds_max = default_bounds(NUM_SENSORS)
bounds_min = bounds_min._replace(
    scatter_length=jnp.array(5.0),
    wall_reflection_rate=jnp.array(0.05),
    sensor_reflection_rate=jnp.array(0.05),
    absorption_length=jnp.array(10.0),
)

# ── Simulators ───────────────────────────────────────────────────────
simulate_event = setup_event_simulator(
    GEOM, NPHOT, temperature=0.2, K=K,
    is_data=False, is_calibration=True)

simulate_data = setup_event_simulator(
    GEOM, NPHOT_TRUE, temperature=0.2, K=K_TRUE,
    is_data=False, is_calibration=True,
    default_detector_params=TRUE_PARAMS)

# ── Generate true data ──────────────────────────────────────────────
key = jax.random.PRNGKey(42)
key, data_key = jax.random.split(key)
true_data = jax.lax.stop_gradient(simulate_data(source, data_key))
print(f"  True data generated: charge_sum={float(jnp.sum(true_data[0])):.1f}")

# ── Initial guess (perturbed from true) ─────────────────────────────
key, init_key = jax.random.split(key)
mults = jax.random.uniform(init_key, (4,), minval=0.5, maxval=1.5)
init_params = DetectorParams(
    scatter_length=jnp.clip(TRUE_PARAMS.scatter_length * mults[0], 5.0, 200.0),
    wall_reflection_rate=jnp.clip(TRUE_PARAMS.wall_reflection_rate * mults[1], 0.05, 0.95),
    sensor_reflection_rate=jnp.clip(TRUE_PARAMS.sensor_reflection_rate * mults[2], 0.05, 0.95),
    absorption_length=jnp.clip(TRUE_PARAMS.absorption_length * mults[3], 10.0, 200.0),
    qe=TRUE_PARAMS.qe,
    qe_corrections=TRUE_PARAMS.qe_corrections,
)

# ── Loss function ────────────────────────────────────────────────────
key, source_key = jax.random.split(key)

@jit
def step_fn(normalized_params):
    params = denormalize_params(normalized_params, bounds_min, bounds_max)
    simulated = simulate_event(source, params, source_key)
    loss = WC_smooth_loss(
        sensor_points, *true_data, *simulated,
        lambda_poisson=1.0, lambda_time=0.0, tau=1.0)
    return loss

grad_fn = jit(value_and_grad(step_fn))

# ── Optimizer ────────────────────────────────────────────────────────
TRAINABLE = {'scatter_length', 'wall_reflection_rate',
             'sensor_reflection_rate', 'absorption_length'}

params = normalize_params(init_params, bounds_min, bounds_max)
mask = make_optimization_mask(params, TRAINABLE)
param_labels = jax.tree.map(lambda m: 'train' if m else 'freeze', mask)

warmup_steps = int(WARMUP_FRAC * ADAM_ITERS)
optimizer = optax.multi_transform({
    'train': optax.adam(learning_rate=ADAM_LR, b1=0.95, b2=0.99),
    'freeze': optax.set_to_zero(),
}, param_labels)
opt_state = optimizer.init(params)

# ── Optimization loop ────────────────────────────────────────────────
loss_history = []
param_history = []

for step in range(ADAM_ITERS):
    loss, grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # Clip normalized params to [0.01, 0.99]
    params = jax.tree.map(lambda p: jnp.clip(p, 0.01, 0.99), params)

    dp = denormalize_params(params, bounds_min, bounds_max)
    loss_history.append(float(loss))
    param_history.append([
        float(dp.scatter_length), float(dp.wall_reflection_rate),
        float(dp.sensor_reflection_rate), float(dp.absorption_length),
    ])

    if step % 10 == 0:
        print(f"  Step {step:3d}: loss={float(loss):.6f}  "
              f"scatter={float(dp.scatter_length):.1f}  "
              f"wall_refl={float(dp.wall_reflection_rate):.3f}")

# ── Save results ─────────────────────────────────────────────────────
final = denormalize_params(params, bounds_min, bounds_max)
results = {
    "loss_curve": loss_history,
    "param_history": param_history,
    "final_scatter_length": float(final.scatter_length),
    "final_wall_reflection": float(final.wall_reflection_rate),
    "final_sensor_reflection": float(final.sensor_reflection_rate),
    "final_absorption_length": float(final.absorption_length),
    "true_scatter_length": 50.0,
    "true_wall_reflection": 0.2,
    "true_sensor_reflection": 0.2,
    "true_absorption_length": 50.0,
    "init_params": [float(init_params.scatter_length), float(init_params.wall_reflection_rate),
                    float(init_params.sensor_reflection_rate), float(init_params.absorption_length)],
}

output = f"baseline_scripts/L2_4_baseline_{BACKEND}.json"
os.makedirs("baseline_scripts", exist_ok=True)
with open(output, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.perf_counter() - t_start
print(f"\nDone in {elapsed:.1f}s → {output}")
