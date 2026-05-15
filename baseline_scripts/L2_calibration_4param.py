"""L2-4: Calibration convergence — 4 scalar parameters, QE frozen.

Mirrors: grad_param_calibration_multi_init_no_qe.ipynb
Settings match the notebook: Nphot=500k, K=8, temperature=None, 500 Adam iters,
warmup schedule, WC_smooth_loss(tau=1.0, lambda_poisson=1.0, lambda_time=0.0).
Geometry: SK_like (cylinder approximation of SK detector).
"""
import os, sys, json, time

BACKEND = os.environ.get("LUCID_BACKEND", "lucid")
if BACKEND == "tools":
    sys.path.insert(0, os.getcwd())
    from tools.geometry import generate_detector
    from tools.simulation import setup_event_simulator
    from tools.detector_params import (
        DetectorParams, laser_source,
        normalize_params, denormalize_params, default_bounds,
        make_optimization_mask,
    )
    from tools.losses import WC_smooth_loss
else:
    sys.path.insert(0, os.getcwd())
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import (
        DetectorParams, laser_source,
        normalize_params, denormalize_params, default_bounds,
        make_optimization_mask,
    )
    from lucid.losses import WC_smooth_loss

GEOM = "config/SK_like_geom_config.json"

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
import optax

print(f"L2-4 Calibration 4-param | Backend: {BACKEND}")
t_start = time.perf_counter()

# ── Settings (matching notebook) ─────────────────────────────────────
NPHOT = 500_000
K = 8
NPHOT_TRUE = 5_000_000     # notebook uses 15M; 5M fits GPU memory
K_TRUE = 12
ADAM_LR = 0.05
ADAM_ITERS = 500
WARMUP_FRAC = 0.4
SOURCE_INTENSITY = 100_000_000

det = generate_detector(GEOM)
sensor_points = jnp.array(det.all_points)
NUM_SENSORS = len(sensor_points)

# True parameters — matching SK_physics_config.json
TRUE_PARAMS = DetectorParams(
    scatter_length=50.0,
    wall_reflection_rate=0.2,
    sensor_reflection_rate=0.2,
    absorption_length=50.0,
    qe=0.065,
    qe_corrections=jnp.ones(NUM_SENSORS),
)

source = laser_source(
    position=[0.0, 0.0, det.H / 2 - 0.1],
    intensity=SOURCE_INTENSITY,
)

bounds_min, bounds_max = default_bounds(NUM_SENSORS)
bounds_min = bounds_min._replace(
    scatter_length=jnp.array(5.0),
    wall_reflection_rate=jnp.array(0.05),
    sensor_reflection_rate=jnp.array(0.05),
    absorption_length=jnp.array(10.0),
)

# ── Simulators ───────────────────────────────────────────────────────
# Grid params matching tools defaults for cylinder
grid_kw = dict(n_cap=150, n_angular=250, n_height=150) if BACKEND == "lucid" else {}

simulate_event = setup_event_simulator(
    GEOM, NPHOT, temperature=None, K=K,
    is_data=False, is_calibration=True,
    max_sensors_per_cell=4, wavelength_mode=False, **grid_kw)

simulate_data = setup_event_simulator(
    GEOM, NPHOT_TRUE, temperature=None, K=K_TRUE,
    is_data=False, is_calibration=True,
    max_sensors_per_cell=4,
    default_detector_params=TRUE_PARAMS, wavelength_mode=False, **grid_kw)

# ── Generate true data ──────────────────────────────────────────────
key = jax.random.PRNGKey(42)
key_source, key_data, key_init = jax.random.split(key, 3)

print("Generating true data...")
true_data = jax.lax.stop_gradient(simulate_data(source, key_data))
print(f"  True data: charge_sum={float(jnp.sum(true_data[0])):.1f}, "
      f"active_sensors={int(jnp.sum(true_data[0] > 0))}")

# ── Initial guess (perturbed from true) ─────────────────────────────
mults = jax.random.uniform(key_init, (4,), minval=0.5, maxval=1.5)
init_params = DetectorParams(
    scatter_length=jnp.clip(TRUE_PARAMS.scatter_length * mults[0], 5.0, 100.0),
    wall_reflection_rate=jnp.clip(TRUE_PARAMS.wall_reflection_rate * mults[1], 0.05, 0.5),
    sensor_reflection_rate=jnp.clip(TRUE_PARAMS.sensor_reflection_rate * mults[2], 0.05, 0.4),
    absorption_length=jnp.clip(TRUE_PARAMS.absorption_length * mults[3], 10.0, 500.0),
    qe=TRUE_PARAMS.qe,
    qe_corrections=TRUE_PARAMS.qe_corrections,
)

print(f"  Init: scatter={float(init_params.scatter_length):.1f}  "
      f"wall_refl={float(init_params.wall_reflection_rate):.3f}  "
      f"sensor_refl={float(init_params.sensor_reflection_rate):.3f}  "
      f"absorption={float(init_params.absorption_length):.1f}")

# ── Loss function ────────────────────────────────────────────────────
@jit
def loss_fn(detector_params):
    simulated = simulate_event(source, detector_params, key_source)
    return WC_smooth_loss(
        sensor_points, *true_data, *simulated,
        lambda_poisson=1.0, lambda_time=0.0, tau=2.0)

@jit
def step_fn(normalized_params):
    return value_and_grad(
        lambda p: loss_fn(denormalize_params(p, bounds_min, bounds_max))
    )(normalized_params)

# ── Optimizer (warmup schedule matching notebook) ────────────────────
TRAINABLE = {'scatter_length', 'wall_reflection_rate',
             'sensor_reflection_rate', 'absorption_length'}

params = normalize_params(init_params, bounds_min, bounds_max)
mask = make_optimization_mask(params, TRAINABLE)
param_labels = jax.tree.map(lambda m: 'train' if m else 'freeze', mask)

warmup_steps = int(WARMUP_FRAC * ADAM_ITERS)
schedule = optax.warmup_constant_schedule(
    init_value=0.0, peak_value=ADAM_LR, warmup_steps=warmup_steps)

optimizer = optax.multi_transform({
    'train': optax.adam(learning_rate=schedule, b1=0.95, b2=0.99),
    'freeze': optax.set_to_zero(),
}, param_labels)
opt_state = optimizer.init(params)

# ── Optimization loop ────────────────────────────────────────────────
loss_history = []
param_history = []

# JIT warmup
_ = step_fn(params)
jax.block_until_ready(_)

t_opt_start = time.perf_counter()
for step in range(ADAM_ITERS):
    loss, grads = step_fn(params)
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

    if step % 50 == 0 or step == ADAM_ITERS - 1:
        print(f"  Step {step:3d}: loss={float(loss):.6f}  "
              f"scatter={float(dp.scatter_length):.1f}  "
              f"wall_refl={float(dp.wall_reflection_rate):.3f}  "
              f"sensor_refl={float(dp.sensor_reflection_rate):.3f}  "
              f"absorption={float(dp.absorption_length):.1f}")

t_opt_end = time.perf_counter()
print(f"  Optimization: {t_opt_end - t_opt_start:.1f}s ({ADAM_ITERS} steps)")

# ── Convergence check ────────────────────────────────────────────────
final = denormalize_params(params, bounds_min, bounds_max)
true_arr = np.array([50.0, 0.2, 0.2, 50.0])
final_arr = np.array([float(final.scatter_length), float(final.wall_reflection_rate),
                      float(final.sensor_reflection_rate), float(final.absorption_length)])
errors = np.abs(final_arr - true_arr)
rel_errors = errors / true_arr

print(f"\n  Convergence check:")
names = ["scatter_length", "wall_reflection", "sensor_reflection", "absorption_length"]
for i, name in enumerate(names):
    print(f"    {name:22s}: true={true_arr[i]:7.2f}  final={final_arr[i]:7.2f}  "
          f"error={errors[i]:7.3f}  rel_error={rel_errors[i]:.3f}")

# ── Save results ─────────────────────────────────────────────────────
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
    "settings": {
        "Nphot": NPHOT, "K": K, "Nphot_true": NPHOT_TRUE, "K_true": K_TRUE,
        "adam_lr": ADAM_LR, "adam_iters": ADAM_ITERS, "warmup_frac": WARMUP_FRAC,
        "temperature": None, "source_intensity": SOURCE_INTENSITY,
        "geometry": GEOM,
    },
}

output = f"baseline_scripts/L2_4_baseline_{BACKEND}.json"
os.makedirs("baseline_scripts", exist_ok=True)
with open(output, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.perf_counter() - t_start
print(f"\nDone in {elapsed:.1f}s → {output}")
