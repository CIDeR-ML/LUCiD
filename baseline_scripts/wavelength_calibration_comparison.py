"""Compare calibration convergence: wavelength_mode=True vs False.

Replicates the grad_param_calibration_multi_init_no_qe notebook:
- SK_like detector, 4-param optimization (scatter, wall_refl, sensor_refl, absorption)
- Laser source at top of detector
- WC_smooth_loss with tau=2.0
- 500 Adam iterations with warmup

Runs three configurations:
1. wavelength_mode=False (scalar DetectorParams, scatter=50, absorption=50)
2. wavelength_mode=True, laser at 405nm (raw medium: L_scat=187m, L_abs=368m)
3. wavelength_mode=True, laser at 350nm (raw medium: L_scat=91m, L_abs=608m)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import (
    DetectorParams, laser_source, normalize_params, denormalize_params,
    default_bounds, make_optimization_mask,
)
from lucid.losses import WC_smooth_loss

GEOM = "config/SK_like_geom_config.json"
GRID_KW = dict(n_cap=150, n_angular=250, n_height=150)

det = generate_detector(GEOM)
sensor_points = jnp.array(det.all_points)
NUM_SENSORS = len(sensor_points)

# ── Settings ─────────────────────────────────────────────────────────
NPHOT = 500_000
K = 8
NPHOT_TRUE = 5_000_000
K_TRUE = 12
ADAM_LR = 0.05
ADAM_ITERS = 500
WARMUP_FRAC = 0.4

# True parameters for scalar mode
TRUE_SCALAR = DetectorParams.from_flat(
    scatter_length=50.0, wall_reflection_rate=0.2,
    sensor_reflection_rate=0.2, absorption_length=50.0,
    qe=0.065, qe_corrections=jnp.ones(NUM_SENSORS))

# For wavelength mode, scatter/absorption come from medium.
# We still need wall/sensor reflection and qe_corrections.
TRUE_WL = DetectorParams.from_flat(
    scatter_length=1.0,  # ignored in wavelength_mode
    wall_reflection_rate=0.2,
    sensor_reflection_rate=0.2,
    absorption_length=1.0,  # ignored in wavelength_mode
    qe=0.065,  # base QE (overridden by wavelength QE weighting)
    qe_corrections=jnp.ones(NUM_SENSORS))


def run_calibration(label, wavelength_mode, laser_wavelength, true_dp):
    """Run 4-param calibration and return results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    source = laser_source(
        position=[0.0, 0.0, det.H / 2 - 0.1],
        intensity=100_000_000,
        wavelength=laser_wavelength)

    sim_true = setup_event_simulator(
        GEOM, NPHOT_TRUE, temperature=None, K=K_TRUE,
        is_calibration=True, default_detector_params=true_dp,
        wavelength_mode=wavelength_mode, **GRID_KW)

    sim_opt = setup_event_simulator(
        GEOM, NPHOT, temperature=None, K=K,
        is_calibration=True, wavelength_mode=wavelength_mode, **GRID_KW)

    key = jax.random.PRNGKey(42)
    ks, kd, ki = jax.random.split(key, 3)

    true_data = jax.lax.stop_gradient(sim_true(source, kd))
    print(f"  True data: charge_sum={float(jnp.sum(true_data[0])):.1f}")

    # Initial guess (same perturbation for all runs)
    mults = jax.random.uniform(ki, (4,), minval=0.5, maxval=1.5)
    init = DetectorParams.from_flat(
        scatter_length=jnp.clip(true_dp.scattering.scatter_length * mults[0], 5., 100.),
        wall_reflection_rate=jnp.clip(true_dp.reflection.wall_reflection_rate * mults[1], 0.05, 0.5),
        sensor_reflection_rate=jnp.clip(true_dp.reflection.sensor_reflection_rate * mults[2], 0.05, 0.4),
        absorption_length=jnp.clip(true_dp.absorption.absorption_length * mults[3], 10., 500.),
        qe=true_dp.response.qe,
        qe_corrections=true_dp.per_pmt.qe_corrections)

    bmin, bmax = default_bounds(NUM_SENSORS)
    bmin = bmin._replace(
        scattering=bmin.scattering._replace(scatter_length=jnp.array(5.)),
        reflection=bmin.reflection._replace(wall_reflection_rate=jnp.array(0.05),
                                            sensor_reflection_rate=jnp.array(0.05)),
        absorption=bmin.absorption._replace(absorption_length=jnp.array(10.)))

    # Only optimize wall_reflection and sensor_reflection (scatter/absorption
    # come from medium in wavelength mode, so they're not optimizable there)
    if wavelength_mode:
        TRAINABLE = {'wall_reflection_rate', 'sensor_reflection_rate'}
    else:
        TRAINABLE = {'scatter_length', 'wall_reflection_rate',
                     'sensor_reflection_rate', 'absorption_length'}

    @jit
    def step_fn(p):
        return value_and_grad(lambda p: WC_smooth_loss(
            sensor_points, *true_data,
            *sim_opt(source, denormalize_params(p, bmin, bmax), ks),
            lambda_poisson=1.0, lambda_time=0.0, tau=2.0))(p)

    params = normalize_params(init, bmin, bmax)
    mask = make_optimization_mask(params, TRAINABLE)
    labels = jax.tree.map(lambda m: 'train' if m else 'freeze', mask)
    sched = optax.warmup_constant_schedule(init_value=0., peak_value=ADAM_LR, warmup_steps=int(WARMUP_FRAC * ADAM_ITERS))
    opt = optax.multi_transform({
        'train': optax.adam(learning_rate=sched, b1=0.95, b2=0.99),
        'freeze': optax.set_to_zero(),
    }, labels)
    opt_state = opt.init(params)

    # Warmup
    _ = step_fn(params); jax.block_until_ready(_)

    t0 = time.perf_counter()
    for i in range(ADAM_ITERS):
        loss, grads = step_fn(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = jax.tree.map(lambda p: jnp.clip(p, 0.01, 0.99), params)

        if i % 100 == 0 or i == ADAM_ITERS - 1:
            dp = denormalize_params(params, bmin, bmax)
            print(f"  Step {i:3d}: loss={float(loss):.6f}  "
                  f"wall_refl={float(dp.reflection.wall_reflection_rate):.3f}  "
                  f"sensor_refl={float(dp.reflection.sensor_reflection_rate):.3f}"
                  + (f"  scatter={float(dp.scattering.scatter_length):.1f}  "
                     f"absorption={float(dp.absorption.absorption_length):.1f}"
                     if not wavelength_mode else ""))

    elapsed = time.perf_counter() - t0
    dp = denormalize_params(params, bmin, bmax)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Final: wall_refl={float(dp.reflection.wall_reflection_rate):.3f} (true=0.200)  "
          f"sensor_refl={float(dp.reflection.sensor_reflection_rate):.3f} (true=0.200)")
    if not wavelength_mode:
        print(f"         scatter={float(dp.scattering.scatter_length):.2f} (true=50.00)  "
              f"absorption={float(dp.absorption.absorption_length):.2f} (true=50.00)")
    return dp, float(loss)


# ── Run all three configurations ─────────────────────────────────────

results = {}

# 1. Scalar mode (current behavior)
dp1, loss1 = run_calibration(
    "SCALAR MODE (wavelength_mode=False, scatter=50, absorption=50)",
    wavelength_mode=False, laser_wavelength=None, true_dp=TRUE_SCALAR)
results["scalar"] = (dp1, loss1)

# 2. Wavelength mode at 405nm
dp2, loss2 = run_calibration(
    "WAVELENGTH MODE (405nm laser, medium physics)",
    wavelength_mode=True, laser_wavelength=405.0, true_dp=TRUE_WL)
results["wl_405"] = (dp2, loss2)

# 3. Wavelength mode at 350nm
dp3, loss3 = run_calibration(
    "WAVELENGTH MODE (350nm laser, medium physics)",
    wavelength_mode=True, laser_wavelength=350.0, true_dp=TRUE_WL)
results["wl_350"] = (dp3, loss3)

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  {'Mode':<25s}  {'wall_refl':>10s}  {'sensor_refl':>12s}  {'loss':>10s}")
print(f"  {'True':25s}  {'0.200':>10s}  {'0.200':>12s}")
for name, (dp, loss) in results.items():
    print(f"  {name:25s}  {float(dp.reflection.wall_reflection_rate):10.3f}  "
          f"{float(dp.reflection.sensor_reflection_rate):12.3f}  {loss:10.6f}")
