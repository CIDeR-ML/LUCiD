# Level 1 Baseline Scripts — Design Document

## Overview

Level 1 covers **forward pass matching and gradient values at specific points**: short, fast tests that verify the core simulation pipeline produces identical numerical results between the baseline (`tools.*`) and the refactored code (`lucid.*`).

All scripts share a common import shim (Section 0) and use the WCTE cylinder config (`config/WCTE_geom_config.json`) unless otherwise noted, because the regular cylinder is simpler and faster than `superk` (which requires a ROOT connection table file). WCTE is a small cylinder (R=2m, H=4m, 2500 sensors, sensor_radius=0.04m) which makes tests fast. NOTE: `SK_like_geom_config.json` has a trailing comma (invalid JSON) and should not be used until fixed.

**Tolerance convention**: All comparisons use `np.allclose(a, b, atol=1e-5, rtol=1e-5)` for forward-pass values and `np.allclose(a, b, atol=1e-4, rtol=1e-4)` for gradients (JAX float32 accumulation can differ at this scale).

---

## Section 0: Common Import Shim

Every script begins with this pattern:

```python
#!/usr/bin/env python3
"""L1-N: <description>"""
import sys, os, json, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

# --- Configurable import prefix ---
# Set LUCID_BASELINE=1 to run against the old tools.* code at /tmp/lucid-baseline/
USE_BASELINE = os.environ.get('LUCID_BASELINE', '0') == '1'

if USE_BASELINE:
    sys.path.insert(0, '/tmp/lucid-baseline')
    from tools.simulation import (
        setup_event_simulator,
        photon_iteration_sample, photon_iteration_update_factors,
        photon_iteration_update_factors_safe,
        make_hits_simulation, make_hits_data, make_hits_likelihood,
        normalize,
    )
    from tools.detector_params import (
        ParticleParams, DetectorParams,
        load_detector_params, isotropic_source, laser_source,
    )
    from tools.geometry import generate_detector
    from tools.losses import WC_loss, WC_smooth_loss, poisson_nll
    from tools.optimization.losses import (
        counts_loss, origin_time_loss, cone_time_loss,
        first_arrival_nll, segment_logsumexp,
    )
    BASE_DIR = '/tmp/lucid-baseline/'
else:
    sys.path.insert(0, '/home/oalterka/desktop_linux/diffWC/diffCherenkov')
    from lucid.simulation import (
        setup_event_simulator,
        photon_iteration_sample, photon_iteration_update_factors,
        photon_iteration_update_factors_safe,
        make_hits_simulation, make_hits_data, make_hits_likelihood,
    )
    from lucid.simulation.optics import normalize
    from lucid.detector_params import (
        ParticleParams, DetectorParams,
        load_detector_params, isotropic_source, laser_source,
    )
    from lucid.geometry import generate_detector
    from lucid.losses import (
        WC_loss, WC_smooth_loss, poisson_nll, counts_loss,
        origin_time_loss, cone_time_loss,
        first_arrival_nll, segment_logsumexp,
    )
    BASE_DIR = '/home/oalterka/desktop_linux/diffWC/diffCherenkov/'

GEOM_CONFIG = os.path.join(BASE_DIR, 'config/WCTE_geom_config.json')
PHYSICS_CONFIG = os.path.join(BASE_DIR, 'config/SK_physics_config.json')

def save_baseline(name, data):
    """Save numpy arrays to .npz for cross-run comparison."""
    out_dir = os.path.join(os.path.dirname(__file__), 'baselines')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'{name}.npz'), **data)

def load_baseline(name):
    """Load previously saved baseline."""
    out_dir = os.path.join(os.path.dirname(__file__), 'baselines')
    return dict(np.load(os.path.join(out_dir, f'{name}.npz'), allow_pickle=True))

def compare(name, computed, atol=1e-5, rtol=1e-5):
    """Compare computed dict against saved baseline."""
    saved = load_baseline(name)
    all_pass = True
    for key in computed:
        a = np.asarray(computed[key])
        b = np.asarray(saved[key])
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(a - b))
            print(f"  FAIL {key}: max diff = {max_diff}")
            all_pass = False
        else:
            print(f"  PASS {key}")
    return all_pass
```

---

## L1-1: Propagator Output

**Filename**: `test_L1_01_propagator.py`

**What it tests**: Fixed axis-aligned rays from the detector center through the cylinder propagator produce identical sensor_weights, positions, and normals.

**Imports** (beyond common shim):
```python
from lucid.propagation.cylinder import create_photon_propagator
# Baseline: from tools.propagation.cylinder import create_photon_propagator
from lucid.geometry import generate_detector
```

**Setup**:
```python
detector = generate_detector(GEOM_CONFIG)
sensor_points = jnp.array(detector.all_points)
sensor_radius = detector.S_radius
R, H = detector.R, detector.H

propagate = create_photon_propagator(
    sensor_points, sensor_radius, r=R, h=H,
    temperature=0.10, max_sensors_per_cell=4)
```

**Computation**:
```python
# 3 axis-aligned rays from origin
origins = jnp.zeros((3, 3))
directions = jnp.array([
    [1.0, 0.0, 0.0],   # +X  (should hit cylinder wall)
    [0.0, 1.0, 0.0],   # +Y  (should hit cylinder wall)
    [0.0, 0.0, 1.0],   # +Z  (should hit top cap)
])

result = propagate(origins, directions)
```

**Values to capture**:
```python
{
    'sensor_weights': np.asarray(result['sensor_weights']),       # (max_sensors_per_cell, 3)
    'positions': np.asarray(result['positions']),                  # (3, 3)
    'normals': np.asarray(result['normals']),                      # (3, 3)
    'sensor_indices': np.asarray(result['sensor_indices']),        # (max_sensors_per_cell, 3)
    'times': np.asarray(result['times']),                          # (max_sensors_per_cell, 3)
    'inside_sensor': np.asarray(result['inside_sensor']),          # (max_sensors_per_cell, 3)
}
```

**Expected runtime**: ~5s (detector construction + JIT compilation of propagator).

---

## L1-2: Forward Pass (Track Mode, Counts-Based)

**Filename**: `test_L1_02_forward_counts.py`

**What it tests**: `setup_event_simulator` in track mode (is_data=False, default_detector_params=True) with fixed ParticleParams and PRNG key produces identical (charges, times) output.

**Imports**: Common shim only.

**Setup**:
```python
simulator = setup_event_simulator(
    GEOM_CONFIG,
    n_photons=1000,
    temperature=0.10,
    K=3,
    is_data=False,
    is_calibration=False,
    max_sensors_per_cell=4,
    physics_config=PHYSICS_CONFIG,
    default_detector_params=True,
    particle='muon',
)
```

**Computation**:
```python
track = ParticleParams(
    energy=jnp.array(800.0),
    position=jnp.array([0.0, 0.0, 0.0]),
    theta=jnp.array(jnp.pi / 3),
    phi=jnp.array(jnp.pi / 4),
    t0=jnp.array(0.0),
)
key = jax.random.PRNGKey(42)

# Track mode returns (charges, times) — the simulator internally uses
# make_hits_likelihood (returns 4-tuple) but wraps into 2-tuple via the
# _simulation_without_data_impl path.
# Actually: track mode returns 4-tuple (log_w, flat_times, flat_indices, total_charge).
# This is because it goes through make_hits_likelihood_fn.
result = simulator(track, key)
```

**Note on return type**: The refactored code's track mode uses `make_hits_likelihood_fn` internally, returning a 4-tuple `(log_w, flat_times, flat_indices, total_charge)`. For this test we capture all elements. If the baseline returns 2-tuple `(charges, times)` instead, the script must handle both cases.

**Values to capture**:
```python
if len(result) == 4:
    log_w, flat_times, flat_indices, total_charge = result
    data = {
        'log_w_shape': np.array(log_w.shape),
        'log_w_sum': np.array(float(jnp.sum(log_w))),
        'log_w_first20': np.asarray(log_w[:20]),
        'flat_times_sum': np.array(float(jnp.sum(flat_times))),
        'flat_indices_first20': np.asarray(flat_indices[:20]),
        'total_charge': np.asarray(total_charge),
        'total_charge_sum': np.array(float(jnp.sum(total_charge))),
    }
elif len(result) == 2:
    charges, times = result
    data = {
        'charges': np.asarray(charges),
        'times': np.asarray(times),
        'charges_sum': np.array(float(jnp.sum(charges))),
        'charges_shape': np.array(charges.shape),
        'times_sum': np.array(float(jnp.sum(times))),
        'n_hit_sensors': np.array(int(jnp.sum(charges > 0))),
    }
```

**Expected runtime**: ~10-15s (SIREN model load + JIT compilation + 1 forward pass at Nphot=1000, K=3).

---

## L1-3: Forward Pass (Track Mode, Likelihood)

**Filename**: `test_L1_03_forward_likelihood.py`

**What it tests**: Track-mode simulator returns the 4-tuple `(log_w, flat_times, flat_indices, total_charge)` with correct shapes and values at a fixed point.

**Imports**: Common shim only.

**Setup**: Same as L1-2 (track mode already returns 4-tuple via `make_hits_likelihood`).

**Computation**:
```python
track = ParticleParams(
    energy=jnp.array(1050.0),
    position=jnp.array([5.0, -3.0, 2.0]),
    theta=jnp.array(1.2),
    phi=jnp.array(0.8),
    t0=jnp.array(0.0),
)
key = jax.random.PRNGKey(99)

result = simulator(track, key)
log_w, flat_times, flat_indices, total_charge = result
```

**Values to capture**:
```python
NUM_SENSORS = 2500  # WCTE
{
    'log_w_shape': np.array(log_w.shape),
    'flat_times_shape': np.array(flat_times.shape),
    'flat_indices_shape': np.array(flat_indices.shape),
    'total_charge_shape': np.array(total_charge.shape),
    # Verify total_charge has correct number of sensors
    'total_charge_len': np.array(len(total_charge)),
    # Aggregate statistics
    'total_charge_sum': np.array(float(jnp.sum(total_charge))),
    'total_charge_max': np.array(float(jnp.max(total_charge))),
    'n_hit_sensors': np.array(int(jnp.sum(total_charge > 0))),
    'log_w_valid_count': np.array(int(jnp.sum(log_w > -20))),
    'flat_times_mean_valid': np.array(float(
        jnp.sum(jnp.where(flat_times > 0, flat_times, 0.0)) /
        (jnp.sum(flat_times > 0) + 1e-8)
    )),
    # First 50 values for exact comparison
    'log_w_first50': np.asarray(log_w[:50]),
    'flat_times_first50': np.asarray(flat_times[:50]),
    'total_charge_first100': np.asarray(total_charge[:100]),
}
```

**Expected runtime**: ~10-15s.

---

## L1-4: Forward Pass (Calibration Mode)

**Filename**: `test_L1_04_forward_calibration.py`

**What it tests**: `setup_event_simulator` with `is_calibration=True` and an `isotropic_source` produces identical `(charges, times)` output.

**Imports**: Common shim only.

**Setup**:
```python
simulator = setup_event_simulator(
    GEOM_CONFIG,
    n_photons=5000,
    temperature=0.10,
    K=3,
    is_data=False,
    is_calibration=True,
    max_sensors_per_cell=4,
    physics_config=PHYSICS_CONFIG,
    default_detector_params=True,
)
```

**Computation**:
```python
source = isotropic_source(
    position=jnp.array([0.0, 0.0, 0.0]),
    intensity=1_000_000.0,
)
key = jax.random.PRNGKey(77)

charges, times = simulator(source, key)
```

**Values to capture**:
```python
{
    'charges': np.asarray(charges),
    'times': np.asarray(times),
    'charges_sum': np.array(float(jnp.sum(charges))),
    'charges_shape': np.array(charges.shape),
    'n_hit_sensors': np.array(int(jnp.sum(charges > 0))),
    'times_mean_hit': np.array(float(
        jnp.sum(jnp.where(charges > 0, times, 0.0)) /
        (jnp.sum(charges > 0) + 1e-8)
    )),
    'charges_first100': np.asarray(charges[:100]),
    'times_first100': np.asarray(times[:100]),
}
```

**Expected runtime**: ~5-8s (no SIREN load needed for calibration mode).

---

## L1-5: Forward Pass (Data Mode)

**Filename**: `test_L1_05_forward_data.py`

**What it tests**: `setup_event_simulator` with `is_data=True` and photon_data from a ROOT file produces identical `(charges, times)` output.

**Imports** (beyond common shim):
```python
from lucid.generate import read_photon_data_from_photonsim
# Baseline: from tools.generate import read_photon_data_from_photonsim
```

**Setup**:
```python
ROOT_FILE = os.path.join(BASE_DIR, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')

# Skip if ROOT file unavailable
if not os.path.exists(ROOT_FILE):
    print("SKIP: ROOT file not found at", ROOT_FILE)
    sys.exit(0)

data_simulator = setup_event_simulator(
    GEOM_CONFIG,
    n_photons=1_000_000,     # padded photon array size
    temperature=0.0,
    K=3,
    is_data=True,
    is_calibration=False,
    max_sensors_per_cell=4,
    physics_config=PHYSICS_CONFIG,
    default_detector_params=True,
    apply_smearing=False,     # disable smearing for determinism
)
```

**Computation**:
```python
# Load ROOT photon data (entry 2)
photon_data = read_photon_data_from_photonsim(ROOT_FILE, entry_idx=2)

# Fixed track params
track = ParticleParams(
    energy=jnp.array(1050.0),
    position=jnp.array([0.0, 0.0, 0.0]),
    theta=jnp.array(jnp.pi / 2),
    phi=jnp.array(jnp.pi / 6),
    t0=jnp.array(0.0),
)

# Prepare photon_data dict with rotation/translation
import lucid.utils as utils  # or tools.utils
# (The notebook code builds photon_data as a dict with rotation/translation fields)
# For simplicity, use apply_rotation=False, apply_translation=False

photon_data_dict = {
    'photon_origins': jnp.array(photon_data['photon_origins']),
    'photon_directions': jnp.array(photon_data['photon_directions']),
    'photon_times': jnp.zeros(photon_data['photon_origins'].shape[0]),
    'N': photon_data['N'],
    'apply_rotation': jnp.array(False),
    'rotation_axis': jnp.array([0.0, 0.0, 1.0]),
    'rotation_angle': jnp.array(0.0),
    'apply_translation': jnp.array(False),
    'translation_vector': jnp.array([0.0, 0.0, 0.0]),
}

key = jax.random.PRNGKey(55)
charges, times = data_simulator(track, key, photon_data_dict)
```

**Values to capture**:
```python
{
    'charges': np.asarray(charges),
    'times': np.asarray(times),
    'charges_sum': np.array(float(jnp.sum(charges))),
    'n_hit_sensors': np.array(int(jnp.sum(charges > 0))),
    'charges_first100': np.asarray(charges[:100]),
    'times_first100': np.asarray(times[:100]),
}
```

**Expected runtime**: ~15-20s (ROOT file load + JIT + propagation).

---

## L1-6: Loss Values at True Parameters

**Filename**: `test_L1_06_loss_values.py`

**What it tests**: Each loss function used by the notebooks produces identical scalar loss values when given the same fixed inputs.

**Imports**: Common shim covers all loss imports.

**Setup**: Create synthetic data tensors once.
```python
NUM_SENSORS = 100  # Use small count for speed
key = jax.random.PRNGKey(42)

# Synthetic "observed" data
k1, k2, k3 = jax.random.split(key, 3)
true_charges = jax.random.uniform(k1, (NUM_SENSORS,), minval=0, maxval=50.0)
true_times = jax.random.uniform(k2, (NUM_SENSORS,), minval=10.0, maxval=100.0)

# Synthetic "predicted" data (slightly offset from true)
pred_charges = true_charges * 1.05 + 0.5
pred_times = true_times + jax.random.normal(k3, (NUM_SENSORS,)) * 2.0

# Sensor positions (random on unit sphere scaled)
k4, k5 = jax.random.split(k3)
sensor_points = jax.random.normal(k4, (NUM_SENSORS, 3))
sensor_points = sensor_points / jnp.linalg.norm(sensor_points, axis=1, keepdims=True) * 16.0

# Vertex position and t0
position = jnp.array([1.0, -2.0, 3.0])
t0 = jnp.array(5.0)

# Likelihood-mode synthetic data
N_PHOTONS = 500
k6, k7, k8 = jax.random.split(k5, 3)
log_w = jax.random.normal(k6, (N_PHOTONS,)) - 5.0
flat_times = jax.random.uniform(k7, (N_PHOTONS,), minval=10.0, maxval=100.0)
flat_indices = jax.random.randint(k8, (N_PHOTONS,), 0, NUM_SENSORS)
t_obs = jax.random.uniform(k5, (NUM_SENSORS,), minval=10.0, maxval=100.0)
```

**Computation**: Evaluate each loss function.
```python
results = {}

# 1. counts_loss (Poisson NLL on aggregated charges)
results['counts_loss'] = np.array(float(counts_loss(true_charges, pred_charges)))

# 2. poisson_nll (alias for counts_loss)
results['poisson_nll'] = np.array(float(poisson_nll(true_charges, pred_charges)))

# 3. origin_time_loss
results['origin_time_loss'] = np.array(float(
    origin_time_loss(position, sensor_points, true_times, true_charges, t0)
))

# 4. origin_time_loss with custom tau
results['origin_time_loss_tau05'] = np.array(float(
    origin_time_loss(position, sensor_points, true_times, true_charges, t0, tau=0.5)
))

# 5. cone_time_loss
results['cone_time_loss'] = np.array(float(
    cone_time_loss(true_charges, pred_times, true_times, t0, tau=0.23)
))

# 6. first_arrival_nll
fa_loss = first_arrival_nll(log_w, flat_times, flat_indices, t_obs, tau=0.15, num_detectors=NUM_SENSORS)
results['first_arrival_nll_sum'] = np.array(float(jnp.sum(fa_loss)))
results['first_arrival_nll_first10'] = np.asarray(fa_loss[:10])

# 7. WC_loss (charge-only)
results['WC_loss_charge'] = np.array(float(
    WC_loss(sensor_points, true_charges, true_times,
            pred_charges, pred_times,
            lambda_poisson=1.0, lambda_time=0.0)
))

# 8. WC_loss (charge + time)
results['WC_loss_both'] = np.array(float(
    WC_loss(sensor_points, true_charges, true_times,
            pred_charges, pred_times,
            lambda_poisson=1.0, lambda_time=1.0)
))

# 9. WC_smooth_loss
results['WC_smooth_loss'] = np.array(float(
    WC_smooth_loss(sensor_points, true_charges, true_times,
                   pred_charges, pred_times,
                   tau=1.0, lambda_poisson=1.0, lambda_time=0.0)
))

# 10. Combined 3-term likelihood loss (as used by notebooks)
from jax import lax
charge_loss = poisson_nll(true_charges, pred_charges)
time_loss_val = jnp.mean(fa_loss[fa_loss < 1e5])  # mask invalid
vertex_loss = origin_time_loss(position, sensor_points, true_times, true_charges, t0)
s = 1e-6
c, t, v = charge_loss + s, time_loss_val + s, vertex_loss + s
combined_3term = (
    jnp.sqrt(c * t * v)
    + jnp.sqrt(c * lax.stop_gradient(t * v))
    + jnp.sqrt(v * lax.stop_gradient(t * c))
)
results['combined_3term'] = np.array(float(combined_3term))

# 11. Product loss (as used by counts-based notebooks)
product_loss = jnp.sqrt(
    (counts_loss(true_charges, pred_charges) + 1e-6) *
    (cone_time_loss(true_charges, pred_times, true_times, t0) + 1e-6) *
    (origin_time_loss(position, sensor_points, true_times, true_charges, t0) + 1e-6)
)
results['product_loss'] = np.array(float(product_loss))
```

**Values to capture**: The `results` dict (12 scalar values + 1 array).

**Expected runtime**: ~2s (pure JAX compute, no simulation).

---

## L1-7: Gradient Values

**Filename**: `test_L1_07_gradients.py`

**What it tests**: `value_and_grad` of loss functions w.r.t. ParticleParams (7 parameters) and DetectorParams (4 calibration scalars) produce identical gradient vectors.

**Imports**: Common shim.

**Setup**: Same synthetic data as L1-6, plus:
```python
# ParticleParams for gradient computation
track = ParticleParams(
    energy=jnp.array(800.0),
    position=jnp.array([1.0, -2.0, 3.0]),
    theta=jnp.array(jnp.pi / 3),
    phi=jnp.array(jnp.pi / 4),
    t0=jnp.array(5.0),
)

# DetectorParams for calibration gradient
det_params = DetectorParams(
    scatter_length=jnp.array(50.0),
    wall_reflection_rate=jnp.array(0.2),
    sensor_reflection_rate=jnp.array(0.2),
    absorption_length=jnp.array(50.0),
    qe=jnp.array(0.065),
    qe_corrections=jnp.ones(NUM_SENSORS),
)
```

**Computation**:
```python
results = {}

# 1. Gradient of counts_loss w.r.t. pred_charges
grad_fn_1 = jax.grad(lambda p: counts_loss(true_charges, p))
g = grad_fn_1(pred_charges)
results['grad_counts_loss_wrt_pred'] = np.asarray(g[:20])

# 2. Gradient of product_loss w.r.t. ParticleParams
#    This requires wrapping loss to accept ParticleParams and run simulator.
#    For a unit test without the full simulator, compute grad of a composed
#    loss w.r.t. position and t0 (the analytically accessible params).
def vertex_loss_fn(pos, t0_val):
    return origin_time_loss(pos, sensor_points, true_times, true_charges, t0_val)

vl_val, (g_pos, g_t0) = jax.value_and_grad(vertex_loss_fn, argnums=(0, 1))(
    track.position, track.t0)
results['vertex_loss_val'] = np.array(float(vl_val))
results['grad_vertex_wrt_pos'] = np.asarray(g_pos)
results['grad_vertex_wrt_t0'] = np.array(float(g_t0))

# 3. Gradient of cone_time_loss w.r.t. t0
def cone_fn(t0_val):
    return cone_time_loss(true_charges, pred_times, true_times, t0_val, tau=0.23)
cone_val, g_cone_t0 = jax.value_and_grad(cone_fn)(track.t0)
results['cone_loss_val'] = np.array(float(cone_val))
results['grad_cone_wrt_t0'] = np.array(float(g_cone_t0))

# 4. Gradient of WC_smooth_loss w.r.t. pred_charges (simulates calibration gradient)
def wc_smooth_fn(pred_q):
    return WC_smooth_loss(sensor_points, true_charges, true_times,
                          pred_q, pred_times,
                          tau=1.0, lambda_poisson=1.0, lambda_time=0.0)
wc_val, g_wc = jax.value_and_grad(wc_smooth_fn)(pred_charges)
results['WC_smooth_val'] = np.array(float(wc_val))
results['grad_WC_smooth_wrt_pred_first20'] = np.asarray(g_wc[:20])

# 5. Gradient of first_arrival_nll w.r.t. log_w
def fa_fn(lw):
    loss_per_sensor = first_arrival_nll(lw, flat_times, flat_indices, t_obs, 0.15, NUM_SENSORS)
    hit_mask = true_charges > 0
    return jnp.sum(jnp.where(hit_mask, loss_per_sensor, 0.0)) / (jnp.sum(hit_mask) + 1e-8)
fa_val, g_fa = jax.value_and_grad(fa_fn)(log_w)
results['fa_nll_val'] = np.array(float(fa_val))
results['grad_fa_wrt_logw_first20'] = np.asarray(g_fa[:20])

# 6. Gradient of 3-term combined loss w.r.t. position
def combined_fn(pos):
    vl = origin_time_loss(pos, sensor_points, true_times, true_charges, track.t0)
    cl = counts_loss(true_charges, pred_charges)
    s = 1e-6
    c, v = cl + s, vl + s
    # Simplified 2-term (time_loss is constant w.r.t. position)
    return jnp.sqrt(c * v) + jnp.sqrt(v * jax.lax.stop_gradient(c))
combined_val, g_combined = jax.value_and_grad(combined_fn)(track.position)
results['combined_val'] = np.array(float(combined_val))
results['grad_combined_wrt_pos'] = np.asarray(g_combined)
```

**Values to capture**: The `results` dict (~15 entries mixing scalars and small arrays).

**Expected runtime**: ~3s.

---

## L1-8: Photon Step Functions

**Filename**: `test_L1_08_photon_step.py`

**What it tests**: `photon_iteration_sample` and `photon_iteration_update_factors` with fixed inputs produce identical 6-tuple outputs (new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor).

**Imports**: Common shim (`photon_iteration_sample`, `photon_iteration_update_factors`, `normalize`).

**Setup**: None beyond imports.

**Computation**:
```python
# Fixed photon state
position = jnp.array([5.0, 3.0, 1.0])
direction = normalize(jnp.array([1.0, 0.5, -0.3]))
time = jnp.array(10.0)
surface_distance = jnp.array(8.5)
normal = normalize(jnp.array([1.0, 0.0, 0.0]))

# Physics params
scatter_length = jnp.array(50.0)
wall_reflection_rate = jnp.array(0.2)
sensor_reflection_rate = jnp.array(0.2)
absorption_length = jnp.array(50.0)
speed_of_light = jnp.array(0.299792 / 1.33)

key = jax.random.PRNGKey(123)

# --- Test 1: photon_iteration_sample with hit_sensor=False (wall hit)
out_sample_wall = photon_iteration_sample(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
    absorption_length,
    jnp.array(False), key, speed_of_light)

# --- Test 2: photon_iteration_sample with hit_sensor=True (sensor hit)
out_sample_sensor = photon_iteration_sample(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
    absorption_length,
    jnp.array(True), key, speed_of_light)

# --- Test 3: photon_iteration_update_factors with hit_sensor=False
out_update_wall = photon_iteration_update_factors(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
    absorption_length,
    jnp.array(False), key, speed_of_light)

# --- Test 4: photon_iteration_update_factors with hit_sensor=True
out_update_sensor = photon_iteration_update_factors(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
    absorption_length,
    jnp.array(True), key, speed_of_light)

results = {}
for label, out in [('sample_wall', out_sample_wall), ('sample_sensor', out_sample_sensor),
                    ('update_wall', out_update_wall), ('update_sensor', out_update_sensor)]:
    new_pos, new_dir, new_time, detect_prob, refl_atten, cont_factor = out
    results[f'{label}_pos'] = np.asarray(new_pos)
    results[f'{label}_dir'] = np.asarray(new_dir)
    results[f'{label}_time'] = np.array(float(new_time))
    results[f'{label}_detect'] = np.array(float(detect_prob))
    results[f'{label}_refl'] = np.array(float(refl_atten))
    results[f'{label}_cont'] = np.array(float(cont_factor))
```

**Values to capture**: 24 entries (6 values x 4 test cases).

**Expected runtime**: ~2s (JIT compile + run 4 function calls).

---

## L1-9: Custom VJP Gradient

**Filename**: `test_L1_09_custom_vjp.py`

**What it tests**: `jax.grad` through `photon_iteration_update_factors_safe` produces identical gradient values, verifying the custom VJP with NaN sanitization works identically.

**Imports**: Common shim (`photon_iteration_update_factors_safe`).

**Setup**: Same fixed photon state as L1-8.

**Computation**:
```python
# Compute gradient of the sum of all 6 outputs w.r.t. each differentiable input.
# We differentiate w.r.t. position (arg 0), direction (arg 1), time (arg 2),
# surface_distance (arg 3), scatter_length (arg 5), wall_reflection_rate (arg 6),
# sensor_reflection_rate (arg 7), absorption_length (arg 8).
# (normal=arg4 and hit_sensor=arg9 and rng_key=arg10 and c=arg11 are not diffable or fixed.)

def scalar_output(position, direction, time, surface_distance,
                  normal, scatter_length, wall_reflection_rate,
                  sensor_reflection_rate, absorption_length,
                  hit_sensor, rng_key, speed_of_light):
    out = photon_iteration_update_factors_safe(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate,
        sensor_reflection_rate, absorption_length,
        hit_sensor, rng_key, speed_of_light)
    # Sum all outputs into a scalar for grad
    return sum(jnp.sum(o) for o in out)

grad_fn = jax.grad(scalar_output, argnums=(0, 1, 2, 3, 5, 6, 7, 8))
grads = grad_fn(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate,
    sensor_reflection_rate, absorption_length,
    jnp.array(False), key, speed_of_light)

results = {}
grad_names = ['pos', 'dir', 'time', 'surface_dist',
              'scatter_len', 'wall_refl', 'sensor_refl', 'abs_len']
for name, g in zip(grad_names, grads):
    results[f'grad_{name}'] = np.asarray(g)

# Also test with hit_sensor=True (sensor path)
grads_sensor = grad_fn(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate,
    sensor_reflection_rate, absorption_length,
    jnp.array(True), key, speed_of_light)
for name, g in zip(grad_names, grads_sensor):
    results[f'grad_sensor_{name}'] = np.asarray(g)
```

**Values to capture**: 16 entries (8 gradient values x 2 modes).

**Expected runtime**: ~3s.

---

## L1-10: Sensor Response Functions

**Filename**: `test_L1_10_sensor_response.py`

**What it tests**: `make_hits_simulation`, `make_hits_likelihood`, and `make_hits_data` with fixed synthetic inputs produce identical outputs.

**Imports**: Common shim (`make_hits_simulation`, `make_hits_data`, `make_hits_likelihood`).

**Setup**:
```python
NUM_SENSORS = 200
N_FLAT = 3000  # simulates K*max_sensors*Nphot flattened

key = jax.random.PRNGKey(42)
k1, k2, k3, k4 = jax.random.split(key, 4)

# Synthetic propagation outputs
flat_weights = jax.random.uniform(k1, (N_FLAT,), minval=0.0, maxval=1.0)
# Zero out ~50% to simulate photons missing sensors
mask = jax.random.bernoulli(k2, 0.5, (N_FLAT,))
flat_weights = flat_weights * mask

flat_indices = jax.random.randint(k3, (N_FLAT,), 0, NUM_SENSORS)
flat_times = jax.random.uniform(k4, (N_FLAT,), minval=1.0, maxval=200.0)
# Zero out times where weights are zero
flat_times = jnp.where(flat_weights > 0, flat_times, 0.0)

qe = 0.065
qe_corrections = jnp.ones(NUM_SENSORS)
```

**Computation**:
```python
results = {}

# 1. make_hits_simulation
charges_sim, times_sim = make_hits_simulation(
    flat_weights, flat_indices, flat_times, NUM_SENSORS,
    qe=qe, qe_corrections=qe_corrections)
results['sim_charges'] = np.asarray(charges_sim)
results['sim_times'] = np.asarray(times_sim)
results['sim_charges_sum'] = np.array(float(jnp.sum(charges_sim)))
results['sim_n_hit'] = np.array(int(jnp.sum(charges_sim > 0)))

# 2. make_hits_likelihood
log_w, safe_times, ret_indices, total_charge = make_hits_likelihood(
    flat_weights, flat_indices, flat_times, NUM_SENSORS,
    qe=qe, qe_corrections=qe_corrections)
results['like_log_w_first50'] = np.asarray(log_w[:50])
results['like_safe_times_first50'] = np.asarray(safe_times[:50])
results['like_total_charge'] = np.asarray(total_charge)
results['like_total_charge_sum'] = np.array(float(jnp.sum(total_charge)))
results['like_valid_count'] = np.array(int(jnp.sum(log_w > -20)))

# 3. make_hits_data (with fixed rng_key, no smearing)
qe_key = jax.random.PRNGKey(999)
charges_data, times_data = make_hits_data(
    flat_weights, flat_indices, flat_times, NUM_SENSORS,
    qe=qe, rng_key=qe_key, apply_smearing=False)
results['data_charges'] = np.asarray(charges_data)
results['data_times'] = np.asarray(times_data)
results['data_charges_sum'] = np.array(float(jnp.sum(charges_data)))
results['data_n_hit'] = np.array(int(jnp.sum(charges_data > 0)))

# 4. make_hits_data (with smearing)
charges_smear, times_smear = make_hits_data(
    flat_weights, flat_indices, flat_times, NUM_SENSORS,
    qe=qe, rng_key=qe_key, apply_smearing=True)
results['smear_charges'] = np.asarray(charges_smear)
results['smear_times'] = np.asarray(times_smear)
results['smear_charges_sum'] = np.array(float(jnp.sum(charges_smear)))
```

**Values to capture**: ~15 entries.

**Expected runtime**: ~2s.

---

## Summary Table

| Script | Filename | Tests | Approx Runtime |
|--------|----------|-------|----------------|
| L1-1 | `test_L1_01_propagator.py` | Cylinder propagator output for 3 axis-aligned rays | 5s |
| L1-2 | `test_L1_02_forward_counts.py` | Track mode forward pass (counts output) | 10-15s |
| L1-3 | `test_L1_03_forward_likelihood.py` | Track mode forward pass (likelihood 4-tuple) | 10-15s |
| L1-4 | `test_L1_04_forward_calibration.py` | Calibration mode with isotropic source | 5-8s |
| L1-5 | `test_L1_05_forward_data.py` | Data mode with ROOT photon file | 15-20s |
| L1-6 | `test_L1_06_loss_values.py` | All 11 loss function values at fixed inputs | 2s |
| L1-7 | `test_L1_07_gradients.py` | Gradients of 6 loss functions w.r.t. params | 3s |
| L1-8 | `test_L1_08_photon_step.py` | photon_iteration_sample and update_factors (4 configs) | 2s |
| L1-9 | `test_L1_09_custom_vjp.py` | Custom VJP gradient through safe wrapper | 3s |
| L1-10 | `test_L1_10_sensor_response.py` | make_hits_simulation/likelihood/data outputs | 2s |
| **Total** | | | **~60-75s** |

## Execution Protocol

1. **Generate baselines** (run once against the fixed baseline code):
   ```bash
   cd /tmp/lucid-baseline
   LUCID_BASELINE=1 python tests/baseline/test_L1_01_propagator.py --save
   # ... repeat for all 10 scripts
   ```

2. **Verify refactored code** (run against lucid/):
   ```bash
   cd /home/oalterka/desktop_linux/diffWC/diffCherenkov
   python tests/baseline/test_L1_01_propagator.py --compare
   # ... repeat for all 10 scripts
   ```

3. **CI integration**: Add `--compare --fail-fast` flag to exit with non-zero status on first mismatch.

## Key Design Decisions

1. **WCTE vs SK config**: L1-1 through L1-4 use `WCTE_geom_config.json` (regular cylinder, R=2m, H=4m, 2500 sensors) rather than `SK_geom_config.json` (superk type, requires ROOT connection table). This avoids ROOT dependency for most tests and is fast due to the small sensor count. Note: `SK_like_geom_config.json` has a trailing comma (invalid JSON) and should be fixed before use.

2. **Small Nphot and K**: L1-2/L1-3/L1-4 use Nphot=1000-5000 and K=3 for speed. This is enough to verify numerical equivalence without waiting for large simulations.

3. **Synthetic data for L1-6/7/8/9/10**: These scripts avoid the full simulator and test individual functions with synthetic tensors. This isolates the test to the specific function being validated.

4. **Deterministic seeds**: Every random operation uses a fixed `jax.random.PRNGKey` so results are reproducible across runs.

5. **Import shim via env var**: `LUCID_BASELINE=1` switches all imports from `lucid.*` to `tools.*`. The scripts share the same test logic for both codebases.

6. **Return type flexibility in L1-2**: The track mode simulator may return 2-tuple or 4-tuple depending on the internal `make_hits_fn` selection. The script handles both cases.
