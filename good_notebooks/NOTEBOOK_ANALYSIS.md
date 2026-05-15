# LUCiD Notebook Analysis — Complete Reference

This document provides a thorough analysis of all 18 notebooks in `good_notebooks/`,
organized by theme. It serves as the reference for building baseline comparison
scripts and regression tests.

---

## Overview

All notebooks use **SK geometry** (`config/SK_geom_config.json`, cylinder, ~11,146 sensors)
with `config/SK_physics_config.json` as the default physics parameters. The primary
entry point is `setup_event_simulator()`.

### Notebook Categories

| Category | Notebooks | Count |
|----------|-----------|-------|
| Track Optimization | tracking_opt_development, tracking_opt_development_likelihood, tracking_opt_with_gif | 3 |
| Parameter Scans | parameter_scans_1D, parameter_scans_1D_likelihood, grad_loss_and_opt_in_2D, grad_loss_and_opt_in_2D_likelihood | 4 |
| Calibration | detector_grad_qe_convergence_multi_source, grad_param_calibration_multi_init_no_qe, laser_source_grad_analysis | 3 |
| Visualization | visualize_3D_track_optimization, track_optimization_visualization, geometry_and_events_3D_visualization, cylinder_2D_displays, data_vs_pred_hit_predictions | 5 |
| Infrastructure | computational_performance_evaluation, train_siren, optimization_vs_variables | 3 |

### Common Configuration

```python
# All notebooks
default_json_filename = '../config/SK_geom_config.json'
PHYSICS_CONFIG = '../config/SK_physics_config.json'

# SK physics params (from SK_physics_config.json)
scatter_length = 50.0
wall_reflection_rate = 0.2
sensor_reflection_rate = 0.2
absorption_length = 50.0
qe = 0.065
qe_corrections = all 1.0
```

---

## 1. Track Optimization Notebooks

### Two Loss Paradigms

The tracking notebooks implement two fundamentally different loss approaches:

**A. Heuristic (counts-based):** `tracking_opt_development.ipynb`
```python
combined_loss = sqrt((vertex_loss + 1e-6) * (counts_loss + 1e-6) * (time_loss + 1e-6))
```
- `vertex_loss` = `origin_time_loss(position, detector_positions, observed_times, observed_counts, t0)`
- `counts_loss` = `counts_loss(observed_counts, simulated_counts)` — Poisson NLL on aggregated charges
- `time_loss` = `cone_time_loss(observed_counts, simulated_time, observed_times, t0)`
- Simulator returns: `(simulated_counts, simulated_time)` — 2-tuple
- No `stop_gradient` applied

**B. Likelihood-based:** `tracking_opt_development_likelihood.ipynb`, `tracking_opt_with_gif.ipynb`
```python
c, t, v, s = charge_loss, time_loss, vertex_loss, 0.
combined = (sqrt((c+s)*(t+s)*(v+s))
          + sqrt((c+s) * stop_gradient((t+s)*(v+s)))
          + sqrt((v+s) * stop_gradient((t+s)*(c+s))))
```
- `charge_loss` = `poisson_nll(observed_counts, total_charge)` — per-sensor Poisson NLL
- `time_loss` = `first_arrival_nll(log_w, flat_times, flat_indices, t_obs_shifted, TAU_TIME, NUM_DETECTORS)` — averaged over hit sensors
- `vertex_loss` = `origin_time_loss_configurable(stop_gradient(position), ..., t0, tau=tau_vtx)` — dynamic tau_vtx
- `tau_vtx = stop_gradient(TAU_VTX_PARAM_A * nrays + TAU_VTX_PARAM_B * energy + TAU_VTX_PARAM_C)`, clipped to [0.05, 0.95]
- Simulator returns: `(log_w, flat_times, flat_indices, total_charge)` — 4-tuple

### 1.1 tracking_opt_development.ipynb

**Purpose:** Full 5-stage pipeline for muon track reconstruction using heuristic loss.

**Stages:**
0. Energy estimation at origin (energy_scan_optimization)
1. Hierarchical position + t0 grid search
2. Hierarchical cone direction search
3. Energy scan at optimal position/direction
4. Adam refinement (250 iterations)

**Setup:**
```python
prediction_simulator = setup_event_simulator(
    default_json_filename, Nphot=50_000, TEMPERATURE=0.10,
    max_sensors_per_cell=4, K=K, is_data=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True)
data_simulator = setup_event_simulator(
    default_json_filename, Nphot=50_000, temperature=0.0,
    K=K, is_data=True, is_calibration=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True)
```

**Adam optimizer:** `optax.adam(lr=0.2, b1=0.9, b2=0.999)`, damping_factor=0.998, initial damping_w=5.0, energy frozen first 25 iters.

**Seeds:** `main_key=PRNGKey(42)`, `opt_key=PRNGKey(12345)`, N_EVENTS=20.

### 1.2 tracking_opt_development_likelihood.ipynb

**Purpose:** Same 5-stage pipeline with likelihood-based loss.

**Key differences from 1.1:**
- Loss: 3-term likelihood with stop_gradient
- Simulator returns 4-tuple (log_w, flat_times, flat_indices, total_charge)
- TAU_TIME=0.15
- Adam: 400 iterations, NO damping, direction-only first 25 iters
- POS_LR_SCALE × 2.0, DIR_LR_SCALE × 5.0
- Initial t0 from stage1 (not TRUE_T0)

### 1.3 tracking_opt_with_gif.ipynb

**Purpose:** Adam from smeared initial guess (no grid search), generates GIF animation.

**Key differences:**
- Skips stages 0-3, goes directly to Adam from smeared initial guess
- Smearing: position_sigma=5.0m, direction_sigma=155.0°, energy_sigma_frac=0.4
- Adam: 500 iterations, WITH damping (5.0, 0.998)
- All params updated from iteration 0 (no phase separation)
- N_EVENTS=1

---

## 2. Parameter Scan Notebooks

### 2.1 parameter_scans_1D.ipynb

**Purpose:** 1D parameter scans over all 7 track params using counts-based loss.

**Setup:** Nphot=150,000, K=7, temperature=0.10, entry_idx=2, PRNGKey(44).

**Scan configs:**
| Param | Range | Points |
|-------|-------|--------|
| X, Y, Z | ±0.5 m | 21 |
| t0 | ±5.0 ns | 21 |
| theta, phi | ±0.3 rad | 21 |
| Energy | ±100 MeV | 21 |

**Multi-event study:** 25 events, X-scan and t0-scan, computes gradient zero-crossing deltas.

### 2.2 parameter_scans_1D_likelihood.ipynb

**Purpose:** Same scans with likelihood-based loss + Nphot variation study.

**Loss:** 3-term likelihood (same as tracking_opt_development_likelihood).

**Nphot study:** Tests [150k, 300k, 600k] with simplified 2-term loss:
```python
combined = sqrt((charge_loss + 1e-6) * (tau * time_loss + 1e-6))
```

### 2.3 grad_loss_and_opt_in_2D.ipynb

**Purpose:** 2D loss landscapes with Adam optimization trajectories, counts-based loss.

**Setup:** Nphot=100,000, entry_idx=1, PRNGKey(45), N_SCAN_POINTS=21, t0 randomized in [-15, 15].

**Scan pairs:** (X,Z), (Y,phi), (phi,E), (X,t0), (t0,phi), (t0,E). Ranges: X/Y/Z ±2.0m, t0 ±4.0ns, theta/phi ±0.5 rad, E ±100 MeV.

**Optimization:** 3 trajectories from Gaussian-perturbed starts, damping_factor=0.998.

### 2.4 grad_loss_and_opt_in_2D_likelihood.ipynb

**Purpose:** Same 2D landscapes with likelihood-based loss.

**Key differences from 2.3:** Nphot=150,000, N_SCAN_POINTS=11, lr=0.1, no damping, direction-only first 25 iters, MAX_ITERATIONS=600.

---

## 3. Calibration Notebooks

### 3.1 detector_grad_qe_convergence_multi_source.ipynb

**Purpose:** Per-sensor QE correction optimization (11,146-dim) using 15 isotropic sources.

**Setup:** Nphot=8M (K=9) simulation, Nphot=15M (K=12) true data, `is_calibration=True`.

**True params (modified):** wall_reflection=0.1, sensor_reflection=0.3, qe_corrections=N(1.0, 0.1, seed=42).

**Loss:** `WC_loss(lambda_poisson=1.0, lambda_time=0.0)` — charge-only.

**Optimizer:** Adam (lr=0.005, b1=0.9, b2=0.99), 150 iterations, plateau LR reduction (patience=16, factor=0.7).

### 3.2 grad_param_calibration_multi_init_no_qe.ipynb

**Purpose:** 4-scalar parameter optimization (scatter, wall/sensor reflection, absorption) with QE frozen.

**Setup:** Nphot=500k (K=8) simulation, Nphot=15M (K=12) true data, single laser source at top.

**Loss:** `WC_smooth_loss(lambda_poisson=1.0, lambda_time=0.0, tau=1.0)`.

**Optimizer:** `optax.multi_transform` — Adam (lr=0.05, b1=0.95, b2=0.99) for trainable, set_to_zero for frozen. Warmup schedule (40% warmup, 500 iterations, 4 random initial guesses). Params optimized in normalized [0,1] space.

### 3.3 laser_source_grad_analysis.ipynb

**Purpose:** 1D and 2D parameter sweep analysis of calibration loss landscape (no optimization).

**Loss:** `WC_smooth_loss(tau=0.5, lambda_poisson=1.0, lambda_time=0.0)`.

**Sweeps:**
- 1D: 4 params × 41 points each
- 2D: 4 pairs (Scatter×Absorption @31pts, Wall×Sensor @41pts, Scatter×Wall @41pts, Scatter×Sensor @41pts)

---

## 4. Visualization Notebooks

### 4.1 visualize_3D_track_optimization.ipynb

**Purpose:** Full Adam optimization on JUNO sphere detector with 3D Plotly visualization.

**Setup:** JUNO config, Nphot=150,000, K=7, temperature=0.10, detector_type=Sphere. Likelihood loss.

**Optimization:** 5 trials from random initial guesses (angle_sigma=15°), 1000 max iterations.

### 4.2 track_optimization_visualization.ipynb

**Purpose:** Post-hoc analysis of pre-computed results from pickle files. No simulation.

**Analysis:** Convergence plots, bootstrap CIs, longitudinal/transverse decomposition, correlation studies, 3D event visualization.

### 4.3 geometry_and_events_3D_visualization.ipynb

**Purpose:** Generate and visualize prediction + data events on SK detector. 3D Plotly disc plots.

**Setup:** Nphot=1M, K=6, temperature=0.0, track at (0,0,0) with theta=π/2, phi=π/6, energy=1050.

### 4.4 cylinder_2D_displays.ipynb

**Purpose:** 2D "unrolled cylinder" displays comparing prediction vs data events.

**Setup:** Nphot=1M, prediction at K=7/temp=0.1, data at K=20/temp=0.0, true_position=[-10,0,0], true_direction=[1,0,0].

### 4.5 data_vs_pred_hit_predictions.ipynb

**Purpose:** Statistical comparison of prediction vs data events. Nrays dependency study.

**Nrays study:** [10k, 25k, 50k, 150k, 250k, 500k] × 30 events each. Metrics: covariance, time residual, charge residual.

---

## 5. Infrastructure Notebooks

### 5.1 computational_performance_evaluation.ipynb

**Purpose:** Wall-clock benchmarks for simulation and gradient computation.

**Parameters:** N = [10k–2M], K = [1–8], 10 warmup + 100 timed runs per (N,K) pair. Both track and calibration modes.

### 5.2 train_siren.ipynb

**Purpose:** Train SIREN neural network for photon density interpolation. Invokes CLI scripts.

**Training:** 50k steps, patience=40, batch_size=65536, lr=1e-4, material=water, particle=electron/muon.

### 5.3 optimization_vs_variables.ipynb

**Purpose:** Analyze reconstruction performance vs Nrays, energy, number of sensors from pre-computed pickle files. Bootstrap CIs, exponential/constant fits.

---

## Key Values for Baseline Scripts

### Level 1: Forward Pass Matching

For each notebook category, the baseline script should verify:

1. **Propagator output:** Fixed rays from center → sensor_weights, positions, normals match
2. **Simulator output:** Fixed ParticleParams + PRNGKey → charges, times match
3. **Loss values:** At true parameters with fixed seed → exact loss value
4. **Gradient values:** At true parameters with fixed seed → exact gradient vector

### Level 2: Optimization Matching

Longer runs that verify convergence behavior:

1. **Grid search:** Position/direction/energy stages produce same candidates
2. **Adam convergence:** Loss curve over iterations matches
3. **Final parameters:** Converged position/direction/energy match within tolerance
4. **Calibration recovery:** Parameter recovery accuracy matches

### Common Seeds Across Notebooks

| Variable | Value | Used in |
|----------|-------|---------|
| main_key | PRNGKey(42) | tracking_opt, param_scans multi-event |
| opt_key | PRNGKey(12345) | Adam optimizer |
| data_key | PRNGKey(44) | 1D param scans single event |
| data_key | PRNGKey(45) | 2D param scans |
| scan_key | PRNGKey(42) | Energy/direction scans |
| smearing_seed | event_idx + 1234 | GIF notebook |
| qe_seed | np.random.seed(42) | QE convergence |
