# Comprehensive Notebook Documentation

This document provides detailed documentation of all 18 notebooks in `good_notebooks/`.

---

## 1. tracking_opt_development.ipynb

### A. Big Picture
Replicates the full 5-stage S3DF optimization pipeline for muon track reconstruction in a water Cherenkov detector: energy scan, position grid search, direction cone search, energy refinement, and Adam gradient descent.

### B. Setup
**Imports:**
```python
import sys; sys.path.append('..')
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import json, os, time, pickle, glob, uproot
from pathlib import Path
from jax import jit, value_and_grad
import optax, subprocess

from lucid.geometry import generate_detector
from lucid.generate import read_photon_data_from_photonsim
from lucid.simulation import setup_event_simulator
from lucid.utils import load_range_params, check_track_endpoint_in_detector
from lucid.detector_params import ParticleParams, load_detector_params

from lucid.optimization.grid_search import (
    load_optimization_config, get_detector_bounds, hierarchical_position_grid_search
)
from lucid.optimization.utils.functions import (
    hierarchical_direction_search_cone, energy_scan_optimization,
    cartesian_to_spherical, spherical_to_cartesian, performance_summary,
    estimate_muon_energy_from_photon_count
)
from lucid.losses import origin_time_loss, counts_loss, cone_time_loss
from lucid.optimization.run import load_config
```

**Config files:** `../config/SK_geom_config.json` (detector geometry), `../config/SK_physics_config.json` (physics parameters), `../s3df_jobs/nrays_config/opt_config_5.json` (optimization config)

**Simulator setup:**
```python
prediction_simulator = setup_event_simulator(
    default_json_filename, Nphot=50_000, TEMPERATURE=0.10,
    max_sensors_per_cell=4, K=config['basic_config']['k'], is_data=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True
)
data_simulator = setup_event_simulator(
    default_json_filename, Nphot=50_000, temperature=0.0,
    K=config['basic_config']['k'], is_data=True, is_calibration=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True
)
```

**Key parameters:** CONFIG_INDEX=5, TEMPERATURE=0.10, N_EVENTS=20, Nphot=50_000, data_dir='../data/water/muon/'

### C. Step-by-step Cell Description
- **Cell 0 (md):** Title and pipeline description
- **Cell 2 (code):** Import all libraries and LUCiD modules
- **Cell 4 (code):** Load nrays_config optimization config (CONFIG_INDEX=5), set Adam defaults (lr=0.2, b1=0.9, b2=0.999)
- **Cell 6-7 (code):** Detector setup: generate_detector, detector_points, detector_bounds, setup prediction and data simulators, load detector_params
- **Cell 8 (code):** Print detector_params
- **Cell 10 (code):** Define `generate_event_data()` -- loads ROOT file, pads photon data to 1M, generates random position/direction/t0, applies rotation+translation, runs data_simulator
- **Cell 12 (code):** Define `create_combined_loss_function()` -- combined_product_loss = sqrt(vertex_loss * counts_loss * time_loss), creates value_and_grad function
- **Cell 14 (code):** Stage 0 function: energy_scan_optimization at origin position [0,0,0] with direction [1/sqrt(3),1/sqrt(3),1/sqrt(3)], energy_guess=1000+-50, delta=700, n_steps=10
- **Cell 16 (code):** Stage 1 function: hierarchical_position_grid_search using config params
- **Cell 18 (code):** Stage 2 function: hierarchical_direction_search_cone using config params
- **Cell 20 (code):** Stage 3 function: energy_scan_optimization at optimal position/direction
- **Cell 22 (code):** Stage 4 function: Adam optimizer refinement, MAX_ITERATIONS=250, energy frozen for first 25 iterations, damping_factor=0.998, parameter clipping to detector bounds
- **Cell 24 (code):** Main processing loop over N_EVENTS=20, full 5-stage pipeline, main_key=PRNGKey(42)
- **Cell 26 (code):** Performance summary statistics
- **Cell 28 (code):** 3D event visualization using create_event_3D_visualization
- **Cell 30 (code):** Visualize all stages for a single event

### D. Key Computations
- **Forward pass:** `prediction_simulator(track, key)` returns `(simulated_counts, simulated_time)`
- **Data generation:** `data_simulator(true_track, key, photon_data)` returns `(hit_counts, hit_times_raw)`
- **Loss function:** `combined_product_loss = sqrt((vertex_loss+1e-6) * (counts_loss+1e-6) * (time_loss+1e-6))`
- **Gradient:** `combined_grad_fn = jit(value_and_grad(combined_product_loss, has_aux=True))`
- **Optimization:** optax.adam(lr=0.2), update_scales per param type, damping_factor=0.998 decaying current_damping_w starting at 5.0
- **Params:** [x, y, z, t0, theta, phi, energy]

### E. Baseline Outputs
- Per-event: final_position_error (m), final_direction_error (deg), final_t0_error, final_energy_error (MeV)
- Aggregate: mean/median/std of all error metrics via performance_summary()
- Timing: mean total event time, mean Adam time, Adam % of total

---

## 2. tracking_opt_development_likelihood.ipynb

### A. Big Picture
Same 5-stage pipeline as notebook 1 but uses **likelihood-based losses** (Poisson NLL for charges, first-arrival NLL for times) with a dynamic tau_vtx parameter instead of aggregated losses.

### B. Setup
Same as notebook 1, plus:
```python
from jax.scipy.special import gammaln
from lucid.losses import (
    first_arrival_nll, origin_time_loss, get_optimal_tau_vtx,
    TAU_VTX_PARAM_A, TAU_VTX_PARAM_B, TAU_VTX_PARAM_C,
)
```

**Key differences:**
- TEMPERATURE = 0.05 (then overridden to 0.1)
- TAU = 0.15 (time constant for first-arrival NLL)
- MAX_ITERATIONS = 400 (Stage 4)
- data_dir points to S3DF path (fallback: '../data/water/muon/')

**Simulator setup:** Same as notebook 1.

### C. Step-by-step Cell Description
- **Cell 2 (code):** Imports with additional gammaln and likelihood loss imports
- **Cell 4 (code):** Same config loading (CONFIG_INDEX=5)
- **Cell 6 (code):** Detector/simulator setup (same except TEMPERATURE=0.1)
- **Cell 8 (code):** Event generation function (identical to notebook 1)
- **Cell 10 (code):** Define likelihood losses: poisson_nll, energy_loss (log ratio), smooth_pinball, origin_time_loss_configurable, and create_combined_loss_function with 3-term combined loss: `sqrt(c*t*v) + sqrt(c*sg(t*v)) + sqrt(v*sg(t*c))`
- **Cell 12 (code):** Stage 0: energy_loss (log ratio of total counts) for initial energy guess
- **Cell 14 (code):** Stage 1: Same hierarchical_position_grid_search (origin_time_loss)
- **Cell 19 (code):** Stage 2: Custom direction search using poisson_nll for loss evaluation, with Rodrigues rotation for cone sampling
- **Cell 23 (code):** Stage 3: energy_loss for energy scan
- **Cell 25 (code):** Stage 4: Adam optimizer, MAX_ITERATIONS=400, Phase 1 (first 25 iters): direction only, then all params; POS_LR_SCALE*2, DIR_LR_SCALE*5
- **Cell 27 (code):** Main processing loop (N_EVENTS=20)
- **Cell 29 (code):** Performance summary

### D. Key Computations
- **Forward pass:** `prediction_simulator(track, key)` returns `(log_w, flat_times, flat_indices, total_charge)`
- **Loss:** 3-term combined: `sqrt(c*t*v) + sqrt(c*sg(t*v)) + sqrt(v*sg(t*c))` where c=poisson_nll, t=first_arrival_nll (masked), v=origin_time_loss_configurable with dynamic tau_vtx
- **Dynamic tau_vtx:** `TAU_VTX_PARAM_A * nrays + TAU_VTX_PARAM_B * energy + TAU_VTX_PARAM_C` (clipped to [0.05, 0.95])
- **Time treatment:** observed_times shifted by t0, track simulated with t0=0

### E. Baseline Outputs
- Same metrics as notebook 1 plus charge_losses, time_losses, tau_vtx_values tracked per iteration

---

## 3. tracking_opt_with_gif.ipynb

### A. Big Picture
Runs Adam optimization from a randomly smeared initial guess (skipping grid search stages 0-3) and generates animated GIF showing the optimization convergence.

### B. Setup
Same as notebook 2 (likelihood-based), plus:
```python
from PIL import Image
import shutil, matplotlib
from lucid.losses import (first_arrival_nll, poisson_nll, origin_time_loss_configurable,
    TAU_VTX_PARAM_A, TAU_VTX_PARAM_B, TAU_VTX_PARAM_C)
from lucid.optimization.utils.visualization import (
    create_event_3D_visualization, create_optimization_path_3d_visualization)
from lucid.optimization.utils.geometry import create_cylinder_surface, create_sphere_surface
from lucid.visualization import create_detector_comparison_display
```

**Key parameters:** N_EVENTS=1, POSITION_SIGMA=5.0m, DIRECTION_SIGMA=155.0deg, ENERGY_SIGMA_FRAC=0.4, T0_SIGMA=0.0, MAX_ITERATIONS=500, TAU_TIME=0.15

### C. Step-by-step Cell Description
- **Cell 2 (code):** All imports
- **Cell 4 (code):** Load config (CONFIG_INDEX=5)
- **Cell 6 (code):** Detector/simulator setup (same as notebooks 1-2)
- **Cell 8 (code):** Event generation function
- **Cell 10 (code):** Combined loss function (3-term likelihood-based, same as notebook 2)
- **Cell 12 (code):** Define `generate_smeared_initial_params()` (Gaussian position smearing, angular direction smearing via Rodrigues rotation, relative energy smearing) and `run_adam_optimization()` with damping
- **Cell 14 (code):** Run single event from smeared initial guess
- **Cell 16 (code):** GIF generation functions: `process_simulator_output_for_visualization()`, `create_optimization_frames()`, `create_gif()`, `clean_optimization_frames()`
- **Cell 18 (code):** Generate GIF using create_detector_comparison_display, iteration_step=10, total_duration=4.8s
- **Cell 20 (code):** Display convergence plots (loss, position/direction/energy errors)

### D. Key Computations
- **Loss:** Same 3-term likelihood as notebook 2
- **Smearing:** Position ~N(0, sigma_m), Direction via Rodrigues rotation ~Rayleigh(sigma_rad), Energy ~N(0, E*frac)
- **Optimizer:** Adam with damping (current_damping_w *= 0.998, starting at 5.0)

### E. Baseline Outputs
- GIF file at `figures/optimization_event_0_charge.gif`
- Convergence PNG at `figures/optimization_convergence.png`
- Final errors: position_error, direction_error, energy_error, t0_error

---

## 4. track_optimization_visualization.ipynb

### A. Big Picture
Post-hoc analysis and publication-quality visualization of pre-computed optimization results loaded from pickle files. Creates convergence plots with histograms, residual analysis, and correlation studies.

### B. Setup
```python
import pickle, sys; sys.path.append('..')
import jax, jax.numpy as jnp, numpy as np
from matplotlib import pyplot as plt
from lucid.geometry import generate_detector
from lucid.generate import read_photon_data_from_photonsim
from lucid.optimization.grid_search import load_optimization_config
import pandas as pd, seaborn as sns
from matplotlib.ticker import FormatStrFormatter
```

**Data loaded:** `/sdf/data/neutrino/cjesus/fixed_LUCiD/nrays/config_3.pkl`

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports
- **Cell 1 (code):** Load results pickle, extract histories (position/direction/t0/energy errors and differences), compute statistics (mean/median/68th/90th percentile), convert energy errors to momentum errors
- **Cell 2 (code):** Bootstrap confidence intervals for 68th percentile metrics, build summary table
- **Cell 3 (code):** Create multi-panel convergence figure with histograms (position, direction, t0, momentum)
- **Cell 4 (code):** Longitudinal/transverse position decomposition using true direction as axis
- **Cell 5 (code):** t0 and energy residual histograms + correlation plots (delta_t0 vs delta_r_parallel, delta_E vs delta_r_parallel)
- **Cell 6-9 (code):** Scatter plots of true quantities vs reconstruction errors with linear fits
- **Cell 10 (code):** Alternative convergence plot style with individual event traces
- **Cell 11 (code):** 3D optimization path visualization using create_optimization_path_3d_visualization
- **Cell 12+ (code):** Simplified 3D event visualization with Cherenkov cone

### D. Key Computations
- Bootstrap percentile CI (n_bootstrap=10000, ci_level=90)
- Momentum conversion: E_total = T_mu + m_mu, p_mu = sqrt(E_total^2 - m_mu^2), conversion_factor = E_total/p_mu

### E. Baseline Outputs
- Summary table with 68% CI for distance error (cm), angle error (deg), t0 error (ns), momentum error (%)
- Longitudinal/transverse position resolution values
- Convergence PDF at `figures/convergence_all_metrics.pdf`

---

## 5. visualize_3D_track_optimization.ipynb

### A. Big Picture
2D parameter scan landscapes with optimization trajectory overlays for the JUNO spherical detector using likelihood-based loss.

### B. Setup
```python
from lucid.geometry import generate_detector
from lucid.generate import read_photon_data_from_photonsim
from lucid.optimization.grid_search import load_optimization_config
from lucid.detector_params import ParticleParams, load_detector_params
from lucid.losses import (first_arrival_nll, poisson_nll, origin_time_loss_configurable,
    TAU_VTX_PARAM_A, TAU_VTX_PARAM_B, TAU_VTX_PARAM_C)
from lucid.simulation import setup_event_simulator
from lucid.optimization.run import load_config
from lucid.optimization.grid_search import get_detector_bounds
```

**Config:** `../config/JUNO_geom_config.json`, `../config/JUNO_physics_config.json`
**Key params:** TEMPERATURE=0.10, K=7, Nphot=150_000, TAU_TIME=0.15

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports including likelihood losses
- **Cell 1 (code):** Configuration and simulator setup for JUNO detector (Sphere type)
- **Cell 2 (code):** Define likelihood-based combined_product_loss (3-term formula, same as notebook 2)
- **Cell 3 (code):** Load ROOT photon data, generate random track, create data event
- **Cell 4+ (code):** Adam optimization from initial guess, 2D parameter scan landscapes, overlay trajectory

### D. Key Computations
- Same 3-term likelihood loss as notebooks 2-3
- 2D parameter scans with contour plots
- Optimization trajectory overlaid on loss landscapes

---

## 6. optimization_vs_variables.ipynb

### A. Big Picture
Plots tracking performance metrics (position, direction, t0, momentum errors) as a function of different variables: number of rays (nrays), particle energy, and number of sensors. Fits exponential/constant curves to the trends.

### B. Setup
```python
import glob, re, numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pickle, sys; sys.path.append('..')
import jax, jax.numpy as jnp, torch
from lucid.geometry import generate_detector
```

**Data files:** Pre-computed pickle files from `/sdf/data/neutrino/cjesus/fixed_LUCiD/` directories (nrays/, energy_50k/, energy_150k_test/, geom_50k/)

### C. Step-by-step Cell Description
- **Cell 0 (code):** Parse S3DF log files to extract per-config 68th percentile position errors with bootstrap CIs, fit exponential curve
- **Cell 1 (code):** Utility functions: extract_histories, compute_statistics, bootstrap_percentile_ci, compute_momentum_errors
- **Cell 2 (code):** Reusable data loading: load_and_process_results, extract_timing_metrics, compute_bootstrap_metrics
- **Cell 3 (code):** Generic plotting function: plot_metrics_vs_variable with exponential or constant fits and asymmetric error bars
- **Cell 4 (code):** Performance vs nrays (configs 0-5, muon_mass=0.105658 GeV)
- **Cell 5 (code):** Performance vs energy (configs 0,3,6 from 50k; 9,12,14 from 150k)
- **Cell 6 (code):** Performance vs number of sensors (configs 0-9 from geom_50k)

### D. Key Computations
- Bootstrap percentile CI with n_bootstrap=1000-2000, ci_level=68-90
- Exponential fit: y = a * b^x + c
- Momentum error conversion: (E_total/p_mu) * (delta_E/E_total) * 100

### E. Baseline Outputs
- PDF figures: `figures/tracking_performance_vs_nrays.pdf`, `figures/tracking_performance_vs_energy.pdf`, `figures/detector_perf_vs_num_sensors.pdf`

---

## 7. detector_grad_qe_convergence_multi_source.ipynb

### A. Big Picture
Optimizes per-sensor quantum efficiency (QE) corrections using gradient descent with multiple isotropic source positions to calibrate the detector.

### B. Setup
```python
from lucid.geometry import generate_detector
from lucid.losses import WC_loss
from lucid.simulation import setup_event_simulator
from lucid.detector_params import (DetectorParams, isotropic_source,
    save_detector_params, load_detector_params)
import jax, jax.numpy as jnp
from jax import value_and_grad, jit
import optax
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Key params:** Nphot=8_000_000, K=9 (prediction), Nphot_True=15_000_000, K_True=12 (data), true_qe=0.2, true_qe_corrections=N(1.0, 0.1)

**Simulator setup:**
```python
simulate_event = setup_event_simulator(..., Nphot=8M, K=9, is_calibration=True)
simulate_data = setup_event_simulator(..., Nphot_True=15M, K_True=12, is_calibration=True, default_detector_params=TRUE_PARAMS)
```

### C. Step-by-step Cell Description
- **Cell 1 (code):** Imports
- **Cell 2 (code):** Detector setup (SK_geom_config.json)
- **Cell 3 (code):** Create synthetic true QE corrections: N(1.0, 0.1) per sensor, TRUE_PARAMS with wall_reflection=0.1, sensor_reflection=0.3
- **Cell 4 (code):** Setup calibration simulators
- **Cell 5 (code):** Create 15 isotropic sources at xy_positions x z_positions, intensity=100M
- **Cell 6 (code):** Generate true data for all 15 sources
- **Cell 7 (code):** Define loss_and_grad_fn using WC_loss (lambda_poisson=1.0, lambda_time=0.0)
- **Cell 8 (code):** Adam optimization loop: 150 steps, lr=0.005 with patience-based reduction (patience=16, factor=0.7, min_lr=1e-5), randomly selects source each step
- **Cell 9 (code):** Results visualization: loss curve, error distribution histogram, true vs final 2D histogram
- **Cell 10 (code):** Evaluate final model on all source locations

### D. Key Computations
- **Loss:** `WC_loss(detector_points, *true_data, *simulated_data, lambda_poisson=1.0, lambda_time=0.0)`
- **Gradient:** `value_and_grad(loss_fn)(qe_corrections)`
- **Optimizer:** optax.inject_hyperparams(optax.adam)(lr=0.005, b1=0.9, b2=0.99) with manual LR reduction

### E. Baseline Outputs
- Mean error, std error, RMSE, max absolute error of QE corrections
- Loss convergence curve
- Per-source final losses

---

## 8. grad_param_calibration_multi_init_no_qe.ipynb

### A. Big Picture
Optimizes 4 scalar detector parameters (scatter_length, wall_reflection_rate, sensor_reflection_rate, absorption_length) while keeping QE frozen, using multiple initial guesses.

### B. Setup
```python
from lucid.geometry import generate_detector
from lucid.losses import WC_smooth_loss
from lucid.simulation import setup_event_simulator
from lucid.detector_params import (DetectorParams, laser_source,
    normalize_params, denormalize_params, default_bounds,
    make_optimization_mask, load_detector_params)
import jax, jax.numpy as jnp
from jax import value_and_grad, jit
import optax
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Key params:** Nphot=500_000, K=8, Nphot_True=15M, K_True=12, source=laser_source at z=H/2-0.1, ADAM_LR=0.05, ADAM_ITERATIONS=500, WARMUP_FRACTION=0.4, N_INIT_GUESSES=4, INIT_FRAC_DIFF=0.5

### C. Step-by-step Cell Description
- **Cell 1 (code):** Imports
- **Cell 2 (code):** Config, detector setup, set bounds (scatter>=5, wall_refl>=0.05, sensor_refl>=0.05, absorption>=10)
- **Cell 3 (code):** Setup calibration simulators
- **Cell 4 (code):** Generate true data, create 4 initial guesses by uniform perturbation of true params
- **Cell 5 (code):** Define loss function using WC_smooth_loss (lambda_poisson=1.0, lambda_time=0.0, tau=1.0), step_fn with normalize/denormalize
- **Cell 6 (code):** run_single_optimization: normalize params to [0,1], warmup_constant_schedule, optax.multi_transform (Adam for trainable, set_to_zero for frozen), clip to [0.01, 0.99]
- **Cell 8 (code):** Run all 4 optimization runs
- **Cell 9 (code):** Summary table: per-guess final loss, MSE, runtime; best guess parameter comparison
- **Cell 10-11 (code):** Visualization: loss evolution (log scale), parameter convergence subplots

### D. Key Computations
- **Loss:** `WC_smooth_loss(..., lambda_poisson=1.0, lambda_time=0.0, tau=1.0)`
- **Normalization:** params mapped to [0,1] via bounds_min/bounds_max
- **Optimizer:** optax.multi_transform with Adam (lr=0.05 with warmup, b1=0.95, b2=0.99)

### E. Baseline Outputs
- Per-parameter: true vs final values and absolute errors
- MSE for each initial guess
- Loss convergence curves
- PDF: `figures/multi_init_no_qe_loss_evolution_v2.png`

---

## 9. laser_source_grad_analysis.ipynb

### A. Big Picture
1D and 2D parameter sweeps for calibration parameters (scatter_length, wall_reflection, sensor_reflection, absorption_length) using a laser source, visualizing loss landscapes and gradient fields.

### B. Setup
```python
from lucid.geometry import generate_detector
from lucid.losses import WC_smooth_loss
from lucid.simulation import setup_event_simulator
from lucid.detector_params import laser_source, load_detector_params
from lucid.gradient_analysis import (SweepParam, sweep_1d, sweep_2d, plot_sweep_1d, plot_sweep_2d)
import jax, jax.numpy as jnp
from jax import jit, value_and_grad
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Key params:** Nphot=500_000, K=8, Nphot_True=15M, K_True=12, source at z=H/2-0.1, intensity=100M

### C. Step-by-step Cell Description
- **Cell 1 (code):** Imports
- **Cell 2 (code):** Detector setup, simulators, generate true data
- **Cell 3 (code):** Define SweepParam configurations: scatter_length (hw=20m), wall_reflection (hw=0.15), sensor_reflection (hw=0.08), absorption_length (hw=80m)
- **Cell 4 (code):** loss_and_grad_fn using WC_smooth_loss(lambda_poisson=1.0, lambda_time=0.0, tau=0.5)
- **Cell 5 (code):** Run 1D sweeps
- **Cell 6 (code):** Plot 1D sweep results
- **Cell 7 (code):** Run 2D sweeps for 4 parameter pairs (31-41 points each)
- **Cell 8 (code):** Plot 2D sweep surfaces

### D. Key Computations
- **Loss:** `WC_smooth_loss(..., lambda_poisson=1.0, lambda_time=0.0, tau=0.5)`
- **Sweeps:** `sweep_1d(loss_and_grad_fn, TRUE_PARAMS, calib_sweeps)`, `sweep_2d(loss_and_grad_fn, TRUE_PARAMS, px, py, num_points=n)`

### E. Baseline Outputs
- `figures/laser_1d_sweeps_v3.png`, `figures/laser_2d_surfaces_v3.png`

---

## 10. parameter_scans_1D.ipynb

### A. Big Picture
1D loss and gradient scans over all 7 track parameters (x, y, z, t0, theta, phi, energy) for single events, plus multi-event analysis of position (X) and t0 gradient zero-crossing accuracy.

### B. Setup
```python
from lucid.geometry import generate_detector
from lucid.utils import load_single_event, save_single_event, generate_random_params, print_particle_params
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.utils import spherical_to_cartesian
from lucid.detector_params import ParticleParams, load_detector_params
from lucid.losses import counts_loss, origin_time_loss, cone_time_loss
import jax, jax.numpy as jnp, numpy as np, time
from jax import jit, grad, vmap, value_and_grad
import uproot
from scipy.interpolate import interp1d
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Data file:** `../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root`
**Key params:** TEMPERATURE=0.10, N_SCAN_POINTS=21, K=7, Nphot=150_000

### C. Step-by-step Cell Description
- **Cell 1 (code):** Imports
- **Cell 3 (code):** Setup detector, data_simulator (K=20, is_data=True), prediction_simulator (K=7, temperature=0.1)
- **Cell 6 (code):** Define combined_product_loss: sqrt(vertex_loss * counts_loss * time_loss), time_loss uses tau=0.23
- **Cell 8 (code):** Generic perform_parameter_scan function
- **Cell 10 (code):** Generate single data-like event: load ROOT entry_idx=2, random position (fraction=0.6), create rotated+translated photon data
- **Cell 12 (code):** Perform all 7 parameter scans with ranges: X/Y/Z=+-0.5m, t0=+-5ns, theta/phi=+-0.3rad, E=+-100MeV
- **Cell 14 (code):** Visualization of all scans (loss and gradient subplots)
- **Cell 16 (code):** Summary statistics table
- **Cell 18-24 (code):** Multi-event X-position analysis (N_EVENTS=25): gradient zero-crossing delta_X statistics
- **Cell 25-30 (code):** Multi-event t0 analysis: gradient zero-crossing delta_t0 statistics

### D. Key Computations
- **Loss:** `sqrt((vertex_loss+1e-6) * (counts_loss+1e-6) * (time_loss+1e-6))`
- **Gradient:** `value_and_grad(combined_product_loss)(params)`
- **Zero-crossing:** Linear interpolation between sign-changing gradient values

### E. Baseline Outputs
- Per-parameter: min loss position, delta from true, zero-crossing detection
- Multi-event: mean/sigma of delta_X, 68th percentile of |delta_X|, expected 3D resolution = p68 * sqrt(3)
- Same for delta_t0

---

## 11. parameter_scans_1D_likelihood.ipynb

### A. Big Picture
Same as notebook 10 but uses likelihood-based loss (first_arrival_nll + poisson_nll + vertex_loss) with dynamic tau_vtx, plus Nphot variation study.

### B. Setup
Same as notebook 10 plus:
```python
from lucid.losses import counts_loss, first_arrival_nll, segment_logsumexp
from lucid.losses import (first_arrival_nll, poisson_nll, origin_time_loss_configurable,
    TAU_VTX_PARAM_A, TAU_VTX_PARAM_B, TAU_VTX_PARAM_C)
from jax.scipy.special import gammaln
```

**Key params:** TAU_TIME=0.15, NRAYS_FLOAT=150_000.0

### C. Step-by-step Cell Description
- **Cell 1 (code):** Imports
- **Cell 3 (code):** Setup (same config files, same Nphot=150k, K=7)
- **Cell 5 (code):** Define likelihood_loss with 3-term combined formula and dynamic tau_vtx
- **Cell 7 (code):** Generate data event (same as notebook 10)
- **Cell 9 (code):** perform_parameter_scan function adapted for likelihood loss
- **Cell 11 (code):** Run 7 parameter scans
- **Cell 13 (code):** Visualize loss and gradient for each parameter
- **Cell 15 (code):** Summary statistics
- **Cell 17 (code):** Nphot variation study: [150k, 300k, 600k] with make_likelihood_loss factory, SCAN_TAU=0.1
- **Cell 18 (code):** Multi-panel comparison plots across Nphot values

### D. Key Computations
- **Loss:** 3-term: `sqrt(c*t*v) + sqrt(c*sg(t*v)) + sqrt(v*sg(t*c))` with dynamic tau_vtx
- **Forward pass:** returns `(log_w, flat_times, flat_indices, total_charge)`

---

## 12. grad_loss_and_opt_in_2D.ipynb

### A. Big Picture
2D parameter scan landscapes with optimization trajectory overlays using the original aggregated loss (counts_loss + cone_time_loss + origin_time_loss). Shows loss surfaces for parameter pairs.

### B. Setup
```python
from lucid.geometry import generate_detector
from lucid.generate import read_photon_data_from_photonsim
from lucid.optimization.grid_search import load_optimization_config
from lucid.detector_params import ParticleParams, load_detector_params
from lucid.simulation import setup_event_simulator
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Key params:** TEMPERATURE=0.10, N_SCAN_POINTS=21, K=7, Nphot=150_000

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports
- **Cell 1 (code):** Setup configuration, detector, simulators
- **Remaining cells:** 2D parameter scans for selected pairs, contour plot visualization, overlay optimization trajectories from Adam runs

### D. Key Computations
- **Loss:** sqrt(vertex_loss * counts_loss * time_loss) (aggregated)
- **2D scans:** Grid evaluation of loss for pairs of parameters

---

## 13. grad_loss_and_opt_in_2D_likelihood.ipynb

### A. Big Picture
Same as notebook 12 but uses likelihood-based loss (Poisson NLL + first-arrival NLL + vertex loss with dynamic tau_vtx).

### B. Setup
Same as notebook 12 plus likelihood loss imports. Same 3-term formula as notebook 2.

### C. Step-by-step Cell Description
Similar structure to notebook 12 but with likelihood loss.

### D. Key Computations
- **Loss:** 3-term likelihood combined formula
- **2D scans:** Grid evaluation for parameter pairs

---

## 14. computational_performance_evaluation.ipynb

### A. Big Picture
Benchmarks simulation and gradient computation wall-clock times as a function of the number of rays (N) and scattering iterations (K), comparing full simulation vs calibration-only mode.

### B. Setup
```python
import sys; sys.path.append('..')
import time, torch, jax
from jax import jit, value_and_grad
import jax.numpy as jnp, numpy as np
from pathlib import Path
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.utils import generate_random_point_inside_cylinder, generate_random_params
from lucid.detector_params import ParticleParams, isotropic_source
from lucid.losses import compute_simplified_loss
import matplotlib.pyplot as plt
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Key params:** K_VALUES=[1,2,...,8], N_VALUES=[10k, 100k, 250k, 500k, 750k, 1M, 1.5M, 2M], WARMUP_RUNS=10, TIMING_RUNS=100, CALC_CALIB_PERF=True

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports
- **Cell 1 (code):** Flag to control calibration performance calculation
- **Cell 2 (code):** benchmark_simulation function: for each K and N, setup simulator, warmup, then time 100 runs. Uses fixed track (energy=800, position=[0,0,0], theta=pi/3, phi=pi/4)
- **Cell 3 (code):** Save results to `output/time_benchmark_results.pkl`
- **Cell 4 (code):** Load results
- **Cell 5-6 (code):** Plot results: simulation time and gradient time vs N for each K
- **Cell 7 (code):** Plot calibration results if available

### D. Key Computations
- **Forward pass timing:** `simulate_event(track, key)` repeated 100 times
- **Gradient timing:** `value_and_grad(loss_fn)(track, key)` repeated 100 times
- **Track used:** energy=800, position=[0,0,0], theta=pi/3, phi=pi/4

### E. Baseline Outputs
- Timing plots: `figures/simulation_timing.png`, `figures/gradient_timing.pdf`
- Mean and std timing per (K, N) combination

---

## 15. train_siren.ipynb

### A. Big Picture
Trains and validates the Physics SIREN neural network model for photon density interpolation using PhotonSim lookup tables.

### B. Setup
Uses `%run` magic commands to invoke CLI tools directly.

### C. Step-by-step Cell Description
- **Cell 0 (code):** `!nvidia-smi` -- check GPU availability
- **Cell 1 (code):** Train SIREN: `%run ../tools/siren/train.py --material water --particle electron --patience 40 --num-steps 50000 --resume`
- **Cell 2 (code):** Validate cutoff thresholds: `%run ../tools/siren/validate.py cutoff --material water --particle muon --thresholds 0.1,1.,2.,4. --energy 1050`
- **Cell 3 (code):** Validate valid-points count: `%run ../tools/siren/validate.py valid-points --material water --particle muon --thresholds 2.0`
- **Cell 4 (code):** Validate integral: `%run ../tools/siren/validate.py integral --material water --particle muon --nphot 1000000`
- **Cell 5 (code):** Validate ray visualization: `%run ../tools/siren/validate.py rays --material water --particle muon`
- **Cell 6 (code):** Energy visualization (electron): `%run ../tools/siren/validate.py energy --material water --particle electron --energies 500,1000,1500 --threshold 0.01 --vmax 200 --output "figures/"`
- **Cell 7 (code):** Energy visualization (muon): same but `--particle muon`

### D. Key Computations
- SIREN training: --batch-size default, --num-steps 50000, --learning-rate default, --patience 40
- Validation: integral accuracy, threshold cutoff analysis, ray pattern visualization

### E. Baseline Outputs
- Trained model checkpoint in `data/{material}/{particle}/siren_training/trained_model/`
- Validation figures in `figures/`

---

## 16. geometry_and_events_3D_visualization.ipynb

### A. Big Picture
Generates interactive 3D visualizations of detector geometries and simulated events using Plotly disc-based rendering.

### B. Setup
```python
import sys; sys.path.append('..')
from pathlib import Path
from lucid.geometry import generate_detector
from lucid.utils import generate_random_params, load_single_event, save_single_event
from lucid.detector_params import ParticleParams, isotropic_source, load_detector_params
import jax, jax.numpy as jnp
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.utils import spherical_to_cartesian
```

**Config:** `../config/SK_physics_config.json`
**Track:** energy=1050.0, position=[0,0,0], theta=pi/2, phi=pi/6, t0=0.0

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports, define track params, create figures dir
- **Cell 1 (code):** Define visualize_detector_geometry: generates equal-charge sensor display
- **Cell 2 (code):** (Commented) Iterate over detector names
- **Cell 3 (code):** detector_names = ['SK']
- **Cell 4 (code):** Generate prediction event: setup_event_simulator (Nphot=1M, K=6, is_data=False), save to output/
- **Cell 5 (code):** Generate data-like event: load ROOT photons, apply rotation/translation, run data_simulator
- **Cell 6 (code):** Define visualization functions: visualize_3D_event_for_detector, visualize_3D_data_event_for_detector
- **Cell 7 (code):** Check for missing sensors
- **Cell 8 (code):** Run visualizations for SK

### D. Key Computations
- **Forward pass:** `simulator(track, key)` for prediction; `data_simulator(track, key, photon_data)` for data
- **Visualization:** `detector.visualize_event_data_plotly_discs(indices, charges, times, ...)`

### E. Baseline Outputs
- PDFs: `figures/{name}_geometry.pdf`, `figures/{name}_3D_event_prediction.pdf`, `figures/{name}_3D_data_like_event.pdf`

---

## 17. cylinder_2D_displays.ipynb

### A. Big Picture
Creates 2D unfolded cylindrical detector displays showing charge and time patterns for both data-like and prediction events.

### B. Setup
```python
import torch, sys; sys.path.append('..')
from lucid.geometry import generate_detector
from lucid.utils import load_single_event, save_single_event, generate_random_params, print_particle_params
from lucid.losses import WC_loss
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.detector_params import ParticleParams, load_detector_params
import jax, jax.numpy as jnp
from lucid.geometry import load_detector_geom
from lucid.utils import sparse_to_full
```

**Config:** `../config/SK_geom_config.json`, `../config/SK_physics_config.json`
**Data file:** `../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root`
**Key params:** Nphot=1M, K=20 (data), K=7 (prediction, temperature=0.1), true_position=[-10,0,0], true_direction=[1,0,0]

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports
- **Cell 1 (code):** Setup detector, simulators (prediction: K=7, temp=0.1; data: K=20, is_data=True)
- **Cell 2 (code):** Generate event: entry_idx=4, true_position=[-10,0,0], true_direction=[1,0,0], apply rotation/translation, run data_simulator
- **Cell 3 (code):** Define create_detector_display function for cylinder unfolding visualization
- **Cell 4 (code):** Save events to h5, load them back, create displays for charge and time

### D. Key Computations
- **Forward pass:** data_simulator returns (charges, times); prediction_simulator returns same
- **Display:** Cylinder unfolding with EllipseCollection colored by charge or time

### E. Baseline Outputs
- PDFs: `figures/data_event_display_charge.pdf`, `figures/data_event_display_time.pdf`, `figures/pred_event_display_charge.pdf`, `figures/pred_event_display_time.pdf`

---

## 18. data_vs_pred_hit_predictions.ipynb

### A. Big Picture
Compares data-like and prediction-like events side by side, analyzing charge/time distributions and their agreement as a function of the number of rays (Nphot).

### B. Setup
```python
import sys, os; sys.path.append(os.path.dirname(os.path.abspath('')))
import numpy as np, jax, jax.numpy as jnp, matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go, plotly.subplots as sp

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.utils import spherical_to_cartesian, base_dir_path
from lucid.optimization.grid_search import get_detector_bounds
from lucid.utils import generate_random_event_params
from lucid.detector_params import ParticleParams, load_detector_params
```

**Config:** `base_dir_path() + 'config/SK_geom_config.json'`, `base_dir_path() + 'config/SK_physics_config.json'`
**Data file:** `../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root`
**Key params:** n_photons=15_000, K=6, entry_idx=2, min_charge=1.0, track_position=[-10,0,0], track_direction=[1,0,0]

### C. Step-by-step Cell Description
- **Cell 0 (code):** Imports
- **Cell 2 (code):** Configuration dict
- **Cell 4 (code):** Setup detector, get bounds
- **Cell 5 (code):** Setup prediction_simulator (is_data=False, temperature=0.0, K=6) and data_simulator (is_data=True, temperature=0.0, K=6)
- **Cell 7 (code):** Load ROOT photons, apply rotation/translation
- **Cell 9 (code):** Simulate both events: prediction_simulator(track, key) and data_simulator(track, key, photon_data)
- **Cell 11 (code):** Statistical comparison: charge distributions, time distributions, charge vs time scatter, summary stats
- **Cell 13-14 (code):** Nrays variation study configuration: NRAYS_VALUES=[10k, 25k, 50k, 150k, 250k, 500k], N_EVENTS=30
- **Cell 15-16 (code):** Helper functions: prepare_photon_data, compute_metrics (covariance, time residual, charge residual), pad photons to 1M
- **Cell 17 (code):** Main loop: for each nrays, setup simulators once, process N_EVENTS events with random track params
- **Cell 18 (code):** Boxplot visualization of metrics vs nrays
- **Cell 20 (code):** Summary statistics table

### D. Key Computations
- **Forward pass (prediction):** `prediction_simulator(track, sim_key)` returns `(charges, times)`
- **Forward pass (data):** `data_simulator(track, sim_key, photon_data)` returns `(charges, times)`
- **Metrics:** covariance-like = mean((d_t - p_t) * (d_q - p_q)), time_residual = mean(d_t - p_t), charge_residual = mean(d_q - p_q)

### E. Baseline Outputs
- Charge/time distribution comparisons
- Box plots of metrics vs nrays
- Summary table of mean +/- std for covariance, time, and charge residuals per nrays
