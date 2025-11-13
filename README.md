# LUCiD: a Light-based Unified Calibration and trackIng Differentiable simulation

🚧 **Under Construction** — Code is being refactored and reorganized. A stable release is coming soon.

A high-performance, differentiable simulation framework for optical particle detectors. This project enables gradient-based optimization of calibration parameters and particle reconstruction using automatic differentiation.

![Repository Overview](figures/combined_3x2_charge_displays.png)

## Overview

LUCiD provides a JAX-based differentiable simulation of light propagation in optical detectors. Key features include:

- **Differentiable ray-tracing** with automatic differentiation for gradient-based optimization
- **Physics-informed neural network (SIREN)** as surrogate model for Cherenkov emission
- **Multi-geometry support**: cylindrical, spherical, and box detectors
- **Particle track reconstruction** via gradient descent with position, direction, initial time, and energy inference
- **Detector calibration** for optical parameters (scattering, absorption, reflection)

## Core Components

### Simulation (`tools/`)
- **`simulation.py`** - Main event simulator with photon propagation and sensor response
- **`geometry/`** - Detector geometries (cylinder, sphere, box) with sensor positions and spatial indexing
- **`propagation/`** - Photon ray-tracing with scattering, reflection, and absorption
- **`siren/`** - Physics SIREN neural network for Cherenkov emission modeling

### Optimization (`tools/optimization/`)
- **`single_track_optimization.py`** - Adam-based gradient descent for track parameters

### Data Generation (`tools/`)
- **`generate.py`** - Interface to PhotonSim ROOT files for data-like events
- **`utils.py`** - Random event generation, coordinate conversions, I/O utilities

## Input Files and PhotonSim Integration

This repository integrates with [PhotonSim](https://github.com/cesarjesusvalls/PhotonSim), a GEANT4 utility to create data-like events and inputs for training a surrogate model for Cherenkov light emission.

To avoid requiring users to install and run PhotonSim, we provide example input data and trained networks to get started. Use:

```bash
./scripts/download_data.sh
```

## Notebooks

Tutorial notebooks in `good_notebooks/` demonstrate key workflows:

1. **`train_siren.ipynb`** - Train Physics SIREN on PhotonSim lookup tables
2. **`geometry_and_events_3D_visualization.ipynb`** - Visualize detector geometries and simulated events
3. **`data_vs_pred_hit_predictions.ipynb`** - Compare data-like vs. prediction-like events
4. **`cylinder_2D_displays.ipynb`** - Create 2D unwrapped detector displays
5. **`parameter_scans_1D.ipynb`** - 1D parameter scans showing loss landscapes
6. **`grad_loss_and_opt_in_2D.ipynb`** - 2D gradient fields with optimization trajectories
7. **`track_optimization_visualization.ipynb`** - Track reconstruction convergence with bootstrap CIs
8. **`visualize_3D_track_optimization.ipynb`** - 3D visualization of optimization paths
9. **`computational_performance_evaluation.ipynb`** - Benchmark simulation and gradient computation
10. **`optimization_vs_variables.ipynb`** - Performance vs. photon count, sensor count, and energy
