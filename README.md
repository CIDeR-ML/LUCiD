# LUCiD: a Light-based Unified Calibration and trackIng Differentiable simulation

🚧 **Under Construction** — Code is being refactored and reorganized. A stable release is coming soon.

A high-performance, differentiable simulation framework for optical particle detectors. This project enables gradient-based optimization of calibration parameters and particle reconstruction using automatic differentiation.

![Repository Overview](figures/combined_3x2_charge_displays.png)

## Input Files and PhotonSim Integration

This repository integrates with [PhotonSim](https://github.com/cesarjesusvalls/PhotonSim), a GEANT4 utility to create data-like events and inputs for training a surrogate model for Cherenkov light emission.

To avoid requiring users to install and run PhotonSim, we provide example input data and trained networks to get started. Use:

```bash
./scripts/download_data.sh

### Details on the software structure and tutorial notebooks will be added soon.
