# OPTIC: LUCiD: a Light-based Unified Calibration and trackIng Differentiable simulation

A high-performance, differentiable simulation framework to simulate optical particle detectors. This project enables gradient-based optimization of calibration parameters and particle reconstruction using automatic differentiation.

![Repository Overview](figures/det_repo_img.png?v=2)

## PhotonSim Integration

This repository integrates with [PhotonSim](https://github.com/cesarjesusvalls/PhotonSim) for training neural networks on Cherenkov photon data:

1. **PhotonSim** generates HDF5 lookup tables with photon density data
2. **diffCherenkov** loads these tables and trains SIREN networks for interpolation

### Details on the software structure and tutorial notebooks will be added soon.
