# OPTIC: LUCiD: a Light-based Unified Calibration and trackIng Differentiable simulation

A high-performance, differentiable simulation framework to simulate optical particle detectors. This project enables gradient-based optimization of calibration parameters and particle reconstruction using automatic differentiation.

## PhotonSim Integration

This repository integrates with [PhotonSim](https://github.com/cesarjesusvalls/PhotonSim) for training neural networks on Cherenkov photon data:

1. **PhotonSim** generates HDF5 lookup tables with photon density data
2. **diffCherenkov** loads these tables and trains SIREN networks for interpolation

### Training on PhotonSim Data

```bash
# Option 1: Train directly on PhotonSim HDF5 lookup table
python scripts/train_photonsim_h5.py \
    --data-path ../PhotonSim/output/3d_lookup_table_density/photon_lookup_table.h5 \
    --output-dir output/photonsim_training \
    --num-steps 5000

# Option 2: Create pre-sampled dataset, then train
python tools/photonsim_data/create_siren_dataset.py \
    --table-path ../PhotonSim/output/3d_lookup_table_density/photon_lookup_table.h5 \
    --output output/siren_dataset \
    --n-samples 5000000

python scripts/train_photonsim_h5.py \
    --data-path output/siren_dataset \
    --num-steps 5000
```

### Project Structure

- `scripts/`: Standalone training and analysis scripts
  - `train_photonsim_h5.py`: **Main training script** (works with both HDF5 tables and sampled datasets)
  - `analyze_trained_siren.py`: Model analysis and visualization
  - `monitor_training.py`: Real-time training progress monitoring
- `tools/`: Reusable library modules
  - `photonsim_h5_dataset.py`: **Unified data loader** (HDF5 tables + sampled datasets)
  - `photonsim_data/`: PhotonSim data processing tools
  - `train_photonsim_siren.py`: SIREN trainer class
- `output/`: All generated files (git-ignored)

### Details on the software structure and tutorial notebooks will be added soon.
