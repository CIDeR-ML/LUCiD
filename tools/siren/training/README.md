# SIREN Training Module

This module provides a refactored, notebook-friendly interface for training JAXSiren networks on PhotonSim data.

## Overview

The training module is organized into several key components:

- **`trainer.py`**: Main `SIRENTrainer` class with configurable training
- **`dataset.py`**: Data loading and management for PhotonSim data
- **`monitor.py`**: Real-time training monitoring and visualization
- **`analyzer.py`**: Post-training analysis and evaluation
- **Moved training scripts**: Legacy scripts moved here for reference

## Quick Start

```python
from siren.training import SIRENTrainer, TrainingConfig, PhotonSimDataset

# Load data
dataset = PhotonSimDataset('path/to/photon_lookup_table.h5')

# Configure training
config = TrainingConfig(
    hidden_features=256,
    num_steps=5000,
    learning_rate=1e-4
)

# Train model
trainer = SIRENTrainer(dataset, config, output_dir='output/training')
history = trainer.train()
```

## Module Structure

```
siren/training/
├── __init__.py                 # Main imports
├── trainer.py                  # SIRENTrainer class
├── dataset.py                  # PhotonSimDataset classes
├── monitor.py                  # TrainingMonitor and callbacks
├── analyzer.py                 # TrainingAnalyzer class
├── README.md                   # This file
├── photonsim_data/            # Dataset creation utilities
│   ├── create_siren_dataset.py
│   └── inspect_siren_dataset.py
└── [legacy scripts]           # Moved from scripts/ and tools/
    ├── train_photonsim_h5.py
    ├── monitor_training.py
    ├── analyze_trained_siren.py
    ├── train_photonsim_siren.py
    └── photonsim_h5_dataset.py
```

## Key Classes

### SIRENTrainer

Main training class with features:
- Configurable model architecture and training parameters
- Automatic checkpointing and logging
- JAX/Flax-based implementation
- Callback system for monitoring
- Built-in validation

### TrainingConfig

Dataclass for training configuration:
```python
config = TrainingConfig(
    # Model architecture
    hidden_features=256,
    hidden_layers=4,
    w0=30.0,
    
    # Training parameters
    learning_rate=1e-4,
    batch_size=16384,
    num_steps=10000,
    
    # Monitoring
    log_every=100,
    checkpoint_every=1000
)
```

### PhotonSimDataset

Dataset loader supporting:
- HDF5 lookup tables from PhotonSim
- Pre-sampled dataset directories
- Automatic normalization
- Train/validation splits
- Batch generation with JAX compatibility

### TrainingMonitor

Real-time monitoring with:
- Live plotting in notebooks
- Progress tracking
- Time estimation
- Metric logging
- Export capabilities

### TrainingAnalyzer

Post-training analysis including:
- Comprehensive evaluation metrics
- Error pattern analysis
- Visualization plots
- Model comparison
- Results export

## Usage Examples

### Basic Training
```python
from siren.training import *

# Load data
dataset = PhotonSimDataset('photon_lookup_table.h5', val_split=0.1)

# Configure training
config = TrainingConfig(num_steps=5000, learning_rate=1e-4)

# Train
trainer = SIRENTrainer(dataset, config)
history = trainer.train()
```

### With Monitoring
```python
# Set up monitoring
monitor = TrainingMonitor('output/training', live_plotting=True)
callback = LiveTrainingCallback(monitor, update_every=100)

# Add to trainer
trainer.add_callback(callback)

# Training will now show live plots
history = trainer.train()
```

### Analysis
```python
# Analyze results
analyzer = TrainingAnalyzer(trainer, dataset)
results = analyzer.evaluate_model(n_samples=50000)
analyzer.plot_comparison()

# Error analysis
error_analysis = analyzer.analyze_error_patterns()
```

### Custom Callbacks
```python
def custom_callback(trainer, step):
    if step % 500 == 0:
        print(f"Custom callback at step {step}")

trainer.add_callback(custom_callback)
```

## Data Format

The module expects PhotonSim data in either:

1. **HDF5 format** (preferred):
   - Path to `.h5` file with PhotonSim lookup table
   - Automatically handles normalization and coordinate grids

2. **Sampled dataset format**:
   - Directory containing `inputs.npy`, `targets.npy`
   - Optional `metadata.json` and `normalization_bounds.npz`

## Configuration Options

### Model Architecture
- `hidden_features`: Number of hidden units (default: 256)
- `hidden_layers`: Number of hidden layers (default: 3)
- `w0`: SIREN frequency parameter (default: 30.0)

### Training Parameters
- `learning_rate`: Initial learning rate (default: 1e-4)
- `batch_size`: Training batch size (default: 16384)
- `num_steps`: Total training steps (default: 10000)
- `weight_decay`: L2 regularization (default: 0.0)

### Learning Rate Schedule
- `scheduler_step_size`: Steps between LR decay (default: 2000)
- `scheduler_gamma`: LR decay factor (default: 0.1)

### Monitoring
- `log_every`: Steps between logging (default: 100)
- `val_every`: Steps between validation (default: 100)
- `checkpoint_every`: Steps between checkpoints (default: 1000)

## Dependencies

- JAX and Flax for neural networks
- NumPy for data handling
- Matplotlib for visualization
- H5py for data loading
- Optax for optimization
- SciPy for analysis

## Example Notebook

See `../notebooks/siren_training_example.ipynb` for a complete workflow example.

## Migration from Legacy Scripts

The legacy training scripts have been moved to this directory but are deprecated. Use the new classes instead:

| Legacy Script | New Class | Usage |
|---------------|-----------|-------|
| `train_photonsim_h5.py` | `SIRENTrainer` | `trainer = SIRENTrainer(dataset, config)` |
| `monitor_training.py` | `TrainingMonitor` | `monitor = TrainingMonitor(output_dir)` |
| `analyze_trained_siren.py` | `TrainingAnalyzer` | `analyzer = TrainingAnalyzer(trainer, dataset)` |

## Output Files

Training produces:
- `final_model.npz`: Trained model parameters
- `config.json`: Training configuration
- `training_history.json`: Loss and metrics history
- `*.png`: Visualization plots
- Checkpoint files during training