# PhotonSim SIREN Module

This directory contains the SIREN (Sinusoidal Representation Networks) implementation for photon simulation, including training and validation tools.

## Structure

- `core.py` - Core SIREN model implementation
- `train.py` - Training script for PhotonSim SIREN models
- `validate.py` - Validation suite for trained models
- `training/` - Training utilities and modules
  - `trainer.py` - Main training logic
  - `dataset.py` - PhotonSim dataset handling
  - `inference.py` - Model inference utilities
  - `monitor.py` - Training monitoring and visualization
  - `analyzer.py` - Training analysis tools

## Training

Train SIREN models for different materials and particles:

```bash
# Train with default settings (water/muon)
python train.py

# Train for specific material/particle
python train.py --material ice --particle electron

# Custom training parameters
python train.py --batch-size 32768 --num-steps 50000 --learning-rate 5e-5

# Resume training from checkpoint
python train.py --resume
```

### Training Options

- `--material` - Material type (default: water)
- `--particle` - Particle type (default: muon)
- `--data-type` - `photon` or `dedx` (default: photon)
- `--h5-path` - Override the conventional .h5 path. `--material` / `--particle` / `--data-type` still drive the output directory (e.g. for training off a non-standard build at `/tmp/photon_lookup_table_mu.h5`).
- `--hidden-features` - Hidden layer size (default: 256)
- `--hidden-layers` - Number of hidden layers (default: 3)
- `--learning-rate` - Initial learning rate (default: 1e-4)
- `--batch-size` - Batch size (default: 65536)
- `--num-steps` - Total training steps (default: 30000)
- `--resume` - Resume from latest checkpoint

Models are saved to: `data/{material}/{particle}/siren_training/trained_model/`

### In-training prediction-vs-truth plots

`lucid-train-siren` saves a PNG every `--prediction-plot-every` steps comparing
the model's current predictions against the source lookup table at a handful
of representative anchors. The figure has two views:

- **Top**: intensity vs angle (or dE/dx for the dedx variant) at four fixed
  `s/s_max` slices (default `0.1, 0.25, 0.5, 0.75`), one panel per slice.
- **Bottom**: intensity vs `s/s_max`, summed over the 2nd axis.

Truth curves are dashed, predictions solid, matched colour per energy.
Output: `data/{material}/{particle}/{photon,dedx}_siren_training/prediction_plots/step_NNNNNN.png`.

Flags:

- `--prediction-plots` / `--no-prediction-plots` — default on.
- `--prediction-plot-every <N>` — refresh interval (default 500).
- `--prediction-plot-energies` — comma list of MeV (default `1000,2000,5000,10000,80000`).
- `--prediction-plot-distance-slices` — comma list in [0, 1] (default `0.1,0.25,0.5,0.75`).

Fresh runs wipe the `prediction_plots/` folder; `--resume` leaves existing
PNGs intact so a single training produces one continuous PNG stream.

## Validation

The validation suite provides three main validation functions:

1. **Cut-off Study** (`cutoff`) - Analyzes different threshold values for photon weight filtering
2. **N-Photon Integral** (`integral`) - Validates photon count predictions vs. real data
3. **Ray Generation Validation** (`rays`) - Validates ray generation patterns across different energies

### Quick Start

```bash
# Run all validations for default water/muon
python validate.py all

# Run all validations for specific material/particle
python validate.py all --material ice --particle electron

# Run specific validation with custom parameters
python validate.py cutoff --energy 500 --thresholds 1,2,4,8 --save
python validate.py integral --energies 100,1000,50 --nphot 1000000 --save
python validate.py rays --energies 200,1000,10 --nphot 500000 --save
```

### Global Options

All validation commands support:
- `--material` - Material type (default: water)
- `--particle` - Particle type (default: muon)
- `--output` - Custom output directory
- `--save` - Save results to output directory

Results are saved to: `data/{material}/{particle}/siren_training/validation/`

### Commands

#### `cutoff` - Cut-off Threshold Analysis

Analyzes the PhotonSim SIREN model output at different threshold values.

```bash
python validate.py cutoff [OPTIONS]
```

**Options:**
- `--energy ENERGY`: Analysis energy in MeV (default: 500)
- `--thresholds THRESHOLDS`: Comma-separated thresholds (default: 1,2,4,8)

#### `integral` - N-Photon Integral Analysis

Compares predicted photon counts with real PhotonSim data. Linear fits use only data ≥ 200 MeV.

```bash
python validate.py integral [OPTIONS]
```

**Options:**
- `--energies ENERGIES`: Energy specification (see formats below)
- `--nphot NPHOT`: Number of photons for analysis (default: 1000000)

**Energy Formats:**
- Explicit list: `200,500,800,1000`
- Range specification: `100,1000,50` (start,end,count)

#### `rays` - Ray Generation Validation

Validates ray generation patterns by creating 2D histograms of photon distribution.

```bash
python validate.py rays [OPTIONS]
```

**Options:**
- `--energies ENERGIES`: Energy specification (same formats as integral)
- `--nphot NPHOT`: Number of photons for ray generation (default: 1000000)

## Data Requirements

Both training and validation require an HDF5 lookup table at
`data/{material}/{particle}/photon_lookup_table.h5`. To build it from
PhotonSim ROOT output, see
[`docs/SIREN_TRAINING_INPUTS.md`](../../docs/SIREN_TRAINING_INPUTS.md).

### s/s_max input axis

Starting with `format_version = "2.0"`, the third SIREN input is
`s / s_max(E) ∈ [0, 1]` rather than absolute distance in mm — this is what
lets the training energy range extend up to 100 GeV. The per-(material,
particle) fit `s_max(E) = A·E^B` is welded into the `.h5` as
`metadata.attrs/smax_{A, B, fit_min_mev, fit_max_mev, quantile,
quantile_multiplier, generated_at_utc}`, so the file is self-contained — no
external `smax_fit.csv` is needed at training or inference time. The trainer
itself doesn't touch these attrs; they exist for the inference path
(`lucid/sources/siren_rays.py`, not yet migrated) to convert physical `s`
into `s/s_max` at sample time.

## Model Paths

The tools automatically handle model paths:
- Models: `data/{material}/{particle}/siren_training/trained_model/photonsim_siren`
- Training output: `data/{material}/{particle}/siren_training/` (checkpoints, history, etc.)
- Validation output: `data/{material}/{particle}/siren_training/validation/`

## Examples

### Complete Workflow

```bash
# 1. Train a model for water/muon
python train.py --material water --particle muon

# 2. Validate the trained model
python validate.py all --material water --particle muon --save

# 3. Train another model for ice/electron
python train.py --material ice --particle electron --batch-size 32768

# 4. Run specific validation
python validate.py integral --material ice --particle electron --energies 100,1000,100
```

### Advanced Usage

```bash
# High-resolution training
python train.py --hidden-layers 4 --hidden-features 512 --num-steps 50000

# Fast validation for testing
python validate.py rays --energies 200,1000,5 --nphot 100000

# Custom output organization
python validate.py all --output my_results/$(date +%Y%m%d)
```

## Notes

- Training uses JAX/Flax for GPU acceleration
- Validation automatically detects trained models
- All tools support automatic path resolution based on material/particle
- Live plotting available during training with `--no-monitoring` to disable