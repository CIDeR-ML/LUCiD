# PhotonSim SIREN Validation Suite

This directory contains the PhotonSim SIREN validation suite, which provides comprehensive validation functionality through a single command-line tool.

## Overview

The validation suite provides three main validation functions:

1. **Cut-off Study** (`cutoff`) - Analyzes different threshold values for photon weight filtering
2. **N-Photon Integral** (`integral`) - Validates photon count predictions vs. real data
3. **Ray Generation Validation** (`rays`) - Validates ray generation patterns across different energies

## Quick Start

```bash
# Run all validations with default parameters (saves to output/siren)
python validation.py all

# Run all validations with custom output directory
python validation.py all --output custom_results/

# Run specific validation and save results
python validation.py cutoff --energy 500 --thresholds 1,2,4,8 --save
python validation.py integral --energies 100,1000,50 --nphot 1000000 --save
python validation.py rays --energies 200,1000,10 --nphot 500000 --save
```

## Commands

### `cutoff` - Cut-off Threshold Analysis

Analyzes the PhotonSim SIREN model output at different threshold values to understand the distribution of significant photon weights.

**Usage:**
```bash
python validation.py cutoff [OPTIONS]
```

**Options:**
- `--energy ENERGY`: Analysis energy in MeV (default: 500)
- `--thresholds THRESHOLDS`: Comma-separated thresholds (default: 1,2,4,8)
- `--output OUTPUT`: Custom output directory (default: output/siren)
- `--save`: Save results to output/siren directory

**Examples:**
```bash
# Default analysis at 500 MeV
python validation.py cutoff

# Custom energy and thresholds
python validation.py cutoff --energy 800 --thresholds 0.5,1,2,4,8,16

# Save results to directory
python validation.py cutoff --output results/cutoff_study/
```

**Output:**
- 2x2 grid of threshold visualizations
- Console output with valid point counts and fractions

### `integral` - N-Photon Integral Analysis

Compares predicted photon counts from the SIREN model with real PhotonSim data to derive correction factors. Linear fits are performed using only data points with energy ≥ 200 MeV for better accuracy.

**Usage:**
```bash
python validation.py integral [OPTIONS]
```

**Options:**
- `--energies ENERGIES`: Energy specification (see formats below)
- `--nphot NPHOT`: Number of photons for analysis (default: 1000000)
- `--output OUTPUT`: Custom output directory (default: output/siren)
- `--save`: Save results to output/siren directory

**Energy Formats:**
- Explicit list: `200,500,800,1000`
- Range specification: `100,1000,50` (start,end,count)
- If not specified, uses default range: 100-1000 MeV with 100 points

**Examples:**
```bash
# Default energy range
python validation.py integral

# Custom energy list
python validation.py integral --energies 200,400,600,800,1000

# Energy range with 50 points
python validation.py integral --energies 100,1000,50 --nphot 2000000
```

**Output:**
- Two-panel plot showing original and corrected data fits
- Console output with linear fit statistics

### `rays` - Ray Generation Validation

Validates ray generation patterns by creating 2D histograms of photon distribution across different energies.

**Usage:**
```bash
python validation.py rays [OPTIONS]
```

**Options:**
- `--energies ENERGIES`: Energy specification (see formats above)
- `--nphot NPHOT`: Number of photons for ray generation (default: 1000000)
- `--output OUTPUT`: Custom output directory (default: output/siren)
- `--save`: Save results to output/siren directory

**Examples:**
```bash
# Default: 20 energies from 200-1000 MeV
python validation.py rays

# Custom energy list
python validation.py rays --energies 200,400,600,800,1000

# Range with fewer points for faster execution
python validation.py rays --energies 200,1000,10 --nphot 500000
```

**Output:**
- Grid of 2D histograms showing ray distribution patterns
- Console output with processing progress

### `all` - Run All Validations

Runs all three validation studies with default parameters.

**Usage:**
```bash
python validation.py all [--output OUTPUT] [--save]
```

**Options:**
- `--output OUTPUT`: Custom output directory (default: output/siren)
- `--save`: Save results to output/siren directory

**Examples:**
```bash
# Run all validations (saves to output/siren by default)
python validation.py all

# Run all validations and save to custom directory
python validation.py all --output validation_results/

# Explicitly save to default directory
python validation.py all --save
```

## Output Files

When `--output` is specified or `--save` is used, the following PNG files are generated:

### Cut-off Study
- `cutoff_study_energy_{energy}.png`: Threshold visualization plots

### Integral Analysis
- `integral_analysis.png`: Original vs. corrected data plots

### Ray Validation
- `rays_validation.png`: Ray distribution grid plots

## Dependencies

The validation suite requires:
- PhotonSim SIREN model (trained)
- PhotonSim dataset (HDF5 format)
- JAX, NumPy, Matplotlib, SciPy
- Tools modules: `siren`, `simulation`, `generate`

## Model and Data Paths

The validator automatically locates:
- **Model**: `../notebooks/output/photonsim_siren_training/trained_model/photonsim_siren`
- **Dataset**: `../data/water/muon/photon_lookup_table.h5`

These paths can be customized in the `PhotonSimValidator` constructor.

## Examples

### Basic Usage
```bash
# Quick validation at single energy
python validation.py cutoff --energy 500

# Validate ray generation at a few energies
python validation.py rays --energies 200,500,800

# Full integral analysis
python validation.py integral --energies 100,1000,100
```

### Advanced Usage
```bash
# High-resolution cut-off study
python validation.py cutoff --energy 800 --thresholds 0.1,0.5,1,2,4,8,16,32

# Fast ray validation for testing
python validation.py rays --energies 200,1000,5 --nphot 100000

# Complete validation suite with output
python validation.py all --output validation_$(date +%Y%m%d_%H%M%S)/
```

## Notes

- The validation suite uses actual PhotonSim training ranges for analysis
- All visualizations are saved as high-resolution PNG files (150 DPI)
- JSON results include detailed statistics and fit parameters
- Progress is reported to console during execution
- The suite handles both individual validation runs and batch processing

## Troubleshooting

**Common Issues:**
1. **Model not found**: Ensure the PhotonSim SIREN model is trained and located at the expected path
2. **Dataset not found**: Verify the PhotonSim HDF5 dataset exists at the expected path
3. **Memory issues**: Reduce `--nphot` parameter for large validation runs
4. **Import errors**: Ensure all required tools modules are available in the Python path

**Performance Tips:**
- Use smaller `--nphot` values for faster testing
- Specify fewer energies for quick validation
- Use output directories to organize results from multiple runs