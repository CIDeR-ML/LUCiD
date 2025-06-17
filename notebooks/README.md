# Testing the Refactored Training Tools

This directory contains notebooks and tests for the newly refactored SIREN training tools.

## Quick Test

1. **Run the import test** (from diffCherenkov directory):
   ```bash
   python test_imports.py
   ```

2. **Run the full validation notebook**:
   ```bash
   cd notebooks/
   jupyter notebook test_training_tools.ipynb
   ```

## Files

- **`test_training_tools.ipynb`** - Comprehensive test notebook that validates all training components
- **`siren_training_example.ipynb`** - Example workflow for training on real PhotonSim data  
- **`README.md`** - This file

## What the Test Does

The test notebook will:

1. ✅ **Import Validation** - Test all module imports with fallback strategies
2. ✅ **Data Handling** - Load real PhotonSim data or create synthetic test data
3. ✅ **Dataset Functionality** - Test batch generation, normalization, train/val splits
4. ✅ **Training Configuration** - Test TrainingConfig dataclass
5. ✅ **Trainer Initialization** - Test SIRENTrainer setup and model initialization
6. ✅ **Monitoring Setup** - Test TrainingMonitor and callback system
7. ✅ **Training Execution** - Run a short training session (100 steps)
8. ✅ **Analysis Tools** - Test TrainingAnalyzer evaluation and error analysis
9. ✅ **Visualization** - Test all plotting functions
10. ✅ **Model Predictions** - Test inference capabilities

## Expected Output

If everything works correctly, you should see:

```
🎉 TRAINING TOOLS TEST SUMMARY
================================

✅ ALL TESTS PASSED!

📊 Test Results:
   ✅ Module imports: Working
   ✅ Data loading: Working (h5_lookup or synthetic)
   ✅ Dataset functionality: Working
   ✅ Training configuration: Working
   ✅ Trainer initialization: Working
   ✅ Monitoring setup: Working
   ✅ Training execution: Working
   ✅ Analysis tools: Working
   ✅ Visualization: Working
   ✅ Model predictions: Working

🚀 Ready for Production Use!
```

## Troubleshooting

### Import Errors
If you get import errors, the test notebook includes multiple fallback strategies:
1. Package import: `from siren.training import ...`
2. Direct module import: `from training import ...`
3. Individual file imports: `from trainer import ...`

### Missing Dependencies
Install required packages:
```bash
pip install jax flax optax numpy matplotlib h5py scipy seaborn
```

### No PhotonSim Data
The test automatically creates synthetic data if no real PhotonSim HDF5 file is found.

### JAX Device Issues
The test will show available JAX devices. For CPU-only testing, that's fine.

## Next Steps

Once the test passes:

1. **Generate real PhotonSim data**:
   ```bash
   cd ../PhotonSim
   python tools/table_generation/create_density_3d_table.py --data-dir data/mu- --visualize
   ```

2. **Use the example workflow**:
   ```bash
   jupyter notebook siren_training_example.ipynb
   ```

3. **Experiment with training parameters** in `TrainingConfig`

## File Structure Expected

```
diffCherenkov/
├── siren/
│   ├── __init__.py
│   └── training/
│       ├── __init__.py
│       ├── trainer.py      # SIRENTrainer, TrainingConfig
│       ├── dataset.py      # PhotonSimDataset
│       ├── monitor.py      # TrainingMonitor
│       ├── analyzer.py     # TrainingAnalyzer
│       └── README.md
├── notebooks/
│   ├── test_training_tools.ipynb
│   ├── siren_training_example.ipynb
│   └── README.md (this file)
├── test_imports.py
└── tools/ (legacy)
```