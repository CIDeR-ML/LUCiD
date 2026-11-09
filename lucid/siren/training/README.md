# SIREN Training Module

Training and validation of the SIREN emitters on PhotonSim lookup tables. Nothing here is on
the forward path: a simulation loads trained emitters through `lucid.siren.emitter`.

## Overview

- **`train.py`**: the `lucid-train-siren` CLI (`python -m lucid.siren.training.train`)
- **`validate.py`**: validation suite for trained emitters (`python -m lucid.siren.training.validate`)
- **`trainer.py`**: `Recipe` and `SIRENTrainer`; **`nominal.json`**: the committed recipe
- **`dataset.py`**: `PhotonSimDataset`, the lookup table and its batching
- **`monitor.py`**: `TrainingMonitor` and `LiveTrainingCallback`, live progress plots
- **`prediction_plot.py`**: `PredictionComparisonCallback`, in-training prediction-vs-truth PDFs
- **`analyzer.py`**: `TrainingAnalyzer`, post-training evaluation
- **`photonsim_data/`**: the lookup-table builders (`lucid-build-photon-table`,
  `lucid-build-dedx-table`)

The training workflow, validation and the table's s/s_max axis are in [`../README.md`](../README.md);
the CLI's flags in `lucid-train-siren --help`.

The package `__init__` imports nothing; import the module you need.

## Quick Start

```python
from lucid.siren.training.dataset import PhotonSimDataset
from lucid.siren.training.trainer import Recipe, SIRENTrainer

recipe = Recipe.load(num_steps=5000)            # nominal.json with overrides
dataset = PhotonSimDataset('path/to/photon_lookup_table.h5', val_split=recipe.val_split,
                           zero_threshold=recipe.zero_threshold,
                           zero_keep_frac=recipe.zero_keep_frac,
                           energy_balance=recipe.energy_balance,
                           target_importance=recipe.target_importance)
trainer = SIRENTrainer(dataset, recipe, output_dir='output/training')
history = trainer.train()
trainer.save_emitter('output/training')         # emitter.json + weights.npz
```

## Key Classes

### Recipe

The working point as a frozen dataclass, read from a JSON file (default
[`nominal.json`](nominal.json), the recipe of every shipped emitter: network size, optimiser,
schedule, batch, steps and the dataset knobs below). Unknown or missing fields raise. It is
recorded into every emitter it trains (`emitter.json` `training.recipe`).

### SIRENTrainer

An MSE loss on log-normalized targets with the recipe's optimiser (`adam`, `adamw`, `sgd`) and,
when `use_patience_scheduler`, a patience-based learning-rate schedule; `init=` starts from a
`weights.npz` (warm start), `checkpoint_every=` writes `checkpoint_step_<N>.npz`, and callbacks
(`add_callback(fn)`, called as `fn(trainer, step)`). `save_emitter(dir)` writes the emitter
through `lucid.siren.emitter.write_emitter`, the loader's own schema.

### PhotonSimDataset

Loads a photon or dE/dx lookup table (`.h5`, auto-detected from `data/photon_table_average`
or `data/dedx_table_average`), normalizes inputs to [-1, 1] and log-normalizes targets to
[0, 1], keeps a fraction of the near-zero bins (`zero_threshold`, `zero_keep_frac`), splits
train/validation, and draws batches, optionally weighted across energies
(`energy_balance`) and by target value (`target_importance`).

### TrainingMonitor

Reads `training_history.json` from a training directory and plots progress with a time
estimate; `LiveTrainingCallback` refreshes it during training.

### TrainingAnalyzer

Post-training evaluation of a trainer and its dataset:

```python
from lucid.siren.training.analyzer import TrainingAnalyzer

analyzer = TrainingAnalyzer(trainer, dataset)
results = analyzer.evaluate_model(n_samples=50000)       # R², RMSE, MAE, percentiles per split
error_analysis = analyzer.analyze_error_patterns()       # errors binned by each input
analyzer.plot_comparison()                               # 3x4 summary figure
```

## Data Format

An HDF5 lookup table as written by `photonsim_data/build_tables.py`: the average table under
`data/`, the bin centers under `coordinates/`, and the s_max and N_photons fits as `metadata`
attributes (carried into the emitter's `emitter.json`; the s_max forms are described in
[`../README.md`](../README.md)).

## Output Files

Under `output_dir` (`lucid-train-siren`):

- `emitter.json` + `weights.npz`: the emitter, with its domain, decoding, the s_max and
  N_photons fits from the table, the recipe, the losses and the commit
- `recipe.json`: the resolved recipe (written at the start, for `TrainingMonitor`)
- `training_history.json`: train/val loss, learning rate and step
- `checkpoint_step_<N>.npz`: with `--checkpoint-every`
- `final_training_progress.pdf`, `prediction_plots/` (what the plots show: the
  `prediction_plot.py` module docstring)
