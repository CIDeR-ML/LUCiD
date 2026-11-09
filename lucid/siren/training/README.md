# SIREN training

Training and validation of SIREN emitters on PhotonSim lookup tables. The simulator does not use
this package; it loads trained emitters through `lucid.siren.emitter`. The command-line workflow
(`lucid-train-siren`, `lucid-siren-promote`, validation) is in [`../README.md`](../README.md).
The package `__init__` imports nothing, so import the module you need.

| module | contents |
|---|---|
| `train.py` | the `lucid-train-siren` command (`python -m lucid.siren.training.train`) |
| `promote.py` | the `lucid-siren-promote` command: a training run, with the t0 fit and band, into `data/` |
| `validate.py` | validation of a trained emitter (`python -m lucid.siren.training.validate`) |
| `trainer.py` | `Recipe` and `SIRENTrainer` |
| `nominal.json` | the recipe of every shipped emitter |
| `dataset.py` | `PhotonSimDataset`: loads a lookup table and draws batches |
| `monitor.py` | `TrainingMonitor` and `LiveTrainingCallback`: progress plots |
| `prediction_plot.py` | `PredictionComparisonCallback`: prediction-vs-truth plots during training |
| `analyzer.py` | `TrainingAnalyzer`: evaluation after training |
| `photonsim_data/` | the table builders `lucid-build-photon-table` and `lucid-build-dedx-table` |

## From Python

```python
from lucid.siren.training.dataset import PhotonSimDataset
from lucid.siren.training.trainer import Recipe, SIRENTrainer

recipe = Recipe.load(num_steps=5000)            # nominal.json with overrides
dataset = PhotonSimDataset('photon_lookup_table.h5', val_split=recipe.val_split,
                           zero_threshold=recipe.zero_threshold,
                           zero_keep_frac=recipe.zero_keep_frac,
                           energy_balance=recipe.energy_balance,
                           target_importance=recipe.target_importance)
trainer = SIRENTrainer(dataset, recipe, output_dir='output/training')
history = trainer.train()
trainer.save_emitter('output/training')         # emitter.json + weights.npz
```

- `Recipe`: the training settings (network size, optimiser, schedule, batch, steps, dataset
  options) as a frozen dataclass read from JSON; an unknown or missing field raises. It is
  recorded in every emitter it trains (`emitter.json`, `training.recipe`).
- `SIRENTrainer`: MSE on log-normalised targets with the recipe's optimiser (`adam`, `adamw` or
  `sgd`) and, with `use_patience_scheduler`, a patience-based learning-rate schedule. `init=`
  warm-starts from a `weights.npz`, `checkpoint_every=` writes `checkpoint_step_<N>.npz`, and
  `add_callback(fn)` registers `fn(trainer, step)`. `save_emitter(dir)` writes through
  `lucid.siren.emitter.write_emitter`.
- `PhotonSimDataset`: reads `data/photon_table_average` or `data/dedx_table_average`, scales
  inputs to [-1, 1] and log-scales targets to [0, 1] (`normalize_inputs` and `encode_target` of
  `lucid.siren.emitter`, which inference inverts), keeps a fraction of the near-zero bins
  (`zero_threshold`, `zero_keep_frac`), splits train and validation, and can weight batches
  across energies (`energy_balance`) and by target value (`target_importance`).
- `TrainingMonitor` reads `training_history.json` from a training directory and plots progress;
  `LiveTrainingCallback` refreshes it during training.
- `TrainingAnalyzer(trainer, dataset)`: `evaluate_model()` (R², RMSE, MAE and percentiles per
  split), `analyze_error_patterns()` (errors binned by each input), `plot_comparison()`.

## Files

A lookup table, as `photonsim_data/build_tables.py` writes it, holds the averaged table under
`data/`, the bin centres under `coordinates/`, and the s_max and N_photons fits as `metadata`
attributes.

A training run writes to its output directory:

- `emitter.json` and `weights.npz`: the emitter, with its domain, decoding, the table's s_max and
  N_photons fits, the recipe, the losses and the commit, but no t0 fit or emission band (added
  when `lucid-siren-promote` installs it);
- `recipe.json`: the resolved recipe, written at the start;
- `training_history.json`: train and validation loss, learning rate, step;
- `checkpoint_step_<N>.npz` with `--checkpoint-every`;
- `final_training_progress.pdf` and `prediction_plots/`.
