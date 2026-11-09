# SIREN emitters

`lucid.siren` holds the trained emission models (SIREN, sinusoidal representation networks) that
the forward simulator draws Cherenkov photons and dE/dx deposits from, and the tools that train
and validate them.

- `net.py` - the SIREN network
- `emitter.py` - `load_emitter(particle, material, kind)` -> `Emitter`: one trained emitter (net,
  weights, domain, the s_max / N_photons / t0 fits, the Cherenkov emission band); `density`, the
  one evaluator; `write_emitter`
- `training/` - training and validation, not on the forward path; its modules and Python API are in
  [`training/README.md`](training/README.md)

## Emitters on disk

- Shipped emitters: `data/{material}/{particle}/{cherenkov,dedx}/` (`emitter.json` tracked,
  `weights.npz` installed by `scripts/download_data.py`), loaded with `load_emitter`. A material
  JSON names the set it runs (`"siren": "water"` for wbls and ice).
- Training output: `--output-dir` (the emitter, history, plots, checkpoints).
- Validation output: `--output`, or `output/siren/{material}/{particle}/` with `--save`.

## Training

The pipeline that produces an emitter (PhotonSim scans, the s_max fit, the lookup tables, the
training scans) is in
[`docs/guides/production/siren-training-inputs.md`](../../docs/guides/production/siren-training-inputs.md).
The tables are built by `lucid-build-photon-table` / `lucid-build-dedx-table`; training is
`lucid-train-siren` (`python -m lucid.siren.training.train`):

```bash
# one seed of the nominal recipe, with the t0 fit the simulator needs (a Cherenkov emitter)
lucid-train-siren --h5-path photon_lookup_table.h5 --output-dir runs/s1 --seed 1 --t0 t0.json
```

The kind (Cherenkov or dE/dx), material and particle come from the table. The working point is
[`training/nominal.json`](training/nominal.json), the recipe of every shipped emitter, unless
`--recipe` names another; every recipe field can be overridden by its flag. `--t0` takes the
`t0.json` written by PhotonSim's `PhotonSim/tools/t0_correction/calculate_t0.py`. `lucid-train-siren --help`
lists the flags; [`docs/reference/cli.md`](../../docs/reference/cli.md) gives the outputs and how
to judge convergence.

### Installing a new emitter

Copy `emitter.json` and `weights.npz` from the output directory to
`data/<material>/<particle>/<cherenkov|dedx>/`. A Cherenkov emitter also needs
`emission_band_nm` (`[lo, hi]` in nm, the band its N_photons(E) counts) at the top level of its
`emitter.json`. The trainer does not write it; without it, `setup_event_simulator` raises for a
wavelength-mode track simulation unless the caller passes `cherenkov_emission_band`. The shipped
Cherenkov emitters carry it.

## Validation

The acceptance checks of a trained emitter:

1. **Energy study** (`energy`) - the density over (opening angle or dE/dx, distance) at a few
   energies (`--kind dedx` for the dE/dx emitter); the paper's `fig_siren_profile`
2. **N-photon integral** (`integral`) - the shape's phase-space integral vs N_photons(E) and the
   table sum
3. **Ray generation** (`rays`) - histograms of the generated rays across energies

```bash
python -m lucid.siren.training.validate all --material water --particle muon --save
python -m lucid.siren.training.validate energy --energies 500,1000,1500 --threshold 0.005 --save
python -m lucid.siren.training.validate energy --kind dedx --save
python -m lucid.siren.training.validate integral --energies 100,1000,50 --nphot 1000000 --save
python -m lucid.siren.training.validate rays --energies 200,1000,10 --nphot 500000 --save
```

Every command takes `--material` (the emitter set, `data/{material}/`; default water),
`--particle` (muon or electron), `--kind` (cherenkov or dedx), `--output DIR` and `--save`
(results to `output/siren/{material}/{particle}/`). `integral` and `rays` are Cherenkov only.
Energy formats: an explicit list `200,500,800` or a range `100,1000,50` (start,end,count).
Only `integral` (and so `all`) reads a lookup table (`data/{material}/{particle}/photon_lookup_table.h5`);
the other checks evaluate the emitter alone. No lookup table is distributed: build one from a
PhotonSim scan first ([SIREN training inputs](../../docs/guides/production/siren-training-inputs.md)).

## The s/s_max input axis

Since the lookup table's `format_version = "2.0"`, the third SIREN input is
`s / s_max(E) ∈ [0, 1]` rather than the absolute distance in mm; this is what lets the training
energy range extend up to 100 GeV. The per-(material, particle) fit `s_max(E)` has one of three
forms (`SMAX_PARAMS` in `emitter.py`): `A*E^B`, `smooth_two_power` (`a, b1, b2, E0`) or
`piecewise` (two `smooth_two_power` pieces joined at `e_join_mev`). The table records it as the
attribute `metadata.attrs/smax_form` plus that form's parameters and the shared fit columns, each
prefixed `smax_`, so no external `smax_fit.csv` is needed at training or inference time. The
trainer copies the fit, and the table's N_photons(E) fit (`nphot_*`), into the emitter's
`emitter.json` (`s_max_mm`, `n_photons`), which the ray samplers read to convert `s/s_max` into a
distance along the track. The shipped muon emitters use `A*E^B`, the electron emitters
`piecewise`.

## Notes

- Training uses JAX, Flax and Optax.
- Live loss plots in Jupyter with `--monitoring` need IPython, which no extra installs.
