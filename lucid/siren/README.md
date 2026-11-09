# SIREN emitters

`lucid.siren` holds the trained emission models that the simulator draws Cherenkov photons and
dE/dx deposits from. Each is a SIREN (sinusoidal representation network) fitted to a Geant4
(PhotonSim) lookup table.

- `net.py`: the SIREN network.
- `emitter.py`: `load_emitter(particle, material, kind)` returns an `Emitter` (network, weights,
  input domain, the s_max(E), N_photons(E) and t0 fits, the Cherenkov emission band); `density`
  evaluates it; `write_emitter` writes one and `validate_record` checks every record written or
  read. `normalize_inputs`, `encode_target` and `decode_target` are the training/inference
  contract: the trainer fits the net to `encode_target(t)`, inference reads
  `decode_target(net(normalize_inputs(x)))`.
- `training/`: training and validation, not used by the simulator
  ([`training/README.md`](training/README.md)).

## Emitters on disk

An emitter is `data/<material>/<particle>/<cherenkov|dedx>/{emitter.json, weights.npz}`.
`emitter.json` is tracked; `weights.npz` comes with `python scripts/download_data.py`. A material
JSON names the emitter set it uses with its `"siren"` key; water, WbLS and ice all use `"water"`.

The third network input is `s / s_max(E)` in [0, 1], the distance along the track over the track
length. `s_max(E)` takes one of the forms in `SMAX_PARAMS` (`A*E^B`, `smooth_two_power`, or
`piecewise`: two `smooth_two_power` pieces joined at `e_join_mev`). The shipped muon emitters use
`A*E^B` and the electron emitters `piecewise`. The lookup table carries the s_max and N_photons
fits as metadata attributes, and training copies them into `emitter.json` (`s_max_mm`,
`n_photons`), where the ray samplers read them.

## Training an emitter

The whole pipeline, from the PhotonSim scans to an installed emitter, is in
[Training a SIREN emitter](../../docs/guides/production/siren-training-inputs.md). Tables are
built with `lucid-build-photon-table` or `lucid-build-dedx-table`, an emitter is trained with
`lucid-train-siren` and installed with `lucid-siren-promote`:

```bash
lucid-train-siren --h5-path photon_lookup_table.h5 --output-dir runs/s1 --seed 1
lucid-siren-promote --run runs/s1 --smax-fit PhotonSim/data/water/mu-/smax_fit.csv \
    --t0 timing_parameters.json --emission-band-nm 274.91 673.83
```

The table determines the kind (Cherenkov or dE/dx), material and particle. The recipe is
[`training/nominal.json`](training/nominal.json), the one every shipped emitter was trained
with, unless `--recipe` names another; each recipe field also has its own flag. `--output-dir`
must not already hold an `emitter.json`. The outputs and how to judge convergence are in the
[CLI reference](../../docs/reference/cli.md).

A training run is not yet an emitter the simulator loads. `lucid-siren-promote` adds, for a
Cherenkov emitter, the t0 fit (`--t0`, the `timing_parameters.json` of PhotonSim's
`tools/t0_correction/calculate_t0.py`) and `emission_band_nm` (`--emission-band-nm`, the band in
nm its N_photons(E) counts; water: 274.91 673.83), checks the run against the `smax_fit.csv` its
table was built with, and writes `data/<material>/<particle>/<cherenkov|dedx>/`. It refuses a
Cherenkov emitter without the t0 fit or the band, and `load_emitter` refuses an emitter that was
not promoted.

## Validating an emitter

```bash
python -m lucid.siren.training.validate all --material water --particle muon --h5-path T --save
python -m lucid.siren.training.validate energy --energies 500,1000,1500 --save
python -m lucid.siren.training.validate energy --kind dedx --save
python -m lucid.siren.training.validate integral --h5-path T --energies 100,1000,50 --nphot 1000000 --save
python -m lucid.siren.training.validate rays --energies 200,1000,10 --nphot 500000 --save
```

- `energy`: the density over (opening angle or dE/dx, distance) at a few energies; the paper's
  `fig_siren_profile` draws the same.
- `integral`: the phase-space integral of the density against N_photons(E) and the table sum.
- `rays`: histograms of generated rays across energies.
- `all`: all three.

Each takes `--material` (the emitter set; default `water`), `--particle` (`muon` or `electron`;
default `muon`), `--output DIR` and `--save` (write to `output/siren/<material>/<particle>/`).
`energy` and `all` also take `--kind cherenkov|dedx`; `integral` and `rays` are Cherenkov only.
`--energies` is a list (`200,500,800`) or a range `start,end,count` (`100,1000,50`). `integral`,
and `all` of a Cherenkov emitter, need `--h5-path`, the photon table the emitter was trained on
(no table is distributed). The other checks need only the emitter.

`--monitoring` (live loss plots in Jupyter) needs IPython, which no LUCiD extra installs.
