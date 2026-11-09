# Training a SIREN emitter

A SIREN emitter is trained on a lookup table built from a PhotonSim energy scan, then promoted
into `data/<material>/<particle>/<cherenkov|dedx>/` with the emission-time (t0) fit and the
emission band the simulator needs. The stages, in order:

| # | stage | command | inputs | outputs |
|---|---|---|---|---|
| 0 | fit `s_max(E)` | `smax.generate_jobs`, `merge.sh`, `smax/analyze.sh` | `smax/configs/<m>_<p>.json` | `$PHOTONSIM_DEV_PATH/data/<m>/<p>/smax_fit.csv` |
| 1 | training-input scan | `siren_inputs.generate_jobs`, `merge.sh` | `configs/<m>_<p>*.json`, `smax_fit.csv` | `training_inputs/<m>/<p>/<E>MeV/photonsim.root` |
| 2 | build the tables | `lucid-build-photon-table`, `lucid-build-dedx-table` | stage 1, `smax_fit.csv` | `training_inputs/<m>/lookup_tables/<particle>/{photon,dedx}_lookup_table.h5` |
| 3 | fit t0 | `t0.generate_jobs` (PhotonSim's `calculate_t0.py`) | stage 1, `smax_fit.csv` | `t0_parametrization/<m>/<p>/timing_parameters.json` |
| 4 | train | `train_siren.generate_jobs`, `rank_seeds.py` | a table, `train_siren/configs/*.json` | `training_tests/<scan>/seed<N>/{emitter.json, weights.npz}` |
| 5 | promote | `lucid-siren-promote` | a run, `smax_fit.csv`; Cherenkov: the t0 fit and band | `data/<m>/<p>/<kind>/{emitter.json, weights.npz}` |
| 6 | validate | `python -m lucid.siren.training.validate` | the emitter; `integral`: its table | `output/siren/<m>/<p>/` |

`<m>/<p>` is PhotonSim's material and particle (`water/mu-`, `water/e-`); `<particle>` and the
`data/` directories use LUCiD's names (`muon`, `electron`). Stage 3 needs only stages 0 and 1,
and can run beside stages 2 and 4. Output paths are under `SIREN_OUTPUT_BASE_PATH`.

## Setup

The cluster stages read the site from `lucid/production/platform/user_paths.sh`, copied from a
template in `lucid/production/platform/templates/`; a setting a stage needs and does not find is
an error that names it.

- `LUCID_IMAGE_PATH`, `SLURM_ACCOUNT`, `SLURM_PARTITION` (CPU jobs), `SIREN_OUTPUT_BASE_PATH`.
- `PHOTONSIM_DEV_PATH`: a PhotonSim checkout (main, May 2026 or later), built in the image
  ([container](container.md)). Stages 0 and 1 run its `build/PhotonSim`, which must fill
  `PhotonHist_TimeDistanceNorm` for stage 3; stage 0 writes and stages 1 to 3 read its
  `data/<m>/<p>/smax_fit.csv`; stage 3 runs its `tools/t0_correction/calculate_t0.py`.
- `SIREN_TRAIN_PARTITION`, a GPU partition (S3DF: `ampere`, `ada`, `hopper`, `turing`), and
  `LUCID_ENV_BASE`, the Python user site holding the GPU jaxlib the image lacks (the template
  says how to make one), for stage 4.
- `LUCID_DEV_PATH` (optional): the jobs run this checkout instead of the image's LUCiD.
- `SIREN_INPUTS_BASE_PATH` (optional): where stages 3 and 4 read earlier stages' outputs;
  default `SIREN_OUTPUT_BASE_PATH`. See [Existing inputs on S3DF](#existing-inputs-on-s3df).

The generators run in the container through `run_in_container.sh`, which submits from the host
with `-s`:

```bash
cd lucid/production/siren_inputs
source ../platform/user_paths.sh          # the commands below use its settings
RUN=../platform/run_in_container.sh
```

## Stages 0 and 1: the PhotonSim scans

Both scans fire one particle along +z from the origin, monoenergetic, one cell per energy, with
decays off and no per-photon output (`smax_planning.py`, `siren_planning.py`). The energies,
events and particle are in the configs; the commands are in
[`smax/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/smax/README.md)
and
[`siren_inputs/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/README.md).

```bash
$RUN lucid.production.siren_inputs.smax.generate_jobs -c smax/configs/water_mu.json -s
./merge.sh $SIREN_OUTPUT_BASE_PATH/smax_parametrization && smax/analyze.sh $SIREN_OUTPUT_BASE_PATH/smax_parametrization
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_mu.json -s
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_mu_lowE.json -s      # 150-490 MeV
./merge.sh $SIREN_OUTPUT_BASE_PATH/training_inputs
```

| particle | stage 0 config | stage 1 configs | stage 1 cells |
|---|---|---|---|
| mu- | `smax/configs/water_mu.json` | `configs/water_mu.json`, `configs/water_mu_lowE.json` | 307, 150 MeV to 100 GeV, 10k events |
| e- | `smax/configs/water_el.json`, `water_el_lowE.json` | `configs/water_el.json` | 339, 1 MeV to 100 GeV, 10k to 100k events |

`s_max(E)` is the track range, fitted per material and particle in one of three forms (`form` in
the CSV): `A*E^B`, `smooth_two_power` (`a, b1, b2, E0`) or `piecewise` (two `smooth_two_power`
pieces joined at `e_join_mev`). Stage 1 macros set `/output/smax <smax_mm> mm`, so PhotonSim
fills, against `s/s_max` in [0, 1] (500 × 500 bins, beyond `s_max` to the overflow):
`PhotonHist_AngleDistanceNorm` (opening angle), `dEdxHist_DistanceNorm` (dE/dx) and
`PhotonHist_TimeDistanceNorm` (photon delay `t − d/c`). The first two make the tables, the third
the t0 fit. The fit travels in the tables' metadata into the emitter, so later stages never read
the CSV again, except promotion, which checks the run against it.

## Stage 2: build the tables

Each table holds every energy of the scan in memory; build it in a batch job. The commands here
run this checkout's modules (`python3 -m`, with the checkout bound at `/opt/LUCiD` and first on
`PYTHONPATH`): an image's console scripts are as old as the image.

```bash
T=$SIREN_OUTPUT_BASE_PATH/training_inputs
srun --partition milano --account "$SLURM_ACCOUNT" --qos "${SLURM_QOS:-normal}" --mem 32G --time 02:00:00 \
    apptainer exec -B "$APPTAINER_BINDS" -B "$(git rev-parse --show-toplevel):/opt/LUCiD" \
    --env PYTHONPATH=/opt/LUCiD "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 -m lucid.siren.training.photonsim_data.build_tables photon \
        --data-dir "$T" --material water --particle mu- \
        --photonsim-dir "$PHOTONSIM_DEV_PATH" --output "$T/water/lookup_tables/muon/photon_lookup_table.h5"
```

and `build_tables dedx` likewise into `dedx_lookup_table.h5` (`lucid-build-photon-table` and
`lucid-build-dedx-table` are the same commands). A Cherenkov table from an older builder, with an
`N_photons(E)` fit not in `logpoly` form, is refused by the trainer: rebuild it. Every `<E>MeV` directory under
`<data-dir>/<material>/<particle>/` is used. Flags:
[command-line tools](../../reference/cli.md#lucid-build-photon-table-lucid-build-dedx-table).
After writing, the builder re-reads the table and checks the required keys, that the distance
edges span [0, 1], the normalisation, one random energy against its `photonsim.root`, the
`smax_*` attributes against `smax_fit.csv`, and the photon table's `nphot_*` attributes. It exits
non-zero on a failure.

## Stage 3: fit t0

The emission time is `t(d, E) = d/c + A(E)(exp((d/λ(E))^β(E)) − 1)`, fitted by PhotonSim's
`calculate_t0.py` from `PhotonHist_TimeDistanceNorm` of the stage 1 cells: `A`, `λ`, `β` per
energy, then `log10 A`, `log10 λ` and `β` each a cubic in `log10 E`. The stage links the cells'
`photonsim.root` into one directory and writes one container job that runs it:

```bash
$RUN lucid.production.siren_inputs.t0.generate_jobs --material water --particle mu- -s
```

It writes `timing_parameters.json` and three plots (`trends.png` shows the cubics against the
per-energy fits). Details:
[`t0/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/t0/README.md).

## Stage 4: train

Not every seed of the nominal recipe converges, so a production emitter comes from a seed scan
ranked by validation loss. The tracked scans, `train_siren/configs/water_{mu,el}_{cherenkov,dedx}_seeds.json`,
are the scans the shipped emitters were chosen from; each names its table as
`${SIREN_INPUTS_BASE_PATH}/training_inputs/...`.

```bash
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c train_siren/configs/water_mu_cherenkov_seeds.json -s
apptainer exec -B "$APPTAINER_BINDS" "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 \
    train_siren/rank_seeds.py $SIREN_OUTPUT_BASE_PATH/training_tests/water_mu_cherenkov_seeds
```

One run is `lucid-train-siren --h5-path T --output-dir D --seed N`; the trainer reads the kind,
material and particle from the table. Flags, outputs and how to judge convergence:
[command-line tools](../../reference/cli.md#lucid-train-siren). Scan configs:
[`train_siren/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/train_siren/README.md).

## Stage 5: promote

A training run carries the net, its domain and decoding, and the table's `s_max(E)` and
`N_photons(E)` fits. `lucid-siren-promote` adds the t0 fit and the emission band (the band, in
nm, over which PhotonSim counted the photons behind `N_photons(E)`; for water 274.91 to 673.83, the
1.84 to 4.51 eV range of PhotonSim's optical table), checks the run's `s_max_mm` against
`smax_fit.csv`, records the provenance, and writes the emitter. It refuses a Cherenkov emitter
without `--t0` or `--emission-band-nm`.

```bash
lucid-siren-promote --run $SIREN_OUTPUT_BASE_PATH/training_tests/water_mu_cherenkov_seeds/seed1 \
    --smax-fit $PHOTONSIM_DEV_PATH/data/water/mu-/smax_fit.csv \
    --t0 $SIREN_OUTPUT_BASE_PATH/t0_parametrization/water/mu-/timing_parameters.json \
    --emission-band-nm 274.91 673.83 \
    --selection "water_mu_cherenkov_seeds/seed1: lowest final val_loss of 10 seeds"
```

A dE/dx run takes only `--run`, `--smax-fit` and `--selection`. The destination is
`data/<material>/<particle>/<kind>/` of the checkout running it (`--dest` for another); an emitter
already there is replaced only with `--force`. In the image, run it as
`python3 -m lucid.siren.training.promote` with the checkout bound over `/opt/LUCiD`.
`load_emitter` reads only promoted emitters.

A material without emitters of its own runs another set: `"siren": "<material>"` in
`config/materials/<material>.json`. To ship new weights, they go into the data asset
(`tools/make_asset.sh`, `data/MANIFEST.sha256`).

## Stage 6: validate

```bash
srun --partition milano --account "$SLURM_ACCOUNT" --qos "${SLURM_QOS:-normal}" --mem 32G --time 01:00:00 \
    apptainer exec -B "$APPTAINER_BINDS" -B "$(git rev-parse --show-toplevel):/opt/LUCiD" \
    --env PYTHONPATH=/opt/LUCiD "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 -m lucid.siren.training.validate all --material water --particle muon \
        --h5-path $SIREN_OUTPUT_BASE_PATH/training_inputs/water/lookup_tables/muon/photon_lookup_table.h5 --save
```

The checks are in
[`lucid/siren/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/siren/README.md#validating-an-emitter).

## Existing inputs on S3DF

The inputs behind the shipped water emitters are on S3DF under
`/sdf/data/neutrino/cjesus/CIDER/SIREN_files` (another user's: read them, never write there). The
S3DF template lists them; set `SIREN_INPUTS_BASE_PATH` to that path to rerun stage 3 or 4 on
them, or pass the paths to stage 2 and 5. Each shipped `emitter.json` names its run
(`provenance.run`), table (`provenance.table`) and t0 scan (`t0.source.scan_dir`).

| stage | path under it |
|---|---|
| 0 | `smax_parametrization/water/{mu-,e-}/<E>MeV/` |
| 1 | `training_inputs/water/{mu-,e-}/<E>MeV/photonsim.root` |
| 2 | `training_inputs/water/lookup_tables/{muon,electron}/{photon,dedx}_lookup_table.h5` (rebuilt 2026-07-01, after the shipped emitters were trained; their `N_photons(E)` fit predates the `logpoly` form, so rebuild them before training) |
| 3 | `t0_parametrization/water/mu-/timing_parameters.json`, `t0_parametrization/water/e-/inputs/timing_parameters.json` (inputs: links into stage 1) |
| 4 | `training_tests/<scan>/seed<N>/` |

The `s_max` fits are PhotonSim's own `data/water/{mu-,e-}/smax_fit.csv`.

## Table layout (`format_version = "2.0"`)

```
photon_lookup_table.h5
├── data/
│   ├── photon_table_raw      (n_energies, 500, 500)
│   └── photon_table_average  (n_energies, 500, 500)   raw / events per energy
├── coordinates/
│   ├── energy_values, energy_centers   (n_energies,)  MeV, non-uniform
│   ├── energy_edges                    (n_energies+1,)
│   ├── angle_edges (501,), angle_centers (500,)       0 to π rad
│   └── distance_edges (501,), distance_centers (500,) s/s_max, 0 to 1
└── metadata/
    ├── attrs: material, particle, distance_axis="s_over_smax", format_version, photonsim_version,
    │          normalization="average", average_units="photons/event", bin counts and ranges,
    │          table_shape, total_photons,
    │          smax_form, smax_<param>..., smax_{fit_min_mev, fit_max_mev, quantile,
    │          quantile_multiplier, generated_at_utc},
    │          nphot_{form, coeffs, u0, du, fit_min_mev, fit_max_mev, rms_frac, max_frac}
    │          (photon table only: the N_photons(E) fit that sets the emitter's yield)
    └── events_per_file       structured (energy: i4, events: i4)
```

The dE/dx table uses `dedx_table_*` for `photon_table_*`, a `dedx_*` axis (0 to 1000 keV/mm)
for the angle, `total_edep` for `total_photons`, `average_units = "MeV/event"` (entries are
weighted by deposited energy) and `data_type = "dedx"`.

## Troubleshooting

- `PhotonHist_AngleDistanceNorm` missing in an `<E>MeV` file, or `distance_edges are not
  [0, 1]`: stage 1 ran without `/output/smax`. Rerun that energy with a config whose generator
  sets `smax_mm`. The t0 stage refuses such inputs too (`has no PhotonHist_TimeDistanceNorm`).
- `--require-list` reports missing energies: stage 1 is incomplete. Resubmit with
  `lucid/production/platform/run_in_container.sh lucid.production.siren_inputs.resubmit_failed
  <scan_dir> --list -s`, then `lucid/production/siren_inputs/merge.sh`, until all are present.
- `No s_max fit at PhotonSim/data/<m>/<p>/smax_fit.csv`: run stage 0 for that material and
  particle.
- `trained on a table whose s_max fit is not ...` (promotion): the table was built from another
  `smax_fit.csv`; promote against that one, or rebuild the table and retrain.
- `invalid emitter record: ... (install an emitter with lucid-siren-promote)`: a run was copied
  into `data/` by hand; promote it instead.
