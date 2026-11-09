# Training a SIREN emitter

A SIREN emitter is trained on a lookup table built from a PhotonSim energy scan. Four stages:

```
stage 0          stage 1                stage 2                    stage 3
smax/   ───►     siren_inputs/   ───►   lucid-build-*-table  ───►  lucid-train-siren
smax_fit.csv     photonsim.root         photon_lookup_table.h5     emitter.json + weights.npz
                 (one per energy)       dedx_lookup_table.h5
```

| stage | tool | output |
|---|---|---|
| 0. fit `s_max(E)` | `lucid/production/siren_inputs/smax/` and `PhotonSim/tools/smax/analyze_smax.py` | `PhotonSim/data/<material>/<particle>/smax_fit.csv` |
| 1. simulate each energy | `lucid/production/siren_inputs/` on a cluster ([runbook](clusters.md)) | `<OUT>/<material>/<particle>/<E>MeV/photonsim.root` |
| 2. build the tables | `lucid-build-photon-table`, `lucid-build-dedx-table` | `photon_lookup_table.h5`, `dedx_lookup_table.h5` |
| 3. train | `lucid-train-siren` | the emitter |

Stage 0 and 1 commands are in
[`lucid/production/siren_inputs/smax/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/smax/README.md)
and
[`lucid/production/siren_inputs/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/README.md).

The tables' distance axis is `s / s_max(E)` in [0, 1] rather than the distance in mm, which is
what lets one emitter cover energies up to 100 GeV. `s_max(E)` is the track range, fitted per
material and particle in one of three forms (`form` in the CSV): `A*E^B`, `smooth_two_power`
(`a, b1, b2, E0`) or `piecewise` (two `smooth_two_power` pieces joined at `e_join_mev`). The fit
travels in the table's metadata and from there into the emitter, so later stages never read the
CSV again.

Stage 1 macros set `/output/smax <smax_mm> mm`, which makes PhotonSim fill
`PhotonHist_AngleDistanceNorm` (opening angle against `s/s_max`, 500 × 500) and
`dEdxHist_DistanceNorm` (dE/dx against `s/s_max`, 500 × 500). Photons beyond `s_max` go to the
overflow bin.

## Stage 2: build the tables

```bash
lucid-build-photon-table --data-dir <OUT> --material water --particle mu- \
    --output data/water/muon/photon_lookup_table.h5
lucid-build-dedx-table   --data-dir <OUT> --material water --particle mu- \
    --output data/water/muon/dedx_lookup_table.h5
```

Every `<E>MeV` directory under `<OUT>/<material>/<particle>/` is used. Flags:
[command-line tools](../../reference/cli.md#lucid-build-photon-table-lucid-build-dedx-table).

After writing, the builder re-reads the table and checks the required keys, that the distance
edges span [0, 1], the normalisation, one random energy against its `photonsim.root`, the
`smax_*` attributes against `smax_fit.csv`, and the photon table's `nphot_*` attributes. It
exits non-zero on a failure.

### Table layout (`format_version = "2.0"`)

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

## Stage 3: train

```bash
lucid-train-siren --h5-path data/water/muon/photon_lookup_table.h5 --output-dir runs/water_muon \
    --seed 1 --t0 t0.json
```

The trainer reads the table kind from the file. Flags, outputs and how to judge convergence:
[command-line tools](../../reference/cli.md#lucid-train-siren). Validating an emitter is described
in [`lucid/siren/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/siren/README.md),
seed scans on a cluster in
[`train_siren/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/siren_inputs/train_siren/README.md).

## Using a new emitter

Copy `emitter.json` and `weights.npz` to `data/<material>/<particle>/<cherenkov|dedx>/` and set
`"siren": "<material>"` in `config/materials/<material>.json`. A Cherenkov emitter also needs:

- its emission-time fit, recorded by training with `--t0` (the `t0.json` written by PhotonSim's
  `calculate_t0.py`);
- `emission_band_nm: [lo, hi]` in `emitter.json`, the wavelength band (nm) over which PhotonSim
  counted the photons behind `N_photons(E)`. The trainer does not write it; add it by hand.
  Without it, a track simulation in wavelength mode raises unless `cherenkov_emission_band` is
  passed.

## Troubleshooting

- `PhotonHist_AngleDistanceNorm` missing in an `<E>MeV` file, or `distance_edges are not
  [0, 1]`: stage 1 ran without `/output/smax`. Rerun that energy with a config whose generator
  sets `smax_mm`.
- `--require-list` reports missing energies: stage 1 is incomplete. Resubmit with
  `lucid/production/platform/run_in_container.sh lucid.production.siren_inputs.resubmit_failed
  <scan_dir> --list -s`, then `lucid/production/siren_inputs/merge.sh`, until all are present.
- `No s_max fit at PhotonSim/data/<m>/<p>/smax_fit.csv`: run stage 0 for that material and
  particle.
