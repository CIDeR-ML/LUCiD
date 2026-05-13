# SIREN training inputs: building `photon_lookup_table.h5`

This document covers the pipeline that turns PhotonSim simulation output
into the HDF5 lookup table consumed by LUCiD's SIREN trainer
(`lucid/siren/training/dataset.py`, class `PhotonSimDataset`).

## Pipeline overview

```
   ┌───────────────┐   per-energy ROOT files   ┌───────────────────┐
   │  PhotonSim    │ ────────────────────────► │  ROOT → h5        │
   │  (G4 binary)  │   <E>MeV/output.root       │  build script     │
   └───────────────┘                            └─────────┬─────────┘
        ▲                                                 │
        │ macro + JSON config                             │ photon_lookup_table.h5
        │                                                 ▼
   diffsim_input/water_lookup_table_*.json     ┌───────────────────┐
                                               │  SIREN trainer    │
                                               │  (PhotonSimDataset)│
                                               └───────────────────┘
```

Three stages, one artifact per stage:

| Stage | Input | Output | Tool |
|---|---|---|---|
| 1. Simulate | `.json` config + `.mac` template | `<E>MeV/output.root` (one per energy) | `lucid-run-job` (orchestrated) or `PhotonSim` directly |
| 2. Aggregate | per-energy ROOTs | `photon_lookup_table.h5` | `lucid-build-photon-table` |
| 3. Train | the .h5 | trained SIREN weights | `lucid-train-siren` |

A parallel dE/dx variant exists (`dEdxHist_Distance` histogram → `dedx_lookup_table.h5`, built by `lucid-build-dedx-table`); the SIREN dataset auto-detects which schema is present.

## Stage 1 — generate per-energy ROOT files

Use the orchestrated S3DF job system documented in
[`lucid/production/s3df_jobs/DiffSimInputs_README.md`](../lucid/production/s3df_jobs/DiffSimInputs_README.md).
The relevant configs are:

| Config | Particle | Energy scan | Histogram written |
|---|---|---|---|
| `PhotonSim/macros/diffsim_input/water_lookup_table_mu.json` | mu- | 10–2000 MeV step 10 | `PhotonHist_AngleDistance` (and `dEdxHist_Distance`) |
| `PhotonSim/macros/diffsim_input/water_lookup_table_el.json` | e- | 10–2000 MeV step 10 | same |

Both configs set `store_individual_photons: false` and `disable_decays: true`. They write the per-event 2D histograms into the ROOT file directly; per-photon arrays are not needed for stage 2.

Output layout (per `$OUTPUT_BASE_PATH/water/monoenergetic/averaged/<particle>/`):

```
<E>MeV/output.root        # one per energy point
```

Each `output.root` must contain:
- `OpticalPhotons` tree — used only for the per-file event count.
- `PhotonHist_AngleDistance` (photon variant) **or** `dEdxHist_Distance` (dE/dx variant) — the 2D histogram that becomes a slice of the 3D table.

Validate the directory before stage 2:

```bash
python -m lucid.siren.training.photonsim_data.check_root_files
# or, with a custom path:
python -c "from lucid.siren.training.photonsim_data.check_root_files import check_root_files; check_root_files('<path>')"
```

## Stage 2 — build the .h5

```bash
# photon angle/distance variant — produces photon_lookup_table.h5
lucid-build-photon-table \
    --data-dir   <stage-1-output>/water/monoenergetic/averaged/mu- \
    --output     data/water/mu- \
    --energy-min 100 --energy-max 2000 --energy-step 10

# dE/dx variant — produces dedx_lookup_table.h5
lucid-build-dedx-table \
    --data-dir   <stage-1-output>/water/monoenergetic/averaged/mu- \
    --output     data/water/mu- \
    --energy-min 10 --energy-max 2000 --energy-step 10
```

**Flags:**

| Flag | Default (photon / dE/dx) | Meaning |
|---|---|---|
| `--data-dir` | `data/mu-` / `data/mu-` | Parent dir; the script reads `<data-dir>/<E>MeV/output.root`. |
| `--output` | `output/3d_lookup_table_average` / `output/3d_dedx_lookup_table` | Output dir; `.h5` lands inside. |
| `--energy-min` | 100 / 10 | Lowest energy point to include. |
| `--energy-max` | 2000 / 2000 | Highest energy point to include. |
| `--energy-step` | 10 / 10 | Step size; missing energies are warned-and-skipped. |
| `--visualize` | off | Saves a 6-panel PNG sanity plot. |
| `--visualize-energy <MeV>` (photon only) | off | Saves a focused PDF slice + comparison plot. |

## Stage 2 output schema

`photon_lookup_table.h5` (photon variant) — `format_version = 1.1`:

```
photon_lookup_table.h5
├── data/
│   ├── photon_table_raw       (n_energies, 500, 500)  raw photon counts
│   └── photon_table_average   (n_energies, 500, 500)  raw / events_per_file
├── coordinates/
│   ├── energy_values          (n_energies,)            integer MeV
│   ├── energy_edges / energy_centers
│   ├── angle_edges            (501,) float64           radians (0 to π)
│   ├── angle_centers          (500,)
│   ├── distance_edges         (501,) float64           mm (0 to 10000)
│   └── distance_centers       (500,)
└── metadata/
    ├── attrs: energy_{min,max,step}, angle_bins=500, distance_bins=500,
    │          angle_range_{min,max} (rad), distance_range_{min,max} (mm),
    │          table_shape, total_photons,
    │          normalization='average', average_units='photons/event',
    │          photonsim_version, format_version
    └── events_per_file        structured (energy: i4, events: i4)
```

`dedx_lookup_table.h5` (dE/dx variant) — same structure with substitutions:

| Photon key | dE/dx key |
|---|---|
| `data/photon_table_raw` | `data/dedx_table_raw` |
| `data/photon_table_average` | `data/dedx_table_average` |
| `coordinates/angle_edges/centers` | `coordinates/dedx_edges/centers` |
| `metadata.attrs/total_photons` | `metadata.attrs/total_entries` |
| `metadata.attrs/average_units = 'photons/event'` | `'entries/event'` |
| (no `data_type` attr) | `metadata.attrs/data_type = 'dedx'` |

Bin ranges differ: dE/dx axis is `0..1000 keV/mm` (vs angle `0..π rad`).

## Stage 3 — hand off to SIREN

Drop the .h5 where the trainer expects it:

```
LUCiD/data/<material>/<particle>/photon_lookup_table.h5
e.g. LUCiD/data/water/mu-/photon_lookup_table.h5
```

Then:

```bash
lucid-train-siren --material water --particle muon
```

`PhotonSimDataset` (`lucid/siren/training/dataset.py`) auto-detects whether `data/photon_table_average` or `data/dedx_table_average` is present and picks the right interpolation grid. See [`lucid/siren/README.md`](../lucid/siren/README.md) for the full training/validation walkthrough.

## Optional: pre-sample to .npy

`lucid/siren/training/photonsim_data/create_siren_dataset.py` samples the .h5 once into `train_inputs.npy` / `train_outputs.npy` for repeated training runs without paying the table-load cost each time. **Caveat:** it currently expects the deprecated `data/photon_table_density` schema produced by an older PhotonSim script no longer in the tree, and won't run unmodified against the average schema written above. Direct .h5 consumption via `PhotonSimDataset` is the recommended path.

## Troubleshooting

- **"File not found for <E> MeV"** — the script silently skips missing energy points. Run `check_root_files.py` to find empty/corrupt files. The build still succeeds, but the affected energy slices in the output table are zero.
- **`--energy-min` mismatch with the JSON config.** The mu- and e- configs scan from 10 MeV (`start_MeV: 10`), but `lucid-build-photon-table` defaults to `--energy-min 100`. Pass `--energy-min 10` if you want everything from 10 MeV in the table.
- **`PhotonHist_AngleDistance` missing from a ROOT file.** That file was probably produced with `store_individual_photons: true`, which writes the per-photon arrays but skips the histogram. Re-run with the `water_lookup_table_*.json` configs (or any other config with `store_individual_photons: false`).
- **dE/dx variant looking for `output_job_*.root`.** Unlike the photon variant, `lucid-build-dedx-table` falls back to `output_job_*.root` and `*.root` patterns when `output.root` is missing. Useful for outputs from S3DF SLURM array jobs that name files per-job-id.

## Related

- [`docs/QUICKSTART_S3DF.md`](QUICKSTART_S3DF.md) — running PhotonSim under SLURM.
- [`lucid/production/s3df_jobs/DiffSimInputs_README.md`](../lucid/production/s3df_jobs/DiffSimInputs_README.md) — Stage 1 details.
- [`lucid/siren/README.md`](../lucid/siren/README.md) — Stage 3 (training) details.
- [`docs/LUCID_DATASET.md`](LUCID_DATASET.md) — separate v3 HDF5 dataset (the four `wc_*_NNNN.h5` files); not the same artifact as `photon_lookup_table.h5`.
