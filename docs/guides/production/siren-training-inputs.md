# SIREN training inputs

How PhotonSim simulation output becomes a `photon_lookup_table.h5` (or
`dedx_lookup_table.h5`) that the SIREN trainer consumes.

The third SIREN input axis is now **`s / s_max(E)` ∈ [0, 1]** rather than
absolute distance in mm. `s_max(E) ≈ A · E^B` is fit per (material, particle);
the fit travels embedded in the .h5 metadata so downstream LUCiD code never has
to re-read it from PhotonSim. The normalised axis is what lets the training
energy range run up to 100 GeV (the old absolute-s axis capped out near 2 GeV).

## Pipeline

```
  Stage 0           Stage 1                Stage 2                Stage 3
  ──────────        ─────────────          ─────────────────      ──────────────
  smax/   ────►     siren_inputs/   ────►  lucid-build-*-table  ──►  lucid-train-siren
  smax_fit.csv      photonsim.root         photon_lookup_table.h5     trained model
                    (one per cell)         dedx_lookup_table.h5
```

Stage outputs:

| Stage | Tool | Output |
|---|---|---|
| 0. Fit `s_max(E)` | `lucid/production/jobs/smax/` + `PhotonSim/tools/smax/analyze_smax.py` | `PhotonSim/data/<material>/<particle>/smax_fit.csv` (one-row CSV with `A, B, fit_min_mev, fit_max_mev, quantile, quantile_multiplier, generated_at_utc`) |
| 1. Per-cell sim | `lucid/production/jobs/siren_inputs/` (SLURM or HTCondor — see `cluster-abstraction.md`) | `<OUT>/<material>/<particle>/<E>MeV/photonsim.root` for each cell |
| 2. Aggregate | `lucid-build-photon-table` / `lucid-build-dedx-table` | `photon_lookup_table.h5` / `dedx_lookup_table.h5` |
| 3. Train | `lucid-train-siren` | SIREN weights |

The Stage 1 macro emits `/output/smax <smax_mm> mm`, which makes PhotonSim
book the two normalised histograms `PhotonHist_AngleDistanceNorm` (opening
angle × `s/s_max`, 500×500) and `dEdxHist_DistanceNorm` (dE/dx × `s/s_max`,
500×500). Photons with `s/s_max > 1` land in ROOT's overflow bin.

Stage 0 details: `lucid/production/jobs/smax/README.md`.
Stage 1 details: `lucid/production/jobs/siren_inputs/README.md`.

## Stage 2 — building the .h5

```bash
# Cherenkov (opening angle vs s/s_max)
lucid-build-photon-table \
    --data-dir   <OUT> \
    --material   water --particle mu- \
    --output     LUCiD/data/water/mu-/photon_lookup_table.h5

# Energy loss (dE/dx vs s/s_max)
lucid-build-dedx-table \
    --data-dir   <OUT> \
    --material   water --particle mu- \
    --output     LUCiD/data/water/mu-/dedx_lookup_table.h5
```

Both share the same CLI:

| Flag | Meaning |
|---|---|
| `--data-dir` | Root containing `<material>/<particle>/<E>MeV/photonsim.root`. |
| `--material` / `--particle` | Identify the cell tree and pick the right `smax_fit.csv`. |
| `--output` | Output `.h5` (file or directory). |
| `--photonsim-dir` | Where to find `smax_fit.csv` (default: `$PHOTONSIM_DEV_PATH` or a sibling of `--data-dir`). |
| `--require-list <cfg.json>` | Assert the config's `energy_list_MeV` is a subset of on-disk cells. |
| `--no-validate` | Skip the post-write self-check. |

Energies are auto-discovered by walking `<E>MeV` subdirectories — the on-disk
set is the truth and no `--energy-min/max/step` is needed.

After write, the builder re-opens the .h5 and asserts: required keys present,
`distance_edges ∈ [0, 1]`, `raw.sum() / events == average.sum()` round-trip,
element-wise equality against the source `photonsim.root` for a random cell,
and `smax_*` attrs match `smax_fit.csv` row-for-row. Failures exit non-zero.

## .h5 schema (`format_version = "2.0"`)

```
photon_lookup_table.h5     # or dedx_lookup_table.h5
├── data/
│   ├── photon_table_raw      (n_energies, 500, 500)
│   └── photon_table_average  (n_energies, 500, 500)    raw / events_per_cell
├── coordinates/
│   ├── energy_values         (n_energies,)             integer MeV, non-uniform
│   ├── energy_centers        same as energy_values
│   ├── energy_edges          (n_energies+1,)
│   ├── angle_edges           (501,)                    0..π rad
│   ├── angle_centers         (500,)
│   ├── distance_edges        (501,)                    [0, 1]   (s/s_max)
│   └── distance_centers      (500,)                    [0, 1]
└── metadata/
    ├── attrs:
    │   material, particle, distance_axis="s_over_smax",
    │   format_version="2.0", photonsim_version,
    │   normalization="average", average_units="photons/event",
    │   angle_bins, distance_bins, angle_range_{min,max},
    │   distance_range_{min,max}, table_shape, total_photons,
    │   smax_{A, B, fit_min_mev, fit_max_mev, quantile,
    │         quantile_multiplier, generated_at_utc}
    └── events_per_file       structured (energy: i4, events: i4)
```

dE/dx variant: substitute `photon_table_*` → `dedx_table_*`,
`angle_*` → `dedx_*` (0..1000 keV/mm), `total_photons` → `total_entries`,
`average_units = "entries/event"`, and add `data_type = "dedx"`.

## Stage 3 — training

Drop the .h5 where the trainer expects it:

```
LUCiD/data/<material>/<particle>/photon_lookup_table.h5
```

Then:

```bash
lucid-train-siren --material water --particle muon
```

`PhotonSimDataset` (`lucid/siren/training/dataset.py`) auto-detects whether the
file has `data/photon_table_average` or `data/dedx_table_average` and picks the
right interpolation grid. Full training/validation walkthrough:
[`lucid/siren/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/siren/README.md).

## Troubleshooting

- **`PhotonHist_AngleDistanceNorm` missing in <E>MeV file** — Stage 1 ran
  without `/output/smax` baked into the macro. Re-run that cell with a config
  whose generator sets `smax_mm`.
- **`distance_edges are not [0, 1]`** — same cause as above (the cell only
  has the legacy absolute-s histogram).
- **`--require-list` missing cells** — Stage 1 didn't finish; run
  `lucid/production/jobs/siren_inputs/resubmit_failed.sh` and `merge.sh`
  until the cell count matches.
- **`No s_max fit at PhotonSim/data/<m>/<p>/smax_fit.csv`** — Stage 0 hasn't
  been run for that (material, particle).

## Related

- [`deploy-s3df.md`](deploy-s3df.md) — running PhotonSim under SLURM at SLAC.
- [`deploy-lxplus.md`](deploy-lxplus.md) — running PhotonSim under HTCondor at CERN.
- [`cluster-abstraction.md`](cluster-abstraction.md) — how the same code drives both.
- [`lucid/production/jobs/smax/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/jobs/smax/README.md) — Stage 0.
- [`lucid/production/jobs/siren_inputs/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/jobs/siren_inputs/README.md) — Stage 1.
- [`lucid/siren/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/siren/README.md) — Stage 3.

## Next steps (not yet wired up)

SIREN inference (`lucid/sources/siren_rays.py`, `lucid/siren/core.py`) still
feeds the model `(E, angle, distance_mm)`. Once the trainer consumes the new
`s/s_max` axis, inference must read `smax_A` / `smax_B` from the .h5 metadata
to convert physical `s` → `s/s_max` at sample time. The fit lives in the .h5
specifically so this can happen without re-reading any external CSV.
