# SIREN-input job submission (s/s_max axis)

SLURM fan-out for PhotonSim jobs that produce SIREN training inputs with
the **`PhotonHist_AngleDistanceNorm`** histogram (opening angle vs.
`s / s_max`). Distinct from the old data-production path
(`../jobs/generate_jobs.sh`) in three ways:

| | Old path (`s3df_jobs/jobs/`) | New SIREN-input path (this dir) |
|---|---|---|
| Macro direction | `/gun/randomDirection true` | `/gun/direction 0 0 1` (matches the hardcoded +Z axis used by PhotonSim's angle histograms) |
| `/output/smax` | not emitted | emitted per-cell, `s_max = A·E^B` from `PhotonSim/data/<material>/<particle>/smax_fit.csv` |
| Output layout | `<OUT>/<detector>/config_NNNNNN/<E>MeV/output_job_NNNNNN.root` | `<OUT>/<material>/<particle>/<E>MeV/photonsim.root` — drop-in for `PhotonSim/tools/siren_inputs/plot_norm_hists.py` |

Both paths share the same `lucid-run-job` core and unified container.

## Quick start

```bash
# 1. Configure your paths (one-time, shared with the old path)
cp ../user_paths.sh.template ../user_paths.sh
vim ../user_paths.sh

# 2. Prepare sbatch scripts only (no submission)
python3 generate_jobs.py -c configs/water_mu_test.json

# 3. Submit
python3 generate_jobs.py -c configs/water_mu_test.json -s

# 4. (Test mode — only the first cell)
python3 generate_jobs.py -c configs/water_mu_test.json -s -t
```

## Config schema

```json
{
  "name": "siren_water_mu_test",
  "description": "...",
  "material": "water",
  "particles": [{"type": "mu-"}],
  "energy_list_MeV": [500, 1000, 2000, 5000],
  "n_events_per_job": 10,
  "include_extrapolated": false
}
```

- `energy_list_MeV` is an explicit non-uniform list (no scan start/stop/step).
- `include_extrapolated: true` lets you process energies below the
  parametrisation's `fit_min_mev`; otherwise those cells are skipped with
  a warning. Override at the CLI with `--include-extrapolated`.
- One job per (particle, energy) cell — `n_events_per_job` is the total
  per-cell statistics.

## Output

```
<OUTPUT_BASE>/<material>/<particle>/<E>MeV/
  ├── photonsim.root        # PhotonHist_AngleDistance(Norm), dEdxHist_Distance, ...
  ├── photonsim_config.json # per-cell dataprod config consumed by lucid-run-job
  ├── submit.sbatch
  ├── job-<slurmid>.out
  ├── job-<slurmid>.err
  └── job_000001.mac        # G4 macro that ran inside the container
```

## Plotting

Once cells finish, point the existing PhotonSim grid plotter at the
output:

```bash
# Single-file 2D panel (one PNG per energy, all TH2Ds side-by-side)
python3 <PhotonSim>/tools/plot_2d_hists.py <OUT>/water/mu-/500MeV/photonsim.root

# Multi-energy grid of PhotonHist_AngleDistanceNorm (s/s_max axis)
python3 <PhotonSim>/tools/siren_inputs/plot_norm_hists.py \
    --input-dir <OUTPUT_BASE> \
    --particles mu- \
    --materials water
```

The grid plotter writes
`angle_distance_norm_<material>_<particle>.png` — all energies in one
figure.

## Downstream: building the .h5 lookup table

`lucid-build-photon-table` consumes the per-energy ROOTs into a single
.h5 keyed by energy. See
[`../../../../docs/SIREN_TRAINING_INPUTS.md`](../../../../docs/SIREN_TRAINING_INPUTS.md)
for the schema and the post-build training step.
