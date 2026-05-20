# SIREN-input job submission (s/s_max axis)

Fan-out for PhotonSim jobs that produce SIREN training inputs with the
**`PhotonHist_AngleDistanceNorm`** histogram (opening angle vs.
`s / s_max`). One code path, two clusters: which one you target is
chosen by the `user_paths.sh` you copy into `../`. See
[`../../../../docs/CLUSTER_ABSTRACTION.md`](../../../../docs/CLUSTER_ABSTRACTION.md)
for the architecture and the per-cluster runbooks
[`QUICKSTART_S3DF.md`](../../../../docs/QUICKSTART_S3DF.md) /
[`QUICKSTART_LXPLUS.md`](../../../../docs/QUICKSTART_LXPLUS.md).

Distinct from the data-production path (`../dataprod/generate_jobs.sh`)
in three ways:

| | Dataprod path (`../dataprod/`) | SIREN-input path (this dir) |
|---|---|---|
| Macro direction | `/gun/randomDirection true` | `/gun/direction 0 0 1` (matches the hardcoded +Z axis used by PhotonSim's angle histograms) |
| `/output/smax` | not emitted | emitted per-cell, `s_max = A·E^B` from `PhotonSim/data/<material>/<particle>/smax_fit.csv` |
| Output layout | `<OUT>/<detector>/config_NNNNNN/<E>MeV/output_job_NNNNNN.root` | `<OUT>/<material>/<particle>/<E>MeV/photonsim.root` — drop-in for `PhotonSim/tools/siren_inputs/plot_norm_hists.py` |

Both paths share the same `lucid-run-job` core and unified container.

## Quick start

```bash
# 1. Configure your paths (one-time, shared across stages)
cp ../user_paths.s3df.sh.template   ../user_paths.sh   # on S3DF, or
cp ../user_paths.lxplus.sh.template ../user_paths.sh   # on LXPLUS
vim ../user_paths.sh

# 2. Prepare sbatch scripts only (no submission)
./generate_jobs.sh -c configs/water_mu_test.json

# 3. Submit smoke test (small grid, flat 100 events/cell, 1 job per cell)
./generate_jobs.sh -c configs/water_mu_test.json -s

# 4. Submit the full scan (322-cell grid, 10k events/cell, ~1h/sub-job)
./generate_jobs.sh -c configs/water_mu.json -s
./generate_jobs.sh -c configs/water_el.json -s

# 5. After each drain, recover any preempted/failed sub-jobs (idempotent;
#    repeat until "missing" is 0 — the cluster's `preemptable` QoS on roma
#    means some sub-jobs get bumped):
./resubmit_failed.sh $SIREN_OUTPUT_BASE_PATH/training_inputs

# 6. Once nothing is missing, hadd per-cell sub-job ROOTs:
./merge.sh $SIREN_OUTPUT_BASE_PATH/training_inputs
```

The `resubmit_failed.sh` truth check is the presence of the
`OpticalPhotons` TTree key in each cell's `output_job_*.root` —
preempted PhotonSim jobs leave basket bytes on disk but never reach
`DataManager::Finalize()`, so the TTree directory entry is the
unambiguous "this job ran to completion" marker. The wrapper bridges
the apptainer/host split (uproot lives in the container, sbatch lives
on the host).

## Config schema

Two forms are supported. Smoke-test configs use a flat
`n_events_per_job` at top level — one SLURM job per (particle, energy)
cell:

```json
{
  "name": "siren_water_mu_test",
  "material": "water",
  "particles": [{"type": "mu-"}],
  "energy_list_MeV": [500, 1000, 2000, 5000, 10000, 50000, 100000],
  "n_events_per_job": 100,
  "include_extrapolated": false
}
```

Full-scan configs use an `events_schedule` block. The generator splits
each cell's `events_per_cell` total into multiple SLURM sub-jobs whose
wall time is ≈ `target_seconds_per_job`, using a linear time model
fitted from real v6 runs (`t/event = a + b · E_MeV`, defaults in
`generate_jobs.py`'s `DEFAULT_TIME_MODEL`):

```json
{
  "name": "siren_water_mu",
  "material": "water",
  "particles": [{"type": "mu-"}],
  "energy_list_MeV": [...322 entries up to 100 GeV...],
  "events_schedule": {
    "events_per_cell": 10000,
    "target_seconds_per_job": 3600,
    "time_model": {
      "a_seconds_per_event":         0.06,
      "b_seconds_per_event_per_mev": 4.22e-4
    }
  },
  "include_extrapolated": false
}
```

- `energy_list_MeV` is an explicit non-uniform list (no scan start/stop/step).
- `include_extrapolated: true` lets you process energies below the
  parametrisation's `fit_min_mev`; otherwise those cells are skipped with
  a warning. Override at the CLI with `--include-extrapolated`.
- Splits are balanced (`events_per_job = ceil(events_per_cell / n_jobs)`)
  so the last sub-job may run slightly short of the target — the total
  event count can slightly overshoot `events_per_cell` because of the
  ceiling.

## Output

Single-job cells (every cell in a smoke test, low-E cells in the full scan):

```
<SIREN_OUTPUT_BASE>/training_inputs/<material>/<particle>/<E>MeV/
  ├── photonsim.root        # PhotonHist_AngleDistance(Norm), dEdxHist_Distance, ...
  ├── photonsim_config.json
  ├── submit.sbatch
  ├── job-001-<slurmid>.out / .err
  └── job_000001.mac
```

Multi-job cells (typically high-E cells when an `events_schedule` with
splitting is active):

```
<SIREN_OUTPUT_BASE>/training_inputs/<material>/<particle>/<E>MeV/
  ├── photonsim.root          # created by merge.sh (hadd of all N sub-jobs)
  ├── photonsim_config.json
  ├── submit_job_001.sbatch ... submit_job_NNN.sbatch
  ├── job-001-*.out / .err ... job-NNN-*.out / .err
  ├── output_job_000001.root ... output_job_NNNNNN.root  (deleted after merge)
  └── job_000001.mac ... job_NNNNNN.mac
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
