# SIREN training inputs

Cluster fan-out for the PhotonSim scans that SIREN emitters are trained on. Each cell of a scan
is one (particle, energy) and fills the `PhotonHist_AngleDistanceNorm` histogram (opening angle
vs `s / s_max`). The same code runs on Slurm (S3DF, NERSC) and HTCondor (LXPLUS); the target is
chosen by the `user_paths.sh` copied into `../platform/`. The whole pipeline, from the s_max scan
to a trained emitter, is in
[SIREN training inputs](../../../docs/guides/production/siren-training-inputs.md); cluster
setup is in [cluster abstraction](../../../docs/guides/production/cluster-abstraction.md) and
[clusters](../../../docs/guides/production/clusters.md).

| stage | where |
|---|---|
| s_max(E) scan and fit | [`smax/`](smax/README.md) |
| training-input scan (this directory) | `generate_jobs.py`, `configs/` |
| lookup table | `lucid-build-photon-table`, `lucid-build-dedx-table` |
| training sweeps | [`train_siren/`](train_siren/README.md) |

Compared with dataset production (`../dataprod/`), these jobs fire the particle along +z
(`/gun/direction 0 0 1`, the axis PhotonSim's angle histograms assume), set `/output/smax` per
cell from the fit in `PhotonSim/data/<material>/<particle>/smax_fit.csv`, run no LUCiD
simulation, and write `<OUT>/<material>/<particle>/<E>MeV/photonsim.root`. Both use
`lucid-run-job` and the same container.

## Quick start

```bash
# 1. Paths (shared by all stages)
cp ../platform/templates/user_paths.s3df.sh.template ../platform/user_paths.sh   # or .lxplus, .nersc
vim ../platform/user_paths.sh

# The generator runs in the container; -s submits from the host.
RUN=../platform/run_in_container.sh

# 2. Prepare the smoke test without submitting, then submit it (7 energies, 100 events per cell)
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_mu_test.json
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_mu_test.json -s

# 3. The full scans (10k events per cell, sub-jobs of about an hour)
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_mu.json -s
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -s

# 4. After each drain, list and resubmit failed or preempted sub-jobs; repeat until none are listed
$RUN lucid.production.siren_inputs.resubmit_failed $SIREN_OUTPUT_BASE_PATH/training_inputs --list -s

# 5. Merge each cell's output_job_*.root into photonsim.root
./merge.sh $SIREN_OUTPUT_BASE_PATH/training_inputs
```

A sub-job counts as complete when its `output_job_*.root` has the `OpticalPhotons` TTree key:
PhotonSim writes it only when it finishes, so a preempted job leaves a file without it.
Cells that already have `photonsim.root` are skipped unless `--no-skip-existing` is given.

## Submission modes

`generate_jobs.py` packages the same work three ways; the per-cell outputs are identical, so
`merge.sh` is the same for all:

- default: one submit file per sub-job (`submit.sbatch` or `submit_job_NNN.sbatch`). Recover with
  `resubmit_failed ... --list -s` as above.
- `--array` (Slurm): one job array per cell, the task id as `--job-id`.
- `--pack[=N]` (NERSC): `N` whole-node jobs (default 10), each running its units in parallel
  across the node's cores. Useful when the shared partition is oversubscribed.

For `--array` and `--pack` there are no per-job submit files, so recover with
`./recover.sh -c <config> [--array | --pack [-n N]]`: it deletes outputs that lack the
`OpticalPhotons` key and reruns the generator in the same mode, which resubmits only the missing
units. `--dry-run` lists without deleting.

Other options: `-t` (first cell only), `-o` (output root; default
`$SIREN_OUTPUT_BASE_PATH/training_inputs`, else `OUTPUT_BASE_PATH`), `-P` (partition; a comma
list round-robins), `-g`, `--photonsim-dir`, `--include-extrapolated`, `--min-energy`,
`--max-energy`. See `--help`.

## Config

A smoke-test config gives a flat event count, one job per cell:

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

A full-scan config has an `events_schedule`. Each cell's `events_per_cell` is split into
sub-jobs of about `target_seconds_per_job`, using the time model `t/event = a + b · E_MeV`
(default in `siren_planning.py`, `DEFAULT_TIME_MODEL`):

```json
{
  "name": "siren_water_mu",
  "material": "water",
  "particles": [{"type": "mu-"}],
  "energy_list_MeV": [...],
  "events_schedule": {
    "events_per_cell": 10000,
    "target_seconds_per_job": 3600,
    "time_model": {"a_seconds_per_event": 0.06, "b_seconds_per_event_per_mev": 4.22e-4}
  },
  "include_extrapolated": false
}
```

- `energy_list_MeV` is an explicit list (322 energies up to 100 GeV in `water_mu.json`).
- `events_schedule.events_per_cell_ranges` (optional, used by `water_el.json`) is a list of
  `{"e_max": <MeV>, "events_per_cell": <n>}`; a cell takes the first entry whose `e_max` exceeds
  its energy, else the top-level `events_per_cell`.
- Energies below the s_max fit's `fit_min_mev` are skipped with a warning unless
  `include_extrapolated` is true (or `--include-extrapolated` is given).
- Sub-jobs are balanced (`events_per_job = ceil(events_per_cell / n_jobs)`), so a cell's total
  can slightly exceed `events_per_cell`.

## Output

```
<SIREN_OUTPUT_BASE>/training_inputs/<material>/<particle>/<E>MeV/
  photonsim.root            # the histograms, made by merge.sh from the output_job_*.root
  photonsim_config.json
  submit.sbatch             # submit_job_001.sbatch ... when the cell is split into sub-jobs
  job-001-<jobid>.out/.err  # one pair per sub-job
  output_job_000001.root    # one per sub-job; merge.sh deletes them unless --keep
  job_000001.mac
```

## Next steps

Plot the histograms with PhotonSim's tools:

```bash
python3 <PhotonSim>/tools/plot_2d_hists.py <OUT>/water/mu-/500MeV/photonsim.root
python3 <PhotonSim>/tools/siren_inputs/plot_norm_hists.py \
    --input-dir <OUT> --particles mu- --materials water     # angle_distance_norm_<material>_<particle>.png
```

Then build the lookup table with `lucid-build-photon-table` (or `lucid-build-dedx-table`) and
train ([SIREN training inputs](../../../docs/guides/production/siren-training-inputs.md),
[`train_siren/`](train_siren/README.md)).
