# s_max scan

The first stage of SIREN training: PhotonSim jobs that fill `PhotonHist_Distance` for each
(particle, energy) cell, from which `PhotonSim/tools/smax/analyze_smax.py` fits the track length
`s_max(E)`. The fit is written to `PhotonSim/data/<material>/<particle>/smax_fit.csv`, which the
training-input scan ([`../README.md`](../README.md)) reads. Its `form` column is `A*E^B`,
`smooth_two_power` or `piecewise` (parameters in `SMAX_PARAMS`, `lucid/siren/emitter.py`).
Because this stage computes s_max, its macros do not set `/output/smax`.

Cluster setup (Slurm or HTCondor) is in
[cluster abstraction](../../../../docs/guides/production/cluster-abstraction.md); the whole
pipeline is in [SIREN training inputs](../../../../docs/guides/production/siren-training-inputs.md).

## Quick start

```bash
# 1. Paths (shared by all stages)
cp ../../platform/templates/user_paths.s3df.sh.template ../../platform/user_paths.sh   # or .lxplus
vim ../../platform/user_paths.sh

# The generator runs in the container; -s submits from the host.
RUN=../../platform/run_in_container.sh

# 2. Smoke test: 4 energies, 50 events each
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_mu_test.json -s

# 3. Prepare the full scans (no submission), then submit them
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_mu.json
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_mu.json -s
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_el.json -s

# 4. When all jobs have finished, merge each cell's output_job_*.root into photonsim.root
../merge.sh $SIREN_OUTPUT_BASE_PATH/smax_parametrization

# 5. Fit s_max(E) and write the CSVs
./analyze.sh $SIREN_OUTPUT_BASE_PATH/smax_parametrization
```

`analyze.sh` runs `analyze_smax.py` in the container and writes `smax_data.csv` and
`smax_fit.csv` to `$PHOTONSIM_DEV_PATH/data/<material>/<particle>/` (or to the PhotonSim checkout
given as its second argument).

Generator options: `-t` (first cell only), `-o` (output root; default
`$SIREN_OUTPUT_BASE_PATH/smax_parametrization`), `-P`, `-g`, `--user-paths`,
`--no-skip-existing`.

## Config

```json
{
  "name": "smax_water_mu",
  "material": "water",
  "particles": [{"type": "mu-"}],
  "energy_list_MeV": [100, 125, 150, ...],
  "events_schedule": {
    "base": 5000,
    "anchor_MeV": 1000,
    "floor": 100
  }
}
```

A cell at energy E gets `base` events at or below `anchor_MeV`, and `base · anchor_MeV / E`
(rounded) above it, never fewer than `floor` (`smax_planning.events_for`). Optionally,
`split_above_MeV` and `target_events_per_job` split each cell above that energy into
`ceil(n_events / target)` jobs that all write `output_job_NNNNNN.root` into the cell directory,
for `../merge.sh` to combine. Instead of a schedule, a flat `n_events_per_job` gives every cell
the same count (`water_mu_test.json`).

| config | energies | schedule | events |
|---|---|---|---|
| `water_mu.json` | 34, 100 MeV to 100 GeV | base 5000, floor 100 | 86.8k |
| `water_el.json` | 34, 100 MeV to 100 GeV | base 10000, floor 1000, split above 10 GeV into 100-event jobs | 180k |
| `water_el_lowE.json` | 27, 1 to 90 MeV | as `water_el.json` | 270k |
| `water_mu_test.json` | 4, 100 MeV to 5 GeV | flat 50 | 200 |

## Output

```
<SIREN_OUTPUT_BASE>/smax_parametrization/<material>/<particle>/<E>MeV/
  photonsim.root            # PhotonHist_Distance (1D, mm), made by ../merge.sh
  photonsim_config.json
  submit.sbatch             # submit_job_001.sbatch ... when the cell is split into jobs
  job-001-<jobid>.out/.err  # one pair per job
  output_job_000001.root    # one per job; merge.sh deletes them unless --keep
  job_000001.mac
```
