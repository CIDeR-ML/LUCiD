# s_max parametrisation scan (Stage 0)

Cluster fan-out (SLURM or HTCondor — see
[`../../../../docs/guides/production/cluster-abstraction.md`](../../../../docs/guides/production/cluster-abstraction.md))
for PhotonSim jobs whose only purpose is to fill `PhotonHist_Distance`
per (particle, energy) cell, so the host-native
`PhotonSim/tools/smax/analyze_smax.py` can fit `s_max(E)` per particle. The
fit has one of three forms, recorded in the CSV's `form` column: `A*E^B`,
`smooth_two_power` or `piecewise` (the parameters of each are in
`SMAX_PARAMS`, `lucid/siren/emitter.py`). The resulting `smax_fit.csv` lives at
`PhotonSim/data/<material>/<particle>/smax_fit.csv` and is the upstream
input that `../generate_jobs.py` reads at submission time.

`/output/smax` is intentionally NOT emitted here — at this stage we are
*computing* `s_max`, not consuming it.

## Quick start

```bash
# 1. Configure your paths (shared across all stages)
cp ../../platform/templates/user_paths.s3df.sh.template   ../../platform/user_paths.sh   # on S3DF, or
cp ../../platform/templates/user_paths.lxplus.sh.template ../../platform/user_paths.sh   # on LXPLUS
vim ../../platform/user_paths.sh

# The generator runs inside the LUCiD container; -s submits from the host.
RUN=../../platform/run_in_container.sh

# 2. Prepare submit scripts only (no submission)
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_mu.json

# 3. Submit the full scan (mu- + e- separately)
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_mu.json -s
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_el.json -s

# 4. Smoke test (4 energies, 50 events each)
$RUN lucid.production.siren_inputs.smax.generate_jobs -c configs/water_mu_test.json -s

# 5. Once all jobs finish, merge per-cell output_job_*.root → photonsim.root
../merge.sh $SIREN_OUTPUT_BASE_PATH/smax_parametrization

# 6. Fit s_max(E) and write the CSVs
./analyze.sh $SIREN_OUTPUT_BASE_PATH/smax_parametrization
```

`analyze.sh` calls `PhotonSim/tools/smax/analyze_smax.py` inside the
container (uses uproot/numpy/matplotlib) against the SLURM output tree
and writes:

```
<PHOTONSIM_DEV_PATH>/data/<material>/<particle>/smax_data.csv
<PHOTONSIM_DEV_PATH>/data/<material>/<particle>/smax_fit.csv
```

`smax_fit.csv` is what `../generate_jobs.py` reads when
generating Stage-1 SIREN inputs.

## Config schema

```json
{
  "name": "smax_water_mu",
  "material": "water",
  "particles": [{"type": "mu-"}],
  "energy_list_MeV": [100, 125, 150, ...],
  "events_schedule": {
    "base":       5000,
    "anchor_MeV": 1000,
    "floor":      100,
    "split_above_MeV":        10000,
    "target_events_per_job":  100
  }
}
```

The schedule is the schedule from `scan_smax.py`: `base` events at and
below `anchor_MeV`, halved per doubling above (rounded to int), never
below `floor`. The default `configs/water_mu.json` uses 10× the
multipliers that produced the existing `smax_data.csv` (34-energy grid
up to 100 GeV, base=5000, floor=100). `configs/water_el.json` uses a
heavier schedule (base=10000, floor=1000) plus job splitting because
high-energy electron showers are expensive — `split_above_MeV` +
`target_events_per_job` fan a cell out into `ceil(n_events/target)`
SLURM jobs that all write into the same cell directory as
`output_job_NNNNNN.root`, ready for `../merge.sh` to `hadd` into
`photonsim.root`.

Alternative — provide a flat per-cell count instead of a schedule:

```json
{ ... "n_events_per_job": 50 }   // overrides events_schedule
```

The smoke-test config `water_mu_test.json` uses this flat form.

## Output layout

For cells with one job (default):

```
<SIREN_OUTPUT_BASE>/smax_parametrization/<material>/<particle>/<E>MeV/
  ├── photonsim.root         # contains PhotonHist_Distance (1D, mm)
  ├── photonsim_config.json
  ├── submit.sbatch
  ├── job-001-<slurmid>.out / .err
  └── job_000001.mac
```

For cells split into N jobs (E > `split_above_MeV` in the e- config):

```
<SIREN_OUTPUT_BASE>/smax_parametrization/<material>/<particle>/<E>MeV/
  ├── photonsim.root          # created by ../merge.sh (hadd of all N)
  ├── photonsim_config.json
  ├── submit_job_001.sbatch ... submit_job_NNN.sbatch
  ├── job-001-*.out/.err ... job-NNN-*.out/.err
  ├── output_job_000001.root ... output_job_NNNNNN.root   (deleted after merge)
  └── job_000001.mac ... job_NNNNNN.mac
```

## Notes on stats budget

Each cell's event count is `smax_planning.events_for` applied to the config's
schedule. Summed over the 34-energy grids (100 MeV to 100 GeV): `water_mu.json`
86.8 k events, `water_el.json` 180 k; `water_el_lowE.json` adds 270 k
(27 energies below 100 MeV at 10 k each). Recompute after editing a schedule.

## Pipeline placement

```
Stage 0 (this dir)        Stage 1                  Stage 2
─────────────────         ─────────────────        ─────────────────
generate_jobs.py    ──►   ../siren_inputs/   ──►   lucid-build-photon-table
analyze.sh                generate_jobs.py         (lucid/siren/training/
                                                    photonsim_data/)
smax_fit.csv        ──►   reads CSV, bakes
                          /output/smax into
                          each per-cell macro
```
