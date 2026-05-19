# s_max parametrisation scan (Stage 0)

Cluster fan-out (SLURM or HTCondor — see
[`../../../../docs/CLUSTER_ABSTRACTION.md`](../../../../docs/CLUSTER_ABSTRACTION.md))
for PhotonSim jobs whose only purpose is to fill `PhotonHist_Distance`
per (particle, energy) cell, so the host-native
`PhotonSim/tools/smax/analyze_smax.py` can fit `s_max(E) ≈ A · E^B` per
particle. The resulting `smax_fit.csv` lives at
`PhotonSim/data/<material>/<particle>/smax_fit.csv` and is the upstream
input that `../siren_inputs/generate_jobs.py` reads at submission time.

`/output/smax` is intentionally NOT emitted here — at this stage we are
*computing* `s_max`, not consuming it.

## Quick start

```bash
# 1. Configure your paths (shared across all stages)
cp ../user_paths.s3df.sh.template   ../user_paths.sh   # on S3DF, or
cp ../user_paths.lxplus.sh.template ../user_paths.sh   # on LXPLUS
vim ../user_paths.sh

# 2. Prepare submit scripts only (no submission)
python3 generate_jobs.py -c configs/water_mu.json

# 3. Submit the full scan (mu- + e- separately)
python3 generate_jobs.py -c configs/water_mu.json -s
python3 generate_jobs.py -c configs/water_el.json -s

# 4. Smoke test (4 energies, 50 events each)
python3 generate_jobs.py -c configs/water_mu_test.json -s

# 5. Once all jobs finish, merge per-cell output_job_*.root → photonsim.root
./merge.sh $OUTPUT_BASE_PATH

# 6. Fit s_max(E) and write the CSVs
./analyze.sh $OUTPUT_BASE_PATH
```

`analyze.sh` calls `PhotonSim/tools/smax/analyze_smax.py` inside the
container (uses uproot/numpy/matplotlib) against the SLURM output tree
and writes:

```
<PHOTONSIM_DEV_PATH>/data/<material>/<particle>/smax_data.csv
<PHOTONSIM_DEV_PATH>/data/<material>/<particle>/smax_fit.csv
```

`smax_fit.csv` is what `../siren_inputs/generate_jobs.py` reads when
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
`output_job_NNNNNN.root`, ready for `merge.sh` to `hadd` into
`photonsim.root`.

Alternative — provide a flat per-cell count instead of a schedule:

```json
{ ... "n_events_per_job": 50 }   // overrides events_schedule
```

The smoke-test config `water_mu_test.json` uses this flat form.

## Output layout

For cells with one job (default):

```
<OUTPUT_BASE>/<material>/<particle>/<E>MeV/
  ├── photonsim.root         # contains PhotonHist_Distance (1D, mm)
  ├── photonsim_config.json
  ├── submit.sbatch
  ├── job-001-<slurmid>.out / .err
  └── job_000001.mac
```

For cells split into N jobs (E > `split_above_MeV` in the e- config):

```
<OUTPUT_BASE>/<material>/<particle>/<E>MeV/
  ├── photonsim.root          # created by merge.sh (hadd of all N)
  ├── photonsim_config.json
  ├── submit_job_001.sbatch ... submit_job_NNN.sbatch
  ├── job-001-*.out/.err ... job-NNN-*.out/.err
  ├── output_job_000001.root ... output_job_NNNNNN.root   (deleted after merge)
  └── job_000001.mac ... job_NNNNNN.mac
```

## Notes on stats budget

| E (MeV) | events | E (MeV) | events |
|--------:|-------:|--------:|-------:|
|     100 |   5000 |    2750 |   1818 |
|     125 |   5000 |    3500 |   1429 |
|     150 |   5000 |    4250 |   1176 |
|     175 |   5000 |    5000 |   1000 |
|     200 |   5000 |    6250 |    800 |
|     275 |   5000 |    7500 |    667 |
|     350 |   5000 |    8750 |    571 |
|     425 |   5000 |   10000 |    500 |
|     500 |   5000 |   20000 |    250 |
|     625 |   5000 |   30000 |    167 |
|     750 |   5000 |   40000 |    125 |
|     875 |   5000 |   50000 |    100 |
|    1000 |   5000 |         |        |
|    1250 |   4000 |         |        |
|    1500 |   3333 |         |        |
|    1750 |   2857 |         |        |
|    2000 |   2500 |         |        |

Total: ~81 k events per particle, ~162 k across mu- + e-.

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
