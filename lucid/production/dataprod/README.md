# Dataset production on a cluster

Fans `lucid-run-job` out over a cluster. Each job runs the whole chain (Geant4 macro, PhotonSim,
LUCiD; sntools first for supernova configs) in one `apptainer exec` of the LUCiD image
([contents](../../../container/README.md)). Slurm (S3DF, NERSC) and HTCondor (LXPLUS) are
supported; the cluster is chosen by the active `../platform/user_paths.sh`. How cluster
targeting works is in [cluster abstraction](../../../docs/guides/production/cluster-abstraction.md);
the image pull and site paths for each cluster are in
[clusters](../../../docs/guides/production/clusters.md).

## Quick start

From this directory:

```bash
# 1. Pull the image (see your cluster's page), then set the paths.
cp ../platform/templates/user_paths.<cluster>.sh.template ../platform/user_paths.sh
${EDITOR:-vim} ../platform/user_paths.sh      # LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, cluster settings

# 2. Prepare one 2-event job per config under <OUT>/<detector>/_test/ (nothing submitted).
./scripts/submit_all_configs.sh -D SK_WAND -t

# 3. Submit every config (the script asks for confirmation).
echo "y" | ./scripts/submit_all_configs.sh -D SK_WAND -s

# 4. Watch the queue.
./scripts/monitor_jobs.slurm.sh -w           # S3DF, NERSC
./scripts/monitor_jobs.htcondor.sh -w        # LXPLUS
```

The `GeV` configs declare no detector and need `-D`; the fan-out refuses a detector that
carries no digitizer and trigger, which among the bundled detectors leaves `SK_WAND` and
`HK_WAND`. The `Solar`, `SN` and `Test` configs declare their own (`HK`), which overrides `-D`.

## Files

| path | purpose |
|---|---|
| `run_job.py` | `lucid-run-job`: one job (macro, PhotonSim, LUCiD, verification) |
| `drivers.py`, `seeds.py` | the event loops and the job's seeds |
| `generators/` | the Geant4 macro, GENIE and sntools steps |
| `fanout.py` | writes the submit scripts for one config |
| `configs/{GeV,SN,Solar,Test}/` | the dataset configs |
| `scripts/submit_all_configs.sh` | `fanout.py` over a glob of configs |
| `scripts/verify_jobs.py` | finds jobs that wrote no complete batch; `--list` to resubmit them |
| `scripts/monitor_jobs.{slurm,htcondor}.sh` | queue watch loops |
| `scripts/cleanup_jobs.sh`, `scripts/clean_output_data.sh` | remove logs, macros, scripts or ROOT files from an output directory |
| `scripts/report_time_performance.py`, `scripts/timing_for_schedule.py`, `scripts/dataprod_metrics_report.py` | timing and metrics reports from job logs |

`../platform/` holds `user_paths.sh` (one template per cluster) and `run_in_container.sh`,
which runs `fanout.py` or `verify_jobs.py` inside the container and submits what they prepare
from the host.

## Dataset configs

A config is `configs/<block>/NN_name.json`: the folder is the block and `NN` the config number.
The blocks, their splits and nominal counts are listed in
[producing datasets](../../../docs/guides/production/index.md). The fields the fan-out reads:

| field | meaning |
|-------|---------|
| `name`, `description` | dataset label (recorded in the HDF5 provenance) and description |
| `material` | `"water"` in every shipped config; the medium itself comes from the detector |
| `detector` | optional; overrides `-D` (the fan-out prints a note) |
| `nominal_train`, `nominal_test` | events per split (`GeV`, `Solar`: a split with 0 is skipped; `SN`, `Test`: one flat dataset of `nominal_test`) |
| `seconds_per_event` | measured PhotonSim + LUCiD wall time per event |
| `target_seconds_per_job` | wall time each job is sized to (default 3600) |
| `primary_source` | `"bomb"`, `"supernova"`, `"genie"`, or absent for a particle gun |
| `particles` / `bomb` / `supernova` | what to generate; a `supernova` block must list `orderings` ([supernova bursts](../README.md#supernova-bursts)) |
| `energy_distribution` | `"uniform"` or `"monoenergetic"` (particle gun) |
| `lucid_seed_base` | optional; separates the LUCiD-side random draws from the Geant4 seed |
| `run_lucid` | `false` stops after PhotonSim |

`lucid-run-job` reads further fields (`selection`, `lucid_options`, `digitizer`, `trigger`,
`cleanup_root_files`, ...). A config without its own `digitizer` and `trigger` takes them from
the detector's physics config; the readout blocks are described in
[digitizer and trigger](../../../docs/reference/digitizer-and-trigger.md).

Jobs per split:

```
events_per_job = floor(target_seconds_per_job / seconds_per_event)
n_jobs         = ceil(nominal / events_per_job)
```

A supernova job is one whole burst (`n_jobs = nominal_test`). A burst in which no interaction
passes the selection is a complete batch of 0 events, which `verify_jobs` counts as done. Train
and test get disjoint master seeds, so they never share an event. `-t` replaces all of this with
one job of 2 events.

## The fan-out

```bash
../platform/run_in_container.sh lucid.production.dataprod.fanout \
    -c <config_json> -D SK_WAND [-s] [-t] [-g] [-P partition] [-o output_base]
```

| option | meaning |
|--------|---------|
| `-c` | dataset config (required) |
| `-D` | detector; required unless the config declares `detector` |
| `-s` | submit (default: prepare only) |
| `-t` | test mode: one job of 2 events, under `<OUT>/<detector>/_test/` |
| `-g` | request 1 GPU per job (otherwise `DEFAULT_GPUS` from `user_paths.sh`) |
| `-P` | partition; a comma list round-robins (below) |
| `-o` | output base (default `OUTPUT_BASE_PATH`) |
| `-j K` | first `job_id` (default 1) |
| `-N M` | run `M` jobs per split |
| `--force-overwrite` | write into a cell that holds submit files from a different config |
| `--user-paths` | the `user_paths.sh` to read (default `../platform/user_paths.sh`) |

Configs with the same number share a cell: `02_mu`, `02_mu_fast`, `02_mu_metrics` and
`02_mu_smoke` all write `config_02/`, and the fan-out refuses a second config in a cell without
`--force-overwrite`.

`scripts/submit_all_configs.sh -D <detector> [-p pattern] [-s] [-t] [-d] [-g] [-P partition] [-o output_base]`
runs the fan-out over every config matching `-p`, a glob under `configs/` matched against
`<block>/<file>` (default `*.json`; e.g. `GeV/*.json`). `-d` prints the commands without running
them; the other flags are passed through. It cannot pass `--force-overwrite`, `-j` or `-N`; call
the fan-out directly for those. With the default glob the three `02_mu_*` variants are refused
after `02_mu` because they share its cell.

### Adding jobs to a dataset

Job `job_id` writes batch `wc_*_<job_id − 1>.h5`, so new jobs join an existing dataset as long
as their ids follow the previous ones:

```bash
N=$(ls <OUTPUT_BASE>/SK_WAND/GeV/test/config_02/submit_job_*.sbatch | wc -l)
../platform/run_in_container.sh lucid.production.dataprod.fanout \
    -c configs/GeV/02_mu.json -D SK_WAND -P milano -j $((N + 1)) -N 5 -s
```

`verify_jobs.py` checks every `submit_job_*` it finds, so a failed job of either wave shows up.

### Several partitions

A comma-separated `SLURM_PARTITION` (or `-P`) spreads jobs over partitions, weighted by an
optional `:N` suffix:

```bash
export SLURM_PARTITION="roma:130,milano:272"
```

Each job is assigned one partition when its script is written, and the fan-out prints the split.
Which partitions to use is cluster-specific; see
[clusters](../../../docs/guides/production/clusters.md).

## Output

The fan-out writes `<OUT>/<detector>/[_test/]<block>/[<split>/]config_NN/`
([details](../../../docs/guides/production/index.md)). Each `config_NN/` is one dataset: the
submit scripts, the job logs (`job_NNNNNN-<jobid>.{out,err}`) and one four-file batch per job
([dataset schema](../../../docs/reference/dataset-schema.md)).

## Monitoring, cleanup and resubmission

```bash
./scripts/monitor_jobs.slurm.sh -w

# Cancel your production jobs (their names start with photonsim_)
squeue -u $USER -h -o '%i %j' | awk '$2 ~ /^photonsim_/ {print $1}' | xargs -r scancel

# Both cleanup scripts only list what they would delete unless given --execute.
./scripts/clean_output_data.sh -d <output_dir> -a --execute     # ROOT and log files; keeps HDF5
./scripts/cleanup_jobs.sh -o <output_dir> -r -a --execute       # logs, macros, submit scripts

# List failed jobs, then resubmit them; repeat until nothing is listed.
../platform/run_in_container.sh lucid.production.dataprod.scripts.verify_jobs \
    <OUTPUT_BASE_PATH>/SK_WAND --list
../platform/run_in_container.sh lucid.production.dataprod.scripts.verify_jobs \
    <OUTPUT_BASE_PATH>/SK_WAND --list -s

# A readable breakdown, without resubmitting
apptainer exec ${LUCID_IMAGE_PATH} python3 ./scripts/verify_jobs.py <OUTPUT_BASE_PATH>/SK_WAND
```

What counts as complete, and the checks after a run, are in
[cluster abstraction](../../../docs/guides/production/cluster-abstraction.md).

## Timing

`report_time_performance.py` reads the per-event timing lines of the job logs and makes
histograms and summary tables:

```bash
python ./scripts/report_time_performance.py --all \
    --base-dir <OUTPUT_BASE_PATH>/SK_WAND --output timing_report.png
```

`timing_for_schedule.py` prints, per config, `seconds_per_event` = median LUCiD per-event time +
PhotonSim elapsed time / n_events. The PhotonSim time includes Geant4 initialisation, so the
value is slightly pessimistic. Put it in the config's `seconds_per_event`.

```bash
python3 ./scripts/timing_for_schedule.py --base-dir <timing_output>/SK_WAND --partition roma
```

## Troubleshooting

- `Error: detector needs to be specified`: pass `-D SK_WAND` (or `HK_WAND`).
- `no 'digitizer' block: the dataset config sets none ...`: the detector (e.g. `SK`, `HK`) has
  no digitizer or trigger; use the `*_WAND` detector, or state the readout in the dataset config.
- `Error: LUCID_IMAGE_PATH not set`: fill in `user_paths.sh`.
- `Error: LUCID_IMAGE_PATH=... does not exist`: the image has not been pulled yet.
- A job failed partway: read its logs in the cell, `config_NN/job_*-*.{out,err}`.
