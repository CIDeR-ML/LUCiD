# Production Jobs (dataprod fan-out)

Cluster fan-out for `lucid-run-job`. The whole pipeline (macro → PhotonSim →
LUCiD for particle-gun and bomb configs; sntools first for supernova) runs
inside the unified LUCiD apptainer image — one `apptainer exec` per job, no
host-side build state. SLURM (S3DF, NERSC) and HTCondor (LXPLUS) are both
supported; the cluster is picked from your active `user_paths.sh`. See
[`docs/guides/production/cluster-abstraction.md`](../../../docs/guides/production/cluster-abstraction.md).

The per-cluster runbook is
[`clusters.md`](../../../docs/guides/production/clusters.md); it holds
the image pull and the site paths.

## Quick start

```bash
# 1. Pull the image and configure paths: see your cluster's deploy page.
cp ../platform/templates/user_paths.<cluster>.sh.template ../platform/user_paths.sh
${EDITOR:-vim} ../platform/user_paths.sh   # LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, cluster vars

# 2. Test with one job per config, 2 events, under <OUT>/<detector>/_test/ (no submission)
./scripts/submit_all_configs.sh -D SK_WAND -t

# 3. Submit all configs
echo "y" | ./scripts/submit_all_configs.sh -D SK_WAND -s

# 4. Monitor
./scripts/monitor_jobs.slurm.sh -w        # on S3DF or NERSC, or
./scripts/monitor_jobs.htcondor.sh -w      # on LXPLUS
```

`-D` is required for the `GeV` configs, which declare no detector; the fan-out
accepts only a frozen `*_WAND` detector for them. The `Solar`, `SN` and `Test`
configs declare their own (`HK`), which overrides `-D`.

## Directory layout

```
dataprod/
├── run_job.py                   # lucid-run-job: one job (macro → PhotonSim → LUCiD → verify)
├── drivers.py, seeds.py         # the event loops; the job's seeds
├── generators/                  # Geant4 macro, GENIE, sntools
├── fanout.py                    # fan one config out into submit scripts
├── configs/{GeV,SN,Solar,Test}/ # dataset configs, NN_name.json
└── scripts/
    ├── submit_all_configs.sh        # fanout.py over a glob of configs
    ├── verify_jobs.py               # flag sub-jobs that wrote no complete batch; --list to resubmit
    ├── monitor_jobs.slurm.sh        # SLURM watch loop
    ├── monitor_jobs.htcondor.sh     # HTCondor watch loop
    ├── cleanup_jobs.sh              # drop logs/macros/scripts (and -x: HDF5) under one dir
    ├── clean_output_data.sh         # drop ROOT/log files from a finished output dir
    └── report_time_performance.py, timing_for_schedule.py, dataprod_metrics_report.py
```

The cluster env block (`user_paths.sh`, one template per cluster) and
`run_in_container.sh`, which runs `fanout.py` and `verify_jobs.py` inside the
container and submits from the host, live in `../platform/`
([how cluster targeting works](../../../docs/guides/production/cluster-abstraction.md)).
What the image contains is in [`container/README.md`](../../../container/README.md).

## Dataset configs

A config lives at `configs/<block>/NN_name.json`; the block is the folder and
`NN` (the filename prefix) is the config number. The blocks, their splits and
nominal counts are listed once, in
[Producing datasets](../../../docs/guides/production/index.md#config-blocks); the
JSONs are the source. The fields the fan-out reads:

| Field | Meaning |
|-------|---------|
| `name`, `description` | dataset label (recorded in the HDF5 provenance) and prose |
| `material` | `"water"` for every shipped config |
| `detector` | optional; when present it wins over `-D` (the fan-out prints a note) |
| `nominal_train`, `nominal_test` | events per split (`GeV`, `Solar`: a split with `0` is skipped; `SN`, `Test`: flat, from `nominal_test`) |
| `seconds_per_event` | measured PhotonSim + LUCiD wall time per event |
| `target_seconds_per_job` | wall time each job is sized to (default 3600) |
| `primary_source` | `"bomb"`, `"supernova"`, `"genie"`, or absent for a particle gun |
| `particles` / `bomb` / `supernova` | what to generate; a `supernova` block must list its mass orderings in `orderings` (no default; see [the supernova section](../README.md#supernova-bursts-sntools)) |
| `energy_distribution` | `"uniform"` or `"monoenergetic"` (particle gun) |
| `lucid_seed_base` | optional; decouples the LUCiD-side draws from the Geant4 seed |
| `run_lucid` | `false` stops after PhotonSim (`--skip-lucid`) |

`lucid-run-job` itself reads further fields (`selection`, `lucid_options`,
`digitizer`/`trigger`, `cleanup_root_files`, ...). A config without its own
`digitizer` and `trigger` takes them from the detector, which must carry them
(the `*_WAND` ones do); the fan-out refuses one that does not. See
[`digitizer-and-trigger.md`](../../../docs/reference/digitizer-and-trigger.md) for the
readout blocks.

Job sizing per split:

```
events_per_job = floor(target_seconds_per_job / seconds_per_event)
n_jobs         = ceil(nominal / events_per_job)
```

Supernova is one all-at-once burst per job (`n_jobs = nominal_test`); a burst
in which no interaction passes the selection is a complete batch of 0 events
(`n_events_requested = 1`), which `verify_jobs` counts as done. Train and
test get disjoint master seeds, so the two datasets never share an event.
`-t` overrides all of this with one job of 2 events.

## Job generation

```bash
../platform/run_in_container.sh lucid.production.dataprod.fanout \
    -c <config_json> -D SK_WAND [-s] [-t] [-g] [-P partition] [-o output_base]
```

| Option | Meaning |
|--------|---------|
| `-c` | dataset config (required) |
| `-D` | detector; required unless the config declares `detector` |
| `-s` | submit (default: prepare only) |
| `-t` | test mode: one job, 2 events, under `<OUT>/<detector>/_test/` |
| `-g` | request 1 GPU per job (otherwise `DEFAULT_GPUS` from `user_paths.sh`) |
| `-P` | partition override; a comma list round-robins (below) |
| `-o` | output base override (default `OUTPUT_BASE_PATH`) |
| `-j K` | first `job_id` (default 1); see "Topping up" |
| `-N M` | force `n_jobs = M` per split |
| `--force-overwrite` | write into a cell that holds submit files from a different config |
| `--user-paths` | `user_paths.sh` to read (default `../platform/user_paths.sh`) |

Configs sharing a number share a cell: `02_mu`, `02_mu_fast`, `02_mu_metrics`
and `02_mu_smoke` all write `config_02/`, and the fan-out refuses the second
one into a cell unless given `--force-overwrite`.

### `submit_all_configs.sh`

Runs the fan-out over every config matching a glob.

```bash
./scripts/submit_all_configs.sh -D SK_WAND [-p pattern] [-s] [-t] [-d] [-g] [-P partition] [-o output_base]
```

| Flag | Meaning |
|------|---------|
| `-p` | glob under `configs/`, matched against `<block>/<file>` (default `*.json` = all; e.g. `GeV/*.json`) |
| `-D` | detector, passed to the fan-out |
| `-s` | submit (default: prepare only) |
| `-t` | test mode (1 job of 2 events per config and split) |
| `-d` | dry run: print the fan-out commands |
| `-g` | request GPU |
| `-P` | partition override |
| `-o` | output base override |

It cannot pass `--force-overwrite`, `-j` or `-N`; call
the fan-out directly for those. With the default glob the three `02_mu_*`
variants are refused after `02_mu` (shared cell, above); that is expected.

### Topping up an existing dataset

Each sub-job writes a `wc_*_<file_index>.h5` batch where `file_index =
job_id − 1`, so new sub-jobs join the existing tree as long as their `job_id`
does not collide with the previous wave:

```bash
# Count what's already there
N=$(ls <OUTPUT_BASE>/SK_WAND/GeV/test/config_02/submit_job_*.sbatch | wc -l)

# Add 5 more sub-jobs of the same size
../platform/run_in_container.sh lucid.production.dataprod.fanout \
    -c configs/GeV/02_mu.json -D SK_WAND -P milano -j $((N + 1)) -N 5 -s
```

`verify_jobs.py` checks every `submit_job_*` it finds, so a preempted sub-job
from either wave shows up the same way.

### Spreading across partitions (weighted round-robin)

A comma-separated `SLURM_PARTITION` (or `-P`) round-robins sub-jobs across
partitions, weighted by an optional `:N` suffix; a single value targets one
partition:

```bash
export SLURM_PARTITION="roma:130,milano:272"   # weight by node count
```

Each sub-job is stamped with one partition at generation time, and the fan-out
prints the resulting split. Which partitions to use and why the choice is
per-job are cluster-specific; see
[S3DF](../../../docs/guides/production/clusters.md#s3df-slac-slurm).

## Output layout

The tree the fan-out writes is described once, in
[Producing datasets](../../../docs/guides/production/index.md#where-a-batch-run-writes):
`<OUT>/<detector>/[_test/]<block>/[<split>/]config_NN/`. Each `config_NN/` cell
is one LUCiD **dataset**, holding the submit scripts, the job logs
(`job_NNNNNN-<jobid>.{out,err}`) and one four-file batch per job; the batch
layout is in [`dataset-schema.md`](../../../docs/reference/dataset-schema.md).

## Monitoring & cleanup

```bash
# Watch the SLURM queue
./scripts/monitor_jobs.slurm.sh -w

# Cancel my production jobs (names start with photonsim_; scancel -n matches whole names)
squeue -u $USER -h -o '%i %j' | awk '$2 ~ /^photonsim_/ {print $1}' | xargs -r scancel

# Wipe ROOT + log files from a finished output dir (keeps HDF5)
./scripts/clean_output_data.sh -d <output_dir> -a --execute

# Drop a config dir's logs, macros and submit scripts (keeps ROOT and HDF5).
# Both scripts are dry-run by default: nothing goes without --execute.
./scripts/cleanup_jobs.sh -o <output_dir> -r -a --execute
```

## Finding and resubmitting failed jobs

What counts as complete, and the post-run checklist, are in
[`cluster-abstraction.md`](../../../docs/guides/production/cluster-abstraction.md#verifying-and-resuming-cluster-agnostic).
The commands:

```bash
# List what's broken
../platform/run_in_container.sh lucid.production.dataprod.scripts.verify_jobs \
    <OUTPUT_BASE_PATH>/SK_WAND --list

# Resubmit it (idempotent; rerun after each drain wave until nothing is listed)
../platform/run_in_container.sh lucid.production.dataprod.scripts.verify_jobs \
    <OUTPUT_BASE_PATH>/SK_WAND --list -s

# Human-readable breakdown, no resubmission
apptainer exec ${LUCID_IMAGE_PATH} \
    python3 ./scripts/verify_jobs.py <OUTPUT_BASE_PATH>/SK_WAND
```

## Per-event timing

`report_time_performance.py` parses the jobs' `*.out` files for the LUCiD
per-event timing line and produces histograms / summary tables:

```bash
python ./scripts/report_time_performance.py --all \
    --base-dir <OUTPUT_BASE_PATH>/SK_WAND \
    --output timing_report.png
```

`timing_for_schedule.py` walks the cells under a detector directory and prints,
per config, `seconds_per_event` = median LUCiD per-event time + PhotonSim
elapsed / n_events (slightly pessimistic: the PhotonSim time includes Geant4
init). Put that value in the config's `seconds_per_event`. (It also prints an
`events_schedule` block, which the fan-out does not read.)

```bash
python3 ./scripts/timing_for_schedule.py \
    --base-dir <timing_output>/SK_WAND \
    --partition roma
```

The current per-config values are the `seconds_per_event` fields of the JSONs.

## Troubleshooting

- **`Error: detector needs to be specified`** → pass `-D SK_WAND` (or `HK_WAND`).
- **`no 'digitizer' block: the dataset config sets none ...`** → `-D SK`/`-D HK` carry no
  digitizer or trigger; use the `*_WAND` name, or state the readout in the dataset config.
- **`Error: LUCID_IMAGE_PATH not set`** → fill in `user_paths.sh`.
- **`Error: LUCID_IMAGE_PATH=... does not exist`** → the `.sif` has not been
  pulled yet; see your cluster's deploy page.
- **Job failed somewhere mid-pipeline** → read the logs in the cell:
  `config_NN/job_*-*.{out,err}`.
