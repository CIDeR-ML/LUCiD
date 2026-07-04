# Production Jobs (dataprod fan-out)

Cluster fan-out for `lucid-run-job`. The whole pipeline (gevgen → gntpc
→ PhotonSim → LUCiD, or macro → PhotonSim → LUCiD) runs inside the
unified LUCiD apptainer image — one `apptainer exec` per job, no
host-side build state. SLURM (S3DF) and HTCondor (LXPLUS) are both
supported; the cluster is picked from your active `user_paths.sh`. See
[`docs/CLUSTER_ABSTRACTION.md`](../../../docs/CLUSTER_ABSTRACTION.md).

For a higher-level walkthrough see
[`docs/QUICKSTART_S3DF.md`](../../../docs/QUICKSTART_S3DF.md) or
[`docs/QUICKSTART_LXPLUS.md`](../../../docs/QUICKSTART_LXPLUS.md).

## Quick start

```bash
# 1. Pull the unified container (one-time) — see your cluster's QUICKSTART
#    for the location.

# 2. Configure paths
cp user_paths.s3df.sh.template   user_paths.sh   # on S3DF, or
cp user_paths.lxplus.sh.template user_paths.sh   # on LXPLUS
vim user_paths.sh   # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, cluster vars

# 3. Test with one job per config (no submission)
./dataprod/submit_all_configs.sh -t

# 4. Submit all configs
echo "y" | ./dataprod/submit_all_configs.sh -s

# 5. Monitor
./dataprod/monitor_jobs.s3df.sh -w        # on S3DF, or
./dataprod/monitor_jobs.lxplus.sh -w      # on LXPLUS
```

## Directory layout

```
jobs/
├── user_paths.s3df.sh.template      # SLURM env block (copy → user_paths.sh)
├── user_paths.lxplus.sh.template    # HTCondor env block
├── user_paths.sh                    # your overrides (gitignored)
├── dataprod/
│   ├── generate_jobs.sh             # fan one config out into submit scripts
│   ├── submit_all_configs.sh
│   ├── verify_jobs.py               # flag finished sub-jobs that wrote no usable batch
│   ├── resubmit_failed.sh           # re-submit the failures verify_jobs found
│   ├── monitor_jobs.s3df.sh         # squeue-flavoured watch loop
│   ├── monitor_jobs.lxplus.sh       # condor_q-flavoured watch loop
│   ├── cleanup_jobs.sh
│   └── report_time_performance.py
└── utils/
    ├── clean_output_data.sh         # drop ROOT/log files from a finished output dir
    └── copy_output_data.sh
```

## Configuration (`user_paths.sh`)

```bash
# Container (apptainer pull from ghcr.io/cider-ml/lucid:latest)
export LUCID_IMAGE_PATH="/sdf/data/neutrino/<user>/software/images/lucid.sif"

# GENIE splines path *inside* the container.
# Default points at the bundled G18_10a_02_11b spline (matches in-repo configs).
export GENIE_XSEC_FILE="/opt/genie_xsec/3_04_00/G18_10a_02_11b/gxspl-min.xml.gz"

# Output base
export OUTPUT_BASE_PATH="/sdf/data/neutrino/<user>/photonsim_output"

# SLURM
export SLURM_PARTITION="ampere"
export SLURM_ACCOUNT="your-account:your-group"

# Resource defaults
export DEFAULT_CPUS="1"
export DEFAULT_MEMORY="39936"   # MB
export DEFAULT_GPUS="1"
export DEFAULT_TIME="23:00:00"
```

The container ships GEANT4 11.3 + ROOT 6.30 + GENIE 3.04 + PhotonSim
+ the LUCiD Python stack, so there is nothing else to install on
the host.

## Job generation

```bash
./jobs/generate_jobs.sh -c <config_json> [-s] [-t] [-g] [-P partition] [-o output_base]
```

Options:
- `-c` JSON config from `lucid/production/configs/` (required)
- `-s` actually submit to SLURM (default: prepare only)
- `-t` test mode — one job per config, `n_events_per_job=2`, passes `--test` to `lucid-run-job`
- `-g` request 1 GPU per job
- `-P` partition override
- `-o` output base override

### `submit_all_configs.sh`

Wrapper that fans every config in `lucid/production/configs/` (matching
a glob) through `generate_jobs.sh`.

```bash
./jobs/submit_all_configs.sh [-p pattern] [-s] [-t] [-d] [-n n_jobs] [-e events] [-g] [-P partition] [-o output_base]
```

| Flag | Meaning |
|------|---------|
| `-p` | config glob (default `dataprod*.json`) |
| `-s` | submit (default: prepare only) |
| `-t` | test mode (1 job per config) |
| `-d` | dry run |
| `-n` | override `n_jobs` |
| `-e` | override `n_events_per_job` |
| `-g` | request GPU |
| `-P` | partition override |
| `-o` | output base override |

### Config schema

See `lucid/production/configs/dataprod_*.json`. The fan-out script
honors:

| Field | Type | Description |
|-------|------|-------------|
| `config_number` | int | unique ID, used as `config_NNNNNN` output folder |
| `name` | string | dataset label, recorded in HDF5 provenance |
| `material` | string | `"water"` (others later) |
| `primary_source` | string | `"particle_gun"`, `"genie"`, `"bomb"`, or `"supernova"` |
| `energy_distribution` | string | `"uniform"` or `"monoenergetic"` |
| `particles` | array | per-particle type + energy ranges |
| `lucid_options` | object | LUCiD writer flags (smearing, translation, etc.) |
| `n_jobs` | int | number of SLURM jobs (one batch each) |
| `n_events_per_job` | int | events per job |
| `events_schedule` | object | *(optional)* time-based fan-out — see below |

`primary_source: "genie"` causes the runner to chain
gevgen → gntpc → PhotonSim → LUCiD, all in-container, using
`$GENIE_XSEC_FILE` (which is exported into the container as
`APPTAINERENV_GENIE_XSEC_FILE`).

`primary_source: "supernova"` chains sntools → rooTracker → PhotonSim →
LUCiD instead, and fans out into `config_NNNNNN/<model>/<ordering>/`
subcases (one per `supernova.models × supernova.orderings`). It needs a
`supernova` block and `SN_ENV_BASE` (sntools+snewpy) rather than GENIE.
See `lucid/production/README.md` § "Supernova bursts (sntools)".

### Time-based fan-out (`events_schedule`)

S3DF's `preemptable` QoS occasionally bumps jobs, and SLURM time limits
are easier to live with when each sub-job is short. To target ~1h per
sub-job, add an `events_schedule` block to the config — `generate_jobs.sh`
will derive `n_jobs` and `n_events_per_job` automatically (overriding the
flat fields). Each sub-job still produces one independent v3 batch
(`file_index = job_id - 1`), so no merge step is needed afterwards.

```json
"events_schedule": {
  "events_per_dataset":    20000,
  "target_seconds_per_job": 3600,
  "seconds_per_event":     2.5
}
```

- `events_per_dataset` — total events for the dataset, summed across
  sub-jobs. The schedule may slightly overshoot this (ceiling division).
- `target_seconds_per_job` — desired wall time per sub-job in seconds.
  `events_per_job = floor(target / seconds_per_event)`.
- `seconds_per_event` — measured wall time per event for this config on
  the partition you're submitting to. Take it from
  `report_time_performance.py` (add ~2 s for the PhotonSim portion).
  Reference numbers (median, PhotonSim + LUCiD) are in the timing table
  near the bottom of this README.

Each generated sbatch passes `--n-events <events_per_job>` to
`lucid-run-job`, so the per-job event count is consistent across the
schedule, the sbatch, and the HDF5 `config/n_events` attr (which is what
`verify_jobs.py` checks against).

The `-n` / `-e` flags on `submit_all_configs.sh` still force flat
splitting — they drop `events_schedule` from the temp config before
fan-out.

### Topping up an existing dataset

Each sub-job writes a `wc_*_<file_index>.h5` shard where `file_index =
job_id − 1`, so new sub-jobs just join the existing tree as long as
their `job_id` doesn't collide with the previous wave. `generate_jobs.sh`
takes two flags for this:

- `-j K` — start the loop at `job_id = K` (default `1`). Set to `N + 1`
  to top up a dataset that already has `N` sub-jobs.
- `-N M` — pin `n_jobs = M` for this invocation, overriding the JSON /
  `events_schedule`. Use with `-j` for "add M more sub-jobs": the new
  sub-jobs still inherit `events_per_job` from the schedule.

```bash
# Count what's already there
N=$(ls <OUTPUT_BASE>/SK_like/config_000001/submit_job_*.sbatch | wc -l)

# Add 5 more sub-jobs (~5 × events_per_job extra events)
./jobs/generate_jobs.sh -c configs/dataprod_01_mu.json -P milano \
    -j $((N + 1)) -N 5 -s
```

Failure detection works the same across waves: `verify_jobs.py` walks
every `submit_job_*.sbatch` and checks the matching shard, so a
preempted sub-job from either wave shows up identically. Run
`resubmit_failed.sh` before a top-up if you want a clean slate.

### Spreading across partitions (weighted round-robin)

A comma-separated `SLURM_PARTITION` (or `-P`) round-robins sub-jobs across
partitions, weighted by an optional `:N` suffix; a single value targets one
partition:

```bash
export SLURM_PARTITION="roma:130,milano:272"   # weight by node count
```

Each sub-job is stamped with one partition by a smooth weighted round-robin,
chosen at generation time, and the fan-out prints the resulting split. Which
partitions to use, the weights, and why the choice is per-job are
cluster-specific — see your cluster's quickstart, e.g.
[`docs/QUICKSTART_S3DF.md`](../../../docs/QUICKSTART_S3DF.md).

## Output layout

Each `config_NNNNNN/` is one LUCiD **dataset**. Each job contributes
one **batch** (`file_index = job_id - 1`) of four parallel HDF5 files:

```
OUTPUT_BASE_PATH/SK_like/           # detector chosen via submit_all_configs.sh -D <name>
├── config_000001/                  # dataset: "single mu-"
│   ├── submit_job_000001.sbatch
│   ├── job_000001-<jobid>.{out,err}
│   ├── sensor/wc_sensor_0000.h5    # ─┐
│   ├── hits/wc_hits_0000.h5        #  ├── batch 0 (job 1)
│   ├── step/wc_step_0000.h5        #  │
│   ├── labl/wc_labl_0000.h5        # ─┘
│   ├── sensor/wc_sensor_0001.h5    # ─┐ batch 1 (job 2)
│   └── ...                         #  │
└── config_000002/
```

See [`docs/LUCID_DATASET.md`](../../../docs/LUCID_DATASET.md) for the
v3 schema (sensor / hits / edep / labl files, `config/` provenance
group, particle categorization, segment merging rules).

## Monitoring & cleanup

```bash
# Watch SLURM queue (only photonsi* jobs)
./jobs/monitor_jobs.sh -w

# Cancel my production jobs
scancel -u $USER -n photonsi

# Wipe ROOT + log files from a finished output dir (keeps HDF5)
./utils/clean_output_data.sh -d <output_dir> -a --execute
```

## Finding and resubmitting failed jobs

`verify_jobs.py` walks a dataprod tree, locates every `submit_job_*.sbatch`,
and flags ones whose v3 batch (`wc_{sensor,hits,step,labl}_<file_index>.h5`)
is missing, unreadable, or whose `config/n_events` attr doesn't match the
per-job event count baked into the sbatch (or its parent config JSON).
Datasets need no merging — each `file_index` batch is its own shard — so
the resubmit loop is a single wave per drain:

```bash
# Show what's broken
./jobs/resubmit_failed.sh <OUTPUT_BASE_PATH>/SK_like --dry-run

# Resubmit
./jobs/resubmit_failed.sh <OUTPUT_BASE_PATH>/SK_like
```

The wrapper runs `verify_jobs.py --list` inside the container (h5py lives
there), then `xargs sbatch`-es on the host. Idempotent — re-run after each
drain wave until "failed" is 0.

For a human-readable breakdown without resubmission, run `verify_jobs.py`
directly inside the container:

```bash
apptainer exec ${LUCID_IMAGE_PATH} \
    python3 ./jobs/verify_jobs.py <OUTPUT_BASE_PATH>/SK_like
```

## Per-event timing

`report_time_performance.py` parses SLURM `*.out` files for the LUCiD
per-event timing line and produces histograms / summary tables.

```bash
python ./jobs/report_time_performance.py --all \
    --base-dir <OUTPUT_BASE_PATH>/SK_like \
    --output timing_report.png
```

To produce numbers in the exact shape an `events_schedule` block wants
(`seconds_per_event` = median LUCiD per-event + PhotonSim_elapsed /
n_events), use `timing_for_schedule.py`. It prints a markdown table and
a JSON block per config you can paste straight into the per-config
JSON:

```bash
python3 ./jobs/timing_for_schedule.py \
    --base-dir <timing_output>/SK_like \
    --partition roma \
    --target-seconds 3600 \
    --events-per-dataset 20000
```

Per-event cost on **Roma** (50-event sample per config, 2026-05-15; ranges
are the post-cleanup energy spec: e- 1–2000 MeV, others 200–2000 MeV).
`seconds_per_event` = median LUCiD per-event + PhotonSim_elapsed / n_events
(slightly pessimistic — PhotonSim_elapsed includes Geant4 init).
Suitable as a default for `events_schedule.seconds_per_event`.

| Config | Description                          | LUCiD median (s) | PhotonSim/event (s) | **s/event** |
|--------|--------------------------------------|---:|---:|---:|
| 000001 | single mu-                            |  3.90 | 0.94 |  **4.83** |
| 000002 | single pi+                            |  2.89 | 0.65 |  **3.54** |
| 000003 | single e-                             |  4.17 | 1.05 |  **5.22** |
| 000004 | single pi-                            |  2.39 | 0.62 |  **3.00** |
| 000005 | single pi0                            |  5.14 | 1.22 |  **6.36** |
| 000006 | single e- (low-E, 1–20 MeV)           |  0.11 | 0.09 |  **0.20** |
| 000007 | mu- + pi+                             |  6.05 | 1.40 |  **7.45** |
| 000008 | e- + pi+                              |  6.55 | 1.47 |  **8.02** |
| 000009 | e- + pi0                              |  9.19 | 2.11 | **11.30** |
| 000010 | mu- + pi+ + pi0                       | 10.76 | 2.42 | **13.18** |
| 000011 | mu- + pi+ + pi-                       |  8.03 | 1.94 |  **9.97** |
| 000012 | e- + pi+ + pi0                        | 10.32 | 2.40 | **12.72** |
| 000013 | GENIE numu                            |  2.15 | 0.58 |  **2.73** |
| 000014 | GENIE nue                             |  2.68 | 0.69 |  **3.37** |
| 000015 | pile-up: (mu-+pi+) + GENIE numu       |  9.31 | 0.60 |  **9.90** |
| 000016 | pile-up: mu- + pi+ particle-gun       |  6.78 | 0.59 |  **7.37** |
| 000017 | pile-up: GENIE numu + GENIE nue       |  6.33 | 0.84 |  **7.17** |
| 000018 | pile-up: 2× particle-bomb (N=1–5/vtx) | 21.14 | 2.75 | **23.89** |

Raw .out files preserved at
`/sdf/data/neutrino/cjesus/new_photonsim_output/timing_20260515/SK_like/config_NNNNNN/`.

## Troubleshooting

- **`Error: LUCID_IMAGE_PATH not set`** → fill in `user_paths.sh`.
- **`Error: LUCID_IMAGE_PATH=... does not exist`** → the `.sif` hasn't
  been pulled yet. Run the `apptainer pull` line above.
- **Job dies with `gxspl-min.xml.gz: No such file`** → the config
  pinned a tune whose spline isn't bundled. Either switch to
  `G18_10a_02_11b` (the in-repo default) or override
  `GENIE_XSEC_FILE` in `user_paths.sh` with a cvmfs-staged spline
  for your tune (cvmfs is bind-mounted into the container).
- **Job failed somewhere mid-pipeline** — read the SLURM logs in the
  config dir: `config_NNNNNN/job_*-*.{out,err}`.
