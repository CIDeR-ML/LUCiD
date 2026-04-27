# S3DF Production Jobs

SLURM fan-out for `lucid-run-job` on S3DF. The whole pipeline (gevgen →
gntpc → PhotonSim → LUCiD, or macro → PhotonSim → LUCiD) runs inside
the unified LUCiD apptainer image — one `apptainer exec` per job, no
host-side build state.

For a higher-level walkthrough see
[`docs/QUICKSTART_S3DF.md`](../../../docs/QUICKSTART_S3DF.md).

## Quick start

```bash
# 1. Pull the unified container (one-time)
apptainer pull /sdf/data/neutrino/<user>/software/images/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

# 2. Configure paths
cp user_paths.sh.template user_paths.sh
vim user_paths.sh   # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, SLURM_*

# 3. Test with one job per config (no submission)
./jobs/submit_all_configs.sh -t

# 4. Submit all configs
echo "y" | ./jobs/submit_all_configs.sh -s

# 5. Monitor
./jobs/monitor_jobs.sh -w
```

## Directory layout

```
s3df_jobs/
├── user_paths.sh.template   # config template — copy to user_paths.sh
├── user_paths.sh            # your overrides (gitignored)
├── jobs/
│   ├── generate_jobs.sh     # fan one config out into sbatch scripts
│   ├── submit_all_configs.sh
│   ├── monitor_jobs.sh
│   ├── cleanup_jobs.sh
│   └── report_time_performance.py
└── utils/
    ├── clean_output_data.sh # drop ROOT/log files from a finished output dir
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
| `output_path` | string | subdir under `OUTPUT_BASE_PATH/<material>/` |
| `primary_source` | string | `"particle_gun"` or `"genie"` |
| `energy_distribution` | string | `"uniform"` or `"monoenergetic"` |
| `particles` | array | per-particle type + energy ranges |
| `lucid_options` | object | LUCiD writer flags (smearing, translation, etc.) |
| `n_jobs` | int | number of SLURM jobs (one batch each) |
| `n_events_per_job` | int | events per job |

`primary_source: "genie"` causes the runner to chain
gevgen → gntpc → PhotonSim → LUCiD, all in-container, using
`$GENIE_XSEC_FILE` (which is exported into the container as
`APPTAINERENV_GENIE_XSEC_FILE`).

## Output layout

Each `config_NNNNNN/` is one LUCiD **dataset**. Each job contributes
one **batch** (`file_index = job_id - 1`) of four parallel HDF5 files:

```
OUTPUT_BASE_PATH/water/uniform_energy/
├── config_000001/                  # dataset: "single mu-"
│   ├── submit_job_000001.sbatch
│   ├── job_000001-<jobid>.{out,err}
│   ├── sensor/wc_sensor_0000.h5    # ─┐
│   ├── inst/wc_inst_0000.h5        #  ├── batch 0 (job 1)
│   ├── seg/wc_seg_0000.h5          #  │
│   ├── labl/wc_labl_0000.h5        # ─┘
│   ├── sensor/wc_sensor_0001.h5    # ─┐ batch 1 (job 2)
│   └── ...                         #  │
└── config_000002/
```

See [`docs/LUCID_DATASET.md`](../../../docs/LUCID_DATASET.md) for the
v3 schema (sensor / inst / seg / labl files, `config/` provenance
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

## Per-event timing

`report_time_performance.py` parses SLURM `*.out` files for the LUCiD
per-event timing line and produces histograms / summary tables.

```bash
python ./jobs/report_time_performance.py --all \
    --base-dir <OUTPUT_BASE_PATH>/water/uniform_energy \
    --output timing_report.png
```

Median LUCiD per-event times (PhotonSim adds ~2 s):

| Config | Description    | Ampere (GPU) | Roma (CPU) | Milano (CPU) |
|--------|----------------|--------------|------------|--------------|
| 000001 | muon           | 0.51 s       | 6.79 s     | 9.04 s       |
| 000002 | charged pion   | 0.28 s       | 7.54 s     | 8.93 s       |
| 000003 | electron       | 0.46 s       | 2.26 s     | 5.94 s       |
| 000006 | low-e electron | 0.06 s       | 0.06 s     | 0.08 s       |
| 000007 | mu + pi mixed  | 0.83 s       | 26.22 s    | 21.59 s      |

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
