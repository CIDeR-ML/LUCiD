# S3DF Quickstart — Production on SLURM

Concise runbook for the S3DF (SLAC) deployment. For a local-machine
workflow with no cluster, see [QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md).

## One-time setup

```bash
# Clone both repos
git clone git@github.com:cesarjesusvalls/PhotonSim.git
git clone git@github.com:CIDeR-ML/LUCiD.git
cd LUCiD

# Configure paths (copies of S3DF-specific install locations)
cd lucid/production/s3df_jobs
cp user_paths.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh          # set PHOTONSIM_BIN, LUCID_PATH, OUTPUT_BASE_PATH, ...

# Build PhotonSim (once)
./utils/build_photonsim.sh
```

## Submit one test run (one job per config, 2 events each)

```bash
cd lucid/production/s3df_jobs/jobs
echo "y" | ./submit_all_configs.sh -t -s -o /sdf/data/<user>/WAND/<date>_test
```

Each config produces:

```
<OUT>/water/uniform_energy/config_NNNNNN/
├── sensor/wc_sensor_0000.h5
├── inst/wc_inst_0000.h5
├── seg/wc_seg_0000.h5
└── labl/wc_labl_0000.h5
```

## Full production

Drop `-t` and use the config's full `n_jobs` / `n_events_per_job`:

```bash
echo "y" | ./submit_all_configs.sh -s -o /sdf/data/<user>/production/<name>
```

Each job of a config writes its own batch (`file_index = job_id - 1`)
into shared `{sensor,inst,seg,labl}/` subdirs — the whole config dir
is one LUCiD dataset.

## Useful commands

```bash
# Monitor
./monitor_jobs.sh -w

# Cancel all my production jobs
scancel -u $USER -n photonsi

# Inspect one output with the viewer (see viewer/README.md for SSH tunneling)
python3 /path/to/LUCiD/viewer/serve_viewer.py <OUT>/water/uniform_energy/config_000001
```

## What each sbatch does

The `submit_job_NNN.sbatch` that `generate_jobs.sh` writes has two
steps:

1. **Bare-host step**: `source utils/setup_environment.sh`, then run
   `$HOST_PYTHON -m lucid.production.run_job --skip-lucid ...`. This
   handles macro generation + the PhotonSim binary (needs GEANT4/ROOT
   shared libs from the host, not the container).

2. **Singularity step**: `singularity exec $SINGULARITY_IMAGE_PATH
   python3 -m lucid.production.run_job --skip-photonsim ...`. This
   runs the LUCiD v3 writer (needs jax / numpy / h5py from the
   container).

## Paths / conventions

- Configs: `LUCiD/lucid/production/configs/dataprod_*.json`
- Runner: `LUCiD/lucid/production/run_job.py` (entry: `lucid-run-job`)
- S3DF wrappers: `LUCiD/lucid/production/s3df_jobs/`
- v3 schema spec: [LUCID_DATASET.md](LUCID_DATASET.md)
