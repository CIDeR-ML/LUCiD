# S3DF Quickstart — Production on SLURM

Concise runbook for the S3DF (SLAC) deployment. For a local-machine
workflow with no cluster, see [QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md);
for Docker on macOS/Linux, see [QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md).

## One-time setup

```bash
# Clone both repos
git clone git@github.com:cesarjesusvalls/PhotonSim.git
git clone git@github.com:CIDeR-ML/LUCiD.git
cd LUCiD

# Pull the container image. Apptainer converts the Docker image to .sif.
apptainer pull \
    /sdf/data/neutrino/<user>/software/images/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

# Configure paths
cd lucid/production/s3df_jobs
cp user_paths.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh          # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, ...
```

Point `LUCID_IMAGE_PATH` in `user_paths.sh` at the `.sif` produced
above.

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

## Dev loop — bind-mount your checkouts

Same pattern as Docker, via `apptainer exec -B`:

```bash
apptainer exec \
    -B "$PWD/LUCiD:/opt/LUCiD" \
    -B "$PWD/PhotonSim:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    lucid-run-job --config /opt/LUCiD/lucid/production/configs/dataprod_01_mu.json \
                  --output-dir /tmp/out --job-id 1 --test
```

To exercise the same bind under the SLURM fan-out (e.g. validate a
LUCiD/PhotonSim source change before a container rebuild), export
either or both of `LUCID_DEV_PATH` / `PHOTONSIM_DEV_PATH` before
running `submit_all_configs.sh` or `generate_jobs.sh`. Each emitted
sbatch then adds `-B <path>:/opt/{LUCiD,PhotonSim}` to its
`apptainer exec`. Unset the vars to go back to the frozen container.

```bash
export LUCID_DEV_PATH="$PWD/LUCiD"
export PHOTONSIM_DEV_PATH="$PWD/PhotonSim"
echo "y" | ./submit_all_configs.sh -t -s -o /sdf/data/<user>/dev_test
```

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

`generate_jobs.sh` emits a single-step sbatch body: the whole chain
(gevgen → gntpc → PhotonSim → LUCiD, or macro gen → PhotonSim → LUCiD
for particle-gun configs) runs inside one `apptainer exec
$LUCID_IMAGE_PATH lucid-run-job ...`. The image contains GEANT4 11.3,
ROOT 6.30, GENIE 3.04, PhotonSim, and the LUCiD Python stack.

In-repo GENIE configs pin tune `G18_10a_02_11b`, whose splines ship
in the container. To run with a different tune, point
`GENIE_XSEC_FILE` in `user_paths.sh` at the matching cvmfs-staged
spline file.

## Refreshing the container image

```bash
apptainer pull --force "$LUCID_IMAGE_PATH" docker://ghcr.io/cider-ml/lucid:latest
```

If you have `LUCID_DEV_PATH` / `PHOTONSIM_DEV_PATH` set (dev-loop binds),
also rebuild the host PhotonSim against the new image:

```bash
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs \
    -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    bash -lc 'cmake --build /opt/PhotonSim/build -j 4'
```

Otherwise the host binary is shadowed in but stale relative to the new
image, and LUCiD will fail with "PhotonSim ROOT file is missing
branches …".

## Override configs (`-n` / `-e`)

`submit_all_configs.sh -n N -e M` materializes a temporary JSON per
config so it can edit `n_jobs` / `n_events_per_job`. That temp file is
read by SLURM at dispatch time, *after* the wrapper has exited, so it
must (a) live on shared FS visible to compute nodes (no node-local
`/lscratch`) and (b) not be auto-deleted by the wrapper. The script
writes them to `${OUTPUT}/overridden_configs.XXXXXX/` and leaves them
there — remove the dir manually after the run if you want.

## Spatial overlap cache

LUCiD writes a small JSON cache (`spatial_overlap_integrals/*.json`).
Resolution order is `$LUCID_OVERLAP_CACHE_DIR` → package dir if
writable → `$XDG_CACHE_HOME/lucid/spatial_overlap_integrals` (default
`~/.cache/lucid/...`). With dev-loop binds the host package dir is
writable, so the cache lives next to the source. On a baked-only
container without binds it falls back to `~/.cache/lucid/`. No setup
needed.

## Paths / conventions

- Configs: `LUCiD/lucid/production/configs/dataprod_*.json`
- Runner: `LUCiD/lucid/production/run_job.py` (entry: `lucid-run-job`)
- S3DF wrappers: `LUCiD/lucid/production/s3df_jobs/`
- v3 schema spec: [LUCID_DATASET.md](LUCID_DATASET.md)
