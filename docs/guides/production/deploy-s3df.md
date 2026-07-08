# S3DF Quickstart — Production on SLURM

Concise runbook for the S3DF (SLAC) deployment. For a local-machine
workflow with no cluster, see [local.md](local.md);
for Docker on macOS/Linux, see [docker.md](docker.md);
for HTCondor on LXPLUS, see [deploy-lxplus.md](deploy-lxplus.md).
The single code path that drives both batch deployments lives at
`LUCiD/lucid/production/cluster_common/` — see
[cluster-abstraction.md](cluster-abstraction.md).

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
cd lucid/production/jobs
cp user_paths.s3df.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh          # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, ...
```

Point `LUCID_IMAGE_PATH` in `user_paths.sh` at the `.sif` produced
above.

## Download example data

`scripts/download_data.sh` pulls the water-muon example data (ROOT +
trained SIREN models) from the CERNBox share. On S3DF, keep the large
files on the shared `neutrino` filesystem and leave only symlinks in the
repo by passing `--store-dir`:

```bash
cd LUCiD
./scripts/download_data.sh --store-dir /sdf/data/neutrino/<user>/CERNBOX
```

This writes the real files under
`/sdf/data/neutrino/<user>/CERNBOX/water/muon/` and points
`data/water/muon/` at them via symlinks. `data/wbls/muon/` is wired up
as relative symlinks into `../../water/muon/`, so wbls reuses the same
ROOT + SIREN files as water. Only the 1000 MeV ROOT is fetched by
default; add `--all-energies` for 500/1000/1500 MeV.

## Submit one test run (one job per config, 2 events each)

```bash
cd lucid/production/jobs/dataprod
echo "y" | ./submit_all_configs.sh -t -s -o /sdf/data/<user>/WAND/<date>_test
```

Each config produces:

```
<OUT>/<detector>/config_NNNNNN/      # default detector: SK_like
├── sensor/wc_sensor_0000.h5
├── hits/wc_hits_0000.h5
├── step/wc_step_0000.h5
└── labl/wc_labl_0000.h5
```

To run the same configs against a different geometry, pass `-D <name>`
to `submit_all_configs.sh` (e.g. `-D HK`, `-D WCTE`). The name selects
`LUCiD/config/<detector>_{geom,physics}_config.json`.

## Full production

Drop `-t` and use the config's full `n_jobs` / `n_events_per_job`:

```bash
echo "y" | ./submit_all_configs.sh -s -o /sdf/data/<user>/production/<name>
```

Each job of a config writes its own batch (`file_index = job_id - 1`)
into shared `{sensor,hits,step,labl}/` subdirs — the whole config dir
is one LUCiD dataset.

## Spreading jobs across roma + milano

The two CPU batch partitions are `roma` (130 nodes) and `milano`
(272 nodes). To use both, set `SLURM_PARTITION` in your `user_paths.sh` to
a comma list weighted by node count; the dataprod fan-out round-robins each
sub-job onto one of them in that proportion:

```bash
export SLURM_PARTITION="roma:130,milano:272"
```

`user_paths.sh` is per-user and gitignored, so set this in your own copy —
it is not shared via the repo. Refresh the counts with
`sinfo -h -p <part> -o %D | awk '{s+=$1} END{print s}'`.

The split is fixed per sub-job at generation time because the account's
SLURM associations are per-partition (`mli:cider-ml@roma`,
`mli:cider-ml@milano` are distinct). As a result SLURM rejects a single
multi-partition request (`--partition=roma,milano`), and `scontrol` cannot
move an already-queued job to the other partition — both fail with an
invalid account/partition error. To rebalance a wave that is already
queued, `scancel` it and resubmit.

## Dev loop — bind-mount your checkouts

Same pattern as Docker, via `apptainer exec -B`:

```bash
apptainer exec \
    -B "$PWD/LUCiD:/opt/LUCiD" \
    -B "$PWD/PhotonSim:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    lucid-run-job --config /opt/LUCiD/lucid/production/configs/GeV/01_mu.json \
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

## Running the JAX stack on a GPU

The container's baked `jaxlib` is **CPU-only** — `jax.devices()` returns
`CpuDevice` even on a GPU node with `--nv`. The CUDA build is not in the
`.sif`; it lives in a separate user-site env that gets layered onto the
container at run time:

```
LUCID_ENV_BASE=/sdf/data/neutrino/<user>/python_envs/lucid
```

This holds the `jax-cuda12-plugin` + `nvidia-*-cu12` wheels matched to the
container's `jax 0.4.38`, plus `optax`. Inject it with
`APPTAINERENV_PYTHONUSERBASE` (this maps to `PYTHONUSERBASE` inside the
container, so its `site-packages` is added to the user-site path) and add
`--nv`. The interactive/Jupyter recipe is in
`JUPYTER_SETUP.md` (lives under `NB_VALIDATION/`);
for a batch job the same two ingredients apply:

```bash
APPTAINERENV_PYTHONUSERBASE=/sdf/data/neutrino/<user>/python_envs/lucid \
APPTAINERENV_PYTHONPATH="" \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs \
    -B "$PWD/LUCiD:/opt/LUCiD" \
    "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 /opt/LUCiD/<your_script>.py
```

`jax.devices()` then returns `CudaDevice`. To run against a bind-mounted
checkout *and* the GPU env, put the checkout's root first on `sys.path` so
its `lucid` wins over the env's editable `lucid` (which points at the
`NB_VALIDATION/` tree). Request a GPU node via `--partition=turing`
(RTX 2080 Ti) or `ampere`/`ada`/`hopper`; the SIREN GPU training is the
only *production* GPU path and uses PyTorch, not JAX.

## Useful commands

```bash
# Monitor
./monitor_jobs.sh -w

# Cancel all my production jobs
scancel -u $USER -n photonsi

# Inspect one output with the viewer (see viewer/README.md for SSH tunneling)
python3 /path/to/LUCiD/viewer/serve_viewer.py <OUT>/SK_like/config_000001
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

- Configs: `LUCiD/lucid/production/configs/{GeV,SN,Solar,Test}/NN_name.json`
- Runner: `LUCiD/lucid/production/run_job.py` (entry: `lucid-run-job`)
- Production wrappers: `LUCiD/lucid/production/jobs/`
- schema spec: [../../reference/dataset-schema.md](../../reference/dataset-schema.md)
