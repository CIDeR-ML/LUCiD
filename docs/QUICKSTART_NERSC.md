# NERSC Quickstart — Production on Perlmutter (SLURM)

Concise runbook for the NERSC / Perlmutter deployment. Pairs with
[QUICKSTART_S3DF.md](QUICKSTART_S3DF.md) (same pipeline, SLURM at SLAC),
[QUICKSTART_LXPLUS.md](QUICKSTART_LXPLUS.md) (HTCondor at CERN), and
[CLUSTER_ABSTRACTION.md](CLUSTER_ABSTRACTION.md) (architecture). The single
code path that drives every batch deployment lives at
`LUCiD/lucid/production/cluster_common/`.

Perlmutter runs SLURM, so this reuses the SLURM planning/fan-out wholesale.
Only the per-cluster `NerscAdapter` (`cluster_common/nersc.py`, selected by
`CLUSTER=nersc`) differs from S3DF, in three ways:

1. **Submit directives** are `--qos` + `--constraint=cpu|gpu`, never
   `--partition` (Perlmutter rejects `--partition`).
2. **Container runtime** is the cvmfs OSG apptainer (Perlmutter ships no
   system apptainer/singularity). `--nv` is only added for GPU jobs.
3. **Accounts** split CPU vs GPU: `dune` for CPU, `dune_g` for GPU.

## Filesystem layout

| Path | Use |
|---|---|
| `/global/u1/<i>/<user>/DIFFSIM/` (= `/global/homes/...`) | Source checkouts (`LUCiD`, `PhotonSim`). Home quota. |
| `/global/cfs/cdirs/dune/users/<user>/software/images/` | Container `.sif` (multi-GB). CFS is backed up and **not purged**. |
| `/global/cfs/cdirs/dune/users/<user>/DORAEMON/` | `OUTPUT_BASE_PATH` — dataset-production outputs (`<detector>/config_NNNNNN/`). |
| `/global/cfs/cdirs/dune/users/<user>/SIREN_files/` | `SIREN_OUTPUT_BASE_PATH` — `training_inputs/`, `smax_parametrization/`, plus `timing_results/`, `training_stable/`, `training_tests/`. |
| `$SCRATCH` (`/pscratch/sd/<i>/<user>`) | Scratch; **purged** — only use for the apptainer pull cache/tmp, never outputs or the image. |

`<i>` is the first letter of `<user>` (e.g. `c` for `cjesus`).

> **Why CFS, not scratch.** `$SCRATCH` is periodically purged. The `.sif`
> image and all outputs live on CFS so they survive. Scratch is used only as
> `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` during the one-time image pull.

## One-time setup

```bash
# Clone both repos (gh is installed + authed; see CLAUDE.md)
cd /global/u1/$(whoami | head -c1)/$(whoami)/DIFFSIM   # or wherever
gh repo clone cesarjesusvalls/PhotonSim
gh repo clone CIDeR-ML/LUCiD
cd LUCiD

# Pull the container image to CFS. Perlmutter has no system apptainer; use
# the cvmfs OSG build, and point its cache/tmp at scratch (not home).
export APPTAINER_BIN=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" \
         /global/cfs/cdirs/dune/users/$(whoami)/software/images
"$APPTAINER_BIN" pull \
    /global/cfs/cdirs/dune/users/$(whoami)/software/images/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

# Configure paths
cd lucid/production/jobs
cp user_paths.nersc.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh        # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, ...
```

`user_paths.sh` defaults to **CPU** (`dune`, `--constraint=cpu`,
`DEFAULT_GPUS=0`). GPU stages flip to `dune_g` + `--constraint=gpu`
automatically when a job requests a GPU.

> **Always bind the local LUCiD (`LUCID_DEV_PATH`).** The LUCiD baked into the
> container image can lag your active branch, so jobs should run your checkout,
> not the frozen copy. The NERSC template sets `LUCID_DEV_PATH` by default;
> leave it set unless you specifically want to test the baked image.

## Submit one test run (one job per config, 2 events each)

```bash
cd lucid/production/jobs/dataprod
echo "y" | ./submit_all_configs.sh -t -s -o /global/cfs/cdirs/dune/users/$(whoami)/DORAEMON/WAND/$(date +%Y%m%d)_test
```

Each config produces:

```
<OUT>/<detector>/config_NNNNNN/      # default detector: SK_like
├── sensor/wc_sensor_0000.h5
├── hits/wc_hits_0000.h5
├── edep/wc_edep_0000.h5
└── labl/wc_labl_0000.h5
```

Pass `-D <name>` to run the same configs against a different geometry
(`-D HK`, `-D WCTE`); the name selects
`LUCiD/config/<detector>_{geom,physics}_config.json`.

## Full production

Drop `-t` and use the config's full `n_jobs` / `n_events_per_job`:

```bash
echo "y" | ./submit_all_configs.sh -s -o /global/cfs/cdirs/dune/users/$(whoami)/DORAEMON/<name>
```

Each job of a config writes its own batch (`file_index = job_id - 1`) into
shared `{sensor,hits,edep,labl}/` subdirs — the whole config dir is one
LUCiD dataset.

## SIREN-input scan (electrons)

```bash
cd lucid/production/jobs/siren_inputs
python3 generate_jobs.py -c configs/water_el.json -t      # prepare, inspect the .sbatch
python3 generate_jobs.py -c configs/water_el.json -t -s   # submit one smoke-test cell
squeue -u $USER
python3 generate_jobs.py -c configs/water_el.json -s      # full scan
./resubmit_failed.sh $SIREN_OUTPUT_BASE_PATH              # recover failed sub-jobs
./merge.sh $SIREN_OUTPUT_BASE_PATH                        # hadd per-cell ROOTs
```

Cells land under `$SIREN_OUTPUT_BASE_PATH/training_inputs/<material>/<particle>/<E>MeV/`.

## Useful commands

```bash
# Monitor (squeue-flavoured, same as S3DF)
./dataprod/monitor_jobs.nersc.sh -w

# Cancel all my production jobs
scancel -u $USER -n photonsi

# Live queue
squeue -u $USER

# Inspect one output with the viewer (see viewer/README.md for SSH tunneling)
python3 /path/to/LUCiD/viewer/serve_viewer.py <OUT>/SK_like/config_000001
```

## Dev loop — bind-mount your checkouts

Export either or both in `user_paths.sh`; each emitted sbatch then adds
`-B <path>:/opt/{LUCiD,PhotonSim}` to its `apptainer exec`:

```bash
export LUCID_DEV_PATH="/global/u1/c/$(whoami)/DIFFSIM/LUCiD"
export PHOTONSIM_DEV_PATH="/global/u1/c/$(whoami)/DIFFSIM/PhotonSim"
```

As on every cluster: if you change PhotonSim **source** under the dev path,
rebuild the host PhotonSim against the container's GEANT4/ROOT before
re-submitting (Python changes in LUCiD are picked up live):

```bash
"$APPTAINER_BIN" exec -B /global/cfs,/global/homes,/pscratch,/cvmfs \
    -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    bash -lc 'cmake --build /opt/PhotonSim/build -j 4'
```

## Refreshing the container image

```bash
export APPTAINER_BIN=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer_cache" APPTAINER_TMPDIR="$SCRATCH/.apptainer_tmp"
"$APPTAINER_BIN" pull --force "$LUCID_IMAGE_PATH" docker://ghcr.io/cider-ml/lucid:latest
```

Then rebuild PhotonSim against the new image (above) if dev-loop binds are set.

## Spatial overlap cache

Same as on S3DF — `$LUCID_OVERLAP_CACHE_DIR` → package dir if writable →
`$XDG_CACHE_HOME/lucid/spatial_overlap_integrals` (default `~/.cache/lucid/...`).
No setup needed.

## Not yet wired: `train_siren` (GPU)

`NerscAdapter.render_train_run` raises `NotImplementedError` on purpose. The
GPU-training defaults in `jobs/train_siren/generate_jobs.py` still carry SLAC
values (`partition: roma`, `account: mli:cider-ml`). Before GPU training runs
here, those need Perlmutter values (`dune_g` + `--constraint=gpu`) and a
NERSC train-run renderer. The CPU stages (dataprod, siren_inputs, smax) are
fully wired. Until then, GPU training stays a documented follow-up — the rest
of the pipeline is unaffected.

## Paths / conventions

- Configs: `LUCiD/lucid/production/configs/dataprod_*.json` (dataprod);
  `jobs/<stage>/configs/*.json` (per-stage).
- Runner: `LUCiD/lucid/production/run_job.py` (entry: `lucid-run-job`).
- Production wrappers: `LUCiD/lucid/production/jobs/`.
- Adapter: `cluster_common/nersc.py` (`CLUSTER=nersc`).
- Architecture: [CLUSTER_ABSTRACTION.md](CLUSTER_ABSTRACTION.md).
