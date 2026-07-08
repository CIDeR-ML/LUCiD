# NERSC (Perlmutter) — production on SLURM

Concise runbook for the NERSC / Perlmutter deployment. Pairs with
[deploy-s3df.md](deploy-s3df.md) (same pipeline, SLURM at SLAC),
[deploy-lxplus.md](deploy-lxplus.md) (HTCondor at CERN), and
[cluster-abstraction.md](cluster-abstraction.md) (architecture). The single
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
| `/global/cfs/cdirs/dune/users/<user>/DORAEMON/` | `OUTPUT_BASE_PATH` — dataset-production outputs (`<detector>/<block>/<split>/config_NN/`). |
| `/global/cfs/cdirs/dune/users/<user>/SIREN_files/` | `SIREN_OUTPUT_BASE_PATH` — `training_inputs/`, `smax_parametrization/`, plus `timing_results/`, `training_stable/`, `training_tests/`. |
| `$SCRATCH` (`/pscratch/sd/<i>/<user>`) | Scratch; **purged** — only use for the apptainer pull cache/tmp, never outputs or the image. |

`<i>` is the first letter of `<user>`.

> **Why CFS, not scratch.** `$SCRATCH` is periodically purged. The `.sif`
> image and all outputs live on CFS so they survive. Scratch is used only as
> `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` during the one-time image pull.

## One-time setup

```bash
# Clone both repos (use gh auth login, or plain git clone with https)
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

Configs live in per-block folders under `lucid/production/configs/<block>/`,
where `<block>` is one of **GeV** (single particles + particle-bomb +
multiparticle + GENIE), **Solar** (low-energy e⁻), **SN** (supernova bursts), or
**Test** (dev). Each file is `NN_name.json` (2-digit, numbering restarts per
block) and declares its own `detector` plus `nominal_train` / `nominal_test`.

Each config produces one dataset per split:

```
<OUT>/<detector>/<block>/<split>/config_NN/     # e.g. HK/GeV/train/config_01
├── sensor/wc_sensor_0000.h5
├── hits/wc_hits_0000.h5
├── step/wc_step_0000.h5
└── labl/wc_labl_0000.h5
```

- **GeV / Solar** carry a `train` / `test` split layer; a config with
  `nominal_train: 0` (multiparticle, GENIE) is **test-only**. Train and test get
  disjoint master seeds, so the two datasets never share an event.
- **SN / Test** are flat (no split layer): `<detector>/SN/config_NN/…`. Supernova
  additionally nests `<model>/<ordering>/` below the config dir.
- Job counts come from `nominal_<split> / (target_seconds_per_job /
  seconds_per_event)`; supernova is one all-at-once burst per job.

The detector for both the output path and the run comes from the config's
`detector` field; the `-D <name>` flag is only a fallback for configs that omit
it (it selects `LUCiD/config/<detector>_{geom,physics}_config.json`).

## Full production

Drop `-t` and use the config's full `n_jobs` / `n_events_per_job`:

```bash
echo "y" | ./submit_all_configs.sh -s -o /global/cfs/cdirs/dune/users/$(whoami)/DORAEMON/<name>
```

Each job of a config writes its own batch (`file_index = job_id - 1`) into
shared `{sensor,hits,step,labl}/` subdirs — the whole config dir is one
LUCiD dataset.

## SIREN-input scan (electrons)

> Run `generate_jobs.py` with **`/usr/bin/python3.11`**: it's a host-side
> submitter (it calls `sbatch`, which only works on the login node, not in the
> container), and the bare `python3` here is 3.6.15 — too old for the script.

**Submission modes** — same work units, different SLURM packaging:
- *(default)* one sbatch per sub-job; simple, fine for small scans.
- `--array` — one job array per cell; far fewer submissions, self-recovering.
- `--pack[=N]` — `N` whole-node `regular` jobs, each running its units in
  parallel across the node's cores (in-container `xargs` pool). **Use this when
  the `shared` partition is oversubscribed** — single-core `shared` jobs get
  starved waiting for fractional slots, whereas a handful of whole-node
  `regular` jobs schedule and then burst 128 units at once. NERSC-only.

Recommended flow (electrons), smoke test first:

```bash
cd lucid/production/jobs/siren_inputs
# smoke: one cell as a packed node job; confirm it lands + output is correct
/usr/bin/python3.11 generate_jobs.py -c configs/water_el.json --pack -t -s
squeue -u $USER
# full scan: ~10 whole-node jobs draining all cells in parallel
/usr/bin/python3.11 generate_jobs.py -c configs/water_el.json --pack -s
# after each drain wave: validate (OpticalPhotons truth key) + resubmit only
# the missing/incomplete units; repeat until "0 invalid"
./recover.sh -c configs/water_el.json
# then hadd each cell's output_job_*.root -> photonsim.root
./merge.sh $SIREN_OUTPUT_BASE_PATH/training_inputs
```

Recovery tool split: **`recover.sh`** drives `--array`/`--pack` recovery
(output-driven — deletes ROOTs lacking the `OpticalPhotons` key, then resubmits
only those). `resubmit_failed.sh` is for the per-job (default) mode used on
S3DF/LXPLUS and does **not** understand array/pack. Note both `recover.sh` and
`merge.sh` take the `…/training_inputs` subdir, not `$SIREN_OUTPUT_BASE_PATH`.

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
re-submitting (Python changes in LUCiD are picked up live).

**First build** — the host `build/` doesn't exist yet (fresh `PhotonSim`
checkout). Configure with the same flags the image uses
(`LUCiD/container/Dockerfile`), then build. The two non-obvious flags:
`-DCMAKE_PREFIX_PATH=/opt/conda` makes cmake pick conda's expat (the system
expat is too old for Geant4), and the linker flags point `ld` at conda's newer
`libstdc++` — the system one lacks the `GLIBCXX_3.4.32` / `CXXABI_1.3.15`
symbols the conda Geant4/Qt libs reference, so the link fails without them:

```bash
"$APPTAINER_BIN" exec -B "$APPTAINER_BINDS" -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    bash -lc 'cmake -S /opt/PhotonSim -B /opt/PhotonSim/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/conda \
        -DCMAKE_EXE_LINKER_FLAGS="-L/opt/conda/lib -Wl,-rpath,/opt/conda/lib" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/conda/lib -Wl,-rpath,/opt/conda/lib" \
        -DGeant4_DIR=$(geant4-config --prefix)/lib/cmake/Geant4 \
        -DROOT_DIR=/opt/root/v6-30-04/cmake \
      && cmake --build /opt/PhotonSim/build -j 4'
```

**Incremental rebuild** — `build/` already configured (after the first build,
or after `git pull` in `PhotonSim/`); just rebuild:

```bash
"$APPTAINER_BIN" exec -B "$APPTAINER_BINDS" -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
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

- Configs: `LUCiD/lucid/production/configs/{GeV,SN,Solar,Test}/NN_name.json`;
  `jobs/<stage>/configs/*.json` (per-stage).
- Runner: `LUCiD/lucid/production/run_job.py` (entry: `lucid-run-job`).
- Production wrappers: `LUCiD/lucid/production/jobs/`.
- Adapter: `cluster_common/nersc.py` (`CLUSTER=nersc`).
- Architecture: [cluster-abstraction.md](cluster-abstraction.md).
