# Cluster deployment: S3DF, NERSC, LXPLUS

Runbook for producing datasets and SIREN inputs on a batch cluster. Three are supported from one
code path, `LUCiD/lucid/production/`: SLURM on S3DF (SLAC), SLURM on NERSC/Perlmutter, and
HTCondor on LXPLUS (CERN). [How cluster targeting works](cluster-abstraction.md) describes that
code path (the adapters, `user_paths.sh`, the container hop, completion checks). For a machine
with no batch system, see [local.md](local.md); for Apptainer or Docker by hand, see
[container.md](container.md).

The first part of this page is the workflow every cluster shares; the site sections after it hold
what differs.

## The common workflow

### One-time setup

```bash
# Clone both repos
git clone https://github.com/cesarjesusvalls/PhotonSim.git
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD

# Pull the container image (Apptainer converts the Docker image to .sif); where the .sif should
# live, and which apptainer to call, is per site (below)
apptainer pull <image-dir>/lucid.sif docker://ghcr.io/cider-ml/lucid:latest

# Configure paths
cd lucid/production/platform
cp templates/user_paths.<cluster>.sh.template user_paths.sh    # s3df, nersc or lxplus
${EDITOR:-vim} user_paths.sh          # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, ...
```

Point `LUCID_IMAGE_PATH` in `user_paths.sh` at the `.sif` pulled above. `user_paths.sh` is
per-user and gitignored. Set `LUCID_DEV_PATH` / `PHOTONSIM_DEV_PATH` in it if you want the host
checkouts to shadow the container's baked copies ([dev loop](#dev-loop-bind-mount-your-checkouts)).

`scripts/download_data.py` installs the trained SIREN weights and the example GEANT4 events (the
`lucid-data-v1` release asset; see [install](../../getting-started/install.md)) into the
checkout's `data/`, once per checkout. wbls and ice reuse water's SIREN emitters
(`config/materials/wbls.json` says `"siren": "water"`), so nothing else is needed for them.

### Submit one test run (one job per config, 2 events each)

```bash
cd lucid/production/dataprod/scripts
echo "y" | ./submit_all_configs.sh -D SK_WAND -t -s -o <OUT>/<date>_test
```

`-D` is required: the GeV configs declare no detector, and without `-D` each one stops with
"detector needs to be specified". Use `SK_WAND` or `HK_WAND`; the fan-out refuses any other
detector for those configs ("no 'digitizer' block": the generic detectors carry no readout).
The `Solar`, `SN` and `Test` configs declare `HK` themselves, which overrides `-D` (the fan-out
prints a note and passes `--detector HK` to their jobs).

`-t` writes under `<OUT>/<detector>/_test/`, mirroring the production tree (layout in
[Producing datasets](index.md#where-a-batch-run-writes)). Each config cell holds four
`{sensor,hits,step,labl}/wc_*_0000.h5` files when the job finishes.

The four `GeV/02_mu*.json` configs (`02_mu`, `_fast`, `_metrics`, `_smoke`) all map to the cell
`config_02/`. After the first, the fan-out refuses each of the others ("✗ Failed" in the summary)
rather than overwrite it. Run a variant on its own with `-p` and a different `-o`, e.g.
`-p 'GeV/02_mu_smoke.json'`.

### Full production

Drop `-t`. Job counts come from each config's `nominal_train` / `nominal_test` and
`seconds_per_event` (see [sizing](#sizing-a-production-run)):

```bash
echo "y" | ./submit_all_configs.sh -D SK_WAND -s -o <OUT>/<name>
```

Each job of a config writes its own batch (`file_index = job_id - 1`) into shared
`{sensor,hits,step,labl}/` subdirs — the whole config dir is one LUCiD dataset. The fan-out's own
flags (topping up a dataset with `-j`/`-N`, `--force-overwrite`, the partition round-robin) are in
[`lucid/production/dataprod/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/dataprod/README.md).

To change the job count of a shipped config, call the fan-out directly with `-N`
(`--n-jobs-override`), or edit `nominal_*` / `target_seconds_per_job` in a copy of the config.
`submit_all_configs.sh` has no job-count flags.

### Sizing a production run

The fan-out splits a config's event budget into equal jobs. The math lives in
`dataprod/fanout.py::_plan`, driven by two fields the config carries
(`lucid/production/dataprod/configs/<block>/NN_name.json`):

- `seconds_per_event` — measured per-event PhotonSim+LUCiD wall time.
- `target_seconds_per_job` — the wall-time target each job is sized to (3600 s in the shipped
  configs).

From those:

```
events_per_job = floor(target_seconds_per_job / seconds_per_event)
n_jobs         = ceil(nominal_events / events_per_job)
core-hours     ≈ nominal_events × seconds_per_event / 3600
```

Each job is **single-core** (`DEFAULT_CPUS=1` in `user_paths.sh`), so "core-hours" is the honest
cost unit.

Worked example — **`GeV/01_pbomb.json`, the 1,000,000-event train split**
(`seconds_per_event = 12.0`, `target_seconds_per_job = 3600`):

```
events_per_job = floor(3600 / 12.0)       = 300
n_jobs         = ceil(1_000_000 / 300)    = 3334
core-hours     ≈ 1_000_000 × 12.0 / 3600  ≈ 3333
```

The shipped configs with more than a smoke-test budget (`target=3600`):

| Config | split | events | `seconds_per_event` | events/job | jobs | core-hours |
|---|---|--:|--:|--:|--:|--:|
| `GeV/01_pbomb` | train | 1,000,000 | 12.0 | 300 | 3,334 | ~3,333 |
| `GeV/01_pbomb` | test | 50,000 | 12.0 | 300 | 167 | ~167 |
| `GeV/02_mu` | test | 50,000 | 4.833 | 744 | 68 | ~67 |
| `GeV/03_pi` | test | 50,000 | 3.273 | 1,099 | 46 | ~45 |
| `GeV/04_e`, `GeV/07_gamma` | test | 50,000 | 5.22 | 689 | 73 | ~72 |
| `GeV/06_pi0` | test | 50,000 | 6.355 | 566 | 89 | ~88 |
| `Solar/01_e_low_energy` | train | 1,000,000 | 0.195 | 18,461 | 55 | ~54 |
| `Solar/01_e_low_energy` | test | 50,000 | 0.195 | 18,461 | 3 | ~3 |
| `Test/01_pile_up_bombs` | flat | 500 | 23.892 | 150 | 4 | ~3 |

GeV `02`–`07` set `nominal_train = 0`: they are test-only. Supernova (`SN/01`) is one
all-at-once burst per job, `n_jobs = nominal_test` (5). The JSONs are the source of these
numbers; recompute after editing one.

#### Estimating on-disk size

Per-event byte size is **not** a stable constant: the two heavy files (`step/` per-(segment,
sensor) hits and `sensor/`/`hits/` digits) scale with occupancy, which depends on the detector,
energy, and pile-up — so a GeV muon in HK and a solar electron differ by orders of magnitude. The
one bounded part is `labl/` (~8 `per_particle` rows and ~600 `per_track` rows per event; see the
[schema](../../reference/dataset-schema.md)), and it is usually the smallest of the four.
**Measure, don't guess:** run one full (non-`--test`) batch, then

```bash
du -sh <OUT>/<detector>/<block>/<split>/config_NN/{sensor,hits,step,labl}/wc_*_0000.h5
```

and divide by that batch's `n_events` for a MB/event figure you can multiply up to the full run.
The `--test` batches (2 events) over-count fixed per-file overhead, so size from a real batch, not
a test one.

### What each job does

The fan-out emits a single-step submit body: the whole chain (Geant4 macro → PhotonSim → LUCiD
writer for the particle-gun / bomb configs; sntools first for supernova; gevgen → gntpc first for
a GENIE config) runs inside one `apptainer exec $LUCID_IMAGE_PATH lucid-run-job ...`. What the
image contains is listed in
[`container/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/container/README.md).

No GENIE config ships today. The image carries the `G18_10a_02_11b` splines, and
`GENIE_XSEC_FILE` in `user_paths.sh` points at them; for another tune, point it at the matching
cvmfs-staged spline file.

The spatial-overlap cache needs no setup; where it lives, and that production jobs do not read
it, is in [container.md](container.md#5-changing-things).

### Verify and resume a run

Preemption and node failures leave some jobs incomplete. The completion check and the post-run
checklist are in
[cluster-abstraction.md](cluster-abstraction.md#verifying-and-resuming-cluster-agnostic). A human
summary (inside the container, which has h5py; add the site's binds):

```bash
cd lucid/production/dataprod/scripts
apptainer exec -B "$APPTAINER_BINDS" "$LUCID_IMAGE_PATH" \
    python3 verify_jobs.py <OUT>/<detector>          # e.g. .../SK_WAND
```

and the resubmission of the failures, idempotent, rerun after each drain wave:

```bash
../../platform/run_in_container.sh lucid.production.dataprod.scripts.verify_jobs \
    <OUT>/<detector> --list -s                       # drop -s to preview
```

**Where logs land** (SLURM). Each dataprod job writes to its own config dir (the `cell_dir`),
named by the render header in `platform/adapters.py`:

```
<OUT>/<detector>/<block>/<split>/config_NN/job_NNNNNN-<slurm_job_id>.out
<OUT>/<detector>/<block>/<split>/config_NN/job_NNNNNN-<slurm_job_id>.err
```

so a failing job's stderr sits right next to the four `{sensor,hits,step,labl}/` subdirs it
should have written. (SLURM-array submissions switch the suffix to `%A_%a`.) On LXPLUS the logs
go elsewhere ([below](#lxplus-cern-htcondor)).

### Useful commands

```bash
# Monitor (from lucid/production/dataprod/scripts)
./monitor_jobs.slurm.sh -w          # S3DF, NERSC (squeue-flavoured)
./monitor_jobs.htcondor.sh -w       # LXPLUS (condor_q-flavoured)

# SLURM: cancel all my production jobs (job names start with photonsim_;
# scancel -n matches whole names only)
squeue -u $USER -h -o '%i %j' | awk '$2 ~ /^photonsim_/ {print $1}' | xargs -r scancel

# Inspect one output with the viewer (see viewer/README.md for SSH tunneling)
python3 /path/to/LUCiD/viewer/serve_viewer.py <OUT>/SK_WAND/GeV/test/config_02
```

### Dev loop: bind-mount your checkouts

How the `LUCID_DEV_PATH` / `PHOTONSIM_DEV_PATH` binds work, and how to build or rebuild PhotonSim
against the image, is in
[cluster-abstraction.md](cluster-abstraction.md#dev-loop-run-your-own-checkouts-in-the-batch-jobs).
One job by hand against your checkouts:

```bash
apptainer exec \
    -B "$PWD/LUCiD:/opt/LUCiD" \
    -B "$PWD/PhotonSim:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu.json \
                  --detector SK_WAND \
                  --output-dir /tmp/out --job-id 1 --test
```

and a test wave with the same binds:

```bash
export LUCID_DEV_PATH="$PWD/LUCiD"
export PHOTONSIM_DEV_PATH="$PWD/PhotonSim"
echo "y" | ./submit_all_configs.sh -D SK_WAND -t -s -o <OUT>/dev_test
```

### Refreshing the container image

```bash
apptainer pull --force "$LUCID_IMAGE_PATH" docker://ghcr.io/cider-ml/lucid:latest
```

If you have `LUCID_DEV_PATH` / `PHOTONSIM_DEV_PATH` set (dev-loop binds), also rebuild the host
PhotonSim against the new image:

```bash
apptainer exec -B "$APPTAINER_BINDS" \
    -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    bash -lc 'cmake --build /opt/PhotonSim/build -j 4'
```

Otherwise the bound binary is stale relative to the new image (see
[cluster-abstraction.md](cluster-abstraction.md#dev-loop-run-your-own-checkouts-in-the-batch-jobs)).

### Paths and conventions

- Configs: `LUCiD/lucid/production/dataprod/configs/{GeV,SN,Solar,Test}/NN_name.json`;
  `siren_inputs/{,smax/,train_siren/}configs/*.json` (per stage).
- Runner: `LUCiD/lucid/production/dataprod/run_job.py` (entry: `lucid-run-job`).
- Production wrappers: `LUCiD/lucid/production/`; adapters: `platform/adapters.py`, selected by
  `CLUSTER=` in `user_paths.sh`.
- Schema: [dataset schema](../../reference/dataset-schema.md).

## S3DF (SLAC, SLURM)

Pull the image to a data or group area, not home (e.g.
`/sdf/data/<facility>/<user>/software/images/lucid.sif`), and copy
`templates/user_paths.s3df.sh.template`. S3DF nodes, batch nodes included, reach GitHub, so
`python scripts/download_data.py` works anywhere. The binds every `apptainer exec` here needs are
the template's `APPTAINER_BINDS` (`/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs`).

**Template defaults.** The shipped S3DF template reserves ~40 GB (`DEFAULT_MEMORY=39936`) and a
23 h wall ceiling (`DEFAULT_TIME`) per job — the ceiling, not the expected ~1 h runtime. Its
`SLURM_PARTITION="ampere"` and `DEFAULT_GPUS="1"` suit the SIREN training stage; every dataprod
job requests `DEFAULT_GPUS` GPUs, so for a CPU dataset production set `DEFAULT_GPUS="0"` and a
CPU partition (next). `SLURM_ACCOUNT` is your own `<facility>:<repo>`.

### Spreading jobs across roma + milano

The two CPU batch partitions are `roma` (130 nodes) and `milano` (272 nodes). To use both, set
`SLURM_PARTITION` in your `user_paths.sh` to a comma list weighted by node count; the dataprod
fan-out round-robins each sub-job onto one of them in that proportion:

```bash
export SLURM_PARTITION="roma:130,milano:272"
```

Refresh the counts with `sinfo -h -p <part> -o %D | awk '{s+=$1} END{print s}'`.

The split is fixed per sub-job at generation time because an account's SLURM associations are
per partition (the same account on `roma` and on `milano` are distinct associations). As a result
SLURM rejects a single multi-partition request (`--partition=roma,milano`), and `scontrol` cannot
move an already-queued job to the other partition — both fail with an invalid account/partition
error. To rebalance a wave that is already queued, `scancel` it and resubmit.

### The reference environment (develop.sif)

The paper's numbers and the test gate come from S3DF's `/sdf/group/neutrino/images/develop.sif`
(`deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim`: Python 3.10.12, jax 0.5.3). The image
does **not** contain flax or optax, so LUCiD needs one install step inside it, into a virtual
environment that sees the image's packages:

```bash
IMAGE=/sdf/group/neutrino/images/develop.sif
VENV=/sdf/data/<facility>/<you>/lucid-venv      # a data or group path
export APPTAINERENV_PYTHONNOUSERSITE=1          # keep ~/.local out (below)
apptainer exec -B /sdf "$IMAGE" python3 -m venv --system-site-packages "$VENV"
apptainer exec -B /sdf "$IMAGE" "$VENV/bin/pip" install -e . -c tests/constraints.txt
apptainer exec --nv -B /sdf "$IMAGE" "$VENV/bin/python" examples/hello_simulate.py
```

Never `pip install --user` into the image: that lands in `~/.local`, which every job of the
account then picks up. A venv made with `--system-site-packages` still puts `~/.local` first on
`sys.path`, so anything already there (a second jax, say) shadows both the venv and the image;
`PYTHONNOUSERSITE=1` takes it out. A dedicated LUCiD image will replace this step.

`ghcr.io/cider-ml/lucid` is the *production* image (GENIE + PhotonSim), not the reference for
published numbers.

### Running the JAX stack on a GPU

The production container's baked `jaxlib` is **CPU-only** — `jax.devices()` returns `CpuDevice`
even on a GPU node with `--nv`. The CUDA build is not in the `.sif`; it lives in a separate
user-site env that gets layered onto the container at run time:

```
LUCID_ENV_BASE=/sdf/data/<facility>/<user>/python_envs/lucid
```

This holds the `jax-cuda12-plugin` + `nvidia-*-cu12` wheels matched to the container's
`jax 0.4.38`, plus `optax`. Inject it with `APPTAINERENV_PYTHONUSERBASE` (this maps to
`PYTHONUSERBASE` inside the container, so its `site-packages` is added to the user-site path) and
add `--nv`. For a batch job the same two ingredients apply:

```bash
APPTAINERENV_PYTHONUSERBASE=/sdf/data/<facility>/<user>/python_envs/lucid \
APPTAINERENV_PYTHONPATH="" \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs \
    -B "$PWD/LUCiD:/opt/LUCiD" \
    "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 /opt/LUCiD/<your_script>.py
```

`jax.devices()` then returns `CudaDevice`. To run against a bind-mounted checkout *and* the GPU
env, put the checkout's root first on `sys.path` so its `lucid` wins over any `lucid` installed in
the env. Request a GPU node via `--partition=turing` (RTX 2080 Ti) or `ampere`/`ada`/`hopper`; the
SIREN GPU training is the only *production* GPU path, and it is JAX (flax + optax).

The image's `jax 0.4.38` (pinned in `container/Dockerfile`, installed with `--no-deps`) is outside
the `jax>=0.5.3,<0.6` range `pyproject.toml` declares for the library.

## NERSC (Perlmutter, SLURM)

Perlmutter runs SLURM, so this reuses the SLURM planning/fan-out wholesale. Only the per-cluster
`NerscAdapter` (`platform/adapters.py`, selected by `CLUSTER=nersc`) differs from S3DF, in three
ways:

1. **Submit directives** are `--qos` + `--constraint=cpu|gpu`, never `--partition` (Perlmutter
   rejects `--partition`).
2. **Container runtime** is the cvmfs OSG apptainer (Perlmutter ships no system
   apptainer/singularity). `--nv` is only added for GPU jobs.
3. **Accounts** split CPU vs GPU: `SLURM_ACCOUNT` is charged for CPU jobs and `SLURM_ACCOUNT_GPU`
   for GPU jobs (set both to your project's allocations).

### Filesystem layout

| Path | Use |
|---|---|
| `/global/u1/<i>/<user>/<dir>/` (= `/global/homes/...`) | Source checkouts (`LUCiD`, `PhotonSim`). Home quota. |
| `/global/cfs/cdirs/<project>/users/<user>/software/images/` | Container `.sif` (multi-GB). CFS is backed up and **not purged**. |
| `/global/cfs/cdirs/<project>/users/<user>/<dir>/` | `OUTPUT_BASE_PATH` — dataset-production outputs (`<detector>/<block>/<split>/config_NN/`). |
| `/global/cfs/cdirs/<project>/users/<user>/<siren-dir>/` | `SIREN_OUTPUT_BASE_PATH` — `training_inputs/`, `smax_parametrization/`, plus `timing_results/`, `training_stable/`, `training_tests/`. |
| `$SCRATCH` (`/pscratch/sd/<i>/<user>`) | Scratch; **purged** — only use for the apptainer pull cache/tmp, never outputs or the image. |

`<i>` is the first letter of `<user>`. The template fills in example project and directory
names; any directories on the same filesystems work.

> **Why CFS, not scratch.** `$SCRATCH` is periodically purged. The `.sif` image and all outputs
> live on CFS so they survive. Scratch is used only as `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR`
> during the one-time image pull.

### One-time setup

```bash
# Clone both repos (gh auth login, or plain git clone with https), in your checkout directory
gh repo clone cesarjesusvalls/PhotonSim
gh repo clone CIDeR-ML/LUCiD
cd LUCiD

# Pull the container image to CFS. Perlmutter has no system apptainer; use
# the cvmfs OSG build, and point its cache/tmp at scratch (not home).
export APPTAINER_BIN=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer_tmp"
IMAGES=/global/cfs/cdirs/<project>/users/$(whoami)/software/images
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$IMAGES"
"$APPTAINER_BIN" pull "$IMAGES/lucid.sif" docker://ghcr.io/cider-ml/lucid:latest

# Configure paths
cd lucid/production/platform
cp templates/user_paths.nersc.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh        # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, the accounts, ...
```

`user_paths.sh` defaults to **CPU** (`SLURM_ACCOUNT`, `--constraint=cpu`, `DEFAULT_GPUS=0`). GPU
stages flip to `SLURM_ACCOUNT_GPU` + `--constraint=gpu` automatically when a job requests a GPU.

> **Always bind the local LUCiD (`LUCID_DEV_PATH`).** The LUCiD baked into the container image
> can lag your active branch, so jobs should run your checkout, not the frozen copy. The NERSC
> template sets `LUCID_DEV_PATH` (to a placeholder you edit); leave it set unless you
> specifically want to test the baked image.

The test and full-production commands are the [common ones](#submit-one-test-run-one-job-per-config-2-events-each),
with `-o` under your CFS output area. Every `apptainer exec` above becomes `"$APPTAINER_BIN" exec`
(the cvmfs build), with the template's `APPTAINER_BINDS` — the dev-loop rebuild and the image
refresh included:

```bash
export APPTAINER_BIN=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer_cache" APPTAINER_TMPDIR="$SCRATCH/.apptainer_tmp"
"$APPTAINER_BIN" pull --force "$LUCID_IMAGE_PATH" docker://ghcr.io/cider-ml/lucid:latest
```

### SIREN-input scan (electrons)

> Run `generate_jobs.py` with **`/usr/bin/python3.11`**: it's a host-side submitter (it calls
> `sbatch`, which only works on the login node, not in the container), and the login node's bare
> `python3` (3.6.15) is too old for the script.

**Submission modes** — same work units, different SLURM packaging:

- *(default)* one sbatch per sub-job; simple, fine for small scans.
- `--array` — one job array per cell; far fewer submissions, self-recovering.
- `--pack[=N]` — `N` whole-node `regular` jobs, each running its units in parallel across the
  node's cores (in-container `xargs` pool). **Use this when the `shared` partition is
  oversubscribed** — single-core `shared` jobs get starved waiting for fractional slots, whereas a
  handful of whole-node `regular` jobs schedule and then burst 128 units at once. NERSC-only.

Recommended flow (electrons), smoke test first:

```bash
cd lucid/production/siren_inputs
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

Recovery tool split: **`recover.sh`** drives `--array`/`--pack` recovery (output-driven —
deletes ROOTs lacking the `OpticalPhotons` key, then resubmits only those). `resubmit_failed`
(through `platform/run_in_container.sh`) is for the per-job (default) mode used on S3DF/LXPLUS and
does **not** understand array/pack. Note both `recover.sh` and `merge.sh` take the
`…/training_inputs` subdir, not `$SIREN_OUTPUT_BASE_PATH`.

Cells land under `$SIREN_OUTPUT_BASE_PATH/training_inputs/<material>/<particle>/<E>MeV/`.

### Not yet wired: `train_siren` (GPU)

`NerscAdapter.render_train_run` raises `NotImplementedError` on purpose. The GPU-training defaults
in `siren_inputs/train_siren/generate_jobs.py` still carry S3DF values (a partition and an
account). Before GPU training runs here, those need Perlmutter values (`SLURM_ACCOUNT_GPU` +
`--constraint=gpu`) and a NERSC train-run renderer. The CPU stages (dataprod, siren_inputs, smax)
are fully wired; the rest of the pipeline is unaffected.

## LXPLUS (CERN, HTCondor)

### Filesystem layout (typical LXPLUS account)

| Path | Use |
|---|---|
| `/afs/cern.ch/work/<i>/<user>/DIFFSIM/` | Source checkouts (`LUCiD`, `PhotonSim`). 100 GB quota. **Per-job logs land here** (`LOG_BASE_PATH`, `DIFFSIM/logs`) — see the gotcha below. |
| `/eos/user/<i>/<user>/DIFFSIM/containers/` | Container `.sif` (multi-GB; AFS work is too small). |
| `/eos/user/<i>/<user>/DIFFSIM/output/` | `OUTPUT_BASE_PATH`: the job outputs. |

`<i>` is the first letter of `<user>`; these are the template's defaults
(`user_paths.lxplus.sh.template`, built from `$USER`).

> **LXPLUS HTCondor gotcha — submit-description logs must NOT be on EOS.** Standard LXPLUS
> schedds (bigbird*) reject submit descriptions whose `output`, `error`, `log`, or `executable`
> paths point at `/eos`. The EosSubmit-schedd workaround requires the executable on EOS too — ours
> is `/usr/bin/apptainer`, so it doesn't apply. The `HTCondorAdapter` therefore writes those three
> log fields under `LOG_BASE_PATH` (AFS in the template), mirroring each cell's path relative to
> `OUTPUT_BASE_PATH`, while the job outputs themselves go to `OUTPUT_BASE_PATH` (EOS). The
> mirroring needs the cell to lie under `OUTPUT_BASE_PATH`: if you set `SIREN_OUTPUT_BASE_PATH`
> (commented out in the template) outside it, the logs fall back to the cell directory on EOS and
> the schedd rejects the submission.

### One-time setup

```bash
# Clone both repos
cd /afs/cern.ch/work/$(whoami | head -c 1)/$(whoami)/DIFFSIM    # or wherever
git clone https://github.com/cesarjesusvalls/PhotonSim.git
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD

# Pull the container image to EOS
mkdir -p /eos/user/$(whoami | head -c 1)/$(whoami)/DIFFSIM/containers
apptainer pull \
    /eos/user/$(whoami | head -c 1)/$(whoami)/DIFFSIM/containers/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

# Configure paths
cd lucid/production/platform
cp templates/user_paths.lxplus.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh        # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, ...
```

For the dev loop, set the binds in `user_paths.sh`; each generated `.sub` then carries them in its
`SingularityBind` classad:

```bash
export LUCID_DEV_PATH="/afs/cern.ch/work/$(whoami | head -c 1)/$(whoami)/DIFFSIM/LUCiD"
export PHOTONSIM_DEV_PATH="/afs/cern.ch/work/$(whoami | head -c 1)/$(whoami)/DIFFSIM/PhotonSim"
```

### SIREN-input scan (electrons, the first thing to run)

The generator runs inside the container (`run_in_container.sh`, which has the Python stack); with
`-s` the wrapper submits what it prepared with `condor_submit` on the host.

```bash
cd lucid/production/siren_inputs
RUN=../platform/run_in_container.sh

# 1. Prepare (no submission). Inspect the rendered .sub before queueing.
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -t

# 2. Submit a single smoke-test cell.
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -t -s
condor_q $USER

# 3. Once the 1-cell test lands and looks right, submit the full 339-cell scan.
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -s

# 4. After each drain wave, recover preempted/failed sub-jobs (idempotent;
#    uproot is in the container, condor_submit on the host).
$RUN lucid.production.siren_inputs.resubmit_failed $OUTPUT_BASE_PATH --list -s

# 5. When missing == 0, hadd per-cell sub-job ROOTs.
./merge.sh $OUTPUT_BASE_PATH
```

(With `SIREN_OUTPUT_BASE_PATH` set, the cells are under `$SIREN_OUTPUT_BASE_PATH/training_inputs/`
instead: pass that to steps 4 and 5, and see the log caveat above.)

Each cell produces:

```
$OUTPUT_BASE_PATH/water/e-/<E>MeV/
├── photonsim.root              ← created by merge.sh
├── photonsim_config.json
├── submit.sub or submit_job_*.sub
└── (during the run) output_job_*.root, deleted after merge
```

and its `job-NNN-<cluster>.<proc>.out / .err / .log` under the mirrored
`$LOG_BASE_PATH/water/e-/<E>MeV/`.

### Walltime — picking a JobFlavour

HTCondor walltime is set via `+JobFlavour` (CERN convention). The default in
`user_paths.sh.template` is `workday` (8 h). The SIREN/smax planner aims for ~1 h per sub-job, so
this is generous; it was picked because some high-energy electron cells (≥ 9 GeV) overshoot a
tighter 2 h cap and get killed by `SYSTEM_PERIODIC_REMOVE`. Override per invocation with
`-P longlunch` (or whatever) on the CLI if you want to be stricter.

| Flavour | Walltime cap |
|---|---|
| espresso | 20 min |
| microcentury | 1 h |
| longlunch | 2 h |
| workday | 8 h |
| tomorrow | 1 d |
| testmatch | 3 d |
| nextweek | 1 w |

### HTCondor commands

```bash
# Live status
condor_q $USER -nobatch
lucid/production/dataprod/scripts/monitor_jobs.htcondor.sh -w   # watch mode

# Cancel everything I own
condor_rm $USER

# Single-job log peek
ls $LOG_BASE_PATH/water/e-/100MeV/*.log
```
