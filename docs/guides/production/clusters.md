# Produce datasets on a cluster: S3DF, NERSC, LXPLUS

The runbook for datasets and SIREN inputs on SLURM at S3DF (SLAC), SLURM at NERSC (Perlmutter)
and HTCondor at LXPLUS (CERN). The common workflow comes first, then what each site changes.
How the code targets the clusters is in [how cluster targeting works](cluster-abstraction.md).

## The common workflow

### One-time setup

```bash
git clone https://github.com/cesarjesusvalls/PhotonSim.git
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD

# the image; where to keep the .sif and which apptainer to call depend on the site (below)
apptainer pull <image-dir>/lucid.sif docker://ghcr.io/cider-ml/lucid:latest

cd lucid/production/platform
cp templates/user_paths.<cluster>.sh.template user_paths.sh    # s3df, nersc or lxplus
${EDITOR:-vim} user_paths.sh          # LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, account, ...
```

`user_paths.sh` is yours and git-ignored. Run `python scripts/download_data.py` once in the
checkout ([install](../../getting-started/install.md#download-the-example-data)).

### Submit one test run (one job per config, 2 events each)

```bash
cd lucid/production/dataprod/scripts
echo "y" | ./submit_all_configs.sh -D SK_WAND -t -s -o <OUT>/<date>_test
```

`submit_all_configs.sh` options: `-D` detector, `-t` test wave, `-s` submit (otherwise only
prepare), `-d` dry run, `-p` config glob under `configs/` (default all), `-o` output root,
`-P` partition, `-g` one GPU per job.

- `-D` is required for the `GeV` configs, and must be `SK_WAND` or `HK_WAND`: other detectors
  have no readout and are refused ("no 'digitizer' block"). The `Solar`, `SN` and `Test` configs
  declare `HK` and use it whatever `-D` says.
- `-t` writes under `<OUT>/<detector>/_test/` ([layout](index.md#where-a-batch-run-writes)).
  Each directory holds the four `{sensor,hits,step,labl}/wc_*_0000.h5` when its job finishes.
- The four `GeV/02_mu*.json` configs share the directory `config_02/`. After the first, the
  fan-out refuses the others rather than overwrite it. Run a variant alone with its own `-o`,
  for example `-p 'GeV/02_mu_smoke.json'`.

### Full production

Drop `-t`:

```bash
echo "y" | ./submit_all_configs.sh -D SK_WAND -s -o <OUT>/<name>
```

Each job writes its own batch (`file_index = job_id − 1`) into the shared
`{sensor,hits,step,labl}/` of its config directory, which is one dataset. To change the number
of jobs, run the fan-out directly with `-N` (`--n-jobs-override`), or edit `nominal_*` or
`target_seconds_per_job` in a copy of the config. The fan-out's other flags (`-j` to top up a
dataset, `--force-overwrite`) are in
[`lucid/production/dataprod/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/lucid/production/dataprod/README.md).

### Sizing a production run

The fan-out splits a config's events into equal single-core jobs, from two config fields:
`seconds_per_event` (measured wall time per event) and `target_seconds_per_job` (3600 in the
shipped configs).

```
events_per_job = floor(target_seconds_per_job / seconds_per_event)
n_jobs         = ceil(nominal_events / events_per_job)
core-hours     ≈ nominal_events × seconds_per_event / 3600
```

| config | split | events | `seconds_per_event` | events/job | jobs | core-hours |
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

A supernova config (`SN/01`) runs one burst per job, so `n_jobs = nominal_test` (5).
Recompute after editing a config.

Disk use per event depends strongly on the detector, energy and pile-up. Measure it: run one
full (not `-t`) batch, then

```bash
du -sh <OUT>/<detector>/<block>/<split>/config_NN/{sensor,hits,step,labl}/wc_*_0000.h5
```

and divide by that batch's `n_events`. Test batches overstate it (fixed per-file overhead).

### What each job does

One `apptainer exec $LUCID_IMAGE_PATH lucid-run-job ...` runs the whole chain inside the
image: GEANT4 macro, PhotonSim, LUCiD writer (sntools first for a supernova config, `gevgen` and
`gntpc` first for a GENIE config). `GENIE_XSEC_FILE` in `user_paths.sh` names the image's
`G18_10a_02_11b` splines.

### Verify and resume a run

Preemption and node failures leave some jobs incomplete. A summary (inside the image, which has
h5py; add the site's binds):

```bash
cd lucid/production/dataprod/scripts
apptainer exec -B "$APPTAINER_BINDS" "$LUCID_IMAGE_PATH" python3 verify_jobs.py <OUT>/<detector>
```

Resubmit the failures, and repeat after each wave until nothing is listed:

```bash
../../platform/run_in_container.sh lucid.production.dataprod.scripts.verify_jobs \
    <OUT>/<detector> --list -s                       # drop -s to preview
```

What "complete" means and a post-run checklist:
[verifying and resuming](cluster-abstraction.md#verifying-and-resuming-cluster-agnostic).

On SLURM, each job's log sits in its config directory, beside the output it should have
written: `config_NN/job_NNNNNN-<slurm_job_id>.{out,err}`. LXPLUS logs go elsewhere
([below](#lxplus-cern-htcondor)).

### Useful commands

```bash
# from lucid/production/dataprod/scripts
./monitor_jobs.slurm.sh -w          # S3DF, NERSC
./monitor_jobs.htcondor.sh -w       # LXPLUS

# SLURM: cancel all production jobs (names start with photonsim_)
squeue -u $USER -h -o '%i %j' | awk '$2 ~ /^photonsim_/ {print $1}' | xargs -r scancel

# look at one output directory (viewer/README.md covers SSH tunnels)
python3 /path/to/LUCiD/viewer/serve_viewer.py <OUT>/SK_WAND/GeV/test/config_02
```

### Dev loop: bind-mount your checkouts

Set `LUCID_DEV_PATH` and `PHOTONSIM_DEV_PATH` in `user_paths.sh` to run your checkouts in every
job; building PhotonSim against the image is described in
[the dev loop](cluster-abstraction.md#dev-loop-run-your-own-checkouts-in-the-batch-jobs). One job
by hand:

```bash
apptainer exec -B "$PWD/LUCiD:/opt/LUCiD" -B "$PWD/PhotonSim:/opt/PhotonSim" "$LUCID_IMAGE_PATH" \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu.json \
                  --detector SK_WAND --output-dir /tmp/out --job-id 1 --test
```

### Refreshing the image

```bash
apptainer pull --force "$LUCID_IMAGE_PATH" docker://ghcr.io/cider-ml/lucid:latest
```

With `PHOTONSIM_DEV_PATH` set, rebuild the host PhotonSim against the new image afterwards
([dev loop](cluster-abstraction.md#dev-loop-run-your-own-checkouts-in-the-batch-jobs)).

## S3DF (SLAC, SLURM)

Keep the image in a data or group area, not in home (for example
`/sdf/data/<facility>/<user>/software/images/lucid.sif`), and copy
`templates/user_paths.s3df.sh.template`. Every `apptainer exec` needs the template's
`APPTAINER_BINDS` (`/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs`). Batch nodes reach GitHub, so
`download_data.py` works anywhere.

The template requests 39936 MB, 23 h and one GPU on `ampere` per job, which suits SIREN training.
Every dataset job requests `DEFAULT_GPUS` GPUs, so for dataset production set
`DEFAULT_GPUS="0"` and a CPU partition. `SLURM_ACCOUNT` is your `<facility>:<repo>`.

### Spreading jobs across roma and milano

Set `SLURM_PARTITION` to a weighted list and the fan-out assigns each job to one partition in
that proportion:

```bash
export SLURM_PARTITION="roma:130,milano:272"     # weights: node counts
```

(Count nodes with `sinfo -h -p <part> -o %D | awk '{s+=$1} END{print s}'`.) The partition is
fixed per job at generation time: SLURM rejects `--partition=roma,milano` for an account whose
associations are per partition, and `scontrol` cannot move a queued job between them. To
rebalance a queued wave, cancel and resubmit it.

### The reference environment (develop.sif)

The paper's numbers and the test gate come from `/sdf/group/neutrino/images/develop.sif`
(Python 3.10.12, jax 0.5.3). The image has no flax or optax, so install LUCiD into a virtual
environment that sees the image's packages:

```bash
IMAGE=/sdf/group/neutrino/images/develop.sif
VENV=/sdf/data/<facility>/<you>/lucid-venv      # a data or group path
export APPTAINERENV_PYTHONNOUSERSITE=1          # keep ~/.local off sys.path
apptainer exec -B /sdf "$IMAGE" python3 -m venv --system-site-packages "$VENV"
apptainer exec -B /sdf "$IMAGE" "$VENV/bin/pip" install -e . -c tests/constraints.txt
apptainer exec --nv -B /sdf "$IMAGE" "$VENV/bin/python" examples/hello_simulate.py
```

Do not `pip install --user` inside the image: it goes to `~/.local`, which every job of the
account then sees, and which precedes the venv on `sys.path` (hence `PYTHONNOUSERSITE=1`).

The production image `ghcr.io/cider-ml/lucid` is not the reference environment for published
numbers.

### Running the JAX stack on a GPU

The production image's `jaxlib` is CPU-only (`jax.devices()` shows `CpuDevice` even with
`--nv`), and its jax is 0.4.38, outside the `>=0.5.3,<0.6` range the library declares. GPU work
layers a user-site environment holding `jax-cuda12-plugin`, the matching `nvidia-*-cu12` wheels
and optax onto the image, through `APPTAINERENV_PYTHONUSERBASE`, plus `--nv`:

```bash
APPTAINERENV_PYTHONUSERBASE=/sdf/data/<facility>/<user>/python_envs/lucid \
APPTAINERENV_PYTHONPATH="" \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs -B "$PWD/LUCiD:/opt/LUCiD" \
    "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 /opt/LUCiD/<your_script>.py
```

`jax.devices()` then shows `CudaDevice`. Put the checkout first on `sys.path` so its `lucid`
wins over one installed in the environment. GPU partitions: `turing` (RTX 2080 Ti), `ampere`,
`ada`, `hopper`. In production, only SIREN training uses a GPU.

## NERSC (Perlmutter, SLURM)

`CLUSTER=nersc` selects `NerscAdapter`, which differs from S3DF in three ways:

1. Jobs use `--qos` and `--constraint=cpu|gpu`; Perlmutter rejects `--partition`.
2. Apptainer comes from cvmfs (Perlmutter has no system Apptainer); `--nv` is added only for
   GPU jobs.
3. CPU jobs charge `SLURM_ACCOUNT`, GPU jobs `SLURM_ACCOUNT_GPU`.

| path | use |
|---|---|
| `/global/u1/<i>/<user>/<dir>/` (`/global/homes/...`) | the `LUCiD` and `PhotonSim` checkouts |
| `/global/cfs/cdirs/<project>/users/<user>/software/images/` | the `.sif` (CFS is backed up and not purged) |
| `/global/cfs/cdirs/<project>/users/<user>/<dir>/` | `OUTPUT_BASE_PATH` |
| `/global/cfs/cdirs/<project>/users/<user>/<siren-dir>/` | `SIREN_OUTPUT_BASE_PATH` (`training_inputs/`, `smax_parametrization/`, ...) |
| `$SCRATCH` (`/pscratch/sd/<i>/<user>`) | purged: only the Apptainer cache and temp during the pull |

`<i>` is the first letter of `<user>`.

### One-time setup

```bash
gh repo clone cesarjesusvalls/PhotonSim
gh repo clone CIDeR-ML/LUCiD
cd LUCiD

export APPTAINER_BIN=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer_tmp"
IMAGES=/global/cfs/cdirs/<project>/users/$(whoami)/software/images
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$IMAGES"
"$APPTAINER_BIN" pull "$IMAGES/lucid.sif" docker://ghcr.io/cider-ml/lucid:latest

cd lucid/production/platform
cp templates/user_paths.nersc.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh        # LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, both accounts, ...
```

The template defaults to CPU jobs (`SLURM_ACCOUNT`, `--constraint=cpu`, `DEFAULT_GPUS=0`); a job
that requests a GPU switches to `SLURM_ACCOUNT_GPU` and `--constraint=gpu`. It also sets
`LUCID_DEV_PATH` to a placeholder: point it at your checkout so jobs run your branch rather than
the image's copy.

The test and production commands are the [common ones](#submit-one-test-run-one-job-per-config-2-events-each)
with `-o` under CFS. Every `apptainer` above becomes `"$APPTAINER_BIN"`.

### SIREN-input scan (electrons)

`generate_jobs.py` submits from the login node (it calls `sbatch`), so run it with
`/usr/bin/python3.11`; the login node's `python3` is too old. Packaging options:

- default: one `sbatch` per job.
- `--array`: one job array per cell.
- `--pack[=N]`: `N` whole-node `regular` jobs (default 10), each running many units in parallel.
  Use it when the `shared` QOS is oversubscribed. NERSC only.

```bash
cd lucid/production/siren_inputs
/usr/bin/python3.11 generate_jobs.py -c configs/water_el.json --pack -t -s   # one-cell smoke test
squeue -u $USER
/usr/bin/python3.11 generate_jobs.py -c configs/water_el.json --pack -s      # full scan
./recover.sh -c configs/water_el.json        # after each wave, until "0 invalid"
./merge.sh $SIREN_OUTPUT_BASE_PATH/training_inputs
```

`recover.sh` recovers `--array` and `--pack` runs: it deletes ROOT files without the
`OpticalPhotons` tree and resubmits those units. `resubmit_failed` is for the default per-job mode
and does not understand arrays or packs. Both `recover.sh` and `merge.sh` take the
`training_inputs` directory. Cells land in
`$SIREN_OUTPUT_BASE_PATH/training_inputs/<material>/<particle>/<E>MeV/`.

### Not yet wired: `train_siren` (GPU)

SIREN training does not run on NERSC yet: `NerscAdapter.render_train_run` raises
`NotImplementedError`, and the GPU defaults in `siren_inputs/train_siren/generate_jobs.py` are
S3DF values. Dataset production, SIREN inputs and `s_max` work.

## LXPLUS (CERN, HTCondor)

| path | use |
|---|---|
| `/afs/cern.ch/work/<i>/<user>/DIFFSIM/` | checkouts (100 GB), and the job logs (`LOG_BASE_PATH`, `DIFFSIM/logs`) |
| `/eos/user/<i>/<user>/DIFFSIM/containers/` | the `.sif` |
| `/eos/user/<i>/<user>/DIFFSIM/output/` | `OUTPUT_BASE_PATH` |

These are the defaults of `user_paths.lxplus.sh.template`.

The standard LXPLUS schedds reject a submit file whose `output`, `error`, `log` or `executable`
is on `/eos`. The adapter therefore writes the logs under `LOG_BASE_PATH` (AFS), mirroring each
job's path relative to `OUTPUT_BASE_PATH`, and the outputs to EOS. This needs the job directory
under `OUTPUT_BASE_PATH`: a `SIREN_OUTPUT_BASE_PATH` outside it puts the logs on EOS and the
submission is rejected.

### One-time setup

```bash
cd /afs/cern.ch/work/$(whoami | head -c 1)/$(whoami)/DIFFSIM
git clone https://github.com/cesarjesusvalls/PhotonSim.git
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD

mkdir -p /eos/user/$(whoami | head -c 1)/$(whoami)/DIFFSIM/containers
apptainer pull /eos/user/$(whoami | head -c 1)/$(whoami)/DIFFSIM/containers/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

cd lucid/production/platform
cp templates/user_paths.lxplus.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh
```

For the dev loop, set `LUCID_DEV_PATH` and `PHOTONSIM_DEV_PATH` to the AFS checkouts; each
submit file then carries them in its `SingularityBind` classad.

### SIREN-input scan (electrons, the first thing to run)

The generator runs inside the image through `run_in_container.sh`; with `-s` the wrapper
submits with `condor_submit` on the host.

```bash
cd lucid/production/siren_inputs
RUN=../platform/run_in_container.sh
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -t       # prepare; inspect the .sub
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -t -s    # one-cell smoke test
condor_q $USER
$RUN lucid.production.siren_inputs.generate_jobs -c configs/water_el.json -s       # all 339 cells
$RUN lucid.production.siren_inputs.resubmit_failed $OUTPUT_BASE_PATH --list -s     # after each wave
./merge.sh $OUTPUT_BASE_PATH                                                       # when nothing is missing
```

With `SIREN_OUTPUT_BASE_PATH` set, the cells are under `$SIREN_OUTPUT_BASE_PATH/training_inputs/`:
pass that to the last two steps (and see the log restriction above). Each cell
`$OUTPUT_BASE_PATH/water/e-/<E>MeV/` ends with `photonsim.root` (made by `merge.sh`),
`photonsim_config.json` and its submit files; its logs are under
`$LOG_BASE_PATH/water/e-/<E>MeV/`.

### Walltime

HTCondor walltime is the `+JobFlavour`. The template uses `workday` (8 h); jobs aim at about
1 h, but some electron cells at 9 GeV and above exceed 2 h. Override with `-P <flavour>`.

| flavour | limit |
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
condor_q $USER -nobatch
lucid/production/dataprod/scripts/monitor_jobs.htcondor.sh -w
condor_rm $USER
ls $LOG_BASE_PATH/water/e-/100MeV/*.log
```
