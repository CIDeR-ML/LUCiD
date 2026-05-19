# LXPLUS Quickstart — Production on HTCondor

Concise runbook for the CERN LXPLUS deployment. Pairs with
[QUICKSTART_S3DF.md](QUICKSTART_S3DF.md) (same pipeline, SLURM at SLAC)
and [CLUSTER_ABSTRACTION.md](CLUSTER_ABSTRACTION.md) (architecture).

## Filesystem layout (typical LXPLUS account)

| Path | Use |
|---|---|
| `/afs/cern.ch/work/<i>/<user>/DIFFSIM/` | Source checkouts (`LUCiD`, `PhotonSim`). 100 GB quota. **Outputs + per-job logs land here** — see the gotcha below. |
| `/eos/user/<i>/<user>/DIFFSIM/containers/` | Container `.sif` (multi-GB; AFS work is too small). |
| `/eos/user/<i>/<user>/DIFFSIM/` | EOS user space; available for bulk staging outside the batch flow. |

`<i>` is the first letter of `<user>` (e.g. `c` for `cjesus`).

> **LXPLUS HTCondor gotcha — outputs must NOT be on EOS.** Standard
> LXPLUS schedds (bigbird*) reject submit descriptions whose `output`,
> `error`, `log`, or `executable` paths point at `/eos`. The
> EosSubmit-schedd workaround requires the executable on EOS too — ours
> is `/usr/bin/apptainer`, so it doesn't apply. The `user_paths.sh.template`
> therefore defaults `OUTPUT_BASE_PATH` to AFS. For an electron SIREN
> scan that comes out to ≈3 GB total, well under the 100 GB AFS quota.
> If you outgrow it, see CERN batchdocs' EOS transfer-plugin workaround
> (not wired into the adapter today).

## One-time setup

```bash
# Clone both repos
cd /afs/cern.ch/work/c/$(whoami)/DIFFSIM    # or wherever
git clone https://github.com/cesarjesusvalls/PhotonSim.git
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD

# Pull the container image to EOS
mkdir -p /eos/user/$(whoami | head -c 1)/$(whoami)/DIFFSIM/containers
apptainer pull \
    /eos/user/$(whoami | head -c 1)/$(whoami)/DIFFSIM/containers/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

# Configure paths
cd lucid/production/jobs
cp user_paths.lxplus.sh.template user_paths.sh
${EDITOR:-vim} user_paths.sh        # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, ...
```

Set `LUCID_DEV_PATH` / `PHOTONSIM_DEV_PATH` in `user_paths.sh` if you want
the host checkouts to shadow the container's baked copies (dev loop).

## SIREN-input scan (electrons, the first thing to run)

```bash
cd lucid/production/jobs/siren_inputs

# 1. Prepare (no submission). Inspect the rendered .sub before queueing.
python3 generate_jobs.py -c configs/water_el.json -t

# 2. Submit a single smoke-test cell.
python3 generate_jobs.py -c configs/water_el.json -t -s
condor_q $USER

# 3. Once the 1-cell test lands and looks right, submit the full 339-cell scan.
python3 generate_jobs.py -c configs/water_el.json -s

# 4. After each drain wave, recover preempted/failed sub-jobs (idempotent).
./resubmit_failed.sh $OUTPUT_BASE_PATH

# 5. When missing == 0, hadd per-cell sub-job ROOTs.
./merge.sh $OUTPUT_BASE_PATH
```

Each cell produces:

```
$OUTPUT_BASE_PATH/water/e-/<E>MeV/
├── photonsim.root              ← created by merge.sh
├── photonsim_config.json
├── submit.sub or submit_job_*.sub
├── job-NNN-<cluster>.<proc>.out / .err / .log
└── (during the run) output_job_*.root, deleted after merge
```

## Walltime — picking a JobFlavour

HTCondor walltime is set via `+JobFlavour` (CERN convention). The default
in `user_paths.sh.template` is `longlunch` (2 h), matching the SIREN/smax
default `target_seconds_per_job = 3600`. Larger cells (very high energy)
may need `workday` (8 h) or `tomorrow` (1 d) — override per-invocation with
`-P workday` on the CLI.

| Flavour | Walltime cap |
|---|---|
| espresso | 20 min |
| microcentury | 1 h |
| longlunch | 2 h |
| workday | 8 h |
| tomorrow | 1 d |
| testmatch | 3 d |
| nextweek | 1 w |

## Useful commands

```bash
# Live status
condor_q $USER -nobatch
./dataprod/monitor_jobs.lxplus.sh -w           # watch mode

# Cancel everything I own
condor_rm $USER

# Single-job log peek
ls $OUTPUT_BASE_PATH/water/e-/100MeV/*.log
```

## Dev-loop — bind-mount your checkouts

Same idea as on S3DF. Export either or both in `user_paths.sh`:

```bash
export LUCID_DEV_PATH="/afs/cern.ch/work/c/$(whoami)/DIFFSIM/LUCiD"
export PHOTONSIM_DEV_PATH="/afs/cern.ch/work/c/$(whoami)/DIFFSIM/PhotonSim"
```

Each generated `.sub` adds `-B <path>:/opt/{LUCiD,PhotonSim}` to its
`apptainer exec`. Unset to go back to the frozen container.

The same caveat as S3DF applies: if you change PhotonSim source under
`LUCID_DEV_PATH`, you must rebuild the host PhotonSim against the
container's GEANT4 / ROOT before re-submitting:

```bash
apptainer exec -B /afs,/eos,/cvmfs \
    -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" \
    bash -lc 'cmake --build /opt/PhotonSim/build -j 4'
```

## Refreshing the container image

```bash
apptainer pull --force "$LUCID_IMAGE_PATH" docker://ghcr.io/cider-ml/lucid:latest
```

Then rebuild PhotonSim against the new image as above if dev-loop binds
are set.

## Spatial overlap cache

Same as on S3DF — `$LUCID_OVERLAP_CACHE_DIR` → package dir if writable →
`$XDG_CACHE_HOME/lucid/spatial_overlap_integrals` (default
`~/.cache/lucid/...`). With dev-loop binds the host package dir is
writable, so the cache lives next to the source. No setup needed.

## Paths / conventions

- Configs: `LUCiD/lucid/production/configs/dataprod_*.json` (dataprod);
  `jobs/<stage>/configs/*.json` (per-stage).
- Runner: `LUCiD/lucid/production/run_job.py` (entry: `lucid-run-job`).
- Production wrappers: `LUCiD/lucid/production/jobs/`.
- Architecture: [CLUSTER_ABSTRACTION.md](CLUSTER_ABSTRACTION.md).
