# Cluster abstraction

How the production pipeline targets multiple batch clusters (currently
SLURM on S3DF, SLURM on NERSC/Perlmutter, and HTCondor on LXPLUS) from a
single code base. The short version: shared planning + a single
per-cluster `ClusterAdapter` class, dispatched at runtime from a sticky
env var in your active `user_paths.sh`.

The runbook, with one section per cluster, is [Cluster deployment](clusters.md): S3DF (SLURM at
SLAC), NERSC (SLURM on Perlmutter; `NerscAdapter` subclasses `SlurmAdapter`: `--qos`/`--constraint`
instead of `--partition`, cvmfs apptainer, separate CPU and GPU accounts) and LXPLUS (HTCondor at
CERN).

## Architecture

```
LUCiD/lucid/production/
├── platform/                    execution plumbing, no physics (importable)
│   ├── adapters.py                  ClusterAdapter ABC, SlurmAdapter, NerscAdapter,
│   │                                HTCondorAdapter; ADAPTERS + get_adapter
│   ├── user_paths.py                source user_paths.sh → env dict
│   ├── run_in_container.sh          run a module in the container; -s submits its
│   │                                [PREPARED] lines from the host
│   ├── user_paths.sh                gitignored; user copies from one template
│   └── templates/                   user_paths.{s3df,nersc,lxplus}.sh.template
│
├── dataprod/                    the dataset line
│   ├── run_job.py                   lucid-run-job (one job)
│   ├── drivers.py, seeds.py         the event loops, the job seeds
│   ├── generators/                  Geant4 macro, GENIE, sntools
│   ├── fanout.py                    Python fan-out
│   ├── configs/                     {GeV,SN,Solar,Test}/NN_name.json
│   └── scripts/                     cluster-neutral shell + reports
│       ├── submit_all_configs.sh        fanout over a glob of configs
│       ├── verify_jobs.py               is_complete; --list for resubmission
│       ├── monitor_jobs.slurm.sh        squeue-flavoured
│       ├── monitor_jobs.htcondor.sh     condor_q-flavoured
│       └── timing / cleanup / clean_output_data helpers
│
├── siren_inputs/                SIREN training inputs
│   ├── generate_jobs.py, merge.sh, recover.sh, resubmit_failed.py   Stage 1
│   ├── siren_planning.py            Stage-1 fan-out math + is_complete_siren
│   ├── smax/ + smax_planning.py     Stage 0
│   └── train_siren/ + train_planning.py   Stage 3 (FLAG_MAP + diff resolution)
│
├── qa/                          validations, trigger_study
└── derived_views/               relabel / de-pile views of a dataset
```

### Key design choices

1. **One canonical scripts tree.** Every entrypoint lives exactly once,
   with its application (`dataprod/`, `siren_inputs/`). Neither cluster is
   privileged; the file tree has no asymmetry between SLURM and HTCondor.

2. **Cluster identity carried entirely by `user_paths.sh`.** Three env
   keys — `CLUSTER`, `CLUSTER_SUBMIT_CMD`, `APPTAINER_BINDS` — turn
   the same script into a SLURM or HTCondor invocation. Switching
   cluster = swap which `user_paths.<cluster>.sh.template` you copied
   into `user_paths.sh`.

3. **`user_paths.sh` is found from the code, not the working directory.**
   Python entry points look up `lucid/production/platform/user_paths.sh` relative to their
   own file (`--user-paths` overrides it), so they work from any directory.

4. **Bash entrypoints read env vars verbatim.** `merge.sh`,
   `recover.sh`, `analyze.sh` source `user_paths.sh` then
   expand `${APPTAINER_BINDS:-/sdf,...}`.

5. **One container hop.** `platform/run_in_container.sh <module> [args] [-s]`
   runs any generator or lister (`fanout`, the three `generate_jobs`,
   `verify_jobs`, `resubmit_failed`) inside the image with the checkout
   bound at `/opt/LUCiD`, and with `-s` submits every `[PREPARED] <path>`
   line the module prints using `$CLUSTER_SUBMIT_CMD` on the host (the
   image has no scheduler client).

## `ClusterAdapter` contract

`platform/adapters.py`:

```python
class ClusterAdapter(ABC):
    name: str                   # "slurm" / "nersc" / "htcondor"
    submit_extension: str       # "sbatch" / "sub"
    submit_cmd: str             # "sbatch" / "condor_submit"

    def __init__(self, env):    # sets self.apptainer_binds: $APPTAINER_BINDS,
        ...                     # else the cluster's _default_binds()

    def render_siren_cell(...)   -> str: ...
    def render_smax_cell(...)    -> str: ...
    def render_dataprod_job(...) -> str: ...
    def render_command_job(...)  -> str: ...   # any shell command in the container
    def render_train_run(...)    -> str: ...
```

`NerscAdapter` subclasses `SlurmAdapter` and adds `render_siren_pack` (the
`--pack` whole-node mode); its `render_train_run` raises
`NotImplementedError` (see [NERSC](clusters.md#not-yet-wired-train_siren-gpu)).

Selection:

```python
from lucid.production.platform.adapters import get_adapter

env = load_user_paths(...)
adapter = get_adapter(env)         # reads env["CLUSTER"], default "slurm"
sb_body = adapter.render_siren_cell(...)
```

`get_adapter` looks the name up in the literal `ADAPTERS` dict at the bottom of
`adapters.py`; nothing registers itself at import time.

## Verifying and resuming (cluster-agnostic)

Completion truth is shared, not per-cluster.

- **Datasets:** `dataprod/scripts/verify_jobs.py::is_complete`. A batch is done
  iff all four `wc_{sensor,hits,step,labl}_NNNN.h5` exist, open with h5py,
  carry `config/complete == True`, agree on `config/n_events`, and hold
  `n_events <= expected`. The count may be lower than requested (the trigger or
  the `min_physics_hits` selection drops events); a crash-killed job is caught
  by the `complete` flag, not by the count.
- **SIREN inputs:** `siren_inputs/siren_planning.py::is_complete_siren` (the
  ROOT file has the `OpticalPhotons` TTree).

Every stage's `verify_jobs` / `resubmit_failed` reuses these, so the resume loop
is the same on every `CLUSTER`: run the stage's lister in `--list` mode inside
the container (it needs h5py/uproot) through `platform/run_in_container.sh`,
which resubmits the listed jobs with `$CLUSTER_SUBMIT_CMD` on the host (`sbatch`
or `condor_submit`) when given `-s`:

```bash
lucid/production/platform/run_in_container.sh \
    lucid.production.dataprod.scripts.verify_jobs <OUT>/<detector> --list -s   # drop -s to preview
```

It is idempotent: rerun it after each drain wave until it lists nothing.

**Post-run checklist.**

1. **All jobs complete** — `verify_jobs` prints `failed/missing: 0`.
2. **Delivered vs requested** — a shortfall against `nominal_*` is expected for
   triggered (`GeV`, `Test`) and `min_physics_hits` (`Solar`, `SN`) configs.
3. **Dropped events are gaps** — `config/source_event_idx` (and the
   `event_NNN/` `source_event_idx` attrs) has holes rather than a dense
   `0..n-1` run; reconcile against the source ROOT's entry count for the
   selection efficiency, and do not treat the gaps as missing output.
4. **Smoke-test first** — before a full wave, run the fan-out with `-t` (one job
   per config, 2 events, under `<OUT>/<detector>/_test/`) and confirm all four
   files appear per config.

## Dev loop: run your own checkouts in the batch jobs

To validate a LUCiD or PhotonSim source change under the batch system without
rebuilding the image, set either or both in `user_paths.sh` (or the shell that
runs the fan-out):

```bash
export LUCID_DEV_PATH=/path/to/LUCiD
export PHOTONSIM_DEV_PATH=/path/to/PhotonSim
```

Every rendered job then binds them over the baked copies
(`-B <path>:/opt/LUCiD`, `-B <path>:/opt/PhotonSim`; on HTCondor through the
`SingularityBind` classad). Unset them to go back to the frozen image. Python
changes in LUCiD are picked up live; a PhotonSim **source** change needs the
host checkout rebuilt against the image's Geant4 / ROOT before resubmitting,
and so does every image refresh (otherwise the bound binary is stale and LUCiD
fails with "PhotonSim ROOT file is missing branches …"):

```bash
apptainer exec -B "$APPTAINER_BINDS" -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" bash -lc 'cmake --build /opt/PhotonSim/build -j 4'
```

**First build** of a fresh PhotonSim checkout (no `build/` yet): configure with
the flags the image uses (`container/Dockerfile`). `-DCMAKE_PREFIX_PATH=/opt/conda`
makes cmake pick conda's expat (the system one is too old for Geant4), and the
linker flags point `ld` at conda's `libstdc++` (the system one lacks the
`GLIBCXX_3.4.32` / `CXXABI_1.3.15` symbols the conda Geant4/Qt libraries need):

```bash
apptainer exec -B "$APPTAINER_BINDS" -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
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

A bound LUCiD checkout is writable, so the spatial-overlap cache lands in it
(resolution order in [container.md](container.md#5-changing-things)).

## Adding a third cluster

1. **New adapter** in `platform/adapters.py`:
   ```python
   class MyClusterAdapter(ClusterAdapter):
       name = "mycluster"
       submit_extension = "myjob"
       submit_cmd = "mysub"
       ...
   ```
2. **Add it to `ADAPTERS`** at the bottom of the same file.
3. **New `user_paths.<cluster>.sh.template`** in `platform/templates/` setting
   `CLUSTER=mycluster` plus the cluster-specific env keys.
4. **New `dataprod/scripts/monitor_jobs.<cluster>.sh`** if the queue tool isn't
   `squeue` or `condor_q`.
5. Run a `-t` smoke wave on the new cluster, then the
   [verification](#verifying-and-resuming-cluster-agnostic) above against it.

No other files change.

## What did NOT get abstracted

- Per-cluster timing-model defaults
  (`siren_planning.DEFAULT_TIME_MODEL`, dataprod `seconds_per_event`).
  These were fit on S3DF runs and may need re-fitting after the first
  large LXPLUS scan. The config schema already supports per-config
  overrides.
