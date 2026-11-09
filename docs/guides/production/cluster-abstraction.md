# How cluster targeting works

One code path in `lucid/production/` submits to SLURM on S3DF, SLURM on NERSC (Perlmutter) and
HTCondor on LXPLUS. The cluster is chosen by your `user_paths.sh`; the commands are the same on
all three. The per-site steps are in [on a cluster](clusters.md).

## Layout

```
lucid/production/
├── platform/                  execution plumbing
│   ├── adapters.py                ClusterAdapter, SlurmAdapter, NerscAdapter, HTCondorAdapter
│   ├── user_paths.py              reads user_paths.sh into an environment
│   ├── run_in_container.sh        runs a module in the image; -s submits what it prepared
│   ├── user_paths.sh              yours, git-ignored, copied from a template
│   └── templates/                 user_paths.{s3df,nersc,lxplus}.sh.template
├── dataprod/                  datasets
│   ├── run_job.py                 lucid-run-job
│   ├── fanout.py                  splits a config into jobs and writes their submit files
│   ├── configs/                   {GeV,SN,Solar,Test}/NN_name.json
│   └── scripts/                   submit_all_configs.sh, verify_jobs.py, monitor_jobs.*.sh, cleanup
├── siren_inputs/              SIREN training inputs (stage 1), smax/ (stage 0), t0/ (stage 3), train_siren/ (stage 4)
├── qa/                        validation studies
└── derived_views/             derived datasets (relabelling, de-piling)
```

## `user_paths.sh`

Three variables make a script a SLURM or an HTCondor submission: `CLUSTER` (`slurm`, `nersc`
or `htcondor`), `CLUSTER_SUBMIT_CMD` (`sbatch` or `condor_submit`) and `APPTAINER_BINDS`.
Switching cluster means copying a different template. The template also sets the image
(`LUCID_IMAGE_PATH`), the output root (`OUTPUT_BASE_PATH`), the account and partition, and the
job resources (`DEFAULT_CPUS`, `DEFAULT_MEMORY`, `DEFAULT_GPUS`, `DEFAULT_TIME`).

Python entry points find `lucid/production/platform/user_paths.sh` relative to their own file,
so they run from any directory (`--user-paths` points at another). The shell scripts source it
too.

`platform/run_in_container.sh <module> [args] [-s]` runs a generator or lister (`fanout`, the
`generate_jobs` modules, `verify_jobs`, `resubmit_failed`) inside the image, with the checkout
bound at `/opt/LUCiD`. With `-s` it submits every `[PREPARED] <path>` line the module printed,
using `$CLUSTER_SUBMIT_CMD` on the host, since the image has no scheduler client.

## Adapters

`platform/adapters.py` has one class per scheduler. Each renders the submit file of every job
kind: a dataset job, a SIREN-input cell, an `s_max` cell, a SIREN training run, and an arbitrary
command in the container. `get_adapter(env)` picks one by `env["CLUSTER"]` (default `slurm`)
from the `ADAPTERS` dict. `NerscAdapter` extends `SlurmAdapter` with `--qos` and `--constraint`
instead of `--partition` and the whole-node `--pack` mode; it cannot render a SIREN training run
yet.

## Verifying and resuming (cluster-agnostic)

Completion is checked the same way on every cluster:

- A dataset batch is done when all four `wc_{sensor,hits,step,labl}_NNNN.h5` open, carry
  `config/complete == True`, agree on `config/n_events`, and hold no more events than requested
  (`dataprod/scripts/verify_jobs.py`). Fewer is normal: the trigger or the selection drops
  events.
- A SIREN-input cell is done when its ROOT file has the `OpticalPhotons` tree.

To resubmit what is missing, run the lister in `--list` mode through the container wrapper:

```bash
lucid/production/platform/run_in_container.sh \
    lucid.production.dataprod.scripts.verify_jobs <OUT>/<detector> --list -s   # drop -s to preview
```

Rerun it after each wave until it lists nothing.

After a run:

1. `verify_jobs` prints `failed/missing: 0`.
2. A shortfall against `nominal_*` is expected for triggered (`GeV`, `Test`) and
   `min_physics_hits` (`Solar`, `SN`) configs.
3. `config/source_event_idx` has gaps where events were dropped. Compare with the ROOT entry
   count for the selection efficiency; the gaps are not missing output.
4. Before a full wave, run a test wave (`-t`: one 2-event job per config, under
   `<OUT>/<detector>/_test/`) and check that all four files appear in each directory.

## Dev loop: run your own checkouts in the batch jobs

To run a LUCiD or PhotonSim change in batch jobs without rebuilding the image, set either or
both in `user_paths.sh`:

```bash
export LUCID_DEV_PATH=/path/to/LUCiD
export PHOTONSIM_DEV_PATH=/path/to/PhotonSim
```

Every job then binds them over `/opt/LUCiD` and `/opt/PhotonSim` (on HTCondor through the
`SingularityBind` classad). Unset them to use the image's copies. Python changes take effect
directly. A PhotonSim source change, and every image refresh, needs the host checkout rebuilt
against the image's GEANT4 and ROOT, or LUCiD fails with "PhotonSim ROOT file is missing
branches":

```bash
apptainer exec -B "$APPTAINER_BINDS" -B "$PHOTONSIM_DEV_PATH:/opt/PhotonSim" \
    "$LUCID_IMAGE_PATH" bash -lc 'cmake --build /opt/PhotonSim/build -j 4'
```

The first build of a fresh PhotonSim clone needs the image's configure flags: conda's expat and
`libstdc++` (the system ones are too old for the conda GEANT4):

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

A bound LUCiD checkout is writable, so new overlap tables are cached in it.

## Adding a cluster

1. Write an adapter class in `platform/adapters.py` (`name`, `submit_extension`, `submit_cmd`
   and the render methods) and add it to `ADAPTERS`.
2. Add `platform/templates/user_paths.<cluster>.sh.template` with `CLUSTER=<name>` and the
   site's variables.
3. Add `dataprod/scripts/monitor_jobs.<cluster>.sh` if the queue tool is not `squeue` or
   `condor_q`.
4. Run a test wave and verify it as above.

## Limits

The job-time models (`seconds_per_event` in the dataset configs, the SIREN-input time model)
were measured on S3DF and may need refitting on another cluster; each config can override them.
