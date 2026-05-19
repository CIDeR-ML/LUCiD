# Cluster abstraction

How the production pipeline targets multiple batch clusters (currently
SLURM on S3DF and HTCondor on LXPLUS) from a single code base. The
short version: shared planning + a single per-cluster
`ClusterAdapter` class, dispatched at runtime from a sticky env var
in your active `user_paths.sh`.

For per-cluster runbooks, see:

- [`QUICKSTART_S3DF.md`](QUICKSTART_S3DF.md) — SLURM at SLAC.
- [`QUICKSTART_LXPLUS.md`](QUICKSTART_LXPLUS.md) — HTCondor at CERN.

## Architecture

```
LUCiD/lucid/production/
├── cluster_common/                  cluster-agnostic logic (importable module)
│   ├── user_paths.py                source user_paths.sh → env dict
│   ├── verify.py                    is_complete_siren / is_complete_dataprod
│   ├── siren_planning.py            Stage-1 fan-out math
│   ├── smax_planning.py             Stage-0 fan-out math
│   ├── train_planning.py            Stage-3 FLAG_MAP + diff resolution
│   ├── dataprod_fanout.py           Dataprod Python fan-out (used by the
│   │                                bash shim in jobs/dataprod/)
│   ├── cluster.py                   ClusterAdapter ABC + SlurmAdapter
│   └── htcondor.py                  HTCondorAdapter + register
│
└── jobs/                            ONE production tree, no per-cluster dirs
    ├── user_paths.s3df.sh.template   cluster env block — SLURM/SLAC values
    ├── user_paths.lxplus.sh.template cluster env block — HTCondor/CERN values
    ├── user_paths.sh                 gitignored; user copies from one template
    ├── README.md                     orientation; points at the QUICKSTART docs
    ├── dataprod/                     dataprod fan-out
    │   ├── generate_jobs.sh          cluster-neutral shim → dataprod_fanout
    │   ├── submit_all_configs.sh
    │   ├── resubmit_failed.sh + verify_jobs.py
    │   ├── monitor_jobs.s3df.sh      squeue-flavoured
    │   ├── monitor_jobs.lxplus.sh    condor_q-flavoured
    │   └── timing/cleanup helpers
    ├── smax/                         Stage 0
    ├── siren_inputs/                 Stage 1
    ├── train_siren/                  Stage 3
    └── utils/                        output-tree maintenance helpers
```

### Key design choices

1. **One canonical scripts tree.** All entrypoints live exactly once
   under `jobs/`. Neither cluster is privileged; the file tree has no
   asymmetry between SLURM and HTCondor.

2. **Cluster identity carried entirely by `user_paths.sh`.** Three env
   keys — `CLUSTER`, `CLUSTER_SUBMIT_CMD`, `APPTAINER_BINDS` — turn
   the same script into a SLURM or HTCondor invocation. Switching
   cluster = swap which `user_paths.<cluster>.sh.template` you copied
   into `user_paths.sh`.

3. **`Path(__file__).parent` (no `.resolve()`).** Python entrypoints
   look up `user_paths.sh` relative to their own directory, which is
   `jobs/<stage>/`. That works regardless of which absolute path the
   user invokes them from.

4. **Bash entrypoints read env vars verbatim.** `merge.sh`,
   `resubmit_failed.sh`, `analyze.sh` source `user_paths.sh` then
   expand `${APPTAINER_BINDS:-/sdf,...}` and
   `${CLUSTER_SUBMIT_CMD:-sbatch}`. Defaults preserve correct
   behaviour for any pre-refactor `user_paths.sh`.

5. **The dataprod bash entrypoint is a Python shim.**
   `jobs/dataprod/generate_jobs.sh` is ~25 lines that exec
   `python3 -m lucid.production.cluster_common.dataprod_fanout`. The
   port preserves the CLI surface and output layout bit-for-bit; the
   bash version's `jq`-driven JSON parsing moved to Python.

## `ClusterAdapter` contract

`cluster_common/cluster.py`:

```python
class ClusterAdapter(ABC):
    name: str                   # "slurm" / "htcondor"
    submit_extension: str       # "sbatch" / "sub"
    submit_cmd: str             # "sbatch" / "condor_submit"
    apptainer_binds: str        # comma-separated

    def render_siren_cell(...)  -> str: ...
    def render_smax_cell(...)   -> str: ...
    def render_dataprod_job(...) -> str: ...
    def render_train_run(...)   -> str: ...
    def queue_status_cmd(user)  -> List[str]: ...
```

Selection:

```python
from lucid.production.cluster_common import htcondor   # registers HTCondor
from lucid.production.cluster_common.cluster import get_adapter

env = load_user_paths(...)
adapter = get_adapter(env)         # reads env["CLUSTER"], default "slurm"
sb_body = adapter.render_siren_cell(...)
```

`htcondor.py` imports `cluster.py` and registers its adapter at module
import time. Every entrypoint imports `htcondor` unconditionally (a
no-op cost when `CLUSTER=slurm`, since `SlurmAdapter` lives in
`cluster.py` and stays the default).

## Adding a third cluster

1. **New adapter** at `cluster_common/<cluster>.py`:
   ```python
   from .cluster import ClusterAdapter, register_adapter

   class MyClusterAdapter(ClusterAdapter):
       name = "mycluster"
       submit_extension = "myjob"
       submit_cmd = "mysub"
       ...

   register_adapter("mycluster", MyClusterAdapter)
   ```
2. **Register the import** in every entrypoint that already imports
   `htcondor` — `siren_inputs/generate_jobs.py`, `smax/generate_jobs.py`,
   `train_siren/generate_jobs.py`, `cluster_common/dataprod_fanout.py`.
   One line per file.
3. **New `user_paths.<cluster>.sh.template`** at `jobs/` setting
   `CLUSTER=mycluster` plus the cluster-specific env keys.
4. **New `dataprod/monitor_jobs.<cluster>.sh`** if the queue tool isn't
   `squeue` or `condor_q`.
5. Run the per-cluster verification from `QUICKSTART_LXPLUS.md` Step
   5/6 against the new cluster.

No other files change.

## What did NOT get abstracted

- `jobs/utils/{copy,clean}_output_data.sh` — hardcode `/sdf/...` paths
  and aren't part of the production loop. Duplicate them for a new
  cluster if anyone needs them.
- The top-level `LUCiD/s3df_jobs/` (track-optimization scans) —
  different workflow, separate concern.
- Per-cluster timing-model defaults
  (`siren_planning.DEFAULT_TIME_MODEL`, dataprod `seconds_per_event`).
  These were fit on S3DF runs and may need re-fitting after the first
  large LXPLUS scan. The config schema already supports per-config
  overrides.
