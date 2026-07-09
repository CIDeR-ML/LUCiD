# `jobs/` — cluster-portable production pipeline

Drives the four production stages (smax fit → SIREN-input scan →
dataprod fan-out → SIREN training) under any supported batch cluster.
The code is cluster-neutral; the cluster identity is carried entirely
by your `user_paths.sh`. Supported today: SLURM (S3DF) and HTCondor
(LXPLUS).

For the architecture see
[`../../../docs/guides/production/cluster-abstraction.md`](../../../docs/guides/production/cluster-abstraction.md).

For per-cluster runbooks see
[`../../../docs/guides/production/deploy-s3df.md`](../../../docs/guides/production/deploy-s3df.md)
and
[`../../../docs/guides/production/deploy-lxplus.md`](../../../docs/guides/production/deploy-lxplus.md).

## Layout

```
jobs/
├── user_paths.s3df.sh.template       cluster env block — SLURM/SLAC values
├── user_paths.lxplus.sh.template     cluster env block — HTCondor/CERN values
├── user_paths.sh                     gitignored; copy from one of the above
├── dataprod/        Data-production fan-out (one dataset per dataprod_*.json)
├── smax/            Stage 0 — fit s_max(E) per (material, particle)
├── siren_inputs/    Stage 1 — per-cell PhotonHist_AngleDistanceNorm ROOTs
├── train_siren/     Stage 3 — SIREN training hyperparam scans (GPU)
└── utils/           Output-tree maintenance helpers
```

`dataprod/` carries two `monitor_jobs.<cluster>.sh` siblings (one for
`squeue`, one for `condor_q`). Pick the matching one when you watch a
running queue.

## One-time setup

Copy the template matching your cluster, edit, run:

```bash
# On S3DF
cp user_paths.s3df.sh.template user_paths.sh && ${EDITOR:-vim} user_paths.sh

# On LXPLUS
cp user_paths.lxplus.sh.template user_paths.sh && ${EDITOR:-vim} user_paths.sh
```

Then any of the entrypoints — `smax/generate_jobs.py`,
`siren_inputs/generate_jobs.py`, `dataprod/generate_jobs.sh`,
`train_siren/generate_jobs.py` — picks up the cluster automatically
from `CLUSTER=` in your `user_paths.sh` and emits the matching submit
description (`.sbatch` for SLURM, `.sub` for HTCondor).

## Adding a third cluster

See the "Adding a third cluster" section of
[`../../../docs/guides/production/cluster-abstraction.md`](../../../docs/guides/production/cluster-abstraction.md).
Short version: write one new adapter under
`lucid/production/cluster_common/<cluster>.py`, drop a
`user_paths.<cluster>.sh.template` alongside the existing two, and
(if your queue tool isn't `squeue` or `condor_q`) drop a matching
`dataprod/monitor_jobs.<cluster>.sh`. No other files change.
