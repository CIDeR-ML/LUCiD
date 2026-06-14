"""Cluster-abstract pieces of the production pipeline.

The submodules here hold logic that is the same on every cluster (SLURM,
HTCondor, ...): config loading, planning math, completion checks. The
cluster-specific bits — submit-description template, submit command,
apptainer bind list — live in `cluster.py` as `ClusterAdapter` subclasses
that are selected at runtime from the `CLUSTER` env var sourced via
`user_paths.sh`.

See `LUCiD/docs/CLUSTER_ABSTRACTION.md` for the architecture overview.
"""
