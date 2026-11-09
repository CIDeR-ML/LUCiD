#!/usr/bin/env bash
# tools/gate.sh on a milano node, inside the owner-named container. The suite
# is minutes, so it does not belong on a shared interactive node.
#
#   sbatch tools/gate.sbatch
#
# The job's exit status is gate.sh's own (0 green, 1/2/3 = L1/L2/L3 failed),
# so `sacct -j <id> --format=State,ExitCode` is the whole result.
#SBATCH --job-name=lucid-gate
#SBATCH --partition=milano
#SBATCH --account=mli:cider-ml
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=gate-%j.log
set -euo pipefail

# NOT from ${BASH_SOURCE[0]}: sbatch copies the script into /var/spool/slurmd, so the
# path it is running from is not in the repo. $SLURM_SUBMIT_DIR is where `sbatch` was
# invoked, which is the repo for the documented `sbatch tools/gate.sbatch`; override
# with $LUCID_REPO to gate a different worktree.
REPO="${LUCID_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -x "$REPO/tools/gate.sh" ]; then
    echo "gate.sbatch: no tools/gate.sh under '$REPO'." >&2
    echo "  Submit from the repo root, or set LUCID_REPO=<worktree>." >&2
    exit 1
fi
IMAGE="${LUCID_IMAGE_PATH:-/sdf/group/neutrino/images/develop.sif}"

# The container ships its own Python with the pinned jax; an inherited
# PYTHONPATH from the submitting shell would shadow it.
export PYTHONPATH=""
# WU-03 H14 asked for PYTHONNOUSERSITE=1 here, so ~/.local cannot shadow the
# container. It is REVERTED, and must stay reverted until flax is either in the
# image or an optional import: develop.sif ships no flax, optax or chex (WU-03
# measured exactly that, commit 12bdddbe), lucid/siren/net.py imports
# flax.linen unconditionally, and lucid/simulation/__init__.py reaches it through
# simulator.py. With the flag set, `import lucid.simulation` raises
# ModuleNotFoundError and the gate collapses: 39 L1 collection errors and 2 L2
# import failures on a tree where nothing is wrong (wave-1 gate run 1, job
# 40914872). The user site-packages are load-bearing on S3DF today.

exec singularity exec --bind /sdf "$IMAGE" bash "$REPO/tools/gate.sh"
