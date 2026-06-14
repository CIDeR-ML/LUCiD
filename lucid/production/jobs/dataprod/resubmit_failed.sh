#!/bin/bash
# Resubmit dataprod jobs that did not finish.
#
# Bridges the apptainer/host split:
#   - verify_jobs.py needs h5py (only inside the unified container) to
#     read each batch's `config/n_events` attr — the truth marker that
#     `lucid-run-job` reached `Finalize()`.
#   - `sbatch` lives on the host, not in the container image.
#
# So this wrapper runs verify_jobs.py in --list mode inside apptainer,
# then xargs-sbatches the resulting paths on the host.
#
# Usage:
#   ./resubmit_failed.sh <scan_output_dir> [--dry-run]
#
# Idempotent — run after each drain wave until "failed" is 0.

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
[ -f "${USER_PATHS}" ] || { echo "error: ${USER_PATHS} not found." >&2; exit 1; }
# shellcheck source=/dev/null
source "${USER_PATHS}"

SCAN_DIR="${1:?usage: $0 <scan_output_dir> [--dry-run]}"
DRY_RUN=0
if [ "${2:-}" = "--dry-run" ]; then DRY_RUN=1; fi

[ -d "${SCAN_DIR}" ] || { echo "error: scan dir not found: ${SCAN_DIR}" >&2; exit 1; }
[ -n "${LUCID_IMAGE_PATH:-}" ] || { echo "error: LUCID_IMAGE_PATH not set." >&2; exit 1; }

# Honor optional LUCID_DEV_PATH so an in-flight checkout of lucid/production
# is the one that gets imported (matches the dev-loop binds in generate_jobs.sh).
DEV_BINDS=""
[ -n "${LUCID_DEV_PATH:-}" ]     && DEV_BINDS="${DEV_BINDS} -B ${LUCID_DEV_PATH}:/opt/LUCiD"
[ -n "${PHOTONSIM_DEV_PATH:-}" ] && DEV_BINDS="${DEV_BINDS} -B ${PHOTONSIM_DEV_PATH}:/opt/PhotonSim"

LIST=$(mktemp)
trap 'rm -f "${LIST}"' EXIT

apptainer exec -B "${APPTAINER_BINDS:-/sdf,/fs,/sdf/scratch,/lscratch}" ${DEV_BINDS} \
    "${LUCID_IMAGE_PATH}" \
    python3 "${SCRIPT_DIR}/verify_jobs.py" --list "${SCAN_DIR}" > "${LIST}"

N=$(wc -l < "${LIST}")
echo "failed/missing jobs: ${N}"

if [ "${N}" -eq 0 ]; then
    echo "Nothing to resubmit."
    exit 0
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: would resubmit ${N} submits. First 10:"
    head -10 "${LIST}"
    exit 0
fi

SUBMIT_CMD="${CLUSTER_SUBMIT_CMD:-sbatch}"
command -v "${SUBMIT_CMD}" >/dev/null || { echo "error: ${SUBMIT_CMD} not on host PATH." >&2; exit 1; }

ok=0; fail=0
while IFS= read -r sb; do
    if "${SUBMIT_CMD}" "${sb}" >/dev/null 2>&1; then
        ok=$((ok + 1))
    else
        echo "  FAILED: ${sb}" >&2
        fail=$((fail + 1))
    fi
done < "${LIST}"
echo "resubmitted: ${ok}  failed: ${fail}"
[ "${fail}" -eq 0 ]
