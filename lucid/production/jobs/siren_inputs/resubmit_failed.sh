#!/bin/bash
# Resubmit SIREN-input sub-jobs that did not finish.
#
# Bridges the apptainer/host split:
#   - resubmit_failed.py needs `uproot` (only inside the unified container)
#     to identify cells whose output ROOT lacks the OpticalPhotons key
#     (the truth marker — see resubmit_failed.py docstring).
#   - `sbatch` lives on the host, not in the container image.
#
# So this wrapper runs the python script in --list mode inside apptainer,
# then xargs-sbatches the resulting paths on the host.
#
# Usage:
#   ./resubmit_failed.sh <scan_output_dir> [--dry-run]
#
# Idempotent — run after each drain wave until "missing" is 0, then merge.sh.

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

LIST=$(mktemp)
trap 'rm -f "${LIST}"' EXIT

apptainer exec -B "${APPTAINER_BINDS:-/sdf,/fs,/sdf/scratch,/lscratch}" \
    "${LUCID_IMAGE_PATH}" \
    python3 "${SCRIPT_DIR}/resubmit_failed.py" --list "${SCAN_DIR}" > "${LIST}"

N=$(wc -l < "${LIST}")
echo "missing/partial sub-jobs: ${N}"

if [ "${N}" -eq 0 ]; then
    echo "Nothing to resubmit. Run merge.sh next."
    exit 0
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: would resubmit ${N} sbatches. First 10:"
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
