#!/bin/bash
# Merge per-job ROOTs into photonsim.root for every cell under a scan-output
# tree, then (optionally) delete the per-job inputs.
#
# Run this AFTER all generate_jobs.py-submitted jobs have finished. Cells that
# ran with n_jobs=1 are also handled (hadd of a single input).
#
# Performance design:
#   * ONE apptainer exec wraps the whole run — avoids per-cell container
#     cold-start (~1-3 s each).
#   * Inside the container, an xargs -P N pool merges cells in parallel.
#   * Each cell hadds into the container's local /tmp (tmpfs), then mv's the
#     finished file to EOS in a single streaming write — avoiding hadd's many
#     small writes hitting EOS metadata each.
#
# Usage:
#   ./merge.sh <scan_output_dir> [--keep] [--force] [-j N]
#
# Options:
#   --keep    Keep the per-job output_job_*.root files after merging (default:
#             delete them so the cell dir is left with just photonsim.root).
#   --force   Re-merge cells that already have photonsim.root.
#   -j N      Parallel hadd workers inside the container (default 4).

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
[ -f "${USER_PATHS}" ] || { echo "error: ${USER_PATHS} not found." >&2; exit 1; }
# shellcheck source=/dev/null
source "${USER_PATHS}"

SCAN_OUTPUT_DIR="${1:?usage: $0 <scan_output_dir> [--keep] [--force] [-j N]}"
shift || true

KEEP=0
FORCE=0
PARALLEL=4
while [ $# -gt 0 ]; do
    case "$1" in
        --keep)  KEEP=1; shift ;;
        --force) FORCE=1; shift ;;
        -j)      PARALLEL="$2"; shift 2 ;;
        *) echo "error: unknown option $1" >&2; exit 2 ;;
    esac
done

[ -d "${SCAN_OUTPUT_DIR}" ] || { echo "error: scan output dir not found: ${SCAN_OUTPUT_DIR}" >&2; exit 1; }
[ -n "${LUCID_IMAGE_PATH:-}" ] || { echo "error: LUCID_IMAGE_PATH not set in user_paths.sh." >&2; exit 1; }

CELLS_LIST=$(mktemp /tmp/merge-cells.XXXXXX)
trap 'rm -f "${CELLS_LIST}"' EXIT

find "${SCAN_OUTPUT_DIR}" -mindepth 3 -maxdepth 3 -type d -name '*MeV' | sort > "${CELLS_LIST}"
N_CELLS=$(wc -l < "${CELLS_LIST}")

echo "scan output:   ${SCAN_OUTPUT_DIR}"
echo "container:     ${LUCID_IMAGE_PATH}"
echo "cells found:   ${N_CELLS}"
echo "parallelism:   ${PARALLEL}"
echo ""

START_TIME=$(date +%s)

apptainer exec \
    -B "${APPTAINER_BINDS:-/sdf,/fs,/sdf/scratch,/lscratch}" \
    --env KEEP="${KEEP}" --env FORCE="${FORCE}" \
    "${LUCID_IMAGE_PATH}" \
    bash -s "${CELLS_LIST}" "${PARALLEL}" <<'INNER_EOF'
set -uo pipefail
LIST="$1"
J="$2"

merge_one() {
    cell="$1"
    merged="${cell}/photonsim.root"
    if [ -f "${merged}" ] && [ "${FORCE:-0}" = "0" ]; then
        echo "  skip (exists): ${cell}"
        return 0
    fi
    shopt -s nullglob
    parts=( "${cell}"/output_job_*.root )
    shopt -u nullglob
    if [ "${#parts[@]}" = 0 ]; then
        if [ ! -f "${merged}" ]; then
            echo "  empty: ${cell}" >&2
        fi
        return 0
    fi
    tmp=$(mktemp /tmp/hadd.XXXXXX.root)
    if hadd -f "${tmp}" "${parts[@]}" >/dev/null 2>&1; then
        mv "${tmp}" "${merged}"
        if [ "${KEEP:-0}" = "0" ]; then
            rm -f "${parts[@]}"
        fi
        echo "  merged ${#parts[@]} → ${merged}"
    else
        rm -f "${tmp}"
        echo "  FAILED hadd: ${cell}" >&2
        return 1
    fi
}
export -f merge_one

xargs -a "${LIST}" -P "${J}" -I CELL bash -c 'merge_one "$@"' _ CELL
INNER_EOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Merge complete ==="
echo "Elapsed:  ${ELAPSED} s"
