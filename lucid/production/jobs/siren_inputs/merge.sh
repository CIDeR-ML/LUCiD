#!/bin/bash
# Merge per-job ROOTs into photonsim.root for every cell under a scan-output
# tree, then (optionally) delete the per-job inputs.
#
# Run this AFTER all generate_jobs.py-submitted jobs have finished. Cells that
# ran with n_jobs=1 are also handled (hadd of a single input).
#
# Usage:
#   ./merge.sh <scan_output_dir> [--keep] [--force]
#
# Options:
#   --keep    Keep the per-job output_job_*.root files after merging (default:
#             delete them so the cell dir is left with just photonsim.root).
#   --force   Re-merge cells that already have photonsim.root.

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
[ -f "${USER_PATHS}" ] || { echo "error: ${USER_PATHS} not found." >&2; exit 1; }
# shellcheck source=/dev/null
source "${USER_PATHS}"

SCAN_OUTPUT_DIR="${1:?usage: $0 <scan_output_dir> [--keep] [--force]}"
shift || true

KEEP=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --keep)  KEEP=1 ;;
        --force) FORCE=1 ;;
        *) echo "error: unknown option $arg" >&2; exit 2 ;;
    esac
done

[ -d "${SCAN_OUTPUT_DIR}" ] || { echo "error: scan output dir not found: ${SCAN_OUTPUT_DIR}" >&2; exit 1; }
[ -n "${LUCID_IMAGE_PATH:-}" ] || { echo "error: LUCID_IMAGE_PATH not set in user_paths.sh." >&2; exit 1; }

echo "scan output:   ${SCAN_OUTPUT_DIR}"
echo "container:     ${LUCID_IMAGE_PATH}"
echo ""

merged=0; skipped=0; empty=0
while IFS= read -r -d '' cell_dir; do
    merged_root="${cell_dir}/photonsim.root"
    # Glob the per-job ROOTs.
    shopt -s nullglob
    parts=( "${cell_dir}"/output_job_*.root )
    shopt -u nullglob

    if [ "${#parts[@]}" -eq 0 ]; then
        if [ ! -f "${merged_root}" ]; then
            echo "  empty: ${cell_dir} (no per-job ROOTs and no photonsim.root)"
            empty=$((empty + 1))
        fi
        continue
    fi

    if [ -f "${merged_root}" ] && [ "${FORCE}" -eq 0 ]; then
        echo "  skip (exists): ${merged_root}"
        skipped=$((skipped + 1))
        continue
    fi

    echo "  merging ${#parts[@]} file(s) → ${merged_root}"
    apptainer exec -B "${APPTAINER_BINDS:-/sdf,/fs,/sdf/scratch,/lscratch}" \
        "${LUCID_IMAGE_PATH}" \
        hadd -f "${merged_root}" "${parts[@]}" > /dev/null

    if [ "${KEEP}" -eq 0 ]; then
        rm -f "${parts[@]}"
    fi
    merged=$((merged + 1))
done < <(find "${SCAN_OUTPUT_DIR}" -mindepth 3 -maxdepth 3 -type d -name '*MeV' -print0)

echo ""
echo "=== Merge complete ==="
echo "Merged:   ${merged}"
echo "Skipped:  ${skipped}"
echo "Empty:    ${empty}"
