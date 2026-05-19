#!/bin/bash
# Run PhotonSim/tools/smax/analyze_smax.py against an s_max-scan output
# tree to fit s_max(E) per (material, particle) and write the
# parametrisation CSVs into PhotonSim/data/<material>/<particle>/.
#
# Run this AFTER all `smax_*` SLURM jobs have finished.
#
# Usage:
#   ./analyze.sh <scan_output_dir>
#
# <scan_output_dir> is the path passed to generate_jobs.py as
# OUTPUT_BASE_PATH (or -o). Equivalent to scan_smax.py's --output-dir.
#
# CSVs are written to PHOTONSIM_DEV_PATH/data/<material>/<particle>/
# (uses the host PhotonSim checkout so the new fit lands where the
# Stage-1 SIREN-input scripts look it up). To target a different
# checkout, pass the path as the second arg.

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
[ -f "${USER_PATHS}" ] || { echo "error: ${USER_PATHS} not found." >&2; exit 1; }
# shellcheck source=/dev/null
source "${USER_PATHS}"

SCAN_OUTPUT_DIR="${1:?usage: $0 <scan_output_dir> [photonsim_checkout]}"
PHOTONSIM_CHECKOUT="${2:-${PHOTONSIM_DEV_PATH:-}}"

[ -d "${SCAN_OUTPUT_DIR}" ] || { echo "error: scan output dir not found: ${SCAN_OUTPUT_DIR}" >&2; exit 1; }
[ -n "${PHOTONSIM_CHECKOUT}" ] || {
    echo "error: PhotonSim checkout not set. Pass as second arg or set PHOTONSIM_DEV_PATH in user_paths.sh." >&2
    exit 1; }
[ -d "${PHOTONSIM_CHECKOUT}/tools/smax" ] || {
    echo "error: ${PHOTONSIM_CHECKOUT}/tools/smax/ not found." >&2
    exit 1; }

echo "scan output:   ${SCAN_OUTPUT_DIR}"
echo "photonsim:     ${PHOTONSIM_CHECKOUT}"
echo "container:     ${LUCID_IMAGE_PATH}"
echo ""

apptainer exec --nv -B "${APPTAINER_BINDS:-/sdf,/fs,/sdf/scratch,/lscratch}" \
    -B "${PHOTONSIM_CHECKOUT}:/opt/PhotonSim" \
    "${LUCID_IMAGE_PATH}" \
    python3 /opt/PhotonSim/tools/smax/analyze_smax.py \
        --output-dir "${SCAN_OUTPUT_DIR}" \
        --data-dir   "${PHOTONSIM_CHECKOUT}/data"
