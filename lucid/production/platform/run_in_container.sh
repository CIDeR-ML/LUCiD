#!/bin/bash
# Run a production Python module inside the LUCiD container, then submit what it prepared.
#
# The container has the Python stack (a modern Python, h5py, uproot) but no `sbatch` or
# `condor_submit`; the host has the scheduler client but perhaps only an old Python (S3DF login
# hosts run 3.6). So the module runs in the container, with this checkout bound at /opt/LUCiD,
# and with -s/--submit every `[PREPARED] <path>` line it prints is submitted from the host with
# $CLUSTER_SUBMIT_CMD (user_paths.sh). Without -s nothing is submitted.
#
# Usage: run_in_container.sh <module> [module args...] [-s|--submit]
#
#   generate (and submit) jobs:
#     run_in_container.sh lucid.production.dataprod.fanout -c <config.json> [-s] [-h]
#     run_in_container.sh lucid.production.siren_inputs.generate_jobs -c <config.json> [-s]
#     run_in_container.sh lucid.production.siren_inputs.smax.generate_jobs -c <config.json> [-s]
#     run_in_container.sh lucid.production.siren_inputs.train_siren.generate_jobs -c <config.json> [-s]
#   resubmit the jobs that did not finish (without -s: list them):
#     run_in_container.sh lucid.production.dataprod.scripts.verify_jobs <scan_dir> --list [-s]
#     run_in_container.sh lucid.production.siren_inputs.resubmit_failed <scan_dir> --list [-s]

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/user_paths.sh"
LUCID_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"

MODULE="${1:?usage: $0 <module> [module args...] [-s|--submit]}"
shift

if [ ! -f "${USER_PATHS}" ]; then
    echo "Error: user_paths.sh not found at ${USER_PATHS}" >&2
    echo "Copy one of ${SCRIPT_DIR}/templates/user_paths.*.sh.template there and configure it." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${USER_PATHS}"

: "${LUCID_IMAGE_PATH:?LUCID_IMAGE_PATH not set in user_paths.sh}"
if [ ! -f "${LUCID_IMAGE_PATH}" ]; then
    echo "Error: LUCID_IMAGE_PATH=${LUCID_IMAGE_PATH} does not exist." >&2
    exit 1
fi

PY_ARGS=()
DO_SUBMIT=0
for a in "$@"; do
    case "$a" in
        -s|--submit) DO_SUBMIT=1 ;;
        *)           PY_ARGS+=("$a") ;;
    esac
done

# The host checkout shadows the image's /opt/LUCiD, so the container runs this code, not the
# baked copy; PHOTONSIM_DEV_PATH does the same for PhotonSim.
APPTAINER_OPTS=()
if [ -n "${APPTAINER_BINDS:-}" ]; then
    APPTAINER_OPTS+=(-B "${APPTAINER_BINDS}")
fi
APPTAINER_OPTS+=(-B "${LUCID_ROOT}:/opt/LUCiD")
if [ -n "${PHOTONSIM_DEV_PATH:-}" ]; then
    APPTAINER_OPTS+=(-B "${PHOTONSIM_DEV_PATH}:/opt/PhotonSim")
fi

TMPOUT="$(mktemp -t lucid-run.XXXXXX)"
trap 'rm -f "${TMPOUT}"' EXIT

# -P: the working directory is not put on sys.path, so a `lucid/` under it cannot shadow
# /opt/LUCiD.
"${APPTAINER_BIN:-apptainer}" exec "${APPTAINER_OPTS[@]}" "${LUCID_IMAGE_PATH}" \
    /opt/conda/bin/python3 -P -m "${MODULE}" ${PY_ARGS[@]+"${PY_ARGS[@]}"} \
    | tee "${TMPOUT}"

if [ "${DO_SUBMIT}" -eq 0 ]; then
    exit 0
fi

: "${CLUSTER_SUBMIT_CMD:?CLUSTER_SUBMIT_CMD not set in user_paths.sh}"
if ! command -v "${CLUSTER_SUBMIT_CMD}" >/dev/null 2>&1; then
    echo "Error: ${CLUSTER_SUBMIT_CMD} not on host PATH; cannot submit." >&2
    exit 1
fi
SUBMITTED=0
FAILED=0
while IFS= read -r sb; do
    if "${CLUSTER_SUBMIT_CMD}" "${sb}" >/dev/null; then
        SUBMITTED=$((SUBMITTED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "  FAILED ${CLUSTER_SUBMIT_CMD} ${sb}" >&2
    fi
done < <(sed -n 's/^\[PREPARED\] //p' "${TMPOUT}")
echo ""
echo "Host submission via ${CLUSTER_SUBMIT_CMD}: ${SUBMITTED} submitted, ${FAILED} failed"
[ "${FAILED}" -eq 0 ]
