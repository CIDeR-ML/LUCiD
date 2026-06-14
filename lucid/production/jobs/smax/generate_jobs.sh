#!/bin/bash
# generate_jobs.sh — cluster-agnostic shim around smax/generate_jobs.py.
#
# Runs the Python fan-out inside the LUCiD container so the host's Python
# version is irrelevant (the refactored fan-out uses `from __future__ import
# annotations`, which the system Python on some clusters — e.g. S3DF login
# hosts running Python 3.6 — refuses to parse).
#
# Submission is split from generation: the container has neither
# `sbatch` (SLURM) nor `condor_submit` (HTCondor) on PATH, so when the
# user passes `-s`/`--submit` we strip it before invoking python, then
# submit each `[PREPARED] <path>` line from the host using
# `$CLUSTER_SUBMIT_CMD` (set in user_paths.sh).
#
# Usage: ./generate_jobs.sh -c <smax_config_json> [-s] [-t] [-g]
#                                                 [-o output_base] [-P partition]
#                                                 [--no-skip-existing]
# See generate_jobs.py for full flag descriptions.

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
# Walk up four levels (smax → jobs → production → lucid → LUCiD).
LUCID_ROOT="$( cd "${SCRIPT_DIR}/../../../.." && pwd )"

case " $* " in
    *" -h "*|*" --help "*) NEEDS_USER_PATHS=0 ;;
    *) NEEDS_USER_PATHS=1 ;;
esac

if [ "$NEEDS_USER_PATHS" -eq 1 ] && [ ! -f "${USER_PATHS}" ]; then
    echo "Error: user_paths.sh not found at ${USER_PATHS}" >&2
    echo "Copy user_paths.sh.template and configure it." >&2
    exit 1
fi

if [ "$NEEDS_USER_PATHS" -eq 0 ]; then
    if [ -f "${USER_PATHS}" ]; then
        # shellcheck disable=SC1090
        source "${USER_PATHS}"
        "${APPTAINER_BIN:-apptainer}" exec \
            ${APPTAINER_BINDS:+-B "${APPTAINER_BINDS}"} \
            -B "${LUCID_ROOT}:/opt/LUCiD" \
            "${LUCID_IMAGE_PATH}" \
            /opt/conda/bin/python3 /opt/LUCiD/lucid/production/jobs/smax/generate_jobs.py -h
        exit 0
    fi
    echo "Help unavailable until user_paths.sh is configured." >&2
    exit 0
fi

# shellcheck disable=SC1090
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

APPTAINER_OPTS=()
if [ -n "${APPTAINER_BINDS:-}" ]; then
    APPTAINER_OPTS+=(-B "${APPTAINER_BINDS}")
fi
APPTAINER_OPTS+=(-B "${LUCID_ROOT}:/opt/LUCiD")
if [ -n "${PHOTONSIM_DEV_PATH:-}" ]; then
    APPTAINER_OPTS+=(-B "${PHOTONSIM_DEV_PATH}:/opt/PhotonSim")
fi

TMPOUT="$(mktemp -t lucid-smax.XXXXXX)"
trap 'rm -f "${TMPOUT}"' EXIT

"${APPTAINER_BIN:-apptainer}" exec "${APPTAINER_OPTS[@]}" "${LUCID_IMAGE_PATH}" \
    /opt/conda/bin/python3 /opt/LUCiD/lucid/production/jobs/smax/generate_jobs.py \
        --user-paths "${USER_PATHS}" "${PY_ARGS[@]}" \
    | tee "${TMPOUT}"

if [ "${DO_SUBMIT}" -eq 1 ]; then
    : "${CLUSTER_SUBMIT_CMD:?CLUSTER_SUBMIT_CMD not set in user_paths.sh}"
    if ! command -v "${CLUSTER_SUBMIT_CMD}" >/dev/null 2>&1; then
        echo "Error: ${CLUSTER_SUBMIT_CMD} not on host PATH; cannot submit." >&2
        exit 1
    fi
    SUBMITTED=0
    FAILED=0
    while IFS= read -r sb; do
        if [ -n "${sb}" ] && [ -f "${sb}" ]; then
            if "${CLUSTER_SUBMIT_CMD}" "${sb}" >/dev/null; then
                SUBMITTED=$((SUBMITTED + 1))
            else
                FAILED=$((FAILED + 1))
                echo "  FAILED ${CLUSTER_SUBMIT_CMD} ${sb}" >&2
            fi
        fi
    done < <(grep '^\[PREPARED\] ' "${TMPOUT}" | sed 's/^\[PREPARED\] //')
    echo ""
    echo "Host submission via ${CLUSTER_SUBMIT_CMD}: ${SUBMITTED} submitted, ${FAILED} failed"
fi
