#!/bin/bash
# generate_jobs.sh — cluster-agnostic shim around the dataprod fan-out.
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
# Usage: ./generate_jobs.sh -c <config_json> [-s] [-t] [-g] [-P partition]
#                                            [-o output_base] [-D detector]
#                                            [-j K] [-N M]
# Pass -h for full flag descriptions.

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
# Walk up four levels (dataprod → jobs → production → lucid → LUCiD) to
# find the LUCiD root — bound into the container so the in-checkout
# `lucid/` package is the one imported (not whatever was baked into the
# image).
LUCID_ROOT="$( cd "${SCRIPT_DIR}/../../../.." && pwd )"

# Skip the user_paths.sh existence check when only asking for help — keeps
# `-h` working in fresh checkouts before the user has copied the template.
case " $* " in
    *" -h "*|*" --help "*) NEEDS_USER_PATHS=0 ;;
    *) NEEDS_USER_PATHS=1 ;;
esac

if [ "$NEEDS_USER_PATHS" -eq 1 ] && [ ! -f "${USER_PATHS}" ]; then
    echo "Error: user_paths.sh not found at ${USER_PATHS}" >&2
    echo "Copy user_paths.sh.template and configure it." >&2
    exit 1
fi

# -h short-circuit: run python in the container (no user_paths needed)
# just to get the help text.
if [ "$NEEDS_USER_PATHS" -eq 0 ]; then
    # The help block needs the python to import cluster_common, which
    # only works inside the container if LUCID_IMAGE_PATH is set. Fall
    # back to a static notice when we can't run python.
    if [ -f "${USER_PATHS}" ]; then
        # shellcheck disable=SC1090
        source "${USER_PATHS}"
        "${APPTAINER_BIN:-apptainer}" exec \
            ${APPTAINER_BINDS:+-B "${APPTAINER_BINDS}"} \
            -B "${LUCID_ROOT}:/opt/LUCiD" \
            "${LUCID_IMAGE_PATH}" \
            /opt/conda/bin/python3 -m lucid.production.cluster_common.dataprod_fanout -h
        exit 0
    fi
    echo "Help unavailable until user_paths.sh is configured." >&2
    exit 0
fi

# shellcheck disable=SC1090
source "${USER_PATHS}"

# Required for the apptainer hop.
: "${LUCID_IMAGE_PATH:?LUCID_IMAGE_PATH not set in user_paths.sh}"
if [ ! -f "${LUCID_IMAGE_PATH}" ]; then
    echo "Error: LUCID_IMAGE_PATH=${LUCID_IMAGE_PATH} does not exist." >&2
    exit 1
fi

# Strip -s/--submit out of the python args; we'll submit host-side after
# generation. Everything else goes through unchanged.
PY_ARGS=()
DO_SUBMIT=0
for a in "$@"; do
    case "$a" in
        -s|--submit) DO_SUBMIT=1 ;;
        *)           PY_ARGS+=("$a") ;;
    esac
done

# Build the apptainer bind list. Always shadow /opt/LUCiD with the host
# checkout so the container sees the current code (the refactored
# cluster_common module won't exist in a stale baked image). Optionally
# shadow /opt/PhotonSim too if the user set PHOTONSIM_DEV_PATH.
APPTAINER_OPTS=()
if [ -n "${APPTAINER_BINDS:-}" ]; then
    APPTAINER_OPTS+=(-B "${APPTAINER_BINDS}")
fi
APPTAINER_OPTS+=(-B "${LUCID_ROOT}:/opt/LUCiD")
if [ -n "${PHOTONSIM_DEV_PATH:-}" ]; then
    APPTAINER_OPTS+=(-B "${PHOTONSIM_DEV_PATH}:/opt/PhotonSim")
fi

# Capture the python's stdout so we can replay it AND parse [PREPARED]
# markers for the post-submit pass.
TMPOUT="$(mktemp -t lucid-dataprod.XXXXXX)"
trap 'rm -f "${TMPOUT}"' EXIT

"${APPTAINER_BIN:-apptainer}" exec "${APPTAINER_OPTS[@]}" "${LUCID_IMAGE_PATH}" \
    /opt/conda/bin/python3 -m lucid.production.cluster_common.dataprod_fanout \
        --user-paths "${USER_PATHS}" "${PY_ARGS[@]}" \
    | tee "${TMPOUT}"

# Host-side submission. Use the cluster's submit command from user_paths.sh
# (sbatch on SLURM, condor_submit on HTCondor).
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
