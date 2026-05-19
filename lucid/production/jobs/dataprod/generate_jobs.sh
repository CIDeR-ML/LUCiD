#!/bin/bash
# generate_jobs.sh — thin shim around the cluster-agnostic Python fan-out.
#
# Historically this file was ~365 lines of bash. The logic was ported to
# `lucid.production.cluster_common.dataprod_fanout` so HTCondor (LXPLUS)
# can share it; this shim preserves the existing CLI surface and the
# `user_paths.sh` discovery rule (next to this file's grand-parent).
#
# Usage: ./generate_jobs.sh -c <config_json> [-s] [-t] [-g] [-P partition]
#                                            [-o output_base] [-D detector]
#                                            [-j K] [-N M]
# Pass -h for full flag descriptions.

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
# Walk up four levels (dataprod → jobs → production → lucid → LUCiD) to
# find the LUCiD root; needed so the Python module can be imported when
# `lucid-sim` isn't pip-installed on the host.
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

# Honour the existing convention: --user-paths defaults to the sibling
# user_paths.sh, regardless of where the user invokes us from.
PYTHONPATH="${LUCID_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" exec python3 \
    -m lucid.production.cluster_common.dataprod_fanout \
    --user-paths "${USER_PATHS}" "$@"
