#!/bin/bash
# Install LUCiD (editable) into the conda base env. Assumes LUCiD source
# is at /opt/LUCiD. Idempotent: skips if `lucid-run-job` is on PATH and
# the package imports.

set -eux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_env.sh"

LUCID_SRC=${LUCID_SRC:-/opt/LUCiD}

if command -v lucid-run-job >/dev/null && python3 -c "import lucid" >/dev/null 2>&1; then
    echo "LUCiD already installed; skipping."
    exit 0
fi

test -d "${LUCID_SRC}" || { echo "ERROR: ${LUCID_SRC} not present." >&2; exit 1; }
cd "${LUCID_SRC}"
pip install --no-deps -e .

command -v lucid-run-job >/dev/null
python3 -c "import lucid.production.run_job as m; print(m.__file__)"
echo "LUCiD installed editable."
