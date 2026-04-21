#!/bin/bash
# 30-second sanity check. Run inside the sandbox/image to confirm the
# environment is consistent before committing to a long build step.
#
# Checks:
#   * Core binaries are on PATH (gcc, g++, gfortran, g77-shim, cmake,
#     ninja, make, python3, mamba, curl, git).
#   * Conda sysroot is activated (CONDA_BUILD_SYSROOT points at a real
#     dir with features.h).
#   * ROOT / libPythia6 / libEGPythia6 paths are present when expected
#     (skipped with a warning if we're mid-build and they don't exist yet).
#
# Exits non-zero if a mandatory check fails.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_env.sh"

fail=0
ok()    { printf '  [ OK ] %s\n' "$*"; }
warn()  { printf '  [warn] %s\n' "$*"; }
bad()   { printf '  [FAIL] %s\n' "$*"; fail=1; }

need_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        ok "$1 -> $(command -v "$1")"
    else
        bad "$1 not on PATH"
    fi
}

echo "=== preflight ==="
for c in gcc g++ gfortran g77 cmake ninja make python3 mamba curl git; do
    need_cmd "$c"
done

# Sysroot for rootcling + c++ builds.
if [ -n "${CONDA_BUILD_SYSROOT:-}" ] && [ -f "${CONDA_BUILD_SYSROOT}/usr/include/features.h" ]; then
    ok "CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT} (features.h present)"
else
    bad "CONDA_BUILD_SYSROOT unset or missing features.h"
fi

# Optional libs (warn if missing — they're populated by later build steps).
if [ -f /opt/conda/lib/libPythia6.so ]; then
    ok "libPythia6.so present"
else
    warn "libPythia6.so not yet built"
fi
if [ -f /opt/root/lib/libEGPythia6.so ]; then
    ok "libEGPythia6.so present"
else
    warn "libEGPythia6.so not yet built (build-root.sh produces it)"
fi
if [ -x /opt/root/bin/root-config ]; then
    ok "root-config -> $(/opt/root/bin/root-config --version)"
else
    warn "ROOT not yet installed"
fi

echo "=== preflight done (fail=${fail}) ==="
exit "${fail}"
