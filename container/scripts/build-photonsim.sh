#!/bin/bash
# Build PhotonSim against conda GEANT4 + source-built ROOT 6.28.
# Assumes PhotonSim source is at /opt/PhotonSim (copied via %files).
# Idempotent: skips if the binary exists.

set -eux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_env.sh"

PHOTONSIM_SRC=${PHOTONSIM_SRC:-/opt/PhotonSim}
BIN=${PHOTONSIM_SRC}/build/PhotonSim

# Always rebuild: even if ${BIN} exists, it was likely linked against host
# libs (common case when the src tree is bind-mounted into a sandbox).

test -d "${PHOTONSIM_SRC}"        || { echo "ERROR: ${PHOTONSIM_SRC} not present." >&2; exit 1; }
test -x /opt/root/bin/root-config || { echo "ERROR: ROOT not installed." >&2; exit 1; }
command -v geant4-config >/dev/null || { echo "ERROR: geant4-config not on PATH." >&2; exit 1; }

# Discard any stale host-built build/ dir copied in via %files.
rm -rf "${PHOTONSIM_SRC}/build"
mkdir -p "${PHOTONSIM_SRC}/build"
cd "${PHOTONSIM_SRC}/build"

cmake -DCMAKE_BUILD_TYPE=Release \
      -DGeant4_DIR=$(geant4-config --prefix)/lib/cmake/Geant4 \
      -DROOT_DIR=/opt/root/cmake \
      ..
make -j"$(nproc)"

test -x "${BIN}"
echo "PhotonSim built: ${BIN}"
