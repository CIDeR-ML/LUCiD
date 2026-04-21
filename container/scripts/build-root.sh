#!/bin/bash
# Build ROOT 6.28.12 from source with -Dpythia6=ON. Installs into
# /opt/root. Idempotent: if /opt/root/bin/root-config exists, exits.
#
# Depends on Pythia6 (build-pythia6.sh) being done first so the cmake
# can point at /opt/conda/lib/libPythia6.so.

set -eux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_env.sh"

if [ -x /opt/root/bin/root-config ] && [ -f /opt/root/lib/libEGPythia6.so ]; then
    echo "ROOT already built with Pythia6 support; skipping."
    exit 0
fi

if [ ! -f /opt/conda/lib/libPythia6.so ]; then
    echo "ERROR: /opt/conda/lib/libPythia6.so missing — run build-pythia6.sh first." >&2
    exit 1
fi

mkdir -p "${BUILD_CACHE_DIR}"
SRC_CACHE="${BUILD_CACHE_DIR}/root-src-v6-28-12"
if [ ! -d "${SRC_CACHE}/.git" ]; then
    rm -rf "${SRC_CACHE}"
    git clone --depth 1 --branch v6-28-12 \
        https://github.com/root-project/root.git "${SRC_CACHE}"
fi

# Persist build dir outside /tmp so incremental rebuilds (when rerunning
# this script after a fix) can reuse cmake's .o cache. /opt/root-build is
# retained across apptainer exec invocations on the sandbox.
BUILD=/opt/root-build
mkdir -p "${BUILD}"
cd "${BUILD}"

# Minimal feature set + enable what GENIE 3.06 links against:
# GenVector, MathMore, Minuit, Minuit2, Geom, EG, Thread, MultiProc,
# ROOTVecOps, ROOTDataFrame, TreePlayer. Pythia6 support gives libEGPythia6.so.
# Env var form of CMAKE_POLICY_VERSION_MINIMUM is inherited by the
# ExternalProject_Add sub-cmakes (cmake 3.31+) so ROOT's bundled PCRE,
# freetype etc. build cleanly on cmake 4.x.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

cmake -S "${SRC_CACHE}" -B . \
    -DCMAKE_INSTALL_PREFIX=/opt/root \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpythia6=ON \
    -DPYTHIA6_LIBRARY=/opt/conda/lib/libPythia6.so \
    -Dpythia8=OFF \
    -Dcxx17=ON \
    -Dminimal=ON \
    -Dbuiltin_pcre=OFF \
    -Dmathmore=ON \
    -Dgenvector=ON \
    -Dminuit2=ON \
    -Ddataframe=ON \
    -Dimt=ON \
    -Dxrootd=OFF \
    -Dssl=OFF \
    -Dhttp=OFF \
    -Dfftw3=OFF \
    -Droofit=OFF \
    -Dtmva=OFF \
    -Dmlp=OFF \
    -Dpyroot=OFF \
    -Dgviz=OFF \
    -Droot7=OFF \
    -Dgnuinstall=OFF

cmake --build . --target install -- -j"$(nproc)"
cd /
rm -rf "${BUILD}"

# Sanity check the pythia6 wrapper is really there.
test -f /opt/root/lib/libEGPythia6.so || {
    echo "ERROR: ROOT built but libEGPythia6.so missing — pythia6 support didn't land." >&2
    exit 1
}
echo "ROOT installed: $(/opt/root/bin/root-config --version), libEGPythia6.so present."
