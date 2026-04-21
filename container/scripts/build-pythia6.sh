#!/bin/bash
# Build libPythia6.so from ROOT's prebuilt tarball, install into
# /opt/conda/lib and /opt/pythia6/. Idempotent: if libPythia6.so is
# already installed, exits cleanly.

set -eux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_env.sh"

if [ -f /opt/conda/lib/libPythia6.so ] && [ -f /opt/pythia6/libPythia6.so ]; then
    echo "Pythia6 already built; skipping."
    exit 0
fi

# Pythia6's build scripts call g77; gfortran is a drop-in replacement.
ln -sf /usr/bin/gfortran /usr/local/bin/g77

mkdir -p "${BUILD_CACHE_DIR}"
TARBALL="${BUILD_CACHE_DIR}/pythia6.tar.gz"
if [ ! -f "${TARBALL}" ]; then
    curl -fsSL -o "${TARBALL}" http://root.cern.ch/download/pythia6.tar.gz
fi

WORK=/tmp/pythia6-build.$$
rm -rf "${WORK}"
mkdir -p "${WORK}"
tar -xzf "${TARBALL}" -C "${WORK}"
cd "${WORK}/pythia6"

# Prefer the 64-bit script; fall back to the 32-bit one if absent.
if [ -f makePythia6.linuxx8664 ]; then
    MK_SCRIPT=makePythia6.linuxx8664
elif [ -f makePythia6.linux ]; then
    MK_SCRIPT=makePythia6.linux
else
    echo "No makePythia6.* script found in tarball" >&2
    ls
    exit 1
fi
# gfortran 10+ defaults to -fno-common; Pythia6 relies on old common-block
# linkage.
sed -i 's/-fPIC/-fPIC -fcommon/g' "./${MK_SCRIPT}"
bash "./${MK_SCRIPT}"

mkdir -p /opt/pythia6
cp libPythia6.so /opt/pythia6/
cp libPythia6.so /opt/conda/lib/

cd /
rm -rf "${WORK}"
echo "Pythia6 built: /opt/pythia6/libPythia6.so ($(stat -c%s /opt/pythia6/libPythia6.so) bytes)"
