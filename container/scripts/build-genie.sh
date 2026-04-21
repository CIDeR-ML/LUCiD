#!/bin/bash
# Build GENIE 3.06.02 from source, in-source install (no --prefix).
# Links against ROOT 6.28 (which provides libEGPythia6.so) and
# libPythia6.so. Idempotent: skips if gevgen and gntpc are present.

set -eux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_env.sh"

if [ -x ${GENIE}/bin/gevgen ] && [ -x ${GENIE}/bin/gntpc ]; then
    echo "GENIE already built; skipping."
    exit 0
fi

test -x /opt/root/bin/root-config       || { echo "ERROR: ROOT missing — run build-root.sh first." >&2; exit 1; }
test -f /opt/root/lib/libEGPythia6.so    || { echo "ERROR: libEGPythia6.so missing from ROOT install." >&2; exit 1; }
test -f /opt/conda/lib/libPythia6.so     || { echo "ERROR: libPythia6.so missing — run build-pythia6.sh first." >&2; exit 1; }

rm -rf "${GENIE}"
git clone --depth 1 --branch R-3_06_02 \
    https://github.com/GENIE-MC/Generator.git "${GENIE}"
cd "${GENIE}"

./configure \
    --disable-profiler \
    --disable-validation-tools \
    --disable-doxygen-doc \
    --disable-test \
    --disable-masterclass \
    --enable-lhapdf6 \
    --enable-fnal \
    --with-pythia6-lib=/opt/pythia6 \
    --with-libxml2-inc=/opt/conda/include/libxml2 \
    --with-libxml2-lib=/opt/conda/lib \
    --with-log4cpp-inc=/opt/conda/include \
    --with-log4cpp-lib=/opt/conda/lib \
    --with-lhapdf6-inc=/opt/conda/include \
    --with-lhapdf6-lib=/opt/conda/lib

make -j"$(nproc)"

test -x ${GENIE}/bin/gevgen && test -x ${GENIE}/bin/gntpc
echo "GENIE built: gevgen + gntpc present in ${GENIE}/bin."
