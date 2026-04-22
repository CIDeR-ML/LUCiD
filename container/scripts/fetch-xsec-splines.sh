#!/bin/bash
# Download GENIE cross-section splines from the FNAL scisoft public mirror.
# Usable both during image build (Docker/Apptainer %post) and on a bare host.
#
#   fetch-xsec-splines.sh [--dest DIR] [--tune TUNE] [--force]
#
#   --dest DIR   destination directory (default: /opt/genie_xsec if writable,
#                else $GENIE_XSEC_CACHE or $HOME/.cache/lucid/genie_xsec)
#   --tune TUNE  GENIE tune name, e.g. G18_02a_00_000 (default)
#   --force      re-download and overwrite even if splines already present
#
# The default tune matches LUCiD's dataprod_13/14 configs. To change tune,
# pass --tune and also set GENIE_XSEC_FILE in your job environment so
# run_genie.py picks up the new path.
#
# Source:  https://scisoft.fnal.gov/scisoft/packages/genie_xsec/v3_06_00/
# The scisoft archive is the same one cvmfs mirrors under
# /cvmfs/larsoft.opensciencegrid.org — public, no auth.

set -eu

TUNE="G18_02a_00_000"
DEST=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dest)  DEST="$2"; shift 2 ;;
        --tune)  TUNE="$2"; shift 2 ;;
        --force) FORCE=1;   shift ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Default dest: /opt/genie_xsec if we can write to /opt (image build),
# otherwise a per-user cache.
if [ -z "${DEST}" ]; then
    if [ -w /opt ] || ( mkdir -p /opt/genie_xsec 2>/dev/null && [ -w /opt/genie_xsec ] ); then
        DEST=/opt/genie_xsec
    else
        DEST="${GENIE_XSEC_CACHE:-${HOME}/.cache/lucid/genie_xsec}"
    fi
fi

# Map tune name "G18_02a_00_000" → scisoft slug "G1802a00000" (strip underscores).
SLUG="$(printf '%s' "${TUNE}" | tr -d '_')"

BASE_URL="https://scisoft.fnal.gov/scisoft/packages/genie_xsec/v3_06_00"
TARBALL_NAME="genie_xsec-3.06.00-noarch-${SLUG}-k250-e1000.tar.bz2"
URL="${BASE_URL}/${TARBALL_NAME}"

TARGET_XML="${DEST}/gxspl-FNALsmall.xml"

mkdir -p "${DEST}"

if [ -f "${TARGET_XML}" ] && [ "${FORCE}" -eq 0 ]; then
    echo "GENIE xsec splines already present at ${TARGET_XML}; skipping download."
    echo "(Pass --force to re-download.)"
    exit 0
fi

echo "=== GENIE xsec splines ==="
echo "    tune:   ${TUNE} (slug ${SLUG})"
echo "    source: ${URL}"
echo "    dest:   ${DEST}"

# Use a temp dir inside DEST to avoid filling /tmp on tiny Docker layers.
WORK="${DEST}/.fetch-$$"
mkdir -p "${WORK}"
trap 'rm -rf "${WORK}"' EXIT

TARBALL="${WORK}/${TARBALL_NAME}"
echo "    downloading (~455 MB compressed)..."
curl -fL --retry 3 --retry-delay 5 -o "${TARBALL}" "${URL}"

echo "    extracting..."
tar -xjf "${TARBALL}" -C "${WORK}"

# The scisoft tarball lays out as  <TUNE-slug>-k250-e1000/data/gxspl-*.xml[.gz]
# We promote just the uncompressed small spline (what GENIE v3 loads
# directly) into $DEST for a stable, predictable path.
SRC_XML="$(find "${WORK}" -maxdepth 5 -name 'gxspl-FNALsmall.xml' -print -quit)"
if [ -z "${SRC_XML}" ]; then
    echo "ERROR: gxspl-FNALsmall.xml not found in tarball." >&2
    find "${WORK}" -maxdepth 5 -type f >&2
    exit 1
fi

mv -f "${SRC_XML}" "${TARGET_XML}"

# Size sanity check: the small spline is ~500 MB uncompressed; anything
# far off that suggests a truncated download.
SIZE="$(stat -c%s "${TARGET_XML}" 2>/dev/null || stat -f%z "${TARGET_XML}")"
if [ "${SIZE}" -lt 100000000 ]; then
    echo "ERROR: ${TARGET_XML} is suspiciously small (${SIZE} bytes)." >&2
    exit 1
fi

echo "    installed: ${TARGET_XML} (${SIZE} bytes)"
echo "Set GENIE_XSEC_FILE=${TARGET_XML} in your job environment."
