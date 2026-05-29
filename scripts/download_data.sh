#!/bin/bash
#
# Download LUCiD example data (water muon) from the CERNBox public share.
#
# Default:             real files land in   LUCiD/data/water/muon/
# With --store-dir D:  real files land in   D/water/muon/   and
#                      LUCiD/data/water/muon/ holds symlinks pointing there.
#
# In both cases data/wbls/muon/ is populated with relative symlinks into
# ../../water/muon/ (ROOT + SIREN), so wbls reuses the water files.
#
# The share holds three muon ROOT energies (500/1000/1500 MeV). Only
# 1000 MeV is fetched by default; use --all-energies for all three. The
# SIREN trained models (Cherenkov + dE/dx) are always fetched.

set -euo pipefail

# --- CERNBox share -----------------------------------------------------------
SHARE_TOKEN="vOhH8P78hnSQKNZ"
export SHARE_TOKEN
DAV_BASE="https://cernbox.cern.ch/remote.php/dav/public-files/${SHARE_TOKEN}"

# --- paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUCID_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${LUCID_ROOT}/data"

# --- defaults ----------------------------------------------------------------
STORE_DIR=""
ENERGIES=(1000)

usage() {
    cat <<EOF
Usage: $(basename "$0") [--store-dir DIR] [--all-energies | --energies "E1 E2"]

Downloads LUCiD water-muon example data (ROOT + trained SIREN models) from the
CERNBox share and wires up data/wbls/muon/ as symlinks reusing the water files.

Options:
  --store-dir DIR     Put the real files under DIR/water/muon/ and leave
                      symlinks in LUCiD/data/water/muon/. Default: store the
                      files directly in LUCiD/data/water/muon/.
                      On S3DF use: --store-dir /sdf/data/neutrino/cjesus/CERNBOX
  --all-energies      Fetch all ROOT energies (500, 1000, 1500 MeV).
  --energies "LIST"   Space-separated ROOT energies in MeV (e.g. "500 1000").
  -h, --help          Show this help.

Default ROOT energy: 1000 MeV only (~0.97 GB).
EOF
}

# --- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --store-dir)          STORE_DIR="${2:?--store-dir needs a path}"; shift 2 ;;
        --store-dir=*)        STORE_DIR="${1#*=}"; shift ;;
        --all-energies|--all) ENERGIES=(500 1000 1500); shift ;;
        --energies)           IFS=' ,' read -r -a ENERGIES <<< "${2:?--energies needs a value}"; shift 2 ;;
        --energies=*)         IFS=' ,' read -r -a ENERGIES <<< "${1#*=}"; shift ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# --- prerequisites -----------------------------------------------------------
for tool in curl python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: '$tool' is required but not found in PATH." >&2
        exit 1
    fi
done

# --- resolve storage roots ---------------------------------------------------
WATER_MUON_DATA="${DATA_DIR}/water/muon"
if [[ -n "$STORE_DIR" ]]; then
    mkdir -p "$STORE_DIR"
    STORE_DIR="$(cd "$STORE_DIR" && pwd)"          # absolutize
    WATER_MUON_STORE="${STORE_DIR}/water/muon"
else
    WATER_MUON_STORE="$WATER_MUON_DATA"
fi
mkdir -p "$WATER_MUON_STORE"

echo "======================================"
echo "LUCiD Data Download"
echo "======================================"
echo "Source     : CERNBox share ${SHARE_TOKEN}"
echo "Energies   : ${ENERGIES[*]} MeV"
echo "Real files : ${WATER_MUON_STORE}"
echo "data/ view : ${WATER_MUON_DATA}"
if [[ -n "$STORE_DIR" ]]; then echo "Mode       : external store + symlinks"; else echo "Mode       : in-repo"; fi
echo ""

# --- helpers -----------------------------------------------------------------
# Download one file via WebDAV, with resume (-C -) and skip-if-already-complete.
dav_get() {
    local rel="$1" dest="$2" url remote_size local_size
    url="${DAV_BASE}/${rel}"
    mkdir -p "$(dirname "$dest")"
    remote_size="$(curl -fsSL -I -u "${SHARE_TOKEN}:" "$url" \
        | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)"
    if [[ -f "$dest" && -n "$remote_size" ]]; then
        local_size="$(wc -c < "$dest" | tr -d '[:space:]')"
        if [[ "$local_size" == "$remote_size" ]]; then
            echo "  [skip] ${dest#"$LUCID_ROOT"/} (complete)"
            return 0
        fi
    fi
    echo "  [get ] ${rel}"
    curl -fSL --retry 3 -C - -u "${SHARE_TOKEN}:" "$url" -o "$dest"
}

# List every file (not dir) under a remote dir; prints paths relative to share root.
dav_list() {
    curl -fsSL -X PROPFIND -H "Depth: infinity" -u "${SHARE_TOKEN}:" "${DAV_BASE}/$1/" \
    | python3 -c '
import sys, re, os, urllib.parse as up
tok = os.environ["SHARE_TOKEN"]
prefix = "/remote.php/dav/public-files/" + tok
xml = sys.stdin.read()
for resp in re.findall(r"<d:response>(.*?)</d:response>", xml, re.S):
    href = re.search(r"<d:href>(.*?)</d:href>", resp, re.S)
    size = re.search(r"<d:getcontentlength>(.*?)</d:getcontentlength>", resp, re.S)
    if not href or size is None:
        continue                      # directory entry -> skip
    p = up.unquote(href.group(1))
    if p.startswith(prefix):
        p = p[len(prefix):]
    print(p.lstrip("/"))
'
}

# Remove a path only if it is a symlink (never touch real files/dirs).
drop_symlink() {
    if [[ -L "$1" ]]; then echo "  [rm  ] stale symlink ${1#"$LUCID_ROOT"/}"; rm -f "$1"; fi
    return 0
}

# --- 1. clear stale SIREN symlinks at the store location ---------------------
drop_symlink "${WATER_MUON_STORE}/siren_training"
drop_symlink "${WATER_MUON_STORE}/dedx_siren_training"

# --- 2. ROOT files -----------------------------------------------------------
echo "Downloading ROOT file(s)..."
for E in "${ENERGIES[@]}"; do
    fname="${E}MeV_100events.root"
    dav_get "ROOT_files/water/mu-/${fname}" "${WATER_MUON_STORE}/${fname}"
done

# --- 3. SIREN trained models -------------------------------------------------
# Remap CERNBox dir names to the repo's layout:
#   cherenkov -> siren_training        dedx -> dedx_siren_training
echo "Downloading SIREN model files..."
SIREN_TMP="$(mktemp)"
dav_list "SIREN_files/water/mu-" > "$SIREN_TMP"
while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    sub="${rel#SIREN_files/water/mu-/}"
    case "$sub" in
        cherenkov/*) dest_sub="siren_training/${sub#cherenkov/}" ;;
        dedx/*)      dest_sub="dedx_siren_training/${sub#dedx/}" ;;
        *)           dest_sub="$sub" ;;
    esac
    dav_get "$rel" "${WATER_MUON_STORE}/${dest_sub}"
done < "$SIREN_TMP"
rm -f "$SIREN_TMP"

# Entries that make up a complete water-muon set (used for the symlink steps).
ENTRIES=(siren_training dedx_siren_training)
for E in "${ENERGIES[@]}"; do ENTRIES+=("${E}MeV_100events.root"); done

# --- 4. external store -> data/ symlinks -------------------------------------
if [[ -n "$STORE_DIR" ]]; then
    echo "Linking data/water/muon -> store..."
    mkdir -p "$WATER_MUON_DATA"
    for entry in "${ENTRIES[@]}"; do
        rm -rf "${WATER_MUON_DATA:?}/${entry}"
        ln -s "${WATER_MUON_STORE}/${entry}" "${WATER_MUON_DATA}/${entry}"
    done
fi

# --- 5. wbls reuses water (relative symlinks) --------------------------------
echo "Wiring up data/wbls/muon -> water (symlinks)..."
WBLS_MUON_DATA="${DATA_DIR}/wbls/muon"
mkdir -p "$WBLS_MUON_DATA"
for entry in "${ENTRIES[@]}"; do
    rm -rf "${WBLS_MUON_DATA:?}/${entry}"
    ln -s "../../water/muon/${entry}" "${WBLS_MUON_DATA}/${entry}"
done

# --- 6. verify ---------------------------------------------------------------
echo ""
echo "Verifying..."
ok=1
check() {
    if [[ -e "$1" ]]; then echo "  ok   ${1#"$LUCID_ROOT"/}"; else echo "  MISS ${1#"$LUCID_ROOT"/}"; ok=0; fi
}
check "${WATER_MUON_DATA}/siren_training/trained_model/photonsim_siren_weights.npz"
check "${WATER_MUON_DATA}/dedx_siren_training/trained_model/dedx_siren_weights.npz"
check "${WBLS_MUON_DATA}/siren_training/trained_model/photonsim_siren_weights.npz"
for E in "${ENERGIES[@]}"; do
    check "${WATER_MUON_DATA}/${E}MeV_100events.root"
    check "${WBLS_MUON_DATA}/${E}MeV_100events.root"
done

echo ""
if [[ "$ok" == 1 ]]; then
    echo "======================================"
    echo "Download complete."
    echo "  water: ${WATER_MUON_DATA}"
    echo "  wbls : ${WBLS_MUON_DATA} (symlinks -> water)"
    echo "======================================"
else
    echo "Some files are missing - see MISS lines above." >&2
    exit 1
fi
