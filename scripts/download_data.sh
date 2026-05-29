#!/bin/bash
#
# Download LUCiD example data (water muon + electron) from the CERNBox
# public share.
#
# Default:             real files land in   LUCiD/data/water/{muon,electron}/
# With --store-dir D:  real files land in   D/water/{muon,electron}/   and
#                      LUCiD/data/water/{muon,electron}/ holds symlinks there.
#
# In both cases data/wbls/{muon,electron}/ is populated with relative symlinks
# into ../../water/{muon,electron}/ (ROOT + SIREN), so wbls reuses the water
# files.
#
# The share holds three ROOT energies per particle (500/1000/1500 MeV). Only
# 1000 MeV is fetched by default; use --all-energies for all three. The SIREN
# trained models (Cherenkov + dE/dx) are always fetched, for both particles.

set -euo pipefail

# --- CERNBox share -----------------------------------------------------------
SHARE_TOKEN="vOhH8P78hnSQKNZ"
export SHARE_TOKEN
DAV_BASE="https://cernbox.cern.ch/remote.php/dav/public-files/${SHARE_TOKEN}"

# --- paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUCID_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${LUCID_ROOT}/data"

# CERNBox uses short particle names (mu-, e-); the repo's data/ tree uses long
# names (muon, electron). Each entry is "<cernbox_name> <repo_name>".
PARTICLES=("mu- muon" "e- electron")

# --- defaults ----------------------------------------------------------------
STORE_DIR=""
ENERGIES=(1000)

usage() {
    cat <<EOF
Usage: $(basename "$0") [--store-dir DIR] [--all-energies | --energies "E1 E2"]

Downloads LUCiD water example data (ROOT + trained SIREN models) for both muon
and electron from the CERNBox share, and wires up data/wbls/{muon,electron}/ as
symlinks reusing the water files.

Options:
  --store-dir DIR     Put the real files under DIR/water/{muon,electron}/ and
                      leave symlinks in LUCiD/data/water/{muon,electron}/.
                      Default: store the files directly in LUCiD/data/.
                      On S3DF use: --store-dir /sdf/data/neutrino/cjesus/CIDER/CERNBOX
  --all-energies      Fetch all ROOT energies (500, 1000, 1500 MeV) per particle.
  --energies "LIST"   Space-separated ROOT energies in MeV (e.g. "500 1000").
  -h, --help          Show this help.

Default ROOT energy: 1000 MeV only, for both mu- and e-.
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

if [[ -n "$STORE_DIR" ]]; then
    mkdir -p "$STORE_DIR"
    STORE_DIR="$(cd "$STORE_DIR" && pwd)"          # absolutize
fi

echo "======================================"
echo "LUCiD Data Download"
echo "======================================"
echo "Source     : CERNBox share ${SHARE_TOKEN}"
echo "Particles  : muon, electron"
echo "Energies   : ${ENERGIES[*]} MeV"
if [[ -n "$STORE_DIR" ]]; then echo "Mode       : external store + symlinks ($STORE_DIR)"; else echo "Mode       : in-repo"; fi
echo ""

# --- helpers -----------------------------------------------------------------
# Download one file via WebDAV, with resume (-C -) and skip-if-already-complete.
dav_get() {
    local rel="$1" dest="$2" url remote_size local_size
    url="${DAV_BASE}/${rel}"
    mkdir -p "$(dirname "$dest")"
    remote_size="$(curl -fsSL --retry 3 -I -u "${SHARE_TOKEN}:" "$url" \
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
    curl -fsSL --retry 3 -X PROPFIND -H "Depth: infinity" -u "${SHARE_TOKEN}:" "${DAV_BASE}/$1/" \
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

# --- verify bookkeeping ------------------------------------------------------
ok=1
check() {
    if [[ -e "$1" ]]; then echo "  ok   ${1#"$LUCID_ROOT"/}"; else echo "  MISS ${1#"$LUCID_ROOT"/}"; ok=0; fi
}

# --- per-particle fetch ------------------------------------------------------
# fetch_particle <cernbox_particle> <repo_particle>   e.g. "mu- muon" / "e- electron"
fetch_particle() {
    local cb="$1" repo="$2"
    local data="${DATA_DIR}/water/${repo}" store wbls="${DATA_DIR}/wbls/${repo}"
    if [[ -n "$STORE_DIR" ]]; then store="${STORE_DIR}/water/${repo}"; else store="$data"; fi
    mkdir -p "$store"

    echo "--------------------------------------"
    echo "water/${repo}  (CERNBox: ${cb})"
    echo "  real files : ${store}"
    echo "  data/ view : ${data}"
    echo ""

    # 1. clear stale SIREN symlinks at the store location
    drop_symlink "${store}/siren_training"
    drop_symlink "${store}/dedx_siren_training"

    # 2. ROOT files
    echo "Downloading ROOT file(s)..."
    local E fname
    for E in "${ENERGIES[@]}"; do
        fname="${E}MeV_100events.root"
        dav_get "ROOT_files/water/${cb}/${fname}" "${store}/${fname}"
    done

    # 3. SIREN trained models. Remap CERNBox dir names to the repo's layout:
    #    cherenkov -> siren_training        dedx -> dedx_siren_training
    echo "Downloading SIREN model files..."
    local siren_tmp rel sub dest_sub
    siren_tmp="$(mktemp)"
    dav_list "SIREN_files/water/${cb}" > "$siren_tmp"
    while IFS= read -r rel; do
        [[ -z "$rel" ]] && continue
        sub="${rel#SIREN_files/water/${cb}/}"
        case "$sub" in
            cherenkov/*) dest_sub="siren_training/${sub#cherenkov/}" ;;
            dedx/*)      dest_sub="dedx_siren_training/${sub#dedx/}" ;;
            *)           dest_sub="$sub" ;;
        esac
        dav_get "$rel" "${store}/${dest_sub}"
    done < "$siren_tmp"
    rm -f "$siren_tmp"

    # Entries that make up a complete set (used for the symlink steps).
    local entries=(siren_training dedx_siren_training) entry
    for E in "${ENERGIES[@]}"; do entries+=("${E}MeV_100events.root"); done

    # 4. external store -> data/ symlinks
    if [[ -n "$STORE_DIR" ]]; then
        echo "Linking data/water/${repo} -> store..."
        mkdir -p "$data"
        for entry in "${entries[@]}"; do
            rm -rf "${data:?}/${entry}"
            ln -s "${store}/${entry}" "${data}/${entry}"
        done
    fi

    # 5. wbls reuses water (relative symlinks)
    echo "Wiring up data/wbls/${repo} -> water (symlinks)..."
    mkdir -p "$wbls"
    for entry in "${entries[@]}"; do
        rm -rf "${wbls:?}/${entry}"
        ln -s "../../water/${repo}/${entry}" "${wbls}/${entry}"
    done

    # 6. verify
    echo ""
    echo "Verifying water/${repo}..."
    check "${data}/siren_training/trained_model/photonsim_siren_weights.npz"
    check "${data}/dedx_siren_training/trained_model/dedx_siren_weights.npz"
    check "${wbls}/siren_training/trained_model/photonsim_siren_weights.npz"
    for E in "${ENERGIES[@]}"; do
        check "${data}/${E}MeV_100events.root"
        check "${wbls}/${E}MeV_100events.root"
    done
    echo ""
}

# --- run for each particle ---------------------------------------------------
for spec in "${PARTICLES[@]}"; do
    # shellcheck disable=SC2086
    fetch_particle ${spec}
done

if [[ "$ok" == 1 ]]; then
    echo "======================================"
    echo "Download complete."
    for spec in "${PARTICLES[@]}"; do
        repo="${spec#* }"
        echo "  water/${repo} : ${DATA_DIR}/water/${repo}"
        echo "  wbls/${repo}  : ${DATA_DIR}/wbls/${repo} (symlinks -> water)"
    done
    echo "======================================"
else
    echo "Some files are missing - see MISS lines above." >&2
    exit 1
fi
