#!/usr/bin/env bash
# Build one LUCiD release asset from a manifest.
#
# Usage:   tools/make_asset.sh [--out DIR] [--force] <manifest>
#          tools/make_asset.sh --row <manifest>
#
# A manifest is tools/assets/<tag>.manifest:
#
#     tag: overlap-tables                  # the asset's name (its tarball is <tag>.tar.gz)
#     release: lucid-archives              # the GitHub release carrying it (default: the tag)
#     commit: eaf2e89f                     # the commit the material ran against
#     summary: the warm overlap cache      # the docs/studies.md cell
#     size: 60 KB                          # the docs/studies.md cell
#     part: spatial_overlap_integrals      # a path in this worktree
#     part: geometry-v2:docs/POOL_MARCHING.md      # a path on another ref
#     extern-root: LUCID_ASSET_SRC         # env var holding the extern root
#     extern: model.npz -> water/muon/model.npz   # on no ref; under that root
#     origin: water/muon/model.npz <- where it was made  # copied into PROVENANCE.txt
#     --- INDEX ---
#     <the asset's INDEX.md, verbatim to the end of the file>
#
# Blank lines and `#` comments are ignored before the marker and kept after it.
#
# Publishing is by hand: the script prints the `gh release create` line and
# stops.  v4-light has no release workflow, and nothing here touches a remote.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: make_asset.sh [--out DIR] [--force] <manifest>
       make_asset.sh --row <manifest>

Builds <tag>.tar.gz from a manifest and writes its checksums beside it, then
prints the `gh release create` line to run by hand.  Nothing is uploaded.

Options:
  --out DIR   Output directory.  Required, either here or as $LUCID_ASSET_OUT;
              there is no default (no script in this tree writes to a personal
              path).
  --force     Overwrite an existing <tag>.tar.gz.  Without it an existing
              tarball is an error, so a rebuild is always deliberate.
  --row       Print this asset's docs/studies.md table row and exit; builds
              nothing, reads no part, needs no --out.
  -h, --help  Show this help.

The asset carries, at the root of the extracted tree: INDEX.md (what the
material was for, the commit it ran against, how to run it, every known
defect, and the CC BY 4.0 licence every asset is under), PROVENANCE.txt (every part, its ref, the resolved commit) and
MANIFEST.sha256 (every file in the asset).

Material that is on no ref (a gitignored cache, trained weights, a ROOT
slice) is an `extern:` part, resolved against the directory named by the
manifest's `extern-root:` env var.  That indirection is why no tracked
manifest carries anyone's absolute path, and an unset root is an error rather
than a silently empty asset.

Two refusals are built in, because a tarball with no provenance is worse than
a deleted directory -- it looks runnable:
  * no INDEX body in the manifest -> no asset;
  * no `commit:` header, or one that does not resolve -> no asset.
EOF
}

die() { printf 'make_asset: %s\n' "$*" >&2; exit 1; }

# --- arguments ---------------------------------------------------------------
OUT_DIR="${LUCID_ASSET_OUT:-}"
FORCE=0
ROW_ONLY=0
MANIFEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT_DIR="${2:?--out needs a directory}"; shift 2 ;;
        --out=*)   OUT_DIR="${1#*=}"; shift ;;
        --force)   FORCE=1; shift ;;
        --row)     ROW_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*)        usage >&2; die "unknown option: $1" ;;
        *)         [[ -z "$MANIFEST" ]] || die "one manifest at a time (got '$MANIFEST' and '$1')"
                   MANIFEST="$1"; shift ;;
    esac
done
[[ -n "$MANIFEST" ]] || { usage >&2; exit 1; }
[[ -f "$MANIFEST" ]] || die "no such manifest: $MANIFEST"

REPO_ROOT="$(git -C "$(dirname "$MANIFEST")" rev-parse --show-toplevel)"

# --- parse the manifest -----------------------------------------------------
# Header keys, then `part:`/`extern:` lines, then everything after the INDEX
# marker verbatim.
TAG=""; RELEASE=""; COMMIT=""; SUMMARY=""; SIZE=""; EXTERN_ROOT_VAR=""
PARTS=()
ORIGINS=()
INDEX_BODY=""
in_index=0

while IFS= read -r line || [[ -n "$line" ]]; do
    if (( in_index )); then
        INDEX_BODY+="$line"$'\n'
        continue
    fi
    case "$line" in
        '--- INDEX ---')  in_index=1 ;;
        '#'*|'')          ;;
        'tag: '*)         TAG="${line#tag: }" ;;
        'commit: '*)      COMMIT="${line#commit: }" ;;
        'release: '*)     RELEASE="${line#release: }" ;;
        'summary: '*)     SUMMARY="${line#summary: }" ;;
        'size: '*)        SIZE="${line#size: }" ;;
        'extern-root: '*) EXTERN_ROOT_VAR="${line#extern-root: }" ;;
        'part: '*)        PARTS+=("part ${line#part: }") ;;
        'extern: '*)      PARTS+=("extern ${line#extern: }") ;;
        'origin: '*)      ORIGINS+=("${line#origin: }") ;;
        *)                die "$MANIFEST: unparsable line: $line" ;;
    esac
done < "$MANIFEST"

[[ -n "$TAG" ]] || die "$MANIFEST: no 'tag:' header"

# --- refusal 2: no recorded commit, no asset --------------------------------
[[ -n "$COMMIT" ]] || die "$TAG: no 'commit:' header -- an asset with no recorded provenance looks runnable and is not"
COMMIT_FULL="$(git -C "$REPO_ROOT" rev-parse --verify "${COMMIT}^{commit}" 2>/dev/null)" \
    || die "$TAG: commit '$COMMIT' does not resolve in $REPO_ROOT"

if (( ROW_ONLY )); then
    printf '| %s | %s | %s | %s |\n' "$TAG" "${SUMMARY:-—}" "${SIZE:-—}" "${COMMIT:0:8}"
    exit 0
fi

# --- refusal 1: no INDEX.md, no asset ---------------------------------------
[[ -n "${INDEX_BODY//[$'\n\t ']/}" ]] \
    || die "$TAG: no INDEX body after the '--- INDEX ---' marker -- refusing to build an asset that cannot say what it is"
(( ${#PARTS[@]} > 0 )) || die "$TAG: no 'part:'/'extern:' lines"

[[ -n "$OUT_DIR" ]] || die "no output directory: pass --out DIR or set LUCID_ASSET_OUT"
mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/$TAG.tar.gz"
if [[ -e "$TARBALL" && $FORCE -eq 0 ]]; then
    die "$TARBALL exists; pass --force to rebuild it"
fi

# --- stage ------------------------------------------------------------------
# Staged inside the destination filesystem, then moved, so a killed job never
# leaves a partial tarball that a later --force-less run takes for complete.
STAGE="$(mktemp -d "$OUT_DIR/.stage.$TAG.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT
ROOT="$STAGE/$TAG"
mkdir -p "$ROOT"

# Every asset is data under one licence, stated in its INDEX.md.
{
    printf '%s' "$INDEX_BODY"
    printf '\n\n## Licence\n\n'
    printf 'Copyright (c) 2024-2026 The LUCiD authors. Licensed under the Creative Commons\n'
    printf 'Attribution 4.0 International licence (CC BY 4.0,\n'
    printf 'https://creativecommons.org/licenses/by/4.0/): cite the LUCiD paper when you use it.\n'
} > "$ROOT/INDEX.md"

PROV="$ROOT/PROVENANCE.txt"
{
    printf 'asset: %s\n' "$TAG"
    printf 'built: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'commit: %s\n' "$COMMIT_FULL"
    printf 'built-by: tools/make_asset.sh from %s\n' "${MANIFEST#"$REPO_ROOT"/}"
    printf '\nparts (kind, source, resolved commit):\n'
} > "$PROV"

for entry in "${PARTS[@]}"; do
    kind="${entry%% *}"
    spec="${entry#* }"
    case "$kind" in
    part)
        if [[ "$spec" == *:* && "$spec" != /* ]]; then
            ref="${spec%%:*}"; path="${spec#*:}"
            ref_commit="$(git -C "$REPO_ROOT" rev-parse --verify "${ref}^{commit}" 2>/dev/null)" \
                || die "$TAG: ref '$ref' does not resolve"
            git -C "$REPO_ROOT" archive --format=tar "$ref_commit" -- "$path" \
                | tar -x -C "$ROOT" \
                || die "$TAG: '$path' is not on $ref"
            printf '  tracked  %s:%s  %s\n' "$ref" "$path" "$ref_commit" >> "$PROV"
        else
            [[ -e "$REPO_ROOT/$spec" ]] || die "$TAG: no such path in the worktree: $spec"
            tar -c -C "$REPO_ROOT" -f - -- "$spec" | tar -x -C "$ROOT"
            printf '  worktree %s  %s\n' "$spec" "$COMMIT_FULL" >> "$PROV"
        fi
        ;;
    extern)
        # `extern: <source> -> <dest in asset>`: material that is not in any
        # ref (gitignored caches, trained weights, a ROOT slice).  Its commit
        # is the asset's recorded commit -- the code it was produced by.
        src="${spec%% -> *}"; dest="${spec##* -> }"
        [[ "$src" != "$spec" ]] || die "$TAG: extern needs '<source> -> <dest>': $spec"
        # The staging root is the builder's own directory, so the record names the
        # variable, not its value; where the material came from is an `origin:` line.
        label="$src"
        if [[ "$src" != /* ]]; then
            [[ -n "$EXTERN_ROOT_VAR" ]] \
                || die "$TAG: relative extern '$src' with no 'extern-root:' header"
            root="${!EXTERN_ROOT_VAR:-}"
            [[ -n "$root" ]] \
                || die "$TAG: \$$EXTERN_ROOT_VAR is unset, so '$src' cannot be resolved"
            label="\$$EXTERN_ROOT_VAR/$src"
            src="$root/$src"
        fi
        [[ -e "$src" ]] || die "$TAG: no such external path: $src"
        mkdir -p "$ROOT/$(dirname "$dest")"
        cp -RL -- "$src" "$ROOT/$dest"
        printf '  extern   %s -> %s  %s\n' "$label" "$dest" "$COMMIT_FULL" >> "$PROV"
        ;;
    esac
done

if (( ${#ORIGINS[@]} > 0 )); then
    printf '\norigins:\n' >> "$PROV"
    printf '  %s\n' "${ORIGINS[@]}" >> "$PROV"
fi

# --- checksums --------------------------------------------------------------
( cd "$ROOT" && find . -type f ! -name MANIFEST.sha256 -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum > MANIFEST.sha256 )

# --- tarball ----------------------------------------------------------------
TMP_TAR="$STAGE/$TAG.tar.gz.part"
tar -c -C "$STAGE" -f - -- "$TAG" | gzip -n -9 > "$TMP_TAR"
mv -f -- "$TMP_TAR" "$TARBALL"
cp -f -- "$ROOT/MANIFEST.sha256" "$OUT_DIR/$TAG.MANIFEST.sha256"
( cd "$OUT_DIR" && sha256sum "$TAG.tar.gz" > "$TAG.tar.gz.sha256" )

printf '%s  %s\n' "$(du -h "$TARBALL" | cut -f1)" "$TARBALL"
printf 'docs/studies.md row:\n'
printf '| %s | %s | %s | %s |\n' "$TAG" "${SUMMARY:-—}" "${SIZE:-—}" "${COMMIT:0:8}"
printf '\nPublish with tools/publish_assets.sh %s (release %s; nothing above touched a remote)\n' \
    "$OUT_DIR" "${RELEASE:-$TAG}"
