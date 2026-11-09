#!/usr/bin/env bash
# Publish every release asset that tools/make_asset.sh built. Each manifest
# tools/assets/<tag>.manifest names its GitHub release with `release:` (default:
# the tag itself); the asset is the file <tag>.tar.gz attached to that release.
#
#   tools/publish_assets.sh <dir>              # dry run: check everything, print each command
#   tools/publish_assets.sh --go <dir>         # create the releases, attach the files
#   tools/publish_assets.sh --go --target <ref> <dir>   # tag a commit other than main
#
# <dir> holds <tag>.tar.gz and <tag>.tar.gz.sha256 for every manifest (make_asset.sh --out).
# Before anything is created, every tarball must match its .sha256, and the two assets the
# installer reads must match the checksums tracked in the tree (data/MANIFEST.sha256,
# analysis/paper/RESULTS.sha256). A release that does not exist is created; a file already
# attached is skipped, never replaced: a changed asset is a new tag. A release that carries one
# asset takes that asset's INDEX.md as its notes; one that carries several lists them.
# Needs `gh auth login` with write access to the repository.
set -euo pipefail

REPO_SLUG="CIDeR-ML/LUCiD"
GO=0
TARGET=main
while [[ $# -gt 0 ]]; do
    case "$1" in
        --go) GO=1; shift ;;
        --target) TARGET="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[[ $# -eq 1 ]] || { echo "usage: tools/publish_assets.sh [--go] [--target REF] <dir>" >&2; exit 2; }
DIR="$(cd "$1" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v gh >/dev/null || { echo "gh (GitHub CLI) is not installed" >&2; exit 1; }

# The installer's assets: what each installs must be what the tree's manifest lists.
declare -A TRACKED=([lucid-data-v1]=data/MANIFEST.sha256
                    [lucid-paper-results-v1]=analysis/paper/RESULTS.sha256)

NOTES="$(mktemp -d)"
trap 'rm -rf "$NOTES"' EXIT
declare -A ASSETS=()        # release -> space-separated tags, in manifest order
releases=()
for m in "$REPO_ROOT"/tools/assets/*.manifest; do
    tag="$(sed -n 's/^tag: //p' "$m" | head -1)"
    release="$(sed -n 's/^release: //p' "$m" | head -1)"
    release="${release:-$tag}"
    tarball="$DIR/$tag.tar.gz"
    [[ -f "$tarball" && -f "$tarball.sha256" ]] || { echo "missing $tarball (or its .sha256)" >&2; exit 1; }
    (cd "$DIR" && sha256sum --quiet -c "$tag.tar.gz.sha256") \
        || { echo "$tarball does not match $tarball.sha256" >&2; exit 1; }
    tar -xzf "$tarball" -C "$NOTES" "$tag/INDEX.md" "$tag/MANIFEST.sha256"
    if [[ -n "${TRACKED[$tag]:-}" ]]; then
        diff <(grep -vE '  \./(INDEX\.md|PROVENANCE\.txt)$' "$NOTES/$tag/MANIFEST.sha256" \
                   | sed 's#  \./#  #' | sort -k2) \
             <(grep -v '^#' "$REPO_ROOT/${TRACKED[$tag]}" | sort -k2) >/dev/null \
            || { echo "$tag: its files do not match ${TRACKED[$tag]}" >&2; exit 1; }
    fi
    [[ -n "${ASSETS[$release]:-}" ]] || releases+=("$release")
    ASSETS[$release]="${ASSETS[$release]:-} $tag"
done
echo "${#releases[@]} releases checked in $DIR; target $TARGET"

run() { if [[ $GO -eq 1 ]]; then "$@"; else echo "would run: $*"; fi; }

for release in "${releases[@]}"; do
    read -r -a tags <<< "${ASSETS[$release]}"
    notes="$NOTES/$release.notes.md"
    if [[ ${#tags[@]} -eq 1 ]]; then
        cp "$NOTES/${tags[0]}/INDEX.md" "$notes"
    else
        { printf '# %s\n\nThe archived research record of LUCiD: one tarball per study, each with an\n' "$release"
          printf '`INDEX.md` (what it was for, the commit it ran against, how to run it), a\n'
          printf '`PROVENANCE.txt` and a `MANIFEST.sha256`. `docs/studies.md` in the repository\n'
          printf 'lists them. Every file here is under CC BY 4.0.\n\n| asset | what it was for | size | ran against |\n|---|---|---|---|\n'
          for tag in "${tags[@]}"; do
              "$REPO_ROOT/tools/make_asset.sh" --row "$REPO_ROOT/tools/assets/$tag.manifest"
          done; } > "$notes"
    fi
    if gh release view "$release" --repo "$REPO_SLUG" >/dev/null 2>&1; then
        attached="$(gh release view "$release" --repo "$REPO_SLUG" --json assets --jq '.assets[].name')"
        echo "exists   $release"
    else
        attached=""
        run gh release create "$release" --repo "$REPO_SLUG" --target "$TARGET" \
            --title "$release" --notes-file "$notes" --latest=false
    fi
    for tag in "${tags[@]}"; do
        if grep -qx "$tag.tar.gz" <<< "$attached"; then
            echo "skip     $release/$tag.tar.gz (already attached)"
        else
            run gh release upload "$release" "$DIR/$tag.tar.gz" --repo "$REPO_SLUG"
        fi
    done
done
[[ $GO -eq 1 ]] || echo "dry run: nothing was created; rerun with --go"
