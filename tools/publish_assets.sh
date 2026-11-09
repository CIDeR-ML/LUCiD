#!/usr/bin/env bash
# Publish every release asset that tools/make_asset.sh built: one GitHub release per
# tools/assets/<tag>.manifest, tagged <tag> on --target, carrying <tag>.tar.gz and its INDEX.md
# as the notes.
#
#   tools/publish_assets.sh <dir>              # dry run: check everything, print each command
#   tools/publish_assets.sh --go <dir>         # create the releases
#   tools/publish_assets.sh --go --target <ref> <dir>   # tag a commit other than main
#
# <dir> holds <tag>.tar.gz and <tag>.tar.gz.sha256 for every manifest (make_asset.sh --out).
# Before anything is created, every tarball must match its .sha256, and the two assets the
# installer reads must match the checksums tracked in the tree (data/MANIFEST.sha256,
# analysis/paper/RESULTS.sha256). A release that already exists is skipped, never replaced: a
# changed asset is a new tag. Needs `gh auth login` with write access to the repository.
set -euo pipefail

REPO_SLUG="CIDeR-ML/LUCiD"
GO=0
TARGET=main
while [[ $# -gt 0 ]]; do
    case "$1" in
        --go) GO=1; shift ;;
        --target) TARGET="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
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
tags=()
for m in "$REPO_ROOT"/tools/assets/*.manifest; do
    tag="$(sed -n 's/^tag: //p' "$m" | head -1)"
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
    tags+=("$tag")
done
echo "${#tags[@]} assets checked in $DIR; target $TARGET"

for tag in "${tags[@]}"; do
    if gh release view "$tag" --repo "$REPO_SLUG" >/dev/null 2>&1; then
        echo "skip     $tag (already released)"
        continue
    fi
    cmd=(gh release create "$tag" "$DIR/$tag.tar.gz" --repo "$REPO_SLUG" --target "$TARGET"
         --title "$tag" --notes-file "$NOTES/$tag/INDEX.md" --latest=false)
    if [[ $GO -eq 1 ]]; then
        "${cmd[@]}"
        echo "released $tag"
    else
        echo "would run: ${cmd[*]}"
    fi
done
[[ $GO -eq 1 ]] || echo "dry run: nothing was created; rerun with --go"
