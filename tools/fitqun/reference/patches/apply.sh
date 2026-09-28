#!/bin/bash
# Re-apply our fixes to a fresh fiTQun tuning checkout.
#
#   apply.sh <fitqun-tree>
#
# We cannot push upstream, so these fixes are carried here indefinitely. They
# are patches rather than whole-file copies for two reasons: the diff is the
# documentation, and if upstream ever touches the same lines this FAILS LOUDLY
# instead of silently reverting or silently overwriting their change.
#
# Upstream sources use CRLF and the patches are generated against LF-normalised
# text, so each target is converted to LF before patching. -l alone is not
# enough: patch still fails to match CRLF context lines.
set -euo pipefail
TREE=${1:?usage: apply.sh <fitqun-tree>}
HERE=$(cd "$(dirname "$0")" && pwd)

# Pinned upstream commits these patches were generated against.
declare -A PINNED=(
  [Utilities]=c0a0916f00ac259d8d4975910b5f029c027e054f
  [WCSimFQTuner]=bfd18d36c39de74871b431052418cbd539e00587
  [fiTQun]=752bfb6b8f71b948e604a883058ba8948cac7904
)
for repo in "${!PINNED[@]}"; do
  [ -d "$TREE/$repo/.git" ] || { echo "no $repo checkout under $TREE"; exit 1; }
  have=$(git -C "$TREE/$repo" rev-parse HEAD)
  [ "$have" = "${PINNED[$repo]}" ] || \
    echo "WARNING: $repo is at $have, patches were made against ${PINNED[$repo]}"
done

declare -A TARGET=(
  [makehistWCSim.cc]=Utilities/timepdf/makehistWCSim.cc
  [fittpdf.cc]=Utilities/timepdf/fittpdf.cc
  [combhists.cc]=Utilities/timepdf/combhists.cc
  [writecprof.cc]=Utilities/cprofile/writecprof.cc
  [runfiTQun.cc]=fiTQun/runfiTQun.cc
  [fiTQun.cc]=fiTQun/fiTQun.cc
)
for p in "$HERE"/*.patch; do
  n=$(basename "$p" .patch)
  t="$TREE/${TARGET[$n]:?unknown patch $n}"
  [ -f "$t" ] || { echo "missing target $t"; exit 1; }
  tr -d '\r' < "$t" > "$t.lf" && mv "$t.lf" "$t"   # normalise before matching
  if patch -l -p0 --dry-run -f "$t" < "$p" >/dev/null 2>&1; then
    patch -l -p0 -f "$t" < "$p" >/dev/null && echo "applied  $n"
  elif patch -l -p0 -R --dry-run -f "$t" < "$p" >/dev/null 2>&1; then
    echo "already  $n"
  else
    echo "CONFLICT $n -- upstream changed these lines; reconcile by hand"; exit 1
  fi
done
echo "all patches applied"
