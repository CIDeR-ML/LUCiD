#!/bin/bash
# 500-iter re-runs of the 1-key combos to find their TRUE floor (they were still descending at
# 250). Readout chosen post-hoc from saved traj (polyak10/final). 50 events, 5 GPUs × 10 events.
# Combos passed as args: B5 C5 D5 (subset ok). D5=1M photons (run only if OOM smoke passed).
set -u
ROOT=/sdf/group/neutrino/omara/LUCiD_unification
cd "$ROOT"
NEV=50
GPUS=${GPUS:-"0 1 2 3 4"}                       # which physical GPUs to use
declare -a GPUARR=($GPUS); NGPU=${#GPUARR[@]}
PERGPU=$(( (NEV + NGPU - 1) / NGPU ))           # ceil, last gpu gets the remainder
declare -A NK=( [B5]=1 [C5]=1 [D5]=1 )
declare -A NP=( [B5]=250000 [C5]=500000 [D5]=1000000 )
run_combo () {
  local NAME=$1
  local OUTDIR="$ROOT/scripts/campaign_recon/out_$NAME"
  mkdir -p "$OUTDIR"
  echo "=== $NAME: NKEYS=${NK[$NAME]} NPH=${NP[$NAME]} NITERS=500 GPUS='$GPUS' -> $OUTDIR ($(date)) ==="
  pids=()
  local i=0
  for g in "${GPUARR[@]}"; do
    local START=$((i*PERGPU)); local CNT=$PERGPU
    [ $((START+CNT)) -gt $NEV ] && CNT=$((NEV-START)); [ $CNT -le 0 ] && { i=$((i+1)); continue; }
    CUDA_VISIBLE_DEVICES=$g NKEYS=${NK[$NAME]} NPH=${NP[$NAME]} FISHER_MODE=ad LR=1 NITERS=500 \
      EVENT_START=$START EVENT_COUNT=$CNT OUT="$OUTDIR" \
      python scripts/campaign_recon/worker.py > "$OUTDIR/gpu$g.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== $NAME done ($(date)); $(ls "$OUTDIR"/ev*.npz 2>/dev/null | wc -l)/$NEV events ==="
}
for combo in "$@"; do run_combo "$combo"; done
echo "=== ALL DONE ($(date)) ==="
