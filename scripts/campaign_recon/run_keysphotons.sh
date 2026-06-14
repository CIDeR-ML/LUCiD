#!/bin/bash
# keys×photons study driver: 3 combos, all AD-Fisher lr=1, 100 events each, 5 GPUs × 20 events.
# A=nkeys4/nph250k (AD-equiv of old recipe), B=nkeys1/nph250k, C=nkeys1/nph500k.
set -u
ROOT=/sdf/group/neutrino/omara/LUCiD_unification
cd "$ROOT"
NEV=100; PERGPU=20; NGPU=5
run_combo () {
  local NAME=$1 NK=$2 NP=$3
  local OUTDIR="$ROOT/scripts/campaign_recon/out_$NAME"
  mkdir -p "$OUTDIR"
  echo "=== combo $NAME: NKEYS=$NK NPH=$NP -> $OUTDIR ($(date)) ==="
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    local START=$((g*PERGPU))
    CUDA_VISIBLE_DEVICES=$g NKEYS=$NK NPH=$NP FISHER_MODE=ad LR=1 NITERS=250 \
      EVENT_START=$START EVENT_COUNT=$PERGPU OUT="$OUTDIR" \
      python scripts/campaign_recon/worker.py > "$OUTDIR/gpu$g.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== combo $NAME done ($(date)); $(ls "$OUTDIR"/ev*.npz 2>/dev/null | wc -l)/$NEV events ==="
}
run_combo A 4 250000
run_combo B 1 250000
run_combo C 1 500000
echo "=== ALL COMBOS DONE ($(date)) ==="
