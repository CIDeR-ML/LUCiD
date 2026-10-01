#!/bin/bash
# LUCiD events -> fiTQun reconstruction -> resolutions, in one step.
#
#   run_reconstruction.sh <sample-dir> <n-events> <tag>
#
# <sample-dir> holds LUCiD's sensor/, labl/ and step/ modality directories. The
# whole chain is: export truth -> convert to WCSim format -> runfiTQunWC ->
# score. Every stage is gated on the artifact it should have produced, because
# ROOT exits non-zero even when a macro succeeds.
set -x
SAMPLE=${1:?usage: $0 <sample-dir> <n-events> <tag> [shard]}
NEV=${2:?missing n events}
TAG=${3:?missing tag}
SHARD=${4:-0000}

# SHARD is explicit because a campaign has several, and `ls | head -1` silently
# pinned every run to 0000. It also matters for held-out bookkeeping: QEEff was
# fitted on mu_metrics shard 0001, so scoring must avoid that shard, and
# mu_metrics shard 0000 reuses the mu_fast2 scoring seed (782985707 2140288342).

L=/afs/cern.ch/work/c/cjesus/DIFFSIM/LUCiD
F=/afs/cern.ch/work/c/cjesus/DIFFSIM/fitqun
E=/eos/project-n/neutrino-generators/cjesus/fitqun
W=$E/work/$TAG
mkdir -p "$W"

export PYTHONPATH=$L
export FITQUN_ROOT=${FITQUN_ROOT_OVERRIDE:-$F/fiTQun}
export LD_LIBRARY_PATH=$F/wcsim_build/lib:$LD_LIBRARY_PATH

need() { [ -s "$1" ] || { echo "MISSING $1 -- stopping"; exit 1; }; }

SENSOR="$SAMPLE/sensor/wc_sensor_$SHARD.h5"
LABL="$SAMPLE/labl/wc_labl_$SHARD.h5"
STEP="$SAMPLE/step/wc_step_$SHARD.h5"
need "$SENSOR"; need "$LABL"; need "$STEP"

cd "$L"
python - <<PY
from lucid.production.fitqun import truth
t = truth.read_tracks("$LABL", "$STEP", max_events=$NEV)
truth.write_text(t, "$W/truth.txt")
print(f"truth: {len(t)} single-primary events")
PY
need "$W/truth.txt"

# The geometry MUST be exported in the sensor file's sensor_idx order; the
# sk_geometry.npz order is a different permutation and mapping hits through it
# attributes every hit to an unrelated PMT.
python tools/fitqun/export_geometry.py config/sk_geometry.npz \
    --sensor "$SENSOR" -o "$W/geom.txt"
need "$W/geom.txt"

# --gates reuses LUCiD's own trigger gates and writes one subevent per gate in
# WCSim's time frame; without it every gate is merged and the prompt peak lands
# wherever the generator's t=0 put it.
GATES="$LABL"
"$E"/bin/lucid_to_wcsim.new --geometry "$W/geom.txt" --sensor "$SENSOR" \
    --gates "$GATES" --truth "$W/truth.txt" -o "$W/events.root" -n "$NEV"
need "$W/events.root"

cd "$F/fiTQun"
PARS=${PARFILE:-$F/fitqun_SK_WAND.parameters.dat}
./runfiTQunWC -p "$PARS" -n "$NEV" \
    -r "$W/fq.root" "$W/events.root" > "$W/fq.log" 2>&1
need "$W/fq.root"

cd "$L"
# --require-fc: score only what fiTQun fits as fully contained, so arms with
# different tunes are compared on the same footing rather than on whatever each
# one happened to flag.
python tools/fitqun/fq_resolution.py --fq "$W/fq.root" \
    --labl "$LABL" --step "$STEP" --pdg 13 --require-fc | tee "$W/resolution.txt"
echo RECON_DONE
