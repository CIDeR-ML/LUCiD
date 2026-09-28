#!/bin/bash
# Run ONE ablation of fiTQun's light model and dump its per-PMT prediction.
#
#   ablate.sh <sample-dir> <n-events> <shard> <name> [override ...]
#
# Each <override> is "fiTQun.Key=value" and is applied to a copy of the live
# tune, so an ablation differs from the baseline in exactly the parameters you
# name and nothing else. The track is pinned at MC truth (TuningMode=4), so the
# comparison is of the FORWARD MODEL alone -- no fitted vertex, direction or
# momentum to absorb a discrepancy and confound the result.
#
# Output is $E/work/abl_<name>/pmt.txt, one row per PMT per event:
#   event icab flgHit flgMask R_cm costheta0 mu muscat chrg tHit tau tres qeeff
# `mu` (direct) and `muscat` (indirect) are separate, so a run can be split by
# which term dominates without re-running.
#
# Knobs that isolate a single term (from fiTQun.parameters.dat):
#   fiTQun.UseScatteredLight        0 off / 1 6D / 2 3D / 4 6D-for-1R
#   fiTQun.WaterAttenuationLengthWCSim   large value -> no attenuation
#   fiTQun.DarkRateWCSim            0 -> no dark term
#   fiTQun.UseTimeLikelihood        0 -> charge-only likelihood
#   fiTQun.TotalChargeConstraintWeight   0 -> drop the total-charge constraint
#   fiTQun.QEEffWCSim               pure multiplicative scale on predicted charge
set -euo pipefail
SAMPLE=${1:?usage: $0 <sample-dir> <n-events> <shard> <name> [key=value ...]}
NEV=${2:?missing n events}
SHARD=${3:?missing shard}
NAME=${4:?missing ablation name}
shift 4

L=/afs/cern.ch/work/c/cjesus/DIFFSIM/LUCiD
F=/afs/cern.ch/work/c/cjesus/DIFFSIM/fitqun
E=/eos/project-n/neutrino-generators/cjesus/fitqun
W=$E/work/abl_$NAME
rm -rf "$W"; mkdir -p "$W"

export PYTHONPATH=$L FITQUN_ROOT=$F/fiTQun
export LD_LIBRARY_PATH=$F/wcsim_build/lib:${LD_LIBRARY_PATH:-}
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
print(f"truth: {len(t)} events")
PY
need "$W/truth.txt"
python tools/sadqun/export_geometry.py config/sk_geometry.npz --sensor "$SENSOR" -o "$W/geom.txt"
need "$W/geom.txt"
"$E"/bin/lucid_to_wcsim.new --geometry "$W/geom.txt" --sensor "$SENSOR" \
    --gates "$LABL" --truth "$W/truth.txt" -o "$W/events.root" -n "$NEV"
need "$W/events.root"

# Apply the overrides to a copy of the LIVE tune, so the only differences from
# the baseline are the ones named on the command line.
PARS=$W/ablation.parameters.dat
sed -e 's/^\( *< *fiTQun.TuningMode *= *\).*$/\1 4 >/' "$F/fitqun_SK_WAND.parameters.dat" > "$PARS"
grep -q "fiTQun.TuningMode" "$PARS" || echo "     < fiTQun.TuningMode = 4 >" >> "$PARS"
for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    if grep -q "< *$k *=" "$PARS"; then
        sed -i "s|^\( *< *$k *= *\).*$|\1 $v >|" "$PARS"
    else
        echo "     < $k = $v >" >> "$PARS"
    fi
    echo "  override: $k = $v"
done
echo "--- effective tune ---"
grep -oE "fiTQun\.(UseScatteredLight|WaterAttenuationLengthWCSim|DarkRateWCSim|QEEffWCSim|UseTimeLikelihood|UseFitCProfile|TotalChargeConstraintWeight|TuningMode) *= *[0-9.e+-]+" "$PARS"

cd "$F/fiTQun"
FITQUN_PMT_DUMP="$W/pmt.txt" ./runfiTQunWC -p "$PARS" -s 0 -n "$NEV" \
    -r "$W/out.root" "$W/events.root" > "$W/run.log" 2>&1
need "$W/pmt.txt"
echo "ABLATION_DONE $NAME  $(wc -l < "$W/pmt.txt") rows"
