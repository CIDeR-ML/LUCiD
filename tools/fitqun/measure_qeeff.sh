#!/bin/bash
# Measure fiTQun's QEEff on a HELD-OUT sample, using fiTQun's own procedure.
#
#   measure_qeeff.sh <sample-dir> <skip> <n> <tag> [pdg]
#
# QEEff is not a quantum efficiency. It is the scalar that turns fiTQun's
# *emitted* photon count into predicted photoelectrons,
#
#     fact = Phi0_local[iPID] * QEEff * QEEffCorr * nphot      (fiTQun.cc:2082)
#
# so predicted charge is strictly linear in it and it absorbs the whole
# normalisation chain at once -- Phi0_local, the nphot convention, the emission
# band the generator used, PMT QE, collection efficiency. It therefore cannot be
# derived from a QE curve: the same curve integrates to anywhere between 6.7%
# and 16.3% depending only on the band chosen. It has to be measured, and
# re-measured whenever the photon yield, QE curve, digitizer or geometry change.
#
# There is no second knob. QEEffCorr, which looks made for exactly this, is
# assigned only inside if(IsItRealData()) (fiTQun.cc:685-700) and is forced to 1
# for a simulated sample, so the whole MC charge scale has to live in QEEff.
#
# THE METHOD. fiTQun measures this itself: fiTQun.TuningMode = 4 makes
# runfiTQun.cc:754 call fiTQun::FitQEEff, which pins all eight track parameters
# at MC truth and floats QEEff alone over [0,0.2] (fiTQun.cc:4770-4845). Because
# the track cannot move, the fit cannot buy agreement by shrinking momentum --
# which is the whole point, and the reason this must not be done by iterating on
# reconstructed momentum.
#
# Reading the result: FitQEEff stores the fitted value through
# SaveDefaultSnglTrkFit(X,0,0,iPID,totmu,QEEff_best,PCflg), i.e. in the slot that
# normally carries the negative log-likelihood. In TuningMode 4, fq1rnll holds
# QEEff, not an NLL.
#
# ORDER MATTERS. FitQEEffWrapper minimises GetOneRngnglogL, the FULL likelihood
# including the time term (unlike FitMu, which explicitly disables it). Measuring
# QEEff against a borrowed or out-of-range time PDF bakes that error into the
# charge scale, so install the tune's own time PDF first.
#
# HELD OUT. -s/-n select an event range (TRuntimeParameters.cxx:380-415). Pass a
# <skip> that puts this range outside whatever events the reported resolutions
# are scored on: fitting the scale on the scoring events flatters the result.
set -euo pipefail
SAMPLE=${1:?usage: $0 <sample-dir> <skip> <n> <tag> [pdg] [shard]}
SKIP=${2:?missing skip count}
NEV=${3:?missing n events}
TAG=${4:?missing tag}
PDG=${5:-13}
SHARD=${6:-0000}

# SHARD is explicit, and it matters. Shards of one campaign can reuse the seeds
# of another: mu_metrics job_000001 and the mu_fast2 scoring sample both carry
# "782985707 2140288342", so its wc_sensor_0000 IS the scoring set. Taking
# `ls wc_sensor_*.h5 | head -1` would fit the charge scale on the very events
# being scored, which is precisely what held-out is supposed to prevent. Check
# the seeds in the campaign's job_*.mac before choosing.

L=/afs/cern.ch/work/c/cjesus/DIFFSIM/LUCiD
F=/afs/cern.ch/work/c/cjesus/DIFFSIM/fitqun
E=/eos/project-n/neutrino-generators/cjesus/fitqun
W=$E/work/$TAG
mkdir -p "$W"

export PYTHONPATH=$L
export FITQUN_ROOT=${FITQUN_ROOT_OVERRIDE:-$F/fiTQun}
export LD_LIBRARY_PATH=$F/wcsim_build/lib:${LD_LIBRARY_PATH:-}
need() { [ -s "$1" ] || { echo "MISSING $1 -- stopping"; exit 1; }; }

SENSOR="$SAMPLE/sensor/wc_sensor_$SHARD.h5"
LABL="$SAMPLE/labl/wc_labl_$SHARD.h5"
STEP="$SAMPLE/step/wc_step_$SHARD.h5"
need "$SENSOR"; need "$LABL"; need "$STEP"

TOT=$((SKIP + NEV))
cd "$L"
python - <<PY
from lucid.production.fitqun import truth
t = truth.read_tracks("$LABL", "$STEP", max_events=$TOT)
truth.write_text(t, "$W/truth.txt")
print(f"truth: {len(t)} events (need {$TOT} to cover skip+n)")
PY
need "$W/truth.txt"

python tools/fitqun/export_geometry.py config/sk_geometry.npz --sensor "$SENSOR" -o "$W/geom.txt"
need "$W/geom.txt"

"$E"/bin/lucid_to_wcsim.new --geometry "$W/geom.txt" --sensor "$SENSOR" \
    --gates "$LABL" --truth "$W/truth.txt" -o "$W/events.root" -n "$TOT"
need "$W/events.root"

# TuningMode=4 on top of the live tune, so the measurement sees exactly the
# tables the fitter will use. Everything else is inherited, deliberately.
PARS=$W/tune4.parameters.dat
sed -e 's/^\( *< *fiTQun.TuningMode *= *\).*$/\1 4 >/' "$F/fitqun_SK_WAND.parameters.dat" > "$PARS"
grep -q "fiTQun.TuningMode" "$PARS" || echo "     < fiTQun.TuningMode = 4 >" >> "$PARS"
echo "TuningMode: $(grep -o 'fiTQun.TuningMode *= *[0-9]*' "$PARS")"

cd "$F/fiTQun"
./runfiTQunWC -p "$PARS" -s "$SKIP" -n "$NEV" \
    -r "$W/qeeff.root" "$W/events.root" > "$W/qeeff.log" 2>&1
need "$W/qeeff.root"

cd "$L"
python tools/fitqun/qeeff_summary.py --fq "$W/qeeff.root" --pdg "$PDG" \
    --manifest "$F/tune_manifest_SK_WAND.json" \
    --held-out "$SAMPLE events $SKIP-$((SKIP+NEV-1))" | tee "$W/qeeff.txt"
echo QEEFF_DONE
