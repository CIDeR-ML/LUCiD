#!/bin/bash
# Turn the time-PDF momentum grid into <pdg>_tpdfpar_<config>_<pmt>.root.
#
#   build_timepdf.sh <pdg> <grid-root> <work-dir>
#
# Per momentum point: truth -> geometry -> WCSim conversion -> makehistWCSim.
# Then combhists stacks the per-momentum 2D histograms into a TH3D over
# momentum, and fittpdf parameterises it.
#
# nwtr is read from the tune's own fiTQun.WaterRefractiveIndex rather than
# hardcoded. makehistWCSim's nwtr and the fitter's WaterRefractiveIndex are
# SEPARATE constants that both happen to default to 1.38, so hardcoding either
# one lets them drift apart -- and a PDF built at one index while the fitter
# subtracts TOF at another carries a systematic nothing else will reveal.
set -euo pipefail
PDG=${1:?usage: build_timepdf.sh <pdg> <grid-root> <work-dir>}
GRID=${2:?missing grid root}
W=${3:?missing work dir}

L=/afs/cern.ch/work/c/cjesus/DIFFSIM/LUCiD
F=/afs/cern.ch/work/c/cjesus/DIFFSIM/fitqun
E=/eos/project-n/neutrino-generators/cjesus/fitqun
NWTR=$(grep -oP "fiTQun.WaterRefractiveIndex\\s*=\\s*\\K[0-9.]+" "$F/fitqun_SK_WAND.parameters.dat")
[ -n "$NWTR" ] || { echo "tune does not set fiTQun.WaterRefractiveIndex"; exit 1; }
echo "using nwtr=$NWTR (from the tune, matching the fitter)"

export PYTHONPATH=$L FITQUN_ROOT=$F/fiTQun
export LD_LIBRARY_PATH=$F/wcsim_build/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$W"; cd "$L"

n=0
for D in "$GRID"/${PDG}_p*/SK_WAND/tpdf/config_*; do
    [ -d "$D/sensor" ] || continue
    TAG=$(basename "$(dirname "$(dirname "$(dirname "$D")")")")
    O="$W/$TAG"; mkdir -p "$O"
    [ -s "$O/events_hist.root" ] && { n=$((n+1)); continue; }   # resumable
    SENSOR=$(ls "$D"/sensor/*.h5 2>/dev/null | head -1) || continue
    LABL=$(ls "$D"/labl/*.h5 | head -1); STEP=$(ls "$D"/step/*.h5 | head -1)
    python - <<PY
from lucid.production.fitqun import truth
t = truth.read_tracks("$LABL", "$STEP")
truth.write_text(t, "$O/truth.txt")
PY
    python tools/sadqun/export_geometry.py config/sk_geometry.npz --sensor "$SENSOR" -o "$O/geom.txt"
    "$E"/bin/lucid_to_wcsim.new --geometry "$O/geom.txt" --sensor "$SENSOR" \
        --truth "$O/truth.txt" -o "$O/events.root"
    ( cd "$O" && "$F"/timepdf_work/makehistWCSim events.root \
        "$F"/fitqun_SK_WAND.parameters.dat $NWTR > makehist.log 2>&1 )
    [ -s "$O/events_hist.root" ] || { echo "FAILED $TAG"; exit 1; }
    n=$((n+1)); echo "PROGRESS $n/? $TAG"
done
echo "histogrammed $n momentum points"

# combhists and fittpdf expect the per-momentum files in the cwd, named by the
# grid's own convention, so stage them there before running.
cd "$W"
root -l -b -q "$F/timepdf_work/combhists.cc($PDG)"
[ -s "${PDG}_tpdfhist.root" ] || { echo "combhists produced nothing"; exit 1; }
root -l -b -q "$F/timepdf_work/fittpdf.cc($PDG,0,1)"
[ -s "${PDG}_tpdfpar.root" ] || { echo "fittpdf produced nothing"; exit 1; }
cp "${PDG}_tpdfpar.root" "$F/fiTQun/const/${PDG}_tpdfpar_SK_WAND_PMT20inch.root"
echo TIMEPDF_DONE
