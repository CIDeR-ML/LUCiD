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
    # A point that was never produced (deliberately capped momenta, or a culled
    # submission) leaves an empty sensor/ behind, so testing the directory is not
    # enough -- test for shards. Missing outright is a skip; shards that exist
    # but fail to histogram stay fatal, because that is a real defect.
    [ -d "$D/sensor" ] || continue
    compgen -G "$D/sensor/wc_sensor_*.h5" >/dev/null || {
        echo "SKIP $(basename "$(dirname "$(dirname "$(dirname "$D")")")"): never produced"
        continue; }
    TAG=$(basename "$(dirname "$(dirname "$(dirname "$D")")")")
    O="$W/$TAG"; mkdir -p "$O"
    [ -s "$O/events_hist.root" ] && { n=$((n+1)); continue; }   # resumable
    # Every shard, not just the first. A momentum point is produced by several
    # jobs writing wc_*_NNNN.h5 side by side, so `head -1` kept a quarter of a
    # grid that cost hundreds of CPU-hours. makehistWCSim names its output after
    # the input with the extension stripped (makehistWCSim.cc:71-77), so each
    # shard lands in its own file and they are summed -- which is what the name
    # combhists expects, <pdg>_<imom>_hist_sum.root, has always meant.
    PARTS=()
    for SENSOR in "$D"/sensor/wc_sensor_*.h5; do
        [ -s "$SENSOR" ] || continue
        IDX=$(basename "$SENSOR" .h5); IDX=${IDX##*_}
        LABL="$D/labl/wc_labl_$IDX.h5"; STEP="$D/step/wc_step_$IDX.h5"
        [ -s "$LABL" ] && [ -s "$STEP" ] || { echo "SKIP $TAG shard $IDX: no labl/step"; continue; }
        python - <<PY
from lucid.production.fitqun import truth
t = truth.read_tracks("$LABL", "$STEP")
truth.write_text(t, "$O/truth_$IDX.txt")
PY
        python tools/sadqun/export_geometry.py config/sk_geometry.npz \
            --sensor "$SENSOR" -o "$O/geom_$IDX.txt"
        "$E"/bin/lucid_to_wcsim.new --geometry "$O/geom_$IDX.txt" --sensor "$SENSOR" \
            --truth "$O/truth_$IDX.txt" -o "$O/events_$IDX.root"
        ( cd "$O" && "$F"/timepdf_work/makehistWCSim "events_$IDX.root" \
            "$F"/fitqun_SK_WAND.parameters.dat $NWTR > "makehist_$IDX.log" 2>&1 )
        [ -s "$O/events_${IDX}_hist.root" ] || { echo "FAILED $TAG shard $IDX"; exit 1; }
        PARTS+=("$O/events_${IDX}_hist.root")
    done
    [ ${#PARTS[@]} -gt 0 ] || { echo "SKIP $TAG: no usable shards"; continue; }
    if [ ${#PARTS[@]} -eq 1 ]; then
        cp "${PARTS[0]}" "$O/events_hist.root"
    else
        hadd -f "$O/events_hist.root" "${PARTS[@]}" > "$O/hadd.log" 2>&1
    fi
    [ -s "$O/events_hist.root" ] || { echo "FAILED $TAG: hadd"; exit 1; }
    echo "  $TAG: ${#PARTS[@]} shards summed"
    n=$((n+1)); echo "PROGRESS $n/? $TAG"
done
echo "histogrammed $n momentum points"

# combhists.cc:22 opens <pdg>_<imom>_hist_sum.root from the cwd and takes imom
# from the FILENAME as the momentum axis coordinate (armom[nmom]=imom), so the
# per-point histograms have to be staged under that name. Range points get
# their band midpoint, which is the only single momentum they can carry.
for O in "$W"/${PDG}_p*; do
    [ -s "$O/events_hist.root" ] || continue
    B=$(basename "$O"); MOM=${B#${PDG}_p}
    case "$MOM" in
        *_*) LO=${MOM%%_*}; HI=${MOM##*_}; IMOM=$(( (LO + HI) / 2 )) ;;
        *)   IMOM=$MOM ;;
    esac
    # Overlapping bands can share a midpoint (130_190 and 150_170 both give 160;
    # 250_400 and 300_350 both give 325), and a plain cp silently let the later
    # one overwrite the earlier, losing a whole sample. Sum instead -- which is
    # what "_hist_sum" meant.
    DEST="$W/${PDG}_${IMOM}_hist_sum.root"
    if [ -s "$DEST" ]; then
        echo "  imom=$IMOM: summing $B into an existing node"
        hadd -f "$DEST.tmp" "$DEST" "$O/events_hist.root" > "$W/hadd_$IMOM.log" 2>&1 \
            && mv "$DEST.tmp" "$DEST"
    else
        cp "$O/events_hist.root" "$DEST"
    fi
done
echo "staged $(ls "$W"/${PDG}_*_hist_sum.root 2>/dev/null | wc -l) momentum points for combhists"

cd "$W"
root -l -b -q "$F/timepdf_work/combhists.cc($PDG)"
[ -s "${PDG}_tpdfhist.root" ] || { echo "combhists produced nothing"; exit 1; }
root -l -b -q "$F/timepdf_work/fittpdf.cc($PDG,0,1)"
[ -s "${PDG}_tpdfpar.root" ] || { echo "fittpdf produced nothing"; exit 1; }
# A size check cannot tell a good parameterisation from one full of NaN. fittpdf
# exits 0 either way, and an all-NaN gtcmnpar/gtcsgpar gives -lnL=nan on every
# event -- which looks like a slow fit, not a broken tune. Refuse to install it.
root -l -b -q "$L/tools/fitqun/check_tpdfpar.C(\"${PDG}_tpdfpar.root\")" | tee tpdfpar_check.log
grep -q "TPDFPAR_OK" tpdfpar_check.log || { echo "tpdfpar failed validation -- NOT installing"; exit 1; }
cp "${PDG}_tpdfpar.root" "$F/fiTQun/const/${PDG}_tpdfpar_SK_WAND_PMT20inch.root"
echo TIMEPDF_DONE
