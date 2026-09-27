#!/bin/bash
# Wait for a charge-PDF merge to appear, adapt it to the loader's naming, install
# it into the tune, and run the reconstruction end to end.
#
#   install_chargepdf_and_recon.sh <cPDFpar.root> <sample-dir> <n-events> <tag>
set -x
SRC=${1:?usage: $0 <cPDFpar.root> <sample-dir> <n-events> <tag>}
SAMPLE=${2:?missing sample dir}; NEV=${3:?missing n events}; TAG=${4:?missing tag}
L=/afs/cern.ch/work/c/cjesus/DIFFSIM/LUCiD
F=/afs/cern.ch/work/c/cjesus/DIFFSIM/fitqun
E=/eos/project-n/neutrino-generators/cjesus/fitqun

# MakecPDFparFile keeps the PMT type in every object name; the loader wants them
# un-suffixed, one file per type. See tools/fitqun/reference/PATCHES.md.
root -l -b -q "$L/tools/fitqun/cpdfpar_for_type.C(\"$SRC\",0,\"$E/const/cPDFpar_PMT20inch.root\")"
[ -s "$E/const/cPDFpar_PMT20inch.root" ] || { echo "adapter produced nothing"; exit 1; }
ls -la "$E/const/cPDFpar_PMT20inch.root"

bash "$L/tools/fitqun/run_reconstruction.sh" "$SAMPLE" "$NEV" "$TAG"
