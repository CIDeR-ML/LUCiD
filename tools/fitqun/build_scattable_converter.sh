#!/bin/bash
# Build TScatTableF's ROOT dictionary and convert LUCiD's HDF5 scattering
# tables into the file fiTQun reads.
#
# TScatTableF is fiTQun's own class, so the ROOT file must be written by a
# program holding its dictionary — which is why this step is here rather than
# in lucid/production/fitqun/scattable.py. Everything needed is in the LUCiD
# container (ROOT 6.30, HDF5) plus the TScatTable sources from the fiTQun
# Utilities checkout.
#
#   ./build_scattable_converter.sh <Utilities/scattable dir> <in.h5> <out.root>
#
# Verified round-trip: element order and all six axis definitions survive, and
# ROOT's own GetIndex() agrees with the dimension-0-fastest convention
# lucid.production.fitqun.scattable writes.
set -euo pipefail

SRC=${1:?usage: $0 <Utilities/scattable dir> <in.h5> <out.root>}
H5=${2:?missing input .h5}
OUT=${3:?missing output .root}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK_DIR:-$(mktemp -d)}

# TScatTableF.h includes TScatTable.h, so both classes are needed.
cp "$SRC"/TScatTable.{h,cc} "$SRC"/TScatTableF.{h,cc} "$WORK/"
cp "$HERE/h5_to_scattable.C" "$WORK/"

cd "$WORK"
# Both classes carry a ClassDef, so both need streamers in the dictionary --
# a TScatTableF-only dictionary links but dies at load on TScatTable::Streamer.
cat > ScatTableLinkDef.h <<'LINKDEF'
#pragma link C++ class TScatTable+;
#pragma link C++ class TScatTableF+;
LINKDEF
rootcling -f TScatTableFDict.cc -c TScatTable.h TScatTableF.h ScatTableLinkDef.h
g++ -fPIC -shared -o libTScatTableF.so TScatTable.cc TScatTableF.cc TScatTableFDict.cc \
    $(root-config --cflags --libs) -I.

# hdf5.h ships with the container's conda prefix, which ACLiC does not search.
HDF5_INC=${HDF5_INC:-/opt/conda/include}
root -l -b -q \
    -e '.L libTScatTableF.so' \
    -e "gInterpreter->AddIncludePath(\"$HDF5_INC\");" \
    "h5_to_scattable.C+(\"$(realpath "$H5")\",\"$OUT\")"

echo "wrote $OUT"
