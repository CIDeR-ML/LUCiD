#!/bin/bash
# Setup for GENIE v3.06.02 on S3DF via the FNAL cvmfs spack stack.
#
# After sourcing, `gevgen` and `gntpc` are on PATH and all GENIE runtime
# deps (lhapdf, log4cpp, pythia6, gsl, libxml2, ROOT 6.28) are on
# LD_LIBRARY_PATH. GENIE_PREFIX and GENIE_XSEC_FILE are exported for the
# Python runner (lucid.production.run_genie).
#
# The cvmfs GENIE binary was built against glibc 2.34 and will not run
# on the S3DF host OS (glibc 2.28). It must be invoked inside the LUCiD
# singularity image (Ubuntu 22.04, glibc 2.35), which is the environment
# the S3DF generate_jobs.sh sbatch template already uses for LUCiD.
#
# We avoid `spack load` (which relies on spack's Python shell helpers and
# has trouble resolving user scopes inside a minimal container) and
# instead discover the required library directories directly from
# `ldd gevgen`. This is stable as long as the spack install hashes don't
# change; point LARSOFT_SPACK at a newer release if the binary moves.

LARSOFT_SPACK=/cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.0
SPACK_OPT="${LARSOFT_SPACK}/spack_env/opt/spack/linux-x86_64_v2"
GENIE_PKG="genie-3.06.02-y7h2qnoz2ubymyzchdekvhubp5s7figs"

if [ ! -d "${SPACK_OPT}/${GENIE_PKG}" ]; then
    echo "setup_genie.sh: ${SPACK_OPT}/${GENIE_PKG} not found — cvmfs not mounted or spack layout changed." >&2
    return 1 2>/dev/null || exit 1
fi

export GENIE_PREFIX="${SPACK_OPT}/${GENIE_PKG}"
export GENIE="${GENIE_PREFIX}"

# Discover all spack dep lib dirs that gevgen needs, deduplicate, prepend.
_genie_dep_libs="$(ldd "${GENIE_PREFIX}/bin/gevgen" 2>/dev/null \
                   | awk '/=>/ {print $3}' \
                   | grep "^${SPACK_OPT}" \
                   | xargs -I {} dirname {} \
                   | sort -u \
                   | tr '\n' ':')"

export PATH="${GENIE_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${GENIE_PREFIX}/lib:${_genie_dep_libs}${LD_LIBRARY_PATH}"

# Default tune xsec splines (override by pre-setting GENIE_XSEC_FILE).
export GENIE_XSEC_FILE="${GENIE_XSEC_FILE:-/cvmfs/larsoft.opensciencegrid.org/products/genie_xsec/v3_06_00/NULL/G1802a00000-k250-e1000/data/gxspl-FNALsmall.xml}"

# GENIE config dir — messenger configs live under $GENIE/config.
export GXMLPATH="${GENIE_PREFIX}/config"

unset _genie_dep_libs

echo "setup_genie.sh: GENIE_PREFIX=${GENIE_PREFIX}"
echo "setup_genie.sh: GENIE_XSEC_FILE=${GENIE_XSEC_FILE}"
command -v gevgen >/dev/null && echo "setup_genie.sh: gevgen: $(command -v gevgen)"
