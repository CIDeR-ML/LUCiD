#!/bin/bash
# Shared environment for every build-*.sh step.
# Sourced by each build script; idempotent; safe to re-source.

# Conda prefix + gcc/binutils activators — needed so CC/CXX point at the
# conda toolchain and CONDA_BUILD_SYSROOT resolves libc headers for
# rootcling and everything that calls it.
export CONDA_PREFIX=/opt/conda
export PATH=/opt/conda/bin:/usr/local/bin:${PATH}

# activate-gcc_linux-64.sh references unbound vars under set -u.
_old_u=$(set +o | grep nounset); set +u
. /opt/conda/etc/conda/activate.d/activate-binutils_linux-64.sh
. /opt/conda/etc/conda/activate.d/activate-gcc_linux-64.sh
eval "$_old_u"; unset _old_u

# ROOT we build from source at /opt/root (see build-root.sh). Harmless
# to export even before it exists — PATH just won't resolve yet.
export ROOTSYS=/opt/root
export PATH=${ROOTSYS}/bin:${PATH}
export LD_LIBRARY_PATH=${ROOTSYS}/lib:/opt/conda/lib:${LD_LIBRARY_PATH:-}

# Pythia6 shared lib lives here (build-pythia6.sh puts it at both
# /opt/pythia6/libPythia6.so and /opt/conda/lib/libPythia6.so so RPATH
# lookups via conda work without extra env).
export PYTHIA6=/opt/pythia6

# GENIE install prefix (build-genie.sh).
export GENIE=/opt/genie
export GXMLPATH=${GENIE}/config
export PATH=${GENIE}/bin:${PATH}
export LD_LIBRARY_PATH=${GENIE}/lib:${LD_LIBRARY_PATH}

# Host-side download cache, bind-mounted in by build-sandbox.sh so each
# iteration reuses pythia6.tar.gz and the ROOT shallow clone.
: "${BUILD_CACHE_DIR:=/build-cache}"
export BUILD_CACHE_DIR
