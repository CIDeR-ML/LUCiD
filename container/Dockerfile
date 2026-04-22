# syntax=docker/dockerfile:1.6
# Docker build of the LUCiD unified image, mirroring lucid-base.def + lucid.def.
# Reuses container/scripts/*.sh step-for-step, so the Apptainer and Docker
# builds stay in lock-step.
#
# Build (from the PARENT dir that contains both LUCiD/ and PhotonSim/):
#   docker build --platform linux/amd64 -f LUCiD/container/Dockerfile \
#       -t lucid:latest .
#
# On Apple Silicon this runs under Rosetta 2 binary translation. conda-forge
# does not ship geant4 for linux-aarch64, so linux/amd64 is the only
# supported arch today.
#
# See docs/QUICKSTART_DOCKER.md for the full runbook.

FROM --platform=linux/amd64 condaforge/miniforge3:latest

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# Layer 1: apt (build toolchain, X/XML/SSL dev headers used by ROOT + GENIE)
# -----------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates git wget curl tar bzip2 xz-utils \
        build-essential gfortran autoconf automake libtool pkg-config \
        libx11-dev libxpm-dev libxft-dev libxext-dev libxml2-dev \
        libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Layer 2: conda (GEANT4 11.3, Python stack, GENIE deps, conda gcc/g++ 13)
# Matches lucid-base.def one-for-one.
# -----------------------------------------------------------------------------
RUN mamba install -y -n base -c conda-forge \
        geant4=11.3 \
        python=3.11 \
        cmake make ninja \
        gxx_linux-64=13 gcc_linux-64=13 \
        gsl lhapdf log4cpp libxml2 libxml2-devel pcre zlib \
        jax=0.4 flax optax numpy scipy h5py uproot awkward matplotlib \
        pandas seaborn plotly ipython ipywidgets scikit-learn pillow \
        pytest tqdm pyyaml \
        tbb-devel \
 && mamba clean -afy \
 && ln -sf /opt/conda/bin/x86_64-conda-linux-gnu-gcc /opt/conda/bin/gcc \
 && ln -sf /opt/conda/bin/x86_64-conda-linux-gnu-g++ /opt/conda/bin/g++

# -----------------------------------------------------------------------------
# Layer 3: build scripts (copy once, reuse across all subsequent build steps)
# -----------------------------------------------------------------------------
COPY LUCiD/container/scripts /opt/container-scripts
RUN chmod +x /opt/container-scripts/*.sh

# -----------------------------------------------------------------------------
# Layer 4: Pythia6 (from ROOT's prebuilt tarball, ~30s)
# -----------------------------------------------------------------------------
RUN /opt/container-scripts/build-pythia6.sh

# -----------------------------------------------------------------------------
# Layer 5: ROOT 6.28.12 from source with -Dpythia6=ON (~45 min; slow layer).
# Changing anything above this line re-triggers the ROOT build — keep
# lower-churn work (conda, apt) strictly above.
# -----------------------------------------------------------------------------
RUN /opt/container-scripts/build-root.sh

# -----------------------------------------------------------------------------
# Layer 6: GENIE 3.06.02 from source against libEGPythia6 (~5 min)
# -----------------------------------------------------------------------------
RUN /opt/container-scripts/build-genie.sh

# -----------------------------------------------------------------------------
# Layer 7: GENIE cross-section splines (G18_02a_00_000, ~455 MB download,
# expands to ~500 MB on disk). Baked into the image so runs are
# self-contained — no cvmfs, no first-run network dependency.
# -----------------------------------------------------------------------------
RUN /opt/container-scripts/fetch-xsec-splines.sh --dest /opt/genie_xsec

# -----------------------------------------------------------------------------
# Layer 8: PhotonSim (edits to PhotonSim only invalidate from here down)
# -----------------------------------------------------------------------------
COPY PhotonSim /opt/PhotonSim
RUN /opt/container-scripts/build-photonsim.sh

# -----------------------------------------------------------------------------
# Layer 9: LUCiD (editable; edits to LUCiD only invalidate this final layer)
# -----------------------------------------------------------------------------
COPY LUCiD /opt/LUCiD
RUN /opt/container-scripts/build-lucid.sh

# -----------------------------------------------------------------------------
# Runtime env — matches lucid-base.def + lucid.def %environment verbatim.
# -----------------------------------------------------------------------------
ENV PATH=/opt/root/bin:/opt/genie/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/root/lib:/opt/conda/lib:/opt/genie/lib \
    ROOTSYS=/opt/root \
    GENIE=/opt/genie \
    GENIE_PREFIX=/opt/genie \
    GXMLPATH=/opt/genie/config \
    PHOTONSIM_BIN=/opt/PhotonSim/build/PhotonSim \
    GENIE_XSEC_FILE=/opt/genie_xsec/gxspl-FNALsmall.xml \
    CC=/opt/conda/bin/gcc \
    CXX=/opt/conda/bin/g++

LABEL org.opencontainers.image.title="LUCiD" \
      org.opencontainers.image.description="GEANT4 11.3 + Pythia6 + ROOT 6.28.12 + GENIE 3.06.02 + PhotonSim + LUCiD"

WORKDIR /work
CMD ["bash"]
