# syntax=docker/dockerfile:1.6
# LUCiD container image, layered on top of the NUISANCE tutorial
# container (ships ROOT 6.30 with Pythia6 support, GENIE 3.04, Pythia6,
# and xsec splines for the AR23_20i_00_000, G18_10a_02_11b, and
# G21_11a_00_000 tunes). Adds conda GEANT4 11.3 + PhotonSim + LUCiD.
#
# Build (from the PARENT dir that contains both LUCiD/ and PhotonSim/):
#   docker build --platform linux/amd64 -f LUCiD/container/Dockerfile \
#       -t lucid:latest .
#
# On S3DF, Apptainer pulls the same image directly:
#   apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest
#
# See docs/guides/production/docker.md and docs/guides/production/deploy-s3df.md.

FROM --platform=linux/amd64 nuisancemc/tutorial:nuint2024

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# Layer 1: miniforge3 (conda/mamba). The base image has Python 3.10 via apt
# but no conda. We need conda to pull GEANT4 11.3 and the jax stack.
# -----------------------------------------------------------------------------
RUN wget -q -O /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh

# Prepend conda to PATH so the subsequent mamba/pip/python invocations pick
# it up. Base image's PATH already points at ROOT and GENIE binaries.
ENV PATH=/opt/conda/bin:${PATH}

# -----------------------------------------------------------------------------
# Layer 2: GEANT4 11.3 + LUCiD's Python deps. GEANT4 11.3 is required because
# PhotonSim.cc uses G4RunManagerFactory::CreateRunManager, an API introduced
# in Geant4 11.0 (the Ubuntu 22.04 apt package ships 10.7, which lacks it).
# -----------------------------------------------------------------------------
RUN mamba install -y -n base -c conda-forge \
        geant4=11.3 \
        python=3.11 \
        cmake make \
        jax=0.4 flax optax numpy scipy h5py uproot awkward matplotlib \
        pandas seaborn plotly ipython ipywidgets scikit-learn pillow \
        pytest tqdm pyyaml \
 && mamba clean -afy

# GEANT4 data libs live under /opt/conda/share/Geant4-11.3.*/data and are
# picked up via geant4-config; prepend conda to LD_LIBRARY_PATH so the
# Geant4 libs resolve at PhotonSim runtime alongside the base image's ROOT.
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

# -----------------------------------------------------------------------------
# Layer 3: build PhotonSim. Finds Geant4 via conda's geant4-config and ROOT
# via the base image's versioned install at /opt/root/v6-30-04.
# -----------------------------------------------------------------------------
COPY PhotonSim /opt/PhotonSim
# CMAKE_PREFIX_PATH=/opt/conda forces cmake to prefer conda's libs (expat,
# xerces-c, zlib, …) over the older system copies under /usr/lib. Geant4
# 11.3's config requires expat >= 2.7.5, while Ubuntu 22.04 ships 2.4.7.
#
# The linker flags point ld at conda's newer libstdc++.so.6. Conda-forge
# Geant4 11.3 needs CXXABI_1.3.15 / GLIBCXX_3.4.32, which the Ubuntu 22.04
# system libstdc++ doesn't export. -Wl,-rpath bakes the lookup into the
# resulting binary so PhotonSim runs with the same runtime it linked
# against.
RUN rm -rf /opt/PhotonSim/build \
 && mkdir -p /opt/PhotonSim/build \
 && cd /opt/PhotonSim/build \
 && cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/opt/conda \
          -DCMAKE_EXE_LINKER_FLAGS="-L/opt/conda/lib -Wl,-rpath,/opt/conda/lib" \
          -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/conda/lib -Wl,-rpath,/opt/conda/lib" \
          -DGeant4_DIR=$(geant4-config --prefix)/lib/cmake/Geant4 \
          -DROOT_DIR=/opt/root/v6-30-04/cmake \
          .. \
 && make -j"$(nproc)" \
 && test -x /opt/PhotonSim/build/PhotonSim

# -----------------------------------------------------------------------------
# Layer 4: install LUCiD as editable. Uses conda's pip (Python 3.11) so the
# lucid-run-job console script lands on conda's PATH.
# -----------------------------------------------------------------------------
COPY LUCiD /opt/LUCiD
# The build context excludes .git/, so setuptools-scm can't infer the
# package version the usual way. Override with a synthetic pretend
# version (normalized dist name of "lucid-sim" → LUCID_SIM).
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LUCID_SIM=0.0.0+docker \
    pip install --no-deps -e /opt/LUCiD \
 && command -v lucid-run-job \
 && python3 -c "import lucid.production.run_job as m; print(m.__file__)"

# -----------------------------------------------------------------------------
# Runtime env. The base image already sets ROOTSYS, GENIE, GENIE_XSEC_DIR,
# GENIE_XSEC_FILE, PYTHIA6, GXMLPATH, ROOT_INCLUDE_PATH, and all the right
# PATH / LD_LIBRARY_PATH entries for ROOT + GENIE + Pythia6 + NUISANCE +
# NEUT + NuWro + LHAPDF.
#
# Geant4 11.3 requires a handful of G4*DATA env vars so it can locate its
# data tables at runtime (photon evaporation, ENSDF state, radioactive
# decay, EM low-energy cross sections, etc.). conda-forge ships these
# under /opt/conda/share/Geant4/data/ and would normally wire them up via
# the activate scripts under /opt/conda/etc/conda/activate.d/activate-
# geant4-data-*.sh. We don't activate the conda env at runtime, so
# replicate those exports statically here. Version suffixes pin the
# exact dataset revisions this image was built with (geant4 11.3 from
# conda-forge).
# -----------------------------------------------------------------------------
ENV PHOTONSIM_BIN=/opt/PhotonSim/build/PhotonSim \
    G4ABLADATA=/opt/conda/share/Geant4/data/ABLA3.3 \
    G4CHANNELINGDATA=/opt/conda/share/Geant4/data/CHANNELING1.0 \
    G4LEDATA=/opt/conda/share/Geant4/data/EMLOW8.6.1 \
    G4ENSDFSTATEDATA=/opt/conda/share/Geant4/data/ENSDFSTATE3.0 \
    G4INCLDATA=/opt/conda/share/Geant4/data/INCL1.2 \
    G4NEUTRONHPDATA=/opt/conda/share/Geant4/data/NDL4.7.1 \
    G4PARTICLEXSDATA=/opt/conda/share/Geant4/data/PARTICLEXS4.1 \
    G4LEVELGAMMADATA=/opt/conda/share/Geant4/data/PhotonEvaporation6.1 \
    G4PIIDATA=/opt/conda/share/Geant4/data/PII1.3 \
    G4RADIOACTIVEDATA=/opt/conda/share/Geant4/data/RadioactiveDecay6.1.2 \
    G4REALSURFACEDATA=/opt/conda/share/Geant4/data/RealSurface2.2 \
    G4SAIDXSDATA=/opt/conda/share/Geant4/data/SAIDDATA2.0

LABEL org.opencontainers.image.title="LUCiD" \
      org.opencontainers.image.description="nuisancemc/tutorial:nuint2024 + conda GEANT4 11.3 + PhotonSim + LUCiD"

WORKDIR /work
CMD ["bash"]
