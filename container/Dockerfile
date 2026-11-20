# syntax=docker/dockerfile:1.6
# LUCiD container image, layered on top of the NUISANCE tutorial
# container (ships ROOT 6.30 with Pythia6 support, GENIE 3.04, Pythia6,
# and xsec splines for the AR23_20i_00_000, G18_10a_02_11b, and
# G21_11a_00_000 tunes). Adds LUCiD's Python stack, PhotonSim with Geant4
# 11.3.2 and ROOT 6.32.16 in a conda environment of its own, and LUCiD.
#
# Build (from the PARENT dir that contains both LUCiD/ and PhotonSim/):
#   docker build --platform linux/amd64 -f LUCiD/container/Dockerfile \
#       -t lucid:latest .
#
# On S3DF, Apptainer pulls the same image directly:
#   apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest
#
# See docs/guides/production/container.md and docs/guides/production/clusters.md.

FROM --platform=linux/amd64 nuisancemc/tutorial:nuint2024

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# Layer 1: miniforge3 (conda/mamba). The base image has Python 3.10 via apt
# but no conda. We need conda for Geant4, ROOT 6.32 and the jax stack.
# -----------------------------------------------------------------------------
RUN wget -q -O /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh

# Prepend conda to PATH so the subsequent mamba/pip/python invocations pick
# it up. Base image's PATH already points at ROOT and GENIE binaries.
ENV PATH=/opt/conda/bin:${PATH}

# -----------------------------------------------------------------------------
# Layer 2: LUCiD's Python stack, at the T1-LArTPC image's versions where it has
# them (Python, jax, numpy, scipy, h5py and HDF5).
# -----------------------------------------------------------------------------
RUN mamba install -y -n base -c conda-forge \
        python=3.11.17 \
        jax=0.6.2 jaxlib=0.6.2 numpy=2.4.6 scipy=1.17.1 h5py=3.16.0 hdf5=1.14.6 \
        flax optax uproot awkward matplotlib \
        pandas seaborn plotly ipython ipywidgets scikit-learn pillow \
        pytest tqdm pyyaml \
 && mamba clean -afy

# Conda's libraries first, as in every earlier image, so the Python stack and
# the processes it starts resolve conda's newer runtime (libstdc++ and others).
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

# -----------------------------------------------------------------------------
# Layer 3: PhotonSim's toolchain, in a conda environment of its own: Geant4
# 11.3.2 (the build that made WAND v4) with ROOT 6.32.16, g++ 14.3 and CMake
# 3.31.8, the T1-LArTPC image's versions. GEANT4 11 is required because
# PhotonSim.cc uses G4RunManagerFactory::CreateRunManager, an API introduced in
# Geant4 11.0.
#
# The event generators of the base image keep their ROOT 6.30, and neither ROOT
# may see the other's headers or libraries: this environment is never on
# LD_LIBRARY_PATH, PhotonSim is configured and built in a clean shell, and it
# runs through /opt/PhotonSim/photonsim, which clears the base image's ROOT
# variables and activates the environment (which also sets Geant4's G4*DATA).
# -----------------------------------------------------------------------------
RUN mamba create -y -n photonsim -c conda-forge \
        geant4=11.3.2=py311h866470e_7 clhep=2.4.7.1 \
        root_base=6.32.16 gxx_linux-64=14.3 cmake=3.31.8 make expat \
        python=3.11.17 \
 && mamba clean -afy

COPY PhotonSim /opt/PhotonSim
COPY LUCiD/container/photonsim /opt/PhotonSim/photonsim
RUN rm -rf /opt/PhotonSim/build \
 && env -i HOME=/root PATH=/usr/bin:/bin bash -c '\
        source /opt/conda/etc/profile.d/conda.sh && conda activate photonsim \
        && cmake -S /opt/PhotonSim -B /opt/PhotonSim/build -DCMAKE_BUILD_TYPE=Release \
        && cmake --build /opt/PhotonSim/build -j"$(nproc)"' \
 && chmod +x /opt/PhotonSim/photonsim \
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
 && python3 -c "import lucid.production.dataprod.run_job as m; print(m.__file__)"

# Warm the overlap-integral lookup cache for the detectors we ship, at the library's default
# temperature (0.2); each table takes a minute or two to build. The cache is gitignored, so it is
# never in the build context. Production jobs run at temperature 0 (the hard step) and never read
# these tables; the warm entries serve interactive use of the image and keep a read-only install
# off the write path.
RUN python3 -c "\
import glob; \
from lucid.geometry.detector_geometry import DetectorGeometry; \
[ (DetectorGeometry.from_config(f), print('warmed', f)) for f in sorted(glob.glob('/opt/LUCiD/config/*_WAND_geom_config.json')) ]" \
 && ls -1 /opt/LUCiD/spatial_overlap_integrals/

# -----------------------------------------------------------------------------
# Runtime env. The base image already sets ROOTSYS, GENIE, GENIE_XSEC_DIR,
# GENIE_XSEC_FILE, PYTHIA6, GXMLPATH, ROOT_INCLUDE_PATH, and all the right
# PATH / LD_LIBRARY_PATH entries for ROOT + GENIE + Pythia6 + NUISANCE +
# NEUT + NuWro + LHAPDF.
#
# PhotonSim runs through /opt/PhotonSim/photonsim (layer 3): the photonsim
# environment it activates sets the G4*DATA variables Geant4 needs to find its
# data sets.
# -----------------------------------------------------------------------------
ENV PHOTONSIM_BIN=/opt/PhotonSim/photonsim

# -----------------------------------------------------------------------------
# Build provenance. The build context excludes .git/, so without this an image
# cannot say which sources it was built from — which is how a .sif drifted four
# months behind the registry unnoticed. Recorded both as a file the image can
# print about itself and as OCI labels readable without running it. lucid.utils.code_version
# reads lucid_sha, lucid_dirty (true/false: tracked changes in the build context) and lucid_ref
# (the release tag, "none" for an untagged build) into every dataset batch's config/git_commit,
# git_dirty and lucid_ref, and lucid.utils.photonsim_version reads photonsim_ref, photonsim_sha
# and geant4 into its photonsim_ref, photonsim_commit and geant4_version; keep the key=value
# format.
# -----------------------------------------------------------------------------
ARG LUCID_SHA=unknown
ARG LUCID_DIRTY=unknown
ARG LUCID_REF=unknown
ARG PHOTONSIM_SHA=unknown
ARG PHOTONSIM_REF=unknown
ARG BUILD_DATE=unknown
RUN printf 'lucid_sha=%s\nlucid_dirty=%s\nlucid_ref=%s\nlucid_source=%s\nphotonsim_sha=%s\nphotonsim_ref=%s\nphotonsim_source=%s\nbuild_date=%s\ngeant4=%s\n' \
      "$LUCID_SHA" "$LUCID_DIRTY" "$LUCID_REF" "https://github.com/CIDeR-ML/LUCiD" \
      "$PHOTONSIM_SHA" "$PHOTONSIM_REF" "https://github.com/cesarjesusvalls/PhotonSim" \
      "$BUILD_DATE" "$(/opt/conda/envs/photonsim/bin/geant4-config --version 2>/dev/null || echo unknown)" \
      > /opt/VERSIONS \
 && cat /opt/VERSIONS

LABEL org.opencontainers.image.title="LUCiD" \
      org.opencontainers.image.description="nuisancemc/tutorial:nuint2024 + PhotonSim (Geant4 11.3.2, ROOT 6.32.16) + LUCiD" \
      org.opencontainers.image.source="https://github.com/CIDeR-ML/LUCiD" \
      org.opencontainers.image.revision="${LUCID_SHA}" \
      org.opencontainers.image.version="${LUCID_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      ml.cider.photonsim.revision="${PHOTONSIM_SHA}" \
      ml.cider.photonsim.ref="${PHOTONSIM_REF}"

WORKDIR /work
CMD ["bash"]
