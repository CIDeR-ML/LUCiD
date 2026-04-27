#!/bin/bash
# User-specific paths configuration for S3DF
# Configured for cjesus on S3DF
#
# All production runs go through one `apptainer exec lucid.sif lucid-run-job
# ...` per SLURM job. The unified container ships GEANT4 + ROOT + GENIE +
# PhotonSim + LUCiD, so the only host-side state is the .sif file and the
# output directory.

# =============================================================================
# Container image
# =============================================================================
# Unified LUCiD container. Pull and convert from ghcr.io with:
#   apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest
export LUCID_IMAGE_PATH="/sdf/data/neutrino/cjesus/software/images/lucid.sif"

# =============================================================================
# GENIE cross-section splines
# =============================================================================
# Path *inside* the container. Points at the G18_10a_02_11b spline bundled in
# the LUCiD image — matches the in-repo dataprod_*.json configs and needs no
# cvmfs at runtime.
export GENIE_XSEC_FILE="/opt/genie_xsec/3_04_00/G18_10a_02_11b/gxspl-min.xml.gz"

# =============================================================================
# Output paths
# =============================================================================
export OUTPUT_BASE_PATH="/sdf/data/neutrino/cjesus/new_photonsim_output"

# =============================================================================
# SLURM configuration
# =============================================================================
export SLURM_PARTITION="roma"
export SLURM_ACCOUNT="mli:cider-ml"

# =============================================================================
# Resource defaults
# =============================================================================
export DEFAULT_CPUS="1"
export DEFAULT_MEMORY="39936"   # MB
export DEFAULT_GPUS="0"
export DEFAULT_TIME="23:00:00"
