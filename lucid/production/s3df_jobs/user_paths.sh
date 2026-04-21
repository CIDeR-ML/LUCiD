#!/bin/bash
# User-specific paths configuration for S3DF
# Configured for cjesus on S3DF

# =============================================================================
# Software installations (host-side)
# =============================================================================
# GEANT4 and ROOT installation paths — sourced by utils/setup_environment.sh
# so that the PhotonSim binary finds its shared libs when run on bare node.
export GEANT4_INSTALL_DIR="/sdf/data/neutrino/cjesus/software/builds/geant4"
export ROOT_INSTALL_DIR="/sdf/data/neutrino/cjesus/software/builds/root"

# Path to the built PhotonSim binary.
export PHOTONSIM_BIN="/sdf/home/c/cjesus/REFACTORED/PhotonSim/build/PhotonSim"

# Python interpreter for the bare-host step (macro gen + PhotonSim subprocess).
# Must be >= 3.8 (run_job uses `from __future__ import annotations`).
# On S3DF roma nodes `python3` is 3.6 and `python3.9` is absent, so pin to 3.8.
export HOST_PYTHON="python3.8"

# =============================================================================
# LUCiD
# =============================================================================
# LUCiD repo root. Added to PYTHONPATH so jobs can import
# `lucid.production.run_job`.
export LUCID_PATH="/sdf/home/c/cjesus/REFACTORED/LUCiD"

# =============================================================================
# Singularity image (hosts the LUCiD Python env with jax / numpy / h5py)
# =============================================================================
export SINGULARITY_IMAGE_PATH="/sdf/group/neutrino/images/develop.sif"

# =============================================================================
# Output paths
# =============================================================================
# Base directory for all production outputs
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
