#!/bin/bash
# Setup script for PhotonSim environment
# This script sets up the environment variables needed for building and running PhotonSim

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"

# Source user paths
if [ -f "${USER_PATHS}" ]; then
    source "${USER_PATHS}"
else
    echo "Error: user_paths.sh not found at ${USER_PATHS}"
    echo "Please create this file with your local paths configuration"
    exit 1
fi

echo "Setting up PhotonSim environment..."

# Use paths from user_paths.sh
export GEANT4_DIR="${GEANT4_INSTALL_DIR}"
export ROOT_DIR="${ROOT_INSTALL_DIR}"

# Set Geant4_DIR for CMake
export Geant4_DIR="${GEANT4_DIR}"

# Source GEANT4 environment
if [ -f "${GEANT4_DIR}/geant4make.sh" ]; then
    source "${GEANT4_DIR}/geant4make.sh"
    echo "✓ GEANT4 environment loaded"
else
    echo "⚠ Warning: GEANT4 setup script not found at ${GEANT4_DIR}/geant4make.sh"
fi

# Source ROOT environment
if [ -f "${ROOT_DIR}/bin/thisroot.sh" ]; then
    source "${ROOT_DIR}/bin/thisroot.sh"
    echo "✓ ROOT environment loaded"
else
    echo "⚠ Warning: ROOT setup script not found at ${ROOT_DIR}/bin/thisroot.sh"
fi

# Add to PATH and LD_LIBRARY_PATH
export PATH="${GEANT4_DIR}/bin:${ROOT_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${GEANT4_DIR}/lib64:${ROOT_DIR}/lib:${LD_LIBRARY_PATH}"

echo "Environment setup complete!"
echo "GEANT4_DIR: ${GEANT4_DIR}"
echo "ROOT_DIR: ${ROOT_DIR}"
echo ""
echo "You can now run ./build_photonsim.sh to build PhotonSim"