#!/bin/bash
# Run script for PhotonSim
# This script sets up the environment and runs PhotonSim with a macro file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PhotonSim Run Script${NC}"
echo "===================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHOTONSIM_DIR="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Check if environment is set up
if [ -z "${GEANT4_DIR}" ] || [ -z "${ROOT_DIR}" ]; then
    echo -e "${YELLOW}Setting up environment...${NC}"
    if [ -f "${SCRIPT_DIR}/setup_environment.sh" ]; then
        source "${SCRIPT_DIR}/setup_environment.sh"
    else
        echo -e "${RED}Error: setup_environment.sh not found!${NC}"
        exit 1
    fi
fi

# Change to PhotonSim directory
cd "${PHOTONSIM_DIR}"

# Check if PhotonSim is built
if [ ! -f "build/PhotonSim" ]; then
    echo -e "${RED}Error: PhotonSim executable not found!${NC}"
    echo "Please run ${SCRIPT_DIR}/build_photonsim.sh first"
    exit 1
fi

# Default macro file
DEFAULT_MACRO="macros/test_muon.mac"

# Use provided macro or default
if [ $# -eq 0 ]; then
    MACRO_FILE="${DEFAULT_MACRO}"
    echo -e "${YELLOW}No macro file specified, using default: ${MACRO_FILE}${NC}"
else
    MACRO_FILE="$1"
fi

# Check if macro file exists
if [ ! -f "${MACRO_FILE}" ]; then
    echo -e "${RED}Error: Macro file not found: ${MACRO_FILE}${NC}"
    echo ""
    echo "Available macro files:"
    ls -1 macros/*.mac 2>/dev/null || echo "No macro files found in macros/"
    exit 1
fi

# Run PhotonSim
echo -e "\n${GREEN}Running PhotonSim with macro: ${MACRO_FILE}${NC}"
echo "=========================================="

cd build
./PhotonSim "../${MACRO_FILE}"

echo -e "\n${GREEN}✓ PhotonSim execution complete${NC}"
echo "Output file: ${PHOTONSIM_DIR}/build/optical_photons.root"