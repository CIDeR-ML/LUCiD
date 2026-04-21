#!/bin/bash
# Check script for PhotonSim dependencies
# This script verifies that all required dependencies are properly installed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PhotonSim Installation Check${NC}"
echo "============================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHOTONSIM_DIR="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Source user paths configuration
USER_PATHS_FILE="${PHOTONSIM_DIR}/s3df_scripts/user_paths.sh"
if [ -f "${USER_PATHS_FILE}" ]; then
    echo -e "${GREEN}✓ Sourcing user paths configuration...${NC}"
    source "${USER_PATHS_FILE}"
else
    echo -e "${YELLOW}⚠ Warning: user_paths.sh not found at ${USER_PATHS_FILE}${NC}"
    echo -e "${YELLOW}  Using default paths - you may need to create this file${NC}"
    # Set default paths as fallback
    export GEANT4_INSTALL_DIR="/sdf/data/neutrino/cjesus/software/builds/geant4"
    export ROOT_INSTALL_DIR="/sdf/data/neutrino/cjesus/software/builds/root"
fi

# Function to check if a command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    fi
}

# Function to check if a directory exists
check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓ $2 found at: $1${NC}"
        return 0
    else
        echo -e "${RED}✗ $2 not found at: $1${NC}"
        return 1
    fi
}

# Function to check if a file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓ $2 found${NC}"
        return 0
    else
        echo -e "${RED}✗ $2 not found at: $1${NC}"
        return 1
    fi
}

echo -e "\n${YELLOW}Checking system dependencies...${NC}"
check_command cmake
check_command make
check_command g++

echo -e "\n${YELLOW}Checking GEANT4 installation...${NC}"
check_directory "${GEANT4_INSTALL_DIR}" "GEANT4 directory"
check_file "${GEANT4_INSTALL_DIR}/Geant4Config.cmake" "Geant4Config.cmake"
check_directory "${GEANT4_INSTALL_DIR}/lib64" "GEANT4 libraries"

echo -e "\n${YELLOW}Checking ROOT installation...${NC}"
check_directory "${ROOT_INSTALL_DIR}" "ROOT directory"
check_file "${ROOT_INSTALL_DIR}/ROOTConfig.cmake" "ROOTConfig.cmake"
check_directory "${ROOT_INSTALL_DIR}/lib" "ROOT libraries"
check_directory "${ROOT_INSTALL_DIR}/include" "ROOT headers"

echo -e "\n${YELLOW}Checking PhotonSim source...${NC}"
check_file "${PHOTONSIM_DIR}/CMakeLists.txt" "CMakeLists.txt"
check_directory "${PHOTONSIM_DIR}/src" "Source directory"
check_directory "${PHOTONSIM_DIR}/include" "Include directory"
check_directory "${PHOTONSIM_DIR}/macros" "Macros directory"

echo -e "\n${YELLOW}Checking environment setup...${NC}"
if [ -z "${GEANT4_INSTALL_DIR}" ] || [ -z "${ROOT_INSTALL_DIR}" ]; then
    echo -e "${YELLOW}Environment not set. Attempting to source setup_environment.sh...${NC}"
    if [ -f "${SCRIPT_DIR}/setup_environment.sh" ]; then
        source "${SCRIPT_DIR}/setup_environment.sh"
        echo -e "${GREEN}✓ Environment loaded${NC}"
    else
        echo -e "${RED}✗ setup_environment.sh not found${NC}"
    fi
else
    echo -e "${GREEN}✓ Environment variables set${NC}"
fi

echo -e "\n${YELLOW}Summary:${NC}"
echo "If all checks passed, you can proceed with:"
echo "  1. source ${SCRIPT_DIR}/setup_environment.sh"
echo "  2. ${SCRIPT_DIR}/build_photonsim.sh"
echo "  3. ${SCRIPT_DIR}/run_photonsim.sh [macro_file]"