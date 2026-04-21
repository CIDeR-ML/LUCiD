#!/bin/bash
# Build script for PhotonSim
# This script configures and builds PhotonSim using CMake

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PhotonSim Build Script${NC}"
echo "========================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHOTONSIM_DIR="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Check if environment is set up
if [ -z "${GEANT4_DIR}" ] || [ -z "${ROOT_DIR}" ]; then
    echo -e "${YELLOW}Environment not set up. Sourcing setup_environment.sh...${NC}"
    if [ -f "${SCRIPT_DIR}/setup_environment.sh" ]; then
        source "${SCRIPT_DIR}/setup_environment.sh"
    else
        echo -e "${RED}Error: setup_environment.sh not found!${NC}"
        echo "Please run this script from the PhotonSim directory"
        exit 1
    fi
fi

# Change to PhotonSim directory
cd "${PHOTONSIM_DIR}"

# Create build directory
echo -e "\n${GREEN}Creating build directory...${NC}"
mkdir -p build
cd build

# Configure with CMake
echo -e "\n${GREEN}Configuring with CMake...${NC}"
cmake -DGeant4_DIR="${GEANT4_DIR}" \
      -DROOT_DIR="${ROOT_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      ..

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuration successful${NC}"
else
    echo -e "${RED}✗ Configuration failed${NC}"
    exit 1
fi

# Build
echo -e "\n${GREEN}Building PhotonSim...${NC}"
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}PhotonSim executable created at: $(pwd)/PhotonSim${NC}"
    echo ""
    echo "To run PhotonSim:"
    echo "  cd ${PHOTONSIM_DIR}/build"
    echo "  ./PhotonSim ../macros/test_fixed_energy.mac"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi