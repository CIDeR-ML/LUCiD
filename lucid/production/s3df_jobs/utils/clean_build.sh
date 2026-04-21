#!/bin/bash
# Clean script for PhotonSim
# This script removes the build directory for a fresh build

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}PhotonSim Clean Script${NC}"
echo "======================"

if [ -d "build" ]; then
    echo -e "${YELLOW}Removing build directory...${NC}"
    rm -rf build
    echo -e "${GREEN}✓ Build directory removed${NC}"
else
    echo -e "${GREEN}Build directory doesn't exist - nothing to clean${NC}"
fi

echo ""
echo "You can now run ./build_photonsim.sh for a fresh build"