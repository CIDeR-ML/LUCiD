#!/bin/bash

# Script to download LUCiD data files from Google Drive
# This downloads the muon simulation data required for running the examples

set -e  # Exit on error

echo "======================================"
echo "LUCiD Data Download Script"
echo "======================================"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "Error: gdown is not installed."
    echo "Please install it using: pip install gdown"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data/water/muon

echo "Downloading muon simulation data..."
echo "This may take several minutes depending on your connection speed."
echo ""

# Download the data
gdown --folder "https://drive.google.com/drive/folders/1zdjj48gYxE7TpzwE7QUGAVskuZpjY6BK" -O data/water/muon/

echo ""
echo "======================================"
echo "Download complete!"
echo "Data saved to: data/water/muon/"
echo "======================================"
