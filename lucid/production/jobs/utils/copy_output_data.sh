#!/bin/bash

# Script to copy ROOT output files from S3DF storage to local data directory
# Maintains the directory structure for organization

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Source paths
SOURCE_DIR="/sdf/data/neutrino/cjesus/photonsim_output"
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Set destination directory relative to repository root
DEST_DIR="$(dirname $(dirname "$SCRIPT_DIR"))/data"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Copy ROOT output files from S3DF storage to local data directory"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -p, --particle TYPE  Copy only files for specific particle type (e.g., mu-)"
    echo "  -e, --energy ENERGY  Copy only files for specific energy (e.g., 500MeV)"
    echo "  -n, --dry-run        Show what would be copied without actually copying"
    echo "  -v, --verbose        Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                   # Copy all ROOT files"
    echo "  $0 -p mu-            # Copy only muon files"
    echo "  $0 -p mu- -e 500MeV  # Copy only 500MeV muon files"
    echo "  $0 -n                # Dry run to see what would be copied"
}

# Parse command line arguments
PARTICLE_TYPE=""
ENERGY=""
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--particle)
            PARTICLE_TYPE="$2"
            shift 2
            ;;
        -e|--energy)
            ENERGY="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Create destination directory if it doesn't exist (unless dry run)
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$DEST_DIR"
    if [ $? -ne 0 ]; then
        print_error "Failed to create destination directory: $DEST_DIR"
        exit 1
    fi
fi

print_info "Source directory: $SOURCE_DIR"
print_info "Destination directory: $DEST_DIR"

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No files will be copied"
fi

# Build find command based on filters
FIND_CMD="find $SOURCE_DIR"

# Add particle type filter if specified
if [ -n "$PARTICLE_TYPE" ]; then
    FIND_CMD="$FIND_CMD -path */$PARTICLE_TYPE/*"
    print_info "Filtering for particle type: $PARTICLE_TYPE"
fi

# Add energy filter if specified
if [ -n "$ENERGY" ]; then
    FIND_CMD="$FIND_CMD -path */$ENERGY/*"
    print_info "Filtering for energy: $ENERGY"
fi

# Complete the find command
FIND_CMD="$FIND_CMD -name '*.root' -type f"

# Count total files to copy
TOTAL_FILES=$(eval $FIND_CMD | wc -l)

if [ $TOTAL_FILES -eq 0 ]; then
    print_warning "No ROOT files found matching the criteria"
    exit 0
fi

print_info "Found $TOTAL_FILES ROOT file(s) to copy"

# Copy files
COPIED=0
FAILED=0

while IFS= read -r file; do
    # Get relative path from source directory
    REL_PATH="${file#$SOURCE_DIR/}"
    
    # Construct destination path
    DEST_FILE="$DEST_DIR/$REL_PATH"
    DEST_FILE_DIR="$(dirname "$DEST_FILE")"
    
    if [ "$VERBOSE" = true ]; then
        echo "Processing: $REL_PATH"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "Would copy: $file -> $DEST_FILE"
        ((COPIED++))
    else
        # Create destination directory
        mkdir -p "$DEST_FILE_DIR"
        if [ $? -ne 0 ]; then
            print_error "Failed to create directory: $DEST_FILE_DIR"
            ((FAILED++))
            continue
        fi
        
        # Copy file
        cp -p "$file" "$DEST_FILE"
        if [ $? -eq 0 ]; then
            if [ "$VERBOSE" = true ]; then
                print_info "Copied: $REL_PATH"
            fi
            ((COPIED++))
        else
            print_error "Failed to copy: $REL_PATH"
            ((FAILED++))
        fi
    fi
done < <(eval $FIND_CMD)

# Print summary
echo ""
if [ "$DRY_RUN" = true ]; then
    print_info "Dry run complete. Would have copied $COPIED file(s)"
else
    print_info "Copy operation complete"
    print_info "Successfully copied: $COPIED file(s)"
    if [ $FAILED -gt 0 ]; then
        print_error "Failed to copy: $FAILED file(s)"
    fi
    
    # Show disk usage
    if command -v du &> /dev/null; then
        SIZE=$(du -sh "$DEST_DIR" 2>/dev/null | cut -f1)
        print_info "Total size of data directory: $SIZE"
    fi
fi