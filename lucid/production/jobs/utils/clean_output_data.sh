#!/bin/bash

# Script to clean PhotonSim output data directory
# This removes old simulation data to prepare for new runs

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default output directory
OUTPUT_DIR="/sdf/data/neutrino/cjesus/photonsim_output"
DRY_RUN=true
CLEAN_ROOT=false
CLEAN_LOGS=false
CLEAN_ALL=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Clean PhotonSim output data directory"
    echo ""
    echo "Options:"
    echo "  -d, --dir DIR        Output directory to clean (default: $OUTPUT_DIR)"
    echo "  -r, --root           Clean ROOT files (.root)"
    echo "  -l, --logs           Clean log files (.out, .err)"
    echo "  -a, --all            Clean all files (ROOT and logs)"
    echo "  --execute            Actually perform the cleanup (default is dry-run)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                   # Dry run - show what would be cleaned"
    echo "  $0 --all --execute   # Clean all files"
    echo "  $0 -r --execute      # Clean only ROOT files"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--root)
            CLEAN_ROOT=true
            shift
            ;;
        -l|--logs)
            CLEAN_LOGS=true
            shift
            ;;
        -a|--all)
            CLEAN_ALL=true
            CLEAN_ROOT=true
            CLEAN_LOGS=true
            shift
            ;;
        --execute)
            DRY_RUN=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

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

print_blue() {
    echo -e "${BLUE}$1${NC}"
}

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    print_error "Output directory not found: $OUTPUT_DIR"
    exit 1
fi

print_blue "=== PhotonSim Output Data Cleanup ==="
echo "Target directory: $OUTPUT_DIR"
echo ""

# Show current status
print_info "Analyzing current directory contents..."
TOTAL_ROOT=$(find "$OUTPUT_DIR" -name "*.root" -type f 2>/dev/null | wc -l)
TOTAL_LOGS=$(find "$OUTPUT_DIR" -name "job-*.out" -o -name "job-*.err" -type f 2>/dev/null | wc -l)
TOTAL_DIRS=$(find "$OUTPUT_DIR" -mindepth 1 -type d 2>/dev/null | wc -l)

if [ $TOTAL_ROOT -eq 0 ] && [ $TOTAL_LOGS -eq 0 ]; then
    print_info "Directory is already clean - no files to remove"
    exit 0
fi

echo "Current contents:"
echo "  ROOT files: $TOTAL_ROOT"
echo "  Log files (job-*.out, job-*.err): $TOTAL_LOGS"
echo "  Subdirectories: $TOTAL_DIRS"
echo ""

# Calculate total size
if command -v du &> /dev/null; then
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "Total directory size: $TOTAL_SIZE"
    echo ""
fi

# If no cleanup options specified, default to showing what would be cleaned
if [ "$CLEAN_ROOT" = false ] && [ "$CLEAN_LOGS" = false ]; then
    print_warning "No cleanup options specified. Use -r, -l, or -a to specify what to clean."
    echo ""
    echo "What would be cleaned with each option:"
    echo "  -r (ROOT files): $TOTAL_ROOT files"
    echo "  -l (Log files): $TOTAL_LOGS files"
    echo "  -a (All files): $((TOTAL_ROOT + TOTAL_LOGS)) files"
    echo ""
    echo "Add --execute to actually perform the cleanup"
    exit 0
fi

# Dry run warning
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No files will actually be deleted"
    echo "Add --execute flag to perform actual cleanup"
    echo ""
fi

# Function to clean files with pattern
clean_files() {
    local pattern="$1"
    local description="$2"
    
    print_info "Searching for $description..."
    
    # Use find with proper pattern matching
    if [[ "$pattern" == *"job-"* ]]; then
        # For log files, use multiple patterns
        FILES=$(find "$OUTPUT_DIR" -name "job-*.out" -o -name "job-*.err" -type f 2>/dev/null)
    else
        FILES=$(find "$OUTPUT_DIR" -name "$pattern" -type f 2>/dev/null)
    fi
    
    COUNT=$(echo "$FILES" | grep -c . 2>/dev/null || echo 0)
    
    if [ $COUNT -eq 0 ]; then
        echo "  No $description found"
        return
    fi
    
    echo "  Found $COUNT $description"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  Would delete:"
        echo "$FILES" | head -5 | sed 's/^/    /'
        if [ $COUNT -gt 5 ]; then
            echo "    ... and $((COUNT-5)) more"
        fi
    else
        echo "  Deleting $COUNT files..."
        echo "$FILES" | xargs rm -f
        print_info "Deleted $COUNT $description"
    fi
    echo ""
}

# Perform cleanup based on options
if [ "$CLEAN_ROOT" = true ]; then
    clean_files "*.root" "ROOT files"
fi

if [ "$CLEAN_LOGS" = true ]; then
    clean_files "job-*" "log files"
fi

# Clean empty directories if not in dry run mode
if [ "$DRY_RUN" = false ]; then
    print_info "Removing empty directories..."
    # Remove empty directories (but keep the main output directory)
    find "$OUTPUT_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null
    print_info "Cleanup complete!"
    echo ""
    
    # Show final status
    FINAL_ROOT=$(find "$OUTPUT_DIR" -name "*.root" -type f 2>/dev/null | wc -l)
    FINAL_LOGS=$(find "$OUTPUT_DIR" -name "job-*.out" -o -name "job-*.err" -type f 2>/dev/null | wc -l)
    FINAL_DIRS=$(find "$OUTPUT_DIR" -mindepth 1 -type d 2>/dev/null | wc -l)
    
    echo "Final contents:"
    echo "  ROOT files: $FINAL_ROOT"
    echo "  Log files: $FINAL_LOGS"
    echo "  Subdirectories: $FINAL_DIRS"
    
    if command -v du &> /dev/null; then
        FINAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
        echo "Final directory size: $FINAL_SIZE"
    fi
else
    print_info "Dry run complete. Use --execute to perform actual cleanup."
fi