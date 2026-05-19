#!/bin/bash
# Script to clean up PhotonSim job outputs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR=""
CLEAN_LOGS=false
CLEAN_MACROS=false
CLEAN_SCRIPTS=false
CLEAN_HDF5=false
DRY_RUN=false
RECURSIVE=false

# Parse command line arguments
while getopts "o:lmsrxah" opt; do
    case $opt in
        o) OUTPUT_DIR="$OPTARG";;
        l) CLEAN_LOGS=true;;
        m) CLEAN_MACROS=true;;
        s) CLEAN_SCRIPTS=true;;
        x) CLEAN_HDF5=true;;
        r) RECURSIVE=true;;
        a) CLEAN_LOGS=true; CLEAN_MACROS=true; CLEAN_SCRIPTS=true;;
        h) echo "Usage: $0 -o <output_dir> [-l] [-m] [-s] [-x] [-r] [-a]"
           echo "  -o: Output directory to clean (required)"
           echo "  -l: Clean log files (job-*.out, job-*.err)"
           echo "  -m: Clean macro files (*.mac)"
           echo "  -s: Clean script files (*.sh, *.sbatch)"
           echo "  -x: Clean HDF5 files (*.h5) - USE WITH CAUTION!"
           echo "  -r: Recursive - clean all subdirectories"
           echo "  -a: Clean all temp files (logs, macros, scripts) - DOES NOT include HDF5"
           echo ""
           echo "Examples:"
           echo "  # Clean logs and macros in a single directory"
           echo "  $0 -o /path/to/output -l -m"
           echo ""
           echo "  # Clean all temp files recursively in a config directory"
           echo "  $0 -o /path/to/config_000001 -r -a"
           echo ""
           echo "  # Clean everything including HDF5 files recursively"
           echo "  $0 -o /path/to/config_000001 -r -a -x"
           echo ""
           echo "By default, this runs in dry-run mode. Files are not actually deleted."
           echo "To actually delete files, edit the script and set DRY_RUN=false"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    esac
done

# Check required parameters
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required (-o option)"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Safety check
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no files will be deleted${NC}"
    echo "To actually delete files, edit the script and set DRY_RUN=false"
    echo ""
fi

# Function to remove files
remove_files() {
    local pattern=$1
    local description=$2
    
    echo -e "${YELLOW}Searching for $description files...${NC}"
    FILES=$(find "$OUTPUT_DIR" -name "$pattern" -type f 2>/dev/null)
    COUNT=$(echo "$FILES" | grep -c . || echo 0)
    
    if [ $COUNT -eq 0 ]; then
        echo "  No $description files found"
        return
    fi
    
    echo "  Found $COUNT $description files"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  Would delete:"
        echo "$FILES" | head -10 | sed 's/^/    /'
        if [ $COUNT -gt 10 ]; then
            echo "    ... and $((COUNT-10)) more"
        fi
    else
        echo -e "  ${RED}Deleting $COUNT files...${NC}"
        echo "$FILES" | xargs rm -f
        echo -e "  ${GREEN}Done${NC}"
    fi
    echo ""
}

# Function to clean a single directory
clean_directory() {
    local dir="$1"

    echo -e "${GREEN}=== Cleaning: $dir ===${NC}"

    # Show current status
    TOTAL_ROOT=$(find "$dir" -maxdepth 1 -name "*.root" -type f 2>/dev/null | wc -l)
    TOTAL_HDF5=$(find "$dir" -maxdepth 1 -name "*.h5" -type f 2>/dev/null | wc -l)
    TOTAL_LOGS=$(find "$dir" -maxdepth 1 \( -name "job-*.out" -o -name "job-*.err" -o -name "job_*.out" -o -name "job_*.err" \) -type f 2>/dev/null | wc -l)
    TOTAL_MACS=$(find "$dir" -maxdepth 1 -name "*.mac" -type f 2>/dev/null | wc -l)
    TOTAL_SCRIPTS=$(find "$dir" -maxdepth 1 \( -name "*.sh" -o -name "*.sbatch" \) -type f 2>/dev/null | wc -l)

    echo "  ROOT files: $TOTAL_ROOT (will be preserved)"
    echo "  HDF5 files: $TOTAL_HDF5 (will be preserved unless -x specified)"
    echo "  Log files: $TOTAL_LOGS"
    echo "  Macro files: $TOTAL_MACS"
    echo "  Script files: $TOTAL_SCRIPTS"
    echo ""

    # Clean files based on options
    if [ "$CLEAN_LOGS" = true ]; then
        remove_files_in_dir "$dir" "job-*.out" "job output log"
        remove_files_in_dir "$dir" "job-*.err" "job error log"
        remove_files_in_dir "$dir" "job_*.out" "job output log"
        remove_files_in_dir "$dir" "job_*.err" "job error log"
    fi

    if [ "$CLEAN_MACROS" = true ]; then
        remove_files_in_dir "$dir" "*.mac" "macro"
    fi

    if [ "$CLEAN_SCRIPTS" = true ]; then
        remove_files_in_dir "$dir" "*.sh" "shell script"
        remove_files_in_dir "$dir" "*.sbatch" "SLURM batch script"
    fi

    if [ "$CLEAN_HDF5" = true ]; then
        echo -e "${RED}WARNING: Cleaning HDF5 files!${NC}"
        remove_files_in_dir "$dir" "*.h5" "HDF5"
    fi

    echo ""
}

# Modified remove_files function for specific directory
remove_files_in_dir() {
    local dir=$1
    local pattern=$2
    local description=$3

    FILES=$(find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null)
    COUNT=$(echo "$FILES" | grep -c '^' || echo 0)

    if [ $COUNT -eq 0 ]; then
        return
    fi

    echo "  Found $COUNT $description files"

    if [ "$DRY_RUN" = true ]; then
        echo "  Would delete:"
        echo "$FILES" | head -5 | sed 's/^/    /'
        if [ $COUNT -gt 5 ]; then
            echo "    ... and $((COUNT-5)) more"
        fi
    else
        echo -e "  ${RED}Deleting $COUNT files...${NC}"
        echo "$FILES" | xargs rm -f
        echo -e "  ${GREEN}Done${NC}"
    fi
}

# Show header
echo -e "${GREEN}=== PhotonSim Job Cleanup ===${NC}"
echo "Target directory: $OUTPUT_DIR"
echo "Recursive mode: $RECURSIVE"
echo ""

# Safety check
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no files will be deleted${NC}"
    echo "To actually delete files, edit the script and set DRY_RUN=false"
    echo ""
fi

if [ "$CLEAN_HDF5" = true ]; then
    echo -e "${RED}WARNING: HDF5 cleanup is ENABLED (-x flag)${NC}"
    echo -e "${RED}This will delete processed LUCiD output files!${NC}"
    echo ""
fi

# Main cleanup logic
if [ "$RECURSIVE" = true ]; then
    echo "Searching for subdirectories to clean..."
    echo ""

    # Find all directories that contain job files (ROOT, mac, or log files)
    SUBDIRS=$(find "$OUTPUT_DIR" -type f \( -name "*.root" -o -name "*.mac" -o -name "job*.out" -o -name "job*.err" -o -name "*.h5" \) -exec dirname {} \; | sort -u)

    if [ -z "$SUBDIRS" ]; then
        echo "No job directories found under $OUTPUT_DIR"
        exit 0
    fi

    DIR_COUNT=$(echo "$SUBDIRS" | wc -l)
    echo "Found $DIR_COUNT directories with job files"
    echo ""

    # Clean each subdirectory
    CURRENT=0
    while IFS= read -r subdir; do
        CURRENT=$((CURRENT + 1))
        echo "[$CURRENT/$DIR_COUNT] Processing: $subdir"
        clean_directory "$subdir"
    done <<< "$SUBDIRS"

else
    # Single directory cleanup
    clean_directory "$OUTPUT_DIR"
fi

# Summary
if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}=== Cleanup complete! ===${NC}"
else
    echo -e "${YELLOW}=== Dry run complete - no files were deleted ===${NC}"
    echo "Edit the script and set DRY_RUN=false to actually delete files"
fi