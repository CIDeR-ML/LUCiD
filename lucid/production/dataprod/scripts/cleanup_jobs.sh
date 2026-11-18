#!/usr/bin/env bash
# Clean up PhotonSim job logs, macros, scripts and (optionally) HDF5 outputs
# under one output directory.
#
# DELETION IS OPT-IN. Without --execute the script only lists what it would
# remove. The previous version defaulted to deleting (DRY_RUN=false with no
# option to set it) while -h claimed dry-run was the default, so a documented
# dry run removed files.

set -euo pipefail

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
DRY_RUN=true
RECURSIVE=false
TOTAL_DELETED=0

usage() {
    cat <<USAGE
Usage: $(basename "$0") -o <output_dir> [-l] [-m] [-s] [-x] [-r] [-a] [-N]
  -o, --output-dir DIR  Output directory to clean (required; must exist)
  -l, --logs            Clean log files (job-*.out, job-*.err, job_*.out, job_*.err)
  -m, --macros          Clean macro files (*.mac)
  -s, --scripts         Clean script files (*.sh, *.sbatch)
  -x, --hdf5            Clean HDF5 files (*.h5) - USE WITH CAUTION!
  -r, --recursive       Recursive - clean all subdirectories
  -a, --all             Clean all temp files (logs, macros, scripts) - DOES NOT include HDF5
  -N, --execute         Actually delete. Without it nothing is removed.
  -h, --help            Show this help.

Examples:
  # List the logs and macros that would go from a single directory
  $(basename "$0") -o /path/to/output -l -m

  # Delete all temp files recursively in a config directory
  $(basename "$0") -o /path/to/config_000001 -r -a --execute

  # Delete everything including HDF5 files recursively
  $(basename "$0") -o /path/to/config_000001 -r -a -x --execute

Dry run is the DEFAULT: nothing is deleted unless -N/--execute is given.
USAGE
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir) OUTPUT_DIR="${2?-o needs a path}"; shift 2 ;;
        -o=*|--output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        -l|--logs)       CLEAN_LOGS=true; shift ;;
        -m|--macros)     CLEAN_MACROS=true; shift ;;
        -s|--scripts)    CLEAN_SCRIPTS=true; shift ;;
        -x|--hdf5)       CLEAN_HDF5=true; shift ;;
        -r|--recursive)  RECURSIVE=true; shift ;;
        -a|--all)        CLEAN_LOGS=true; CLEAN_MACROS=true; CLEAN_SCRIPTS=true; shift ;;
        -N|--execute)    DRY_RUN=false; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Invalid option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Check required parameters. An empty or unset -o is an error, never a default:
# every deletion path below interpolates it.
if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "Error: output directory is required (-o/--output-dir)" >&2
    exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Error: output directory not found: ${OUTPUT_DIR}" >&2
    exit 1
fi

if [[ "$CLEAN_LOGS" == false && "$CLEAN_MACROS" == false \
      && "$CLEAN_SCRIPTS" == false && "$CLEAN_HDF5" == false ]]; then
    echo "Error: nothing selected. Pass at least one of -l/-m/-s/-x/-a." >&2
    usage >&2
    exit 1
fi

# Remove (or list) one pattern in one directory, non-recursively.
# File lists reach rm through -print0 | xargs -0, so a name with a space or a
# newline cannot split into two arguments.
remove_files_in_dir() {
    local dir="$1" pattern="$2" description="$3" count
    count=$(find "$dir" -maxdepth 1 -name "$pattern" -type f -printf '.' 2>/dev/null | wc -c)
    if [[ "$count" -eq 0 ]]; then
        return 0
    fi

    echo "  Found $count $description files"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would delete:"
        find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null \
            | head -5 | sed 's/^/    /'
        if [[ "$count" -gt 5 ]]; then
            echo "    ... and $((count - 5)) more"
        fi
    else
        echo -e "  ${RED}Deleting $count files...${NC}"
        find "$dir" -maxdepth 1 -name "$pattern" -type f -print0 2>/dev/null \
            | xargs -0 --no-run-if-empty rm -f
        TOTAL_DELETED=$((TOTAL_DELETED + count))
        echo -e "  ${GREEN}Done${NC}"
    fi
}

# Function to clean a single directory
clean_directory() {
    local dir="$1"

    echo -e "${GREEN}=== Cleaning: $dir ===${NC}"

    # Show current status
    TOTAL_ROOT=$(find "$dir" -maxdepth 1 -name "*.root" -type f -printf '.' 2>/dev/null | wc -c)
    TOTAL_HDF5=$(find "$dir" -maxdepth 1 -name "*.h5" -type f -printf '.' 2>/dev/null | wc -c)
    TOTAL_LOGS=$(find "$dir" -maxdepth 1 \( -name "job-*.out" -o -name "job-*.err" -o -name "job_*.out" -o -name "job_*.err" \) -type f -printf '.' 2>/dev/null | wc -c)
    TOTAL_MACS=$(find "$dir" -maxdepth 1 -name "*.mac" -type f -printf '.' 2>/dev/null | wc -c)
    TOTAL_SCRIPTS=$(find "$dir" -maxdepth 1 \( -name "*.sh" -o -name "*.sbatch" \) -type f -printf '.' 2>/dev/null | wc -c)

    echo "  ROOT files: $TOTAL_ROOT (will be preserved)"
    echo "  HDF5 files: $TOTAL_HDF5 (will be preserved unless -x specified)"
    echo "  Log files: $TOTAL_LOGS"
    echo "  Macro files: $TOTAL_MACS"
    echo "  Script files: $TOTAL_SCRIPTS"
    echo ""

    # Clean files based on options
    if [[ "$CLEAN_LOGS" == true ]]; then
        remove_files_in_dir "$dir" "job-*.out" "job output log"
        remove_files_in_dir "$dir" "job-*.err" "job error log"
        remove_files_in_dir "$dir" "job_*.out" "job output log"
        remove_files_in_dir "$dir" "job_*.err" "job error log"
    fi

    if [[ "$CLEAN_MACROS" == true ]]; then
        remove_files_in_dir "$dir" "*.mac" "macro"
    fi

    if [[ "$CLEAN_SCRIPTS" == true ]]; then
        remove_files_in_dir "$dir" "*.sh" "shell script"
        remove_files_in_dir "$dir" "*.sbatch" "SLURM batch script"
    fi

    if [[ "$CLEAN_HDF5" == true ]]; then
        echo -e "${RED}WARNING: Cleaning HDF5 files!${NC}"
        remove_files_in_dir "$dir" "*.h5" "HDF5"
    fi

    echo ""
}

# Show header
echo -e "${GREEN}=== PhotonSim Job Cleanup ===${NC}"
echo "Target directory: $OUTPUT_DIR"
echo "Recursive mode: $RECURSIVE"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no files will be deleted${NC}"
    echo "Pass -N/--execute to actually delete files."
    echo ""
fi

if [[ "$CLEAN_HDF5" == true ]]; then
    echo -e "${RED}WARNING: HDF5 cleanup is ENABLED (-x flag)${NC}"
    echo -e "${RED}This will delete processed LUCiD output files!${NC}"
    echo ""
fi

# Main cleanup logic
if [[ "$RECURSIVE" == true ]]; then
    echo "Searching for subdirectories to clean..."
    echo ""

    # Find all directories that contain job files (ROOT, mac, log or HDF5).
    # %h\0 + sort -z keeps a directory name containing a space or a newline in
    # one piece, which `-exec dirname {} \;` piped to `sort -u` did not.
    SUBDIRS=$(find "$OUTPUT_DIR" -type f \
                   \( -name "*.root" -o -name "*.mac" -o -name "job*.out" \
                      -o -name "job*.err" -o -name "*.h5" \) \
                   -printf '%h\0' | sort -z -u | tr '\0' '\n')

    if [[ -z "$SUBDIRS" ]]; then
        echo "No job directories found under $OUTPUT_DIR"
        exit 0
    fi

    DIR_COUNT=$(printf '%s\n' "$SUBDIRS" | wc -l)
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
if [[ "$DRY_RUN" == false ]]; then
    echo -e "${GREEN}=== Cleanup complete: ${TOTAL_DELETED} files deleted ===${NC}"
else
    echo -e "${YELLOW}=== Dry run complete - no files were deleted ===${NC}"
    echo "Pass -N/--execute to actually delete files."
fi
