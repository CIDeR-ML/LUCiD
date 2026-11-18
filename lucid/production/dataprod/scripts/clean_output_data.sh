#!/usr/bin/env bash
#
# Remove ROOT and/or log files from a finished PhotonSim output directory.
#
# The directory is a REQUIRED argument. It used to default to one developer's
# absolute path (/sdf/data/neutrino/cjesus/photonsim_output), so running the
# script with no arguments on a shared filesystem pointed it at somebody
# else's data.
#
# Deletion is opt-in: without --execute nothing is removed.

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

OUTPUT_DIR=""
DRY_RUN=true
CLEAN_ROOT=false
CLEAN_LOGS=false

usage() {
    cat <<USAGE
Usage: $(basename "$0") -d DIR [-r] [-l] [-a] [--execute]
Remove ROOT and/or log files from a PhotonSim output directory.

Options:
  -d, --dir DIR        Output directory to clean (REQUIRED; must exist)
  -r, --root           Clean ROOT files (*.root)
  -l, --logs           Clean log files (job-*.out, job-*.err)
  -a, --all            Clean all files (ROOT and logs)
      --execute        Actually perform the cleanup (default is dry-run)
  -h, --help           Show this help message

Examples:
  $(basename "$0") -d /path/to/output -a              # Dry run - list what would go
  $(basename "$0") -d /path/to/output -a --execute    # Delete ROOT files and logs
  $(basename "$0") -d /path/to/output -r --execute    # Delete only ROOT files

There is no default directory and no default selection: -d and at least one of
-r/-l/-a are required.
USAGE
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)   OUTPUT_DIR="${2?-d needs a path}"; shift 2 ;;
        -d=*|--dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        -r|--root)  CLEAN_ROOT=true; shift ;;
        -l|--logs)  CLEAN_LOGS=true; shift ;;
        -a|--all)   CLEAN_ROOT=true; CLEAN_LOGS=true; shift ;;
        --execute)  DRY_RUN=false; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

print_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }
print_blue()    { echo -e "${BLUE}$1${NC}"; }

# An empty or unset directory is an error, never a default: every deletion
# path below interpolates it.
if [[ -z "${OUTPUT_DIR}" ]]; then
    print_error "an output directory is required (-d/--dir)"
    usage >&2
    exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    print_error "Output directory not found: ${OUTPUT_DIR}"
    exit 1
fi

if [[ "$CLEAN_ROOT" == false && "$CLEAN_LOGS" == false ]]; then
    print_error "nothing selected. Pass at least one of -r/-l/-a."
    usage >&2
    exit 1
fi

# Count matches for one selection. `find -printf '.' | wc -c` counts without
# building a list, and never yields the two-line "0" that
# `grep -c … || echo 0` produced (which then aborted `[ $COUNT -eq 0 ]` with
# "too many arguments").
count_root() {
    find "${OUTPUT_DIR}" -type f -name "*.root" -printf '.' 2>/dev/null | wc -c
}

# -type f applies to BOTH log patterns. Written as
# `-name "job-*.out" -o -name "job-*.err" -type f` the -type f bound only to
# the second alternative, so every *directory* named job-*.out was counted
# too, and every log that was not a *.err was matched without a type test.
count_logs() {
    find "${OUTPUT_DIR}" -type f \
         \( -name "job-*.out" -o -name "job-*.err" \) -printf '.' 2>/dev/null | wc -c
}

print_blue "=== PhotonSim Output Data Cleanup ==="
echo "Target directory: ${OUTPUT_DIR}"
echo ""

print_info "Analyzing current directory contents..."
TOTAL_ROOT=$(count_root)
TOTAL_LOGS=$(count_logs)
TOTAL_DIRS=$(find "${OUTPUT_DIR}" -mindepth 1 -type d -printf '.' 2>/dev/null | wc -c)

if [[ "$TOTAL_ROOT" -eq 0 && "$TOTAL_LOGS" -eq 0 ]]; then
    print_info "Directory is already clean - no files to remove"
    exit 0
fi

echo "Current contents:"
echo "  ROOT files: ${TOTAL_ROOT}"
echo "  Log files (job-*.out, job-*.err): ${TOTAL_LOGS}"
echo "  Subdirectories: ${TOTAL_DIRS}"
echo ""

if command -v du >/dev/null 2>&1; then
    TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)
    echo "Total directory size: ${TOTAL_SIZE}"
    echo ""
fi

if [[ "$DRY_RUN" == true ]]; then
    print_warning "DRY RUN MODE - No files will actually be deleted"
    echo "Add --execute to perform the actual cleanup"
    echo ""
fi

# Remove (or list) one selection. The file list reaches rm through
# -print0 | xargs -0, so a name with a space or a newline cannot split into
# two arguments.
clean_selection() {
    local what="$1" description="$2" count
    if [[ "$what" == root ]]; then count=$(count_root); else count=$(count_logs); fi

    print_info "Searching for ${description}..."
    if [[ "$count" -eq 0 ]]; then
        echo "  No ${description} found"
        echo ""
        return 0
    fi

    echo "  Found ${count} ${description}"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would delete:"
        if [[ "$what" == root ]]; then
            find "${OUTPUT_DIR}" -type f -name "*.root" 2>/dev/null | head -5 | sed 's/^/    /'
        else
            find "${OUTPUT_DIR}" -type f \( -name "job-*.out" -o -name "job-*.err" \) \
                 2>/dev/null | head -5 | sed 's/^/    /'
        fi
        if [[ "$count" -gt 5 ]]; then
            echo "    ... and $((count - 5)) more"
        fi
    else
        echo "  Deleting ${count} files..."
        if [[ "$what" == root ]]; then
            find "${OUTPUT_DIR}" -type f -name "*.root" -print0 2>/dev/null \
                | xargs -0 --no-run-if-empty rm -f
        else
            find "${OUTPUT_DIR}" -type f \( -name "job-*.out" -o -name "job-*.err" \) \
                 -print0 2>/dev/null | xargs -0 --no-run-if-empty rm -f
        fi
        print_info "Deleted ${count} ${description}"
    fi
    echo ""
}

if [[ "$CLEAN_ROOT" == true ]]; then
    clean_selection root "ROOT files"
fi

if [[ "$CLEAN_LOGS" == true ]]; then
    clean_selection logs "log files"
fi

if [[ "$DRY_RUN" == false ]]; then
    print_info "Removing empty directories..."
    # -mindepth 1 keeps OUTPUT_DIR itself. -delete only ever sees directories
    # that are already empty.
    find "${OUTPUT_DIR}" -mindepth 1 -depth -type d -empty -delete 2>/dev/null || true
    print_info "Cleanup complete!"
    echo ""

    echo "Final contents:"
    echo "  ROOT files: $(count_root)"
    echo "  Log files: $(count_logs)"
    echo "  Subdirectories: $(find "${OUTPUT_DIR}" -mindepth 1 -type d -printf '.' 2>/dev/null | wc -c)"

    if command -v du >/dev/null 2>&1; then
        echo "Final directory size: $(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)"
    fi
else
    print_info "Dry run complete. Use --execute to perform actual cleanup."
fi
