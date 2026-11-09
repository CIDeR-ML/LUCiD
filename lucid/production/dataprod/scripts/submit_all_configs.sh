#!/bin/bash
# Script to submit multiple JSON configurations
# Usage: ./submit_all_configs.sh -D detector [-p pattern] [-s] [-t] [-d] [-g] [-P partition] [-o output_base]
# Per-config job counts come from each config (nominal_*, seconds_per_event); to change one, run
# the fan-out on that config with -N (--n-jobs-override).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Bundled dataprod configs live under LUCiD/lucid/production/dataprod/configs/<block>/
# (block = GeV, Solar, SN, Test), one NN_name.json per config.
CONFIG_DIR="${SCRIPT_DIR}/../configs"
FANOUT=("${SCRIPT_DIR}/../../platform/run_in_container.sh" lucid.production.dataprod.fanout)

# Default values. Configs live one level down in per-block folders. -p is a
# path glob matched against <block>/<file>: default "*.json" = all blocks;
# "GeV/*.json" = one block; "GeV/0[1-6]_*.json" = a subset.
PATTERN="*.json"
SUBMIT_JOBS=false
TEST_MODE=false
DRY_RUN=false
USE_GPU=false
PARTITION_OVERRIDE=""
OUTPUT_OVERRIDE=""
DETECTOR_OVERRIDE=""

# Parse command line arguments
while getopts "p:stdgP:o:D:h" opt; do
    case $opt in
        p) PATTERN="$OPTARG";;
        s) SUBMIT_JOBS=true;;
        t) TEST_MODE=true;;
        d) DRY_RUN=true;;
        g) USE_GPU=true;;
        P) PARTITION_OVERRIDE="$OPTARG";;
        o) OUTPUT_OVERRIDE="$OPTARG";;
        D) DETECTOR_OVERRIDE="$OPTARG";;
        h) echo "Usage: $0 -D detector [-p pattern] [-s] [-t] [-d] [-g] [-P partition] [-o output_base]"
           echo "  -p: Path glob under configs/ (default: *.json = all; e.g. GeV/*.json)"
           echo "  -s: Submit jobs to SLURM (default: prepare only)"
           echo "  -t: Test mode - create only one job per config"
           echo "  -d: Dry run - show what would be submitted without doing it"
           echo "  -g: Enable GPU mode (request 1 GPU per job)"
           echo "  -P: SLURM partition override"
           echo "  -o: Output base path override"
           echo "  -D: Detector geometry (required unless a config declares its own)"
           echo ""
           echo "Examples:"
           echo "  # Dry run to see all configs"
           echo "  $0 -d"
           echo ""
           echo "  # Submit all configs, GPU mode"
           echo "  $0 -D SK_WAND -g -P ampere -s"
           echo ""
           echo "  # Test all configs (1 job each, no submit)"
           echo "  $0 -D SK_WAND -t"
           echo ""
           echo "  # Submit to specific output directory"
           echo "  $0 -D SK_WAND -P roma -o /path/to/output -s"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    esac
done

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Find matching config files
mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -mindepth 2 -maxdepth 2 -type f -path "*/$PATTERN" | sort)

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "No configuration files found matching pattern: $PATTERN"
    echo "Looking in: $CONFIG_DIR"
    exit 1
fi

echo "=== Submit All Configurations ==="
echo "Config directory: $CONFIG_DIR"
echo "Pattern: $PATTERN"
echo "Found ${#CONFIG_FILES[@]} configuration file(s)"
echo ""

# Show override settings
if [ "$USE_GPU" = true ]; then
    echo "GPU mode: enabled"
fi
if [ -n "$PARTITION_OVERRIDE" ]; then
    echo "Partition override: $PARTITION_OVERRIDE"
fi
if [ -n "$OUTPUT_OVERRIDE" ]; then
    echo "Output base override: $OUTPUT_OVERRIDE"
fi
echo ""

# Show what will be processed
echo "Configurations to process:"
for config_file in "${CONFIG_FILES[@]}"; do
    config_name=$(basename "$config_file")
    config_desc=$(jq -r '.name' "$config_file" 2>/dev/null || echo "N/A")
    echo "  - $config_name ($config_desc)"
done
echo ""

# Build command arguments for the fan-out
CMD_ARGS=""
if [ "$SUBMIT_JOBS" = true ]; then
    CMD_ARGS="$CMD_ARGS -s"
    echo "Mode: Generate and SUBMIT jobs to SLURM"
else
    echo "Mode: Generate jobs only (no submission)"
fi

if [ "$TEST_MODE" = true ]; then
    CMD_ARGS="$CMD_ARGS -t"
    echo "Test mode: Only 1 job per configuration"
fi

if [ "$USE_GPU" = true ]; then
    CMD_ARGS="$CMD_ARGS -g"
fi

if [ -n "$PARTITION_OVERRIDE" ]; then
    CMD_ARGS="$CMD_ARGS -P $PARTITION_OVERRIDE"
fi

if [ -n "$OUTPUT_OVERRIDE" ]; then
    CMD_ARGS="$CMD_ARGS -o $OUTPUT_OVERRIDE"
fi

if [ -n "$DETECTOR_OVERRIDE" ]; then
    CMD_ARGS="$CMD_ARGS -D $DETECTOR_OVERRIDE"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - Would execute the following:"
    for config_file in "${CONFIG_FILES[@]}"; do
        echo "  ${FANOUT[*]} -c $config_file $CMD_ARGS"
    done
    echo ""
    echo "Run without -d flag to actually execute"
    exit 0
fi

# Confirm before proceeding if submitting
if [ "$SUBMIT_JOBS" = true ]; then
    echo "WARNING: This will submit jobs to SLURM!"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Process each config file
SUCCESS_COUNT=0
FAIL_COUNT=0

for config_file in "${CONFIG_FILES[@]}"; do
    config_name=$(basename "$config_file")

    echo "=========================================="
    echo "Processing: $config_name"
    echo "=========================================="

    if "${FANOUT[@]}" -c "$config_file" $CMD_ARGS; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ Success: $config_name"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ Failed: $config_name"
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "=== Summary ==="
echo "=========================================="
echo "Total configurations: ${#CONFIG_FILES[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ "$SUBMIT_JOBS" = true ]; then
    echo "Jobs have been submitted to SLURM"
    echo "Monitor with: ${SCRIPT_DIR}/monitor_jobs.<slurm|htcondor>.sh -w"
else
    echo "Jobs have been prepared but not submitted"
    echo "To submit individual configs, run:"
    echo "  ${FANOUT[*]} -c <config_file> -s"
fi
echo ""
