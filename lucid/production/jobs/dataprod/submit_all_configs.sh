#!/bin/bash
# Script to submit multiple JSON configurations
# Usage: ./submit_all_configs.sh [-p pattern] [-s] [-t] [-d] [-n n_jobs] [-e events] [-g] [-P partition]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Bundled dataprod configs live under LUCiD/lucid/production/configs/<block>/
# (block = GeV, Solar, SN, Test), one NN_name.json per config.
CONFIG_DIR="${SCRIPT_DIR}/../../configs"

# Default values. Configs live one level down in per-block folders. -p is a
# path glob matched against <block>/<file>: default "*.json" = all blocks;
# "GeV/*.json" = one block; "GeV/0[1-6]_*.json" = a subset.
PATTERN="*.json"
SUBMIT_JOBS=false
TEST_MODE=false
DRY_RUN=false
N_JOBS_OVERRIDE=""
N_EVENTS_OVERRIDE=""
USE_GPU=false
PARTITION_OVERRIDE=""
OUTPUT_OVERRIDE=""
DETECTOR_OVERRIDE=""

# Parse command line arguments
while getopts "p:stdn:e:gP:o:D:h" opt; do
    case $opt in
        p) PATTERN="$OPTARG";;
        s) SUBMIT_JOBS=true;;
        t) TEST_MODE=true;;
        d) DRY_RUN=true;;
        n) N_JOBS_OVERRIDE="$OPTARG";;
        e) N_EVENTS_OVERRIDE="$OPTARG";;
        g) USE_GPU=true;;
        P) PARTITION_OVERRIDE="$OPTARG";;
        o) OUTPUT_OVERRIDE="$OPTARG";;
        D) DETECTOR_OVERRIDE="$OPTARG";;
        h) echo "Usage: $0 [-p pattern] [-s] [-t] [-d] [-n n_jobs] [-e events] [-g] [-P partition] [-o output_base] [-D detector]"
           echo "  -p: Path glob under configs/ (default: *.json = all; e.g. GeV/*.json)"
           echo "  -s: Submit jobs to SLURM (default: prepare only)"
           echo "  -t: Test mode - create only one job per config"
           echo "  -d: Dry run - show what would be submitted without doing it"
           echo "  -n: Override number of jobs per config"
           echo "  -e: Override events per job"
           echo "  -g: Enable GPU mode (request 1 GPU per job)"
           echo "  -P: SLURM partition override"
           echo "  -o: Output base path override"
           echo "  -D: Detector geometry (default: SK_like)"
           echo ""
           echo "Examples:"
           echo "  # Dry run to see all configs"
           echo "  $0 -d"
           echo ""
           echo "  # Submit all configs with 2 jobs, 100 events, GPU mode"
           echo "  $0 -n 2 -e 100 -g -P ampere -s"
           echo ""
           echo "  # Test all configs (1 job each, no submit)"
           echo "  $0 -t"
           echo ""
           echo "  # Submit to specific output directory"
           echo "  $0 -n 10 -e 100 -P roma -o /path/to/output -s"
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
if [ -n "$N_JOBS_OVERRIDE" ]; then
    echo "Jobs per config override: $N_JOBS_OVERRIDE"
fi
if [ -n "$N_EVENTS_OVERRIDE" ]; then
    echo "Events per job override: $N_EVENTS_OVERRIDE"
fi
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

# Build command arguments for generate_jobs.sh
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

# Override configs go on shared FS and persist past sbatch return; see docs/guides/production/deploy-s3df.md.
TEMP_DIR=""
if [ -n "$N_JOBS_OVERRIDE" ] || [ -n "$N_EVENTS_OVERRIDE" ]; then
    BASE_FOR_TEMP="${OUTPUT_OVERRIDE:-$OUTPUT_BASE_PATH}"
    mkdir -p "$BASE_FOR_TEMP" 2>/dev/null || true
    TEMP_DIR=$(mktemp -d -p "$BASE_FOR_TEMP" overridden_configs.XXXXXX 2>/dev/null \
                  || mktemp -d "$HOME/.lucid_overridden_configs.XXXXXX")
    echo "Override configs: $TEMP_DIR (persistent — remove after jobs finish)"
fi

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - Would execute the following:"
    for config_file in "${CONFIG_FILES[@]}"; do
        config_to_use="$config_file"
        if [ -n "$TEMP_DIR" ]; then
            config_to_use="${TEMP_DIR}/$(basename "$config_file")"
        fi
        echo "  ${SCRIPT_DIR}/generate_jobs.sh -c $config_to_use $CMD_ARGS"
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

    # Determine which config file to use
    config_to_use="$config_file"

    # Create modified config if overrides are specified
    if [ -n "$TEMP_DIR" ]; then
        temp_config="${TEMP_DIR}/${config_name}"

        # Start with original config
        cp "$config_file" "$temp_config"

        # Apply n_jobs override
        if [ -n "$N_JOBS_OVERRIDE" ]; then
            jq --argjson njobs "$N_JOBS_OVERRIDE" '.n_jobs = $njobs' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
        fi

        # Apply n_events_per_job override
        if [ -n "$N_EVENTS_OVERRIDE" ]; then
            jq --argjson nevents "$N_EVENTS_OVERRIDE" '.n_events_per_job = $nevents' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
        fi

        # If the user is forcing flat n_jobs/n_events via -n/-e, drop any
        # events_schedule block so generate_jobs.sh doesn't override our
        # override with time-based planning.
        if [ -n "$N_JOBS_OVERRIDE" ] || [ -n "$N_EVENTS_OVERRIDE" ]; then
            jq 'del(.events_schedule)' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
        fi

        config_to_use="$temp_config"
    fi

    # Execute generate_jobs.sh
    if ${SCRIPT_DIR}/generate_jobs.sh -c "$config_to_use" $CMD_ARGS; then
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
    echo "Monitor with: ${SCRIPT_DIR}/monitor_jobs.sh -w"
else
    echo "Jobs have been prepared but not submitted"
    echo "To submit individual configs, run:"
    echo "  ${SCRIPT_DIR}/generate_jobs.sh -c <config_file> -s"
fi
echo ""
