#!/bin/bash
# Generate validation HTML visualizations from a dataprod configuration
#
# Usage: ./generate_validation_htmls.sh [OPTIONS]
#
# Options:
#   -c, --config NUM     Dataprod config number (default: 07)
#   -n, --n-events NUM   Number of events to generate (default: 5)
#   -o, --output PATH    Output directory for HTML files (default: validation_html/track_segments_test)
#   -h, --help           Show this help message
#
# Example:
#   ./generate_validation_htmls.sh -c 07 -n 5
#   ./generate_validation_htmls.sh --config 10 --n-events 10 --output /path/to/output

set -e

# Default values
CONFIG_NUM="07"
N_EVENTS=5
OUTPUT_DIR=""

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUCID_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHOTONSIM_DIR="$(cd "${LUCID_DIR}/../PhotonSim" && pwd)"
DATAPROD_CONFIG_DIR="${PHOTONSIM_DIR}/macros/data_production_config"
TMP_DIR="/sdf/data/neutrino/cjesus/tmp/validation_generation"
SINGULARITY_IMAGE="/sdf/data/neutrino/cjesus/software/images/lucid.sif"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_NUM="$2"
            shift 2
            ;;
        -n|--n-events)
            N_EVENTS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${LUCID_DIR}/validation_html/track_segments_test"
fi

# Append config subfolder to output directory
OUTPUT_DIR="${OUTPUT_DIR}/config${CONFIG_NUM}"

# Find the config file
CONFIG_FILE=$(ls ${DATAPROD_CONFIG_DIR}/dataprod_${CONFIG_NUM}*.json 2>/dev/null | head -1)
if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found for config number ${CONFIG_NUM}"
    echo "Available configs:"
    ls ${DATAPROD_CONFIG_DIR}/dataprod_*.json | xargs -n1 basename
    exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE" .json)

echo "========================================================================"
echo "VALIDATION HTML GENERATION"
echo "========================================================================"
echo "Config file: $CONFIG_FILE"
echo "Config name: $CONFIG_NAME"
echo "Number of events: $N_EVENTS"
echo "Output directory: $OUTPUT_DIR"
echo "Temp directory: $TMP_DIR"
echo ""

# Create directories
mkdir -p "$TMP_DIR"
mkdir -p "$OUTPUT_DIR"

# Parse config file to get particle info
echo "Parsing configuration..."
N_PARTICLES=$(jq '.particles | length' "$CONFIG_FILE")
echo "  Particles per event: $N_PARTICLES"

# Build particle commands for macro
PARTICLE_COMMANDS=""
for (( p=0; p<$N_PARTICLES; p++ )); do
    PARTICLE_TYPE=$(jq -r ".particles[$p].type" "$CONFIG_FILE")
    ENERGY_MIN=$(jq -r ".particles[$p].energy_min_MeV" "$CONFIG_FILE")
    ENERGY_MAX=$(jq -r ".particles[$p].energy_max_MeV" "$CONFIG_FILE")
    echo "  - $PARTICLE_TYPE: ${ENERGY_MIN}-${ENERGY_MAX} MeV"
    PARTICLE_COMMANDS="${PARTICLE_COMMANDS}/gun/addPrimaryWithEnergyRange ${PARTICLE_TYPE} ${ENERGY_MIN} ${ENERGY_MAX} MeV
"
done
echo ""

# Create macro file
MACRO_FILE="${TMP_DIR}/${CONFIG_NAME}_validation.mac"
ROOT_FILE="${TMP_DIR}/${CONFIG_NAME}_validation.root"

echo "Creating macro file: $MACRO_FILE"
cat > "$MACRO_FILE" << EOF
# Validation macro generated from ${CONFIG_NAME}
/output/filename ${ROOT_FILE}
/run/initialize
/photon/storeIndividual true

# Particles
${PARTICLE_COMMANDS}
/gun/randomDirection true
/gun/position 0 0 0 m

# Run events
/run/beamOn ${N_EVENTS}
EOF

echo ""
echo "========================================================================"
echo "STEP 1: Running PhotonSim"
echo "========================================================================"
cd "$TMP_DIR"
${PHOTONSIM_DIR}/build/PhotonSim "$MACRO_FILE"

if [ ! -f "$ROOT_FILE" ]; then
    echo "Error: PhotonSim failed to create ROOT file"
    exit 1
fi
echo "ROOT file created: $ROOT_FILE"
ls -lh "$ROOT_FILE"

echo ""
echo "========================================================================"
echo "STEP 2: Running LUCiD processing (v3 four-file output)"
echo "========================================================================"
DATASET_ROOT="${TMP_DIR}/${CONFIG_NAME}_validation"
mkdir -p "$DATASET_ROOT"

singularity exec -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE} python \
    ${LUCID_DIR}/lucid/production/generate_events_with_particles.py \
    --root-file "$ROOT_FILE" \
    --output "$DATASET_ROOT" \
    --dataset-name "${CONFIG_NAME}_validation" \
    --apply-smearing \
    --apply-translation \
    --n-events "$N_EVENTS" \
    --batch-size "$N_EVENTS"

SENSOR_FILE="${DATASET_ROOT}/sensor/wc_sensor_0000.h5"
if [ ! -f "$SENSOR_FILE" ]; then
    echo "Error: LUCiD failed to create v3 batch file (expected $SENSOR_FILE)"
    exit 1
fi
echo "v3 dataset created under: $DATASET_ROOT"
ls -lh "${DATASET_ROOT}/"*/*.h5

echo ""
echo "========================================================================"
echo "STEP 3: Generating HTML visualizations"
echo "========================================================================"
cd "$LUCID_DIR"

for (( event=0; event<$N_EVENTS; event++ )); do
    echo "Generating visualization for event $event..."
    singularity exec -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE} python \
        lucid/production/visualize_particle_events.py \
        "$DATASET_ROOT" \
        config/SK_like_geom_config.json \
        --event "$event" \
        --file-index 0 \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "========================================================================"
echo "COMPLETE"
echo "========================================================================"
echo "Generated HTML files:"
ls -lh ${OUTPUT_DIR}/*.html | tail -${N_EVENTS}
echo ""
echo "Output directory: $OUTPUT_DIR"
