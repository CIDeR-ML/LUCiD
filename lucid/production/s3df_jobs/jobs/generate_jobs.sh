#!/bin/bash
# generate_jobs.sh — fan out a dataprod JSON config into SLURM sbatch scripts
# that invoke `lucid-run-job` (macro + PhotonSim on the bare compute node, then
# LUCiD v3 writer inside the singularity image with jax/numpy/h5py).
#
# Usage: ./generate_jobs.sh -c <config_json> [-s] [-t] [-g] [-P partition] [-o output_base]

set -eu -o pipefail

# --- Locate ourselves --------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UTILS_DIR="${SCRIPT_DIR}/../utils"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"

if [ -f "${USER_PATHS}" ]; then
    # shellcheck source=/dev/null
    source "${USER_PATHS}"
else
    echo "Error: user_paths.sh not found at ${USER_PATHS}" >&2
    echo "Copy user_paths.sh.template and configure it." >&2
    exit 1
fi

# --- Defaults & CLI ----------------------------------------------------------
CONFIG_FILE=""
SUBMIT_JOBS=false
TEST_MODE=false
USE_GPU=false
PARTITION_OVERRIDE=""
OUTPUT_OVERRIDE=""

while getopts "c:stgP:o:h" opt; do
    case $opt in
        c) CONFIG_FILE="$OPTARG";;
        s) SUBMIT_JOBS=true;;
        t) TEST_MODE=true;;
        g) USE_GPU=true;;
        P) PARTITION_OVERRIDE="$OPTARG";;
        o) OUTPUT_OVERRIDE="$OPTARG";;
        h) cat <<EOF
Usage: $0 -c <config_json> [-s] [-t] [-g] [-P partition] [-o output_base]
  -c  Path to dataprod JSON config (required)
  -s  Submit jobs to SLURM (default: prepare only)
  -t  Test mode — create only one job (and pass --test to lucid-run-job)
  -g  Enable GPU mode (request 1 GPU per job)
  -P  SLURM partition override
  -o  OUTPUT_BASE_PATH override

See lucid/production/configs/ for example JSON configs.
EOF
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    esac
done

[ -n "$CONFIG_FILE" ]    || { echo "Error: -c <config_json> is required" >&2; exit 1; }
[ -f "$CONFIG_FILE" ]    || { echo "Error: config file not found: $CONFIG_FILE" >&2; exit 1; }
command -v jq >/dev/null || { echo "Error: jq is required" >&2; exit 1; }

echo "=== lucid-run-job SLURM fan-out ==="
echo ""

# --- Parse JSON config --------------------------------------------------------
CONFIG_NUMBER=$(jq -r '.config_number' "$CONFIG_FILE")
CONFIG_NAME=$(jq -r '.name' "$CONFIG_FILE")
# Filesystem/SLURM-safe slug (e.g. "mu- + pi+" -> "muminus_plus_piplus")
CONFIG_NAME_SLUG=$(echo "$CONFIG_NAME" | sed 's/+/plus/g; s/-/minus/g; s/ /_/g; s/__*/_/g')
CONFIG_DESC=$(jq -r '.description // ""' "$CONFIG_FILE")
MATERIAL=$(jq -r '.material' "$CONFIG_FILE")
OUTPUT_PATH=$(jq -r '.output_path' "$CONFIG_FILE")
PRIMARY_SOURCE=$(jq -r '.primary_source // "particle_gun"' "$CONFIG_FILE")
ENERGY_DIST=$(jq -r '.energy_distribution // "uniform"' "$CONFIG_FILE")
RUN_LUCID=$(jq -r '.run_lucid // true' "$CONFIG_FILE")
N_JOBS=$(jq -r '.n_jobs' "$CONFIG_FILE")
N_EVENTS=$(jq -r '.n_events_per_job' "$CONFIG_FILE")
N_PARTICLES=$(jq '.particles // [] | length' "$CONFIG_FILE")
USE_CONFIG_NUMBER=$(jq -r 'if has("use_config_number") then .use_config_number else true end' "$CONFIG_FILE")

if [ "$TEST_MODE" = true ]; then
    N_EVENTS=2
fi

# --- Validate -----------------------------------------------------------------
[ "$CONFIG_NAME"   != "null" ] || { echo "Error: config missing required field 'name'"   >&2; exit 1; }
[ "$MATERIAL"      != "null" ] || { echo "Error: config missing required field 'material'"    >&2; exit 1; }
[ "$OUTPUT_PATH"   != "null" ] || { echo "Error: config missing required field 'output_path'" >&2; exit 1; }

if [ "$PRIMARY_SOURCE" != "genie" ] && [ "$ENERGY_DIST" != "monoenergetic" ] && [ "$ENERGY_DIST" != "uniform" ]; then
    echo "Error: energy_distribution must be 'monoenergetic' or 'uniform' (got: $ENERGY_DIST)" >&2
    exit 1
fi

if [ "$USE_CONFIG_NUMBER" == "true" ] && { [ "$CONFIG_NUMBER" == "null" ] || [ "$CONFIG_NUMBER" == "-1" ]; }; then
    echo "Error: config_number is required when use_config_number is true" >&2
    exit 1
fi

[ -n "${LUCID_PATH:-}" ] || { echo "Error: LUCID_PATH not set in user_paths.sh" >&2; exit 1; }

if [ "$PRIMARY_SOURCE" = "genie" ]; then
    [ -n "${LUCID_IMAGE_PATH:-}" ] || {
        echo "Error: LUCID_IMAGE_PATH not set (required for primary_source=genie)." >&2
        echo "Build the unified container with LUCiD/container/lucid.def and point" >&2
        echo "LUCID_IMAGE_PATH at the resulting .sif." >&2
        exit 1; }
    [ -f "${LUCID_IMAGE_PATH}" ] || {
        echo "Error: LUCID_IMAGE_PATH=${LUCID_IMAGE_PATH} does not exist." >&2
        exit 1; }
else
    [ -n "${PHOTONSIM_BIN:-}" ] || { echo "Error: PHOTONSIM_BIN not set in user_paths.sh" >&2; exit 1; }
fi

# --- Effective resources ------------------------------------------------------
EFFECTIVE_OUTPUT_BASE="${OUTPUT_OVERRIDE:-$OUTPUT_BASE_PATH}"
EFFECTIVE_PARTITION="${PARTITION_OVERRIDE:-$SLURM_PARTITION}"
if [ "$USE_GPU" = true ]; then EFFECTIVE_GPUS="1"; else EFFECTIVE_GPUS="$DEFAULT_GPUS"; fi

OUTPUT_BASE_DIR="${EFFECTIVE_OUTPUT_BASE}/${MATERIAL}/${OUTPUT_PATH}"
if [ "$USE_CONFIG_NUMBER" == "true" ]; then
    CONFIG_DIR="${OUTPUT_BASE_DIR}/config_$(printf "%06d" "$CONFIG_NUMBER")"
else
    CONFIG_DIR="${OUTPUT_BASE_DIR}"
fi
mkdir -p "$CONFIG_DIR"

# Pre-create the v3 four-subdir layout so concurrent jobs never race on mkdir.
if [ "$RUN_LUCID" == "true" ]; then
    mkdir -p "$CONFIG_DIR/sensor" "$CONFIG_DIR/inst" "$CONFIG_DIR/seg" "$CONFIG_DIR/labl"
fi

# --- Summary ------------------------------------------------------------------
if [ "$USE_CONFIG_NUMBER" == "true" ]; then
    echo "Configuration: $CONFIG_NAME (config_$(printf '%06d' "$CONFIG_NUMBER"))"
else
    echo "Configuration: $CONFIG_NAME (no config subfolder)"
fi
echo "Description:   $CONFIG_DESC"
echo "Energy dist:   $ENERGY_DIST"
echo "Particles:     $N_PARTICLES"
echo "Jobs:          $N_JOBS (events/job=$N_EVENTS)"
echo "Partition:     $EFFECTIVE_PARTITION  GPUs=$EFFECTIVE_GPUS"
echo "Output dir:    $CONFIG_DIR"
echo ""

# --- Helpers ------------------------------------------------------------------
emit_sbatch() {
    # Args: slurm_out_dir, job_id (6-digit), job_name, [override_energy_MeV]
    local out_dir="$1"
    local job_id="$2"
    local job_name="$3"
    local override_energy="${4:-}"

    local override_flag=""
    [ -n "$override_energy" ] && override_flag="--override-energy-MeV ${override_energy}"
    local test_flag=""
    [ "$TEST_MODE" = true ] && test_flag="--test"

    local gpu_line=""
    [ "$EFFECTIVE_GPUS" != "0" ] && gpu_line="#SBATCH --gpus=${EFFECTIVE_GPUS}"

    local sbatch_path="${out_dir}/submit_job_${job_id}.sbatch"

    # Common SBATCH header.
    cat > "$sbatch_path" <<EOFSBATCH
#!/bin/bash
#SBATCH --partition=${EFFECTIVE_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --job-name=${job_name}
#SBATCH --output=${out_dir}/job_${job_id}-%j.out
#SBATCH --error=${out_dir}/job_${job_id}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${DEFAULT_CPUS}
${gpu_line}
#SBATCH --mem=${DEFAULT_MEMORY}
#SBATCH --time=${DEFAULT_TIME}

set -eu -o pipefail
echo "SLURM Job ID: \${SLURM_JOB_ID}"
echo "Job started:  \$(date)"
echo "Node:         \$(hostname)"

EOFSBATCH

    if [ "$PRIMARY_SOURCE" = "genie" ]; then
        # Single-step body: everything runs inside the unified LUCiD image
        # (GEANT4 + ROOT + GENIE + PhotonSim + LUCiD). No bare-host state.
        local skip_lucid_flag=""
        [ "$RUN_LUCID" != "true" ] && skip_lucid_flag="--skip-lucid"

        cat >> "$sbatch_path" <<EOFSBATCH
# Unified container: GEANT4 + ROOT + GENIE + PhotonSim + LUCiD.
# Cross-section splines live on /cvmfs (526 MB), so we bind it into the
# container rather than bake into the image.
export APPTAINERENV_GENIE_XSEC_FILE=${GENIE_XSEC_FILE}
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs \\
    ${LUCID_IMAGE_PATH} \\
    lucid-run-job \\
        --config "${CONFIG_FILE}" \\
        --output-dir "${out_dir}" \\
        --job-id ${job_id} ${test_flag} ${override_flag} ${skip_lucid_flag}

EOFSBATCH
    else
        # Classic two-step body: PhotonSim on bare host, LUCiD in legacy image.
        cat >> "$sbatch_path" <<EOFSBATCH
# Host-side env: GEANT4 + ROOT shared libs for the PhotonSim binary.
source ${UTILS_DIR}/setup_environment.sh
export PHOTONSIM_BIN=${PHOTONSIM_BIN}
export PYTHONPATH=${LUCID_PATH}:\${PYTHONPATH:-}
export APPTAINERENV_PYTHONPATH=\${PYTHONPATH}

# Step A: macro generation + PhotonSim on the bare node (no jax needed).
${HOST_PYTHON:-python3} -m lucid.production.run_job \\
    --config "${CONFIG_FILE}" \\
    --output-dir "${out_dir}" \\
    --job-id ${job_id} \\
    --skip-lucid ${test_flag} ${override_flag}

EOFSBATCH

        if [ "$RUN_LUCID" == "true" ]; then
            cat >> "$sbatch_path" <<EOFSBATCH
# Step B: LUCiD v3 writer inside the legacy singularity image.
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch \\
    ${SINGULARITY_IMAGE_PATH} \\
    python3 -m lucid.production.run_job \\
        --config "${CONFIG_FILE}" \\
        --output-dir "${out_dir}" \\
        --job-id ${job_id} \\
        --skip-photonsim ${test_flag}

EOFSBATCH
        fi
    fi

    cat >> "$sbatch_path" <<EOFSBATCH
echo "Job ended: \$(date)"
EOFSBATCH

    chmod +x "$sbatch_path"
    echo "$sbatch_path"
}

create_readme() {
    local readme="${CONFIG_DIR}/README.md"
    local config_id
    if [ "$USE_CONFIG_NUMBER" == "true" ]; then
        config_id="$(printf '%06d' "$CONFIG_NUMBER")"
    else
        config_id="N/A (use_config_number: false)"
    fi

    cat > "$readme" <<EOF
# Dataset: ${CONFIG_NAME}

- **Config number**: ${config_id}
- **Description**: ${CONFIG_DESC}
- **Material**: ${MATERIAL}
- **Energy distribution**: ${ENERGY_DIST}
- **Jobs**: ${N_JOBS}  ×  **Events/job**: ${N_EVENTS}  =  **Total events**: $((N_JOBS * N_EVENTS))
- **Particles per event**: ${N_PARTICLES}
- **Generated**: $(date)

## Layout

This directory is one LUCiD dataset. Each job contributes one v3 batch
(\`file_index = job_id - 1\`) consisting of four files spread across
\`sensor/\`, \`inst/\`, \`seg/\`, \`labl/\`. See
\`LUCiD/docs/LUCID_DATASET.md\` for the schema.
EOF
}

# --- Fan-out ------------------------------------------------------------------
SUBMITTED=0
PREPARED=0

if [ "$ENERGY_DIST" == "uniform" ]; then
    JOBS_TO_PROCESS=$N_JOBS
    [ "$TEST_MODE" = true ] && JOBS_TO_PROCESS=1

    for (( j=1; j<=JOBS_TO_PROCESS; j++ )); do
        JOB_ID=$(printf "%06d" "$j")
        if [ "$USE_CONFIG_NUMBER" == "true" ]; then
            JOB_NAME="photonsim_config$(printf '%06d' "$CONFIG_NUMBER")_${JOB_ID}"
        else
            JOB_NAME="photonsim_${CONFIG_NAME_SLUG}_${JOB_ID}"
        fi
        sb=$(emit_sbatch "$CONFIG_DIR" "$JOB_ID" "$JOB_NAME")
        PREPARED=$((PREPARED+1))
        if [ "$SUBMIT_JOBS" = true ]; then
            sbatch "$sb" >/dev/null && SUBMITTED=$((SUBMITTED+1))
        fi
    done

elif [ "$ENERGY_DIST" == "monoenergetic" ]; then
    # Derive list of energies (single value or scan)
    SINGLE_E=$(jq -r '.energy_MeV // empty' "$CONFIG_FILE")
    SCAN_START=$(jq -r '.energy_scan.start_MeV // empty' "$CONFIG_FILE")
    SCAN_STOP=$(jq -r '.energy_scan.stop_MeV // empty' "$CONFIG_FILE")
    SCAN_STEP=$(jq -r '.energy_scan.step_MeV // empty' "$CONFIG_FILE")

    ENERGIES=()
    if [ -n "$SINGLE_E" ]; then
        ENERGIES=("$SINGLE_E")
    elif [ -n "$SCAN_START" ] && [ -n "$SCAN_STOP" ] && [ -n "$SCAN_STEP" ]; then
        for (( e=SCAN_START; e<=SCAN_STOP; e+=SCAN_STEP )); do ENERGIES+=("$e"); done
    else
        echo "Error: monoenergetic requires either 'energy_MeV' or 'energy_scan.{start,stop,step}_MeV'" >&2
        exit 1
    fi

    JOBS_TO_PROCESS=$N_JOBS
    [ "$TEST_MODE" = true ] && JOBS_TO_PROCESS=1

    for energy in "${ENERGIES[@]}"; do
        e_int=$(printf "%.0f" "$energy")
        ENERGY_DIR="${CONFIG_DIR}/${e_int}MeV"
        mkdir -p "$ENERGY_DIR"
        if [ "$RUN_LUCID" == "true" ]; then
            mkdir -p "$ENERGY_DIR/sensor" "$ENERGY_DIR/inst" "$ENERGY_DIR/seg" "$ENERGY_DIR/labl"
        fi

        for (( j=1; j<=JOBS_TO_PROCESS; j++ )); do
            JOB_ID=$(printf "%06d" "$j")
            if [ "$USE_CONFIG_NUMBER" == "true" ]; then
                JOB_NAME="photonsim_config$(printf '%06d' "$CONFIG_NUMBER")_${e_int}MeV_${JOB_ID}"
            else
                JOB_NAME="photonsim_${CONFIG_NAME_SLUG}_${e_int}MeV_${JOB_ID}"
            fi
            sb=$(emit_sbatch "$ENERGY_DIR" "$JOB_ID" "$JOB_NAME" "$energy")
            PREPARED=$((PREPARED+1))
            if [ "$SUBMIT_JOBS" = true ]; then
                sbatch "$sb" >/dev/null && SUBMITTED=$((SUBMITTED+1))
            fi
        done

        [ "$TEST_MODE" = true ] && break  # one energy in test mode
    done
fi

create_readme

echo ""
echo "=== Fan-out complete ==="
echo "Prepared:  $PREPARED"
echo "Submitted: $SUBMITTED"
echo "Config dir: $CONFIG_DIR"
echo ""
echo "Monitor with: ${SCRIPT_DIR}/monitor_jobs.sh -w"
