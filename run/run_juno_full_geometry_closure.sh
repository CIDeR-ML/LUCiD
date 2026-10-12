#!/usr/bin/env bash
# JUNO full-geometry event-wise synthetic closure.
#
# This worker script never invokes sudo or srun.  Submit it through the
# collaboration's existing shared-account sbatch entry point.
#SBATCH --job-name=lucid_synthetic_closure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/stage/liukainan/LUCiD
#SBATCH --output=/junofs/C/liukainan/LUCiD/slurm/calibration_%j.slurm.out
#SBATCH --error=/junofs/C/liukainan/LUCiD/slurm/calibration_%j.slurm.err

set -euo pipefail

WORK=/mnt/stage/liukainan/LUCiD
# Pin the closure worker to the collaboration's Python 3.13 + ROCm 7
# interpreter.  Do not inherit a generic or stale LUCID_PY from the submit
# shell; CLOSURE_PYTHON remains an explicit diagnostic override.
export LUCID_PY="${CLOSURE_PYTHON:-/opt/gentoo/usr/bin/python3.13}"
if [[ ! -x "$LUCID_PY" ]]; then
    echo "Missing closure Python interpreter: $LUCID_PY" >&2
    exit 2
fi
# Disable XLA command-buffer replay because the current ROCm stack has
# reported malformed AQL packets when replaying compiled transport graphs.
export LUCID_DISABLE_XLA_COMMAND_BUFFER=1
source "$WORK/run/env.sh"
# Disable the XLA GPU autotuner: with ROCm 7 + Python 3.13 the
# autotuner picks kernel configurations for multi-source graphs whose
# SECOND execution emits malformed AQL packets / SIGSEGV (issue:
# ISSUE-rocm7-aql-crash-forward-vjp.md; bisect stage I, job 5547209).
# Default algorithms replay safely; single-source graphs were unaffected.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"

: "${OUTDIR:?Set OUTDIR to a new closure output directory}"
: "${SOURCES:?Set SOURCES to space-separated x,y,z positions}"
CLOSURE_MODE=${CLOSURE_MODE:-full}
OBSERVATION_ROOT=${OBSERVATION_ROOT:-}

if [[ -e "$OUTDIR/closure_result.json" ]]; then
    echo "Refusing to overwrite completed result: $OUTDIR" >&2
    exit 2
fi

export JAX_PLATFORMS=rocm
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK"

N_EVENTS=${N_EVENTS:-2}
MODEL_PROFILE=${MODEL_PROFILE:-full}
DATA_N_PHOTONS=${DATA_N_PHOTONS:-400000}
FIT_N_PHOTONS=${FIT_N_PHOTONS:-100000}
K=${K:-64}
INTENSITY=${INTENSITY:-100000}
EVENT_EXPOSURE=${EVENT_EXPOSURE:-1}
EVENT_TIME_OFFSET_NS=${EVENT_TIME_OFFSET_NS:-0}
BACKGROUND_RATE_HZ=${BACKGROUND_RATE_HZ:-30000}
TIME_MIN_NS=${TIME_MIN_NS:-0}
TIME_MAX_NS=${TIME_MAX_NS:-1000}
TIME_BIN_NS=${TIME_BIN_NS:-2}
INTERPOLATION_SIGMA_NS=${INTERPOLATION_SIGMA_NS:-}
WINDOW_EXPANSION_NS=${WINDOW_EXPANSION_NS:-50}
TRUNCATE_SIGMA=${TRUNCATE_SIGMA:-5}
OPTIMIZATION_PLAN=${OPTIMIZATION_PLAN:-joint-only}
JOINT_STEPS=${JOINT_STEPS:-16}
STAGE1_STEPS=${STAGE1_STEPS:-64}
STAGE2_STEPS=${STAGE2_STEPS:-64}
STAGE3_STEPS=${STAGE3_STEPS:-64}
LEARNING_RATE=${LEARNING_RATE:-0.001}
GRAD_CLIP=${GRAD_CLIP:-10000}
START_FRACTION=${START_FRACTION:-0.1}
AD_FD_RELATIVE_STEP=${AD_FD_RELATIVE_STEP:-0}
DATA_ENSEMBLE=${DATA_ENSEMBLE:-2}
FIT_ENSEMBLE=${FIT_ENSEMBLE:-2}
MC_VALIDATION_ENSEMBLE=${MC_VALIDATION_ENSEMBLE:-2}
SEED=${SEED:-42}
FIT_SEED_OFFSET=${FIT_SEED_OFFSET:-1000003}
MC_VALIDATION_SEED_OFFSET=${MC_VALIDATION_SEED_OFFSET:-2000003}
HELDOUT_SEED_OFFSET=${HELDOUT_SEED_OFFSET:-3000003}
VALIDATION_EVERY=${VALIDATION_EVERY:-2}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
MIN_IMPROVEMENT=${MIN_IMPROVEMENT:-0.0001}
SOURCE_CHUNK_SIZE=${SOURCE_CHUNK_SIZE:-0}
OBJECTIVE_EXECUTION=${OBJECTIVE_EXECUTION:-fused}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-4}
DATA_IMPORTANCE=${DATA_IMPORTANCE:-config}
FIT_IMPORTANCE=${FIT_IMPORTANCE:-config}
RESTART_CHECKPOINT=${RESTART_CHECKPOINT:-}
HELDOUT_SOURCE_INDICES=${HELDOUT_SOURCE_INDICES:-}
SCRIPT_ARGS=${SCRIPT_ARGS:-}
SUPPORT_DIAGNOSTIC_ONLY=${SUPPORT_DIAGNOSTIC_ONLY:-0}

read -r -a SOURCE_ITEMS <<< "$SOURCES"
read -r -a HELDOUT_ITEMS <<< "$HELDOUT_SOURCE_INDICES"
read -r -a EXTRA_ARGS <<< "$SCRIPT_ARGS"

SOURCE_ARGS=()
SOURCE_INDEX_ARGS=()
for index in "${!SOURCE_ITEMS[@]}"; do
    source=${SOURCE_ITEMS[$index]}
    SOURCE_ARGS+=(--source-position "$source")
    SOURCE_INDEX_ARGS+=(--source-index "$index")
done
HELDOUT_ARGS=()
for index in "${HELDOUT_ITEMS[@]}"; do
    HELDOUT_ARGS+=(--heldout-source-index "$index")
done
RESTART_ARGS=()
if [[ -n "$RESTART_CHECKPOINT" ]]; then
    RESTART_ARGS+=(--restart-checkpoint "$RESTART_CHECKPOINT")
fi

echo "=== synthetic closure start: $(hostname) $(date) ==="
echo "sources=${SOURCE_ITEMS[*]}"
echo "mode=$CLOSURE_MODE outdir=$OUTDIR"
echo "model_profile=$MODEL_PROFILE"
echo "plan=$OPTIMIZATION_PLAN K=$K"
echo "events=$N_EVENTS data_photons=$DATA_N_PHOTONS \
fit_photons=$FIT_N_PHOTONS source_chunk=$SOURCE_CHUNK_SIZE"
echo "objective_execution=$OBJECTIVE_EXECUTION"
echo "ensembles=data:$DATA_ENSEMBLE fit:$FIT_ENSEMBLE \
mc_validation:$MC_VALIDATION_ENSEMBLE"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "ROCm Queue is idle after a phase means normal process teardown."
"$LUCID_PY" -c \
    'import sys, h5py, jax, jaxlib, pandas
print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("jax:", jax.__version__, "jaxlib:", jaxlib.__version__)
print("jax devices:", jax.devices())'

COMMON_ARGS=(
    --config config/JUNO_laser_config.json \
    --model-profile "$MODEL_PROFILE" \
    --n-events "$N_EVENTS" \
    --data-n-photons "$DATA_N_PHOTONS" \
    --fit-n-photons "$FIT_N_PHOTONS" \
    --k "$K" \
    --intensity "$INTENSITY" \
    --event-exposure "$EVENT_EXPOSURE" \
    --event-time-offset-ns "$EVENT_TIME_OFFSET_NS" \
    --background-rate-hz "$BACKGROUND_RATE_HZ" \
    --time-min-ns "$TIME_MIN_NS" \
    --time-max-ns "$TIME_MAX_NS" \
    --time-bin-ns "$TIME_BIN_NS" \
    --truncate-sigma "$TRUNCATE_SIGMA" \
    --window-stability-expansion-ns "$WINDOW_EXPANSION_NS" \
    --optimization-plan "$OPTIMIZATION_PLAN" \
    --joint-steps "$JOINT_STEPS" \
    --stage1-steps "$STAGE1_STEPS" \
    --stage2-steps "$STAGE2_STEPS" \
    --stage3-steps "$STAGE3_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --grad-clip "$GRAD_CLIP" \
    --start-fraction "$START_FRACTION" \
    --ad-fd-relative-step "$AD_FD_RELATIVE_STEP" \
    --data-ensemble-size "$DATA_ENSEMBLE" \
    --fit-ensemble-size "$FIT_ENSEMBLE" \
    --mc-validation-ensemble-size "$MC_VALIDATION_ENSEMBLE" \
    --seed "$SEED" \
    --fit-seed-offset "$FIT_SEED_OFFSET" \
    --mc-validation-seed-offset "$MC_VALIDATION_SEED_OFFSET" \
    --heldout-seed-offset "$HELDOUT_SEED_OFFSET" \
    --validation-every "$VALIDATION_EVERY" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --min-improvement "$MIN_IMPROVEMENT" \
    --source-chunk-size "$SOURCE_CHUNK_SIZE" \
    --objective-execution "$OBJECTIVE_EXECUTION" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --data-importance-normalization "$DATA_IMPORTANCE" \
    --fit-importance-normalization "$FIT_IMPORTANCE"
)
if [[ -n "$INTERPOLATION_SIGMA_NS" ]]; then
    COMMON_ARGS+=(--interpolation-sigma-ns "$INTERPOLATION_SIGMA_NS")
fi
if [[ "$SUPPORT_DIAGNOSTIC_ONLY" == "1" ]]; then
    COMMON_ARGS+=(--support-diagnostic-only)
fi

OBSERVATION_ARGS=()
if [[ "$CLOSURE_MODE" == "full" ]]; then
    GENERATION_ROOT="$OUTDIR/generation"
    mkdir -p "$GENERATION_ROOT"
    for index in "${!SOURCE_ITEMS[@]}"; do
        source=${SOURCE_ITEMS[$index]}
        source_out=$(printf '%s/source_%02d' "$GENERATION_ROOT" "$index")
        if [[ -e "$source_out/closure_observations.h5" ]]; then
            echo "Refusing stale source fragment: $source_out" >&2
            exit 2
        fi
        echo "=== generate source $index: $source ==="
        "$LUCID_PY" -u scripts/run_juno_full_geometry_closure.py \
            --phase generate \
            --source-position "$source" \
            --source-index "$index" \
            --outdir "$source_out" \
            "${COMMON_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        OBSERVATION_ARGS+=(
            --observation-input "$source_out/closure_observations.h5"
        )
    done
elif [[ "$CLOSURE_MODE" == "fit-only" ]]; then
    if [[ -z "$OBSERVATION_ROOT" ]]; then
        echo "fit-only requires OBSERVATION_ROOT" >&2
        exit 2
    fi
    for index in "${!SOURCE_ITEMS[@]}"; do
        fragment=$(printf \
            '%s/source_%02d/closure_observations.h5' \
            "$OBSERVATION_ROOT" "$index")
        if [[ ! -f "$fragment" ]]; then
            echo "Missing source fragment: $fragment" >&2
            exit 2
        fi
        OBSERVATION_ARGS+=(--observation-input "$fragment")
    done
    echo "reusing observations from $OBSERVATION_ROOT"
else
    echo "Unsupported CLOSURE_MODE: $CLOSURE_MODE" >&2
    exit 2
fi

echo "=== fit all ${#SOURCE_ITEMS[@]} source(s) in a fresh process ==="
"$LUCID_PY" -u scripts/run_juno_full_geometry_closure.py \
    --phase fit \
    "${SOURCE_ARGS[@]}" \
    "${SOURCE_INDEX_ARGS[@]}" \
    "${OBSERVATION_ARGS[@]}" \
    --outdir "$OUTDIR" \
    "${COMMON_ARGS[@]}" \
    "${HELDOUT_ARGS[@]}" \
    "${RESTART_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

if [[ "$SUPPORT_DIAGNOSTIC_ONLY" == "1" ]]; then
    echo "=== support-only diagnostic done: $(date) ==="
    exit 0
fi

echo "=== render closure diagnostics ==="
"$LUCID_PY" scripts/plot_juno_closure_result.py \
    --result "$OUTDIR/closure_result.json" \
    --output "$OUTDIR/closure_diagnostics.pdf" \
    --summary "$OUTDIR/closure_diagnostics.json"

echo "=== synthetic closure done: $(date) ==="
