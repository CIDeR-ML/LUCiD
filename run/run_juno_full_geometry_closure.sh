#!/usr/bin/env bash
# Minimal JUNO event-wise synthetic closure worker.

#SBATCH --job-name=lucid_minimal_closure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK=$(cd -- "$SCRIPT_DIR/.." && pwd)
export LUCID_PY="${CLOSURE_PYTHON:-/opt/gentoo/usr/bin/python3.13}"
if [[ ! -x "$LUCID_PY" ]]; then
    echo "Missing closure Python interpreter: $LUCID_PY" >&2
    exit 2
fi

export LUCID_DISABLE_XLA_COMMAND_BUFFER=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"
export JAX_PLATFORMS=rocm
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}"

: "${OUTDIR:?Set OUTDIR to a new closure output directory}"
: "${SOURCES:?Set SOURCES to space-separated x,y,z positions}"
if [[ -e "$OUTDIR/closure_result.json" ]]; then
    echo "Refusing to overwrite completed result: $OUTDIR" >&2
    exit 2
fi

N_EVENTS=${N_EVENTS:-100}
DATA_N_PHOTONS=${DATA_N_PHOTONS:-4000000}
FIT_N_PHOTONS=${FIT_N_PHOTONS:-4000000}
K=${K:-32}
INTENSITY=${INTENSITY:-400000}
TIME_MIN_NS=${TIME_MIN_NS:-0}
TIME_MAX_NS=${TIME_MAX_NS:-1000}
TIME_BIN_NS=${TIME_BIN_NS:-0.5}
INTERPOLATION_SIGMA_NS=${INTERPOLATION_SIGMA_NS:-0.5}
WINDOW_EXPANSION_NS=${WINDOW_EXPANSION_NS:-50}
TRUNCATE_SIGMA=${TRUNCATE_SIGMA:-5}
JOINT_STEPS=${JOINT_STEPS:-600}
LEARNING_RATE=${LEARNING_RATE:-0.001}
GRAD_CLIP=${GRAD_CLIP:-10000}
START_FRACTION=${START_FRACTION:-0.1}
AD_FD_RELATIVE_STEP=${AD_FD_RELATIVE_STEP:-0.001}
DATA_ENSEMBLE=${DATA_ENSEMBLE:-2}
FIT_ENSEMBLE=${FIT_ENSEMBLE:-1}
MC_VALIDATION_ENSEMBLE=${MC_VALIDATION_ENSEMBLE:-2}
SEED=${SEED:-42}
FIT_SEED_OFFSET=${FIT_SEED_OFFSET:-1000003}
MC_VALIDATION_SEED_OFFSET=${MC_VALIDATION_SEED_OFFSET:-2000003}
VALIDATION_EVERY=${VALIDATION_EVERY:-2}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
MIN_IMPROVEMENT=${MIN_IMPROVEMENT:-0.0001}
SOURCE_CHUNK_SIZE=${SOURCE_CHUNK_SIZE:-1}
OBJECTIVE_EXECUTION=${OBJECTIVE_EXECUTION:-per-source}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-4}
RESTART_CHECKPOINT=${RESTART_CHECKPOINT:-}
SCRIPT_ARGS=${SCRIPT_ARGS:-}

read -r -a SOURCE_ITEMS <<< "$SOURCES"
read -r -a EXTRA_ARGS <<< "$SCRIPT_ARGS"
SOURCE_ARGS=()
SOURCE_INDEX_ARGS=()
for index in "${!SOURCE_ITEMS[@]}"; do
    SOURCE_ARGS+=(--source-position "${SOURCE_ITEMS[$index]}")
    SOURCE_INDEX_ARGS+=(--source-index "$index")
done
RESTART_ARGS=()
if [[ -n "$RESTART_CHECKPOINT" ]]; then
    RESTART_ARGS+=(--restart-checkpoint "$RESTART_CHECKPOINT")
fi

COMMON_ARGS=(
    --config config/JUNO_laser_config.json
    --n-events "$N_EVENTS"
    --data-n-photons "$DATA_N_PHOTONS"
    --fit-n-photons "$FIT_N_PHOTONS"
    --k "$K"
    --intensity "$INTENSITY"
    --time-min-ns "$TIME_MIN_NS"
    --time-max-ns "$TIME_MAX_NS"
    --time-bin-ns "$TIME_BIN_NS"
    --interpolation-sigma-ns "$INTERPOLATION_SIGMA_NS"
    --truncate-sigma "$TRUNCATE_SIGMA"
    --window-stability-expansion-ns "$WINDOW_EXPANSION_NS"
    --joint-steps "$JOINT_STEPS"
    --learning-rate "$LEARNING_RATE"
    --grad-clip "$GRAD_CLIP"
    --start-fraction "$START_FRACTION"
    --ad-fd-relative-step "$AD_FD_RELATIVE_STEP"
    --data-ensemble-size "$DATA_ENSEMBLE"
    --fit-ensemble-size "$FIT_ENSEMBLE"
    --mc-validation-ensemble-size "$MC_VALIDATION_ENSEMBLE"
    --seed "$SEED"
    --fit-seed-offset "$FIT_SEED_OFFSET"
    --mc-validation-seed-offset "$MC_VALIDATION_SEED_OFFSET"
    --validation-every "$VALIDATION_EVERY"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --min-improvement "$MIN_IMPROVEMENT"
    --source-chunk-size "$SOURCE_CHUNK_SIZE"
    --objective-execution "$OBJECTIVE_EXECUTION"
    --checkpoint-every "$CHECKPOINT_EVERY"
)

cd "$WORK"
echo "=== minimal closure start: $(hostname) $(date) ==="
echo "sources=${SOURCE_ITEMS[*]} outdir=$OUTDIR"
echo "events=$N_EVENTS data_photons=$DATA_N_PHOTONS fit_photons=$FIT_N_PHOTONS"
echo "K=$K objective_execution=$OBJECTIVE_EXECUTION"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
"$LUCID_PY" -c \
    'import sys, jax, jaxlib
print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("jax:", jax.__version__, "jaxlib:", jaxlib.__version__)
print("jax devices:", jax.devices())'

GENERATION_ROOT="$OUTDIR/generation"
mkdir -p "$GENERATION_ROOT"
OBSERVATION_ARGS=()
for index in "${!SOURCE_ITEMS[@]}"; do
    source=${SOURCE_ITEMS[$index]}
    source_out=$(printf '%s/source_%02d' "$GENERATION_ROOT" "$index")
    if [[ -e "$source_out/closure_observations.h5" ]]; then
        echo "Refusing stale source fragment: $source_out" >&2
        exit 2
    fi
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

"$LUCID_PY" -u scripts/run_juno_full_geometry_closure.py \
    --phase fit \
    "${SOURCE_ARGS[@]}" \
    "${SOURCE_INDEX_ARGS[@]}" \
    "${OBSERVATION_ARGS[@]}" \
    --outdir "$OUTDIR" \
    "${COMMON_ARGS[@]}" \
    "${RESTART_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

"$LUCID_PY" scripts/plot_juno_closure_result.py \
    --result "$OUTDIR/closure_result.json" \
    --output "$OUTDIR/closure_diagnostics.pdf" \
    --summary "$OUTDIR/closure_diagnostics.json"
echo "=== minimal closure done: $(date) ==="
