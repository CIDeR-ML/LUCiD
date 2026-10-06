#!/usr/bin/env bash
# Per-source Jacobian and numerical-time-basis Fisher diagnostics.
#SBATCH --job-name=lucid_fisher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/stage/liukainan/LUCiD

set -euo pipefail

WORK=/mnt/stage/liukainan/LUCiD
export LUCID_PY="${CLOSURE_PYTHON:-/opt/gentoo/usr/bin/python3.13}"
export LUCID_DISABLE_XLA_COMMAND_BUFFER=1
source "$WORK/run/env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"
export JAX_PLATFORMS=rocm
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

: "${OUTDIR:?Set a new Fisher output directory}"
: "${SOURCES:?Set space-separated source positions}"
N_PHOTONS=${N_PHOTONS:-100000}
N_EVENTS=${N_EVENTS:-100}
K=${K:-32}
INTENSITY=${INTENSITY:-100000}
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-1}
BACKGROUND_RATE_HZ=${BACKGROUND_RATE_HZ:-30000}
TIME_MAX_NS=${TIME_MAX_NS:-1000}
TIME_BIN_NS=${TIME_BIN_NS:-2}
SEED=${SEED:-42}

if [[ -e "$OUTDIR/fisher_result.json" ]]; then
    echo "Refusing to overwrite completed Fisher result: $OUTDIR" >&2
    exit 2
fi
mkdir -p "$OUTDIR"
read -r -a SOURCE_ITEMS <<< "$SOURCES"
SOURCE_ARGS=()
for source in "${SOURCE_ITEMS[@]}"; do
    SOURCE_ARGS+=(--source-position "$source")
done

cd "$WORK"
echo "=== Fisher start: $(hostname) $(date) ==="
echo "sources=${SOURCE_ITEMS[*]} photons=$N_PHOTONS K=$K"
"$LUCID_PY" scripts/check_juno_synthetic_fisher.py \
    "${SOURCE_ARGS[@]}" \
    --output "$OUTDIR/fisher_result.json" \
    --n-photons "$N_PHOTONS" \
    --k "$K" \
    --intensity "$INTENSITY" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --n-events "$N_EVENTS" \
    --background-rate-hz "$BACKGROUND_RATE_HZ" \
    --time-max-ns "$TIME_MAX_NS" \
    --time-bin-ns "$TIME_BIN_NS" \
    --seed "$SEED"
"$LUCID_PY" scripts/plot_juno_synthetic_fisher.py \
    --result "$OUTDIR/fisher_result.json" \
    --output "$OUTDIR/fisher_diagnostics.pdf"
echo "=== Fisher done: $(date) ==="
