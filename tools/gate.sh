#!/usr/bin/env bash
# The light gate, in one command (docs/development.md, "The gate"). Three steps, no
# options, run from anywhere:
#
#   L1  pytest tests/ -q                       -- the existing suite stays green
#   L2  every analysis/paper/fig_*.py that offers --plot-results re-plots the
#       output already there (imports and --help are L1:
#       tests/paper/test_fig_entrypoints.py)
#   L3  the production entry points still run at their smallest local size
#
# L4 -- a reviewer reads the diff -- is not something a script can do.
#
# Exit status, so a caller can tell the steps apart without parsing the log:
#   0  all three green
#   1  L1 failed
#   2  L2 failed
#   3  L3 failed
# All three steps always run, and each runs to the end and reports every
# failure it found, so one invocation shows everything that is wrong. The
# status names the FIRST step that had a failure.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Node-local scratch, never $HOME. Slurm exports a $TMPDIR the container does
# not always have bound, so the root is created, checked and only then used:
# an unwritable one must not abort the gate before it has run a single test.
TMPROOT="${TMPDIR:-${LSCRATCH:-/tmp}}"
mkdir -p -- "$TMPROOT" 2>/dev/null || true
if [ ! -w "$TMPROOT" ]; then
    TMPROOT=/tmp
fi
TMP="$(mktemp -d "$TMPROOT/lucid-gate.XXXXXX")"
trap 'rm -rf -- "$TMP"' EXIT

PY="${PYTHON:-python3}"

# ---------------------------------------------------------------- L1
FAILED_STEP=0
note_failure() {
    if [ "$FAILED_STEP" -eq 0 ]; then
        FAILED_STEP="$1"
    fi
}

echo "=== L1  pytest tests/ -q"
if ! "$PY" -m pytest tests/ -q; then
    echo "GATE: L1 failed -- the suite is not green" >&2
    note_failure 1
fi

# ---------------------------------------------------------------- L2
# The 14 figure scripts are the paper's user interface. That each imports and
# parses --help is L1's tests/paper/test_fig_entrypoints.py; this step adds the
# one thing pytest does not do, a real invocation.
#
# --plot-results is the smallest real invocation 12 of them offer, and it means
# "re-plot the output that is already there" -- so it runs only for a figure
# whose own data directory holds something. The paper output tree is generated,
# not committed, and four of the twelve (calib_convergence,
# calib_loss_geometry, detector_gallery, loss_landscape) raise rather than skip when their
# input is missing, so running them against an empty tree tests nothing and
# fails for the wrong reason. fig_photon_event_display and fig_siren_profile
# have no plot-only path at all; L1 covers them.
echo "=== L2  analysis/paper/fig_*.py"

# Does this figure have output to re-plot? A module that names its own data
# directory (FIGURE) is checked against it; one that does not is left to its
# own skip handling. paths.data_dir() creates the directory it returns, so the
# test is for a FILE in it, not for the directory.
replot_has_input() {
    "$PY" - "$1" <<'PYEOF'
import importlib
import sys

mod = importlib.import_module(sys.argv[1])
figure = getattr(mod, 'FIGURE', None)
if figure is None:
    sys.exit(0)
from analysis.paper.utils import paths
d = paths.data_dir(figure)
sys.exit(0 if any(q.is_file() for q in d.rglob('*')) else 1)
PYEOF
}

L2_BAD=0
for f in analysis/paper/fig_*.py; do
    mod="analysis.paper.$(basename -- "$f" .py)"
    if ! grep -q -- '--plot-results' "$f"; then
        continue
    fi
    echo "--- $mod"
    if ! replot_has_input "$mod"; then
        echo "GATE: no output under analysis/paper/output/data for $mod" >&2
        echo "GATE: --plot-results did NOT run for it. Generate its data to cover L2 fully." >&2
        continue
    fi
    if ! "$PY" -m "$mod" --plot-results --out "$TMP/figures"; then
        echo "GATE: $mod --plot-results failed" >&2
        L2_BAD=1
    fi
done
if [ "$L2_BAD" -ne 0 ]; then
    echo "GATE: L2 failed" >&2
    note_failure 2
fi

# ---------------------------------------------------------------- L3
# The smallest shipped dataset config, run through both production entry
# points: the worker (lucid.production.dataprod.run_job, one job of two events with the
# Geant4 stage skipped) and the host-side fan-out (dataprod_fanout with no
# --submit, which is its dry run -- it resolves the detector, the digitizer and
# the trigger, plans the splits and renders the argv without queueing
# anything).
echo "=== L3  production, smallest shipped config"
CFG="lucid/production/dataprod/configs/GeV/02_mu_smoke.json"
L3_BAD=0

if ! "$PY" -m lucid.production.dataprod.run_job --help >/dev/null; then
    echo "GATE: lucid.production.dataprod.run_job --help does not parse" >&2
    L3_BAD=1
fi

# run_job --skip-photonsim reads the Geant4 output a previous stage left at
# <output-dir>/output_job_<job-id>.root. data/ is gitignored, so a fresh clone
# has none and this step reports that it did not run rather than passing.
# Either 1 GeV muon file will do: the lucid-data release asset ships a verbatim
# 5-event slice of the 100-event file.
DEMO_ROOT=""
for candidate in data/water/muon/1000MeV_5events.root data/water/muon/1000MeV_100events.root; do
    if [ -f "$candidate" ]; then
        DEMO_ROOT="$candidate"
        break
    fi
done
if [ -n "$DEMO_ROOT" ]; then
    mkdir -p -- "$TMP/run"
    cp -- "$DEMO_ROOT" "$TMP/run/output_job_000001.root"
    if ! "$PY" -m lucid.production.dataprod.run_job --config "$CFG" --detector SK_WAND \
            --output-dir "$TMP/run" --job-id 1 --n-events 2 --master-seed 1 --skip-photonsim \
            --keep-root; then
        echo "GATE: run_job on 2 events failed" >&2
        L3_BAD=1
    fi
else
    echo "GATE: no data/water/muon/1000MeV_{5,100}events.root -- the 2-event run did NOT run." >&2
    echo "GATE: install it with python scripts/download_data.py to cover L3 fully." >&2
fi

IMAGE="${LUCID_IMAGE_PATH:-/sdf/group/neutrino/images/develop.sif}"
if [ -f "$IMAGE" ]; then
    cat > "$TMP/user_paths.sh" <<SH
export CLUSTER="slurm"
export CLUSTER_SUBMIT_CMD="sbatch"
export APPTAINER_BINDS="/sdf"
export LUCID_IMAGE_PATH="$IMAGE"
export OUTPUT_BASE_PATH="$TMP/dataprod"
export SLURM_PARTITION="milano"
export SLURM_ACCOUNT="mli:cider-ml"
export DEFAULT_CPUS="1"
export DEFAULT_MEMORY="4096"
export DEFAULT_GPUS="0"
export DEFAULT_TIME="01:00:00"
SH
    # -D SK_WAND, not SK_like: a config with no "detector" of its
    # own takes its digitizer and trigger from the detector, and only the
    # frozen *_WAND descriptions carry them. -t is the two-event test plan.
    if ! "$PY" -m lucid.production.dataprod.fanout \
            -c "$REPO/$CFG" -D SK_WAND -t --user-paths "$TMP/user_paths.sh"; then
        echo "GATE: dataprod fan-out dry run failed" >&2
        L3_BAD=1
    fi
else
    echo "GATE: no container at $IMAGE -- the fan-out dry run did NOT run." >&2
    echo "GATE: set LUCID_IMAGE_PATH to cover L3 fully." >&2
fi

if [ "$L3_BAD" -ne 0 ]; then
    echo "GATE: L3 failed" >&2
    note_failure 3
fi

if [ "$FAILED_STEP" -ne 0 ]; then
    echo "=== GATE: L$FAILED_STEP failed. Every failure is named above." >&2
    exit "$FAILED_STEP"
fi
echo "=== GATE: L1, L2 and L3 green. L4 is the reviewer."
