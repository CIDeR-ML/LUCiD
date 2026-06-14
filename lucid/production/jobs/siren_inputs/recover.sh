#!/bin/bash
# Validate SIREN-input pack/array outputs and resubmit only the failed units.
#
# Output-driven recovery for --pack / --array runs. Those modes don't write a
# per-job submit*.sbatch per unit, so resubmit_failed.py (which enumerates
# submit*.sbatch) can't drive recovery for them. Instead this works off the
# OUTPUTS, bridging the container/host split the same way resubmit_failed.sh
# does (uproot lives in the container, sbatch on the host):
#
#   Phase 1 (container): walk this config's output_job_*.root and delete any
#            that are unreadable or lack the OpticalPhotons truth key (a unit
#            that crashed/was preempted mid-run never reaches Finalize(), so the
#            key is absent even if the file exists).
#   Phase 2 (host): re-run generate_jobs.py in --pack/--array mode. Its
#            skip-on-existence logic now sees the deleted units as missing and
#            resubmits ONLY those (plus any never-created).
#
# Idempotent: run after each drain wave until "0 invalid", then merge.sh.
# For per-job (single) mode on S3DF/LXPLUS use resubmit_failed.sh instead.
#
# Usage:
#   ./recover.sh -c <config.json> [--pack [-n N] | --array] [--dry-run] [--force]
#     --dry-run  validate + list invalid outputs, delete/submit nothing
#     -n N       pack bins for the resubmit (default 1: one node drains the lot)
#     --force    proceed even if siren jobs are still queued (unsafe mid-wave)

set -eu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_PATHS="${SCRIPT_DIR}/../user_paths.sh"
[ -f "${USER_PATHS}" ] || { echo "error: ${USER_PATHS} not found." >&2; exit 1; }
# shellcheck source=/dev/null
source "${USER_PATHS}"

CONFIG=""; MODE="--pack"; PACK_BINS=1; DRY=0; FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config) CONFIG="$2"; shift 2 ;;
        --pack)      MODE="--pack"; shift ;;
        --array)     MODE="--array"; shift ;;
        -n)          PACK_BINS="$2"; shift 2 ;;
        --dry-run)   DRY=1; shift ;;
        --force)     FORCE=1; shift ;;
        *) echo "error: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "${CONFIG}" ] || { echo "usage: $0 -c <config.json> [--pack [-n N] | --array] [--dry-run] [--force]" >&2; exit 2; }
[ -f "${CONFIG}" ] || { echo "error: config not found: ${CONFIG}" >&2; exit 1; }
[ -n "${LUCID_IMAGE_PATH:-}" ] || { echo "error: LUCID_IMAGE_PATH not set in user_paths.sh." >&2; exit 1; }
[ -n "${SIREN_OUTPUT_BASE_PATH:-}" ] || { echo "error: SIREN_OUTPUT_BASE_PATH not set." >&2; exit 1; }

OUT_BASE="${SIREN_OUTPUT_BASE_PATH}/training_inputs"
APPTAINER="${APPTAINER_BIN:-apptainer}"
# Prefer a modern host python for the resubmit (the bare python3 may be too old).
PYBIN="$(command -v python3.11 || command -v python3)"

# dev-loop binds so the container's uproot validation + lucid-run-job pick up
# the host checkouts (the truth check imports lucid.production.cluster_common).
DEV_BINDS=""
[ -n "${LUCID_DEV_PATH:-}" ]     && DEV_BINDS="${DEV_BINDS} -B ${LUCID_DEV_PATH}:/opt/LUCiD"
[ -n "${PHOTONSIM_DEV_PATH:-}" ] && DEV_BINDS="${DEV_BINDS} -B ${PHOTONSIM_DEV_PATH}:/opt/PhotonSim"

# Safety: deleting outputs while jobs are still writing them would lose data.
ACTIVE=$(squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -c '^siren' || true)
if [ "${DRY}" -eq 0 ] && [ "${ACTIVE}" -gt 0 ] && [ "${FORCE}" -eq 0 ]; then
    echo "error: ${ACTIVE} siren job(s) still in the queue — recovering now could delete" >&2
    echo "       outputs that running jobs are still writing. Wait for the wave to drain" >&2
    echo "       (squeue -u \$USER), or pass --force to override." >&2
    exit 1
fi

echo "config:    ${CONFIG}"
echo "scan base: ${OUT_BASE}"
echo "mode:      ${MODE}$( [ "${MODE}" = "--pack" ] && echo " (bins=${PACK_BINS})" )"
echo "dry-run:   ${DRY}"
echo ""

# --- Phase 1: validate (and unless --dry-run, delete) invalid outputs ---------
"${APPTAINER}" exec -B "${APPTAINER_BINDS:-/global/cfs,/global/homes,/global/u1,/pscratch,/dvs_ro,/cvmfs}" ${DEV_BINDS} \
    "${LUCID_IMAGE_PATH}" \
    python3 - "${CONFIG}" "${OUT_BASE}" "${DRY}" <<'PYEOF'
import sys, os, json, glob
sys.path.insert(0, "/opt/LUCiD")
from pathlib import Path
from lucid.production.cluster_common.verify import is_complete_siren

config_path, out_base, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
cfg = json.load(open(config_path))
material = cfg["material"]
particles = [p["type"] for p in cfg["particles"]]

bad, total = [], 0
for particle in particles:
    base = os.path.join(out_base, material, particle)
    for f in sorted(glob.glob(os.path.join(base, "*MeV", "output_job_*.root"))):
        total += 1
        if not is_complete_siren(Path(f)):
            bad.append(f)

print(f"checked {total} output ROOTs under {material}/{particles}")
print(f"invalid (missing OpticalPhotons / unreadable): {len(bad)}")
for b in bad[:30]:
    print("  INVALID:", b)
if len(bad) > 30:
    print(f"  ... and {len(bad) - 30} more")

if dry:
    print("\n--dry-run: nothing deleted. Re-run without --dry-run to delete + resubmit.")
elif bad:
    for b in bad:
        os.remove(b)
    print(f"\ndeleted {len(bad)} invalid output ROOTs (will be regenerated).")
else:
    print("\nAll present outputs valid.")
PYEOF

if [ "${DRY}" -eq 1 ]; then
    echo ""
    echo "Dry run complete — no deletion or resubmission."
    exit 0
fi

# --- Phase 2: resubmit the now-missing units (deleted-invalid + never-created)-
echo ""
echo "=== resubmitting missing units (${MODE}) ==="
if [ "${MODE}" = "--pack" ]; then
    "${PYBIN}" "${SCRIPT_DIR}/generate_jobs.py" -c "${CONFIG}" --pack "${PACK_BINS}" -s
else
    "${PYBIN}" "${SCRIPT_DIR}/generate_jobs.py" -c "${CONFIG}" --array -s
fi
