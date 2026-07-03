#!/bin/bash
# De-pile a v3 pile-up dataset into single-interaction events via a SLURM array.
# TEMPLATE: edit SRC / DST / --array to point at your dataset. Each array task owns
# a disjoint file-index range, so the whole thing is resumable — just resubmit and
# already-finished files are skipped (depile.py is per-file, atomic .part->rename).
#
# The example below de-piles config_000018 (2x particle-bomb) -> config_000018_separated.
#SBATCH --partition=roma
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=depile18
#SBATCH --array=0-99
#SBATCH --output=/sdf/data/neutrino/cjesus/DORAEMON/WAND/latest/SK_like/config_000018_separated/logs/task_%a-%A.out
#SBATCH --error=/sdf/data/neutrino/cjesus/DORAEMON/WAND/latest/SK_like/config_000018_separated/logs/task_%a-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=03:00:00
set -eu -o pipefail

IMG=/sdf/data/neutrino/cjesus/software/images/lucid.sif
SCRIPT=/sdf/home/c/cjesus/REFACTORED/LUCiD/lucid/production/derived_views/depile.py
SRC=/sdf/data/neutrino/cjesus/DORAEMON/WAND/latest/SK_like/config_000018
DST=/sdf/data/neutrino/cjesus/DORAEMON/WAND/latest/SK_like/config_000018_separated

PER_TASK=100                       # 100 tasks x 100 files = 10000 files (indices 0..9999)
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK - 1 ))

echo "task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID}) on $(hostname): indices ${START}..${END}  $(date)"
for i in $(seq "$START" "$END"); do
    # depile is per-file, atomic (.part->rename) and resumable (skips matching dest).
    # Each task owns a disjoint index range -> no cross-job file clashes.
    apptainer exec -B /sdf,/lscratch "$IMG" python3 "$SCRIPT" "$SRC" "$DST" "$i" \
        || echo "WARN: index $i failed (will be retried on resubmit)"
done
echo "task ${SLURM_ARRAY_TASK_ID} done $(date)"
