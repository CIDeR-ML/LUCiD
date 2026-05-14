#!/usr/bin/env python3
"""Resubmit SIREN-input sub-jobs that did not finish.

Walks the scan output tree, finds every `submit*.sbatch` whose matching
`output_job_NNNNNN.root` is missing OR doesn't have the `OpticalPhotons`
TTree key (the truth marker — preempted jobs never reach
DataManager::Finalize() so the TTree directory entry is never written
even though basket bytes may be on disk), and re-submits just those
sbatches via `sbatch`.

Idempotent — run it after each drain wave until the "missing" count is
zero. Safe to point at a partially-finished run.

Requires `uproot`; run inside the unified container:

  apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch \
      /sdf/data/neutrino/cjesus/software/images/lucid.sif \
      python3 resubmit_failed.py <scan_output_dir> [-s]
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import uproot

# submit.sbatch  -> job_id = 1
# submit_job_NNN.sbatch -> job_id = int(NNN)
SBATCH_RE = re.compile(r"^submit(?:_job_(\d{3}))?\.sbatch$")


def job_id_from_sbatch(name: str) -> Optional[int]:
    m = SBATCH_RE.match(name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def expected_root(sbatch: Path) -> Path:
    jid = job_id_from_sbatch(sbatch.name)
    return sbatch.parent / f"output_job_{jid:06d}.root"


def is_complete(root_path: Path) -> bool:
    """True iff the ROOT file exists AND has the OpticalPhotons TTree key.

    Preempted jobs may leave file bytes on disk (basket flushes), but the
    directory entries — including OpticalPhotons — are only written by
    DataManager::Finalize() at clean exit. So the key's presence is the
    binary truth marker.
    """
    if not root_path.is_file():
        return False
    try:
        with uproot.open(root_path) as f:
            return "OpticalPhotons" in f
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("scan_dir", type=Path,
                   help="Scan output dir (containing <material>/<particle>/<E>MeV/).")
    p.add_argument("-s", "--submit", action="store_true",
                   help="Actually re-submit (default: dry-run report).")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only resubmit the first N missing sbatches "
                        "(for staggered resubmission to avoid slurmctld churn).")
    args = p.parse_args()

    if not args.scan_dir.is_dir():
        print(f"error: scan dir not found: {args.scan_dir}", file=sys.stderr)
        return 2
    if args.submit and shutil.which("sbatch") is None:
        print("error: sbatch not on PATH.", file=sys.stderr)
        return 2

    sbatches = sorted(args.scan_dir.rglob("submit*.sbatch"))
    if not sbatches:
        print(f"error: no submit*.sbatch under {args.scan_dir}", file=sys.stderr)
        return 1

    complete = missing_or_partial = 0
    missing = []  # type: List[Path]
    for sb in sbatches:
        out = expected_root(sb)
        if is_complete(out):
            complete += 1
        else:
            missing_or_partial += 1
            missing.append(sb)

    print(f"scan dir:            {args.scan_dir}")
    print(f"truth check:         OpticalPhotons key in output_job_*.root")
    print(f"total sbatches:      {len(sbatches):,}")
    print(f"complete:            {complete:,}")
    print(f"missing/partial:     {missing_or_partial:,}")
    print("")

    if not missing:
        print("Nothing to resubmit. Run merge.sh next.")
        return 0

    if args.limit > 0:
        missing = missing[: args.limit]
        print(f"--limit {args.limit}: only resubmitting the first "
              f"{len(missing)} missing sbatches.")

    if not args.submit:
        print("dry-run; would resubmit (first 10):")
        for sb in missing[:10]:
            rel = sb.relative_to(args.scan_dir)
            print(f"  {rel}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print("")
        print("Re-run with -s to actually submit.")
        return 0

    ok = fail = 0
    for sb in missing:
        r = subprocess.run(["sbatch", str(sb)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        if r.returncode != 0:
            print(f"  FAILED: {sb.relative_to(args.scan_dir)}: "
                  f"{r.stderr.strip()}", file=sys.stderr)
            fail += 1
        else:
            ok += 1
    print(f"resubmitted: {ok}  failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
