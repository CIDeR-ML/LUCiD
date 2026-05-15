#!/usr/bin/env python3
"""Identify SIREN-input sub-jobs that did not finish.

Walks a scan output tree, finds every `submit*.sbatch` whose matching
`output_job_NNNNNN.root` is missing OR doesn't have the `OpticalPhotons`
TTree key (the truth marker — preempted jobs never reach
DataManager::Finalize() so the TTree directory entry is never written
even though basket bytes may be on disk).

Two modes:

  default  — human-readable summary on stderr, sample paths on stdout
  --list   — machine-readable: one missing sbatch path per line on stdout,
             nothing on stderr unless an error. Pipe into `xargs sbatch`.

Requires `uproot`. The submit step needs `sbatch` (host PATH); since uproot
lives inside the container and sbatch lives on the host, the recommended
driver is the sibling `resubmit_failed.sh` wrapper which bridges the two.
"""

import argparse
import re
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
    """True iff the ROOT file exists AND has the OpticalPhotons TTree key."""
    if not root_path.is_file():
        return False
    try:
        with uproot.open(root_path) as f:
            return "OpticalPhotons" in f
    except Exception:
        return False


def find_missing(scan_dir: Path) -> List[Path]:
    sbatches = sorted(scan_dir.rglob("submit*.sbatch"))
    return [sb for sb in sbatches if not is_complete(expected_root(sb))]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("scan_dir", type=Path,
                   help="Scan output dir (containing <material>/<particle>/<E>MeV/).")
    p.add_argument("--list", action="store_true",
                   help="Machine-readable: print one missing sbatch path per "
                        "line on stdout. Pipe to `xargs sbatch`.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only emit the first N missing sbatches.")
    args = p.parse_args()

    if not args.scan_dir.is_dir():
        print(f"error: scan dir not found: {args.scan_dir}", file=sys.stderr)
        return 2

    sbatches = sorted(args.scan_dir.rglob("submit*.sbatch"))
    if not sbatches:
        print(f"error: no submit*.sbatch under {args.scan_dir}", file=sys.stderr)
        return 1

    missing = [sb for sb in sbatches if not is_complete(expected_root(sb))]
    if args.limit > 0:
        missing = missing[: args.limit]

    if args.list:
        for sb in missing:
            print(sb)
        return 0

    # Human-readable mode.
    print(f"scan dir:            {args.scan_dir}")
    print(f"truth check:         OpticalPhotons key in output_job_*.root")
    print(f"total sbatches:      {len(sbatches):,}")
    print(f"complete:            {len(sbatches) - len(missing):,}")
    print(f"missing/partial:     {len(missing):,}")
    print("")
    if not missing:
        print("Nothing to resubmit. Run merge.sh next.")
        return 0
    print("first 10 missing:")
    for sb in missing[:10]:
        print(f"  {sb.relative_to(args.scan_dir)}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    print("")
    print("To resubmit, run the wrapper:  ./resubmit_failed.sh <scan_dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
