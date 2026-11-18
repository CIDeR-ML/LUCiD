#!/usr/bin/env python3
"""Identify SIREN-input sub-jobs that did not finish.

Walks a scan output tree, finds every `submit*.sbatch` whose matching
`output_job_NNNNNN.root` is missing OR doesn't have the `OpticalPhotons`
TTree key (the truth marker — preempted jobs never reach
DataManager::Finalize() so the TTree directory entry is never written
even though basket bytes may be on disk).

Two modes:

  default  — human-readable summary on stderr, sample paths on stdout
  --list   — one `[PREPARED] <sbatch>` line per missing sub-job on stdout.

Requires `uproot`, which lives in the container, while `sbatch` lives on the
host; `platform/run_in_container.sh` bridges the two and submits the listed
sub-jobs with -s:

    lucid/production/platform/run_in_container.sh \
        lucid.production.siren_inputs.resubmit_failed <scan_dir> --list -s
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

from lucid.production.siren_inputs.siren_planning import is_complete_siren as is_complete

# submit.{sbatch|sub}        -> job_id = 1
# submit_job_NNN.{sbatch|sub} -> job_id = int(NNN)
SUBMIT_RE = re.compile(r"^submit(?:_job_(\d{3}))?\.(?:sbatch|sub)$")


def job_id_from_submit(name: str) -> Optional[int]:
    m = SUBMIT_RE.match(name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def expected_root(submit_path: Path) -> Path:
    jid = job_id_from_submit(submit_path.name)
    return submit_path.parent / f"output_job_{jid:06d}.root"


def _iter_submits(scan_dir: Path) -> List[Path]:
    submits = sorted(list(scan_dir.rglob("submit*.sbatch"))
                     + list(scan_dir.rglob("submit*.sub")))
    return submits


def find_missing(scan_dir: Path) -> List[Path]:
    return [sb for sb in _iter_submits(scan_dir)
            if not is_complete(expected_root(sb))]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("scan_dir", type=Path,
                   help="Scan output dir (containing <material>/<particle>/<E>MeV/).")
    p.add_argument("--list", action="store_true",
                   help="Print one `[PREPARED] <sbatch>` line per missing "
                        "sub-job, for platform/run_in_container.sh -s to submit.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only emit the first N missing sbatches.")
    args = p.parse_args()

    if not args.scan_dir.is_dir():
        print(f"error: scan dir not found: {args.scan_dir}", file=sys.stderr)
        return 2

    submits = _iter_submits(args.scan_dir)
    if not submits:
        print(f"error: no submit*.{{sbatch,sub}} under {args.scan_dir}",
              file=sys.stderr)
        return 1

    missing = [sb for sb in submits if not is_complete(expected_root(sb))]
    if args.limit > 0:
        missing = missing[: args.limit]

    if args.list:
        for sb in missing:
            print(f"[PREPARED] {sb}")
        return 0

    # Human-readable mode.
    print(f"scan dir:            {args.scan_dir}")
    print(f"truth check:         OpticalPhotons key in output_job_*.root")
    print(f"total submits:       {len(submits):,}")
    print(f"complete:            {len(submits) - len(missing):,}")
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
    print("To resubmit:  lucid/production/platform/run_in_container.sh "
          "lucid.production.siren_inputs.resubmit_failed <scan_dir> --list -s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
