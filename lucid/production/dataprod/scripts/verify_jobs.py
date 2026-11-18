#!/usr/bin/env python3
"""Identify dataprod jobs that did not finish.

Walks a dataprod output tree and finds every `submit_job_NNNNNN.sbatch`
whose batch (`file_index = job_id - 1`) fails `lucid.dataset.layout.check_batch`
-- the same check every job runs on its own output -- or holds more events than
the job requested (fewer is selection, not a failure).

This is the dataset-prod analog of `siren_inputs/resubmit_failed.py`,
but datasets don't need merging — each `file_index` batch is an
independent shard.

Two modes:

  default  — human-readable summary on stderr, sample paths on stdout
  --list   — one `[PREPARED] <sbatch>` line per failed job on stdout.

Requires h5py, which lives in the container, while `sbatch` lives on the
host; `platform/run_in_container.sh` bridges the two and submits the
listed jobs with -s:

    lucid/production/platform/run_in_container.sh \
        lucid.production.dataprod.scripts.verify_jobs <scan_dir> --list -s
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from lucid.dataset.layout import check_batch


def is_complete(cell_dir: Path, file_index: int,
                expected_events: Optional[int]) -> Tuple[bool, str]:
    """True iff batch ``file_index`` passes :func:`lucid.dataset.layout.check_batch` with at most
    ``expected_events`` events (fewer is selection, not a failure).

    Returns (ok, short_reason). `short_reason` is "" when ok, else the first failure line.
    """
    ok, messages = check_batch(cell_dir, file_index, max_events=expected_events)
    if ok:
        return True, ""
    return False, next((m for m in messages if not m.startswith(("OK", "EMPTY_KEPT"))),
                       "failed the batch check")


# submit_job_NNNNNN.{sbatch|sub}  -> job_id = int(NNNNNN)
SUBMIT_RE = re.compile(r"^submit_job_(\d+)\.(?:sbatch|sub)$")


def job_id_from_submit(name: str) -> Optional[int]:
    m = SUBMIT_RE.match(name)
    return int(m.group(1)) if m else None


def _read_sbatch(sbatch: Path) -> str:
    try:
        return sbatch.read_text()
    except OSError:
        return ""


def find_config_json_for(sbatch: Path) -> Optional[Path]:
    """Locate the dataprod JSON config that produced this sbatch.

    The fan-out writes the absolute path to the config into the
    sbatch body as `--config "<path>"`. We grep that out so we know
    the expected `n_events_per_job`.
    """
    m = re.search(r'--config\s+"?([^"\s\\]+\.json)"?', _read_sbatch(sbatch))
    if not m:
        return None
    p = Path(m.group(1))
    return p if p.is_file() else None


def expected_n_events(sbatch: Path, override: Optional[int]) -> Optional[int]:
    """Per-job expected event count.

    Priority (highest first):
      1. CLI override.
      2. `--n-events N` baked into the sbatch — what `events_schedule`
         injects when the fan-out splits a config by time.
      3. `n_events_per_job` from the config JSON referenced by --config.
    """
    if override is not None:
        return override
    text = _read_sbatch(sbatch)
    m = re.search(r'--n-events\s+(\d+)\b', text)
    if m:
        return int(m.group(1))
    cfg_path = find_config_json_for(sbatch)
    if cfg_path is None:
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        return int(cfg["n_events_per_job"])
    except (OSError, KeyError, ValueError):
        return None


def iter_sbatches(scan_dir: Path) -> Iterator[Path]:
    """Every submit_job_*.{sbatch,sub} under the scan dir, sorted."""
    submits = (list(scan_dir.rglob("submit_job_*.sbatch"))
               + list(scan_dir.rglob("submit_job_*.sub")))
    yield from sorted(submits)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "scan_dir", type=Path,
        help="Dataprod output root (e.g. $OUTPUT_BASE_PATH/SK_like). "
             "Walked recursively for submit_job_*.sbatch.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print one `[PREPARED] <sbatch>` line per failed job, for "
             "platform/run_in_container.sh -s to submit.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="If >0, only emit the first N failed sbatches.",
    )
    p.add_argument(
        "--expected-n-events", type=int, default=None,
        help="Override expected n_events_per_job (default: read it from the "
             "config JSON referenced by each sbatch).",
    )
    args = p.parse_args()

    if not args.scan_dir.is_dir():
        print(f"error: scan dir not found: {args.scan_dir}", file=sys.stderr)
        return 2

    sbatches = list(iter_sbatches(args.scan_dir))
    if not sbatches:
        print(f"error: no submit_job_*.sbatch under {args.scan_dir}",
              file=sys.stderr)
        return 1

    failed: List[Tuple[Path, str]] = []
    for sb in sbatches:
        jid = job_id_from_submit(sb.name)
        if jid is None:
            continue
        file_index = jid - 1
        expected = expected_n_events(sb, args.expected_n_events)
        ok, reason = is_complete(sb.parent, file_index, expected)
        if not ok:
            failed.append((sb, reason))

    if args.limit > 0:
        failed = failed[: args.limit]

    if args.list:
        for sb, _ in failed:
            print(f"[PREPARED] {sb}")
        return 0

    print(f"scan dir:         {args.scan_dir}")
    print(f"truth check:      4× wc_*.h5 exist AND config/n_events matches")
    print(f"total sbatches:   {len(sbatches):,}")
    print(f"complete:         {len(sbatches) - len(failed):,}")
    print(f"failed/missing:   {len(failed):,}")
    print("")
    if not failed:
        print("Nothing to resubmit.")
        return 0
    print("first 10 failed:")
    for sb, reason in failed[:10]:
        print(f"  {sb.relative_to(args.scan_dir)}  [{reason}]")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")
    print("")
    print("To resubmit:  lucid/production/platform/run_in_container.sh "
          "lucid.production.dataprod.scripts.verify_jobs <scan_dir> --list -s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
