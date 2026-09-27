#!/usr/bin/env python3
"""Turn a stage's log into a measured percentage and ETA.

Every long stage in this pipeline prints a line carrying the index it is working
on -- ``fitpdf`` prints ``k=150: Observed q=...``, one per q bin -- so an ETA does
not need new instrumentation, only the total and the file's age. Quoting a
deadline (a queue's walltime cap) as if it were an ETA, which is easy to do by
accident, says nothing about whether the work will finish.

Stages emit ``PROGRESS <i>/<n>`` (flushed), so the total travels with the
progress and nothing has to be told how long a loop is:

    ./eta.py --log s_fit0.log                 # any stage, no arguments
    ./eta.py --log a.log --log b.log          # several at once

The rate is measured between the log's first write and its last, so it is the
stage's own throughput rather than an assumption. With ``--watch`` it reprints
until the stage finishes.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


def total_from_root(path, hist="hst2d_type0") -> int:
    """q-bin count straight from the 2D charge PDF, so the total is never guessed."""
    import uproot
    with uproot.open(path) as f:
        return int(f[hist].axis(1).traits.__class__ and len(f[hist].axis(1).edges()) - 1)


PROGRESS_RX = re.compile(r"PROGRESS\s+(\d+)\s*/\s*(\d+)(?:\s+t=([0-9.]+))?")


def progress(log: Path, pattern: str):
    """``(index, total, first_mtime, last_mtime)``.

    A ``PROGRESS i/n`` line wins, because it carries its own total. ``pattern``
    is the fallback for a stage not yet emitting one.
    """
    try:
        text = log.read_text(errors="ignore")
    except OSError:
        return None, None, None, None
    st = log.stat()
    hits = PROGRESS_RX.findall(text)
    if hits:
        idx = max(int(i) for i, _, _ in hits)
        total = int(hits[-1][1])
        stamps = [float(t) for _, _, t in hits if t]
        if len(stamps) >= 2:
            # Rate from the line's own clock: independent of the filesystem.
            return idx, total, min(stamps), max(stamps)
        return idx, total, getattr(st, "st_ctime", None), st.st_mtime
    rx = re.compile(pattern)
    best = None
    try:
        text = log.read_text(errors="ignore")
    except OSError:
        return None, None, None
    for m in rx.finditer(text):
        v = int(m.group(1))
        best = v if best is None else max(best, v)
    return best, None, getattr(st, "st_ctime", None), st.st_mtime


def report(log: Path, pattern: str, total=None) -> bool:
    idx, found_total, started, last = progress(log, pattern)
    total = found_total or total
    if idx is None or not total:
        if not log.exists():
            print(f"  {log.name}: not created yet")
        else:
            # An AFS size can be stale across machines; say so rather than imply
            # the stage is hung.
            age = time.time() - log.stat().st_mtime
            print(f"  {log.name}: no progress line yet (last write {age:.0f}s ago; "
                  f"on AFS this can be a stale cached size -- try `fs flush`)")
        return False
    elapsed = max(last - started, 1e-6)
    frac = idx / total
    rate = idx / elapsed
    remaining = (total - idx) / rate if rate > 0 else float("inf")
    bar_n = int(28 * frac)
    print(f"  [{'#'*bar_n}{'.'*(28-bar_n)}] {100*frac:5.1f}%  {idx}/{total}  "
          f"{rate*60:.1f}/min  elapsed {elapsed/60:.1f} min  "
          f"ETA {remaining/60:.1f} min")
    return idx >= total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True, action="append",
                   help="repeatable")
    p.add_argument("--pattern", default=None,
                   help="fallback regex for a stage not emitting PROGRESS; "
                        "it has no clock, so its rate is unreliable")
    p.add_argument("--total", type=int, default=None)
    p.add_argument("--total-from", type=Path, default=None,
                   help="pdf2d.root to read the q-bin count from")
    p.add_argument("--hist", default="hst2d_type0")
    a = p.parse_args(argv)

    total = a.total
    if total is None and a.total_from:
        total = total_from_root(a.total_from, a.hist)
    for log in a.log:
        print(f"{log}")
        report(log, a.pattern or r"(?!)", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
