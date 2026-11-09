#!/usr/bin/env python3
"""Fail if a tracked file exceeds 1 MB or a tracked notebook carries outputs.

The repo has twice grown artefacts `.gitignore` was supposed to stop — a 32 MB
notebook with 18 MB of inlined plotly, and force-added GIFs — so the check is
on the tracked set, not on the working tree. Run from the repo root; exits 1
with one line per offender.
"""
import json
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1024 * 1024


def tracked(pattern=None):
    cmd = ["git", "ls-files", "-z"] + ([pattern] if pattern else [])
    out = subprocess.check_output(cmd).decode()
    return [p for p in out.split("\0") if p]


def main():
    bad = []
    for name in tracked():
        path = Path(name)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            bad.append(f"{name}: {path.stat().st_size / 1024:.0f} kB > 1024 kB")
    for name in tracked("*.ipynb"):
        try:
            doc = json.loads(Path(name).read_text())
        except (OSError, ValueError) as exc:
            bad.append(f"{name}: unreadable notebook ({exc})")
            continue
        for i, cell in enumerate(doc.get("cells", [])):
            if cell.get("outputs") or cell.get("execution_count") is not None:
                bad.append(f"{name}: cell {i} carries outputs or an execution count")
    for line in bad:
        print(line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
