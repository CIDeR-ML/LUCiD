#!/usr/bin/env python3
"""Enforce the two shell-script rules (docs/development.md) on the scripts handed to it.

S1: ``set -euo pipefail`` appears in the first 30 lines. Ten of the tracked
scripts had no ``set -e`` family at all and three of those delete data, which
is how a failed ``cd`` goes on to remove the wrong directory.

S5: no personal or site-absolute write target, and no listener on every
interface. The paths are matched as text anywhere in the file -- a read-only
mention is still a path that will not exist on the next machine.

Usage: shell_header_check.py FILE [FILE ...]   (a pre-commit `local` hook)
"""
import re
import sys

HEADER = re.compile(r"^\s*set\s+-euo\s+pipefail\s*$", re.M)
HEAD_LINES = 30

# text -> why it is refused
FORBIDDEN = {
    "/afs/cern.ch": "a CERN AFS path is not a path on this cluster",
    "/eos/": "an EOS path is not a path on this cluster",
    "/sdf/home/": "home is a 30 GB backed-up quota: never stage data or builds there",
    "$HOME/": "no personal state; roots come from user_paths.sh with :?",
    "${HOME}/": "no personal state; roots come from user_paths.sh with :?",
    "--bind 0.0.0.0": "binding every interface exposes the service beyond the node",
}


def check(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        return [f"{path}: unreadable ({exc})"]

    problems = []
    head = "\n".join(text.splitlines()[:HEAD_LINES])
    if not HEADER.search(head):
        problems.append(f"{path}: no 'set -euo pipefail' in the first {HEAD_LINES} lines (S1)")
    for needle, why in FORBIDDEN.items():
        for n, line in enumerate(text.splitlines(), 1):
            if needle in line:
                problems.append(f"{path}:{n}: {needle!r} -- {why} (S5)")
                break
    return problems


def main(argv):
    problems = []
    for path in argv:
        problems.extend(check(path))
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
