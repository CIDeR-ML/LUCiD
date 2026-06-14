"""Source `user_paths.sh` and read its exported env.

Single canonical implementation. Replaces the three near-identical copies
that used to live in jobs/{siren_inputs,smax,train_siren}/generate_jobs.py.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Dict


def load_user_paths(path: Path) -> Dict[str, str]:
    """Source the shell file in a subshell and capture its env.

    Raises FileNotFoundError if `path` doesn't exist (with a hint about the
    .template sibling), and CalledProcessError if the shell exits non-zero
    while sourcing (typically a syntax error in user_paths.sh).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"user_paths.sh not found at {path}. "
            f"Copy {path.parent / 'user_paths.sh.template'} and configure it."
        )
    proc = subprocess.run(
        ["bash", "-c", f"set -e; source {shlex.quote(str(path))} && env"],
        capture_output=True, text=True, check=True,
    )
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out
