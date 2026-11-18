"""Source `user_paths.sh` and read its exported env.

Shared by every generator under lucid/production.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Dict, List


def load_user_paths(path: Path) -> Dict[str, str]:
    """Source the shell file in a subshell and capture its env.

    Raises FileNotFoundError if `path` doesn't exist (with a hint about the
    .template sibling), and CalledProcessError if the shell exits non-zero
    while sourcing (typically a syntax error in user_paths.sh).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"user_paths.sh not found at {path}. "
            f"Copy one of {Path(__file__).parent / 'templates'}/user_paths.*.sh.template "
            f"there and configure it."
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


def missing_settings(env: Dict[str, str], *keys: str) -> List[str]:
    """The ``keys`` a stage needs that ``env`` (from :func:`load_user_paths`) does not set, plus
    ``SLURM_ACCOUNT`` on a Slurm cluster (every Slurm submit file charges it)."""
    need = list(keys)
    if (env.get("CLUSTER") or "slurm").lower() in ("slurm", "nersc"):
        need.append("SLURM_ACCOUNT")
    return [k for k in need if not env.get(k)]
