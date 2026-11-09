"""What a paper run records about the code that produced it."""
import json
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = ('jax', 'jaxlib', 'numpy', 'optax')


def run_provenance():
    """The commit, whether the tree had uncommitted changes, and the package versions."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO_ROOT), *args], text=True).strip()
    return dict(git_commit=git('rev-parse', 'HEAD'),
                git_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
                versions=json.dumps({p: version(p) for p in _PACKAGES}))


def write_run_record(directory, script, args):
    """Write ``<directory>/<script>.json``: a figure script's arguments and :func:`run_provenance`.

    Called after each step with the directory the step wrote to, so a figure or cache that is
    copied elsewhere can be traced to the command and the commit that made it.
    """
    name = Path(script).stem
    record = dict(script=name, args=vars(args), written=datetime.now().isoformat(),
                  **run_provenance())
    path = Path(directory) / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + '\n')
    return path
