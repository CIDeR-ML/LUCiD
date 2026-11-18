#!/usr/bin/env python3
"""Refuse binary assets in the repository.

`.gitignore` has been bypassed by force-adds before -- the GIFs under
`good_notebooks/figures/` are 13 MB of it -- so the extension list is enforced
by a hook rather than by a pattern a `git add -f` can step over. A render
belongs in CI output or in the docs build, not in git history, where it is
permanent.

Usage: no_binary_extensions.py FILE [FILE ...]   (a pre-commit `local` hook)
"""
import sys
from pathlib import PurePosixPath

REFUSED = {
    ".gif", ".mp4", ".mov", ".avi", ".webm",            # renders
    ".so", ".dylib", ".dll", ".a", ".o", ".pyd",        # build output
    ".whl", ".jar", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",  # archives
    ".pdf", ".pyc", ".pkl", ".pickle", ".db", ".sqlite",
}

# Shipped data the project genuinely carries. Keep this short and specific:
# every entry is a promise that the file is an input, not an artefact.
ALLOWED = (
    "config/",                    # detector geometry npz / geofiles
    "lucid/resources/",           # packaged resources
    "landing/assets/",            # the project page's own images
    "docs/assets/",
)


def main(argv):
    bad = []
    for name in argv:
        path = PurePosixPath(name)
        if path.suffix.lower() not in REFUSED:
            continue
        if any(name.startswith(prefix) for prefix in ALLOWED):
            continue
        bad.append(f"{name}: {path.suffix} is a binary artefact; git history is permanent")
    for line in bad:
        print(line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
