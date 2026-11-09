#!/usr/bin/env python3
"""Fail if a Markdown page points at a repository file that does not exist.

Checks every tracked ``*.md`` outside ``.tidy/``, for three kinds of reference:

- a Markdown link or image, ``[text](target)``, whose target is a path rather than a URL; it
  resolves against the page's directory, and a ``#fragment`` is ignored;
- a ``--8<-- "path"`` or ``--8<-- "path:section"`` snippet include; the path resolves against the
  repo root, and a section must be marked ``[start:section]`` in the file;
- a repository path in backticks, ``dir/file.ext``: at least one ``/``, a file extension from
  ``EXTENSIONS``, no placeholder (``<x>``, ``*``, ``$X``, ``NAME``, ``NNNN``). It must name a
  tracked file or a file of the ``lucid-data`` release (``data/MANIFEST.sha256``): relative to the
  page's directory, to the repo root (a leading ``LUCiD/`` is the root), or as the tail of one
  (``platform/adapters.py``, ``GeV/02_mu.json``). Not checked: a gitignored path (generated, such
  as ``user_paths.sh``), a path into another repository (``PhotonSim/...``), and the pages in
  ``NO_PATH_CHECK``, whose job is to name files that are not in the tree.

Run from the repo root; exits 1 with one line per broken reference.
"""
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

LINK = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)\)')
SNIPPET = re.compile(r'--8<--\s+"([^":]+)(?::([\w-]+))?"')
BACKTICK = re.compile(r'`([\w.-]+(?:/[\w.-]+)+\.([A-Za-z]\w*))`')
EXTENSIONS = {'py', 'md', 'json', 'sh', 'yml', 'yaml', 'toml', 'txt', 'cff', 'npz', 'h5', 'root',
              'ipynb', 'html', 'css', 'js', 'sbatch', 'template', 'manifest', 'sha256', 'png',
              'pdf', 'csv', 'gz', 'mac'}
PLACEHOLDER = re.compile(r'NAME|N{3,}|X{3,}')
EXTERNAL = ('PhotonSim/',)
NO_PATH_CHECK = {'THIRD_PARTY.md'}      # the third-party inventory, history included


def tracked():
    out = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    return {f for f in out if f}


def release_files():
    """The files ``scripts/download_data.py`` installs under ``data/``."""
    rows = Path('data/MANIFEST.sha256').read_text().splitlines()
    return {'data/' + row.split()[1] for row in rows if row.strip() and not row.startswith('#')}


def norm(path):
    """``path`` with ``.`` and ``..`` folded, as a repo-relative POSIX string (None above it)."""
    parts = []
    for p in PurePosixPath(path).parts:
        if p == '..':
            if not parts:
                return None
            parts.pop()
        elif p != '.':
            parts.append(p)
    return '/'.join(parts)


def ignored(paths):
    """The subset of ``paths`` that .gitignore covers."""
    if not paths:
        return set()
    run = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], input='\n'.join(paths),
                         stdout=subprocess.PIPE, universal_newlines=True)
    return set(run.stdout.split())


def main():
    files = tracked()
    known = files | release_files()
    tails = {'/'.join(f.split('/')[i:]) for f in known for i in range(1, f.count('/') + 1)}
    pages = sorted(f for f in files if f.endswith('.md') and not f.startswith('.tidy/'))
    bad = []
    for name in pages:
        text = Path(name).read_text()
        here = PurePosixPath(name).parent
        for t in LINK.findall(text):
            if not re.match(r'[a-z]+:|#', t) and not Path(here / t.split('#')[0]).exists():
                bad.append(f'{name}: link {t}')
        for path, section in SNIPPET.findall(text):
            if not Path(path).exists():
                bad.append(f'{name}: include {path}')
            elif section and f'[start:{section}]' not in Path(path).read_text():
                bad.append(f'{name}: include {path}:{section} (no such section)')
        if name in NO_PATH_CHECK:
            continue
        missing = []
        for t, ext in sorted(set(BACKTICK.findall(text))):
            if ext not in EXTENSIONS or PLACEHOLDER.search(t) or t.startswith(EXTERNAL):
                continue
            root = t[len('LUCiD/'):] if t.startswith('LUCiD/') else t
            if not {norm(here / t), norm(root)} & known and norm(root) not in tails:
                missing.append((t, {c for c in (norm(here / t), norm(root)) if c}))
        skip = ignored(sorted(set().union(*(c for _, c in missing))))
        bad += [f'{name}: path `{t}`' for t, c in missing if not c & skip]
    for line in bad:
        print(line)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
