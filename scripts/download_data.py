#!/usr/bin/env python3
"""Install LUCiD's example data, the ``lucid-data`` release asset, into ``data/``.

    python scripts/download_data.py                  # fetch it from the GitHub release
    python scripts/download_data.py --from-dir DIR   # offline, or a CI cache: DIR/<asset>
    python scripts/download_data.py --verify-only    # check data/ against data/MANIFEST.sha256
    python scripts/download_data.py --paper          # the paper's results instead (below)

``--paper`` installs the ``lucid-paper-results`` asset instead: the reconstructed scans and the
other cached inputs the paper's figures draw, checked against ``analysis/paper/RESULTS.sha256``,
into ``analysis/paper/output/data/`` (where ``fig_*.py --plot-results`` reads them).

The asset is one tarball. Every file ``data/MANIFEST.sha256`` lists is read out of it and checked
against its sha256; only when all of them match are they moved into ``data/``, so a corrupted or
truncated download changes nothing. The overlap tables (rows under ``spatial_overlap_integrals/``)
go to the repository root instead, the first place ``lucid.overlap`` looks for them. A file already in ``data/`` that matches is left alone; one
that does not is replaced.
Standard library only, so it runs wherever Python does (macOS has no ``sha256sum``).
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
PAPER_DATA = REPO / 'analysis' / 'paper' / 'output' / 'data'
# A new version of an asset is a new release: change its tag and its manifest together.
RELEASE = 'lucid-data-v1'
PAPER_RELEASE = 'lucid-paper-results-v1'
MANIFESTS = {RELEASE: DATA / 'MANIFEST.sha256',
             PAPER_RELEASE: REPO / 'analysis' / 'paper' / 'RESULTS.sha256'}
TABLES = 'spatial_overlap_integrals/'


def url(release):
    return f'https://github.com/CIDeR-ML/LUCiD/releases/download/{release}/{release}.tar.gz'


def target(rel, release=RELEASE):
    """Where a manifest row installs: a paper result under ``analysis/paper/output/data/``, an
    overlap table at the repository root, the rest in ``data/``."""
    if release == PAPER_RELEASE:
        return PAPER_DATA / rel
    return (REPO if rel.startswith(TABLES) else DATA) / rel


def fetch(link, dest, attempts=6, wait=2.0):
    """Download ``link`` to ``dest``, retrying a server error or a dropped connection.

    GitHub's release downloads answer 504 to a share of requests at times; each retry waits twice
    as long as the last. A client error (404: the release is not there) is not retried.
    """
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(link, dest)
            return
        except urllib.error.HTTPError as err:
            if err.code < 500 or i == attempts - 1:
                raise
            reason = f'HTTP {err.code}'
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            if i == attempts - 1:
                raise
            reason = str(getattr(err, 'reason', err))
        print(f'  {reason}; retrying in {wait * 2 ** i:.0f} s', file=sys.stderr)
        time.sleep(wait * 2 ** i)


def read_manifest(path):
    """``{path relative to data/: sha256}`` from a ``sha256sum``-format file; ``#`` starts a comment."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            digest, rel = line.split(maxsplit=1)
            rows[rel] = digest
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def status(rows, release):
    """``{rel: 'ok' | 'MISSING' | 'MISMATCH'}`` for every manifest row."""
    def one(path, digest):
        if not path.is_file():
            return 'MISSING'
        return 'ok' if sha256(path) == digest else 'MISMATCH'
    return {rel: one(target(rel, release), digest) for rel, digest in rows.items()}


def install(archive, rows, release):
    """Check every manifest file inside ``archive``, then move each to its ``target``.

    Raises ``ValueError`` naming the first file that is absent from the archive or does not match
    its sha256; nothing is then moved. Only manifest paths are read out of the archive,
    so a member elsewhere in it cannot write anywhere.
    """
    with tempfile.TemporaryDirectory(dir=DATA, prefix='.install-') as stage, \
            tarfile.open(archive) as tar:
        for rel, digest in rows.items():
            try:
                member = tar.extractfile(f'{release}/{rel}')
            except KeyError:
                raise ValueError(f'{archive} has no {release}/{rel}') from None
            staged = Path(stage) / rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            with open(staged, 'wb') as f:
                shutil.copyfileobj(member, f)
            if sha256(staged) != digest:
                raise ValueError(f'{rel} in {archive} does not match its sha256 in {MANIFESTS[release]}')
        for rel in rows:
            dest = target(rel, release)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(Path(stage) / rel, dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--paper', action='store_true',
                    help=f'install {PAPER_RELEASE} (the paper figures\' inputs) instead of {RELEASE}')
    ap.add_argument('--from-dir', type=Path, metavar='DIR',
                    help='install from DIR/<asset>.tar.gz (a local copy of the release asset) '
                         'instead of downloading it')
    ap.add_argument('--verify-only', action='store_true',
                    help='check the installed files against the manifest and exit; no network')
    args = ap.parse_args(argv)
    release = PAPER_RELEASE if args.paper else RELEASE
    manifest, asset = MANIFESTS[release], f'{release}.tar.gz'
    rows = read_manifest(manifest)

    before = status(rows, release)
    if args.verify_only:
        for rel, state in before.items():
            print(f'  {state:8s} {rel}')
        bad = sum(state != 'ok' for state in before.values())
        print(f'{len(rows) - bad} of {len(rows)} files match {manifest.relative_to(REPO)}')
        return 1 if bad else 0
    todo = {rel: rows[rel] for rel, state in before.items() if state != 'ok'}
    if not todo:
        print(f'every file of {release} is already installed')
        return 0
    for rel in todo:
        if before[rel] == 'MISMATCH':
            print(f'  replacing {rel} (its sha256 differs from the manifest)')

    try:
        if args.from_dir:
            install(args.from_dir / asset, todo, release)
        else:
            with tempfile.TemporaryDirectory(dir=DATA, prefix='.download-') as tmp:
                print(f'fetching {url(release)}')
                fetch(url(release), Path(tmp) / asset)
                install(Path(tmp) / asset, todo, release)
    except (OSError, EOFError, ValueError, tarfile.TarError, zlib.error) as err:
        print(f'error: {err}; nothing was installed', file=sys.stderr)
        return 1
    print(f'installed {len(todo)} files of {release}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
