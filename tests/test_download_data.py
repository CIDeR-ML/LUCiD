"""scripts/download_data.py installs only checked bytes, and its manifest is well formed.

The installs run the script against a copy of it in a temporary checkout, with a synthetic asset
in a ``--from-dir`` directory, so no test needs the network or the real assets.
"""
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tarfile

import pytest
from tests.helpers.paths import REPO

FILES = {'water/muon/cherenkov/weights.npz': b'the weights',
         'water/muon/1000MeV_5events.root': b'five events',
         'spatial_overlap_integrals/gaussian_overlap_r0.250000_sigma0.050000_ksig8_nd400_f64.json': b'{}'}


def _installed(root, rel):
    """An overlap table installs at the checkout root (where lucid.overlap looks), the rest in data/."""
    return root / rel if rel.startswith('spatial_overlap_integrals/') else root / 'data' / rel


def _release():
    text = (REPO / 'scripts' / 'download_data.py').read_text()
    return re.search(r"^RELEASE = '([^']+)'", text, re.M).group(1)


@pytest.fixture
def checkout(tmp_path):
    """A checkout holding the script and a manifest for FILES, and an asset directory for it."""
    root = tmp_path / 'LUCiD'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(REPO / 'scripts' / 'download_data.py', root / 'scripts')
    (root / 'data').mkdir()
    (root / 'data' / 'MANIFEST.sha256').write_text('# a comment\n' + ''.join(
        f'{hashlib.sha256(body).hexdigest()}  {rel}\n' for rel, body in FILES.items()))
    (tmp_path / 'assets').mkdir()
    return root


def _write_asset(asset_dir, files, release=None):
    release = release or _release()
    with tarfile.open(asset_dir / f'{release}.tar.gz', 'w:gz') as tar:
        for rel, body in files.items():
            info = tarfile.TarInfo(f'{release}/{rel}')
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))


def _run(root, *args):
    return subprocess.run([sys.executable, str(root / 'scripts' / 'download_data.py'), *args],
                          capture_output=True, text=True)


def test_from_dir_installs_every_file_and_verify_only_agrees(checkout):
    _write_asset(checkout.parent / 'assets', FILES)
    r = _run(checkout, '--from-dir', str(checkout.parent / 'assets'))
    assert r.returncode == 0, r.stderr
    for rel, body in FILES.items():
        assert _installed(checkout, rel).read_bytes() == body
    assert _run(checkout, '--verify-only').returncode == 0


def test_a_corrupted_file_installs_nothing(checkout):
    _write_asset(checkout.parent / 'assets', {**FILES, 'water/muon/1000MeV_5events.root': b'five evens'})
    r = _run(checkout, '--from-dir', str(checkout.parent / 'assets'))
    assert r.returncode == 1
    assert 'does not match' in r.stderr
    assert sorted(p.name for p in (checkout / 'data').iterdir()) == ['MANIFEST.sha256']
    assert not (checkout / 'spatial_overlap_integrals').exists()


def test_a_truncated_asset_installs_nothing(checkout):
    asset_dir = checkout.parent / 'assets'
    _write_asset(asset_dir, FILES)
    tarball = asset_dir / f'{_release()}.tar.gz'
    tarball.write_bytes(tarball.read_bytes()[:tarball.stat().st_size // 2])
    r = _run(checkout, '--from-dir', str(asset_dir))
    assert r.returncode == 1
    assert 'nothing was installed' in r.stderr
    assert sorted(p.name for p in (checkout / 'data').iterdir()) == ['MANIFEST.sha256']
    assert not (checkout / 'spatial_overlap_integrals').exists()


def test_verify_only_catches_a_truncated_file(checkout):
    _write_asset(checkout.parent / 'assets', FILES)
    assert _run(checkout, '--from-dir', str(checkout.parent / 'assets')).returncode == 0
    (checkout / 'data' / 'water/muon/cherenkov/weights.npz').write_bytes(b'the wei')
    r = _run(checkout, '--verify-only')
    assert r.returncode == 1
    assert 'MISMATCH' in r.stdout


def test_paper_installs_the_results_under_output_data(checkout):
    """--paper installs its own asset, checked against analysis/paper/RESULTS.sha256, where the
    figures' --plot-results reads it, and leaves data/ alone."""
    text = (REPO / 'scripts' / 'download_data.py').read_text()
    release = re.search(r"^PAPER_RELEASE = '([^']+)'", text, re.M).group(1)
    results = {'nrays/nrays_mu_5k/nrays_mu_5k.h5': b'a study', 'juno_cher_scint/charges.npz': b'q'}
    (checkout / 'analysis' / 'paper').mkdir(parents=True)
    (checkout / 'analysis' / 'paper' / 'RESULTS.sha256').write_text(''.join(
        f'{hashlib.sha256(body).hexdigest()}  {rel}\n' for rel, body in results.items()))
    _write_asset(checkout.parent / 'assets', results, release)
    r = _run(checkout, '--paper', '--from-dir', str(checkout.parent / 'assets'))
    assert r.returncode == 0, r.stderr
    for rel, body in results.items():
        assert (checkout / 'analysis' / 'paper' / 'output' / 'data' / rel).read_bytes() == body
    assert sorted(p.name for p in (checkout / 'data').iterdir()) == ['MANIFEST.sha256']
    assert _run(checkout, '--paper', '--verify-only').returncode == 0


@pytest.mark.parametrize('manifest', ['data/MANIFEST.sha256', 'analysis/paper/RESULTS.sha256'])
def test_manifest_rows_are_well_formed(manifest):
    """Every row is a sha256 and a unique relative path."""
    rows = [line.split() for line in (REPO / manifest).read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    assert rows
    for row in rows:
        assert len(row) == 2, row
        digest, path = row
        assert re.fullmatch(r'[0-9a-f]{64}', digest), row
        assert not path.startswith('/') and '..' not in path.split('/'), row
    assert len({path for _, path in rows}) == len(rows)
