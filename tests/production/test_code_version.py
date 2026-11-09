"""The code version a dataset batch records (``lucid.utils.code_version``).

The production image has no ``.git/`` (``container/Dockerfile.dockerignore``), so every production
batch recorded ``git_commit = 'unknown'`` although the image writes its commit into
``/opt/VERSIONS``. The image paths are arguments here, so a fake install and record stand in.
"""
import functools
import subprocess
import sys
import types

import h5py
import pytest

from lucid import utils
from lucid.utils import code_version

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _fresh_cache():
    code_version.cache_clear()
    yield
    code_version.cache_clear()


def _image(tmp_path, record):
    """A git-less install dir and its version record, as the image lays them out."""
    install = tmp_path / "opt" / "LUCiD"
    install.mkdir(parents=True)
    versions = tmp_path / "opt" / "VERSIONS"
    if record is not None:
        versions.write_text(record)
    return str(install), (str(install), str(versions))


def _record(**over):
    keys = dict(lucid_sha=SHA, lucid_dirty="false", lucid_source="https://github.com/CIDeR-ML/LUCiD",
                photonsim_sha="f" * 40, build_date="2026-11-08T00:00:00Z")
    keys.update(over)
    return "".join(f"{k}={v}\n" for k, v in keys.items() if v is not None)


def test_the_image_install_takes_its_commit_and_dirty_flag_from_the_version_record(tmp_path):
    install, image = _image(tmp_path, _record())
    assert code_version(install, image) == (SHA, False)
    code_version.cache_clear()
    install, image = _image(tmp_path / "d", _record(lucid_dirty="true"))
    assert code_version(install, image) == (SHA, True)


def test_an_image_record_without_lucid_dirty_leaves_dirty_unknown(tmp_path):
    """Images built before ``lucid_dirty`` was recorded say nothing about it."""
    install, image = _image(tmp_path, _record(lucid_dirty=None))
    assert code_version(install, image) == (SHA, None)


@pytest.mark.parametrize("record", [None, _record(lucid_sha="unknown"), _record(lucid_sha=""),
                                    _record(lucid_sha=None), _record(lucid_sha="main")])
def test_an_unknown_image_commit_stays_unknown(tmp_path, record):
    """No record, the Dockerfile's ``ARG LUCID_SHA=unknown`` default, or a non-SHA value."""
    install, image = _image(tmp_path, record)
    assert code_version(install, image)[0] == "unknown"


def test_a_gitless_tree_that_is_not_the_image_install_is_unknown(tmp_path):
    """The record describes the image's own install; a copied tree elsewhere is not that code."""
    _, image = _image(tmp_path, _record())
    other = tmp_path / "copy" / "LUCiD"
    other.mkdir(parents=True)
    assert code_version(str(other), image) == ("unknown", None)


def _git(cwd, *args):
    return subprocess.check_output(["git", "-C", str(cwd), "-c", "user.name=t",
                                    "-c", "user.email=t@t", *args], text=True).strip()


def test_a_git_checkout_gives_head_and_whether_tracked_files_changed(tmp_path):
    repo = tmp_path / "LUCiD"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "c")
    head = _git(repo, "rev-parse", "HEAD")
    _, image = _image(tmp_path / "img", _record())   # a checkout wins over any image record
    assert code_version(str(repo), image) == (head, False)
    (repo / "untracked.txt").write_text("scratch\n")  # untracked files do not count
    code_version.cache_clear()
    assert code_version(str(repo), image) == (head, False)
    (repo / "a.py").write_text("x = 2\n")
    code_version.cache_clear()
    assert code_version(str(repo), image) == (head, True)
    # a tree that only sits inside a repository is not that repository's code
    sub = repo / "vendored"
    sub.mkdir()
    code_version.cache_clear()
    assert code_version(str(sub), image) == ("unknown", None)


def test_the_running_checkout_is_found():
    """This test suite runs from a checkout: a real commit, and ``git_commit`` agrees."""
    commit, dirty = code_version()
    assert len(commit) == 40 and dirty in (True, False)
    assert utils.git_commit() == commit


@pytest.mark.parametrize("dirty", [False, True, None])
def test_the_batch_records_the_image_commit_and_dirty_flag(tmp_path, monkeypatch, dirty):
    """``run_particles`` stamps ``config/git_commit`` and ``git_dirty`` (absent when not known)
    on all four files, from the image record when lucid runs from the image install."""
    import tests.production.test_event_generation as teg
    from lucid.production.dataprod import drivers
    from lucid.simulation.readout import Readout

    flag = {False: "false", True: "true", None: None}[dirty]
    install, image = _image(tmp_path, _record(lucid_dirty=flag))
    monkeypatch.setattr(drivers, "code_version",
                        functools.partial(code_version, install, image))
    files = {}
    monkeypatch.setitem(sys.modules, "uproot", types.SimpleNamespace(open=lambda p: files[str(p)]))
    ev = teg._single_muon_event()
    for c in ev["chunks"]:
        c["EventID"] = 0
    files[str(tmp_path / "a.root")] = teg._RootFile([ev])
    out = tmp_path / "out"
    drivers.run_particles(
        teg._StubSimulator(teg.Cylinder(teg.R_DET, teg.H_DET, 120, teg.S_RADIUS)),
        str(tmp_path / "a.root"), output_dir=str(out), file_index=0, n_events=1, master_seed=1,
        job_id=1, readout=Readout.from_blocks(), dataset_name="t", apply_smearing=False,
        apply_translation=False)
    for sub in ("sensor", "hits", "step", "labl"):
        with h5py.File(out / sub / f"wc_{sub}_0000.h5", "r") as f:
            attrs = f["config"].attrs
            assert attrs["git_commit"] == SHA
            if dirty is None:
                assert "git_dirty" not in attrs
            else:
                assert bool(attrs["git_dirty"]) is dirty
