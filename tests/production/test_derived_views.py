"""The derived-view scripts (lucid/production/derived_views/relabel_*.py) relabel every batch.

They copied the whole dataset and then relabelled only ``sorted(glob(...))[0]``, so every batch
after the first kept its old labels; relabel_cone also overwrote the dark-noise code. Each script
runs here as a command on a two-batch dataset of one muon + Michel event per batch (the muon is
particle 0, its decay electron particle 1).
"""

import os
import subprocess
import sys
from pathlib import Path

import h5py

from lucid.dataset.layout import batch_paths
from lucid.dataset.schema import EMISSION_PROCESS_DARK
from lucid.production.dataprod import drivers
from lucid.simulation.readout import Readout
from tests.production.test_event_generation import (
    SKI_NO_DARK,
    _muon_michel_event,
    _StubSimulator,
)

REPO = Path(__file__).resolve().parents[2]
VIEWS = REPO / "lucid" / "production" / "derived_views"


def _two_batches(tmp_path, fake_root, detector):
    src = tmp_path / "src"
    for k in (0, 1):
        path = fake_root(tmp_path / f"ps{k}.root", [_muon_michel_event()])
        drivers.run_particles(
            _StubSimulator(detector),
            path,
            output_dir=str(src),
            file_index=k,
            n_events=1,
            master_seed=7,
            job_id=k + 1,
            readout=Readout.from_blocks(SKI_NO_DARK),
            dataset_name="t",
            apply_smearing=False,
            apply_translation=False,
            pad_size_buckets=(64,),
        )
    for k in (0, 1):
        with h5py.File(batch_paths(src, k)["labl"], "r") as f:
            assert sorted(set(f["event_000/per_track/particle_idx"][()])) == [0, 1]
    return src


def _run(script, src, dst, param):
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])
    )
    subprocess.run(
        [sys.executable, str(VIEWS / script), str(src), str(dst), str(param)],
        check=True,
        env=env,
        capture_output=True,
    )


def _read(dst, k, sub, name):
    with h5py.File(batch_paths(dst, k)[sub], "r") as f:
        return f[f"event_000/{name}"][()]


def test_merge_folds_the_decay_electron_into_the_muon_in_every_batch(tmp_path, fake_root, detector):
    src = _two_batches(tmp_path, fake_root, detector)
    _run("relabel_particles.py", src, tmp_path / "dst", "merge")
    for k in (0, 1):
        assert set(_read(tmp_path / "dst", k, "labl", "per_track/particle_idx")) == {0}, k
        assert set(_read(tmp_path / "dst", k, "hits", "particle_idx")) == {0}, k


def test_segments_rebuild_the_particles_of_every_batch(tmp_path, fake_root, detector):
    """The decay electron is an EM child of the muon, so it joins the muon's particle."""
    src = _two_batches(tmp_path, fake_root, detector)
    _run("relabel_segments.py", src, tmp_path / "dst", 30)
    for k in (0, 1):
        assert _read(tmp_path / "dst", k, "labl", "per_particle/category").size == 1, k
        assert set(_read(tmp_path / "dst", k, "hits", "particle_idx")) == {0}, k


def test_cone_labels_every_batch_and_keeps_dark_noise(tmp_path, fake_root, detector):
    src = _two_batches(tmp_path, fake_root, detector)
    with h5py.File(batch_paths(src, 1)["hits"], "r+") as f:  # one hit made dark noise
        f["event_000/emission_process"][0] = EMISSION_PROCESS_DARK
        f["event_000/particle_idx"][0] = -1
    # a 180-degree cone holds every PMT: all light is relabelled 1 (it was 0, Cherenkov)
    _run("relabel_cone.py", src, tmp_path / "dst", 180)
    for k in (0, 1):
        emis = _read(tmp_path / "dst", k, "hits", "emission_process")
        light = emis[1:] if k == 1 else emis
        assert set(light) == {1}, k
        assert set(_read(tmp_path / "dst", k, "step", "sensor_hits/emission_process")) == {1}, k
    assert _read(tmp_path / "dst", 1, "hits", "emission_process")[0] == EMISSION_PROCESS_DARK
