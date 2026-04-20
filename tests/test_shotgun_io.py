"""Tests for shotgun HDF5 IO roundtrips."""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pytest

from lucid.production.photon_shotgun.io import (
    sparsify_waveform, densify_waveform,
    save_shotgun_waveform, load_shotgun_waveform,
    save_shotgun_per_photon, load_shotgun_per_photon,
    StreamingWaveformWriter, StreamingPerPhotonWriter,
)
from lucid.sources import shotgun_source


@pytest.fixture
def tiny_waveform():
    rng = np.random.default_rng(0)
    wf = np.zeros((3, 10, 20), dtype=np.float32)
    for c in range(3):
        idx_s = rng.integers(0, 10, size=5)
        idx_t = rng.integers(0, 20, size=5)
        for s, t in zip(idx_s, idx_t):
            wf[c, s, t] += rng.uniform(0.5, 2.0)
    return wf


def test_sparsify_densify_roundtrip(tiny_waveform):
    ci, sid, tb, ch, offsets = sparsify_waveform(tiny_waveform)
    wf2 = densify_waveform(ci, sid, tb, ch,
                           n_cases=tiny_waveform.shape[0],
                           num_sensors=tiny_waveform.shape[1],
                           n_time_bins=tiny_waveform.shape[2])
    assert np.array_equal(wf2, tiny_waveform)


def test_entries_per_case_offsets(tiny_waveform):
    ci, _, _, _, offsets = sparsify_waveform(tiny_waveform)
    assert offsets.shape == (tiny_waveform.shape[0] + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == len(ci)
    for c in range(tiny_waveform.shape[0]):
        start, end = offsets[c], offsets[c + 1]
        assert np.all(ci[start:end] == c)


def test_waveform_hdf5_roundtrip(tmp_path, tiny_waveform):
    path = tmp_path / "wf.h5"
    n_detected = np.asarray([10, 12, 15], dtype=np.int32)
    n_dropped = np.asarray([0, 1, 0], dtype=np.int32)
    src = shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=4, wavelength=400.0)

    save_shotgun_waveform(
        str(path), tiny_waveform, n_dropped, n_detected,
        window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=0.4,
        source=src, save_source=False,
    )

    out = load_shotgun_waveform(str(path), dense=True)
    assert np.array_equal(out['waveform'], tiny_waveform)
    assert np.array_equal(out['n_detected'], n_detected)
    assert np.array_equal(out['n_dropped'], n_dropped)
    assert out['meta']['mode'] == 'waveform'
    assert float(out['meta']['window_ns']) == 500.0
    assert float(out['meta']['bin_width_ns']) == 1.0


def test_streaming_waveform_writer_roundtrip(tmp_path):
    """Streaming writer ↔ load_shotgun_waveform produce identical COO arrays to
    the equivalent one-shot save_shotgun_waveform."""
    rng = np.random.default_rng(1)
    n_cases_total = 7
    num_sensors, n_time_bins = 8, 10
    wf_full = np.zeros((n_cases_total, num_sensors, n_time_bins), dtype=np.float32)
    for c in range(n_cases_total):
        for _ in range(4):
            s = rng.integers(0, num_sensors)
            t = rng.integers(0, n_time_bins)
            wf_full[c, s, t] += rng.uniform(0.5, 2.0)
    nd_full = rng.integers(0, 3, size=n_cases_total).astype(np.int32)
    ndet_full = rng.integers(10, 30, size=n_cases_total).astype(np.int32)

    waveform_config = dict(window_ns=10.0, bin_width_ns=1.0, tts_sigma_ns=1.0,
                           t_min_ns=0.0, smear_time=True, smear_charge=True)

    path = tmp_path / "stream_wf.h5"
    with StreamingWaveformWriter(str(path),
                                 num_sensors=num_sensors, n_time_bins=n_time_bins,
                                 waveform_config=waveform_config,
                                 n_photons=100, K=4, save_source=False) as w:
        # Write in two uneven chunks
        w.append(wf_full[:3], nd_full[:3], ndet_full[:3])
        w.append(wf_full[3:], nd_full[3:], ndet_full[3:])

    out = load_shotgun_waveform(str(path), dense=True)
    assert int(out['meta']['n_cases']) == n_cases_total
    assert int(out['meta']['num_sensors']) == num_sensors
    assert int(out['meta']['n_time_bins']) == n_time_bins
    assert np.array_equal(out['waveform'], wf_full)
    assert np.array_equal(out['n_dropped'], nd_full)
    assert np.array_equal(out['n_detected'], ndet_full)
    # Per-case entry offsets correct
    assert out['entries_per_case'][0] == 0
    assert out['entries_per_case'][-1] == out['case_idx'].shape[0]


def test_streaming_per_photon_writer_roundtrip(tmp_path):
    n_cases_total, n_photons = 5, 4
    detected = np.array([
        [True, False, True, False],
        [False, True, True, True],
        [True, True, False, False],
        [False, False, False, False],
        [True, False, False, True],
    ])
    sensor_id = np.array([
        [3, -1, 5, -1],
        [-1, 7, 2, 1],
        [6, 4, -1, -1],
        [-1, -1, -1, -1],
        [0, -1, -1, 2],
    ], dtype=np.int32)
    hit_time = np.array([
        [12.3, 0.0, 45.6, 0.0],
        [0.0, 33.1, 77.8, 19.2],
        [8.1, 2.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [5.5, 0.0, 0.0, 99.9],
    ], dtype=np.float32)

    path = tmp_path / "stream_pp.h5"
    with StreamingPerPhotonWriter(str(path), n_photons=n_photons,
                                  tts_sigma_ns=1.0, K=4, save_source=False) as w:
        w.append(detected[:2], sensor_id[:2], hit_time[:2])
        w.append(detected[2:], sensor_id[2:], hit_time[2:])

    out = load_shotgun_per_photon(str(path))
    assert int(out['meta']['n_cases']) == n_cases_total
    assert int(out['meta']['n_photons']) == n_photons
    assert np.array_equal(out['detected'], detected)
    assert np.array_equal(out['sensor_id'], sensor_id)
    assert np.allclose(out['hit_time'], hit_time)


def test_per_photon_hdf5_roundtrip(tmp_path):
    detected = np.array([[True, False, True], [False, True, True]])
    sensor_id = np.array([[5, -1, 7], [-1, 2, 4]], dtype=np.int32)
    hit_time = np.array([[12.3, 0.0, 45.6], [0.0, 33.1, 77.8]], dtype=np.float32)

    path = tmp_path / "pp.h5"
    save_shotgun_per_photon(
        str(path), detected, sensor_id, hit_time,
        tts_sigma_ns=0.4, save_source=False,
    )
    out = load_shotgun_per_photon(str(path))
    assert np.array_equal(out['detected'], detected)
    assert np.array_equal(out['sensor_id'], sensor_id)
    assert np.allclose(out['hit_time'], hit_time)
    assert out['meta']['mode'] == 'per_photon'
