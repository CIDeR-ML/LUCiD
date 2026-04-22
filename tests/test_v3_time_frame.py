"""Verify v3 writers apply the t0 shift consistently.

seg.time, inst.T, and sensor.T must all be stored in detector-frame
(G4-absolute minus t0). labl/per_interaction/t0 is the truth emission
time (per interaction — one row for single-interaction events).
"""
import h5py
import numpy as np
import pytest

from lucid.sources.event_io import (
    write_sensor_config_v3, write_inst_config_v3,
    write_seg_config_v3, write_labl_config_v3,
    save_sensor_event_v3, save_inst_event_v3,
    save_seg_event_v3, save_labl_event_v3,
    read_sensor_event_v3, read_inst_event_v3,
    read_seg_event_v3, read_labl_event_v3,
)

from tests._v3_event_fixture import build_synthetic_event


def _write_batch(tmp_path, t0):
    cfg, ev, sp = build_synthetic_event(t0=t0)
    src = np.array([ev['source_event_idx']], dtype=np.uint32)
    paths = {
        'sensor': tmp_path / 'wc_sensor_0000.h5',
        'inst':   tmp_path / 'wc_inst_0000.h5',
        'seg':    tmp_path / 'wc_seg_0000.h5',
        'labl':   tmp_path / 'wc_labl_0000.h5',
    }
    with h5py.File(paths['sensor'], 'w') as fs, \
         h5py.File(paths['inst'],   'w') as fi, \
         h5py.File(paths['seg'],    'w') as fg, \
         h5py.File(paths['labl'],   'w') as fl:
        write_sensor_config_v3(fs, cfg, src, sp)
        write_inst_config_v3(fi, cfg, src, sp)
        write_seg_config_v3(fg, cfg, src)
        write_labl_config_v3(fl, cfg, src)
        save_sensor_event_v3(fs, ev, seq_idx=0)
        save_inst_event_v3(fi, ev, seq_idx=0)
        save_seg_event_v3(fg, ev, seq_idx=0)
        save_labl_event_v3(fl, ev, seq_idx=0)
    return paths, ev


@pytest.mark.parametrize("t0", [0.0, 7.5, -3.2, 15.0])
def test_seg_time_is_t0_shifted(tmp_path, t0):
    paths, ev = _write_batch(tmp_path, t0)
    seg = read_seg_event_v3(str(paths['seg']), 0)
    expected = ev['segments']['time'] - np.float32(t0)
    np.testing.assert_allclose(seg['time'], expected, atol=1e-5)


@pytest.mark.parametrize("t0", [0.0, 7.5, -3.2])
def test_inst_and_sensor_T_are_t0_shifted(tmp_path, t0):
    paths, ev = _write_batch(tmp_path, t0)
    inst = read_inst_event_v3(str(paths['inst']), 0)
    sensor = read_sensor_event_v3(str(paths['sensor']), 0)

    # Expected detector-frame times (non-zero only for hit sensors)
    expected_t = np.array([50.0, 52.0, 55.0, 60.0, 61.0]) - np.float32(t0)
    np.testing.assert_allclose(inst['T'], expected_t, atol=1e-5)
    np.testing.assert_allclose(sensor['T'], expected_t, atol=1e-5)


def test_labl_t0_matches_input(tmp_path):
    paths, ev = _write_batch(tmp_path, 9.125)
    labl = read_labl_event_v3(str(paths['labl']), 0)
    # Single-interaction fixture → 1 row in per_interaction/
    assert len(labl['per_interaction']['t0']) == 1
    assert float(labl['per_interaction']['t0'][0]) == pytest.approx(ev['t0'])
