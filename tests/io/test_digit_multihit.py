"""Roundtrip test for the digit-list schema: multi-hit, digit_idx, dark, per_window.

The dense build_synthetic_event path collapses to one digit/sensor; this exercises
the digitizer's actual output shape and the trigger's per_window table.
"""
import h5py
import numpy as np
import pytest

from lucid.sources.writer import (
    write_sensor_config, write_hits_config, write_step_config, write_labl_config,
    save_sensor_event, save_hits_event, save_step_event, save_labl_event,
)
from lucid.sources.reader import (
    read_sensor_event, read_hits_event, read_step_event, read_labl_event,
)
from tests.io._event_fixture import build_synthetic_digit_event


@pytest.fixture
def digit_batch(tmp_path):
    cfg, ev, pos = build_synthetic_digit_event()
    paths = {k: tmp_path / f'wc_{k}_0000.h5' for k in ('sensor', 'hits', 'step', 'labl')}
    src = np.array([ev['source_event_idx']], np.uint32)
    with h5py.File(paths['sensor'], 'w') as fs, h5py.File(paths['hits'], 'w') as fi, \
         h5py.File(paths['step'], 'w') as fg, h5py.File(paths['labl'], 'w') as fl:
        write_sensor_config(fs, cfg, src, pos); write_hits_config(fi, cfg, src, pos)
        write_step_config(fg, cfg, src); write_labl_config(fl, cfg, src)
        save_sensor_event(fs, ev, 0); save_hits_event(fi, ev, 0)
        save_step_event(fg, ev, 0); save_labl_event(fl, ev, 0)
    return paths


def test_sensor_multihit(digit_batch):
    s = read_sensor_event(str(digit_batch['sensor']), 0)
    assert list(s['sensor_idx']) == [5, 6, 5, 7]          # 4 digits
    assert int((s['sensor_idx'] == 5).sum()) == 2         # sensor 5 multi-hit
    np.testing.assert_allclose(s['PE'], [3.0, 2.0, 1.5, 1.0])


def test_hits_digit_idx_fk_and_dark(digit_batch):
    s = read_sensor_event(str(digit_batch['sensor']), 0)
    h = read_hits_event(str(digit_batch['hits']), 0)
    # digit_idx present + FK valid: hits.sensor == sensor[digit_idx]
    assert 'digit_idx' in h
    assert h['digit_idx'].min() >= 0 and h['digit_idx'].max() < len(s['sensor_idx'])
    assert (h['sensor_idx'] == s['sensor_idx'][h['digit_idx']]).all()
    # dark row: emission_process==2, particle_idx==-1
    dark = h['emission_process'] == 2
    assert int(dark.sum()) == 1 and int(h['particle_idx'][dark][0]) == -1
    # charge conservation: sum hits PE over a digit == sensor digit PE
    per_digit = np.zeros(len(s['PE'])); np.add.at(per_digit, h['digit_idx'], h['PE'])
    np.testing.assert_allclose(per_digit, s['PE'], atol=1e-4)


def test_step_sensor_hits_digit_idx(digit_batch):
    s = read_sensor_event(str(digit_batch['sensor']), 0)
    st = read_step_event(str(digit_batch['step']), 0)
    sh = st['sensor_hits']
    assert 'digit_idx' in sh
    assert (sh['sensor_idx'] == s['sensor_idx'][sh['digit_idx']]).all()   # FK valid
    assert (sh['emission_process'] != 2).all()                            # no dark in segments


def test_per_window_roundtrip_and_csr(digit_batch):
    s = read_sensor_event(str(digit_batch['sensor']), 0)
    labl = read_labl_event(str(digit_batch['labl']), 0)
    assert 'per_window' in labl
    pw = labl['per_window']
    np.testing.assert_allclose(pw['window_start'], [40., 2990.])
    np.testing.assert_allclose(pw['window_end'], [60., 3010.])
    off = pw['digit_offsets']
    assert list(off) == [0, 2, 4]
    assert off[0] == 0 and off[-1] == len(s['sensor_idx'])   # CSR spans all digits
    assert (np.diff(off) >= 0).all()                          # monotonic


def test_trigger_config_attrs(digit_batch):
    with h5py.File(digit_batch['sensor'], 'r') as f:
        a = f['config'].attrs
        assert str(a['trigger']).strip("b'\"") == 'sliding_window' or a['trigger'] == 'sliding_window'
        assert int(a['trigger_n_thr']) == 30
        assert float(a['trigger_window_ns']) == 200.0


def test_float64_preserves_subns_on_second_scale_offset(tmp_path):
    """The float64 time schema must hold sub-ns light timing on a burst-scale
    (~1e9 ns) t0 offset — the supernova motivation. float32 would collapse the
    0.4 ns TDC steps into a single value at 1e9."""
    import h5py, numpy as np
    from lucid.sources.writer import write_sensor_config, save_sensor_event
    from lucid.sources.reader import read_sensor_event
    from tests.io._event_fixture import build_synthetic_digit_event

    cfg, ev, pos = build_synthetic_digit_event()
    offset = 1.0e9                                   # 1 s in ns (mid-burst)
    ev['sensor_digits']['T'] = np.array(
        [offset + 0.0, offset + 0.4, offset + 0.8, offset + 1.2], np.float64)
    p = tmp_path / 'wc_sensor_0000.h5'
    with h5py.File(p, 'w') as f:
        write_sensor_config(f, cfg, np.array([0], np.uint32), pos)
        save_sensor_event(f, ev, 0)
    T = read_sensor_event(str(p), 0)['T']
    assert T.dtype == np.float64
    # the four 0.4 ns steps survive exactly (float32 would round them together)
    np.testing.assert_allclose(np.diff(T), [0.4, 0.4, 0.4], atol=1e-6)
    assert len(np.unique(T)) == 4
