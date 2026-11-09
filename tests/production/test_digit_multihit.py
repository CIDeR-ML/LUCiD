"""Roundtrip test for the digit-list schema: multi-hit, digit_idx, dark, per_window.

The dense build_synthetic_event path collapses to one digit/sensor; this exercises
the digitizer's actual output shape and the trigger's per_window table.
"""
import h5py
import numpy as np
import pytest

from lucid.dataset.writer import (
    write_sensor_config, write_hits_config, write_step_config, write_labl_config,
    save_sensor_event, save_hits_event, save_step_event, save_labl_event,
)
from lucid.dataset.reader import (
    read_sensor_event, read_hits_event, read_step_event, read_labl_event,
)
from tests.helpers.event_fixture import build_synthetic_digit_event


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


@pytest.fixture
def digitized_batch(tmp_path):
    """The synthetic event's labels with a payload from the real hit-making chain.

    Two particles (4 segments) light 12 sensors with prompt (~100 ns) and
    delayed (~3 us) bursts; ski windows them, 200 kHz dark noise is mixed in,
    the trigger keeps two gates. No readout smearing, so a digit's charge is
    exactly the sum of its deposits. Written through the four writers.
    """
    from lucid.simulation.digitizer import digitize_and_decompose, resolve_model_config
    from lucid.simulation.trigger import TriggerConfig, apply_trigger
    cfg, ev, pos = build_synthetic_digit_event()
    rng = np.random.default_rng(3)
    n = 300
    burst = rng.integers(0, 2, n)
    t = np.where(burst == 0, 100.0, 3000.0) + rng.exponential(5.0, n)
    seg = rng.integers(0, 4, n)
    sd, hits, segh = digitize_and_decompose(
        sensor_idx=rng.integers(0, 12, n), charge=np.ones(n), t_true=t,
        t_reco=t, particle_idx=np.where(seg < 2, 0, 1),
        segment_idx=seg, emission_process=np.zeros(n, np.int64), n_sensors=20,
        model=resolve_model_config("ski"), rng=rng, dark_rate_khz=200.0,
        readout_pad_ns=500.0, apply_resolution=False)
    trig = apply_trigger(sd, hits, segh, TriggerConfig(window_ns=200.0, n_thr=8,
                                                       pad_before_ns=30.0, pad_after_ns=30.0))
    assert trig is not None
    ev = dict(ev)
    ev['sensor_digits'], ev['hits_sparse'], ev['segment_sensor_hits'], ev['per_window'] = trig
    paths = {k: tmp_path / f'wc_{k}_0000.h5' for k in ('sensor', 'hits', 'step', 'labl')}
    src = np.array([ev['source_event_idx']], np.uint32)
    with h5py.File(paths['sensor'], 'w') as fs, h5py.File(paths['hits'], 'w') as fi, \
         h5py.File(paths['step'], 'w') as fg, h5py.File(paths['labl'], 'w') as fl:
        write_sensor_config(fs, cfg, src, pos); write_hits_config(fi, cfg, src, pos)
        write_step_config(fg, cfg, src); write_labl_config(fl, cfg, src)
        save_sensor_event(fs, ev, 0); save_hits_event(fi, ev, 0)
        save_step_event(fg, ev, 0); save_labl_event(fl, ev, 0)
    return paths


def test_digit_fk_charge_conservation_and_dark_on_disk(digitized_batch):
    """Every decomposition row points at a digit on its own sensor; per digit the
    hits rows (dark included) sum to the digit charge, the step rows (no dark)
    to the digit's non-dark charge; dark rows carry particle_idx -1."""
    s = read_sensor_event(str(digitized_batch['sensor']), 0)
    h = read_hits_event(str(digitized_batch['hits']), 0)
    sh = read_step_event(str(digitized_batch['step']), 0)['sensor_hits']
    nd = len(s['sensor_idx'])
    assert nd > 12                                         # multi-hit: more digits than sensors
    for tbl in (h, sh):
        assert tbl['digit_idx'].min() >= 0 and tbl['digit_idx'].max() < nd
        np.testing.assert_array_equal(tbl['sensor_idx'], s['sensor_idx'][tbl['digit_idx']])
    dark = h['emission_process'] == 2
    assert dark.any() and (~dark).any()
    assert (h['particle_idx'][dark] == -1).all() and (h['particle_idx'][~dark] >= 0).all()
    np.testing.assert_allclose(np.bincount(h['digit_idx'], weights=h['PE'], minlength=nd),
                               s['PE'], rtol=1e-6)
    assert (sh['emission_process'] != 2).all()
    np.testing.assert_allclose(
        np.bincount(sh['digit_idx'], weights=sh['PE'], minlength=nd),
        np.bincount(h['digit_idx'][~dark], weights=h['PE'][~dark], minlength=nd), rtol=1e-6)
    # a digit opens at its first deposit: per digit, the earliest T (and T_reco;
    # equal here, no TTS) among its decomposition rows, each a min over its own
    # deposits, is the digit T
    for col in ('T', 'T_reco'):
        first = np.full(nd, np.inf)
        np.minimum.at(first, h['digit_idx'], h[col])
        np.testing.assert_array_equal(first, s['T'], err_msg=col)


def test_triggered_windows_hold_their_digits(digitized_batch):
    s = read_sensor_event(str(digitized_batch['sensor']), 0)
    pw = read_labl_event(str(digitized_batch['labl']), 0)['per_window']
    off = pw['digit_offsets']
    assert off[0] == 0 and off[-1] == len(s['sensor_idx']) and len(off) == 3
    for w in range(len(off) - 1):
        t = s['T'][off[w]:off[w + 1]]
        assert ((t >= pw['window_start'][w]) & (t <= pw['window_end'][w])).all()


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
    0.4 ns TDC steps into a single value at 1e9 (its step there is 64 ns).

    Checked for every time column the digitizer path writes: sensor T, hits T /
    T_reco, step time and sensor_hits T / T_reco, per_window edges and
    per_interaction / per_event t0.
    """
    cfg, ev, pos = build_synthetic_digit_event()
    off = 1.0e9                                       # 1 s in ns (mid-burst)
    steps = off + np.array([0.0, 0.4, 0.8, 1.2])
    ev = dict(ev)
    ev['sensor_digits'] = dict(ev['sensor_digits'], T=steps)
    ev['hits_sparse'] = dict(ev['hits_sparse'], T=steps, T_reco=steps + 0.1)
    ev['segment_sensor_hits'] = dict(ev['segment_sensor_hits'], T=steps[:3],
                                     T_reco=steps[:3] + 0.1)
    ev['segments'] = dict(ev['segments'], time=steps)
    ev['per_window'] = dict(ev['per_window'], window_start=off + np.array([-0.4, 0.4]),
                            window_end=off + np.array([0.2, 1.6]))
    ev['interaction_metadata'] = [dict(ev['interaction_metadata'][0], t0=off + 0.3)]
    paths = {k: tmp_path / f'wc_{k}_0000.h5' for k in ('sensor', 'hits', 'step', 'labl')}
    src = np.array([0], np.uint32)
    with h5py.File(paths['sensor'], 'w') as fs, h5py.File(paths['hits'], 'w') as fi, \
         h5py.File(paths['step'], 'w') as fg, h5py.File(paths['labl'], 'w') as fl:
        write_sensor_config(fs, cfg, src, pos); write_hits_config(fi, cfg, src, pos)
        write_step_config(fg, cfg, src); write_labl_config(fl, cfg, src)
        save_sensor_event(fs, ev, 0); save_hits_event(fi, ev, 0)
        save_step_event(fg, ev, 0); save_labl_event(fl, ev, 0)
    s = read_sensor_event(str(paths['sensor']), 0)
    h = read_hits_event(str(paths['hits']), 0)
    st = read_step_event(str(paths['step']), 0)
    lb = read_labl_event(str(paths['labl']), 0)
    cols = {'sensor/T': (s['T'], steps), 'hits/T': (h['T'], steps),
            'hits/T_reco': (h['T_reco'], steps + 0.1), 'step/time': (st['time'], steps),
            'step/sensor_hits/T': (st['sensor_hits']['T'], steps[:3]),
            'step/sensor_hits/T_reco': (st['sensor_hits']['T_reco'], steps[:3] + 0.1),
            'per_window/window_start': (lb['per_window']['window_start'], off + np.array([-0.4, 0.4])),
            'per_window/window_end': (lb['per_window']['window_end'], off + np.array([0.2, 1.6])),
            'per_interaction/t0': (lb['per_interaction']['t0'], [off + 0.3]),
            'per_event/t0': (np.atleast_1d(lb['per_event']['t0']), [off + 0.3])}
    for name, (got, want) in cols.items():
        got = np.asarray(got)
        assert got.dtype == np.float64, name
        np.testing.assert_allclose(got - off, np.asarray(want) - off, rtol=0, atol=1e-6,
                                   err_msg=name)
