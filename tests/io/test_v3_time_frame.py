"""Verify v3 writer time-frame semantics.

From v4 on, save_* no longer applies any t0 shift: input times are
expected already in absolute detector frame (generate_events_from_photonsim_*
adds per-interaction t0 before saving). labl/per_interaction/t0 carries
the per-interaction metadata for downstream reconstruction.
"""
import h5py
import numpy as np
import pytest

from lucid.sources.event_io import (
    write_sensor_config_v3, write_hits_config_v3,
    write_edep_config_v3, write_labl_config_v3,
    save_sensor_event_v3, save_hits_event_v3,
    save_edep_event_v3, save_labl_event_v3,
    read_sensor_event_v3, read_hits_event_v3,
    read_edep_event_v3, read_labl_event_v3,
)

from tests.io._v3_event_fixture import build_synthetic_event


def _write_batch(tmp_path, t0):
    cfg, ev, sp = build_synthetic_event(t0=t0)
    src = np.array([ev['source_event_idx']], dtype=np.uint32)
    paths = {
        'sensor': tmp_path / 'wc_sensor_0000.h5',
        'hits':   tmp_path / 'wc_hits_0000.h5',
        'edep':    tmp_path / 'wc_edep_0000.h5',
        'labl':   tmp_path / 'wc_labl_0000.h5',
    }
    with h5py.File(paths['sensor'], 'w') as fs, \
         h5py.File(paths['hits'],   'w') as fi, \
         h5py.File(paths['edep'],    'w') as fg, \
         h5py.File(paths['labl'],   'w') as fl:
        write_sensor_config_v3(fs, cfg, src, sp)
        write_hits_config_v3(fi, cfg, src, sp)
        write_edep_config_v3(fg, cfg, src)
        write_labl_config_v3(fl, cfg, src)
        save_sensor_event_v3(fs, ev, seq_idx=0)
        save_hits_event_v3(fi, ev, seq_idx=0)
        save_edep_event_v3(fg, ev, seq_idx=0)
        save_labl_event_v3(fl, ev, seq_idx=0)
    return paths, ev


@pytest.mark.parametrize("t0", [0.0, 7.5, -3.2, 15.0])
def test_edep_time_passthrough(tmp_path, t0):
    """Stored seg.time equals the input — save_* applies no shift."""
    paths, ev = _write_batch(tmp_path, t0)
    seg = read_edep_event_v3(str(paths['edep']), 0)
    np.testing.assert_allclose(seg['time'], ev['segments']['time'], atol=1e-5)


@pytest.mark.parametrize("t0", [0.0, 7.5, -3.2])
def test_hits_and_sensor_T_passthrough(tmp_path, t0):
    """Stored inst.T and sensor.T equal the input — save_* applies no shift."""
    paths, ev = _write_batch(tmp_path, t0)
    inst = read_hits_event_v3(str(paths['hits']), 0)
    sensor = read_sensor_event_v3(str(paths['sensor']), 0)

    expected_t = np.array([50.0, 52.0, 55.0, 60.0, 61.0])
    np.testing.assert_allclose(inst['T'], expected_t, atol=1e-5)
    np.testing.assert_allclose(sensor['T'], expected_t, atol=1e-5)


def test_labl_t0_matches_input(tmp_path):
    paths, ev = _write_batch(tmp_path, 9.125)
    labl = read_labl_event_v3(str(paths['labl']), 0)
    # Single-interaction fixture → 1 row in per_interaction/
    assert len(labl['per_interaction']['t0']) == 1
    assert float(labl['per_interaction']['t0'][0]) == pytest.approx(ev['t0'])
