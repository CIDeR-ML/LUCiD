"""Per-track Cherenkov totals in labl agree with the per-segment counts in step."""
import h5py
import numpy as np
import pytest

from lucid.dataset.writer import (
    write_sensor_config, write_hits_config,
    write_step_config, write_labl_config,
    save_sensor_event, save_hits_event,
    save_step_event, save_labl_event,
)
from lucid.dataset.reader import (
    read_step_event,
)

from tests.helpers.event_fixture import build_synthetic_event


@pytest.fixture
def batch_paths(tmp_path):
    cfg, ev, sp = build_synthetic_event()
    src = np.array([ev['source_event_idx']], dtype=np.uint32)
    p_sensor = tmp_path / 'wc_sensor_0000.h5'
    p_hits = tmp_path / 'wc_hits_0000.h5'
    p_step = tmp_path / 'wc_step_0000.h5'
    p_labl = tmp_path / 'wc_labl_0000.h5'
    with h5py.File(p_sensor, 'w') as fs, h5py.File(p_hits, 'w') as fi, \
         h5py.File(p_step, 'w') as fg, h5py.File(p_labl, 'w') as fl:
        write_sensor_config(fs, cfg, src, sp)
        write_hits_config(fi, cfg, src, sp)
        write_step_config(fg, cfg, src)
        write_labl_config(fl, cfg, src)
        save_sensor_event(fs, ev, seq_idx=0)
        save_hits_event(fi, ev, seq_idx=0)
        save_step_event(fg, ev, seq_idx=0)
        save_labl_event(fl, ev, seq_idx=0)
    return p_step, p_labl, ev


def test_sum_ncherenkov_per_track(batch_paths):
    """On disk, labl per_track/n_cherenkov == sum of step n_cherenkov over the track's segments.

    The fixture's segment counts [2, 1, 1, 1] group by track to [3, 0, 2].
    """
    p_step, p_labl, _ = batch_paths
    edep = read_step_event(str(p_step), 0)
    with h5py.File(p_labl, 'r') as f:
        per_track = f['event_000/per_track/n_cherenkov'][:]
    summed = np.bincount(edep['track_idx'], weights=edep['n_cherenkov'],
                         minlength=per_track.size).astype(np.int32)
    np.testing.assert_array_equal(per_track, summed)
    np.testing.assert_array_equal(per_track, [3, 0, 2])
