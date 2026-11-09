"""Verify writer time-frame semantics.

From v4 on, save_* no longer applies any t0 shift: input times are
expected already in absolute detector frame (the production drivers add
per-interaction t0 before saving). labl/per_interaction/t0 carries
the per-interaction metadata for downstream reconstruction.
"""
import numpy as np

from lucid.dataset.reader import (
    read_sensor_event, read_hits_event, read_step_event,
)

from tests.helpers.event_fixture import build_synthetic_event, write_batch


def test_writers_store_times_unshifted(tmp_path):
    """sensor T, hits T and step time are stored exactly as handed in.

    The fixture's labl t0 is 7.5 ns; a writer that subtracted (or added) it
    would move every column by 7.5 ns.
    """
    cfg, ev, sp = build_synthetic_event(t0=7.5)
    paths = write_batch(tmp_path, cfg, ev, sp)
    seg = read_step_event(str(paths['step']), 0)
    inst = read_hits_event(str(paths['hits']), 0)
    sensor = read_sensor_event(str(paths['sensor']), 0)
    expected_t = np.array([50.0, 52.0, 55.0, 60.0, 61.0])
    np.testing.assert_allclose(seg['time'], ev['segments']['time'], atol=1e-5)
    np.testing.assert_allclose(inst['T'], expected_t, atol=1e-5)
    np.testing.assert_allclose(sensor['T'], expected_t, atol=1e-5)
