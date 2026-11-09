"""Where batches are, in which order their events come back, and the batch check."""
import h5py

from lucid.dataset.layout import batch_paths, check_batch, list_batches
from lucid.dataset.reader import iter_events
from lucid.dataset.writer import write_batch

from tests.helpers.event_fixture import build_synthetic_event


def test_events_come_back_in_numeric_order(tmp_path):
    """``event_{i:03d}`` sorts lexicographically wrong past 999: event_1000 before event_101."""
    path = tmp_path / 'f.h5'
    with h5py.File(path, 'w') as f:
        for i in (1000, 101, 99, 0):
            f.create_group(f'event_{i:03d}')
        f.create_group('config')
    assert [i for i, _ in iter_events(path)] == [0, 99, 101, 1000]
    assert [name for _, name in iter_events(path)][-1] == 'event_1000'


def test_written_batches_are_found_and_pass_the_check(tmp_path):
    cfg, ev, sensor_positions = build_synthetic_event()
    for k in (10, 2):
        paths = write_batch(tmp_path, k, [ev], cfg, sensor_positions)
        assert paths == [str(p) for p in batch_paths(tmp_path, k).values()]
    assert list_batches(tmp_path) == [2, 10]
    ok, messages = check_batch(tmp_path, 10, expected_dataset_name=cfg['dataset_name'])
    assert ok, messages
    ok, messages = check_batch(tmp_path, 10, expected_dataset_name='another name')
    assert not ok and any('BAD_DATASET_NAME' in m for m in messages)
