"""Where batches are, in which order their events come back, and the batch check."""
import h5py

from lucid.dataset.layout import batch_paths, check_batch, list_batches
from lucid.dataset.reader import iter_events
from lucid.dataset.writer import write_batch
from lucid.production.dataprod.scripts.verify_jobs import is_complete

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


def _batch(root, n_events, file_index=0):
    cfg, ev, sensor_positions = build_synthetic_event()
    write_batch(root, file_index, [ev] * n_events, cfg, sensor_positions)
    ok, messages = check_batch(root, file_index)
    assert ok, messages


def _fails_both(root, tag, file_index=0, expected_events=None):
    """check_batch and verify_jobs (which calls it) both fail the batch, for the same reason."""
    ok, messages = check_batch(root, file_index, max_events=expected_events)
    assert not ok and any(m.startswith(tag) for m in messages), messages
    ok, reason = is_complete(root, file_index, expected_events)
    assert not ok and reason.startswith(tag), reason


def test_a_file_without_the_complete_flag_fails(tmp_path):
    """The writer sets config/complete only after the last event, so its absence means the file
    is not one this writer finished."""
    _batch(tmp_path, 1)
    with h5py.File(batch_paths(tmp_path, 0)['hits'], 'a') as f:
        del f['config'].attrs['complete']
    _fails_both(tmp_path, 'INCOMPLETE')


def test_a_gap_in_the_events_fails(tmp_path):
    _batch(tmp_path, 3)
    for path in batch_paths(tmp_path, 0).values():
        with h5py.File(path, 'a') as f:
            del f['event_001']
    _fails_both(tmp_path, 'BAD_EVENTS')


def test_a_lost_last_event_fails(tmp_path):
    _batch(tmp_path, 2)
    with h5py.File(batch_paths(tmp_path, 0)['labl'], 'a') as f:
        del f['event_001']
    _fails_both(tmp_path, 'BAD_EVENTS')


def test_a_file_from_another_attempt_fails(tmp_path):
    _batch(tmp_path, 1)
    with h5py.File(batch_paths(tmp_path, 0)['labl'], 'a') as f:
        f['config'].attrs['run_id'] = 'another attempt'
    _fails_both(tmp_path, 'RUN_ID_MISMATCH')


def test_a_wrong_file_index_fails(tmp_path):
    _batch(tmp_path, 1)
    with h5py.File(batch_paths(tmp_path, 0)['step'], 'a') as f:
        f['config'].attrs['file_index'] = 7
    _fails_both(tmp_path, 'BAD_FILE_INDEX')


def test_disagreeing_event_counts_fail(tmp_path):
    _batch(tmp_path, 2)
    with h5py.File(batch_paths(tmp_path, 0)['hits'], 'a') as f:
        f['config'].attrs['n_events'] = 1
        del f['event_001']
    _fails_both(tmp_path, 'N_EVENTS_MISMATCH')


def test_more_events_than_requested_fail(tmp_path):
    _batch(tmp_path, 2)
    _fails_both(tmp_path, 'TOO_MANY_EVENTS', expected_events=1)
    assert is_complete(tmp_path, 0, 2) == (True, "")


def test_only_the_declared_modalities_are_checked(tmp_path):
    _batch(tmp_path, 1)
    batch_paths(tmp_path, 0)['step'].unlink()
    assert not check_batch(tmp_path, 0)[0]
    ok, messages = check_batch(tmp_path, 0, modalities=('sensor', 'hits', 'labl'))
    assert ok, messages


def test_qa_counts_missing_batches_from_the_first_one(tmp_path):
    """A production need not start at file_index 0: only a hole between the first and the last
    batch is a missing one."""
    from lucid.production.qa import validations
    for k in (5, 7):
        _batch(tmp_path, 1, file_index=k)
    passed, _, info = validations.check_integrity(tmp_path)
    assert not passed and info["missing"] == [6] and info["corrupt"] == []
