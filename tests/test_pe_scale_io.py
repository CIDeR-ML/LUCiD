"""Tests for compare/pe_scale_io.py raw-cache FSMP streaming."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from compare import compare_sim_vs_real as response
from compare import pe_scale_io


def test_all_channel_fsmp_cache_streams_histograms_and_file_audit(
    tmp_path, monkeypatch
):
    """The all-channel cache retains binned counts, exposure and audit data."""
    schema = pa.schema(
        [
            pa.field("eid", pa.int64()),
            pa.field("ch", pa.int64()),
            pa.field("offset", pa.uint16()),
            pa.field("step", pa.uint16()),
            pa.field("PEt", pa.list_(pa.float64())),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([1, 1, 1, 2]),
            pa.array([5, 6, 5, 6]),
            pa.array([0, 0, 0, 0], type=pa.uint16()),
            pa.array([10, 10, 20, 10], type=pa.uint16()),
            pa.array([[0.1, 0.15], [0.2], [0.3], [0.4]]),
        ],
        schema=schema,
    )
    fsmp_root = tmp_path / "fsmp"
    fsmp_file = fsmp_root / "001" / "one.pq"
    fsmp_file.parent.mkdir(parents=True)
    pq.write_table(table, fsmp_file)
    monkeypatch.setattr(
        response,
        "_monitor_arrays",
        lambda _: (np.array([1, 2]), np.array([0.0, 0.0])),
    )

    records = pe_scale_io.read_fsmp_histograms_by_channel(
        [fsmp_file],
        fsmp_root,
        tmp_path / "rtraw",
        channels=np.array([5, 6]),
        edges=np.array([0.0, 0.2, 0.4]),
        run=1,
    )

    np.testing.assert_array_equal(
        records["histogram"], np.array([[2, 1], [0, 1]])
    )
    np.testing.assert_array_equal(records["histogram_entries"], [3, 1])
    assert int(records["physical_events"]) == 2
    assert int(records["event_samples"]) == 3
    np.testing.assert_array_equal(records["file_selected_pe"], [5])
    np.testing.assert_array_equal(records["file_event_samples"], [3])
    np.testing.assert_array_equal(records["file_channels_seen"], [2])
    assert records["files"].tolist() == [str(fsmp_file)]
