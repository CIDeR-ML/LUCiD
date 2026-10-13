"""Tests for the JUNO forward-simulation Parquet output contract."""

import numpy as np
import pyarrow as pa

from lucid.juno_laser.data import SIMULATION_SCHEMA, simulation_table


def test_simulation_table_preserves_all_channels_and_pe_times():
    """Full-channel output retains empty channels and stable PE ordering."""
    table = simulation_table(
        7,
        np.array([2, 0, 2]),
        np.array([3.0, 1.0, 2.0]),
        n_channels=4,
        nu_lc=np.array([10.0, 11.0, 12.0, 13.0]),
    )

    assert table.schema == SIMULATION_SCHEMA
    assert table.column("eid").to_pylist() == [7, 7, 7, 7]
    assert table.column("ch").to_pylist() == [0, 1, 2, 3]
    assert table.column("PEt").to_pylist() == [
        [1.0],
        [],
        [3.0, 2.0],
        [],
    ]
    assert table.column("nu_lc").to_pylist() == [
        10.0,
        11.0,
        12.0,
        13.0,
    ]


def test_simulation_schema_matches_fsmp_types():
    """The forward writer keeps the documented FSMP-compatible types."""
    expected = {
        "eid": pa.uint64(),
        "ch": pa.uint16(),
        "offset": pa.uint16(),
        "step": pa.uint16(),
        "PEt": pa.list_(pa.float64()),
        "e": pa.list_(pa.int8()),
        "q": pa.list_(pa.float64()),
        "nu_lc": pa.float64(),
    }

    assert {field.name: field.type for field in SIMULATION_SCHEMA} == expected


def test_simulation_table_rejects_channel_outside_detector():
    """A PMT ID outside ``n_channels`` fails before writing Parquet."""
    with np.testing.assert_raises_regex(ValueError, "exceed n_channels"):
        simulation_table(
            0,
            np.array([3]),
            np.array([1.5]),
            n_channels=3,
        )
