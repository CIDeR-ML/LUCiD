"""Build FSMP-compatible Parquet output for JUNO laser simulation.

The simulation output is one FSMP/BAPPE-compatible Parquet event table per
source position.  Its rows retain ``eid``, ``ch``, ``step``, ``PEt`` and
``nu_lc`` so the table can be consumed by the same reader contract as FSMP.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


SIMULATION_SCHEMA = pa.schema(
    [
        pa.field("eid", pa.uint64()),
        pa.field("ch", pa.uint16()),
        pa.field("offset", pa.uint16()),
        pa.field("step", pa.uint16()),
        pa.field("PEt", pa.list_(pa.float64())),
        pa.field("e", pa.list_(pa.int8())),
        pa.field("q", pa.list_(pa.float64())),
        pa.field("nu_lc", pa.float64()),
    ]
)


def simulation_table(
    event_id,
    pmt_ids,
    pe_times,
    *,
    n_channels: int | None = None,
    nu_lc=None,
):
    """Build one FSMP-style Parquet batch for a simulated event.

    Input ``pmt_ids`` and ``pe_times`` are flat accepted-PE arrays.  Output
    rows are grouped by PMT, while the event axis remains explicit in
    ``eid``.  ``offset`` and ``step`` are zero because this simulator has no
    FSMP time offset or Markov-chain step.  When ``n_channels`` is supplied,
    one row is emitted for every channel, including empty ``PEt`` lists as
    required by the FSMP full-step contract.  ``nu_lc`` may be a scalar or a
    per-channel array.
    """
    if int(event_id) < 0 or int(event_id) > np.iinfo(np.uint64).max:
        raise ValueError("event_id does not fit the uint64 Parquet schema")
    pmt_ids = np.asarray(pmt_ids, dtype=np.int64)
    pe_times = np.asarray(pe_times, dtype=np.float64)
    if pmt_ids.ndim != 1 or pe_times.ndim != 1:
        raise ValueError("pmt_ids and pe_times must be one-dimensional")
    if pmt_ids.shape != pe_times.shape:
        raise ValueError("pmt_ids and pe_times must have matching lengths")
    if np.any((pmt_ids < 0) | (pmt_ids > np.iinfo(np.uint16).max)):
        raise ValueError("PMT ids do not fit the uint16 Parquet schema")

    if n_channels is not None:
        n_channels = int(n_channels)
        if not 0 < n_channels <= np.iinfo(np.uint16).max + 1:
            raise ValueError("n_channels must fit the uint16 channel schema")
        if np.any(pmt_ids >= n_channels):
            raise ValueError("PMT ids exceed n_channels")

    order = np.argsort(pmt_ids, kind="mergesort")
    sorted_pmts = pmt_ids[order]
    unique_pmts, starts = np.unique(sorted_pmts, return_index=True)
    stops = np.r_[starts[1:], len(sorted_pmts)]
    pe_values = [
        pe_times[order[start:stop]].tolist()
        for start, stop in zip(starts, stops)
    ]
    if n_channels is not None:
        grouped = [[] for _ in range(n_channels)]
        for channel, values in zip(unique_pmts, pe_values):
            grouped[int(channel)] = values
        unique_pmts = np.arange(n_channels, dtype=np.int64)
        pe_values = grouped

    rows = len(unique_pmts)
    if nu_lc is None:
        nu_values = [np.nan] * rows
    else:
        nu_array = np.asarray(nu_lc, dtype=np.float64)
        if nu_array.ndim == 0:
            nu_values = [float(nu_array)] * rows
        elif n_channels is not None and nu_array.shape == (n_channels,):
            nu_values = nu_array.tolist()
        elif n_channels is None and nu_array.shape == (rows,):
            nu_values = nu_array.tolist()
        else:
            raise ValueError("nu_lc must be scalar or match output channels")
    arrays = [
        pa.array(np.full(rows, event_id, dtype=np.uint64), type=pa.uint64()),
        pa.array(unique_pmts.astype(np.uint16), type=pa.uint16()),
        pa.array(np.zeros(rows, dtype=np.uint16), type=pa.uint16()),
        pa.array(np.zeros(rows, dtype=np.uint16), type=pa.uint16()),
        pa.array(pe_values, type=pa.list_(pa.float64())),
        pa.array(
            [[None] * len(values) for values in pe_values],
            type=pa.list_(pa.int8()),
        ),
        pa.array(
            [[None] * len(values) for values in pe_values],
            type=pa.list_(pa.float64()),
        ),
        pa.array(nu_values, type=pa.float64()),
    ]
    return pa.Table.from_arrays(arrays, schema=SIMULATION_SCHEMA)
