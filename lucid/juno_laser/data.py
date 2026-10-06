"""Input handling for JUNO laser per-PE calibration data.

The simulation output is one FSMP/BAPPE-compatible Parquet event table per
source position.  Its rows retain ``eid``, ``ch``, ``step``, ``PEt`` and
``nu_lc`` so the table can be consumed by the same reader contract as FSMP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .time_grid import TimeGrid
from .eventwise import EventwisePeObservation


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


def event_file(data_dir: str | Path) -> Path:
    """Return the single consolidated Parquet event table for a source.

    The returned path follows the FSMP/BAPPE full-step file contract.  Event
    selection is applied after reading the table, rather than by selecting
    individual files, because one table contains many event IDs.
    """
    root = Path(data_dir).expanduser()
    if root.is_file():
        if root.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError(f"simulation event file must be Parquet: {root}")
        return root
    files = sorted(root.glob("*.parquet"))
    files.extend(sorted(root.glob("*.pq")))
    if not files:
        raise FileNotFoundError(f"No simulation Parquet file found in {root}")
    if len(files) > 1:
        raise ValueError(
            "expected one consolidated Parquet file in "
            f"{root}, found {len(files)}"
        )
    return files[0]


def _read_event_table(path: Path):
    """Read the FSMP/BAPPE columns and restore absolute PE times."""
    table = pq.read_table(
        path,
        columns=["eid", "ch", "offset", "step", "PEt", "nu_lc"],
    )
    frame = table.to_pandas()
    frame["PEt"] = [
        np.asarray([] if values is None else values, dtype=np.float64)
        + float(offset)
        for values, offset in zip(frame["PEt"], frame["offset"])
    ]
    return frame


def _event_arrays(frame):
    """Flatten selected BAPPE rows to per-PE event, channel and time arrays."""
    lengths = frame["PEt"].map(len).to_numpy(dtype=np.int64)
    if lengths.sum():
        times = np.concatenate(frame.loc[lengths > 0, "PEt"].to_numpy())
        events = np.repeat(
            frame.loc[lengths > 0, "eid"].to_numpy(dtype=np.int64),
            lengths[lengths > 0],
        )
        channels = np.repeat(
            frame.loc[lengths > 0, "ch"].to_numpy(dtype=np.int64),
            lengths[lengths > 0],
        )
    else:
        events = np.empty(0, dtype=np.int64)
        times = np.empty(0, dtype=np.float64)
        channels = np.empty(0, dtype=np.int64)
    return events, channels, times


def _select_fsmp_step(frame, *, posterior_step: int, max_events: int):
    """Select one prescribed FSMP posterior state for each physical event."""
    if posterior_step < 0:
        raise ValueError("posterior_step must be non-negative")
    event_labels = frame["eid"].drop_duplicates().to_numpy(dtype=np.int64)
    if max_events > 0:
        event_labels = event_labels[:max_events]
    selected = frame[
        frame["eid"].isin(event_labels) & (frame["step"] == posterior_step)
    ].copy()
    present = set(selected["eid"].unique().tolist())
    missing = [int(item) for item in event_labels if int(item) not in present]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"posterior step {posterior_step} is missing for physical "
            f"event IDs {preview}"
        )
    duplicate = selected.duplicated(subset=["eid", "ch", "step"])
    if duplicate.any():
        raise ValueError(
            "FSMP input has duplicate rows for the same (eid, ch, step)"
        )
    return selected, event_labels


def load_eventwise_pe_observation(
    data_dir: str | Path,
    *,
    n_sensors: int,
    time_grid: TimeGrid,
    posterior_step: int,
    event_exposure,
    event_time_offset_ns=0.0,
    max_events: int = 0,
) -> EventwisePeObservation:
    """Load one FSMP posterior step while retaining physical zero-PE events.

    ``eid`` is the physical laser-event ledger.  ``step`` indexes posterior
    samples within that event and is therefore not counted as an independent
    exposure.  This prototype selects one prescribed step for every retained
    event; it does not marginalize over posterior samples.  ``event_exposure``
    is an externally measured or explicitly fixed laser normalization.  The
    FSMP ``nu_lc`` column is not interpreted as exposure.
    """
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid")
    frame = _read_event_table(event_file(data_dir))
    frame, event_labels = _select_fsmp_step(
        frame, posterior_step=posterior_step, max_events=max_events
    )
    exposures = np.asarray(event_exposure, dtype=np.float64)
    if exposures.ndim == 0:
        exposures = np.full(len(event_labels), float(exposures))
    if exposures.shape != (len(event_labels),):
        raise ValueError(
            "event exposure must be scalar or match retained events"
        )
    if np.any(~np.isfinite(exposures)) or np.any(exposures <= 0.0):
        raise ValueError("event exposures must be finite and positive")
    time_offsets = np.asarray(event_time_offset_ns, dtype=np.float64)
    if time_offsets.ndim == 0:
        time_offsets = np.full(len(event_labels), float(time_offsets))
    if time_offsets.shape != (len(event_labels),):
        raise ValueError(
            "event time offset must be scalar or match retained events"
        )
    if np.any(~np.isfinite(time_offsets)):
        raise ValueError("event time offsets must be finite")
    invalid_channel = (frame["ch"] < 0) | (frame["ch"] >= n_sensors)
    rows_per_event = frame.groupby("eid", sort=False).size()
    incomplete = rows_per_event[rows_per_event != n_sensors].index.tolist()
    if invalid_channel.any():
        incomplete.extend(frame.loc[invalid_channel, "eid"].unique().tolist())
    incomplete = sorted({int(label) for label in incomplete})
    if incomplete:
        raise ValueError(
            "FSMP full-step contract requires one row per channel; "
            f"incomplete physical event IDs {incomplete[:5]}"
        )
    events, channels, times = _event_arrays(frame)
    event_lookup = {
        int(label): index for index, label in enumerate(event_labels)
    }
    remapped_events = np.fromiter(
        (event_lookup[int(item)] for item in events),
        dtype=np.int32,
        count=len(events),
    )
    valid = (
        (channels >= 0)
        & (channels < n_sensors)
        & np.isfinite(times)
        & (times >= time_grid.time_min_ns)
        & (times < time_grid.time_max_ns)
    )
    return EventwisePeObservation(
        event_ids=remapped_events[valid],
        pmt_ids=channels[valid],
        pe_times_ns=times[valid],
        n_events=len(event_labels),
        n_sensors=n_sensors,
        time_grid=time_grid,
        event_exposures=exposures,
        event_time_offsets_ns=time_offsets,
        event_labels=event_labels,
        posterior_step=posterior_step,
    )
