#!/usr/bin/env python
"""Compare FSMP data with LUCiD and JUNOsw laser simulations.

Common primitives come from :mod:`compare.compare_sim`. This entry module
owns input/cache orchestration and the CLI; metric construction, CSV output
and PDF plotting live in dedicated ``laser_compare_*`` modules.

FSMP ``PEt`` is a list column.  Its values are converted to NumPy arrays;
``offset`` is added before subtracting the monitor time, and every PE in each
stored row is retained.
LUCiD simulation input uses the same ``eid``/``ch``/``offset``/``step``/``PEt``
contract as FSMP.  Only consolidated Parquet input is accepted; ``--sim-dir``
may contain a shell-style wildcard for matching one source-position folder.
The default plot panel contains 72 fixed geometry-stratified PMTs; a larger
seeded theta/type-stratified panel is available via --channel-mode
stratified.  All inputs are restricted to that same subset.  FSMP dark noise
is retained; its per-channel sideband estimate is added to every simulation
source.  FSMP loading covers every input file once, and source caches are
validated by content signature so unchanged FSMP/JUNOsw caches are reused
when only the LUCiD source changes.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Make direct ``python compare/compare_sim_vs_real.py`` execution resolve the
# repository package without requiring a caller-specific PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import compare_sim as base
except ImportError:  # Direct ``python compare/compare_sim_vs_real.py``.
    import compare_sim as base

from compare.laser_compare_metrics import (
    JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
    build_channel_histograms_multi,
    consensus_l1_alignment,
    density_hist,
    estimate_peak_from_density,
    processing_string,
    resolve_irf_sigmas,
    simulation_qe_factors,
)
from compare.laser_compare_channels import (
    representative_channels,
    stratified_channels,
)
from compare.laser_compare_csv import (
    write_channel_manifest,
    write_comparison_stats_csv,
    write_fsmp_file_manifest,
)
from compare.laser_compare_plotting import (
    plot_amplitude_data,
    plot_data,
    plot_summary_data,
    pooled_group_shape,
)
NORTH_CHANNEL_COUNT = base.NORTH_CHANNEL_COUNT
BRIEF_FSMP_FILE_LIMIT = base.BRIEF_FSMP_FILE_LIMIT
BRIEF_SIM_FILE_LIMIT = base.BRIEF_SIM_FILE_LIMIT

DEFAULT_RUN_TABLE = "/mnt/stage/haoch/laser/run_532.csv"
DEFAULT_RTRAW_DIR = "/mnt/stage/haoch/laser/split_rtraw"
DEFAULT_FSMP_ROOT = "/junofs/C/PE"
DEFAULT_PMT_MAP = "/junofs/C/junoid_J25.7.3.pq"
DEFAULT_PMT_FILE = "/junofs/C/PMT/STD/9541.h5"
DEFAULT_LUCID_LABEL = "LUCiD"
DEFAULT_JUNOSW_LABEL = "JUNOsw (J26.3.2)"
DEFAULT_IRF_SIGMA_HAMAMATSU = 1.3
DEFAULT_IRF_SIGMA_NNVT = 7.0
FSMP_CACHE_SCHEMA = 7

resolve_run = base.resolve_run
iter_fsmp_files = base.iter_fsmp_files
rtraw_path_for = base.rtraw_path_for
read_monitor_times = base.read_monitor_times
select_channels = base.select_channels
read_pmt_info = base.read_pmt_info
load_data = base.load_data


def attach_pmt_coordinates(fsmp_data, pmt_map_file):
    """Attach detector z, theta and phi coordinates to selected PMTs."""
    result = base.attach_pmt_coordinates(fsmp_data, pmt_map_file)
    table = pq.read_table(pmt_map_file, columns=["copy_number", "phi"])
    channel = table["copy_number"].to_numpy(zero_copy_only=False)
    phi = table["phi"].to_numpy(zero_copy_only=False)
    phi_by_channel = dict(zip(channel, phi))
    result["phi"] = np.asarray(
        [phi_by_channel.get(value, np.nan) for value in result["channels"]],
        dtype=np.float64,
    )
    return result




def _arrow_pet_values(column):
    """Return flattened PE values and row lengths from an Arrow column."""
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        offsets = column.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        lengths = np.diff(offsets)
        valid = column.is_valid().to_numpy(zero_copy_only=False)
        if not np.all(valid):
            return _python_pet_values(column.to_pylist())
        lengths = np.where(valid, lengths, 0)
        values = column.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float64), lengths, valid

    return _python_pet_values(column.to_pylist())


def _python_pet_values(rows):
    """Flatten generic Python list values when Arrow buffers are unsuitable."""
    values = []
    lengths = []
    valid = []
    for value in rows:
        if value is None:
            valid.append(False)
            lengths.append(0)
        elif np.ndim(value) == 0:
            valid.append(True)
            lengths.append(1)
            values.append(float(value))
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            valid.append(True)
            lengths.append(len(array))
            values.extend(array.tolist())
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(valid, dtype=bool),
    )


def _monitor_arrays(path):
    """Load one rtraw monitor file as sorted unique NumPy arrays."""
    frame = read_monitor_times(path)
    eid = frame["eid"].to_numpy(dtype=np.int64)
    time = frame["t_monitor"].to_numpy(dtype=np.float64)
    order = np.argsort(eid, kind="mergesort")
    eid, time = eid[order], time[order]
    eid, first = np.unique(eid, return_index=True)
    return eid, time[first]


def _monitor_match(eid, monitor_eid, monitor_time):
    """Vectorized monitor lookup for a batch of FSMP event ids."""
    pos = np.searchsorted(monitor_eid, eid)
    valid = pos < len(monitor_eid)
    matched = np.zeros(len(eid), dtype=np.float64)
    matched[valid] = monitor_time[pos[valid]]
    valid[valid] = monitor_eid[pos[valid]] == eid[valid]
    return matched, valid


def _append_fsmp_batch(
    chunks,
    batch,
    channels,
    monitor_eid,
    monitor_time,
    first_pe_only=False,
    monitor_join=True,
):
    """Convert one Arrow record batch to flat, monitor-corrected arrays."""
    eid = batch.column("eid").to_numpy(zero_copy_only=False).astype(np.int64)
    channel = batch.column("ch").to_numpy(zero_copy_only=False).astype(np.int64)
    offset = (
        batch.column("offset").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    values, lengths, valid_pet = _arrow_pet_values(batch.column("PEt"))
    if monitor_join:
        monitor, valid_monitor = _monitor_match(
            eid, monitor_eid, monitor_time
        )
    else:
        monitor = np.zeros(len(eid), dtype=np.float64)
        valid_monitor = np.ones(len(eid), dtype=bool)
    valid_row = valid_pet & valid_monitor & (lengths > 0)
    if not np.any(valid_row):
        return 0

    if first_pe_only:
        row_start = np.cumsum(lengths, dtype=np.int64) - lengths
        if np.any(valid_row):
            chunks.append(
                (
                    channel[valid_row],
                    eid[valid_row],
                    values[row_start[valid_row]]
                    + offset[valid_row]
                    - monitor[valid_row],
                    np.ones(np.count_nonzero(valid_row), dtype=np.int64),
                )
            )
        return int(np.count_nonzero(valid_row))

    row_channel = np.repeat(channel, lengths)
    row_eid = np.repeat(eid, lengths)
    row_offset = np.repeat(offset, lengths)
    row_monitor = np.repeat(monitor, lengths)
    value_valid = np.repeat(valid_row, lengths)
    if np.any(value_valid):
        chunks.append(
            (
                row_channel[value_valid],
                row_eid[value_valid],
                values[value_valid]
                + row_offset[value_valid]
                - row_monitor[value_valid],
                np.ones(np.count_nonzero(value_valid), dtype=np.int64),
            )
        )
    return int(np.count_nonzero(value_valid))


def read_fsmp_times_by_channel(
    fsmp_files,
    fsmp_dir,
    rtraw_dir,
    channels,
    run,
    file_limit=None,
    print_first_row=False,
    batch_size=250_000,
    target_pe=None,
    first_pe_only=False,
    monitor_join=True,
):
    """Read FSMP Parquet until the detector-wide PE target is reached.

    The current file is always read to completion. Once its accepted PE count
    makes the cumulative total reach ``target_pe``, later files are skipped.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if target_pe is not None and target_pe < 0:
        raise ValueError("target_pe must be non-negative")
    if target_pe is not None:
        target_pe = int(target_pe)
    channels = np.asarray(channels, dtype=np.int64)
    chunks = []
    readable = 0
    loaded_pe = 0
    physical_events = 0
    event_samples = 0
    printed = False
    iterator = (
        fsmp_files
        if file_limit is not None
        else tqdm(fsmp_files, desc=f"RUN {run} FSMP files")
    )
    for path in iterator:
        if file_limit is not None and readable >= file_limit:
            break
        try:
            if monitor_join:
                monitor_eid, monitor_time = _monitor_arrays(
                    rtraw_path_for(path, fsmp_dir, rtraw_dir)
                )
            else:
                monitor_eid = np.empty(0, dtype=np.int64)
                monitor_time = np.empty(0, dtype=np.float64)
            parquet = pq.ParquetFile(path)
            file_event_steps = set()
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=["eid", "ch", "offset", "step", "PEt"],
                use_threads=False,
            ):
                row_channel = batch.column("ch").to_numpy(zero_copy_only=False)
                row_eid = batch.column("eid").to_numpy(zero_copy_only=False)
                row_step = batch.column("step").to_numpy(zero_copy_only=False)
                monitor_keep = (
                    np.isin(row_eid, monitor_eid)
                    if monitor_join
                    else np.ones(len(row_eid), dtype=bool)
                )
                if np.any(monitor_keep):
                    pairs = np.unique(
                        np.column_stack(
                            (row_eid[monitor_keep], row_step[monitor_keep])
                        ),
                        axis=0,
                    )
                    file_event_steps.update(map(tuple, pairs.tolist()))
                keep = np.isin(row_channel, channels)
                if not np.any(keep):
                    continue
                selected = batch.filter(pa.array(keep))
                if print_first_row and not printed and selected.num_rows:
                    printed = True
                    print(
                        "FSMP sample row:", selected.slice(0, 1).to_pylist()[0]
                    )
                loaded_pe += _append_fsmp_batch(
                    chunks,
                    selected,
                    channels,
                    monitor_eid,
                    monitor_time,
                    first_pe_only=first_pe_only,
                    monitor_join=monitor_join,
                )
            physical_events += (
                len(monitor_eid)
                if monitor_join
                else len(file_event_steps)
            )
            event_samples += len(file_event_steps)
            readable += 1
            if target_pe is not None and loaded_pe >= target_pe:
                break
        except (OSError, KeyError, ValueError, pa.ArrowException) as error:
            if file_limit is None:
                raise
            print(f"Skip unreadable FSMP file {path}: {error}")

    if not chunks:
        return {
            "ch": np.empty(0, dtype=np.int64),
            "eid": np.empty(0, dtype=np.int64),
            "time": np.empty(0, dtype=np.float64),
            "pe_count": np.empty(0, dtype=np.int64),
            "physical_events": np.asarray(physical_events, dtype=np.int64),
            "event_samples": np.asarray(event_samples, dtype=np.int64),
        }
    channel, eid, time, pe_count = map(np.concatenate, zip(*chunks))
    return {
        "ch": channel.astype(np.int64, copy=False),
        "eid": eid.astype(np.int64, copy=False),
        "time": time.astype(np.float64, copy=False),
        "pe_count": pe_count.astype(np.int64, copy=False),
        "physical_events": np.asarray(physical_events, dtype=np.int64),
        "event_samples": np.asarray(event_samples, dtype=np.int64),
    }


def read_fsmp_histograms_by_channel(
    fsmp_files,
    fsmp_dir,
    rtraw_dir,
    channels,
    edges,
    run,
    file_limit=None,
    batch_size=250_000,
    target_pe=None,
    first_pe_only=False,
    monitor_join=True,
):
    """Stream FSMP into per-channel histograms for an all-channel cache.

    This intentionally stores no PE-time list.  With ``first_pe_only``, one
    PE is retained per nonempty PEt record, matching the reference counter.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    channels = np.asarray(channels, dtype=np.int64)
    histogram = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_pe = np.zeros(len(channels), dtype=np.int64)
    physical_events = event_samples = loaded_pe = readable = 0
    manifest_path, manifest_pe, manifest_exposure, manifest_coverage = (
        [], [], [], []
    )
    iterator = (
        fsmp_files
        if file_limit is not None
        else tqdm(fsmp_files, desc=f"RUN {run} FSMP files")
    )
    for path in iterator:
        if file_limit is not None and readable >= file_limit:
            break
        try:
            if monitor_join:
                monitor_eid, monitor_time = _monitor_arrays(
                    rtraw_path_for(path, fsmp_dir, rtraw_dir)
                )
            else:
                monitor_eid = np.empty(0, dtype=np.int64)
                monitor_time = np.empty(0, dtype=np.float64)
            parquet = pq.ParquetFile(path)
            file_event_steps, file_pe, seen = set(), 0, set()
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=["eid", "ch", "offset", "step", "PEt"],
                use_threads=False,
            ):
                row_channel = batch.column("ch").to_numpy(
                    zero_copy_only=False
                )
                row_eid = batch.column("eid").to_numpy(zero_copy_only=False)
                row_step = batch.column("step").to_numpy(
                    zero_copy_only=False
                )
                monitor_keep = (
                    np.isin(row_eid, monitor_eid)
                    if monitor_join
                    else np.ones(len(row_eid), dtype=bool)
                )
                if np.any(monitor_keep):
                    pairs = np.unique(
                        np.column_stack(
                            (row_eid[monitor_keep], row_step[monitor_keep])
                        ),
                        axis=0,
                    )
                    file_event_steps.update(map(tuple, pairs.tolist()))
                values, lengths, valid_pet = _arrow_pet_values(
                    batch.column("PEt")
                )
                if monitor_join:
                    monitor, valid_monitor = _monitor_match(
                        row_eid, monitor_eid, monitor_time
                    )
                else:
                    monitor = np.zeros(len(row_eid), dtype=np.float64)
                    valid_monitor = np.ones(len(row_eid), dtype=bool)
                row_index = np.searchsorted(channels, row_channel)
                selected = valid_pet & valid_monitor & (lengths > 0)
                selected &= row_index < len(channels)
                selected[selected] = (
                    channels[row_index[selected]] == row_channel[selected]
                )
                if not np.any(selected):
                    continue
                if first_pe_only:
                    row_start = (
                        np.cumsum(lengths, dtype=np.int64) - lengths
                    )
                    flat_channel = row_channel[selected]
                    flat_time = (
                        values[row_start[selected]]
                        + batch.column("offset").to_numpy(
                            zero_copy_only=False
                        )[selected]
                        - monitor[selected]
                    )
                else:
                    flat_selected = np.repeat(selected, lengths)
                    flat_channel = np.repeat(row_channel, lengths)[
                        flat_selected
                    ]
                    flat_time = (
                        values
                        + np.repeat(
                            batch.column("offset").to_numpy(
                                zero_copy_only=False
                            ),
                            lengths,
                        )
                        - np.repeat(monitor, lengths)
                    )[flat_selected]
                _accumulate_histogram(
                    flat_channel, flat_time, channels, edges, histogram, entries
                )
                flat_index = np.searchsorted(channels, flat_channel)
                np.add.at(total_pe, flat_index, 1)
                file_pe += int(len(flat_channel))
                seen.update(np.unique(flat_channel).tolist())
            physical_events += (
                len(monitor_eid)
                if monitor_join
                else len(file_event_steps)
            )
            file_exposure = len(file_event_steps)
            event_samples += file_exposure
            manifest_path.append(str(path))
            manifest_pe.append(file_pe)
            manifest_exposure.append(file_exposure)
            manifest_coverage.append(len(seen))
            loaded_pe += file_pe
            readable += 1
            if target_pe is not None and loaded_pe >= target_pe:
                break
        except (OSError, KeyError, ValueError, pa.ArrowException) as error:
            if file_limit is None:
                raise
            print(f"Skip unreadable FSMP file {path}: {error}")
    return {
        "histogram": histogram,
        "histogram_entries": entries,
        "total_pe": total_pe,
        "physical_events": np.asarray(physical_events, dtype=np.int64),
        "event_samples": np.asarray(event_samples, dtype=np.int64),
        "files": np.asarray(manifest_path),
        "file_selected_pe": np.asarray(manifest_pe, dtype=np.int64),
        "file_event_samples": np.asarray(manifest_exposure, dtype=np.int64),
        "file_channels_seen": np.asarray(manifest_coverage, dtype=np.int64),
    }


def selected_channels(args, require_files=True, files=None):
    """Select final plot channels and optionally locate FSMP files."""
    files = list(iter_fsmp_files(args.fsmp_dir)) if files is None else files
    if require_files and not files:
        raise FileNotFoundError(
            f"No FSMP Parquet files found in {args.fsmp_dir}"
        )
    if args.north:
        channels = np.arange(NORTH_CHANNEL_COUNT, dtype=np.int64)
    elif args.channel_mode == "representative":
        channels = representative_channels(args.pmt_map)
    elif args.channel_mode == "stratified":
        channels = stratified_channels(
            args.pmt_map,
            stratum_size=args.stratum_size,
            seed=args.seed,
        )
    elif args.channel_mode == "all":
        channels = np.arange(base.FULL_CHANNEL_COUNT, dtype=np.int64)
    else:
        channels = select_channels(args.number, args.seed)
    mode = "north" if args.north else args.channel_mode
    print(f"Channel mode: {mode}; selected {len(channels)} channels")
    print("Selected channels:", " ".join(map(str, channels)))
    return channels, files


def _accumulate_histogram(pmt, times, channels, edges, histogram, entries):
    """Accumulate selected PMT times into an integer histogram."""
    index = np.searchsorted(channels, pmt)
    valid = index < len(channels)
    valid[valid] = channels[index[valid]] == pmt[valid]
    if not np.any(valid):
        return
    index = index[valid]
    bins = np.searchsorted(edges, times[valid], side="right") - 1
    valid_bin = (bins >= 0) & (bins < histogram.shape[1])
    np.add.at(entries, index, valid_bin.astype(np.int64))
    if np.any(valid_bin):
        np.add.at(histogram, (index[valid_bin], bins[valid_bin]), 1)


def _density_from_histogram(histogram, entries, edges):
    width = float(edges[1] - edges[0])
    norm = entries[:, None] * width
    return np.divide(
        histogram.astype(np.float64),
        norm,
        out=np.zeros_like(histogram, dtype=np.float64),
        where=norm > 0,
    )


def _simulation_files(path, pattern):
    if path.is_file():
        return [path]
    return sorted(path.rglob(pattern))


def iter_sim_files(sim_dir):
    """Return consolidated LUCiD Parquet files, expanding path wildcards."""
    raw_path = str(sim_dir)
    if any(token in raw_path for token in "*?["):
        roots = [Path(value) for value in sorted(glob.glob(raw_path))]
    else:
        roots = [Path(sim_dir)]
    files = []
    for path in roots:
        if path.is_file():
            if path.suffix.lower() not in {".parquet", ".pq"}:
                raise ValueError(
                    f"LUCiD simulation input must be Parquet: {path}"
                )
            files.append(path)
        elif path.is_dir():
            files.extend(_simulation_files(path, "*.parquet"))
            files.extend(_simulation_files(path, "*.pq"))
    return list(dict.fromkeys(files))


def iter_junosw_sim_files(sim_dir, file_glob="*.h5"):
    """Return JUNOsw HDF5 files from a file or directory."""
    return _simulation_files(Path(sim_dir), file_glob)


def _source_signature(files):
    """Return a JSON signature of path/size/mtime for cache validation."""
    records = []
    for path in files:
        stat = Path(path).stat()
        records.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return json.dumps(records)


def _effective_fsmp_files(args, files):
    """Return the FSMP files the reader will actually attempt."""
    limit = args.fsmp_file_limit
    if limit is None and args.brief:
        limit = BRIEF_FSMP_FILE_LIMIT
    return files if limit is None else files[:limit]


def _planned_sim_files(args, kind, sim_dir=None):
    """Return the simulation files a cache build would read."""
    if kind == "lucid":
        root = sim_dir if sim_dir is not None else args.lucid_sim_dir[0]
        files = iter_sim_files(root)
        limit = args.lucid_file_limit
    else:
        files = iter_junosw_sim_files(
            args.junosw_sim_dir, args.junosw_sim_file_glob
        )
        limit = args.junosw_file_limit
    if limit is not None:
        files = files[:limit]
    if args.brief:
        files = files[:BRIEF_SIM_FILE_LIMIT]
    return files


def _read_lucid_parquet(path, edges, chunk_size, channels):
    """Accumulate one FSMP-style LUCiD Parquet file into histograms."""
    parquet = pq.ParquetFile(path)
    histogram = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_pe = np.zeros(len(channels), dtype=np.int64)
    metadata = parquet.schema_arrow.metadata or {}
    metadata_events = metadata.get(b"n_events")
    event_ids = None if metadata_events is not None else set()
    for batch in parquet.iter_batches(
        batch_size=chunk_size,
        columns=["eid", "ch", "offset", "PEt"],
        use_threads=False,
    ):
        row_channel = batch.column("ch").to_numpy(zero_copy_only=False)
        row_eid = batch.column("eid").to_numpy(zero_copy_only=False)
        row_offset = batch.column("offset").to_numpy(zero_copy_only=False)
        values, lengths, valid_pet = _arrow_pet_values(batch.column("PEt"))
        if event_ids is not None:
            event_ids.update(np.unique(row_eid).tolist())
        row_index = np.searchsorted(channels, row_channel)
        valid_row = valid_pet & (lengths > 0) & (row_index < len(channels))
        valid_row[valid_row] = (
            channels[row_index[valid_row]] == row_channel[valid_row]
        )
        if not np.any(valid_row):
            continue
        flat_channel = np.repeat(row_channel, lengths)
        flat_offset = np.repeat(row_offset, lengths)
        flat_valid = np.repeat(valid_row, lengths)
        if np.any(flat_valid):
            flat_times = values + flat_offset
            _accumulate_histogram(
                flat_channel[flat_valid],
                flat_times[flat_valid],
                channels,
                edges,
                histogram,
                entries,
            )
            valid_index = np.searchsorted(
                channels, flat_channel[flat_valid]
            )
            np.add.at(total_pe, valid_index, 1)
    event_count = (
        int(metadata_events.decode())
        if metadata_events is not None
        else len(event_ids)
    )
    return histogram, entries, total_pe, event_count


def read_lucid_sim_histograms(sim_files, edges, chunk_size, channels):
    """Read LUCiD Parquet files into density and count arrays."""
    if chunk_size <= 0:
        raise ValueError("--sim-chunk-size must be positive")
    channels = np.asarray(channels, dtype=np.int64)
    histogram = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_pe = np.zeros(len(channels), dtype=np.int64)
    events = np.zeros(len(channels), dtype=np.int64)
    for path in tqdm(sim_files, desc="LUCiD simulation files"):
        try:
            parquet_hist = _read_lucid_parquet(
                path, edges, chunk_size, channels
            )
        except (
            OSError,
            KeyError,
            ValueError,
            TypeError,
            pa.ArrowException,
        ) as error:
            print(f"Skip invalid LUCiD file {path}: {error}")
            continue
        histogram += parquet_hist[0]
        entries += parquet_hist[1]
        total_pe += parquet_hist[2]
        events += parquet_hist[3]
    return (
        _density_from_histogram(histogram, entries, edges),
        entries,
        total_pe,
        events,
    )


def _junosw_event_count(files, sim_group):
    """Count generated JUNOsw events, including events without selected hits."""
    count = 0
    for path in files:
        with base.h5.File(path, "r") as stream:
            header_path = f"{sim_group}/Header"
            if header_path in stream:
                count += len(stream[header_path])
                continue
            track = stream[f"{sim_group}/Track"]
            event_id = base.read_compound_fields(track, ("event_id",))[
                "event_id"
            ]
            count += len(np.unique(event_id))
    return count


def _read_junosw_histograms(files, args, edges, channels):
    """Delegate JUNOsw Track/CDHit matching to the reference module."""
    density, entries = base.read_sim_histograms(
        files,
        args.junosw_sim_group,
        edges,
        args.sim_chunk_size,
        args.junosw_sim_match_mode,
        channels,
    )
    event_count = _junosw_event_count(files, args.junosw_sim_group)
    return (
        density,
        entries,
        entries.copy(),
        np.full(len(channels), event_count, dtype=np.int64),
    )


def _compute_sim_cache(
    args, channels, kind, sim_dir=None, label=None, output=None
):
    """Build one simulation cache in the common source schema."""
    edges = np.linspace(args.data_range[0], args.data_range[1], args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    files = _planned_sim_files(args, kind, sim_dir=sim_dir)
    if not files:
        raise FileNotFoundError(
            f"No {kind} simulation input files for a new cache"
        )
    if kind == "lucid":
        result = read_lucid_sim_histograms(
            files, edges, args.sim_chunk_size, channels
        )
        label = label if label is not None else args.lucid_label[0]
        output = output if output is not None else args.sim_data[0]
    else:
        result = _read_junosw_histograms(files, args, edges, channels)
        label, output = args.junosw_label, args.junosw_sim_data
    density, entries, total_pe, events = result
    if not np.any(entries):
        raise RuntimeError(f"No {kind} simulation time entries")
    data = {
        "channels": channels,
        "edges": edges,
        "centers": centers,
        "simulation_raw_density": density,
        "simulation_entries": entries,
        "simulation_total_pe": total_pe,
        "simulation_events": events,
        "amplitude_schema": np.asarray(3, dtype=np.int16),
        "label": np.asarray(label),
        "kind": np.asarray(kind),
        "source_signature": np.asarray(_source_signature(files)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f"Wrote {output}")
    return data


def compute_fsmp_data(args, channels, fsmp_files):
    """Build the FSMP cache from Arrow/NumPy arrays."""
    target_pe = getattr(args, "fsmp_target_pe", None)
    monitor_join = True
    signature = _source_signature(_effective_fsmp_files(args, fsmp_files))
    if target_pe is not None:
        print(
            "FSMP PE target from selected-channel LUCiD simulation: "
            f"{int(target_pe)}"
        )
    pmt_type, tts = read_pmt_info(channels, args.pmt_map, args.pmt)
    if args.channel_mode == "all" and not args.north:
        edges = np.linspace(
            args.data_range[0], args.data_range[1], args.bins + 1
        )
        records = read_fsmp_histograms_by_channel(
            fsmp_files,
            args.fsmp_dir,
            args.rtraw_dir,
            channels,
            edges,
            args.run,
            args.fsmp_file_limit
            if args.fsmp_file_limit is not None
            else (BRIEF_FSMP_FILE_LIMIT if args.brief else None),
            target_pe=target_pe,
            first_pe_only=False,
            monitor_join=monitor_join,
        )
        if not np.any(records["total_pe"]):
            raise RuntimeError(
                f"No RUN {args.run} FSMP entries after input selection"
            )
        data = {
            "channels": channels,
            "pmt_type": pmt_type,
            "tts": tts,
            "fsmp_histogram": records["histogram"],
            "fsmp_histogram_entries": records["histogram_entries"],
            "fsmp_total_pe": records["total_pe"],
            "fsmp_histogram_edges": edges,
            "fsmp_physical_events": records["physical_events"],
            "fsmp_event_samples": records["event_samples"],
            "fsmp_files_read": records["files"],
            "fsmp_file_selected_pe": records["file_selected_pe"],
            "fsmp_file_event_samples": records["file_event_samples"],
            "fsmp_file_channels_seen": records["file_channels_seen"],
            "fsmp_cache_schema": np.asarray(
                FSMP_CACHE_SCHEMA, dtype=np.int16
            ),
            "source_signature": np.asarray(signature),
        }
    else:
        records = read_fsmp_times_by_channel(
            fsmp_files,
            args.fsmp_dir,
            args.rtraw_dir,
            channels,
            args.run,
            file_limit=args.fsmp_file_limit
            if args.fsmp_file_limit is not None
            else (BRIEF_FSMP_FILE_LIMIT if args.brief else None),
            print_first_row=args.print_fsmp_row,
            target_pe=target_pe,
            first_pe_only=False,
            monitor_join=monitor_join,
        )
        if not len(records["time"]):
            raise RuntimeError(
                f"No RUN {args.run} FSMP entries after monitor matching"
            )
        data = {
            "channels": channels,
            "pmt_type": pmt_type,
            "tts": tts,
            "fsmp_ch": records["ch"],
            "fsmp_eid": records["eid"],
            "fsmp_time": records["time"],
            "fsmp_pe_count": records["pe_count"],
            "fsmp_total_pe": np.bincount(
                records["ch"], minlength=len(channels)
            ),
            "fsmp_physical_events": records["physical_events"],
            "fsmp_event_samples": records["event_samples"],
            "fsmp_cache_schema": np.asarray(
                FSMP_CACHE_SCHEMA, dtype=np.int16
            ),
            "source_signature": np.asarray(signature),
        }
    data["fsmp_target_pe"] = np.asarray(
        -1 if target_pe is None else target_pe, dtype=np.int64
    )
    args.fsmp_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fsmp_data, **data)
    print(f"Wrote {args.fsmp_data}")
    return data






def parse_args(argv=None):
    """Parse the three-source comparison command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-z", type=float, default=0.0, dest="z")
    parser.add_argument(
        "--run-table", type=Path, default=Path(DEFAULT_RUN_TABLE)
    )
    parser.add_argument("--fsmp-dir", type=Path, default=None)
    parser.add_argument(
        "--rtraw-dir", type=Path, default=Path(DEFAULT_RTRAW_DIR)
    )
    parser.add_argument("--pmt-map", type=Path, default=Path(DEFAULT_PMT_MAP))
    parser.add_argument("--pmt", type=Path, default=Path(DEFAULT_PMT_FILE))
    # 修改为支持多个 LUCiD 目录
    parser.add_argument(
        "--sim-dir",
        "--lucid-sim-dir",
        dest="lucid_sim_dir",
        type=Path,
        nargs="+",
        help="LUCiD simulation directories; multiple sources are allowed",
    )
    parser.add_argument("--sim-chunk-size", type=int, default=10_000_000)
    # 修改为支持多个 LUCiD 标签
    parser.add_argument(
        "--lucid-label", nargs='+', default=None,
        help="Labels for LUCiD sources (must match number of --lucid-sim-dir)"
    )
    parser.add_argument("--junosw-sim-dir", type=Path)
    parser.add_argument("--junosw-sim-group", default="Sim")
    parser.add_argument(
        "--junosw-sim-match-mode",
        choices=("auto", "event", "global"),
        default="auto",
    )
    parser.add_argument("--junosw-sim-file-glob", default="*.h5")
    parser.add_argument("--junosw-label", default=DEFAULT_JUNOSW_LABEL)
    qe_group = parser.add_mutually_exclusive_group()
    qe_group.add_argument(
        "--lucid-qe-correction",
        dest="apply_lucid_qe",
        action="store_true",
        default=True,
        help="scale only LUCiD amplitudes by the common 532 nm QE (default)",
    )
    qe_group.add_argument(
        "--no-lucid-qe-correction",
        dest="apply_lucid_qe",
        action="store_false",
        help="leave LUCiD amplitudes at their raw QE=1 scale",
    )
    parser.add_argument(
        "--lucid-qe-effective",
        type=float,
        default=JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
        metavar="QE",
        help="common LUCiD-only QE factor; J26.3.2 gives 0.097 at 532 nm",
    )
    parser.add_argument(
        "--source-intensity-scale",
        type=float,
        nargs="+",
        default=None,
        metavar="SCALE",
        help=(
            "one global intensity scale per simulation source; "
            "defaults to 1"
        ),
    )
    parser.add_argument(
        "--channel-mode",
        choices=("representative", "random", "all", "stratified"),
        default="representative",
        help="channel panel; representative is the fixed default",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1000,
        dest="number",
        help="number of channels when --channel-mode=random",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratum-size",
        type=int,
        default=28,
        help="PMTs per theta/type stratum for --channel-mode=stratified",
    )
    parser.add_argument("--north", action="store_true")
    parser.add_argument("--bins", type=int, default=600)
    parser.add_argument(
        "--plot-components",
        nargs="+",
        choices=("time", "summary", "amplitude"),
        default=("time", "summary", "amplitude"),
        help="PDF components to render",
    )
    parser.add_argument(
        "--data-range",
        type=float,
        nargs=2,
        default=(-100.0, 500.0),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--plot-range",
        type=float,
        nargs=2,
        default=(0.0, 500.0),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--prompt-window",
        type=float,
        nargs=2,
        default=(-10.0, 20.0),
        metavar=("LOW", "HIGH"),
        help="diagnostic window in ns relative to the median simulation peak",
    )
    parser.add_argument(
        "--late-window",
        type=float,
        nargs=2,
        default=(20.0, 600.0),
        metavar=("LOW", "HIGH"),
        help="diagnostic window in ns relative to the median simulation peak",
    )
    parser.add_argument(
        "--alignment",
        choices=("l1", "peak"),
        default="l1",
        help="common FSMP alignment against all simulations with equal weight",
    )
    parser.add_argument("--long", action="store_true")
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="disable the default simulation-side per-PMT TTS convolution",
    )
    parser.add_argument("--irf-sigma", type=float, default=0.0)
    parser.add_argument("--irf-sigma-hamamatsu", type=float, default=None)
    parser.add_argument("--irf-sigma-nnvt", type=float, default=None)
    dark_group = parser.add_mutually_exclusive_group()
    dark_group.add_argument(
        "--add-dark-to-simulation",
        dest="add_dark_to_simulation",
        action="store_true",
        default=True,
        help="keep FSMP dark noise and add its estimate to MC (default)",
    )
    dark_group.add_argument(
        "--no-add-dark-to-simulation",
        dest="add_dark_to_simulation",
        action="store_false",
        help="keep FSMP dark noise without adding it to MC",
    )
    parser.add_argument(
        "--dark-margin",
        type=float,
        default=30.0,
        help="signal-free margin before the FSMP peak in ns",
    )
    parser.add_argument("--print-fsmp-row", action="store_true")
    parser.add_argument(
        "--fsmp-file-limit",
        type=int,
        default=None,
        help="read at most this many FSMP parquet files (raw-data smoke only)",
    )
    parser.add_argument(
        "--lucid-file-limit",
        type=int,
        default=None,
        help="read at most this many LUCiD parquet files (raw-data smoke only)",
    )
    parser.add_argument(
        "--junosw-file-limit",
        type=int,
        default=None,
        help="read at most this many JUNOsw files (raw-data smoke only)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="read plot-only source caches from a separate output root",
    )
    parser.add_argument("--output-root", type=Path, default=Path("compare_sim"))
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args(argv)
    if args.long:
        args.data_range = (args.data_range[0], 1000.0)
        args.plot_range = (args.plot_range[0], 1000.0)
    for name in ("prompt_window", "late_window"):
        window = getattr(args, name)
        if not np.all(np.isfinite(window)) or window[1] <= window[0]:
            parser.error(f"--{name.replace('_', '-')} requires LOW < HIGH")
    if (
        not np.isfinite(args.lucid_qe_effective)
        or not 0.0 < args.lucid_qe_effective <= 1.0
    ):
        parser.error("--lucid-qe-effective must be in (0, 1]")
    if not np.isfinite(args.dark_margin) or args.dark_margin < 0.0:
        parser.error("--dark-margin must be finite and non-negative")
    for name in ("fsmp_file_limit", "lucid_file_limit", "junosw_file_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.bins <= 0:
        parser.error("--bins must be positive")
    if args.stratum_size <= 0:
        parser.error("--stratum-size must be positive")
    if args.source_intensity_scale is not None and (
        not np.all(np.isfinite(args.source_intensity_scale))
        or np.any(np.asarray(args.source_intensity_scale) <= 0.0)
    ):
        parser.error("--source-intensity-scale values must be positive")
    if args.cache_root is not None and not args.plot_only:
        parser.error("--cache-root requires --plot-only")
    args.plot_components = tuple(dict.fromkeys(args.plot_components))
    if not args.plot_components:
        parser.error("--plot-components requires at least one component")
    if args.channel_mode == "all" and not args.north:
        parser.error(
            "--channel-mode all is provided by compare/plot_pe_scale.py; "
            "the response comparison needs a smaller channel selection"
        )
    if not any((args.lucid_sim_dir, args.junosw_sim_dir)):
        parser.error("provide at least one simulation input")
    try:
        args.irf_sigma_hamamatsu, args.irf_sigma_nnvt = resolve_irf_sigmas(
            args.irf_sigma, args.irf_sigma_hamamatsu, args.irf_sigma_nnvt
        )
    except ValueError as error:
        parser.error(str(error))
    args.run = resolve_run(args.z, args.run_table)
    args.fsmp_dir = args.fsmp_dir or Path(DEFAULT_FSMP_ROOT) / str(args.run)
    args.rtraw_dir = args.rtraw_dir / str(args.run)
    args.output_dir = args.output_root / str(args.run)
    cache_root = args.cache_root or args.output_root
    args.cache_dir = cache_root / str(args.run)
    if args.north:
        cache_suffix = "north"
    elif args.channel_mode == "random":
        cache_suffix = str(args.number)
    elif args.channel_mode == "stratified":
        cache_suffix = f"stratified{args.stratum_size}_seed{args.seed}"
    else:
        cache_suffix = args.channel_mode
    if args.long:
        cache_suffix += "_long"
    limits = (
        args.fsmp_file_limit,
        args.lucid_file_limit,
        args.junosw_file_limit,
    )
    if any(value is not None for value in limits):
        cache_suffix += "_" + "_".join(
            f"{name}{value if value is not None else 'all'}"
            for name, value in zip(("fsmp", "lucid", "junosw"), limits)
        )
    tag = args.output_tag.strip().replace(" ", "_")
    result_suffix = f"{cache_suffix}_{tag}" if tag else cache_suffix
    args.opt = args.output_dir / f"compare_sim_{result_suffix}.pdf"
    args.summary_opt = (
        args.output_dir / f"compare_sim_{result_suffix}_summary.pdf"
    )
    args.amplitude_opt = (
        args.output_dir / f"compare_sim_{result_suffix}_amplitude.pdf"
    )
    args.fsmp_data = args.cache_dir / f"fsmp_data_{cache_suffix}_all.npz"

    if args.lucid_sim_dir:
        if args.lucid_label is None:
            args.lucid_label = [
                f"LUCiD {i + 1}" for i in range(len(args.lucid_sim_dir))
            ]
        elif len(args.lucid_label) != len(args.lucid_sim_dir):
            parser.error("Number of --lucid-label must match --lucid-sim-dir")
        args.sim_data = [
            args.cache_dir / f"sim_data_{cache_suffix}_{i}.npz"
            for i in range(len(args.lucid_sim_dir))
        ]
    else:
        args.sim_data = []

    args.junosw_sim_data = (
        args.cache_dir / f"junosw_sim_data_{cache_suffix}.npz"
    )
    args.aligned_data = (
        args.output_dir / f"aligned_data_{result_suffix}.npz"
    )
    args.stats_csv = (
        args.output_dir / f"comparison_stats_{result_suffix}.csv"
    )
    args.channel_manifest = (
        args.output_dir / f"channel_selection_{result_suffix}_all.csv"
    )
    args.fsmp_file_manifest = (
        args.output_dir / f"fsmp_file_manifest_{result_suffix}.csv"
    )
    return args


def _load_matching_cache(
    path,
    channels,
    kind=None,
    target_pe=None,
    require_exposure=False,
    signature=None,
):
    """Load a source cache only when channels, kind and content match."""
    if not path.exists():
        return None
    data = load_data(path)
    if not np.array_equal(data.get("channels"), channels):
        print(f"Cached channels in {path} do not match; recomputing.")
        return None
    if kind is not None:
        cached_kind = str(np.asarray(data.get("kind", "")).item())
        if cached_kind != kind:
            print(f"Cached source kind in {path} does not match; recomputing.")
            return None
        schema = int(np.asarray(data.get("amplitude_schema", 0)).item())
        if schema != 3:
            print(f"Cached amplitude schema in {path} is stale; recomputing.")
            return None
    if require_exposure:
        fields = {"fsmp_physical_events", "fsmp_event_samples"}
        if not fields.issubset(data):
            print(
                f"Cached FSMP exposure fields in {path} are missing; "
                "recomputing."
            )
            return None
        schema = int(np.asarray(data.get("fsmp_cache_schema", 0)).item())
        if schema != FSMP_CACHE_SCHEMA:
            print(f"Cached FSMP schema in {path} is stale; recomputing.")
            return None
    if target_pe is not None:
        cached_target = int(
            np.asarray(data.get("fsmp_target_pe", -1)).item()
        )
        if cached_target != int(target_pe):
            print(
                f"Cached FSMP PE target in {path} does not match; "
                "recomputing."
            )
            return None
    if signature is not None:
        cached_signature = data.get("source_signature")
        if cached_signature is None or str(
            np.asarray(cached_signature).item()
        ) != signature:
            print(
                f"Cached source files for {path} changed or lack a "
                "signature; recomputing."
            )
            return None
    return data


def main(argv=None):
    """Compute/load caches and render the three-source comparison."""
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = list(iter_fsmp_files(args.fsmp_dir))
    if not args.plot_only and not files:
        raise FileNotFoundError(
            f"No FSMP Parquet files found in {args.fsmp_dir}"
        )
    channels, _ = selected_channels(
        args, require_files=False, files=files
    )
    fsmp_files_planned = _effective_fsmp_files(args, files)
    fsmp_signature = (
        _source_signature(fsmp_files_planned)
        if fsmp_files_planned
        else None
    )
    source_signatures = []
    if args.lucid_sim_dir:
        for sim_dir in args.lucid_sim_dir:
            planned = _planned_sim_files(args, "lucid", sim_dir=sim_dir)
            source_signatures.append(
                _source_signature(planned) if planned else None
            )
    if args.junosw_sim_dir is not None:
        planned = _planned_sim_files(args, "junosw")
        source_signatures.append(
            _source_signature(planned) if planned else None
        )
    combined_signatures = json.dumps(
        {"fsmp": fsmp_signature, "sources": source_signatures}
    )
    if args.plot_only:
        fsmp_data = load_data(args.fsmp_data)
        cached_signature = fsmp_data.get("source_signature")
        if fsmp_signature is not None and (
            cached_signature is None
            or str(np.asarray(cached_signature).item()) != fsmp_signature
        ):
            raise ValueError(
                "--plot-only FSMP cache source signature mismatch; "
                "recompute without --plot-only"
            )
        required_fsmp = {"fsmp_physical_events", "fsmp_event_samples"}
        if not required_fsmp.issubset(fsmp_data):
            raise ValueError(
                "--plot-only FSMP cache lacks amplitude exposure fields; "
                "recompute without --plot-only"
            )
        schema = int(np.asarray(fsmp_data.get("fsmp_cache_schema", 0)).item())
        if schema != FSMP_CACHE_SCHEMA:
            raise ValueError(
                "--plot-only FSMP cache predates physical-event amplitude "
                "normalization; recompute without --plot-only"
            )
        if not np.array_equal(fsmp_data.get("channels"), channels):
            raise ValueError(
                "--plot-only requires a matching selected-channel FSMP cache; "
                "recompute without --plot-only"
            )
        if (
            args.channel_mode == "all"
            and not args.north
            and "fsmp_histogram" not in fsmp_data
        ):
            raise ValueError(
                "--channel-mode all requires the new histogram FSMP cache; "
                "recompute once without --plot-only"
            )
        sources = []
        if args.lucid_sim_dir:
            for index, _ in enumerate(args.lucid_sim_dir):
                source = _load_matching_cache(
                    args.sim_data[index],
                    channels,
                    "lucid",
                    signature=source_signatures[index],
                )
                if source is None:
                    raise ValueError(
                        "--plot-only requires matching selected-channel "
                        "LUCiD caches"
                    )
                sources.append(source)
        if args.junosw_sim_dir is not None:
            source = _load_matching_cache(
                args.junosw_sim_data,
                channels,
                "junosw",
                signature=source_signatures[-1],
            )
            if source is None:
                raise ValueError(
                    "--plot-only requires a matching selected-channel "
                    "JUNOsw cache"
                )
            sources.append(source)
    else:
        sources = []
        if args.lucid_sim_dir:
            for index, sim_dir in enumerate(args.lucid_sim_dir):
                source = None
                if not args.force:
                    source = _load_matching_cache(
                        args.sim_data[index],
                        channels,
                        "lucid",
                        signature=source_signatures[index],
                    )
                if source is None:
                    source = _compute_sim_cache(
                        args,
                        channels,
                        "lucid",
                        sim_dir=sim_dir,
                        label=args.lucid_label[index],
                        output=args.sim_data[index],
                    )
                sources.append(source)
        if args.junosw_sim_dir is not None:
            source = None
            if not args.force:
                source = _load_matching_cache(
                    args.junosw_sim_data,
                    channels,
                    "junosw",
                    signature=source_signatures[-1],
                )
            sources.append(
                source or _compute_sim_cache(args, channels, "junosw")
            )
        # FSMP reads every input file once; caches are reused by content
        # signature so LUCiD iterations do not trigger FSMP/JUNOsw rereads.
        args.fsmp_target_pe = None
        fsmp_data = None
        if not args.force:
            fsmp_data = _load_matching_cache(
                args.fsmp_data,
                channels,
                require_exposure=True,
                signature=fsmp_signature,
            )
        if (
            args.channel_mode == "all"
            and not args.north
            and fsmp_data is not None
            and "fsmp_histogram" not in fsmp_data
        ):
            print("Existing all-channel FSMP cache is not histogram-based.")
            fsmp_data = None
        if fsmp_data is None:
            fsmp_data = compute_fsmp_data(args, channels, files)
    if not sources:
        raise RuntimeError("No simulation source was loaded")
    coordinate_fields = {"z", "theta", "phi"}
    if not coordinate_fields.issubset(fsmp_data):
        fsmp_data = attach_pmt_coordinates(fsmp_data, args.pmt_map)
    selection_mode = "north" if args.north else args.channel_mode
    write_channel_manifest(
        args.channel_manifest,
        fsmp_data,
        args.run,
        args.z,
        args.irf_sigma_hamamatsu,
        args.irf_sigma_nnvt,
        selection_mode=selection_mode,
    )
    write_fsmp_file_manifest(args.fsmp_file_manifest, fsmp_data)
    qe_factors = simulation_qe_factors(
        sources,
        apply_lucid_qe=args.apply_lucid_qe,
        lucid_qe_eff=args.lucid_qe_effective,
    )
    source_scale = (
        np.ones(len(sources), dtype=np.float64)
        if args.source_intensity_scale is None
        else np.asarray(args.source_intensity_scale, dtype=np.float64)
    )
    if source_scale.shape != (len(sources),):
        raise ValueError(
            "--source-intensity-scale must provide one value per "
            "simulation source"
        )
    if "fsmp_histogram" in fsmp_data:
        raise ValueError(
            "The all-channel FSMP cache is histogram-only. Use "
            "compare/plot_pe_scale.py for PE-scale diagnostics."
        )
    processing = processing_string(
        not args.no_tts,
        args.irf_sigma_hamamatsu,
        args.irf_sigma_nnvt,
        args.add_dark_to_simulation,
        simulation_qe_factors=qe_factors,
        source_intensity_scale=source_scale,
        target_pe=getattr(
            args,
            "fsmp_target_pe",
            fsmp_data.get("fsmp_target_pe"),
        ),
        prompt_window=args.prompt_window,
        late_window=args.late_window,
        alignment=args.alignment,
    )
    data = None
    if args.aligned_data.exists() and not args.force:
        candidate = load_data(args.aligned_data)
        labels = np.asarray(
            [str(np.asarray(source["label"]).item()) for source in sources]
        )
        kinds = np.asarray(
            [str(np.asarray(source["kind"]).item()) for source in sources]
        )
        if (
            str(candidate.get("processing", "")) == processing
            and np.array_equal(candidate.get("channels"), fsmp_data["channels"])
            and np.array_equal(candidate.get("edges"), sources[0]["edges"])
            and np.array_equal(candidate.get("simulation_labels"), labels)
            and np.array_equal(candidate.get("simulation_kinds"), kinds)
            and str(candidate.get("source_signatures", ""))
            == combined_signatures
        ):
            data = candidate
    if data is None:
        data = build_channel_histograms_multi(
            fsmp_data,
            sources,
            use_tts=not args.no_tts,
            irf_sigma_hamamatsu=args.irf_sigma_hamamatsu,
            irf_sigma_nnvt=args.irf_sigma_nnvt,
            add_dark_to_simulation=args.add_dark_to_simulation,
            dark_margin=args.dark_margin,
            simulation_qe_factor=qe_factors,
            simulation_source_scale=source_scale,
            prompt_window=args.prompt_window,
            late_window=args.late_window,
            alignment=args.alignment,
        )
        data["source_signatures"] = np.asarray(combined_signatures)
        np.savez_compressed(args.aligned_data, **data)
        print(f"Wrote {args.aligned_data}")
    if "time" in args.plot_components:
        plot_data(data, args.opt, args.plot_range, args.run)
    if "summary" in args.plot_components:
        plot_summary_data(
            data, args.summary_opt, args.plot_range, args.run, args.z
        )
    if "amplitude" in args.plot_components:
        plot_amplitude_data(
            data,
            args.amplitude_opt,
            args.plot_range,
            args.run,
        )
    write_comparison_stats_csv(args.stats_csv, data, args.run, args.z)




if __name__ == "__main__":
    main()
