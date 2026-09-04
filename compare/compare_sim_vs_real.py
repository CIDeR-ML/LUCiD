#!/usr/bin/env python
"""Compare FSMP data with LUCiD and JUNOsw laser simulations.

Common JUNOsw matching, histogram, alignment, convolution and plotting
primitives are imported from :mod:`compare.compare_sim`.  This module owns
only the three-source orchestration and the FSMP-specific Arrow reader.

FSMP ``PEt`` is a list column.  Its values are converted to NumPy arrays and
``offset`` is added before subtracting the monitor time.  ``all`` mode keeps
every PE in every stored row; ``first`` mode keeps the first PE per row.
LUCiD simulation input uses the same ``eid``/``ch``/``offset``/``step``/``PEt``
contract as FSMP.  Only consolidated Parquet input is accepted; ``--sim-dir``
may contain a shell-style wildcard for matching one source-position folder.
The channel subset is selected before reading any source.  All three inputs
are therefore restricted to the same reproducible channel subset, and FSMP
loading stops after reaching that subset's first LUCiD PE total, finishing the
current file first.
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
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


PMT_TYPE_NAME = base.PMT_TYPE_NAME
NORTH_CHANNEL_COUNT = base.NORTH_CHANNEL_COUNT
METRIC_HISTOGRAM_BINS = base.METRIC_HISTOGRAM_BINS
BRIEF_FSMP_FILE_LIMIT = base.BRIEF_FSMP_FILE_LIMIT
BRIEF_SIM_FILE_LIMIT = base.BRIEF_SIM_FILE_LIMIT
FIGURE_SIZE = base.FIGURE_SIZE

DEFAULT_RUN_TABLE = "/mnt/stage/haoch/laser/run_532.csv"
DEFAULT_RTRAW_DIR = "/mnt/stage/haoch/laser/split_rtraw"
DEFAULT_FSMP_ROOT = "/junofs/C/PE"
DEFAULT_PMT_MAP = "/junofs/C/junoid_J25.7.3.pq"
DEFAULT_PMT_FILE = "/junofs/C/PMT/STD/9541.h5"
DEFAULT_LUCID_LABEL = "LUCiD"
DEFAULT_JUNOSW_LABEL = "JUNO official"
DEFAULT_IRF_SIGMA_HAMAMATSU = 1.3
DEFAULT_IRF_SIGMA_NNVT = 7.0

FSMP_COLOR = "tab:blue"
SOURCE_COLORS = {"lucid": "tab:orange", "junosw": "tab:green"}
FALLBACK_SOURCE_COLORS = ("tab:red", "tab:purple", "tab:brown", "tab:pink")
DIFFERENCE_AXIS_COLOR = "0.25"
RATIO_AXIS_COLOR = "#7f3c8d"

# Shared public helpers.  Keeping these aliases makes reuse explicit and
# preserves the useful notebook-level API without duplicating implementations.
resolve_run = base.resolve_run
iter_fsmp_files = base.iter_fsmp_files
rtraw_path_for = base.rtraw_path_for
read_monitor_times = base.read_monitor_times
select_channels = base.select_channels
read_pmt_info = base.read_pmt_info
attach_pmt_coordinates = base.attach_pmt_coordinates
estimate_peak_from_density = base.estimate_peak_from_density
estimate_peak_from_times = base.estimate_peak_from_times
density_hist = base.density_hist
l1_distance = base.l1_distance
wasserstein_distance = base.wasserstein_distance
local_l1_alignment = base.local_l1_alignment
convolve_density = base.convolve_density
configure_plot_style = base.configure_plot_style
plot_metric_distribution = base.plot_metric_distribution
load_data = base.load_data


def resolve_irf_sigmas(global_sigma=0.0, hamamatsu_sigma=None, nnvt_sigma=None):
    """Resolve the Gaussian IRF placeholder width for both PMT types."""
    values = {
        "global": global_sigma,
        "hamamatsu": (
            global_sigma if hamamatsu_sigma is None else hamamatsu_sigma
        ),
        "nnvt": global_sigma if nnvt_sigma is None else nnvt_sigma,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"IRF sigma for {name} must be finite and >= 0 ns")
    return float(values["hamamatsu"]), float(values["nnvt"])


def irf_sigma_for_pmt_type(pmt_type, hamamatsu_sigma, nnvt_sigma):
    """Return the configured IRF sigma for one PMT type."""
    return {1: float(hamamatsu_sigma), 2: float(nnvt_sigma)}.get(
        int(pmt_type), 0.0
    )


def irf_config_string(hamamatsu_sigma, nnvt_sigma):
    """Serialize type-dependent IRF settings for a cache key."""
    if np.isclose(hamamatsu_sigma, nnvt_sigma, rtol=0.0, atol=1e-12):
        return f"{hamamatsu_sigma:g}"
    return f"hamamatsu:{hamamatsu_sigma:g}|nnvt:{nnvt_sigma:g}"


def processing_string(
    use_tts,
    hamamatsu_sigma,
    nnvt_sigma,
    dark_subtract,
    mode,
    target_pe=None,
):
    """Serialize all response-processing settings into a cache key."""
    value = (
        f"tts={int(use_tts)},"
        f"irf={irf_config_string(hamamatsu_sigma, nnvt_sigma)},"
        f"dark={int(dark_subtract)},fsmp={mode}"
    )
    if target_pe is not None:
        value += f",fsmp_target={int(target_pe)}"
    return value


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


def _first_values(values, lengths):
    """Find the earliest finite PE in each variable-length row."""
    result = np.full(len(lengths), np.nan, dtype=np.float64)
    starts = np.cumsum(np.r_[0, lengths[:-1]])
    for index, (start, length) in enumerate(zip(starts, lengths)):
        if length:
            result[index] = np.nanmin(values[start : start + length])
    return result


def _append_fsmp_batch(
    chunks, batch, channels, monitor_eid, monitor_time, mode
):
    """Convert one Arrow record batch to flat, monitor-corrected arrays."""
    eid = batch.column("eid").to_numpy(zero_copy_only=False).astype(np.int64)
    channel = batch.column("ch").to_numpy(zero_copy_only=False).astype(np.int64)
    offset = (
        batch.column("offset").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    values, lengths, valid_pet = _arrow_pet_values(batch.column("PEt"))
    monitor, valid_monitor = _monitor_match(eid, monitor_eid, monitor_time)
    valid_row = valid_pet & valid_monitor & (lengths > 0)
    if not np.any(valid_row):
        return 0

    if mode == "first":
        first = _first_values(values, lengths)
        keep = valid_row & np.isfinite(first)
        if np.any(keep):
            chunks.append(
                (
                    channel[keep],
                    eid[keep],
                    first[keep] + offset[keep] - monitor[keep],
                    lengths[keep],
                )
            )
        return int(np.count_nonzero(keep))

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
    time_mode="all",
    print_first_row=False,
    batch_size=250_000,
    target_pe=None,
):
    """Read FSMP Parquet until the detector-wide PE target is reached.

    The current file is always read to completion. Once its accepted PE count
    makes the cumulative total reach ``target_pe``, later files are skipped.
    """
    if time_mode not in {"first", "all"}:
        raise ValueError(f"Unknown FSMP time mode: {time_mode}")
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
            monitor_eid, monitor_time = _monitor_arrays(
                rtraw_path_for(path, fsmp_dir, rtraw_dir)
            )
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=["eid", "ch", "offset", "step", "PEt"],
                use_threads=False,
            ):
                row_channel = batch.column("ch").to_numpy(zero_copy_only=False)
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
                    time_mode,
                )
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
        }
    channel, eid, time, pe_count = map(np.concatenate, zip(*chunks))
    return {
        "ch": channel.astype(np.int64, copy=False),
        "eid": eid.astype(np.int64, copy=False),
        "time": time.astype(np.float64, copy=False),
        "pe_count": pe_count.astype(np.int64, copy=False),
    }


def selected_channels(args, require_files=True, files=None):
    """Select final plot channels and optionally locate FSMP files."""
    files = list(iter_fsmp_files(args.fsmp_dir)) if files is None else files
    if require_files and not files:
        raise FileNotFoundError(
            f"No FSMP Parquet files found in {args.fsmp_dir}"
        )
    channels = (
        np.arange(NORTH_CHANNEL_COUNT, dtype=np.int64)
        if args.north
        else select_channels(args.number, args.seed)
    )
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


def _read_lucid_parquet(path, edges, chunk_size, channels, event_pairs):
    """Accumulate one FSMP-style LUCiD Parquet file into histograms."""
    parquet = pq.ParquetFile(path)
    histogram = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_pe = np.zeros(len(channels), dtype=np.int64)
    for batch in parquet.iter_batches(
        batch_size=chunk_size,
        columns=["eid", "ch", "offset", "PEt"],
        use_threads=False,
    ):
        row_channel = batch.column("ch").to_numpy(zero_copy_only=False)
        row_eid = batch.column("eid").to_numpy(zero_copy_only=False)
        row_offset = batch.column("offset").to_numpy(zero_copy_only=False)
        values, lengths, valid_pet = _arrow_pet_values(batch.column("PEt"))
        row_index = np.searchsorted(channels, row_channel)
        valid_row = valid_pet & (lengths > 0) & (row_index < len(channels))
        valid_row[valid_row] = (
            channels[row_index[valid_row]] == row_channel[valid_row]
        )
        if np.any(valid_row):
            event_pairs.append(
                np.column_stack(
                    (row_index[valid_row], row_eid[valid_row])
                ).astype(np.uint64, copy=False)
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
    return histogram, entries, total_pe


def read_lucid_sim_histograms(sim_files, edges, chunk_size, channels):
    """Read LUCiD Parquet files into density and count arrays."""
    if chunk_size <= 0:
        raise ValueError("--sim-chunk-size must be positive")
    channels = np.asarray(channels, dtype=np.int64)
    histogram = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_pe = np.zeros(len(channels), dtype=np.int64)
    events = np.zeros(len(channels), dtype=np.int64)
    event_pairs = []
    for path in tqdm(sim_files, desc="LUCiD simulation files"):
        try:
            parquet_hist = _read_lucid_parquet(
                path, edges, chunk_size, channels, event_pairs
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
    if event_pairs:
        unique_pairs = np.unique(np.concatenate(event_pairs), axis=0)
        np.add.at(events, unique_pairs[:, 0].astype(np.int64), 1)
    return (
        _density_from_histogram(histogram, entries, edges),
        entries,
        total_pe,
        events,
    )


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
    return (
        density,
        entries,
        entries.copy(),
        np.full(len(channels), -1, dtype=np.int64),
    )


def _compute_sim_cache(args, channels, kind, sim_dir=None, label=None, output=None):
    """Build one simulation cache in the common source schema."""
    edges = np.linspace(args.data_range[0], args.data_range[1], args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if kind == "lucid":
        files = iter_sim_files(sim_dir if sim_dir is not None else args.lucid_sim_dir[0])
        if args.brief:
            files = files[:BRIEF_SIM_FILE_LIMIT]
        if not files:
            raise FileNotFoundError(
                f"No LUCiD Parquet files found in {sim_dir}"
            )
        result = read_lucid_sim_histograms(
            files, edges, args.sim_chunk_size, channels
        )
        label = label if label is not None else args.lucid_label[0]
        output = output if output is not None else args.sim_data[0]
    else:
        files = iter_junosw_sim_files(
            args.junosw_sim_dir, args.junosw_sim_file_glob
        )
        if args.brief:
            files = files[:BRIEF_SIM_FILE_LIMIT]
        if not files:
            raise FileNotFoundError(
                f"No JUNOsw HDF5 files found in {args.junosw_sim_dir}"
            )
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
        "label": np.asarray(label),
        "kind": np.asarray(kind),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f"Wrote {output}")
    return data


def compute_fsmp_data(args, channels, fsmp_files):
    """Build the FSMP cache from Arrow/NumPy arrays."""
    target_pe = getattr(args, "fsmp_target_pe", None)
    if target_pe is not None:
        print(
            "FSMP PE target from selected-channel LUCiD simulation: "
            f"{int(target_pe)}"
        )
    pmt_type, tts = read_pmt_info(channels, args.pmt_map, args.pmt)
    records = read_fsmp_times_by_channel(
        fsmp_files,
        args.fsmp_dir,
        args.rtraw_dir,
        channels,
        args.run,
        BRIEF_FSMP_FILE_LIMIT if args.brief else None,
        args.fsmp_time_mode,
        args.print_fsmp_row,
        target_pe=target_pe,
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
        "fsmp_time_mode": np.asarray(args.fsmp_time_mode),
        "fsmp_target_pe": np.asarray(
            -1 if target_pe is None else target_pe, dtype=np.int64
        ),
    }
    args.fsmp_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fsmp_data, **data)
    print(f"Wrote {args.fsmp_data}")
    return data


def _records_by_channel(data):
    """Group FSMP cache arrays while retaining event and PE metadata."""
    channel = np.asarray(data["fsmp_ch"], dtype=np.int64)
    order = np.argsort(channel, kind="mergesort")
    sorted_channel = channel[order]
    unique, starts = np.unique(sorted_channel, return_index=True)
    stops = np.r_[starts[1:], len(order)]
    eid = np.asarray(
        data.get("fsmp_eid", np.arange(len(channel))), dtype=np.int64
    )
    count = np.asarray(
        data.get("fsmp_pe_count", np.ones(len(channel))), dtype=np.int64
    )
    return {
        int(value): {
            "time": np.asarray(data["fsmp_time"])[order[start:stop]],
            "eid": eid[order[start:stop]],
            "pe_count": count[order[start:stop]],
        }
        for value, start, stop in zip(unique, starts, stops)
    }


def estimate_dark_window(times, peak, margin_ns):
    """Return the finite pre-peak interval used for dark-rate estimation."""
    values = np.asarray(times, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(peak):
        return None
    low, high = float(np.min(values)), float(peak - margin_ns)
    return None if high <= low else (low, high)


def _empty_dark_stats():
    return {
        "prepeak_pe": 0,
        "rate_per_ns": 0.0,
        "expected_pe": 0.0,
        "removed_pe": 0.0,
        "window_ns": 0.0,
        "window_low_ns": np.nan,
        "window_high_ns": np.nan,
        "density_floor": 0.0,
        "normalization_pe": 0,
    }


def subtract_dark_noise(
    times, density, edges, peak, margin_ns, normalization_count=None
):
    """Subtract a non-negative pre-peak density floor and return audit data."""
    stats = _empty_dark_stats()
    window = estimate_dark_window(times, peak, margin_ns)
    if window is None or density is None:
        return density, stats
    low, high = window
    width = high - low
    prepeak = int(np.count_nonzero((times >= low) & (times < high)))
    rate = prepeak / width
    normalization_count = (
        len(times) if normalization_count is None else int(normalization_count)
    )
    floor = rate / max(normalization_count, 1)
    corrected = np.clip(density - floor, 0.0, None)
    removed = np.clip(density - corrected, 0.0, None)
    stats.update(
        {
            "prepeak_pe": prepeak,
            "rate_per_ns": float(rate),
            "expected_pe": float(rate * (edges[-1] - edges[0])),
            "removed_pe": float(
                np.sum(removed) * (edges[1] - edges[0]) * normalization_count
            ),
            "window_ns": float(width),
            "window_low_ns": low,
            "window_high_ns": high,
            "density_floor": float(floor),
            "normalization_pe": normalization_count,
        }
    )
    return corrected, stats


def _source_arrays(data):
    """Normalize source cache fields to ``[source, channel, ...]``."""
    density = np.asarray(data["simulation_density"], dtype=np.float64)
    raw = np.asarray(
        data.get("simulation_raw_density", density), dtype=np.float64
    )
    entries = np.asarray(data["simulation_entries"], dtype=np.int64)
    peaks = np.asarray(
        data.get("simulation_peak", np.empty(0)), dtype=np.float64
    )
    if density.ndim == 2:
        density, raw, entries = density[None], raw[None], entries[None]
        if peaks.size:
            peaks = peaks[None]
    if not peaks.size:
        peaks = np.full(entries.shape, np.nan)
    labels = data.get("simulation_labels")
    if labels is None:
        labels = np.asarray(
            [str(np.asarray(data.get("label", "Simulation")).item())]
        )
    labels = np.asarray(labels).astype(str)
    if len(labels) != density.shape[0]:
        labels = np.asarray(
            [f"Simulation {i + 1}" for i in range(density.shape[0])]
        )
    return labels, raw, density, entries, peaks


def _metric_arrays(data):
    """Return labels, densities and source-by-channel distance arrays."""
    labels, _, density, _, _ = _source_arrays(data)
    run = np.asarray(data["run_density"], dtype=np.float64)
    if run.ndim == 2:
        run = run[None]
    width = float(data["edges"][1] - data["edges"][0])
    l1 = np.asarray(data.get("l1_distance", []), dtype=np.float64)
    wasserstein = np.asarray(
        data.get("wasserstein_distance", []), dtype=np.float64
    )
    if l1.size == 0:
        l1 = np.asarray(
            [
                [l1_distance(a, b, width) for a, b in zip(run[i], density[i])]
                for i in range(len(labels))
            ]
        )
    if wasserstein.size == 0:
        wasserstein = np.asarray(
            [
                [
                    wasserstein_distance(a, b, width)
                    for a, b in zip(run[i], density[i])
                ]
                for i in range(len(labels))
            ]
        )
    if l1.ndim == 1:
        l1 = l1[None]
    if wasserstein.ndim == 1:
        wasserstein = wasserstein[None]
    return labels, run, density, l1, wasserstein


def _source_counts(data, labels):
    """Normalize count fields to source-by-channel arrays."""
    _, _, density, entries, _ = _source_arrays(data)
    shape = density.shape[:2]

    def get(name, fallback, fill=0):
        value = np.asarray(data.get(name, data.get(fallback)), dtype=np.int64)
        if value.ndim == 1:
            value = value[None]
        if value.size == 0:
            return np.full(shape, fill, dtype=np.int64)
        if value.shape != shape:
            value = np.broadcast_to(value, shape).copy()
        return value

    return (
        get("run_entries", "simulation_entries"),
        get("run_events", "run_entries"),
        get("run_total_pe", "run_entries"),
        entries,
        get("simulation_events", "simulation_entries", -1),
        get("simulation_total_pe", "simulation_entries"),
    )


def _simulation_colors(data, labels):
    kinds = np.asarray(
        data.get("simulation_kinds", ["simulation"] * len(labels))
    ).astype(str)
    colors = []
    lucid_count = 0  # 记录 LUCiD 源的数量
    for index, (kind, label) in enumerate(zip(kinds, labels)):
        key, text = kind.lower(), str(label).lower()
        if key == "lucid" or "lucid" in text:
            if lucid_count == 0:
                colors.append(SOURCE_COLORS["lucid"])
            else:
                # 第2个及之后的LUCiD使用备用颜色
                colors.append(
                    FALLBACK_SOURCE_COLORS[(lucid_count - 1) % len(FALLBACK_SOURCE_COLORS)]
                )
            lucid_count += 1
        elif key == "junosw" or "juno" in text:
            colors.append(SOURCE_COLORS["junosw"])
        else:
            colors.append(
                FALLBACK_SOURCE_COLORS[index % len(FALLBACK_SOURCE_COLORS)]
            )
    return colors


def _format_count(value):
    value = int(value)
    return "n/a" if value < 0 else f"{value:,}"


def _processing_title(data):
    raw = str(np.asarray(data.get("processing", "")).item())
    flags = dict(item.split("=", 1) for item in raw.split(",") if "=" in item)
    irf = flags.get("irf", "unknown")
    if "|" in irf:
        values = dict(item.split(":", 1) for item in irf.split("|"))
        irf_title = (
            f"IRF sigma: Hamamatsu={values.get('hamamatsu', '?')} ns, "
            f"NNVT={values.get('nnvt', '?')} ns"
        )
    else:
        irf_title = f"IRF sigma={irf} ns"
    tts = flags.get("tts", "unknown")
    dark = flags.get("dark", "unknown")
    tts_text = "on" if tts == "1" else "off" if tts == "0" else tts
    dark_text = "on" if dark == "1" else "off" if dark == "0" else dark
    return (
        f"simulation TTS={tts_text}; {irf_title}; "
        f"FSMP dark subtraction={dark_text}"
    )


def write_channel_manifest(
    path,
    data,
    run,
    source_z,
    irf_sigma_hamamatsu=0.0,
    irf_sigma_nnvt=0.0,
):
    """Write selected channel metadata, including configured IRF widths."""
    channels = np.asarray(data["channels"], dtype=np.int64)
    irf = np.asarray(
        [
            irf_sigma_for_pmt_type(
                pmt_type, irf_sigma_hamamatsu, irf_sigma_nnvt
            )
            for pmt_type in data["pmt_type"]
        ],
        dtype=np.float64,
    )
    fields = [
        "source_z_m",
        "run",
        "channel",
        "pmt_type",
        "pmt_type_name",
        "z_mm",
        "theta_deg",
        "tts_ns",
        "irf_sigma_ns",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, channel in enumerate(channels):
            writer.writerow(
                {
                    "source_z_m": float(source_z),
                    "run": int(run),
                    "channel": int(channel),
                    "pmt_type": int(data["pmt_type"][index]),
                    "pmt_type_name": PMT_TYPE_NAME.get(
                        int(data["pmt_type"][index]), "Unknown"
                    ),
                    "z_mm": float(data["z"][index]),
                    "theta_deg": float(data["theta"][index]),
                    "tts_ns": float(data["tts"][index]),
                    "irf_sigma_ns": float(irf[index]),
                }
            )
    print(f"Wrote {path}")


def _source_slug(label, index):
    slug = "".join(
        char.lower() if char.isalnum() else "_" for char in str(label)
    ).strip("_")
    return slug or f"simulation_{index + 1}"


def write_comparison_stats_csv(path, data, run, source_z):
    """Write per-channel metrics, counts and dark-noise audit fields."""
    labels, _, _, l1, wasserstein = _metric_arrays(data)
    entries, events, total_pe, sim_entries, sim_events, sim_total = (
        _source_counts(data, labels)
    )
    processing = _processing_title(data)
    fields = [
        "run",
        "source_z_m",
        "channel",
        "pmt_type",
        "z_mm",
        "theta_deg",
        "tts_ns",
        "irf_sigma_ns",
        "fsmp_time_mode",
        "processing",
        "fsmp_events",
        "fsmp_pe",
        "fsmp_time_samples",
    ]
    for index, label in enumerate(labels):
        prefix = _source_slug(label, index)
        fields.extend(
            [
                f"{prefix}_events",
                f"{prefix}_pe",
                f"{prefix}_time_samples",
                f"{prefix}_l1",
                f"{prefix}_wasserstein_ns",
                f"{prefix}_peak_ns",
            ]
        )
    fields.extend(
        f"fsmp_dark_{key}"
        for key in (
            "prepeak_pe",
            "rate_per_ns",
            "window_ns",
            "window_low_ns",
            "window_high_ns",
            "density_floor",
            "normalization_pe",
            "expected_pe",
            "removed_pe",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for channel_index, channel in enumerate(data["channels"]):
            row = {
                "run": int(run),
                "source_z_m": float(source_z),
                "channel": int(channel),
                "pmt_type": int(data["pmt_type"][channel_index]),
                "z_mm": float(data["z"][channel_index]),
                "theta_deg": float(data["theta"][channel_index]),
                "tts_ns": float(data["tts"][channel_index]),
                "irf_sigma_ns": float(
                    data["irf_sigma_by_channel"][channel_index]
                ),
                "fsmp_time_mode": str(
                    np.asarray(data.get("fsmp_time_mode", "unknown")).item()
                ),
                "processing": processing,
                "fsmp_events": int(events[0, channel_index]),
                "fsmp_pe": int(total_pe[0, channel_index]),
                "fsmp_time_samples": int(entries[0, channel_index]),
            }
            for index, label in enumerate(labels):
                prefix = _source_slug(label, index)
                row.update(
                    {
                        f"{prefix}_events": int(
                            sim_events[index, channel_index]
                        ),
                        f"{prefix}_pe": int(sim_total[index, channel_index]),
                        f"{prefix}_time_samples": int(
                            sim_entries[index, channel_index]
                        ),
                        f"{prefix}_l1": float(l1[index, channel_index]),
                        f"{prefix}_wasserstein_ns": float(
                            wasserstein[index, channel_index]
                        ),
                        f"{prefix}_peak_ns": float(
                            data["simulation_peak"][index, channel_index]
                        ),
                    }
                )
            for key in (
                "prepeak_pe",
                "rate_per_ns",
                "window_ns",
                "window_low_ns",
                "window_high_ns",
                "density_floor",
                "normalization_pe",
                "expected_pe",
                "removed_pe",
            ):
                row[f"fsmp_dark_{key}"] = np.asarray(data[f"run_dark_{key}"])[
                    0, channel_index
                ]
            writer.writerow(row)
    print(f"Wrote {path}")


def plot_data(data, output, plot_range, run):
    """Write per-channel three-way density, residual and metric pages."""
    configure_plot_style()
    labels, run_density, sim_density, l1, wasserstein = _metric_arrays(data)
    colors = _simulation_colors(data, labels)
    with PdfPages(output) as pdf:
        for index, channel in enumerate(data["channels"]):
            pmt = PMT_TYPE_NAME.get(int(data["pmt_type"][index]), "Unknown")
            title = (
                f"Channel {int(channel)} | {pmt}\n"
                f"TTS={data['tts'][index]:.2f} ns; "
                f"z={data['z'][index]:.2f} mm; "
                f"theta={data['theta'][index]:.2f} deg"
            )
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                data["centers"],
                run_density[0, index],
                where="mid",
                color=FSMP_COLOR,
                linestyle="--",
                label=f"FSMP RUN {run}",
            )
            for source, (label, color) in enumerate(zip(labels, colors)):
                ax.step(
                    data["centers"],
                    sim_density[source, index],
                    where="mid",
                    color=color,
                    label=str(label),
                )
            ax.text(
                0.98,
                0.98,
                "\n".join(
                    f"{label}: L1={l1[i,index]:.4g}, "
                    f"W={wasserstein[i,index]:.4g} ns"
                    for i, label in enumerate(labels)
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            ax.set(
                title=title,
                xlim=plot_range,
                xlabel="Time [ns]",
                ylabel="Normalized density",
                yscale="log",
            )
            ax.set_ylim(bottom=1e-6)
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, difference = plt.subplots(figsize=FIGURE_SIZE)
            lines = []
            for source, (label, color) in enumerate(zip(labels, colors)):
                lines.append(
                    difference.step(
                        data["centers"],
                        sim_density[source, index] - run_density[0, index],
                        where="mid",
                        color=color,
                        label=f"difference: {label} - FSMP",
                    )[0]
                )
            difference.axhline(0.0, color=DIFFERENCE_AXIS_COLOR, linestyle="--")
            ratio_ax = difference.twinx()
            ratios = []
            for source, (label, color) in enumerate(zip(labels, colors)):
                ratio = np.divide(
                    sim_density[source, index],
                    run_density[0, index],
                    out=np.full_like(run_density[0, index], np.nan),
                    where=run_density[0, index] > 0,
                )
                ratios.append(
                    ratio_ax.step(
                        data["centers"],
                        ratio,
                        where="mid",
                        color=color,
                        linestyle=":",
                        label=f"ratio: {label} / FSMP",
                    )[0]
                )
            ratio_ax.axhline(1.0, color=RATIO_AXIS_COLOR, linestyle="--")
            ratio_ax.set_yscale("log")
            difference.set(
                title=title,
                xlim=plot_range,
                xlabel="Time [ns]",
                ylabel="Simulation - FSMP density",
            )
            ratio_ax.set_ylabel("Simulation / FSMP ratio")
            difference.legend(
                lines + ratios,
                [line.get_label() for line in lines + ratios],
                frameon=False,
                fontsize="x-small",
                loc="lower right",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        for name, mask in (
            ("Hamamatsu PMT", data["pmt_type"] == 1),
            ("NNVT PMT", data["pmt_type"] == 2),
            ("All PMTs", np.ones(len(data["channels"]), dtype=bool)),
        ):
            for source, label in enumerate(labels):
                plot_metric_distribution(
                    pdf, l1[source, mask], f"{name} {label} L1", "L1 distance"
                )
                plot_metric_distribution(
                    pdf,
                    wasserstein[source, mask],
                    f"{name} {label} Wasserstein",
                    "Wasserstein distance [ns]",
                )
    print(f"Wrote {output}")


def _normalize(density, width):
    area = np.sum(density) * width
    return density / area if area > 0 else np.zeros_like(density)


def plot_summary_data(data, output, plot_range, run, source_z):
    """Write PMT-type/theta summary pages for all simulation sources."""
    configure_plot_style()
    labels, run_density, sim_density, _, _ = _metric_arrays(data)
    _, _, _, sim_entries, _, _ = _source_counts(data, labels)
    colors = _simulation_colors(data, labels)
    width = float(data["edges"][1] - data["edges"][0])
    if np.isclose(source_z, 0.0):
        theta_groups = [("", np.ones(len(data["theta"]), dtype=bool))]
    else:
        theta_groups = [
            (f"theta={value:.2f} deg", np.isclose(data["theta"], value))
            for value in np.unique(data["theta"][np.isfinite(data["theta"])])
        ]
    with PdfPages(output) as pdf:
        for theta_title, theta_mask in theta_groups:
            for pmt_title, pmt_id in (
                ("Hamamatsu PMT average", 1),
                ("NNVT PMT average", 2),
            ):
                mask = theta_mask & (data["pmt_type"] == pmt_id)
                if not np.any(mask):
                    continue
                mean_run = _normalize(
                    np.mean(run_density[0, mask], axis=0), width
                )
                means = [
                    (
                        _normalize(np.mean(sim_density[i, mask], axis=0), width)
                        if np.any(mask & (sim_entries[i] > 0))
                        else np.zeros(len(data["centers"]))
                    )
                    for i in range(len(labels))
                ]
                title = (
                    f"{theta_title + ', ' if theta_title else ''}{pmt_title}"
                )
                fig, ax = plt.subplots(figsize=FIGURE_SIZE)
                ax.step(
                    data["centers"],
                    mean_run,
                    where="mid",
                    color=FSMP_COLOR,
                    linestyle="--",
                    label=f"FSMP RUN {run}",
                )
                for label, mean, color in zip(labels, means, colors):
                    ax.step(
                        data["centers"],
                        mean,
                        where="mid",
                        color=color,
                        label=str(label),
                    )
                ax.text(
                    0.98,
                    0.98,
                    "\n".join(
                        f"{label}: "
                        f"L1={l1_distance(mean_run, mean, width):.4g}, "
                        f"W={wasserstein_distance(mean_run, mean, width):.4g} "
                        "ns"
                        for label, mean in zip(labels, means)
                    ),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )
                ax.set(
                    title=f"{title}\nFSMP RUN {run} vs {', '.join(labels)}",
                    xlim=plot_range,
                    xlabel="Time [ns]",
                    ylabel="Normalized density",
                    yscale="log",
                )
                ax.set_ylim(bottom=1e-6)
                ax.legend(frameon=False, loc="upper left")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                fig, difference = plt.subplots(figsize=FIGURE_SIZE)
                lines = []
                ratios = []
                for label, mean, color in zip(labels, means, colors):
                    lines.append(
                        difference.step(
                            data["centers"],
                            mean - mean_run,
                            where="mid",
                            color=color,
                            label=f"difference: {label} - FSMP",
                        )[0]
                    )
                difference.axhline(
                    0.0, color=DIFFERENCE_AXIS_COLOR, linestyle="--"
                )
                ratio_ax = difference.twinx()
                for label, mean, color in zip(labels, means, colors):
                    ratio = np.divide(
                        mean,
                        mean_run,
                        out=np.full_like(mean_run, np.nan),
                        where=mean_run > 0,
                    )
                    ratios.append(
                        ratio_ax.step(
                            data["centers"],
                            ratio,
                            where="mid",
                            color=color,
                            linestyle=":",
                            label=f"ratio: {label} / FSMP",
                        )[0]
                    )
                ratio_ax.axhline(1.0, color=RATIO_AXIS_COLOR, linestyle="--")
                ratio_ax.set_yscale("log")
                difference.set(
                    title=title,
                    xlim=plot_range,
                    xlabel="Time [ns]",
                    ylabel="Simulation - FSMP density",
                )
                ratio_ax.set_ylabel("Simulation / FSMP ratio")
                difference.legend(
                    lines + ratios,
                    [line.get_label() for line in lines + ratios],
                    frameon=False,
                    fontsize="x-small",
                    loc="lower right",
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Wrote {output}")


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
        "--sim-dir", "--lucid-sim-dir", dest="lucid_sim_dir", type=Path, 
        nargs='+', help="LUCiD simulation directories (provide multiple for multiple sources)"
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
    parser.add_argument("-n", type=int, default=1000, dest="number")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--north", action="store_true")
    parser.add_argument("--bins", type=int, default=600)
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
        "--dark-subtract",
        dest="dark_subtract",
        action="store_true",
        default=True,
        help="enable FSMP dark-noise subtraction (default)",
    )
    dark_group.add_argument(
        "--no-dark-subtract",
        dest="dark_subtract",
        action="store_false",
        help="disable FSMP dark-noise subtraction",
    )
    parser.add_argument("--dark-window", type=float, default=30.0)
    parser.add_argument(
        "--fsmp-time-mode", choices=("first", "all"), default="all"
    )
    parser.add_argument("--print-fsmp-row", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("compare_sim"))
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args(argv)
    if args.long:
        args.data_range = (args.data_range[0], 1000.0)
        args.plot_range = (args.plot_range[0], 1000.0)
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
    suffix = "north" if args.north else str(args.number)
    if args.long:
        suffix += "_long"
    tag = args.output_tag.strip().replace(" ", "_")
    suffix = f"{suffix}_{tag}" if tag else suffix
    args.opt = args.output_dir / f"compare_sim_{suffix}.pdf"
    args.summary_opt = args.output_dir / f"compare_sim_{suffix}_summary.pdf"
    args.fsmp_data = (
        args.output_dir / f"fsmp_data_{suffix}_{args.fsmp_time_mode}.npz"
    )

    if args.lucid_sim_dir:
        if args.lucid_label is None:
            args.lucid_label = [
                f"LUCiD {i + 1}" for i in range(len(args.lucid_sim_dir))
            ]
        elif len(args.lucid_label) != len(args.lucid_sim_dir):
            parser.error("Number of --lucid-label must match --lucid-sim-dir")
        args.sim_data = [
            args.output_dir / f"sim_data_{suffix}_{i}.npz"
            for i in range(len(args.lucid_sim_dir))
        ]
    else:
        args.sim_data = []

    args.junosw_sim_data = (
        args.output_dir / f"junosw_sim_data_{suffix}.npz"
    )
    args.aligned_data = args.output_dir / f"aligned_data_{suffix}.npz"
    args.stats_csv = args.output_dir / f"comparison_stats_{suffix}.csv"
    args.channel_manifest = (
        args.output_dir
        / f"channel_selection_{suffix}_{args.fsmp_time_mode}.csv"
    )
    return args


def _load_matching_cache(path, channels, kind=None, target_pe=None):
    """Load a source cache only when its channel selection and kind match."""
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
    if args.plot_only:
        fsmp_data = load_data(args.fsmp_data)
        if not np.array_equal(fsmp_data.get("channels"), channels):
            raise ValueError(
                "--plot-only requires a matching selected-channel FSMP cache; "
                "recompute without --plot-only"
            )
        sources = []
        if args.lucid_sim_dir:
            for index, _ in enumerate(args.lucid_sim_dir):
                source = _load_matching_cache(
                    args.sim_data[index], channels, "lucid"
                )
                if source is None:
                    raise ValueError(
                        "--plot-only requires matching selected-channel "
                        "LUCiD caches"
                    )
                sources.append(source)
        if args.junosw_sim_dir is not None:
            source = _load_matching_cache(
                args.junosw_sim_data, channels, "junosw"
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
                        args.sim_data[index], channels, "lucid"
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
                    args.junosw_sim_data, channels, "junosw"
                )
            sources.append(
                source or _compute_sim_cache(args, channels, "junosw")
            )
        # Limit FSMP input using the selected-channel LUCiD PE count.
        lucid_sources = [
            source for source in sources
            if str(np.asarray(source.get("kind", "")).item()) == "lucid"
        ]
        lucid_source = lucid_sources[0] if lucid_sources else None
        lucid_total_pe = (
            int(
                np.asarray(
                    lucid_source.get(
                        "simulation_total_pe",
                        lucid_source["simulation_entries"],
                    )
                ).sum()
            )
            if lucid_source is not None
            else None
        )
        args.fsmp_target_pe = lucid_total_pe
        if lucid_total_pe is not None:
            print(
                "FSMP target from selected-channel LUCiD simulation: "
                f"{args.fsmp_target_pe} PE"
            )
        fsmp_data = None
        if not args.force:
            fsmp_data = _load_matching_cache(
                args.fsmp_data,
                channels,
                target_pe=args.fsmp_target_pe,
            )
        if fsmp_data is None:
            fsmp_data = compute_fsmp_data(args, channels, files)
    if not sources:
        raise RuntimeError("No simulation source was loaded")
    fsmp_data = attach_pmt_coordinates(fsmp_data, args.pmt_map)
    write_channel_manifest(
        args.channel_manifest,
        fsmp_data,
        args.run,
        args.z,
        args.irf_sigma_hamamatsu,
        args.irf_sigma_nnvt,
    )
    processing = processing_string(
        not args.no_tts,
        args.irf_sigma_hamamatsu,
        args.irf_sigma_nnvt,
        args.dark_subtract,
        args.fsmp_time_mode,
        target_pe=getattr(
            args,
            "fsmp_target_pe",
            fsmp_data.get("fsmp_target_pe"),
        ),
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
        ):
            data = candidate
    if data is None:
        data = build_channel_histograms_multi(
            fsmp_data,
            sources,
            use_tts=not args.no_tts,
            irf_sigma_hamamatsu=args.irf_sigma_hamamatsu,
            irf_sigma_nnvt=args.irf_sigma_nnvt,
            dark_subtract=args.dark_subtract,
            dark_window=args.dark_window,
        )
        np.savez_compressed(args.aligned_data, **data)
        print(f"Wrote {args.aligned_data}")
    plot_data(data, args.opt, args.plot_range, args.run)
    plot_summary_data(data, args.summary_opt, args.plot_range, args.run, args.z)
    write_comparison_stats_csv(args.stats_csv, data, args.run, args.z)


def build_channel_histograms_multi(
    fsmp_data,
    sim_data_list,
    alignment="l1",
    use_tts=True,
    irf_sigma_hamamatsu=0.0,
    irf_sigma_nnvt=0.0,
    dark_subtract=False,
    dark_window=30.0,
):
    """Build aligned density and diagnostic arrays for simulation sources."""
    if alignment not in {"l1", "peak"} or not sim_data_list:
        raise ValueError(
            "alignment must be l1/peak and at least one source is required"
        )
    channels = np.asarray(fsmp_data["channels"], dtype=np.int64)
    pmt_type = np.asarray(fsmp_data["pmt_type"], dtype=np.int16)
    tts = np.asarray(fsmp_data["tts"], dtype=np.float64)
    irf = np.asarray(
        [
            irf_sigma_for_pmt_type(value, irf_sigma_hamamatsu, irf_sigma_nnvt)
            for value in pmt_type
        ]
    )
    edges = np.asarray(sim_data_list[0]["edges"], dtype=np.float64)
    centers = np.asarray(sim_data_list[0]["centers"], dtype=np.float64)
    if any(
        not np.array_equal(edges, source["edges"])
        for source in sim_data_list[1:]
    ):
        raise ValueError("simulation sources use different histogram edges")
    width = float(edges[1] - edges[0])
    records = _records_by_channel(fsmp_data)
    n_sources = len(sim_data_list)
    run_density = []
    raw_density = []
    sim_density = []
    run_peak = []
    sim_peak = []
    run_entries = []
    sim_entries = []
    run_events = []
    run_total = []
    sim_events = []
    sim_total = []
    l1_values = []
    wasserstein_values = []
    dark_stats = []
    for channel, channel_tts, channel_irf in tqdm(
        list(zip(channels, tts, irf)), desc="Channel histograms"
    ):
        record = records.get(
            int(channel),
            {"time": np.empty(0), "eid": np.empty(0), "pe_count": np.empty(0)},
        )
        times = np.asarray(record["time"], dtype=np.float64)
        peak_run = estimate_peak_from_times(times, len(edges) - 1)
        source_result = []
        for source in sim_data_list:
            source_channels = np.asarray(source["channels"], dtype=np.int64)
            source_index = np.searchsorted(source_channels, channel)
            if (
                source_index >= len(source_channels)
                or source_channels[source_index] != channel
            ):
                raw = np.zeros(len(edges) - 1)
                entries = total = event_count = 0
            else:
                raw = np.asarray(source["simulation_raw_density"])[source_index]
                entries = int(
                    np.asarray(source["simulation_entries"])[source_index]
                )
                total = int(
                    np.asarray(
                        source.get(
                            "simulation_total_pe", source["simulation_entries"]
                        )
                    )[source_index]
                )
                event_count = int(
                    np.asarray(
                        source.get(
                            "simulation_events", [-1] * len(source_channels)
                        )
                    )[source_index]
                )
            observed = (
                convolve_density(raw, width, channel_tts) if use_tts else raw
            )
            if channel_irf > 0:
                observed = convolve_density(observed, width, channel_irf)
            source_result.append((raw, observed, entries, total, event_count))
        reference_peak = estimate_peak_from_density(
            centers, source_result[0][1]
        )
        shift = reference_peak - peak_run
        shift = float(shift) if np.isfinite(shift) else 0.0
        if alignment == "l1":
            shift, aligned, _ = local_l1_alignment(
                times, source_result[0][1], edges, shift
            )
        else:
            aligned = density_hist(times + shift, edges)
        normalization = int(np.histogram(times + shift, bins=edges)[0].sum())
        aligned, stats = (
            subtract_dark_noise(
                times, aligned, edges, peak_run, dark_window, normalization
            )
            if dark_subtract
            else (aligned, _empty_dark_stats())
        )
        dark_stats.append(stats)
        run_density.append([aligned] * n_sources)
        run_peak.append([peak_run] * n_sources)
        run_entries.append([len(times)] * n_sources)
        run_events.append([len(np.unique(record["eid"]))] * n_sources)
        run_total.append([int(np.sum(record["pe_count"]))] * n_sources)
        raw_density.append([item[0] for item in source_result])
        sim_density.append([item[1] for item in source_result])
        sim_peak.append(
            [
                estimate_peak_from_density(centers, item[1])
                for item in source_result
            ]
        )
        sim_entries.append([item[2] for item in source_result])
        sim_total.append([item[3] for item in source_result])
        sim_events.append([item[4] for item in source_result])
        l1_values.append(
            [l1_distance(aligned, item[1], width) for item in source_result]
        )
        wasserstein_values.append(
            [
                wasserstein_distance(aligned, item[1], width)
                for item in source_result
            ]
        )

    def transpose(values, dtype=float):
        """Convert channel-major lists to source-major cache arrays."""
        array = np.asarray(values, dtype=dtype)
        return array.transpose(1, 0, 2) if array.ndim == 3 else array.T

    result = {
        "channels": channels,
        "pmt_type": pmt_type,
        "tts": tts,
        "irf_sigma_by_channel": irf,
        "z": np.asarray(fsmp_data["z"], dtype=float),
        "theta": np.asarray(fsmp_data["theta"], dtype=float),
        "edges": edges,
        "centers": centers,
        "run_density": transpose(run_density),
        "simulation_raw_density": transpose(raw_density),
        "simulation_density": transpose(sim_density),
        "run_peak": transpose(run_peak),
        "simulation_peak": transpose(sim_peak),
        "run_entries": transpose(run_entries, np.int64),
        "simulation_entries": transpose(sim_entries, np.int64),
        "run_events": transpose(run_events, np.int64),
        "run_total_pe": transpose(run_total, np.int64),
        "simulation_events": transpose(sim_events, np.int64),
        "simulation_total_pe": transpose(sim_total, np.int64),
        "l1_distance": transpose(l1_values),
        "wasserstein_distance": transpose(wasserstein_values),
        "simulation_labels": np.asarray(
            [
                str(np.asarray(source.get("label", "Simulation")).item())
                for source in sim_data_list
            ]
        ),
        "simulation_kinds": np.asarray(
            [
                str(np.asarray(source.get("kind", "simulation")).item())
                for source in sim_data_list
            ]
        ),
        "processing": np.asarray(
            processing_string(
                use_tts,
                irf_sigma_hamamatsu,
                irf_sigma_nnvt,
                dark_subtract,
                fsmp_data.get("fsmp_time_mode", "unknown"),
                target_pe=fsmp_data.get("fsmp_target_pe"),
            )
        ),
        "fsmp_time_mode": np.asarray(
            fsmp_data.get("fsmp_time_mode", "unknown")
        ),
    }
    for key in (
        "prepeak_pe",
        "rate_per_ns",
        "window_ns",
        "window_low_ns",
        "window_high_ns",
        "density_floor",
        "normalization_pe",
        "expected_pe",
        "removed_pe",
    ):
        values = np.asarray([item[key] for item in dark_stats])
        dtype = np.int64 if key in {"prepeak_pe", "normalization_pe"} else float
        result[f"run_dark_{key}"] = np.repeat(
            values.astype(dtype)[None, :], n_sources, axis=0
        )
    return result


def compute_lucid_sim_data(args, channels):
    """Compatibility wrapper for the LUCiD cache builder."""
    return _compute_sim_cache(args, channels, "lucid")


def compute_junosw_sim_data(args, channels):
    """Compatibility wrapper for the JUNOsw cache builder."""
    return _compute_sim_cache(args, channels, "junosw")


def read_junosw_sim_histograms(
    sim_files, sim_group, edges, chunk_size, match_mode, channels
):
    """Delegate Track/CDHit matching to ``compare_sim``."""

    class _Options:
        junosw_sim_group = sim_group
        sim_chunk_size = chunk_size
        junosw_sim_match_mode = match_mode

    return _read_junosw_histograms(sim_files, _Options(), edges, channels)


def build_channel_histograms(
    fsmp_data,
    sim_data,
    alignment="l1",
    use_tts=True,
    irf_sigma=0.0,
    irf_sigma_hamamatsu=None,
    irf_sigma_nnvt=None,
    dark_subtract=False,
    dark_window=30.0,
):
    """Build a one-source result through the multi-source implementation."""
    hamamatsu, nnvt = resolve_irf_sigmas(
        irf_sigma, irf_sigma_hamamatsu, irf_sigma_nnvt
    )
    return build_channel_histograms_multi(
        fsmp_data,
        [sim_data],
        alignment=alignment,
        use_tts=use_tts,
        irf_sigma_hamamatsu=hamamatsu,
        irf_sigma_nnvt=nnvt,
        dark_subtract=dark_subtract,
        dark_window=dark_window,
    )


def simulation_arrays(data):
    """Compatibility wrapper returning normalized source arrays."""
    return _source_arrays(data)


def metric_arrays(data):
    """Compatibility wrapper returning L1 and Wasserstein arrays."""
    _, _, _, l1, wasserstein = _metric_arrays(data)
    return l1, wasserstein


def simulation_colors(data, labels):
    """Compatibility wrapper for stable source colors."""
    return _simulation_colors(data, labels)


def processing_title(data):
    """Compatibility wrapper for cache processing descriptions."""
    return _processing_title(data)


def source_count_arrays(data, labels):
    """Compatibility wrapper for source-by-channel count arrays."""
    return _source_counts(data, labels)


def format_count(value):
    """Compatibility wrapper for human-readable integer counts."""
    return _format_count(value)


if __name__ == "__main__":
    main()
