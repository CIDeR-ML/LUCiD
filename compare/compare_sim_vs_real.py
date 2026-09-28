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
The channel subset is selected before reading any source.  All three inputs
are therefore restricted to the same reproducible channel subset, and FSMP
loading stops after reaching that subset's first LUCiD PE total, finishing the
current file first.
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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

from compare.laser_compare_irf import (
    build_channel_kernels,
    irf_descriptor,
)
from compare.laser_compare_metrics import (
    JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
    build_channel_histograms_multi,
    consensus_l1_alignment,
    density_hist,
    estimate_peak_from_density,
    processing_string,
    simulation_qe_factors,
)
from compare.laser_compare_csv import (
    write_channel_manifest,
    write_comparison_stats_csv,
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

resolve_run = base.resolve_run
iter_fsmp_files = base.iter_fsmp_files
rtraw_path_for = base.rtraw_path_for
read_monitor_times = base.read_monitor_times
select_channels = base.select_channels
load_channel_selection = base.load_channel_selection
read_pmt_info = base.read_pmt_info
attach_pmt_coordinates = base.attach_pmt_coordinates
load_data = base.load_data




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
    chunks, batch, channels, monitor_eid, monitor_time
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
            monitor_eid, monitor_time = _monitor_arrays(
                rtraw_path_for(path, fsmp_dir, rtraw_dir)
            )
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
                monitor_keep = np.isin(row_eid, monitor_eid)
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
                )
            physical_events += len(monitor_eid)
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


def selected_channels(args, require_files=True, files=None):
    """Select final plot channels and optionally locate FSMP files."""
    files = list(iter_fsmp_files(args.fsmp_dir)) if files is None else files
    if require_files and not files:
        raise FileNotFoundError(
            f"No FSMP Parquet files found in {args.fsmp_dir}"
        )
    if args.channel_selection is not None:
        channels = load_channel_selection(args.channel_selection)
    elif args.north:
        channels = np.arange(NORTH_CHANNEL_COUNT, dtype=np.int64)
    else:
        channels = select_channels(args.number, args.seed)
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
    if kind == "lucid":
        root = sim_dir if sim_dir is not None else args.lucid_sim_dir[0]
        files = iter_sim_files(root)
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
        "amplitude_schema": np.asarray(2, dtype=np.int16),
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
        "fsmp_physical_events": records["physical_events"],
        "fsmp_event_samples": records["event_samples"],
        "fsmp_target_pe": np.asarray(
            -1 if target_pe is None else target_pe, dtype=np.int64
        ),
    }
    args.fsmp_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fsmp_data, **data)
    print(f"Wrote {args.fsmp_data}")
    return data






def _apply_cli_labels(sources, lucid_labels, junosw_label):
    """Override cached source labels with the CLI-provided ones.

    Input: loaded or computed source caches and the CLI label values.
    Output: none; each source cache dict is updated in place so plots and
    CSV exports carry the labels of the current invocation.
    """
    lucid_index = 0
    for source in sources:
        kind = str(np.asarray(source.get("kind", "")).item())
        if kind == "lucid":
            source["label"] = np.asarray(lucid_labels[lucid_index])
            lucid_index += 1
        elif kind == "junosw":
            source["label"] = np.asarray(junosw_label)


def source_amplitude_totals(sources):
    """Selected-channel total MC amplitude (signal PE/event) per source.

    Input: loaded simulation source caches with per-channel PE totals and
    event counts.
    Output: list of floats, one total amplitude per source.
    """
    totals = []
    for source in sources:
        total_pe = source.get("simulation_total_pe")
        if total_pe is None:
            total_pe = source["simulation_entries"]
        total_pe = np.asarray(total_pe, dtype=np.float64)
        events = np.asarray(source["simulation_events"], dtype=np.float64)
        mean_pe = np.divide(
            total_pe, events, out=np.zeros_like(total_pe), where=events > 0
        )
        totals.append(float(np.sum(mean_pe)))
    return totals


def auto_source_intensity_scale(sources, qe_factors=None):
    """Correction multipliers scaling LUCiD sources to the JUNOsw level.

    Input: loaded simulation source caches and the per-source QE factors
    matching the amplitude figure's display convention (LUCiD carries the
    equivalent 532 nm QE, JUNOsw 1); every LUCiD source is divided by its
    QE-weighted LUCiD:JUNOsw total-amplitude ratio over the selected
    channels, the JUNOsw reference source keeps multiplier 1.
    Output: float64 ndarray of per-source multipliers; raises when no
    usable JUNOsw source or non-positive totals are present.
    """
    totals = source_amplitude_totals(sources)
    if qe_factors is None:
        qe_factors = np.ones(len(sources), dtype=np.float64)
    weighted = [
        factor * value for factor, value in zip(qe_factors, totals)
    ]
    reference = next(
        (
            value
            for source, value in zip(sources, weighted)
            if str(np.asarray(source.get("kind", "")).item()) == "junosw"
        ),
        None,
    )
    if reference is None or reference <= 0:
        raise ValueError(
            "auto source-intensity scale needs a JUNOsw source"
        )
    scales = np.ones(len(sources), dtype=np.float64)
    for index, (source, value) in enumerate(zip(sources, weighted)):
        kind = str(np.asarray(source.get("kind", "")).item())
        if kind != "lucid":
            continue
        if value <= 0:
            raise ValueError(
                "auto source-intensity scale needs positive MC totals"
            )
        scales[index] = reference / value
    return scales


def report_source_intensity(sources, source_scale, mode, qe_factors=None):
    """Print totals, QE factors, weighted ratios and the multipliers.

    Input: simulation sources, the resolved per-source multipliers, the
    mode string ("auto" or "explicit") and the per-source QE factors used
    by the amplitude figure.
    Output: none; logs one line per source.
    """
    totals = source_amplitude_totals(sources)
    if qe_factors is None:
        qe_factors = np.ones(len(sources), dtype=np.float64)
    weighted = [
        factor * value for factor, value in zip(qe_factors, totals)
    ]
    reference = next(
        (
            value
            for source, value in zip(sources, weighted)
            if str(np.asarray(source.get("kind", "")).item()) == "junosw"
        ),
        None,
    )
    for source, value, qe, scale in zip(
        sources, totals, qe_factors, source_scale
    ):
        label = str(np.asarray(source.get("label", "Simulation")).item())
        kind = str(np.asarray(source.get("kind", "")).item())
        line = f"Total MC amplitude {label}: {value:.6g} PE/event"
        if qe != 1.0:
            line += f"; QE factor {qe:.6g}"
        if reference and kind != "junosw":
            ratio = qe * value / reference
            line += f"; QE-weighted LUCiD:JUNOsw = {ratio:.6g}"
        print(f"{line}; applied scale = {scale:.6g} ({mode})")


def write_amplitude_intensity_scale(
    path,
    run,
    sources,
    source_scale,
    qe_factors,
    channel_count,
    dark_window,
):
    """Record the amplitude-figure source-intensity scales as JSON.

    Input: output path, run number, simulation sources, the applied
    per-source multipliers and QE factors, the selected-channel count and
    the dark sideband margin in ns.
    Output: none; writes the JSON record next to the amplitude figure.
    """
    records = []
    for source, scale, qe in zip(sources, source_scale, qe_factors):
        records.append(
            {
                "kind": str(np.asarray(source.get("kind", "")).item()),
                "label": str(
                    np.asarray(source.get("label", "Simulation")).item()
                ),
                "qe_factor": float(qe),
                "source_intensity_scale": float(scale),
            }
        )
    reference = next(
        (item["label"] for item in records if item["kind"] == "junosw"),
        None,
    )
    payload = {
        "run": int(run),
        "reference_source": reference,
        "common_valid_pmt_count": int(channel_count),
        "dark_margin_ns": float(dark_window),
        "sources": records,
    }
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {path}")


def build_parser():
    """Build the argument parser shared by the comparison CLIs.

    Input: none.  Output: the argparse.ArgumentParser (the TTS-vs-IRF
    check adds its own flags on top before parsing).
    """
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
    parser.add_argument("-n", type=int, default=1000, dest="number")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--north", action="store_true")
    parser.add_argument(
        "--channel-selection",
        type=Path,
        default=None,
        help="CSV with a `channel` column; overrides -n/--seed/--north",
    )
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
    parser.add_argument(
        "--irf-h5",
        type=Path,
        default=None,
        help="measured per-channel IRF kernel file; enabling it "
        "replaces the per-PMT TTS convolution (the IRF contains "
        "the TTS)",
    )
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
    parser.add_argument("--dark-window", type=float, default=30.0)
    parser.add_argument(
        "--source-intensity-scale",
        type=float,
        nargs="+",
        default=None,
        metavar="SCALE",
        help=(
            "one global intensity scale per simulation source; "
            "default: auto JUNOsw/LUCiD total-amplitude ratio"
        ),
    )
    parser.add_argument("--print-fsmp-row", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("compare_sim"))
    parser.add_argument("--output-tag", default="")
    return parser


def parse_args(argv=None, parser=None):
    """Parse the three-source comparison command line.

    Input: optional argv and an optional pre-built parser (used by the
    TTS-vs-IRF check to add its own flags).
    Output: the finalized namespace.
    """
    if parser is None:
        parser = build_parser()
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
    if args.source_intensity_scale is not None:
        n_sources = len(args.lucid_sim_dir or []) + int(
            args.junosw_sim_dir is not None
        )
        scales = np.asarray(args.source_intensity_scale, dtype=np.float64)
        if (
            scales.shape != (n_sources,)
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0.0)
        ):
            parser.error(
                "--source-intensity-scale needs one positive value per "
                "simulation source"
            )
    if not any((args.lucid_sim_dir, args.junosw_sim_dir)):
        parser.error("provide at least one simulation input")
    args.run = resolve_run(args.z, args.run_table)
    args.fsmp_dir = args.fsmp_dir or Path(DEFAULT_FSMP_ROOT) / str(args.run)
    args.rtraw_dir = args.rtraw_dir / str(args.run)
    args.output_dir = args.output_root / str(args.run)
    if args.channel_selection is not None:
        suffix = args.channel_selection.stem
    elif args.north:
        suffix = "north"
    else:
        suffix = str(args.number)
    if args.long:
        suffix += "_long"
    tag = args.output_tag.strip().replace(" ", "_")
    suffix = f"{suffix}_{tag}" if tag else suffix
    args.opt = args.output_dir / f"compare_sim_{suffix}.pdf"
    args.summary_opt = args.output_dir / f"compare_sim_{suffix}_summary.pdf"
    args.amplitude_opt = (
        args.output_dir / f"compare_sim_{suffix}_amplitude.pdf"
    )
    args.fsmp_data = args.output_dir / f"fsmp_data_{suffix}_all.npz"

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
        args.output_dir / f"channel_selection_{suffix}_all.csv"
    )
    return args


def _load_matching_cache(
    path,
    channels,
    kind=None,
    target_pe=None,
    require_exposure=False,
):
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
        schema = int(np.asarray(data.get("amplitude_schema", 0)).item())
        if schema != 2:
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


def expected_source_labels(sources, smearing_variants=False):
    """Expected source label/kind arrays for the aligned-cache check.

    Input: simulation sources; smearing_variants appends the (TTS) and
    (IRF) suffixes used by the TTS-vs-IRF check.
    Output: (labels, kinds) numpy arrays.
    """
    labels = [
        str(np.asarray(source["label"]).item()) for source in sources
    ]
    kinds = [
        str(np.asarray(source["kind"]).item()) for source in sources
    ]
    if smearing_variants:
        labels = [
            f"{label} ({variant})"
            for label in labels
            for variant in ("TTS", "IRF")
        ]
        kinds = [kind for kind in kinds for _ in ("TTS", "IRF")]
    return np.asarray(labels), np.asarray(kinds)


def prepare_inputs(args):
    """Load channels, FSMP data, simulation sources and smearing inputs.

    Input: parsed arguments.  Output: SimpleNamespace with fsmp_data,
    sources, channels, source_scale, use_tts, irf_kernels,
    irf_fwhm_by_channel and irf_description.
    """
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
        required_fsmp = {"fsmp_physical_events", "fsmp_event_samples"}
        if not required_fsmp.issubset(fsmp_data):
            raise ValueError(
                "--plot-only FSMP cache lacks amplitude exposure fields; "
                "recompute without --plot-only"
            )
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
                require_exposure=True,
            )
        if fsmp_data is None:
            fsmp_data = compute_fsmp_data(args, channels, files)
    if not sources:
        raise RuntimeError("No simulation source was loaded")
    _apply_cli_labels(sources, args.lucid_label, args.junosw_label)
    qe_factors = simulation_qe_factors(
        sources, args.apply_lucid_qe, args.lucid_qe_effective
    )
    if args.source_intensity_scale is None:
        source_scale = auto_source_intensity_scale(sources, qe_factors)
        scale_mode = "auto"
    else:
        source_scale = np.asarray(
            args.source_intensity_scale, dtype=np.float64
        )
        scale_mode = "explicit"
    report_source_intensity(sources, source_scale, scale_mode, qe_factors)
    write_amplitude_intensity_scale(
        args.output_dir / "amplitude_intensity_scale.json",
        args.run,
        sources,
        source_scale,
        qe_factors,
        len(channels),
        args.dark_window,
    )
    fsmp_data = attach_pmt_coordinates(fsmp_data, args.pmt_map)
    use_tts = not args.no_tts
    irf_kernels = None
    irf_fwhm_by_channel = {}
    irf_description = "0"
    if args.irf_h5 is not None:
        bin_width = float(
            sources[0]["edges"][1] - sources[0]["edges"][0]
        )
        irf_kernels, irf_meta, irf_orphans = build_channel_kernels(
            args.irf_h5, args.pmt_map, bin_width
        )
        irf_fwhm_by_channel = {
            channel: info["fwhm"] for channel, info in irf_meta.items()
        }
        irf_description = irf_descriptor(args.irf_h5, irf_kernels)
        missing = [
            int(channel)
            for channel in fsmp_data["channels"]
            if int(channel) not in irf_kernels
        ]
        print(
            f"Measured IRF: kernels for {len(irf_kernels)} channels; "
            f"{len(missing)} selected channels have no measured "
            f"kernel; {len(irf_orphans)} IRF serials are not in "
            "the PMT map"
        )
        if use_tts:
            print("Measured IRF enabled: TTS convolution disabled")
            use_tts = False
    write_channel_manifest(
        args.channel_manifest,
        fsmp_data,
        args.run,
        args.z,
        irf_fwhm_by_channel=irf_fwhm_by_channel,
    )
    return SimpleNamespace(
        fsmp_data=fsmp_data,
        sources=sources,
        channels=channels,
        source_scale=source_scale,
        use_tts=use_tts,
        irf_kernels=irf_kernels,
        irf_fwhm_by_channel=irf_fwhm_by_channel,
        irf_description=irf_description,
    )


def emit_results(args, data, time_difference_panels=True):
    """Render all figures and write the comparison statistics CSV.

    Input: parsed arguments, the aligned metrics dictionary and the
    difference-panel switch.  Output: none (files written).
    """
    plot_data(
        data,
        args.opt,
        args.plot_range,
        args.run,
        time_difference_panels=time_difference_panels,
    )
    plot_summary_data(data, args.summary_opt, args.plot_range, args.run, args.z)
    plot_amplitude_data(
        data,
        args.amplitude_opt,
        args.plot_range,
        args.run,
        apply_lucid_qe=args.apply_lucid_qe,
        lucid_qe_eff=args.lucid_qe_effective,
    )
    write_comparison_stats_csv(args.stats_csv, data, args.run, args.z)


def main(argv=None):
    """Compute/load caches and render the three-source comparison."""
    args = parse_args(argv)
    prepared = prepare_inputs(args)
    processing = processing_string(
        prepared.use_tts,
        prepared.irf_description,
        args.add_dark_to_simulation,
        target_pe=getattr(
            args,
            "fsmp_target_pe",
            prepared.fsmp_data.get("fsmp_target_pe"),
        ),
        prompt_window=args.prompt_window,
        late_window=args.late_window,
        alignment=args.alignment,
        source_intensity_scale=prepared.source_scale,
    )
    data = None
    if args.aligned_data.exists() and not args.force:
        candidate = load_data(args.aligned_data)
        labels, kinds = expected_source_labels(prepared.sources)
        if (
            str(candidate.get("processing", "")) == processing
            and np.array_equal(
                candidate.get("channels"), prepared.fsmp_data["channels"]
            )
            and np.array_equal(
                candidate.get("edges"), prepared.sources[0]["edges"]
            )
            and np.array_equal(candidate.get("simulation_labels"), labels)
            and np.array_equal(candidate.get("simulation_kinds"), kinds)
        ):
            data = candidate
    if data is None:
        data = build_channel_histograms_multi(
            prepared.fsmp_data,
            prepared.sources,
            use_tts=prepared.use_tts,
            irf_kernels=prepared.irf_kernels,
            irf_fwhm_by_channel=prepared.irf_fwhm_by_channel,
            irf_description=prepared.irf_description,
            add_dark_to_simulation=args.add_dark_to_simulation,
            dark_window=args.dark_window,
            prompt_window=args.prompt_window,
            late_window=args.late_window,
            alignment=args.alignment,
            simulation_source_scale=prepared.source_scale,
        )
        np.savez_compressed(args.aligned_data, **data)
        print(f"Wrote {args.aligned_data}")
    emit_results(args, data)



if __name__ == "__main__":
    main()
