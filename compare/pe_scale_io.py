"""Build, load and validate all-channel caches for raw PE-scale plots."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from compare import compare_sim as base
from compare import compare_sim_vs_real as response


RAW_CACHE_SCHEMA = 1
RAW_TIME_RANGE_NS = (-1000.0, 1000.0)
RAW_TIME_BINS = 2000


def _load_npz(path):
    """Load an NPZ cache into ordinary NumPy arrays.

    Input: ``path`` is a cache file written by the laser comparison reader.
    Output: dictionary whose values remain valid after the NPZ file closes.
    """
    with np.load(path, allow_pickle=False) as cache:
        return {name: cache[name] for name in cache.files}


def _scalar(cache, name, path):
    """Return one required scalar cache field as a Python value."""
    if name not in cache:
        raise ValueError(f"{path} lacks required field {name}")
    value = np.asarray(cache[name])
    if value.ndim != 0:
        raise ValueError(f"{path} field {name} must be scalar")
    return value.item()


def _array(cache, name, path, channels):
    """Return one required channel-length array from a cache."""
    if name not in cache:
        raise ValueError(f"{path} lacks required field {name}")
    value = np.asarray(cache[name])
    if value.shape != channels.shape:
        raise ValueError(
            f"{path} field {name} shape {value.shape} does not match "
            f"channels {channels.shape}"
        )
    return value


def valid_raw_pe_scale_cache(
    path, channels, fields, kind=None, first_pe_only=False
):
    """Check whether one cache has matching channels and required fields.

    Input: cache path, expected channels, fields, source kind and PE mode.
    Output: ``True`` only when the cache is suitable for raw PE-scale use.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            valid = (
                all(name in cache.files for name in fields)
                and np.array_equal(cache["channels"], channels)
            )
            if kind is not None:
                valid &= str(np.asarray(cache["kind"]).item()) == kind
            if "fsmp_monitor_join" in cache.files:
                valid &= not bool(np.asarray(cache["fsmp_monitor_join"]).item())
            if "fsmp_first_pe_only" in cache.files:
                valid &= bool(
                    np.asarray(cache["fsmp_first_pe_only"]).item()
                ) == bool(first_pe_only)
            elif first_pe_only:
                valid = False
            return valid
    except (KeyError, OSError, ValueError):
        return False


def _write_raw_cache(path, data):
    """Write one compressed raw PE-scale cache and report its location."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Wrote {path}")


def _build_fsmp_raw_cache(args, channels, edges, output):
    """Read all FSMP PEs without monitor joining into one raw cache.

    Input: build arguments, detector channels, raw histogram edges and output.
    Output: no return; writes all-PE channel counts and read audit fields.
    """
    files = list(base.iter_fsmp_files(args.fsmp_dir))
    if not files:
        raise FileNotFoundError(
            f"No FSMP Parquet files found in {args.fsmp_dir}"
        )
    pmt_type, tts = base.read_pmt_info(channels, args.pmt_map, args.pmt)
    records = response.read_fsmp_histograms_by_channel(
        files,
        args.fsmp_dir,
        args.rtraw_dir,
        channels,
        edges,
        args.run,
        file_limit=args.fsmp_file_limit,
        target_pe=None,
        first_pe_only=args.fsmp_first_pe,
        monitor_join=False,
    )
    if not np.any(records["total_pe"]):
        raise RuntimeError(f"No RUN {args.run} FSMP PE records were selected")
    _write_raw_cache(
        output,
        {
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
            "fsmp_monitor_join": np.asarray(False),
            "fsmp_first_pe_only": np.asarray(args.fsmp_first_pe),
            "raw_pe_scale_cache_schema": np.asarray(
                RAW_CACHE_SCHEMA, dtype=np.int16
            ),
        },
    )


def _simulation_raw_cache_data(channels, edges, result, label, kind):
    """Return one common simulation raw-cache dictionary from reader output."""
    density, entries, total_pe, events = result
    if not np.any(entries):
        raise RuntimeError(f"No {kind} simulation PE records were selected")
    return {
        "channels": channels,
        "edges": edges,
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "simulation_raw_density": density,
        "simulation_entries": entries,
        "simulation_total_pe": total_pe,
        "simulation_events": events,
        "amplitude_schema": np.asarray(2, dtype=np.int16),
        "label": np.asarray(label),
        "kind": np.asarray(kind),
        "raw_pe_scale_cache_schema": np.asarray(
            RAW_CACHE_SCHEMA, dtype=np.int16
        ),
    }


def _build_lucid_raw_cache(args, channels, edges, output):
    """Read all LUCiD PEs into one raw PE-scale cache.

    Input: build arguments, detector channels, raw histogram edges and output.
    Output: no return; writes all-PE LUCiD channel counts.
    """
    files = response.iter_sim_files(args.lucid_sim_dir)
    if args.lucid_file_limit is not None:
        files = files[:args.lucid_file_limit]
    if not files:
        raise FileNotFoundError(
            f"No LUCiD Parquet files found in {args.lucid_sim_dir}"
        )
    result = response.read_lucid_sim_histograms(
        files, edges, args.sim_chunk_size, channels
    )
    _write_raw_cache(
        output,
        _simulation_raw_cache_data(
            channels, edges, result, args.lucid_label, "lucid"
        ),
    )


def _build_junosw_raw_cache(args, channels, edges, output):
    """Read all JUNOsW PEs into one raw PE-scale cache.

    Input: build arguments, detector channels, raw histogram edges and output.
    Output: no return; writes all-PE JUNOsW channel counts.
    """
    files = response.iter_junosw_sim_files(
        args.junosw_sim_dir, args.junosw_sim_file_glob
    )
    if args.junosw_file_limit is not None:
        files = files[:args.junosw_file_limit]
    if not files:
        raise FileNotFoundError(
            f"No JUNOsW HDF5 files found in {args.junosw_sim_dir}"
        )
    reader_args = SimpleNamespace(
        junosw_sim_group=args.junosw_sim_group,
        sim_chunk_size=args.sim_chunk_size,
        junosw_sim_match_mode=args.junosw_sim_match_mode,
    )
    result = response._read_junosw_histograms(
        files, reader_args, edges, channels
    )
    _write_raw_cache(
        output,
        _simulation_raw_cache_data(
            channels, edges, result, args.junosw_label, "junosw"
        ),
    )


def _require_build_path(value, option):
    """Raise a clear error when a missing cache has no source directory."""
    if value is None:
        raise ValueError(f"{option} is required to build the missing raw cache")


def ensure_raw_pe_scale_caches(args):
    """Reuse valid raw caches and build only missing source caches.

    Input: plotting arguments with ``--build-missing`` source parameters.
    Output: no return; requested cache files exist on successful completion.
    """
    channels = np.arange(base.FULL_CHANNEL_COUNT, dtype=np.int64)
    edges = np.linspace(*RAW_TIME_RANGE_NS, RAW_TIME_BINS + 1)
    fsmp_fields = {"fsmp_total_pe", "fsmp_physical_events", "pmt_type"}
    simulation_fields = {"simulation_total_pe", "simulation_events", "kind"}
    if not valid_raw_pe_scale_cache(
        args.fsmp_cache,
        channels,
        fsmp_fields,
        first_pe_only=args.fsmp_first_pe,
    ):
        _require_build_path(args.fsmp_dir, "--fsmp-dir")
        _build_fsmp_raw_cache(args, channels, edges, args.fsmp_cache)
    else:
        print(f"Reuse {args.fsmp_cache}")
    if not valid_raw_pe_scale_cache(
        args.lucid_cache, channels, simulation_fields, kind="lucid"
    ):
        _require_build_path(args.lucid_sim_dir, "--lucid-sim-dir")
        _build_lucid_raw_cache(args, channels, edges, args.lucid_cache)
    else:
        print(f"Reuse {args.lucid_cache}")
    if not valid_raw_pe_scale_cache(
        args.junosw_cache, channels, simulation_fields, kind="junosw"
    ):
        _require_build_path(args.junosw_sim_dir, "--junosw-sim-dir")
        _build_junosw_raw_cache(args, channels, edges, args.junosw_cache)
    else:
        print(f"Reuse {args.junosw_cache}")


def load_fsmp_raw_cache(path, pmt_map, with_histogram=False):
    """Load an all-PE FSMP cache and attach PMT theta.

    Input: cache must contain all-PE ``fsmp_total_pe`` and its recorded
    physical-event exposure. The cache must have been produced without
    monitor joining. Output: channel arrays needed by raw-count PE scale;
    ``with_histogram`` also loads the cached histogram and its edges for
    sideband dark-noise estimation.
    """
    path = Path(path)
    cache = _load_npz(path)
    channels = np.asarray(cache.get("channels", []), dtype=np.int64)
    if not len(channels):
        raise ValueError(f"{path} has no channels")
    table = pq.read_table(pmt_map, columns=["copy_number", "theta"])
    index = {
        int(channel): position
        for position, channel in enumerate(
            table["copy_number"].to_numpy(zero_copy_only=False)
        )
    }
    theta_all = table["theta"].to_numpy(zero_copy_only=False)
    theta = np.asarray(
        [theta_all[index[channel]] if channel in index else np.nan
         for channel in channels],
        dtype=np.float64,
    )
    result = {
        "channels": channels,
        "pmt_type": _array(cache, "pmt_type", path, channels).astype(np.int16),
        "theta": theta,
        "total_pe": _array(cache, "fsmp_total_pe", path, channels).astype(
            np.float64
        ),
        "events": float(_scalar(cache, "fsmp_physical_events", path)),
    }
    if with_histogram:
        edges = np.asarray(cache["fsmp_histogram_edges"], dtype=np.float64)
        histogram = np.asarray(cache["fsmp_histogram"], dtype=np.int64)
        if histogram.shape != (len(channels), len(edges) - 1):
            raise ValueError(f"{path} histogram shape does not match edges")
        result["histogram_edges"] = edges
        result["histogram"] = histogram
    return result


def load_simulation_raw_cache(path, channels):
    """Load one simulation cache for the raw all-PE count comparison.

    Input: ``path`` is a LUCiD or JUNOsW cache and ``channels`` defines the
    FSMP order.  Output: source label, kind, all-PE counts and event exposure.
    """
    path = Path(path)
    cache = _load_npz(path)
    cached_channels = np.asarray(cache.get("channels", []), dtype=np.int64)
    if not np.array_equal(cached_channels, channels):
        raise ValueError(f"{path} channel order does not match FSMP cache")
    return {
        "label": str(_scalar(cache, "label", path)),
        "kind": str(_scalar(cache, "kind", path)),
        "total_pe": _array(cache, "simulation_total_pe", path, channels).astype(
            np.float64
        ),
        "events": _array(cache, "simulation_events", path, channels).astype(
            np.float64
        ),
    }


DARK_PEAK_MIN_COUNTS = 100


def fsmp_dark_counts(total_pe, histogram, edges, margin_ns=30.0):
    """Estimate per-channel dark counts from the pre-peak sideband.

    Input: all-PE channel counts, the cached raw histogram and its bin
    edges, and the safety margin before each channel's direct-light peak.
    Channels with fewer than ``DARK_PEAK_MIN_COUNTS`` PE use the
    all-channel aggregate peak.  Output: expected dark counts on the full
    histogram span and an audit dictionary with the window definition.
    """
    total_pe = np.asarray(total_pe, dtype=np.float64)
    histogram = np.asarray(histogram, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.float64)
    if not np.isfinite(margin_ns) or margin_ns < 0.0:
        raise ValueError("dark margin must be a non-negative number")
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = float(np.median(np.diff(edges)))
    span = float(edges[-1] - edges[0])
    global_peak = float(centers[int(np.argmax(histogram.sum(axis=0)))])
    peak_time = centers[np.argmax(histogram, axis=1)]
    low_count = total_pe < DARK_PEAK_MIN_COUNTS
    peak_time = np.where(low_count, global_peak, peak_time)
    window_end = np.searchsorted(centers, peak_time - margin_ns)
    cumulative = np.cumsum(histogram, axis=1, dtype=np.int64)
    rows = np.arange(len(total_pe))
    clipped = np.clip(window_end - 1, 0, histogram.shape[1] - 1)
    prepeak = np.where(window_end > 0, cumulative[rows, clipped], 0)
    width = window_end.astype(np.float64) * bin_width
    valid = width > 0.0
    dark_pe = np.zeros(len(total_pe), dtype=np.float64)
    dark_pe[valid] = prepeak[valid] * (span / width[valid])
    info = {
        "margin_ns": float(margin_ns),
        "global_peak_ns": global_peak,
        "span_ns": span,
        "low_count_channels": int(np.count_nonzero(low_count)),
        "no_window_channels": int(np.count_nonzero(~valid)),
    }
    return dark_pe, info


def raw_pe_scale_inputs(fsmp, simulations, lucid_qe_effective=0.097,
                        dark_pe=None, min_pe_per_event=0.0,
                        per_event=False):
    """Build the all-PE raw-count ratio used by the reference PDF.

    Input: validated FSMP and simulation cache dictionaries;
    ``dark_pe`` optionally holds per-channel dark counts subtracted from
    the FSMP numerator.
    Output: arrays with one all-PE count ratio per PMT and simulation source.  
    This preserves raw-PDF definition: no monitor join, event normalization, peak
    gate, dark subtraction or source-intensity scale.  With
    ``dark_pe`` the corrected ratio is added as ``pe_scale_dark``;
    ``min_pe_per_event`` filters channels whose corrected FSMP counts
    fall below that per-event exposure (raw ``pe_scale`` stays raw).
    With ``per_event`` both ratios use per-event averaged PE counts.
    """
    if not 0.0 < lucid_qe_effective <= 1.0:
        raise ValueError("LUCiD QE factor must be in (0, 1]")
    if not np.isfinite(min_pe_per_event) or min_pe_per_event < 0.0:
        raise ValueError("min_pe_per_event must be a non-negative number")
    fsmp_mean = fsmp["total_pe"]
    corrected = None
    if dark_pe is not None:
        dark_pe = np.asarray(dark_pe, dtype=np.float64)
        if dark_pe.shape != fsmp_mean.shape or not np.all(
            np.isfinite(dark_pe)
        ):
            raise ValueError("dark_pe must be finite per channel")
        corrected = fsmp_mean - dark_pe
    numerator = fsmp_mean
    numerator_dark = corrected
    if per_event:
        numerator = numerator / fsmp["events"]
        if corrected is not None:
            numerator_dark = corrected / fsmp["events"]
    values = []
    labels = []
    kinds = []
    factors = []
    good = []
    candidate = np.isfinite(fsmp["theta"])
    candidate &= np.isin(fsmp["pmt_type"], (1, 2))
    for source in simulations:
        factor = (
            lucid_qe_effective
            if source["kind"].lower() == "lucid"
            else 1.0
        )
        mean = source["total_pe"] * factor
        if per_event:
            mean = mean / source["events"]
        valid = candidate & (fsmp_mean > 0.0) & (mean > 0.0)
        values.append(
            np.divide(
                numerator,
                mean,
                out=np.full(len(fsmp_mean), np.nan),
                where=valid,
            )
        )
        labels.append(source["label"])
        kinds.append(source["kind"])
        factors.append(factor)
        good.append(valid)
    result = {
        "channels": fsmp["channels"],
        "theta": fsmp["theta"],
        "fsmp_total_pe": fsmp_mean,
        "simulation_labels": np.asarray(labels),
        "simulation_kinds": np.asarray(kinds),
        "simulation_qe_factor": np.asarray(factors),
        "simulation_total_pe": np.asarray(
            [source["total_pe"] * factor for source, factor in
             zip(simulations, factors)]
        ),
        "pe_scale": np.asarray(values),
        "pe_scale_source_good_mask": np.asarray(good),
        "pe_per_event": np.asarray(bool(per_event)),
    }
    if corrected is not None:
        min_pe = float(min_pe_per_event) * fsmp["events"]
        dark_values = []
        dark_good = []
        for source, factor in zip(simulations, factors):
            mean = source["total_pe"] * factor
            if per_event:
                mean = mean / source["events"]
            valid = (
                candidate
                & (corrected > 0.0)
                & (corrected >= min_pe)
                & (mean > 0.0)
            )
            dark_values.append(
                np.divide(
                    numerator_dark,
                    mean,
                    out=np.full(len(corrected), np.nan),
                    where=valid,
                )
            )
            dark_good.append(valid)
        result["fsmp_dark_pe"] = dark_pe
        result["fsmp_min_pe_per_event"] = float(min_pe_per_event)
        result["fsmp_numerator_pe"] = corrected
        result["pe_scale_dark"] = np.asarray(dark_values)
        result["pe_scale_dark_source_good_mask"] = np.asarray(dark_good)
    return result
