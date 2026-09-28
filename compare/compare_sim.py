#!/usr/bin/env python
"""Compare LUCiD laser simulation with FSMP distributions."""

import argparse
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


PMT_TYPE_NAME = {
    1: "Hamamatsu PMT",
    2: "NNVT PMT",
}
FULL_CHANNEL_COUNT = 17612
NORTH_CHANNEL_COUNT = 145
METRIC_HISTOGRAM_BINS = 200
BRIEF_FSMP_FILE_LIMIT = 200
BRIEF_SIM_FILE_LIMIT = 20
FIGURE_SIZE = (5.5, 3.5)
RANDOM_SIM_ROOT = Path(
    "/mnt/stage/haoch/junores/laser_random_direction/532"
)


def resolve_run(z_m, run_table_file):
    run_table = pd.read_csv(run_table_file)
    required_columns = {"run", "z[m]"}
    missing_columns = required_columns.difference(run_table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{run_table_file} is missing columns: {missing}")

    matches = run_table[np.isclose(run_table["z[m]"], z_m, rtol=0.0, atol=1e-9)]
    if len(matches) == 0:
        raise ValueError(f"No run in {run_table_file} has z={z_m:g} m")
    if len(matches) > 1:
        runs = ", ".join(map(str, matches["run"].tolist()))
        raise ValueError(f"Multiple runs in {run_table_file} have z={z_m:g} m: {runs}")
    return int(matches.iloc[0]["run"])


def parse_args():
    psr = argparse.ArgumentParser()
    psr.add_argument(
        "-z",
        dest="z",
        type=float,
        default=0.0,
        metavar="Z_M",
        help="source z coordinate in m; defaults to 0",
    )
    psr.add_argument(
        "--run-table",
        dest="run_table",
        type=Path,
        default=Path("run_532.csv"),
        help="CSV mapping run numbers to z coordinates in metres",
    )
    psr.add_argument(
        "--sim-chunk-size",
        dest="sim_chunk_size",
        type=int,
        default=1_000_000,
        help="rows per chunk while scanning Sim/Track",
    )
    psr.add_argument(
        "--sim-group",
        dest="sim_group",
        type=str,
        default="Sim",
        help="simulation group containing Track and CDHit datasets",
    )
    psr.add_argument(
        "--sim-match-mode",
        choices=("auto", "event", "global"),
        default="auto",
        help="match CDHit to Track by event ranges when possible, or by global combined keys",
    )
    psr.add_argument(
        "--force",
        action="store_true",
        help="recompute cached FSMP and simulation data even if cache files exist",
    )
    psr.add_argument(
        "--plot-only",
        action="store_true",
        help="only plot from cached FSMP and simulation data; do not compute raw distributions",
    )
    psr.add_argument(
        "--brief",
        action="store_true",
        help="quick mode using only the first 200 FSMP files and first 20 simulation files",
    )
    psr.add_argument(
        "-n",
        dest="number",
        type=int,
        default=1000,
        help="number of channels to select",
    )
    psr.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=0,
        help="random seed for channel selection",
    )
    psr.add_argument(
        "--north",
        action="store_true",
        help="select fixed channels 0-144 instead of randomly selecting channels",
    )
    psr.add_argument(
        "--random",
        action="store_true",
        help=(
            "for -z 0, use laser_random_direction simulation data while "
            "reusing the normal FSMP cache"
        ),
    )
    psr.add_argument(
        "--bins",
        dest="bins",
        type=int,
        default=600,
        help="number of histogram bins for cached full-range distributions",
    )
    psr.add_argument(
        "--data-range",
        dest="data_range",
        type=float,
        nargs=2,
        default=(-100.0, 500.0),
        metavar=("LOW", "HIGH"),
        help="full time range saved to the simulation cache and used for alignment",
    )
    psr.add_argument(
        "--plot-range",
        dest="plot_range",
        type=float,
        nargs=2,
        default=(0.0, 500.0),
        metavar=("LOW", "HIGH"),
        help="x range shown in the pdf",
    )
    psr.add_argument(
        "--long",
        action="store_true",
        help="extend the cached data and plotted x range to 1000 ns and add _long to output names",
    )
    psr.add_argument(
        "--pmt-map",
        dest="pmt_map",
        type=Path,
        default=Path("/junofs/C/junoid_J25.7.3.pq"),
        help="PMT type table: copy_number -> pmt_type",
    )
    psr.add_argument(
        "--pmt",
        dest="pmt",
        type=Path,
        default=Path("/junofs/C/PMT/STD/9541.h5"),
        help="PMT calibration h5 with PMT/TTS",
    )
    args = psr.parse_args()
    if args.long:
        args.data_range = (args.data_range[0], 1000.0)
        args.plot_range = (args.plot_range[0], 1000.0)
    args.run = resolve_run(args.z, args.run_table)
    z_mm = args.z * 1000.0
    z_dir = "0" if z_mm == 0 else f"{z_mm:g}"
    if args.random and not np.isclose(args.z, 0.0, rtol=0.0, atol=1e-12):
        psr.error("--random is only supported with -z 0")
    args.fsmp_dir = Path("FSMP/PE") / str(args.run)
    args.rtraw_dir = Path("split_rtraw") / str(args.run)
    sim_root = RANDOM_SIM_ROOT if args.random else Path("laser/532")
    args.sim_dir = sim_root / z_dir
    output_root = "compare_sim_brief" if args.brief else "compare_sim"
    args.output_dir = Path(output_root) / str(args.run)
    fsmp_suffix = "north" if args.north else str(args.number)
    if args.long:
        fsmp_suffix += "_long"
    output_suffix = fsmp_suffix
    if args.random:
        output_suffix += "_random"
    args.opt = args.output_dir / f"compare_sim_{output_suffix}.pdf"
    args.summary_opt = args.output_dir / f"compare_sim_{output_suffix}_summary.pdf"
    args.fsmp_data = args.output_dir / f"fsmp_data_{fsmp_suffix}.npz"
    args.sim_data = args.output_dir / f"sim_data_{output_suffix}.npz"
    args.aligned_data = args.output_dir / f"aligned_data_{output_suffix}.npz"
    return args


def iter_fsmp_files(fsmp_dir):
    yield from sorted(fsmp_dir.glob("*/*.pq"))


def iter_sim_files(sim_dir):
    if sim_dir.is_file():
        return [sim_dir]
    return sorted(sim_dir.rglob("*.h5"))


def rtraw_path_for(fsmp_file, fsmp_dir, rtraw_dir):
    rel = fsmp_file.relative_to(fsmp_dir)
    return rtraw_dir / rel.with_suffix(".h5")


def read_monitor_times(rtraw_file):
    with h5.File(rtraw_file, "r") as f:
        monitor = f["CdMonitorWaveform"]["Waveform"][:]
        cdtrig = f["CdTrigger/Trigger"][:]

    monitor = monitor[monitor["pmt_id"] == 43303]

    df_trig = pd.DataFrame(
        {
            "id": cdtrig["id"],
            "eid": (
                cdtrig["trigger_time"]["sec"].astype(np.int64) * 1_000_000_000
                + cdtrig["trigger_time"]["nanosec"].astype(np.int64)
            )
            // 8,
        }
    )
    df_monitor = pd.DataFrame(
        {
            "id": monitor["id"],
            "t_monitor": np.argmin(monitor["adc"], axis=1),
        }
    )
    return pd.merge(df_trig, df_monitor, on="id", how="inner")[["eid", "t_monitor"]]


def first_pet_with_offset(pet, offset):
    if pet is None or len(pet) == 0:
        return np.nan
    return float(np.min(pet)) + float(offset)


def select_channels(number, seed):
    if number <= 0 or number > FULL_CHANNEL_COUNT:
        raise ValueError(f"-n must be between 1 and {FULL_CHANNEL_COUNT}")

    channels = np.arange(FULL_CHANNEL_COUNT, dtype=np.int64)
    if number == FULL_CHANNEL_COUNT:
        return channels

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(channels, size=number, replace=False))
    return selected.astype(np.int64)


def load_channel_selection(path):
    """Fixed channel list from a channel-selection CSV.

    Input: CSV path with a ``channel`` column.
    Output: sorted int64 ndarray of channel ids.
    """
    frame = pd.read_csv(path)
    return np.sort(frame["channel"].to_numpy(dtype=np.int64))


def selected_channels(args):
    fsmp_files = list(iter_fsmp_files(args.fsmp_dir))
    if not fsmp_files:
        raise FileNotFoundError(f"No second-level RUN {args.run} pq files found in {args.fsmp_dir}")

    if args.north:
        channels = np.arange(NORTH_CHANNEL_COUNT, dtype=np.int64)
        print("North mode: fixed channels 0-144")
    else:
        channels = select_channels(args.number, args.seed)
    print("Selected channels:", " ".join(map(str, channels)))
    return channels, fsmp_files


def read_pmt_types(channels, pmt_map_file):
    pmt_table = pd.read_parquet(pmt_map_file, columns=["copy_number", "pmt_type"])
    pmt_table = pmt_table[pmt_table["copy_number"].isin(channels)]
    pmt_type_by_ch = dict(zip(pmt_table["copy_number"], pmt_table["pmt_type"]))
    return np.array([pmt_type_by_ch.get(ch, -1) for ch in channels], dtype=np.int16)


def read_pmt_coordinates(channels, pmt_map_file):
    pmt_table = pd.read_parquet(pmt_map_file, columns=["copy_number", "z", "theta"])
    pmt_table = pmt_table[pmt_table["copy_number"].isin(channels)]
    z_by_ch = dict(zip(pmt_table["copy_number"], pmt_table["z"]))
    theta_by_ch = dict(zip(pmt_table["copy_number"], pmt_table["theta"]))

    z = np.array([z_by_ch.get(ch, np.nan) for ch in channels], dtype=np.float64)
    theta = np.array([theta_by_ch.get(ch, np.nan) for ch in channels], dtype=np.float64)
    return z, theta


def read_pmt_info(channels, pmt_map_file, pmt_file):
    pmt_type = read_pmt_types(channels, pmt_map_file)

    with h5.File(pmt_file, "r") as f:
        tts_values = f["PMT"]["TTS"][:]

    tts = np.array([tts_values[ch] for ch in channels], dtype=np.float64)
    return pmt_type, tts


def attach_pmt_coordinates(fsmp_data, pmt_map_file):
    z, theta = read_pmt_coordinates(fsmp_data["channels"], pmt_map_file)
    fsmp_data = dict(fsmp_data)
    fsmp_data["z"] = z
    fsmp_data["theta"] = theta
    return fsmp_data


def read_fsmp_times_by_channel(
    fsmp_files,
    fsmp_dir,
    rtraw_dir,
    channels,
    run,
    file_limit=None,
):
    chunks = []
    readable_files = 0
    if len(channels) == FULL_CHANNEL_COUNT and np.array_equal(channels, np.arange(FULL_CHANNEL_COUNT)):
        channel_filter = None
    else:
        channel_filter = [("ch", "in", channels.tolist())]

    if file_limit is None:
        file_iterator = tqdm(fsmp_files, desc=f"RUN {run} files")
        readable_progress = None
    else:
        file_iterator = fsmp_files
        readable_progress = tqdm(
            total=min(file_limit, len(fsmp_files)),
            desc=f"RUN {run} readable files",
        )

    for fsmp_file in file_iterator:
        if file_limit is not None and readable_files >= file_limit:
            break

        rtraw_file = rtraw_path_for(fsmp_file, fsmp_dir, rtraw_dir)
        try:
            data = pd.read_parquet(
                fsmp_file,
                columns=["eid", "ch", "offset", "PEt"],
                filters=channel_filter,
            )
        except Exception as error:
            if file_limit is None:
                raise
            print(f"Skip unreadable FSMP file {fsmp_file}: {error}")
            continue

        readable_files += 1
        if readable_progress is not None:
            readable_progress.update(1)
        if data.empty:
            continue

        data["first_pet"] = [
            first_pet_with_offset(pet, offset)
            for pet, offset in zip(data["PEt"], data["offset"])
        ]
        data = data.dropna(subset=["first_pet"])[["eid", "ch", "first_pet"]]
        if data.empty:
            continue

        monitor = read_monitor_times(rtraw_file)
        data = pd.merge(data, monitor, on="eid", how="inner")
        if data.empty:
            continue

        data["time"] = data["first_pet"] - data["t_monitor"]
        chunks.append(data[["ch", "time"]])

    if readable_progress is not None:
        readable_progress.close()

    if not chunks:
        return pd.DataFrame({"ch": [], "time": []})

    return pd.concat(chunks, ignore_index=True)


def read_compound_fields(dataset, fields, start=None, stop=None):
    view = dataset.fields(fields)
    if start is None and stop is None:
        return view[:]
    if stop is None:
        return view[start]
    return view[start:stop]


def make_event_track_key(event_id, track_id):
    return (event_id.astype(np.int64) << 32) | track_id.astype(np.uint32).astype(np.int64)


def unique_track_init_times(track, wanted_keys, chunk_size):
    selected = []
    track_fields = ("event_id", "track_id", "init_t")
    wanted_keys = np.sort(wanted_keys)

    for start in range(0, len(track), chunk_size):
        stop = min(start + chunk_size, len(track))
        data = read_compound_fields(track, track_fields, start, stop)
        key = make_event_track_key(data["event_id"], data["track_id"])
        idx = np.searchsorted(wanted_keys, key)
        mask = idx < len(wanted_keys)
        mask[mask] = wanted_keys[idx[mask]] == key[mask]
        if np.any(mask):
            selected.append(
                np.rec.fromarrays(
                    [key[mask], data["init_t"][mask]],
                    names=("key", "init_t"),
                )
            )

    if not selected:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    data = np.concatenate(selected)
    order = np.argsort(data["key"], kind="mergesort")
    key = data["key"][order]
    init_t = data["init_t"][order]
    _, first = np.unique(key, return_index=True)
    return key[first], init_t[first].astype(np.float64, copy=False)


def sorted_event_ranges(values):
    if len(values) == 0:
        return {}
    if np.any(values[1:] < values[:-1]):
        return None

    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, len(values)]
    return {int(values[start]): (int(start), int(stop)) for start, stop in zip(starts, stops)}


def sorted_event_ranges_dataset(dataset, chunk_size):
    ranges = {}
    current_event = None
    current_start = 0
    previous_event = None

    for start in range(0, len(dataset), chunk_size):
        stop = min(start + chunk_size, len(dataset))
        data = read_compound_fields(dataset, ("event_id",), start, stop)
        values = data["event_id"]
        if len(values) == 0:
            continue
        if previous_event is not None and values[0] < previous_event:
            return None
        if np.any(values[1:] < values[:-1]):
            return None

        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        chunk_starts = np.r_[0, changes]
        chunk_stops = np.r_[changes, len(values)]

        for local_start, local_stop in zip(chunk_starts, chunk_stops):
            event_id = int(values[local_start])
            absolute_start = start + int(local_start)
            absolute_stop = start + int(local_stop)

            if current_event is None:
                current_event = event_id
                current_start = absolute_start
            elif event_id != current_event:
                ranges[current_event] = (current_start, absolute_start)
                current_event = event_id
                current_start = absolute_start

            previous_event = event_id
            if absolute_stop == len(dataset):
                ranges[current_event] = (current_start, absolute_stop)

    if current_event is not None and current_event not in ranges:
        ranges[current_event] = (current_start, len(dataset))
    return ranges


def add_times_to_channel_hist(hist, entries, channels, pmt_id, time, edges):
    idx = np.searchsorted(channels, pmt_id)
    valid_ch = idx < len(channels)
    valid_ch[valid_ch] = channels[idx[valid_ch]] == pmt_id[valid_ch]
    if not np.any(valid_ch):
        return

    selected_time = time[valid_ch]
    selected_idx = idx[valid_ch]
    bin_idx = np.searchsorted(edges, selected_time, side="right") - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < hist.shape[1])
    if not np.any(valid_bin):
        return

    selected_idx = selected_idx[valid_bin]
    bin_idx = bin_idx[valid_bin]
    np.add.at(hist, (selected_idx, bin_idx), 1)
    np.add.at(entries, selected_idx, 1)


def fill_histogram_by_event_ranges(hits, hit_ranges, track, track_ranges, edges, channels):
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    unmatched = 0

    for event_id, (hit_start, hit_stop) in hit_ranges.items():
        if event_id not in track_ranges:
            unmatched += hit_stop - hit_start
            continue

        track_start, track_stop = track_ranges[event_id]
        track_data = read_compound_fields(track, ("track_id", "init_t"), track_start, track_stop)
        if len(track_data) == 0:
            unmatched += hit_stop - hit_start
            continue

        order = np.argsort(track_data["track_id"], kind="mergesort")
        track_id = track_data["track_id"][order]
        init_t = track_data["init_t"][order].astype(np.float64, copy=False)
        unique_track_id, first = np.unique(track_id, return_index=True)
        unique_init_t = init_t[first]

        event_hits = hits[hit_start:hit_stop]
        idx = np.searchsorted(unique_track_id, event_hits["track_id"])
        valid = idx < len(unique_track_id)
        valid[valid] = unique_track_id[idx[valid]] == event_hits["track_id"][valid]
        unmatched += int((~valid).sum())
        if not np.any(valid):
            continue

        time = event_hits["hit_time"][valid].astype(np.float64, copy=False) - unique_init_t[idx[valid]]
        add_times_to_channel_hist(
            hist,
            entries,
            channels,
            event_hits["pmt_id"][valid].astype(np.int64, copy=False),
            time,
            edges,
        )

    return hist, entries, unmatched


def fill_histogram_by_global_keys(hits, track, edges, chunk_size, channels):
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    hit_key = make_event_track_key(hits["event_id"], hits["track_id"])
    wanted_keys = np.unique(hit_key)
    track_key, init_t = unique_track_init_times(track, wanted_keys, chunk_size)
    if len(track_key) == 0:
        return hist, entries, len(hits)

    idx = np.searchsorted(track_key, hit_key)
    valid = idx < len(track_key)
    valid[valid] = track_key[idx[valid]] == hit_key[valid]
    unmatched = int((~valid).sum())
    if np.any(valid):
        time = hits["hit_time"][valid].astype(np.float64, copy=False) - init_t[idx[valid]]
        add_times_to_channel_hist(
            hist,
            entries,
            channels,
            hits["pmt_id"][valid].astype(np.int64, copy=False),
            time,
            edges,
        )

    return hist, entries, unmatched


def read_sim_histograms(sim_files, sim_group, edges, chunk_size, match_mode, channels):
    channels = np.asarray(channels, dtype=np.int64)
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    unmatched = 0
    event_mode_used = 0
    global_mode_used = 0

    for sim_file in tqdm(sim_files, desc="Simulation files"):
        with h5.File(sim_file, "r") as f:
            cd_hit = f[f"{sim_group}/CDHit"]
            track = f[f"{sim_group}/Track"]

            hits = read_compound_fields(cd_hit, ("event_id", "track_id", "pmt_id", "hit_time"))
            if len(hits) == 0:
                continue

            file_hist = None
            if match_mode in ("auto", "event"):
                hit_ranges = sorted_event_ranges(hits["event_id"])
                track_ranges = sorted_event_ranges_dataset(track, chunk_size)
                if hit_ranges is not None and track_ranges is not None:
                    file_hist, file_entries, file_unmatched = fill_histogram_by_event_ranges(
                        hits,
                        hit_ranges,
                        track,
                        track_ranges,
                        edges,
                        channels,
                    )
                    event_mode_used += 1
                elif match_mode == "event":
                    raise RuntimeError(f"{sim_file} is not sorted by event_id")

            if file_hist is None:
                file_hist, file_entries, file_unmatched = fill_histogram_by_global_keys(
                    hits,
                    track,
                    edges,
                    chunk_size,
                    channels,
                )
                global_mode_used += 1

            hist += file_hist
            entries += file_entries
            unmatched += file_unmatched

    bin_width = edges[1] - edges[0]
    norm = entries[:, np.newaxis] * bin_width
    density = np.divide(
        hist.astype(np.float64),
        norm,
        out=np.zeros_like(hist, dtype=np.float64),
        where=norm > 0,
    )

    if unmatched:
        print(f"Simulation hits without matched (event_id, track_id): {unmatched}")
    if event_mode_used or global_mode_used:
        print(f"Simulation match mode files: event={event_mode_used}, global={global_mode_used}")

    return density, entries


def estimate_peak_from_density(centers, density):
    if len(density) == 0 or np.all(density == 0):
        return np.nan
    return centers[np.argmax(density)]


def estimate_peak_from_times(times, bins):
    if len(times) == 0:
        return np.nan
    low, high = np.percentile(times, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return np.nan
    counts, edges = np.histogram(times, bins=bins, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    return estimate_peak_from_density(centers, counts)


def gaussian_kernel(bin_width, sigma):
    if sigma <= 0:
        return np.array([1.0])
    radius = max(int(np.ceil(5 * sigma / bin_width)), 1)
    x = np.arange(-radius, radius + 1) * bin_width
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def density_hist(times, edges):
    values, _ = np.histogram(times, bins=edges)
    bin_width = edges[1] - edges[0]
    norm = values.sum() * bin_width
    if norm == 0:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    return values.astype(np.float64) / norm


def l1_distance(density_a, density_b, bin_width):
    return float(np.sum(np.abs(density_a - density_b)) * bin_width)


def wasserstein_distance(density_a, density_b, bin_width):
    cdf_a = np.cumsum(density_a) * bin_width
    cdf_b = np.cumsum(density_b) * bin_width
    return float(np.sum(np.abs(cdf_a - cdf_b)) * bin_width)


def local_l1_alignment(times, sim_density, edges, coarse_shift, window=10.0):
    bin_width = edges[1] - edges[0]
    if not np.isfinite(coarse_shift):
        coarse_shift = 0.0

    step = max(bin_width / 10.0, 0.01)
    offsets = np.arange(-window, window + step / 2.0, step)
    best_shift = float(coarse_shift)
    best_density = density_hist(times + best_shift, edges)
    best_l1 = l1_distance(best_density, sim_density, bin_width)

    for offset in offsets:
        candidate_shift = float(coarse_shift + offset)
        candidate_density = density_hist(times + candidate_shift, edges)
        candidate_l1 = l1_distance(candidate_density, sim_density, bin_width)
        if candidate_l1 < best_l1:
            best_shift = candidate_shift
            best_density = candidate_density
            best_l1 = candidate_l1

    return best_shift, best_density, best_l1


def convolve_kernel(density, kernel):
    """Convolve a density with a unit-sum kernel, then renormalize.

    Input: density values on a uniform grid and the unit-sum kernel.
    Output: the convolved density, normalized back to unit area.
    """
    conv = np.convolve(density, kernel, mode="same")
    total = conv.sum()
    if total > 0:
        conv = conv / total
    return conv


def convolve_density(density, bin_width, sigma):
    kernel = gaussian_kernel(bin_width, sigma)
    return convolve_kernel(density, kernel)


def times_by_channel(fsmp_ch, fsmp_time):
    order = np.argsort(fsmp_ch, kind="mergesort")
    sorted_ch = fsmp_ch[order]
    sorted_time = fsmp_time[order]
    unique_ch, starts = np.unique(sorted_ch, return_index=True)
    stops = np.r_[starts[1:], len(sorted_ch)]
    return {
        int(ch): sorted_time[start:stop]
        for ch, start, stop in zip(unique_ch, starts, stops)
    }


def build_channel_histograms(fsmp_data, sim_data, alignment="l1"):
    if alignment not in {"l1", "peak"}:
        raise ValueError(f"Unknown alignment mode: {alignment}")

    channels = fsmp_data["channels"].astype(np.int64, copy=False)
    pmt_type = fsmp_data["pmt_type"].astype(np.int16, copy=False)
    tts = fsmp_data["tts"].astype(np.float64, copy=False)
    z = fsmp_data["z"].astype(np.float64, copy=False)
    theta = fsmp_data["theta"].astype(np.float64, copy=False)
    fsmp_lookup = times_by_channel(
        fsmp_data["fsmp_ch"].astype(np.int64, copy=False),
        fsmp_data["fsmp_time"].astype(np.float64, copy=False),
    )

    edges = sim_data["edges"].astype(np.float64, copy=False)
    centers = sim_data["centers"].astype(np.float64, copy=False)
    sim_channels = sim_data["channels"].astype(np.int64, copy=False)
    sim_raw_density_by_channel = sim_data["simulation_raw_density"].astype(np.float64, copy=False)
    sim_entries_by_channel = sim_data["simulation_entries"].astype(np.int64, copy=False)
    bin_width = edges[1] - edges[0]
    bins = len(edges) - 1

    run_density = []
    sim_raw_density_all = []
    sim_density = []
    run_peak = []
    sim_peak = []
    fsmp_counts = []
    sim_counts = []
    l1_distances = []
    wasserstein_distances = []

    for ch, ch_tts in tqdm(list(zip(channels, tts)), desc="Channel histograms"):
        fsmp_time = fsmp_lookup.get(int(ch), np.array([], dtype=np.float64))
        sim_idx = np.searchsorted(sim_channels, ch)
        if sim_idx >= len(sim_channels) or sim_channels[sim_idx] != ch:
            sim_raw_density = np.zeros(len(edges) - 1, dtype=np.float64)
            sim_entries = 0
        else:
            sim_raw_density = sim_raw_density_by_channel[sim_idx]
            sim_entries = int(sim_entries_by_channel[sim_idx])

        sim_conv_density = convolve_density(sim_raw_density, bin_width, ch_tts)
        this_sim_peak = estimate_peak_from_density(centers, sim_conv_density)

        this_run_peak = estimate_peak_from_times(fsmp_time, bins)
        this_shift = this_sim_peak - this_run_peak
        if not np.isfinite(this_shift):
            this_shift = 0.0

        if alignment == "l1":
            this_shift, fsmp_aligned_density, this_l1 = local_l1_alignment(
                fsmp_time,
                sim_conv_density,
                edges,
                this_shift,
            )
        else:
            fsmp_aligned_density = density_hist(fsmp_time + this_shift, edges)
            this_l1 = l1_distance(fsmp_aligned_density, sim_conv_density, bin_width)
        this_wasserstein = wasserstein_distance(fsmp_aligned_density, sim_conv_density, bin_width)

        run_density.append(fsmp_aligned_density)
        sim_raw_density_all.append(sim_raw_density)
        sim_density.append(sim_conv_density)
        run_peak.append(this_run_peak)
        sim_peak.append(this_sim_peak)
        fsmp_counts.append(len(fsmp_time))
        sim_counts.append(sim_entries)
        l1_distances.append(this_l1)
        wasserstein_distances.append(this_wasserstein)

    return {
        "channels": channels.astype(np.int64),
        "pmt_type": pmt_type.astype(np.int16),
        "tts": tts.astype(np.float64),
        "z": z.astype(np.float64),
        "theta": theta.astype(np.float64),
        "edges": edges.astype(np.float64),
        "centers": centers.astype(np.float64),
        "run_density": np.asarray(run_density, dtype=np.float64),
        "simulation_raw_density": np.asarray(sim_raw_density_all, dtype=np.float64),
        "simulation_density": np.asarray(sim_density, dtype=np.float64),
        "run_peak": np.asarray(run_peak, dtype=np.float64),
        "simulation_peak": np.asarray(sim_peak, dtype=np.float64),
        "run_entries": np.asarray(fsmp_counts, dtype=np.int64),
        "simulation_entries": np.asarray(sim_counts, dtype=np.int64),
        "l1_distance": np.asarray(l1_distances, dtype=np.float64),
        "wasserstein_distance": np.asarray(wasserstein_distances, dtype=np.float64),
    }


def compute_fsmp_data(args, channels, fsmp_files):
    pmt_type, tts = read_pmt_info(channels, args.pmt_map, args.pmt)
    fsmp_df = read_fsmp_times_by_channel(
        fsmp_files,
        args.fsmp_dir,
        args.rtraw_dir,
        channels,
        args.run,
        BRIEF_FSMP_FILE_LIMIT if args.brief else None,
    )
    if fsmp_df.empty:
        raise RuntimeError(f"No RUN {args.run} time entries after channel selection and monitor join.")

    fsmp_df = fsmp_df.sort_values("ch", kind="mergesort")
    data = {
        "channels": channels.astype(np.int64),
        "pmt_type": pmt_type.astype(np.int16),
        "tts": tts.astype(np.float64),
        "fsmp_ch": fsmp_df["ch"].to_numpy(dtype=np.int64),
        "fsmp_time": fsmp_df["time"].to_numpy(dtype=np.float64),
    }
    args.fsmp_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fsmp_data, **data)
    print(f"Wrote {args.fsmp_data}")
    return data


def compute_sim_data(args, channels):
    if args.sim_chunk_size <= 0:
        raise ValueError("--sim-chunk-size must be positive")

    sim_files = iter_sim_files(args.sim_dir)
    if args.brief:
        sim_files = sim_files[:BRIEF_SIM_FILE_LIMIT]
    if not sim_files:
        raise FileNotFoundError(f"No simulation h5 files found in {args.sim_dir}")

    edges = np.linspace(args.data_range[0], args.data_range[1], args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sim_raw_density, sim_entries = read_sim_histograms(
        sim_files,
        args.sim_group,
        edges,
        args.sim_chunk_size,
        args.sim_match_mode,
        channels,
    )
    if np.sum(sim_entries) == 0:
        raise RuntimeError("No simulation time entries.")

    data = {
        "channels": channels.astype(np.int64),
        "edges": edges.astype(np.float64),
        "centers": centers.astype(np.float64),
        "simulation_raw_density": sim_raw_density.astype(np.float64),
        "simulation_entries": sim_entries.astype(np.int64),
    }
    args.sim_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.sim_data, **data)
    print(f"Wrote {args.sim_data}")
    return data


def load_data(path):
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def metric_arrays(data):
    edges = data["edges"]
    bin_width = edges[1] - edges[0]
    run_density = data["run_density"]
    sim_density = data["simulation_density"]

    if "l1_distance" in data:
        l1 = data["l1_distance"].astype(np.float64, copy=False)
    else:
        l1 = np.array(
            [l1_distance(run, sim, bin_width) for run, sim in zip(run_density, sim_density)],
            dtype=np.float64,
        )

    if "wasserstein_distance" in data:
        wasserstein = data["wasserstein_distance"].astype(np.float64, copy=False)
    else:
        wasserstein = np.array(
            [wasserstein_distance(run, sim, bin_width) for run, sim in zip(run_density, sim_density)],
            dtype=np.float64,
        )

    return l1, wasserstein


def plot_metric_distribution(pdf, values, title, xlabel):
    finite = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    if len(finite) == 0:
        ax.text(0.5, 0.5, "No finite entries", transform=ax.transAxes, ha="center", va="center")
        mean = np.nan
        std = np.nan
    else:
        ax.hist(finite, bins=METRIC_HISTOGRAM_BINS, histtype="step", color="black")
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.2, label="Mean")
        ax.legend(frameon=False)

    ax.set_title(f"{title}\nmean={mean:.6g}, std={std:.6g}, N={len(finite)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def plot_data(data, output, plot_range, run):
    configure_plot_style()

    channels = data["channels"]
    pmt_type = data["pmt_type"]
    tts = data["tts"]
    z = data["z"]
    theta = data["theta"]
    centers = data["centers"]
    run_density = data["run_density"]
    sim_density = data["simulation_density"]
    l1, wasserstein = metric_arrays(data)

    with PdfPages(output) as pdf:
        for i, ch in enumerate(channels):
            difference = sim_density[i] - run_density[i]
            ratio = np.divide(
                sim_density[i],
                run_density[i],
                out=np.full_like(sim_density[i], np.nan),
                where=run_density[i] > 0,
            )
            pmt_name = PMT_TYPE_NAME.get(int(pmt_type[i]), f"Unknown type {int(pmt_type[i])}")
            channel_title = (
                rf"Channel {int(ch)} ({pmt_name})" "\n"
                rf"TTS={tts[i]:.2f} ns, "
                rf"z={z[i]:.2f} mm, $\theta$={theta[i]:.2f}$^\circ$"
            )

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                centers,
                run_density[i],
                where="mid",
                linewidth=1.4,
                label=f"RUN{run}",
            )
            ax.step(
                centers,
                sim_density[i],
                where="mid",
                linewidth=1.4,
                label="Simulation",
            )
            metric_text = f"L1 = {l1[i]:.6g}\nWasserstein = {wasserstein[i]:.6g} ns"
            ax.text(
                0.98,
                0.98,
                metric_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

            ax.set_title(channel_title)
            ax.set_xlim(plot_range)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-6)
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Normalized density")
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, difference_ax = plt.subplots(figsize=FIGURE_SIZE)
            difference_line = difference_ax.step(
                centers,
                difference,
                where="mid",
                linewidth=1.3,
                color="tab:blue",
                label="Difference",
            )[0]
            difference_ax.axhline(
                0.0,
                color="tab:blue",
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )

            ratio_ax = difference_ax.twinx()
            ratio_line = ratio_ax.step(
                centers,
                ratio,
                where="mid",
                linewidth=1.3,
                color="tab:red",
                label="Ratio",
            )[0]
            ratio_ax.axhline(
                1.0,
                color="tab:red",
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )
            ratio_ax.set_yscale("log")

            difference_ax.set_title(channel_title)
            difference_ax.set_xlim(plot_range)
            difference_ax.set_xlabel("Time [ns]")
            difference_ax.set_ylabel("Simulation - RUN density", color="tab:blue")
            difference_ax.tick_params(axis="y", colors="tab:blue")
            difference_ax.spines["left"].set_color("tab:blue")
            ratio_ax.set_ylabel("Simulation / RUN ratio", color="tab:red")
            ratio_ax.tick_params(axis="y", colors="tab:red")
            ratio_ax.spines["right"].set_color("tab:red")
            difference_ax.legend(
                [difference_line, ratio_line],
                [difference_line.get_label(), ratio_line.get_label()],
                frameon=False,
                loc="upper left",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        groups = [
            ("Hamamatsu PMT", pmt_type == 1),
            ("NNVT PMT", pmt_type == 2),
            ("All PMTs", np.ones(len(pmt_type), dtype=bool)),
        ]
        for title, mask in groups:
            plot_metric_distribution(
                pdf,
                l1[mask],
                f"{title} L1 distance distribution",
                "L1 distance",
            )
        for title, mask in groups:
            plot_metric_distribution(
                pdf,
                wasserstein[mask],
                f"{title} Wasserstein distance distribution",
                "Wasserstein distance [ns]",
            )

    print(f"Wrote {output}")


def normalize_density(density, bin_width):
    area = np.sum(density) * bin_width
    if area <= 0:
        return np.zeros_like(density)
    return density / area


def plot_summary_data(data, output, plot_range, run, source_z):
    configure_plot_style()

    centers = data["centers"]
    bin_width = data["edges"][1] - data["edges"][0]
    pmt_type = data["pmt_type"]
    theta = data["theta"]
    run_density = data["run_density"]
    sim_density = data["simulation_density"]
    run_entries = data["run_entries"]
    sim_entries = data["simulation_entries"]

    pmt_groups = [
        ("Hamamatsu PMT average", 1),
        ("NNVT PMT average", 2),
    ]
    if np.isclose(source_z, 0.0, rtol=0.0, atol=1e-12):
        theta_groups = [("", np.ones(len(theta), dtype=bool))]
    else:
        finite_theta = np.unique(theta[np.isfinite(theta)])
        theta_groups = [
            (
                rf"$\theta$={this_theta:.2f}$^\circ$",
                np.isclose(theta, this_theta, rtol=0.0, atol=1e-9),
            )
            for this_theta in finite_theta
        ]
        if not theta_groups:
            raise ValueError("No finite PMT theta values available for the summary PDF")
    summary_groups = [
        (theta_title, theta_mask, pmt_title, type_id)
        for theta_title, theta_mask in theta_groups
        for pmt_title, type_id in pmt_groups
    ]

    with PdfPages(output) as pdf:
        for theta_title, theta_mask, pmt_title, type_id in summary_groups:
            title = f"{theta_title}, {pmt_title}" if theta_title else pmt_title
            type_mask = theta_mask & (pmt_type == type_id)
            run_mask = type_mask & (run_entries > 0)
            sim_mask = type_mask & (sim_entries > 0)

            if np.any(run_mask):
                mean_run = normalize_density(np.mean(run_density[run_mask], axis=0), bin_width)
            else:
                mean_run = np.zeros(len(centers), dtype=np.float64)
            if np.any(sim_mask):
                mean_sim = normalize_density(np.mean(sim_density[sim_mask], axis=0), bin_width)
            else:
                mean_sim = np.zeros(len(centers), dtype=np.float64)

            difference = mean_sim - mean_run
            ratio = np.divide(
                mean_sim,
                mean_run,
                out=np.full_like(mean_sim, np.nan),
                where=mean_run > 0,
            )
            summary_l1 = l1_distance(mean_run, mean_sim, bin_width)
            summary_wasserstein = wasserstein_distance(mean_run, mean_sim, bin_width)
            summary_metrics = (
                f"L1 = {summary_l1:.6g}\n"
                f"Wasserstein = {summary_wasserstein:.6g} ns"
            )

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(centers, mean_run, where="mid", linewidth=1.4, label=f"RUN{run}")
            ax.step(centers, mean_sim, where="mid", linewidth=1.4, label="Simulation")
            ax.text(
                0.98,
                0.98,
                summary_metrics,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_title(
                f"{title}\n"
                f"RUN channels={int(run_mask.sum())}, simulation channels={int(sim_mask.sum())}"
            )
            ax.set_xlim(plot_range)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-6)
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Normalized density")
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, difference_ax = plt.subplots(figsize=FIGURE_SIZE)
            difference_line = difference_ax.step(
                centers,
                difference,
                where="mid",
                linewidth=1.3,
                color="tab:blue",
                label="Difference",
            )[0]
            difference_ax.axhline(
                0.0,
                color="tab:blue",
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )

            ratio_ax = difference_ax.twinx()
            ratio_line = ratio_ax.step(
                centers,
                ratio,
                where="mid",
                linewidth=1.3,
                color="tab:red",
                label="Ratio",
            )[0]
            ratio_ax.axhline(
                1.0,
                color="tab:red",
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )
            ratio_ax.set_yscale("log")

            difference_ax.set_title(
                f"{title}\n"
                f"RUN channels={int(run_mask.sum())}, simulation channels={int(sim_mask.sum())}"
            )
            difference_ax.set_xlim(plot_range)
            difference_ax.set_xlabel("Time [ns]")
            difference_ax.set_ylabel("Simulation - RUN density", color="tab:blue")
            difference_ax.tick_params(axis="y", colors="tab:blue")
            difference_ax.spines["left"].set_color("tab:blue")
            ratio_ax.set_ylabel("Simulation / RUN ratio", color="tab:red")
            ratio_ax.tick_params(axis="y", colors="tab:red")
            ratio_ax.spines["right"].set_color("tab:red")
            difference_ax.legend(
                [difference_line, ratio_line],
                [difference_line.get_label(), ratio_line.get_label()],
                frameon=False,
                loc="upper left",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {output}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"z={args.z:g} m -> RUN {args.run}; "
        f"FSMP={args.fsmp_dir}; simulation={args.sim_dir}; output={args.output_dir}"
    )
    if args.brief:
        print(
            f"Brief mode: first {BRIEF_FSMP_FILE_LIMIT} FSMP files and "
            f"first {BRIEF_SIM_FILE_LIMIT} simulation files"
        )

    if args.plot_only:
        if not args.fsmp_data.exists():
            raise FileNotFoundError(f"{args.fsmp_data} does not exist for --plot-only")
        if not args.sim_data.exists():
            raise FileNotFoundError(f"{args.sim_data} does not exist for --plot-only")
        fsmp_data = load_data(args.fsmp_data)
        sim_data = load_data(args.sim_data)
    else:
        channels, fsmp_files = selected_channels(args)
        if args.fsmp_data.exists() and not args.force:
            cached_fsmp_data = load_data(args.fsmp_data)
            if np.array_equal(cached_fsmp_data.get("channels"), channels):
                fsmp_data = cached_fsmp_data
                print(f"Load cached FSMP data from {args.fsmp_data}; use --force to recompute.")
            else:
                print(f"Cached channels in {args.fsmp_data} do not match; recomputing.")
                fsmp_data = compute_fsmp_data(args, channels, fsmp_files)
        else:
            fsmp_data = compute_fsmp_data(args, channels, fsmp_files)

        if args.sim_data.exists() and not args.force:
            cached_sim_data = load_data(args.sim_data)
            if np.array_equal(cached_sim_data.get("channels"), channels):
                sim_data = cached_sim_data
                print(f"Load cached simulation data from {args.sim_data}; use --force to recompute.")
            else:
                print(f"Cached channels in {args.sim_data} do not match; recomputing.")
                sim_data = compute_sim_data(args, channels)
        else:
            sim_data = compute_sim_data(args, channels)

    channel_pdf_needed = not args.opt.exists()
    summary_pdf_needed = not args.summary_opt.exists()
    if not channel_pdf_needed:
        print(f"Skip existing channel PDF {args.opt}")
    if not summary_pdf_needed:
        print(f"Skip existing summary PDF {args.summary_opt}")
    if not channel_pdf_needed and not summary_pdf_needed:
        return

    data = None
    if args.aligned_data.exists() and not args.force:
        cached_aligned_data = load_data(args.aligned_data)
        compatible = (
            np.array_equal(cached_aligned_data.get("channels"), fsmp_data.get("channels"))
            and np.array_equal(cached_aligned_data.get("edges"), sim_data.get("edges"))
        )
        cached_mode = str(cached_aligned_data.get("alignment_mode", ""))
        if compatible and (not channel_pdf_needed or cached_mode == "l1"):
            data = cached_aligned_data
            print(f"Load cached aligned data from {args.aligned_data}.")
        else:
            print(f"Cached aligned data in {args.aligned_data} is incompatible; recomputing.")

    if data is None:
        fsmp_data = attach_pmt_coordinates(fsmp_data, args.pmt_map)
        alignment = "l1" if channel_pdf_needed else "peak"
        if alignment == "peak":
            print("Only summary PDF is missing; use direct peak alignment.")
        data = build_channel_histograms(fsmp_data, sim_data, alignment=alignment)
        data["alignment_mode"] = np.asarray(alignment)
        np.savez_compressed(args.aligned_data, **data)
        print(f"Wrote {args.aligned_data}")

    if channel_pdf_needed:
        plot_data(data, args.opt, args.plot_range, args.run)
    if summary_pdf_needed:
        plot_summary_data(data, args.summary_opt, args.plot_range, args.run, args.z)


if __name__ == "__main__":
    main()
