#!/usr/bin/env python
"""compare_sim_and_real —— LUCiD/JUNO 官方模拟 vs FSMP 真实数据对比

功能：
  - FSMP 加载/预处理：逐 pq 读 (eid, ch, offset, PEt) → 每行取
    首达 first_pet = min(PEt)+offset → 与 monitor PMT (split_rtraw, pmt 43303)
    逐事例对齐：time = first_pet − t_monitor。这里不施加 TTS/RIF；
    默认 `--fsmp-time-mode all` 会把每个匹配到 monitor 的 PEt 都放入分布；
    `first` 模式才只取每行最早的 PEt。FSMP 另行记录事件数和 PE 总数。
    暗噪声在 FSMP 自身的原始时间坐标中估计：窗口为
    [该通道最早 FSMP 时间, FSMP 主峰 - dark_window]，用窗口内 PE 数除以
    窗口宽度得到计数率，再换算为归一化密度平底并从整个 FSMP 密度扣除。
  - 画图/指标：每通道密度 (step) + 差分/比值双轴 + L1/Wasserstein
    指标分布 + 按 PMT 类型/θ 分组的 summary PDF；只在模拟侧逐通道
    使用 PMT 表中的 TTS 高斯卷积，并可叠加全局或按 PMT 类型配置的
    RIF 高斯占位；支持 l1/peak 两种对齐。

LUCiD 模拟加载两种方式（--sim-dir/--lucid-sim-dir 与 --sim-npz/--lucid-sim-npz 二选一）：
  --sim-dir <目录>：流式逐文件累计每通道直方图（内存 = 单文件 + 直方图，可跑 10000 事件）；
  三点源数据存储于/junofs/C/liukainan/LUCiD/outputs

官方 HDF5 模拟通过 --junosw-sim-dir 指定；其文件内部默认读取
    `Sim/Track` 和 `Sim/CDHit`，数据取自/mnt/stage/haoch/junores/laser。

用法（先在可写目录运行，或用 --output-root 指定绝对路径）：
  # 中心 (z=0, run 15040):LUCiD + JUNOsw + FSMP
  python compare/compare_sim_vs_real.py -z 0 \
      --sim-dir outputs/lucid_center \
      --junosw-sim-dir /path/to/laser/532/0 \
      --output-root /mnt/stage/liukainan/compare_sim_out -n 1000
  # +15m (run 15030):
  python compare/compare_sim_vs_real.py -z 15 \
      --sim-dir outputs/juno_532nm_laser_1500cm_300w_10000event_v41 \
      --junosw-sim-dir /path/to/laser/532/1500 \
      --output-root /mnt/stage/liukainan/compare_sim_out -n 1000
  # -15m (run 15050):
  python compare/compare_sim_vs_real.py -z -15 \
      --sim-dir outputs/juno_532nm_laser_-1500cm_300w_10000event_v41 \
      --junosw-sim-dir /path/to/laser/532/-1500 \
      --output-root /mnt/stage/liukainan/compare_sim_out -n 1000

依赖: numpy/pandas/pyarrow/h5py/matplotlib/tqdm。
"""
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
DEFAULT_RUN_TABLE = "/mnt/stage/haoch/laser/run_532.csv"
DEFAULT_RTRAW_DIR = "/mnt/stage/haoch/laser/split_rtraw"
DEFAULT_FSMP_ROOT = "/junofs/C/PE"
DEFAULT_PMT_MAP = "/junofs/C/junoid_J25.7.3.pq"
DEFAULT_PMT_FILE = "/junofs/C/PMT/STD/9541.h5"
DEFAULT_LUCID_LABEL = "LUCiD"
DEFAULT_JUNOSW_LABEL = "JUNO official"
DEFAULT_RIF_SIGMA_HAMAMATSU = 1.3
DEFAULT_RIF_SIGMA_NNVT = 7.0

# Keep the source identity stable across both the density and diagnostic pages.
# The line style on an even page then identifies which y-axis quantity is shown.
FSMP_COLOR = "tab:blue"
SOURCE_COLORS = {
    "lucid": "tab:orange",
    "junosw": "tab:green",
}
FALLBACK_SOURCE_COLORS = ("tab:red", "tab:purple", "tab:brown", "tab:pink")
DIFFERENCE_AXIS_COLOR = "0.25"
RATIO_AXIS_COLOR = "#7f3c8d"


def resolve_rif_sigmas(global_sigma=0.0, hamamatsu_sigma=None, nnvt_sigma=None):
    """Resolve the effective Gaussian RIF sigma for the two PMT types.

    ``--rif-sigma`` remains the legacy global fallback.  A type-specific
    argument overrides it for that PMT type, so old commands keep working.
    """
    values = {
        "global": global_sigma,
        "hamamatsu": global_sigma if hamamatsu_sigma is None else hamamatsu_sigma,
        "nnvt": global_sigma if nnvt_sigma is None else nnvt_sigma,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"RIF sigma for {name} must be finite and >= 0 ns")
    return float(values["hamamatsu"]), float(values["nnvt"])


def rif_sigma_for_pmt_type(pmt_type, hamamatsu_sigma, nnvt_sigma):
    """Return the resolved RIF sigma for one PMT type."""
    return {
        1: float(hamamatsu_sigma),
        2: float(nnvt_sigma),
    }.get(int(pmt_type), 0.0)


def rif_config_string(hamamatsu_sigma, nnvt_sigma):
    """Serialize the effective type-dependent RIF configuration for caches."""
    if np.isclose(hamamatsu_sigma, nnvt_sigma, rtol=0.0, atol=1e-12):
        return f"{hamamatsu_sigma:g}"
    return f"hamamatsu:{hamamatsu_sigma:g}|nnvt:{nnvt_sigma:g}"


def processing_string(
    use_tts,
    hamamatsu_sigma,
    nnvt_sigma,
    dark_subtract,
    fsmp_time_mode,
):
    """Serialize all response-processing settings into a cache key."""
    return (
        f"tts={int(use_tts)},"
        f"rif={rif_config_string(hamamatsu_sigma, nnvt_sigma)},"
        f"dark={int(dark_subtract)},"
        f"fsmp={fsmp_time_mode}"
    )


def count_pet_values(pet):
    """Return the number of PE timestamps stored in one FSMP PEt cell."""
    if pet is None:
        return 0
    try:
        return int(np.asarray(pet).size)
    except (TypeError, ValueError):
        return 1


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
        default=Path(DEFAULT_RUN_TABLE),
        help=f"CSV mapping run numbers to z coordinates (default {DEFAULT_RUN_TABLE})",
    )
    # ---- LUCiD 模拟输入（二选一；保留 --sim-dir/--sim-npz 旧接口） ----
    psr.add_argument(
        "--sim-dir",
        "--lucid-sim-dir",
        dest="lucid_sim_dir",
        type=Path,
        default=None,
        help="LUCiD per-event npz directory (hit_pmt_ids + pe_hit_times)",
    )
    psr.add_argument(
        "--sim-npz",
        "--lucid-sim-npz",
        dest="lucid_sim_npz",
        type=Path,
        default=None,
        help="single merged npz (hit_pmt_ids + pe_hit_times); read via memmap in chunks",
    )
    psr.add_argument(
        "--merge-sim-to",
        dest="merge_sim_to",
        type=Path,
        default=None,
        help="merge all per-event npz under --sim-dir into one npz and exit "
             "(raw PE; 10000 events ≈ 140 GB, use with care)",
    )
    psr.add_argument(
        "--sim-chunk-size",
        dest="sim_chunk_size",
        type=int,
        default=10_000_000,
        help="rows per chunk while scanning the (merged) simulation array",
    )
    psr.add_argument(
        "--sim-file-glob",
        dest="sim_file_glob",
        type=str,
        default="*_ev*.npz",
        help="basename glob for per-event npz under --sim-dir (default '*_ev*.npz' to "
             "exclude cache/non-event npz such as comparison_cache/*)",
    )
    psr.add_argument(
        "--lucid-label",
        default=DEFAULT_LUCID_LABEL,
        help=f"label used for the LUCiD curve (default: {DEFAULT_LUCID_LABEL})",
    )
    # ----  JUNOsw HDF5 模拟（可与 LUCiD 同时提供） ----
    psr.add_argument(
        "--junosw-sim-dir",
        type=Path,
        default=None,
        help="JUNO official simulation HDF5 file or directory containing *.h5",
    )
    psr.add_argument(
        "--junosw-sim-group",
        default="Sim",
        help="HDF5 group containing Track and CDHit (default: Sim)",
    )
    psr.add_argument(
        "--junosw-sim-match-mode",
        choices=("auto", "event", "global"),
        default="auto",
        help="JUNO HDF5 Track/CDHit matching mode (default: auto)",
    )
    psr.add_argument(
        "--junosw-sim-file-glob",
        default="*.h5",
        help="file glob for JUNO HDF5 simulation files (default: *.h5)",
    )
    psr.add_argument(
        "--junosw-label",
        default=DEFAULT_JUNOSW_LABEL,
        help=f"label used for the JUNO official curve (default: {DEFAULT_JUNOSW_LABEL})",
    )
    # ---- FSMP 输入 ----
    psr.add_argument(
        "--fsmp-dir",
        dest="fsmp_dir",
        type=Path,
        default=None,
        help=f"FSMP pq root containing */*.pq; default {DEFAULT_FSMP_ROOT}/<run>",
    )
    psr.add_argument(
        "--rtraw-dir",
        dest="rtraw_dir",
        type=Path,
        default=Path(DEFAULT_RTRAW_DIR),
        help=f"split_rtraw h5 root for monitor alignment (default {DEFAULT_RTRAW_DIR})",
    )
    psr.add_argument(
        "--pmt-map",
        dest="pmt_map",
        type=Path,
        default=Path(DEFAULT_PMT_MAP),
        help="PMT type table: copy_number -> pmt_type / z / theta",
    )
    psr.add_argument(
        "--pmt",
        dest="pmt",
        type=Path,
        default=Path(DEFAULT_PMT_FILE),
        help=(
            "PMT calibration h5 with PMT/TTS; the stored TTS value is passed "
            "directly as the Gaussian convolution sigma in ns"
        ),
    )
    # ---- 通道选择（随机 n 或北 0-144） ----
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
    # ---- 直方图/范围 ----
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
    # ---- 模拟侧处理开关 ----
    psr.add_argument(
        "--no-tts",
        dest="no_tts",
        action="store_true",
        help="skip the per-PMT TTS Gaussian convolution on the simulation density "
             "(TTS 作用于模拟侧, 使其带上真实 PMT 时间展宽; 关闭则用纯到达时间)",
    )
    psr.add_argument(
        "--rif-sigma",
        dest="rif_sigma",
        type=float,
        default=0.0,
        help="temporary Gaussian RIF (response impulse function) stand-in sigma in ns, "
             "used for both PMT types unless type-specific values are supplied "
             "(0 = off; legacy interface)",
    )
    psr.add_argument(
        "--rif-sigma-hamamatsu",
        type=float,
        default=None,
        help=(
            "Gaussian RIF stand-in sigma [ns] for Hamamatsu/HPK PMTs; "
            "defaults to --rif-sigma. The literature-based test value is "
            f"{DEFAULT_RIF_SIGMA_HAMAMATSU:g} ns, but this is a TTS reference "
            "value rather than a measured independent electronics IRF."
        ),
    )
    psr.add_argument(
        "--rif-sigma-nnvt",
        type=float,
        default=None,
        help=(
            "Gaussian RIF stand-in sigma [ns] for NNVT PMTs; defaults to "
            f"--rif-sigma. The literature-based test value is "
            f"{DEFAULT_RIF_SIGMA_NNVT:g} ns, but NNVT timing is not strictly "
            "Gaussian."
        ),
    )
    # ---- FSMP 侧预处理开关 ----
    psr.add_argument(
        "--dark-subtract",
        dest="dark_subtract",
        action="store_true",
        help="subtract the per-channel pre-peak dark-noise floor from the FSMP density",
    )
    psr.add_argument(
        "--dark-window",
        dest="dark_window",
        type=float,
        default=30.0,
        help="pre-peak margin [ns] used to estimate the dark floor (laser light not yet arrived)",
    )
    psr.add_argument(
        "--fsmp-time-mode",
        choices=("first", "all"),
        default="all",
        help=(
            "FSMP time samples to plot: first=min(PEt)+offset per waveform row, "
            "or all=every PEt+offset; default all"
        ),
    )
    psr.add_argument(
        "--print-fsmp-row",
        action="store_true",
        help="print the first non-empty raw FSMP row while reading each run",
    )
    # ---- 缓存/输出 ----
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
        "--output-root",
        dest="output_root",
        type=Path,
        default=Path("compare_sim"),
        help="directory holding per-run outputs (default ./compare_sim)",
    )
    psr.add_argument(
        "--output-tag",
        default="",
        help=(
            "optional tag appended to PDF/aligned-cache names, allowing "
            "multiple response-processing jobs in one output root"
        ),
    )
    args = psr.parse_args()
    if args.long:
        args.data_range = (args.data_range[0], 1000.0)
        args.plot_range = (args.plot_range[0], 1000.0)
    if args.merge_sim_to is not None and args.lucid_sim_dir is None:
        psr.error("--merge-sim-to requires --sim-dir/--lucid-sim-dir")
    if args.lucid_sim_dir is not None and args.lucid_sim_npz is not None:
        psr.error("specify at most one of --sim-dir/--lucid-sim-dir and --sim-npz/--lucid-sim-npz")
    if (
        args.merge_sim_to is None
        and args.lucid_sim_dir is None
        and args.lucid_sim_npz is None
        and args.junosw_sim_dir is None
    ):
        psr.error(
            "provide at least one simulation input: LUCiD (--sim-dir/--sim-npz) "
            "or JUNO official (--junosw-sim-dir)"
        )
    try:
        args.rif_sigma_hamamatsu, args.rif_sigma_nnvt = resolve_rif_sigmas(
            args.rif_sigma,
            args.rif_sigma_hamamatsu,
            args.rif_sigma_nnvt,
        )
    except ValueError as exc:
        psr.error(str(exc))
    args.run = resolve_run(args.z, args.run_table)
    if args.fsmp_dir is None:
        args.fsmp_dir = Path(DEFAULT_FSMP_ROOT) / str(args.run)
    args.rtraw_dir = Path(args.rtraw_dir) / str(args.run)
    args.output_dir = Path(args.output_root) / str(args.run)
    fsmp_suffix = "north" if args.north else str(args.number)
    if args.long:
        fsmp_suffix += "_long"
    args.opt = args.output_dir / f"compare_sim_{fsmp_suffix}.pdf"
    args.summary_opt = args.output_dir / f"compare_sim_{fsmp_suffix}_summary.pdf"
    args.fsmp_data = (
        args.output_dir / f"fsmp_data_{fsmp_suffix}_{args.fsmp_time_mode}.npz"
    )
    # Keep sim_data as the legacy LUCiD cache name. The official simulation
    # uses a separate cache so both inputs can coexist.
    args.sim_data = args.output_dir / f"sim_data_{fsmp_suffix}.npz"
    args.junosw_sim_data = args.output_dir / f"junosw_sim_data_{fsmp_suffix}.npz"
    output_tag = args.output_tag.strip().replace(" ", "_") if args.output_tag else ""
    output_suffix = f"{fsmp_suffix}_{output_tag}" if output_tag else fsmp_suffix
    args.aligned_data = args.output_dir / f"aligned_data_{output_suffix}.npz"
    args.opt = args.output_dir / f"compare_sim_{output_suffix}.pdf"
    args.summary_opt = args.output_dir / f"compare_sim_{output_suffix}_summary.pdf"
    args.stats_csv = args.output_dir / f"comparison_stats_{output_suffix}.csv"
    args.channel_manifest = (
        args.output_dir
        / f"channel_selection_{output_suffix}_{args.fsmp_time_mode}.csv"
    )
    return args


def iter_fsmp_files(fsmp_dir):
    yield from sorted(fsmp_dir.glob("*/*.pq"))


def iter_sim_files(sim_dir, file_glob="*_ev*.npz"):
    if sim_dir.is_file():
        return [sim_dir]
    files = sorted(sim_dir.rglob(file_glob))
    return [f for f in files if "_traj_" not in f.name]


def iter_junosw_sim_files(sim_dir, file_glob="*.h5"):
    """Return official JUNO HDF5 simulation files from a file or directory."""
    if sim_dir.is_file():
        return [sim_dir]
    return sorted(sim_dir.rglob(file_glob))


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

    # The comparison code treats the stored value as Gaussian sigma.  It does
    # not convert from FWHM; verify the calibration-file convention upstream.
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
    time_mode="all",
    print_first_row=False,
):
    if time_mode not in {"first", "all"}:
        raise ValueError(f"Unknown FSMP time mode: {time_mode}")
    chunks = []
    readable_files = 0
    printed_row = False
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
                columns=["eid", "ch", "offset", "step", "PEt", "e", "q", "nu_lc"],
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

        if print_first_row and not printed_row:
            row = data.iloc[0]
            pet = row["PEt"]
            print(
                f"FSMP sample row from {fsmp_file}:\n"
                f"  eid={row['eid']!r}\n"
                f"  ch={row['ch']!r}\n"
                f"  offset={row['offset']!r}\n"
                f"  step={row['step']!r}\n"
                f"  PEt={pet!r}\n"
                f"  PEt_count={count_pet_values(pet)}\n"
                f"  e={row['e']!r}\n"
                f"  q={row['q']!r}\n"
                f"  nu_lc={row['nu_lc']!r}",
                flush=True,
            )
            printed_row = True

        data["pe_count"] = [count_pet_values(pet) for pet in data["PEt"]]
        data["first_pet"] = [
            first_pet_with_offset(pet, offset)
            for pet, offset in zip(data["PEt"], data["offset"])
        ]
        data = data.dropna(subset=["first_pet"])
        if time_mode == "first":
            data = data[["eid", "ch", "first_pet", "pe_count"]]
        else:
            data = data[["eid", "ch", "offset", "PEt", "pe_count"]]
        if data.empty:
            continue

        monitor = read_monitor_times(rtraw_file)
        data = pd.merge(data, monitor, on="eid", how="inner")
        if data.empty:
            continue

        if time_mode == "first":
            data["time"] = data["first_pet"] - data["t_monitor"]
            chunks.append(data[["eid", "ch", "time", "pe_count"]])
        else:
            expanded = data[["eid", "ch", "offset", "PEt", "t_monitor"]].copy()
            expanded["PEt"] = expanded["PEt"].map(
                lambda value: np.asarray(value, dtype=np.float64).reshape(-1).tolist()
            )
            expanded = expanded.explode("PEt", ignore_index=True)
            expanded["PEt"] = pd.to_numeric(expanded["PEt"], errors="coerce")
            expanded = expanded.dropna(subset=["PEt"])
            if expanded.empty:
                continue
            expanded["time"] = (
                expanded["PEt"].astype(np.float64)
                + expanded["offset"].astype(np.float64)
                - expanded["t_monitor"].astype(np.float64)
            )
            expanded["pe_count"] = 1
            chunks.append(expanded[["eid", "ch", "time", "pe_count"]])

    if readable_progress is not None:
        readable_progress.close()

    if not chunks:
        return pd.DataFrame({"eid": [], "ch": [], "time": [], "pe_count": []})

    return pd.concat(chunks, ignore_index=True)


# =====================================================================
# LUCiD 模拟读取
# =====================================================================
def _accumulate_times_into_hist(
    pmt, t, channels, edges, hist, entries, total_entries=None
):
    """把一组的 (pmt_id, time) 累计进 (n_channels, n_bins) 直方图 (channels 须升序)."""
    idx = np.searchsorted(channels, pmt)
    valid = idx < len(channels)
    valid[valid] = channels[idx[valid]] == pmt[valid]
    if not np.any(valid):
        return
    sel_t = t[valid]
    sel_idx = idx[valid]
    if total_entries is not None:
        np.add.at(total_entries, sel_idx, 1)
    bin_idx = np.searchsorted(edges, sel_t, side="right") - 1
    ok = (bin_idx >= 0) & (bin_idx < hist.shape[1])
    if not np.any(ok):
        return
    sel_idx = sel_idx[ok]
    bin_idx = bin_idx[ok]
    np.add.at(hist, (sel_idx, bin_idx), 1)
    np.add.at(entries, sel_idx, 1)


def read_lucid_sim_histograms(sim_files, edges, chunk_size, channels):
    """流式读取 LUCiD per-event npz: 每文件 (hit_pmt_ids, pe_hit_times) -> 每通道直方图.

    LUCiD 时间已相对激光出射 (t0=0), 无需 track/init_t 匹配；逐文件累计,
    内存 = 单文件数组 + 直方图（10000 事件无压力）。输出结构与
    read_sim_histograms 一致: (density, entries)。
    容错: 缺少 hit_pmt_ids/pe_hit_times 或损坏的 npz 会跳过并计数 (不中断)."""
    channels = np.asarray(channels, dtype=np.int64)
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_entries = np.zeros(len(channels), dtype=np.int64)
    events = np.zeros(len(channels), dtype=np.int64)
    skipped = 0

    for sim_file in tqdm(sim_files, desc="LUCiD simulation files"):
        try:
            data = np.load(sim_file)
            pmt = data["hit_pmt_ids"].astype(np.int64, copy=False)
            t = data["pe_hit_times"].astype(np.float64, copy=False)
        except (KeyError, OSError, ValueError, TypeError) as error:
            skipped += 1
            print(f"Skip non-event/broken simulation file {sim_file}: {error}")
            continue
        if len(t):
            _accumulate_times_into_hist(
                pmt,
                t,
                channels,
                edges,
                hist,
                entries,
                total_entries,
            )
            selected_pmt = pmt[np.isin(pmt, channels)]
            if selected_pmt.size:
                selected_idx = np.searchsorted(
                    channels, np.unique(selected_pmt)
                )
                events[selected_idx] += 1

    if skipped:
        print(f"Skipped {skipped} non-event/broken simulation file(s)")
    bin_width = edges[1] - edges[0]
    norm = entries[:, np.newaxis] * bin_width
    density = np.divide(
        hist.astype(np.float64),
        norm,
        out=np.zeros_like(hist, dtype=np.float64),
        where=norm > 0,
    )
    return density, entries, total_entries, events


def read_lucid_sim_histograms_merged(npz_path, edges, chunk_size, channels):
    """读取合并后的单 npz (memmap + 分块), 避免大文件整读内存外泄."""
    channels = np.asarray(channels, dtype=np.int64)
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_entries = np.zeros(len(channels), dtype=np.int64)
    with np.load(npz_path, mmap_mode="r") as data:
        n_total = len(data["pe_hit_times"])
        for start in tqdm(range(0, n_total, chunk_size), desc="Merged npz chunks"):
            stop = min(start + chunk_size, n_total)
            pmt = data["hit_pmt_ids"][start:stop].astype(np.int64, copy=False)
            t = data["pe_hit_times"][start:stop].astype(np.float64, copy=False)
            _accumulate_times_into_hist(
                pmt,
                t,
                channels,
                edges,
                hist,
                entries,
                total_entries,
            )
    bin_width = edges[1] - edges[0]
    norm = entries[:, np.newaxis] * bin_width
    density = np.divide(
        hist.astype(np.float64),
        norm,
        out=np.zeros_like(hist, dtype=np.float64),
        where=norm > 0,
    )
    # A merged PE array has no event boundary metadata.  Use -1 to make the
    # missing event count explicit in titles and summaries.
    events = np.full(len(channels), -1, dtype=np.int64)
    return density, entries, total_entries, events


def merge_lucid_sim(sim_files, out_path):
    """把所有 per-event npz 的 (hit_pmt_ids, pe_hit_times) 拼接成单个 npz.

    注意: 10000 事件原始 PE ≈ 140 GB（float64 时间 + int32 PMT）;
    若非必要建议直接用 --sim-dir 流式, 缓存 sim_data 才是小体积中间量."""
    parts_id, parts_t = [], []
    total = 0
    for f in tqdm(sim_files, desc="Merging LUCiD npz"):
        d = np.load(f)
        parts_id.append(d["hit_pmt_ids"])
        parts_t.append(d["pe_hit_times"])
        total += len(d["pe_hit_times"])
    ids = np.concatenate(parts_id).astype(np.int32)
    ts = np.concatenate(parts_t).astype(np.float64)
    print(f"Merged {len(parts_id)} files, {total:,} PE -> {out_path}")
    np.savez(out_path, hit_pmt_ids=ids, pe_hit_times=ts)


def read_compound_fields(dataset, fields, start=None, stop=None):
    """Read selected fields from an HDF5 compound dataset, optionally by slice."""
    view = dataset.fields(fields)
    if start is None and stop is None:
        return view[:]
    if stop is None:
        return view[start]
    return view[start:stop]


def make_event_track_key(event_id, track_id):
    """Encode an (event_id, track_id) pair into one sortable integer key."""
    return (event_id.astype(np.int64) << 32) | track_id.astype(np.uint32).astype(np.int64)


def unique_track_init_times(track, wanted_keys, chunk_size):
    """Read init_t only for Track rows referenced by CDHit keys."""
    selected = []
    wanted_keys = np.sort(wanted_keys)
    for start in range(0, len(track), chunk_size):
        stop = min(start + chunk_size, len(track))
        data = read_compound_fields(track, ("event_id", "track_id", "init_t"), start, stop)
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
    """Return event -> contiguous row range, or None if values are not sorted."""
    if len(values) == 0:
        return {}
    if np.any(values[1:] < values[:-1]):
        return None

    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, len(values)]
    return {
        int(values[start]): (int(start), int(stop))
        for start, stop in zip(starts, stops)
    }


def sorted_event_ranges_dataset(dataset, chunk_size):
    """Build event ranges without loading the full Track dataset."""
    ranges = {}
    current_event = None
    current_start = 0
    previous_event = None

    for start in range(0, len(dataset), chunk_size):
        stop = min(start + chunk_size, len(dataset))
        values = read_compound_fields(dataset, ("event_id",), start, stop)["event_id"]
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


def fill_histogram_by_event_ranges(hits, hit_ranges, track, track_ranges, edges, channels):
    """Match CDHit and Track by event ranges, then fill relative hit-time histograms."""
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_entries = np.zeros(len(channels), dtype=np.int64)
    event_sets = [set() for _ in channels]
    unmatched = 0

    for event_id, (hit_start, hit_stop) in hit_ranges.items():
        if event_id not in track_ranges:
            unmatched += hit_stop - hit_start
            continue
        track_start, track_stop = track_ranges[event_id]
        track_data = read_compound_fields(
            track, ("track_id", "init_t"), track_start, track_stop
        )
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

        selected_pmt = event_hits["pmt_id"][valid].astype(np.int64, copy=False)
        selected_idx = np.searchsorted(channels, selected_pmt)
        channel_valid = selected_idx < len(channels)
        channel_valid[channel_valid] = (
            channels[selected_idx[channel_valid]] == selected_pmt[channel_valid]
        )
        if np.any(channel_valid):
            selected_idx = selected_idx[channel_valid]
            np.add.at(total_entries, selected_idx, 1)
            for channel_index in np.unique(selected_idx):
                event_sets[int(channel_index)].add(int(event_id))

        time = (
            event_hits["hit_time"][valid].astype(np.float64, copy=False)
            - unique_init_t[idx[valid]]
        )
        _accumulate_times_into_hist(
            event_hits["pmt_id"][valid].astype(np.int64, copy=False),
            time,
            channels,
            edges,
            hist,
            entries,
        )

    events = np.asarray([len(values) for values in event_sets], dtype=np.int64)
    return hist, entries, total_entries, events, unmatched


def fill_histogram_by_global_keys(hits, track, edges, chunk_size, channels):
    """Match CDHit and Track by globally encoded (event_id, track_id) keys."""
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_entries = np.zeros(len(channels), dtype=np.int64)
    hit_key = make_event_track_key(hits["event_id"], hits["track_id"])
    wanted_keys = np.unique(hit_key)
    track_key, init_t = unique_track_init_times(track, wanted_keys, chunk_size)
    if len(track_key) == 0:
        return (
            hist,
            entries,
            total_entries,
            np.zeros(len(channels), dtype=np.int64),
            len(hits),
        )

    idx = np.searchsorted(track_key, hit_key)
    valid = idx < len(track_key)
    valid[valid] = track_key[idx[valid]] == hit_key[valid]
    unmatched = int((~valid).sum())
    if np.any(valid):
        selected_pmt = hits["pmt_id"][valid].astype(np.int64, copy=False)
        selected_idx = np.searchsorted(channels, selected_pmt)
        channel_valid = selected_idx < len(channels)
        channel_valid[channel_valid] = (
            channels[selected_idx[channel_valid]] == selected_pmt[channel_valid]
        )
        total_event_ids = [set() for _ in channels]
        if np.any(channel_valid):
            valid_idx = selected_idx[channel_valid]
            np.add.at(total_entries, valid_idx, 1)
            for channel_index in np.unique(valid_idx):
                total_event_ids[int(channel_index)].update(
                    hits["event_id"][valid][channel_valid & (selected_idx == channel_index)].tolist()
                )
        events = np.asarray([len(values) for values in total_event_ids], dtype=np.int64)
        time = (
            hits["hit_time"][valid].astype(np.float64, copy=False)
            - init_t[idx[valid]]
        )
        _accumulate_times_into_hist(
            hits["pmt_id"][valid].astype(np.int64, copy=False),
            time,
            channels,
            edges,
            hist,
            entries,
        )
    else:
        events = np.zeros(len(channels), dtype=np.int64)
    return hist, entries, total_entries, events, unmatched


def read_junosw_sim_histograms(
    sim_files, sim_group, edges, chunk_size, match_mode, channels
):
    """Read official JUNO HDF5 simulation files using the original matching logic."""
    channels = np.asarray(channels, dtype=np.int64)
    hist = np.zeros((len(channels), len(edges) - 1), dtype=np.int64)
    entries = np.zeros(len(channels), dtype=np.int64)
    total_entries = np.zeros(len(channels), dtype=np.int64)
    events = np.zeros(len(channels), dtype=np.int64)
    unmatched = 0
    event_mode_used = 0
    global_mode_used = 0

    for sim_file in tqdm(sim_files, desc="JUNO official simulation files"):
        with h5.File(sim_file, "r") as f:
            cd_hit = f[f"{sim_group}/CDHit"]
            track = f[f"{sim_group}/Track"]
            hits = read_compound_fields(
                cd_hit, ("event_id", "track_id", "pmt_id", "hit_time")
            )
            if len(hits) == 0:
                continue

            file_result = None
            if match_mode in ("auto", "event"):
                hit_ranges = sorted_event_ranges(hits["event_id"])
                track_ranges = sorted_event_ranges_dataset(track, chunk_size)
                if hit_ranges is not None and track_ranges is not None:
                    file_result = fill_histogram_by_event_ranges(
                        hits, hit_ranges, track, track_ranges, edges, channels
                    )
                    event_mode_used += 1
                elif match_mode == "event":
                    raise RuntimeError(f"{sim_file} is not sorted by event_id")

            if file_result is None:
                file_result = fill_histogram_by_global_keys(
                    hits, track, edges, chunk_size, channels
                )
                global_mode_used += 1

            (
                file_hist,
                file_entries,
                file_total_entries,
                file_events,
                file_unmatched,
            ) = file_result
            hist += file_hist
            entries += file_entries
            total_entries += file_total_entries
            events += file_events
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
        print(f"JUNO official hits without matched (event_id, track_id): {unmatched}")
    print(
        f"JUNO official match mode files: event={event_mode_used}, "
        f"global={global_mode_used}"
    )
    return density, entries, total_entries, events


def compute_lucid_sim_data(args, channels):
    if args.sim_chunk_size <= 0:
        raise ValueError("--sim-chunk-size must be positive")

    edges = np.linspace(args.data_range[0], args.data_range[1], args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    if args.lucid_sim_npz is not None:
        print(f"Reading merged LUCiD simulation npz: {args.lucid_sim_npz}")
        (
            sim_raw_density,
            sim_entries,
            sim_total_pe,
            sim_events,
        ) = read_lucid_sim_histograms_merged(
            args.lucid_sim_npz, edges, args.sim_chunk_size, channels
        )
    else:
        sim_files = iter_sim_files(args.lucid_sim_dir, args.sim_file_glob)
        if args.brief:
            sim_files = sim_files[:BRIEF_SIM_FILE_LIMIT]
        if not sim_files:
            raise FileNotFoundError(f"No LUCiD npz files found in {args.lucid_sim_dir}")
        (
            sim_raw_density,
            sim_entries,
            sim_total_pe,
            sim_events,
        ) = read_lucid_sim_histograms(
            sim_files, edges, args.sim_chunk_size, channels
        )
    if np.sum(sim_entries) == 0:
        raise RuntimeError("No simulation time entries.")

    data = {
        "channels": channels.astype(np.int64),
        "edges": edges.astype(np.float64),
        "centers": centers.astype(np.float64),
        "simulation_raw_density": sim_raw_density.astype(np.float64),
        "simulation_entries": sim_entries.astype(np.int64),
        "simulation_total_pe": sim_total_pe.astype(np.int64),
        "simulation_events": sim_events.astype(np.int64),
        "label": np.asarray(args.lucid_label),
        "kind": np.asarray("lucid"),
    }
    args.sim_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.sim_data, **data)
    print(f"Wrote {args.sim_data}")
    return data


def compute_junosw_sim_data(args, channels):
    if args.sim_chunk_size <= 0:
        raise ValueError("--sim-chunk-size must be positive")

    sim_files = iter_junosw_sim_files(
        args.junosw_sim_dir, args.junosw_sim_file_glob
    )
    if args.brief:
        sim_files = sim_files[:BRIEF_SIM_FILE_LIMIT]
    if not sim_files:
        raise FileNotFoundError(
            f"No JUNO official HDF5 files found in {args.junosw_sim_dir}"
        )

    edges = np.linspace(args.data_range[0], args.data_range[1], args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    (
        sim_raw_density,
        sim_entries,
        sim_total_pe,
        sim_events,
    ) = read_junosw_sim_histograms(
        sim_files,
        args.junosw_sim_group,
        edges,
        args.sim_chunk_size,
        args.junosw_sim_match_mode,
        channels,
    )
    if np.sum(sim_entries) == 0:
        raise RuntimeError("No JUNO official simulation time entries.")

    data = {
        "channels": channels.astype(np.int64),
        "edges": edges.astype(np.float64),
        "centers": centers.astype(np.float64),
        "simulation_raw_density": sim_raw_density.astype(np.float64),
        "simulation_entries": sim_entries.astype(np.int64),
        "simulation_total_pe": sim_total_pe.astype(np.int64),
        "simulation_events": sim_events.astype(np.int64),
        "label": np.asarray(args.junosw_label),
        "kind": np.asarray("junosw"),
    }
    args.junosw_sim_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.junosw_sim_data, **data)
    print(f"Wrote {args.junosw_sim_data}")
    return data


# Backward-compatible function name for callers that imported the old helper.
compute_sim_data = compute_lucid_sim_data


def compute_fsmp_data(args, channels, fsmp_files):
    pmt_type, tts = read_pmt_info(channels, args.pmt_map, args.pmt)
    fsmp_df = read_fsmp_times_by_channel(
        fsmp_files,
        args.fsmp_dir,
        args.rtraw_dir,
        channels,
        args.run,
        BRIEF_FSMP_FILE_LIMIT if args.brief else None,
        time_mode=args.fsmp_time_mode,
        print_first_row=args.print_fsmp_row,
    )
    if fsmp_df.empty:
        raise RuntimeError(f"No RUN {args.run} time entries after channel selection and monitor join.")

    fsmp_df = fsmp_df.sort_values("ch", kind="mergesort")
    data = {
        "channels": channels.astype(np.int64),
        "pmt_type": pmt_type.astype(np.int16),
        "tts": tts.astype(np.float64),
        "fsmp_ch": fsmp_df["ch"].to_numpy(dtype=np.int64),
        "fsmp_eid": fsmp_df["eid"].to_numpy(dtype=np.int64),
        "fsmp_pe_count": fsmp_df["pe_count"].to_numpy(dtype=np.int64),
        "fsmp_time": fsmp_df["time"].to_numpy(dtype=np.float64),
        "fsmp_time_mode": np.asarray(args.fsmp_time_mode),
    }
    args.fsmp_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fsmp_data, **data)
    print(f"Wrote {args.fsmp_data}")
    return data


def load_data(path):
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def simulation_arrays(data):
    """Normalize one-simulation and multi-simulation caches to shape [S, C, ...]."""
    density = np.asarray(data["simulation_density"], dtype=np.float64)
    raw_density = np.asarray(
        data.get("simulation_raw_density", density), dtype=np.float64
    )
    entries = np.asarray(data["simulation_entries"], dtype=np.int64)
    peaks = np.asarray(data.get("simulation_peak", np.full(entries.shape, np.nan)))

    if density.ndim == 2:
        density = density[np.newaxis, ...]
    if raw_density.ndim == 2:
        raw_density = raw_density[np.newaxis, ...]
    if entries.ndim == 1:
        entries = entries[np.newaxis, ...]
    if peaks.ndim == 1:
        peaks = peaks[np.newaxis, ...]

    labels = data.get("simulation_labels")
    if labels is None:
        label = data.get("simulation_label", data.get("label", "Simulation"))
        labels = np.asarray([str(np.asarray(label).item())])
    else:
        labels = np.asarray(labels).astype(str)
    if len(labels) != density.shape[0]:
        labels = np.asarray(
            [f"Simulation {i + 1}" for i in range(density.shape[0])]
        )
    return labels, raw_density, density, entries, peaks


def metric_arrays(data):
    edges = data["edges"]
    bin_width = edges[1] - edges[0]
    run_density = data["run_density"]
    _, _, sim_density, _, _ = simulation_arrays(data)
    if run_density.ndim == 2:
        run_density = run_density[np.newaxis, ...]

    if "l1_distance" in data:
        l1 = np.asarray(data["l1_distance"], dtype=np.float64)
        if l1.ndim == 1:
            l1 = l1[np.newaxis, ...]
    else:
        l1 = np.asarray(
            [
                [
                    l1_distance(run, sim, bin_width)
                    for run, sim in zip(run_density, source_density)
                ]
                for source_density in sim_density
            ],
            dtype=np.float64,
        )

    if "wasserstein_distance" in data:
        wasserstein = np.asarray(data["wasserstein_distance"], dtype=np.float64)
        if wasserstein.ndim == 1:
            wasserstein = wasserstein[np.newaxis, ...]
    else:
        wasserstein = np.asarray(
            [
                [
                    wasserstein_distance(run, sim, bin_width)
                    for run, sim in zip(run_density, source_density)
                ]
                for source_density in sim_density
            ],
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


def format_count(value):
    value = int(value)
    return "n/a" if value < 0 else f"{value:,}"


def simulation_colors(data, labels):
    """Return stable curve colors for the simulation sources in ``data``."""
    kinds = np.asarray(
        data.get("simulation_kinds", np.full(len(labels), "simulation"))
    ).astype(str)
    colors = []
    for index, (kind, label) in enumerate(zip(kinds, labels)):
        kind = kind.lower()
        label = str(label).lower()
        if kind in SOURCE_COLORS:
            colors.append(SOURCE_COLORS[kind])
        elif "lucid" in label:
            colors.append(SOURCE_COLORS["lucid"])
        elif "juno" in label:
            colors.append(SOURCE_COLORS["junosw"])
        else:
            colors.append(FALLBACK_SOURCE_COLORS[index % len(FALLBACK_SOURCE_COLORS)])
    return colors


def processing_title(data):
    """Describe the simulation-side response processing stored in the cache."""
    raw = str(np.asarray(data.get("processing", "")).item())
    flags = dict(
        item.split("=", 1)
        for item in raw.split(",")
        if "=" in item
    )
    tts_enabled = flags.get("tts", "unknown")
    rif_config = flags.get("rif", "unknown")
    dark_enabled = flags.get("dark", "unknown")
    if "|" in rif_config and ":" in rif_config:
        rif_values = {}
        for item in rif_config.split("|"):
            name, value = item.split(":", 1)
            rif_values[name] = value
        rif_title = (
            f"RIF sigma: Hamamatsu={rif_values.get('hamamatsu', '?')} ns, "
            f"NNVT={rif_values.get('nnvt', '?')} ns"
        )
    else:
        rif_title = f"RIF sigma={rif_config} ns"
    return (
        f"simulation TTS convolution={'on' if tts_enabled == '1' else 'off' if tts_enabled == '0' else tts_enabled}; "
        f"{rif_title}; FSMP dark subtraction={'on' if dark_enabled == '1' else 'off' if dark_enabled == '0' else dark_enabled}"
    )


def source_count_arrays(data, labels):
    """Return run/simulation count arrays normalized to [source, channel]."""
    _, _, sim_density, sim_entries, _ = simulation_arrays(data)
    n_sources, n_channels = sim_density.shape[:2]
    shape = (n_sources, n_channels)
    run_entries = np.asarray(data["run_entries"], dtype=np.int64)
    if run_entries.ndim == 1:
        run_entries = run_entries[np.newaxis, ...]
    if run_entries.shape != shape:
        run_entries = np.broadcast_to(run_entries, shape).copy()
    run_events = normalized_source_counts(
        data, "run_events", "run_entries", shape
    )
    run_total_pe = normalized_source_counts(
        data, "run_total_pe", "run_entries", shape
    )
    sim_total_pe = normalized_source_counts(
        data, "simulation_total_pe", "simulation_entries", shape
    )
    sim_events = normalized_source_counts(
        data,
        "simulation_events",
        "simulation_entries",
        shape,
        fill_value=-1,
    )
    if len(labels) != n_sources:
        raise ValueError("Simulation label count does not match source count")
    return (
        run_entries,
        run_events,
        run_total_pe,
        sim_entries,
        sim_events,
        sim_total_pe,
    )


def count_lines_for_channel(
    run,
    labels,
    run_entries,
    run_events,
    run_total_pe,
    sim_entries,
    sim_events,
    sim_total_pe,
    channel_index,
):
    lines = [
        f"FSMP RUN {run}: PE={format_count(run_total_pe[0, channel_index])}"
    ]
    lines.extend(
        f"{label}: PE={format_count(sim_total_pe[source_index, channel_index])}"
        for source_index, label in enumerate(labels)
    )
    return "\n".join(lines)


def write_channel_manifest(
    path,
    data,
    run,
    source_z,
    rif_sigma_hamamatsu=0.0,
    rif_sigma_nnvt=0.0,
):
    """Write selected channels, detector coordinates and PMT metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = np.asarray(data["channels"], dtype=np.int64)
    pmt_type = np.asarray(data["pmt_type"], dtype=np.int16)
    tts = np.asarray(data["tts"], dtype=np.float64)
    rif_sigma = np.asarray(
        data.get(
            "rif_sigma_by_channel",
            [
                rif_sigma_for_pmt_type(
                    pmt, rif_sigma_hamamatsu, rif_sigma_nnvt
                )
                for pmt in pmt_type
            ],
        ),
        dtype=np.float64,
    )
    z = np.asarray(data["z"], dtype=np.float64)
    theta = np.asarray(data["theta"], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "source_z_m": np.full(len(channels), source_z, dtype=np.float64),
            "run": np.full(len(channels), run, dtype=np.int64),
            "channel": channels,
            "pmt_type": pmt_type,
            "pmt_type_name": [
                PMT_TYPE_NAME.get(int(value), f"Unknown type {int(value)}")
                for value in pmt_type
            ],
            "z_mm": z,
            "theta_deg": theta,
            "tts_ns": tts,
            "rif_sigma_ns": rif_sigma,
        }
    )
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")


def source_column_slug(label, index):
    """Create a stable ASCII column prefix from a simulation label."""
    slug = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(label)
    ).strip("_")
    return slug or f"simulation_{index + 1}"


def write_comparison_stats_csv(path, data, run, source_z):
    """Write queryable per-channel counts, titles, metrics and dark-noise stats."""
    channels = np.asarray(data["channels"], dtype=np.int64)
    pmt_type = np.asarray(data["pmt_type"], dtype=np.int16)
    tts = np.asarray(data["tts"], dtype=np.float64)
    rif_sigma = np.asarray(
        data.get("rif_sigma_by_channel", np.zeros(len(channels), dtype=np.float64)),
        dtype=np.float64,
    )
    z = np.asarray(data["z"], dtype=np.float64)
    theta = np.asarray(data["theta"], dtype=np.float64)
    labels, _, _, _, _ = simulation_arrays(data)
    l1, wasserstein = metric_arrays(data)
    (
        run_entries,
        run_events,
        run_total_pe,
        sim_entries,
        sim_events,
        sim_total_pe,
    ) = source_count_arrays(data, labels)

    processing = processing_title(data)
    fsmp_mode = str(np.asarray(data.get("fsmp_time_mode", "unknown")).item())
    n_selected = len(channels)
    rows = []
    for channel_index, channel in enumerate(channels):
        pmt_name = PMT_TYPE_NAME.get(
            int(pmt_type[channel_index]),
            f"Unknown type {int(pmt_type[channel_index])}",
        )
        title_short = (
            f"Channel {int(channel)} | {pmt_name} | "
            f"z={z[channel_index]:.2f} mm | "
            f"theta={theta[channel_index]:.2f} deg"
        )
        title_full = (
            f"Channel {int(channel)} ({pmt_name}); "
            f"PMT TTS={tts[channel_index]:.2f} ns; "
            f"z={z[channel_index]:.2f} mm; "
            f"theta={theta[channel_index]:.2f} deg; "
            f"FSMP RUN {run} vs {', '.join(map(str, labels))}; "
            f"{processing}"
        )
        summary_title = (
            f"FSMP RUN {run} vs {', '.join(map(str, labels))}; "
            f"selected channels: "
            f"FSMP={int(np.count_nonzero(run_entries[0] > 0))}; "
            + "; ".join(
                f"{label}={int(np.count_nonzero(sim_entries[source_index] > 0))}"
                for source_index, label in enumerate(labels)
            )
        )
        row = {
            "run": int(run),
            "source_z_m": float(source_z),
            "channel": int(channel),
            "pmt_type": int(pmt_type[channel_index]),
            "pmt_type_name": pmt_name,
            "z_mm": float(z[channel_index]),
            "theta_deg": float(theta[channel_index]),
            "tts_ns": float(tts[channel_index]),
            "rif_sigma_ns": float(rif_sigma[channel_index]),
            "fsmp_time_mode": fsmp_mode,
            "processing": processing,
            "selected_channel_count": n_selected,
            "pdf_title_short": title_short,
            "pdf_title_full": title_full,
            "summary_title_base": summary_title,
            "fsmp_events": int(run_events[0, channel_index]),
            "fsmp_pe": int(run_total_pe[0, channel_index]),
            "fsmp_time_samples": int(run_entries[0, channel_index]),
            "fsmp_dark_prepeak_pe": int(
                np.asarray(data.get("run_dark_prepeak_pe", np.zeros_like(run_entries)))[
                    0, channel_index
                ]
            ),
            "fsmp_dark_rate_per_ns": float(
                np.asarray(data.get("run_dark_rate_per_ns", np.zeros_like(run_entries, dtype=float)))[
                    0, channel_index
                ]
            ),
            "fsmp_dark_window_ns": float(
                np.asarray(
                    data.get(
                        "run_dark_window_ns",
                        np.zeros_like(run_entries, dtype=float),
                    )
                )[0, channel_index]
            ),
            "fsmp_dark_window_low_ns": float(
                np.asarray(
                    data.get(
                        "run_dark_window_low_ns",
                        np.full_like(run_entries, np.nan, dtype=float),
                    )
                )[0, channel_index]
            ),
            "fsmp_dark_window_high_ns": float(
                np.asarray(
                    data.get(
                        "run_dark_window_high_ns",
                        np.full_like(run_entries, np.nan, dtype=float),
                    )
                )[0, channel_index]
            ),
            "fsmp_dark_density_floor_per_ns": float(
                np.asarray(
                    data.get(
                        "run_dark_density_floor",
                        np.zeros_like(run_entries, dtype=float),
                    )
                )[0, channel_index]
            ),
            "fsmp_dark_normalization_pe": int(
                np.asarray(
                    data.get(
                        "run_dark_normalization_pe",
                        np.zeros_like(run_entries, dtype=np.int64),
                    )
                )[0, channel_index]
            ),
            "fsmp_dark_expected_pe_data_range": float(
                np.asarray(
                    data.get(
                        "run_dark_expected_pe",
                        np.zeros_like(run_entries, dtype=float),
                    )
                )[0, channel_index]
            ),
            "fsmp_dark_removed_pe_data_range": float(
                np.asarray(
                    data.get(
                        "run_dark_removed_pe",
                        np.zeros_like(run_entries, dtype=float),
                    )
                )[0, channel_index]
            ),
        }
        for source_index, label in enumerate(labels):
            prefix = source_column_slug(label, source_index)
            row[f"{prefix}_events"] = int(sim_events[source_index, channel_index])
            row[f"{prefix}_pe"] = int(sim_total_pe[source_index, channel_index])
            row[f"{prefix}_time_samples"] = int(sim_entries[source_index, channel_index])
            row[f"{prefix}_l1"] = float(l1[source_index, channel_index])
            row[f"{prefix}_wasserstein_ns"] = float(
                wasserstein[source_index, channel_index]
            )
            row[f"{prefix}_peak_ns"] = float(
                np.asarray(data["simulation_peak"])[source_index, channel_index]
            )
            row[f"{prefix}_selected_channel_count"] = int(
                np.count_nonzero(sim_entries[source_index] > 0)
            )
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    print(
        f"Dark-noise audit RUN {run}: "
        f"prepeak PE={format_count(frame['fsmp_dark_prepeak_pe'].sum())}; "
        f"expected PE in data range={frame['fsmp_dark_expected_pe_data_range'].sum():.3f}; "
        f"removed equivalent PE={frame['fsmp_dark_removed_pe_data_range'].sum():.3f}"
    )
    print(f"Wrote {path}")


def plot_data(data, output, plot_range, run):
    configure_plot_style()

    channels = data["channels"]
    pmt_type = data["pmt_type"]
    tts = data["tts"]
    z = data["z"]
    theta = data["theta"]
    centers = data["centers"]
    run_density = np.asarray(data["run_density"], dtype=np.float64)
    if run_density.ndim == 2:
        run_density = run_density[np.newaxis, ...]
    labels, _, sim_density, _, _ = simulation_arrays(data)
    l1, wasserstein = metric_arrays(data)
    (
        run_entries,
        run_events,
        run_total_pe,
        sim_entries,
        sim_events,
        sim_total_pe,
    ) = source_count_arrays(data, labels)
    source_colors = simulation_colors(data, labels)

    with PdfPages(output) as pdf:
        for i, ch in enumerate(channels):
            pmt_name = PMT_TYPE_NAME.get(int(pmt_type[i]), f"Unknown type {int(pmt_type[i])}")
            channel_title = (
                f"Channel {int(ch)} | {pmt_name}\n"
                f"PMT TTS sigma={tts[i]:.2f} ns; z={z[i]:.2f} mm; "
                f"theta={theta[i]:.2f} deg"
            )

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                centers,
                run_density[0, i],
                where="mid",
                linewidth=1.2,
                linestyle="--",
                color=FSMP_COLOR,
                label=f"FSMP RUN {run}",
            )
            for source_idx, (label, color) in enumerate(zip(labels, source_colors)):
                ax.step(
                    centers,
                    sim_density[source_idx, i],
                    where="mid",
                    linewidth=1.4,
                    color=color,
                    label=str(label),
                )
            metric_text = (
                count_lines_for_channel(
                    run,
                    labels,
                    run_entries,
                    run_events,
                    run_total_pe,
                    sim_entries,
                    sim_events,
                    sim_total_pe,
                    i,
                )
                + "\n"
                + "\n".join(
                    f"{labels[source_idx]}: L1={l1[source_idx, i]:.4g}, "
                    f"W={wasserstein[source_idx, i]:.4g} ns"
                    for source_idx in range(len(labels))
                )
            )
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
            difference_lines = []
            difference_labels = []
            for source_idx, (label, color) in enumerate(zip(labels, source_colors)):
                difference = sim_density[source_idx, i] - run_density[0, i]
                difference_line = difference_ax.step(
                    centers,
                    difference,
                    where="mid",
                    linewidth=1.3,
                    color=color,
                    linestyle="-",
                    label=f"difference: {label} - FSMP",
                )[0]
                difference_lines.append(difference_line)
                difference_labels.append(difference_line.get_label())
            difference_ax.axhline(
                0.0,
                color=DIFFERENCE_AXIS_COLOR,
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )

            ratio_ax = difference_ax.twinx()
            ratio_lines = []
            ratio_labels = []
            for source_idx, (label, color) in enumerate(zip(labels, source_colors)):
                ratio = np.divide(
                    sim_density[source_idx, i],
                    run_density[0, i],
                    out=np.full_like(run_density[0, i], np.nan),
                    where=run_density[0, i] > 0,
                )
                ratio_line = ratio_ax.step(
                    centers,
                    ratio,
                    where="mid",
                    linewidth=1.3,
                    color=color,
                    linestyle=":",
                    label=f"ratio: {label} / FSMP",
                )[0]
                ratio_lines.append(ratio_line)
                ratio_labels.append(ratio_line.get_label())
            ratio_ax.axhline(
                1.0,
                color=RATIO_AXIS_COLOR,
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )
            ratio_ax.set_yscale("log")

            difference_ax.set_title(channel_title)
            difference_ax.set_xlim(plot_range)
            difference_ax.set_xlabel("Time [ns]")
            difference_ax.set_ylabel(
                "Difference: simulation - FSMP density", color=DIFFERENCE_AXIS_COLOR
            )
            difference_ax.tick_params(axis="y", colors=DIFFERENCE_AXIS_COLOR)
            difference_ax.spines["left"].set_color(DIFFERENCE_AXIS_COLOR)
            ratio_ax.set_ylabel(
                "Ratio: simulation / FSMP", color=RATIO_AXIS_COLOR
            )
            ratio_ax.tick_params(axis="y", colors=RATIO_AXIS_COLOR)
            ratio_ax.spines["right"].set_color(RATIO_AXIS_COLOR)
            difference_ax.legend(
                difference_lines + ratio_lines,
                difference_labels + ratio_labels,
                frameon=False,
                loc="lower right",
                fontsize="x-small",
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
            for source_idx, label in enumerate(labels):
                plot_metric_distribution(
                    pdf,
                    l1[source_idx, mask],
                    f"{title} {label} L1 distance distribution",
                    "L1 distance",
                )
        for title, mask in groups:
            for source_idx, label in enumerate(labels):
                plot_metric_distribution(
                    pdf,
                    wasserstein[source_idx, mask],
                    f"{title} {label} Wasserstein distance distribution",
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
    run_density = np.asarray(data["run_density"], dtype=np.float64)
    if run_density.ndim == 2:
        run_density = run_density[np.newaxis, ...]
    labels, _, sim_density, sim_entries, _ = simulation_arrays(data)
    (
        run_entries,
        run_events,
        run_total_pe,
        sim_entries,
        sim_events,
        sim_total_pe,
    ) = source_count_arrays(data, labels)
    source_colors = simulation_colors(data, labels)

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
            mean_sims = []
            mean_runs = []
            summary_metrics = []
            for source_idx, label in enumerate(labels):
                run_mask = type_mask & (run_entries[source_idx] > 0)
                source_mask = type_mask & (sim_entries[source_idx] > 0)
                if np.any(run_mask):
                    mean_run = normalize_density(
                        np.mean(run_density[source_idx, run_mask], axis=0),
                        bin_width,
                    )
                else:
                    mean_run = np.zeros(len(centers), dtype=np.float64)
                if np.any(source_mask):
                    mean_sim = normalize_density(
                        np.mean(sim_density[source_idx, source_mask], axis=0),
                        bin_width,
                    )
                else:
                    mean_sim = np.zeros(len(centers), dtype=np.float64)
                mean_runs.append(mean_run)
                mean_sims.append(mean_sim)
                summary_metrics.append(
                    f"{label}: L1={l1_distance(mean_run, mean_sim, bin_width):.4g}, "
                    f"W={wasserstein_distance(mean_run, mean_sim, bin_width):.4g} ns"
                )

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                centers,
                mean_runs[0],
                where="mid",
                linewidth=1.2,
                linestyle="--",
                color=FSMP_COLOR,
                label=f"FSMP RUN {run}",
            )
            for label, mean_sim, color in zip(labels, mean_sims, source_colors):
                ax.step(
                    centers,
                    mean_sim,
                    where="mid",
                    linewidth=1.4,
                    color=color,
                    label=str(label),
                )
            count_lines = [
                (
                    f"FSMP RUN {run}: channels={int(np.count_nonzero(type_mask & (run_entries[0] > 0)))}; "
                    f"PE={format_count(np.sum(run_total_pe[0, type_mask]))}"
                )
            ]
            count_lines.extend(
                (
                    f"{label}: channels={int(np.count_nonzero(type_mask & (sim_entries[source_idx] > 0)))}; "
                    f"PE={format_count(np.sum(sim_total_pe[source_idx, type_mask]))}"
                )
                for source_idx, label in enumerate(labels)
            )
            ax.text(
                0.98,
                0.98,
                "\n".join(count_lines + summary_metrics),
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_title(
                f"{title}\n"
                f"FSMP RUN {run} vs {', '.join(labels)}\n"
                "equal-weight mean over selected channels"
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
            difference_lines = []
            difference_labels = []
            for label, mean_sim, color in zip(labels, mean_sims, source_colors):
                difference_line = difference_ax.step(
                    centers,
                    mean_sim - mean_runs[0],
                    where="mid",
                    linewidth=1.3,
                    color=color,
                    linestyle="-",
                    label=f"difference: {label} - FSMP",
                )[0]
                difference_lines.append(difference_line)
                difference_labels.append(difference_line.get_label())
            difference_ax.axhline(
                0.0,
                color=DIFFERENCE_AXIS_COLOR,
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )

            ratio_ax = difference_ax.twinx()
            ratio_lines = []
            ratio_labels = []
            for label, mean_sim, color in zip(labels, mean_sims, source_colors):
                ratio = np.divide(
                    mean_sim,
                    mean_runs[0],
                    out=np.full_like(mean_runs[0], np.nan),
                    where=mean_runs[0] > 0,
                )
                ratio_line = ratio_ax.step(
                    centers,
                    ratio,
                    where="mid",
                    linewidth=1.3,
                    color=color,
                    linestyle=":",
                    label=f"ratio: {label} / FSMP",
                )[0]
                ratio_lines.append(ratio_line)
                ratio_labels.append(ratio_line.get_label())
            ratio_ax.axhline(
                1.0,
                color=RATIO_AXIS_COLOR,
                linewidth=1.8,
                linestyle="--",
                alpha=0.8,
            )
            ratio_ax.set_yscale("log")

            difference_ax.set_title(
                f"{title}\n"
                f"FSMP RUN {run} vs {', '.join(labels)}"
            )
            difference_ax.set_xlim(plot_range)
            difference_ax.set_xlabel("Time [ns]")
            difference_ax.set_ylabel(
                "Difference: simulation - FSMP density", color=DIFFERENCE_AXIS_COLOR
            )
            difference_ax.tick_params(axis="y", colors=DIFFERENCE_AXIS_COLOR)
            difference_ax.spines["left"].set_color(DIFFERENCE_AXIS_COLOR)
            ratio_ax.set_ylabel(
                "Ratio: simulation / FSMP", color=RATIO_AXIS_COLOR
            )
            ratio_ax.tick_params(axis="y", colors=RATIO_AXIS_COLOR)
            ratio_ax.spines["right"].set_color(RATIO_AXIS_COLOR)
            difference_ax.legend(
                difference_lines + ratio_lines,
                difference_labels + ratio_labels,
                frameon=False,
                loc="lower right",
                fontsize="x-small",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {output}")


# =====================================================================
# 对齐
# =====================================================================
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


def convolve_density(density, bin_width, sigma):
    """Convolve a density with a zero-mean Gaussian whose sigma is in ns."""
    kernel = gaussian_kernel(bin_width, sigma)
    # ``np.convolve(..., mode="same")`` returns the longer input length.
    # Explicitly crop the full convolution so short test ranges remain
    # shape-stable even when the Gaussian kernel is wider than the range.
    full = np.convolve(density, kernel, mode="full")
    start = (len(kernel) - 1) // 2
    conv = full[start : start + len(density)]
    area = conv.sum() * bin_width
    if area > 0:
        conv = conv / area
    return conv


def estimate_dark_window(times, peak, margin_ns):
    """Return the FSMP pre-peak window in the FSMP time coordinate.

    The lower edge is the earliest finite FSMP time for this channel.  This
    makes the estimate independent of the simulation histogram range; the
    upper edge keeps the configured safety margin before the FSMP peak.
    """
    finite_times = np.asarray(times, dtype=np.float64)
    finite_times = finite_times[np.isfinite(finite_times)]
    if len(finite_times) == 0 or not np.isfinite(peak):
        return None
    lo = float(np.min(finite_times))
    hi = float(peak) - margin_ns
    if hi <= lo:
        return None
    return lo, hi


def empty_dark_stats():
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
    fsmp_time,
    fsmp_density,
    edges,
    peak,
    bin_width,
    margin_ns,
    normalization_count=None,
    return_stats=False,
):
    """Subtract the estimated dark floor and optionally return audit statistics."""
    stats = empty_dark_stats()
    if len(fsmp_time) == 0 or fsmp_density is None:
        return (fsmp_density, stats) if return_stats else fsmp_density
    dark_window = estimate_dark_window(fsmp_time, peak, margin_ns)
    if dark_window is None:
        return (fsmp_density, stats) if return_stats else fsmp_density
    lo, hi = dark_window
    window_ns = hi - lo
    prepeak_pe = int(np.sum((fsmp_time >= lo) & (fsmp_time < hi)))
    dark_rate = prepeak_pe / window_ns
    if normalization_count is None:
        normalization_count = len(fsmp_time)
    normalization_count = int(normalization_count)
    dark_floor = dark_rate / max(normalization_count, 1)
    corrected = np.clip(fsmp_density - dark_floor, 0.0, None)
    removed_density = np.clip(fsmp_density - corrected, 0.0, None)
    stats.update(
        {
            "prepeak_pe": prepeak_pe,
            "rate_per_ns": float(dark_rate),
            "expected_pe": float(dark_rate * (edges[-1] - edges[0])),
            "removed_pe": float(
                np.sum(removed_density) * bin_width * normalization_count
            ),
            "window_ns": float(window_ns),
            "window_low_ns": float(lo),
            "window_high_ns": float(hi),
            "density_floor": float(dark_floor),
            "normalization_pe": normalization_count,
        }
    )
    return (corrected, stats) if return_stats else corrected


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


def fsmp_records_by_channel(fsmp_data):
    """Group FSMP first-hit rows and their event/PE metadata by channel."""
    fsmp_ch = fsmp_data["fsmp_ch"].astype(np.int64, copy=False)
    fsmp_time = fsmp_data["fsmp_time"].astype(np.float64, copy=False)
    fsmp_eid = np.asarray(
        fsmp_data.get("fsmp_eid", np.arange(len(fsmp_time))),
        dtype=np.int64,
    )
    fsmp_pe_count = np.asarray(
        fsmp_data.get("fsmp_pe_count", np.ones(len(fsmp_time))),
        dtype=np.int64,
    )
    order = np.argsort(fsmp_ch, kind="mergesort")
    sorted_ch = fsmp_ch[order]
    unique_ch, starts = np.unique(sorted_ch, return_index=True)
    stops = np.r_[starts[1:], len(sorted_ch)]
    return {
        int(ch): {
            "time": fsmp_time[order[start:stop]],
            "eid": fsmp_eid[order[start:stop]],
            "pe_count": fsmp_pe_count[order[start:stop]],
        }
        for ch, start, stop in zip(unique_ch, starts, stops)
    }


def normalized_source_counts(data, key, fallback_key, shape, fill_value=0):
    """Return a source-by-channel count array with backward-compatible fallback."""
    values = np.asarray(data.get(key, data.get(fallback_key)), dtype=np.int64)
    if values.size == 0:
        return np.full(shape, fill_value, dtype=np.int64)
    if values.ndim == 1:
        values = values[np.newaxis, ...]
    if values.shape != shape:
        raise ValueError(
            f"{key} has shape {values.shape}, expected source/channel shape {shape}"
        )
    return values


def build_channel_histograms(fsmp_data, sim_data, alignment="l1",
                             use_tts=True, rif_sigma=0.0,
                             rif_sigma_hamamatsu=None, rif_sigma_nnvt=None,
                             dark_subtract=False, dark_window=30.0):
    if alignment not in {"l1", "peak"}:
        raise ValueError(f"Unknown alignment mode: {alignment}")

    channels = fsmp_data["channels"].astype(np.int64, copy=False)
    pmt_type = fsmp_data["pmt_type"].astype(np.int16, copy=False)
    tts = fsmp_data["tts"].astype(np.float64, copy=False)
    rif_sigma_hamamatsu, rif_sigma_nnvt = resolve_rif_sigmas(
        rif_sigma, rif_sigma_hamamatsu, rif_sigma_nnvt
    )
    rif_sigmas = np.asarray(
        [
            rif_sigma_for_pmt_type(pmt, rif_sigma_hamamatsu, rif_sigma_nnvt)
            for pmt in pmt_type
        ],
        dtype=np.float64,
    )
    z = fsmp_data["z"].astype(np.float64, copy=False)
    theta = fsmp_data["theta"].astype(np.float64, copy=False)
    fsmp_lookup = fsmp_records_by_channel(fsmp_data)

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
    run_event_counts = []
    run_total_pe = []
    sim_event_counts = []
    sim_total_pe = []
    run_dark_prepeak_pe = []
    run_dark_rate_per_ns = []
    run_dark_window_ns = []
    run_dark_window_low_ns = []
    run_dark_window_high_ns = []
    run_dark_density_floor = []
    run_dark_normalization_pe = []
    run_dark_expected_pe = []
    run_dark_removed_pe = []

    for ch, ch_tts, ch_rif_sigma in tqdm(
        list(zip(channels, tts, rif_sigmas)), desc="Channel histograms"
    ):
        fsmp_record = fsmp_lookup.get(
            int(ch),
            {"time": np.array([]), "eid": np.array([]), "pe_count": np.array([])},
        )
        fsmp_time = fsmp_record["time"]
        sim_idx = np.searchsorted(sim_channels, ch)
        if sim_idx >= len(sim_channels) or sim_channels[sim_idx] != ch:
            sim_raw_density = np.zeros(len(edges) - 1, dtype=np.float64)
            sim_entries = 0
        else:
            sim_raw_density = sim_raw_density_by_channel[sim_idx]
            sim_entries = int(sim_entries_by_channel[sim_idx])

        # 模拟侧时间响应: 先逐通道 TTS 卷积 (真实 PMT 时间展宽, 可关), 再叠 RIF 高斯占位 (可关)
        sim_conv_density = sim_raw_density
        if use_tts:
            sim_conv_density = convolve_density(sim_conv_density, bin_width, ch_tts)
        if ch_rif_sigma > 0.0:
            sim_conv_density = convolve_density(
                sim_conv_density, bin_width, ch_rif_sigma
            )
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
        # FSMP 侧: 扣除前端 (主峰前) 暗噪声平底 (可关)
        if dark_subtract:
            normalization_count = int(
                np.histogram(fsmp_time + this_shift, bins=edges)[0].sum()
            )
            fsmp_aligned_density, dark_stats = subtract_dark_noise(
                fsmp_time,
                fsmp_aligned_density,
                edges,
                this_run_peak,
                bin_width,
                dark_window,
                normalization_count=normalization_count,
                return_stats=True,
            )
            this_l1 = l1_distance(fsmp_aligned_density, sim_conv_density, bin_width)
        else:
            dark_stats = empty_dark_stats()
        this_wasserstein = wasserstein_distance(fsmp_aligned_density, sim_conv_density, bin_width)

        run_density.append(fsmp_aligned_density)
        sim_raw_density_all.append(sim_raw_density)
        sim_density.append(sim_conv_density)
        run_peak.append(this_run_peak)
        sim_peak.append(this_sim_peak)
        fsmp_counts.append(len(fsmp_time))
        sim_counts.append(sim_entries)
        run_event_counts.append(len(np.unique(fsmp_record["eid"])))
        run_total_pe.append(int(np.sum(fsmp_record["pe_count"])))
        sim_event_counts.append(-1)
        sim_total_pe.append(sim_entries)
        run_dark_prepeak_pe.append(dark_stats["prepeak_pe"])
        run_dark_rate_per_ns.append(dark_stats["rate_per_ns"])
        run_dark_window_ns.append(dark_stats["window_ns"])
        run_dark_window_low_ns.append(dark_stats["window_low_ns"])
        run_dark_window_high_ns.append(dark_stats["window_high_ns"])
        run_dark_density_floor.append(dark_stats["density_floor"])
        run_dark_normalization_pe.append(dark_stats["normalization_pe"])
        run_dark_expected_pe.append(dark_stats["expected_pe"])
        run_dark_removed_pe.append(dark_stats["removed_pe"])
        l1_distances.append(this_l1)
        wasserstein_distances.append(this_wasserstein)

    return {
        "channels": channels.astype(np.int64),
        "pmt_type": pmt_type.astype(np.int16),
        "tts": tts.astype(np.float64),
        "rif_sigma_by_channel": rif_sigmas,
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
        "run_events": np.asarray(run_event_counts, dtype=np.int64),
        "run_total_pe": np.asarray(run_total_pe, dtype=np.int64),
        "simulation_events": np.asarray(sim_event_counts, dtype=np.int64),
        "simulation_total_pe": np.asarray(sim_total_pe, dtype=np.int64),
        "run_dark_prepeak_pe": np.asarray(run_dark_prepeak_pe, dtype=np.int64),
        "run_dark_rate_per_ns": np.asarray(run_dark_rate_per_ns, dtype=np.float64),
        "run_dark_window_ns": np.asarray(run_dark_window_ns, dtype=np.float64),
        "run_dark_window_low_ns": np.asarray(
            run_dark_window_low_ns, dtype=np.float64
        ),
        "run_dark_window_high_ns": np.asarray(
            run_dark_window_high_ns, dtype=np.float64
        ),
        "run_dark_density_floor": np.asarray(
            run_dark_density_floor, dtype=np.float64
        ),
        "run_dark_normalization_pe": np.asarray(
            run_dark_normalization_pe, dtype=np.int64
        ),
        "run_dark_expected_pe": np.asarray(run_dark_expected_pe, dtype=np.float64),
        "run_dark_removed_pe": np.asarray(run_dark_removed_pe, dtype=np.float64),
        "l1_distance": np.asarray(l1_distances, dtype=np.float64),
        "wasserstein_distance": np.asarray(wasserstein_distances, dtype=np.float64),
        "processing": np.asarray(
            processing_string(
                use_tts,
                rif_sigma_hamamatsu,
                rif_sigma_nnvt,
                dark_subtract,
                fsmp_data.get("fsmp_time_mode", "unknown"),
            ),
            dtype="<U128",
        ),
        "fsmp_time_mode": np.asarray(
            fsmp_data.get("fsmp_time_mode", "unknown")
        ),
    }


def build_channel_histograms_multi(
    fsmp_data,
    sim_data_list,
    alignment="l1",
    use_tts=True,
    rif_sigma=0.0,
    rif_sigma_hamamatsu=None,
    rif_sigma_nnvt=None,
    dark_subtract=False,
    dark_window=30.0,
):
    """Build aligned comparison arrays for one or more simulation sources."""
    if alignment not in {"l1", "peak"}:
        raise ValueError(f"Unknown alignment mode: {alignment}")
    if not sim_data_list:
        raise ValueError("At least one simulation source is required")

    channels = fsmp_data["channels"].astype(np.int64, copy=False)
    pmt_type = fsmp_data["pmt_type"].astype(np.int16, copy=False)
    tts = fsmp_data["tts"].astype(np.float64, copy=False)
    rif_sigma_hamamatsu, rif_sigma_nnvt = resolve_rif_sigmas(
        rif_sigma, rif_sigma_hamamatsu, rif_sigma_nnvt
    )
    rif_sigmas = np.asarray(
        [
            rif_sigma_for_pmt_type(pmt, rif_sigma_hamamatsu, rif_sigma_nnvt)
            for pmt in pmt_type
        ],
        dtype=np.float64,
    )
    z = fsmp_data["z"].astype(np.float64, copy=False)
    theta = fsmp_data["theta"].astype(np.float64, copy=False)
    fsmp_lookup = fsmp_records_by_channel(fsmp_data)

    edges = sim_data_list[0]["edges"].astype(np.float64, copy=False)
    centers = sim_data_list[0]["centers"].astype(np.float64, copy=False)
    for sim_data in sim_data_list[1:]:
        if not np.array_equal(sim_data["edges"], edges):
            raise ValueError("Simulation sources use different histogram edges")

    bin_width = edges[1] - edges[0]
    bins = len(edges) - 1
    n_sources = len(sim_data_list)

    run_density = [[] for _ in range(n_sources)]
    sim_raw_density = [[] for _ in range(n_sources)]
    sim_density = [[] for _ in range(n_sources)]
    run_peak = [[] for _ in range(n_sources)]
    sim_peak = [[] for _ in range(n_sources)]
    run_entries = [[] for _ in range(n_sources)]
    sim_entries = [[] for _ in range(n_sources)]
    l1_distances = [[] for _ in range(n_sources)]
    wasserstein_distances = [[] for _ in range(n_sources)]
    run_event_counts = [[] for _ in range(n_sources)]
    run_total_pe = [[] for _ in range(n_sources)]
    sim_event_counts = [[] for _ in range(n_sources)]
    sim_total_pe = [[] for _ in range(n_sources)]
    run_dark_prepeak_pe = [[] for _ in range(n_sources)]
    run_dark_rate_per_ns = [[] for _ in range(n_sources)]
    run_dark_window_ns = [[] for _ in range(n_sources)]
    run_dark_window_low_ns = [[] for _ in range(n_sources)]
    run_dark_window_high_ns = [[] for _ in range(n_sources)]
    run_dark_density_floor = [[] for _ in range(n_sources)]
    run_dark_normalization_pe = [[] for _ in range(n_sources)]
    run_dark_expected_pe = [[] for _ in range(n_sources)]
    run_dark_removed_pe = [[] for _ in range(n_sources)]

    for ch_index, (ch, ch_tts, ch_rif_sigma) in enumerate(
        tqdm(list(zip(channels, tts, rif_sigmas)), desc="Channel histograms")
    ):
        fsmp_record = fsmp_lookup.get(
            int(ch),
            {"time": np.array([]), "eid": np.array([]), "pe_count": np.array([])},
        )
        fsmp_time = fsmp_record["time"]
        this_run_peak = estimate_peak_from_times(fsmp_time, bins)
        source_results = []

        for source_index, sim_data in enumerate(sim_data_list):
            sim_channels = sim_data["channels"].astype(np.int64, copy=False)
            sim_idx = np.searchsorted(sim_channels, ch)
            if sim_idx >= len(sim_channels) or sim_channels[sim_idx] != ch:
                raw = np.zeros(len(edges) - 1, dtype=np.float64)
                n_sim = 0
                n_sim_total_pe = 0
                n_sim_events = 0
            else:
                raw = sim_data["simulation_raw_density"][sim_idx].astype(
                    np.float64, copy=False
                )
                n_sim = int(sim_data["simulation_entries"][sim_idx])
                n_sim_total_pe = int(
                    np.asarray(
                        sim_data.get("simulation_total_pe", sim_data["simulation_entries"])
                    )[sim_idx]
                )
                n_sim_events = int(
                    np.asarray(
                        sim_data.get(
                            "simulation_events",
                            np.full(len(sim_channels), -1, dtype=np.int64),
                        )
                    )[sim_idx]
                )

            observed = raw
            if use_tts:
                observed = convolve_density(observed, bin_width, ch_tts)
            if ch_rif_sigma > 0.0:
                observed = convolve_density(
                    observed, bin_width, ch_rif_sigma
                )

            this_sim_peak = estimate_peak_from_density(centers, observed)
            source_results.append(
                (raw, observed, this_sim_peak, n_sim, n_sim_total_pe, n_sim_events)
            )

        # Align FSMP once to the first simulation source. All simulation
        # curves then share one time coordinate for a genuine three-way plot.
        reference_peak = source_results[0][2]
        shift = reference_peak - this_run_peak
        if not np.isfinite(shift):
            shift = 0.0
        if alignment == "l1":
            shift, aligned, _ = local_l1_alignment(
                fsmp_time,
                source_results[0][1],
                edges,
                shift,
            )
        else:
            aligned = density_hist(fsmp_time + shift, edges)
        if dark_subtract:
            normalization_count = int(
                np.histogram(fsmp_time + shift, bins=edges)[0].sum()
            )
            aligned, dark_stats = subtract_dark_noise(
                fsmp_time,
                aligned,
                edges,
                this_run_peak,
                bin_width,
                dark_window,
                normalization_count=normalization_count,
                return_stats=True,
            )
        else:
            dark_stats = empty_dark_stats()

        for source_index, (
            raw,
            observed,
            this_sim_peak,
            n_sim,
            n_sim_total_pe,
            n_sim_events,
        ) in enumerate(
            source_results
        ):
            run_density[source_index].append(aligned)
            sim_raw_density[source_index].append(raw)
            sim_density[source_index].append(observed)
            run_peak[source_index].append(this_run_peak)
            sim_peak[source_index].append(this_sim_peak)
            run_entries[source_index].append(len(fsmp_time))
            sim_entries[source_index].append(n_sim)
            run_event_counts[source_index].append(
                len(np.unique(fsmp_record["eid"]))
            )
            run_total_pe[source_index].append(
                int(np.sum(fsmp_record["pe_count"]))
            )
            sim_event_counts[source_index].append(n_sim_events)
            sim_total_pe[source_index].append(n_sim_total_pe)
            run_dark_prepeak_pe[source_index].append(dark_stats["prepeak_pe"])
            run_dark_rate_per_ns[source_index].append(
                dark_stats["rate_per_ns"]
            )
            run_dark_window_ns[source_index].append(dark_stats["window_ns"])
            run_dark_window_low_ns[source_index].append(
                dark_stats["window_low_ns"]
            )
            run_dark_window_high_ns[source_index].append(
                dark_stats["window_high_ns"]
            )
            run_dark_density_floor[source_index].append(
                dark_stats["density_floor"]
            )
            run_dark_normalization_pe[source_index].append(
                dark_stats["normalization_pe"]
            )
            run_dark_expected_pe[source_index].append(
                dark_stats["expected_pe"]
            )
            run_dark_removed_pe[source_index].append(
                dark_stats["removed_pe"]
            )
            l1_distances[source_index].append(
                l1_distance(aligned, observed, bin_width)
            )
            wasserstein_distances[source_index].append(
                wasserstein_distance(aligned, observed, bin_width)
            )

    labels = [
        str(np.asarray(sim_data.get("label", f"Simulation {i + 1}")).item())
        for i, sim_data in enumerate(sim_data_list)
    ]
    kinds = [
        str(np.asarray(sim_data.get("kind", "simulation")).item())
        for sim_data in sim_data_list
    ]
    return {
        "channels": channels,
        "pmt_type": pmt_type,
        "tts": tts,
        "rif_sigma_by_channel": rif_sigmas,
        "z": z,
        "theta": theta,
        "edges": edges,
        "centers": centers,
        "run_density": np.asarray(run_density, dtype=np.float64),
        "simulation_raw_density": np.asarray(sim_raw_density, dtype=np.float64),
        "simulation_density": np.asarray(sim_density, dtype=np.float64),
        "run_peak": np.asarray(run_peak, dtype=np.float64),
        "simulation_peak": np.asarray(sim_peak, dtype=np.float64),
        "run_entries": np.asarray(run_entries, dtype=np.int64),
        "simulation_entries": np.asarray(sim_entries, dtype=np.int64),
        "run_events": np.asarray(run_event_counts, dtype=np.int64),
        "run_total_pe": np.asarray(run_total_pe, dtype=np.int64),
        "simulation_events": np.asarray(sim_event_counts, dtype=np.int64),
        "simulation_total_pe": np.asarray(sim_total_pe, dtype=np.int64),
        "run_dark_prepeak_pe": np.asarray(run_dark_prepeak_pe, dtype=np.int64),
        "run_dark_rate_per_ns": np.asarray(
            run_dark_rate_per_ns, dtype=np.float64
        ),
        "run_dark_window_ns": np.asarray(
            run_dark_window_ns, dtype=np.float64
        ),
        "run_dark_window_low_ns": np.asarray(
            run_dark_window_low_ns, dtype=np.float64
        ),
        "run_dark_window_high_ns": np.asarray(
            run_dark_window_high_ns, dtype=np.float64
        ),
        "run_dark_density_floor": np.asarray(
            run_dark_density_floor, dtype=np.float64
        ),
        "run_dark_normalization_pe": np.asarray(
            run_dark_normalization_pe, dtype=np.int64
        ),
        "run_dark_expected_pe": np.asarray(
            run_dark_expected_pe, dtype=np.float64
        ),
        "run_dark_removed_pe": np.asarray(
            run_dark_removed_pe, dtype=np.float64
        ),
        "l1_distance": np.asarray(l1_distances, dtype=np.float64),
        "wasserstein_distance": np.asarray(wasserstein_distances, dtype=np.float64),
        "simulation_labels": np.asarray(labels),
        "simulation_kinds": np.asarray(kinds),
        "processing": np.asarray(
            processing_string(
                use_tts,
                rif_sigma_hamamatsu,
                rif_sigma_nnvt,
                dark_subtract,
                fsmp_data.get("fsmp_time_mode", "unknown"),
            ),
            dtype="<U128",
        ),
        "fsmp_time_mode": np.asarray(
            fsmp_data.get("fsmp_time_mode", "unknown")
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"z={args.z:g} m -> RUN {args.run}; "
        f"FSMP={args.fsmp_dir}; rtraw={args.rtraw_dir}; "
        f"LUCiD={args.lucid_sim_dir or args.lucid_sim_npz}; "
        f"JUNO={args.junosw_sim_dir}; output={args.output_dir}"
    )

    if args.merge_sim_to is not None:
        sim_files = iter_sim_files(args.lucid_sim_dir, args.sim_file_glob)
        merge_lucid_sim(sim_files, args.merge_sim_to)
        return

    if args.brief:
        print(
            f"Brief mode: first {BRIEF_FSMP_FILE_LIMIT} FSMP files and "
            f"first {BRIEF_SIM_FILE_LIMIT} simulation files"
        )

    if args.plot_only:
        if not args.fsmp_data.exists():
            raise FileNotFoundError(f"{args.fsmp_data} does not exist for --plot-only")
        fsmp_data = load_data(args.fsmp_data)
        sim_sources = []
        if args.lucid_sim_dir is not None or args.lucid_sim_npz is not None:
            if not args.sim_data.exists():
                raise FileNotFoundError(
                    f"{args.sim_data} does not exist for LUCiD --plot-only"
                )
            sim_sources.append(load_data(args.sim_data))
        if args.junosw_sim_dir is not None:
            if not args.junosw_sim_data.exists():
                raise FileNotFoundError(
                    f"{args.junosw_sim_data} does not exist for JUNO --plot-only"
                )
            sim_sources.append(load_data(args.junosw_sim_data))
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

        sim_sources = []
        source_specs = []
        if args.lucid_sim_dir is not None or args.lucid_sim_npz is not None:
            source_specs.append(("lucid", args.sim_data, compute_lucid_sim_data))
        if args.junosw_sim_dir is not None:
            source_specs.append(("junosw", args.junosw_sim_data, compute_junosw_sim_data))

        for kind, cache_path, compute in source_specs:
            if cache_path.exists() and not args.force:
                cached_sim_data = load_data(cache_path)
                channels_ok = np.array_equal(
                    cached_sim_data.get("channels"), channels
                )
                kind_ok = str(cached_sim_data.get("kind", kind)) == kind
                count_fields_ok = (
                    "simulation_total_pe" in cached_sim_data
                    and "simulation_events" in cached_sim_data
                )
                if channels_ok and kind_ok and count_fields_ok:
                    sim_sources.append(cached_sim_data)
                    print(f"Load cached {kind} simulation data from {cache_path}.")
                    continue
                if channels_ok and kind_ok and not count_fields_ok:
                    print(
                        f"Cached {kind} simulation data lacks total PE/event counts; "
                        "recomputing it once."
                    )
                print(f"Cached {kind} simulation data is incompatible; recomputing.")
            sim_sources.append(compute(args, channels))

    if not sim_sources:
        raise RuntimeError("No simulation source was loaded.")

    fsmp_data = attach_pmt_coordinates(fsmp_data, args.pmt_map)
    write_channel_manifest(
        args.channel_manifest,
        fsmp_data,
        args.run,
        args.z,
        args.rif_sigma_hamamatsu,
        args.rif_sigma_nnvt,
    )

    # 处理配置 (tts/rif/dark) 变化时, 旧的 aligned 缓存与旧 PDF 均失效, 需要重算重画
    processing = processing_string(
        not args.no_tts,
        args.rif_sigma_hamamatsu,
        args.rif_sigma_nnvt,
        args.dark_subtract,
        args.fsmp_time_mode,
    )
    cached_aligned_data = None
    aligned_ok = False
    if args.aligned_data.exists() and not args.force:
        cached_aligned_data = load_data(args.aligned_data)
        source_labels = np.asarray(
            [
                str(np.asarray(sim.get("label", "Simulation")).item())
                for sim in sim_sources
            ]
        )
        source_kinds = np.asarray(
            [
                str(np.asarray(sim.get("kind", "simulation")).item())
                for sim in sim_sources
            ]
        )
        aligned_ok = (
            np.array_equal(
                cached_aligned_data.get("channels"), fsmp_data.get("channels")
            )
            and np.array_equal(
                cached_aligned_data.get("edges"), sim_sources[0].get("edges")
            )
            and str(cached_aligned_data.get("processing", "")) == processing
            and str(
                np.asarray(
                    cached_aligned_data.get("fsmp_time_mode", "")
                ).item()
            )
            == args.fsmp_time_mode
            and np.array_equal(
                cached_aligned_data.get("simulation_labels"), source_labels
            )
            and np.array_equal(
                cached_aligned_data.get("simulation_kinds"), source_kinds
            )
            and all(
                key in cached_aligned_data
                for key in (
                    "run_dark_prepeak_pe",
                    "run_dark_rate_per_ns",
                    "run_dark_window_ns",
                    "run_dark_window_low_ns",
                    "run_dark_window_high_ns",
                    "run_dark_density_floor",
                    "run_dark_normalization_pe",
                    "run_dark_expected_pe",
                    "run_dark_removed_pe",
                )
            )
        )

    channel_pdf_needed = not args.opt.exists() or not aligned_ok
    summary_pdf_needed = not args.summary_opt.exists() or not aligned_ok
    stats_csv_needed = not args.stats_csv.exists() or not aligned_ok
    if not channel_pdf_needed:
        print(f"Skip existing channel PDF {args.opt}")
    if not summary_pdf_needed:
        print(f"Skip existing summary PDF {args.summary_opt}")
    if not stats_csv_needed:
        print(f"Skip existing comparison CSV {args.stats_csv}")
    if not channel_pdf_needed and not summary_pdf_needed and not stats_csv_needed:
        return

    data = None
    if aligned_ok:
        cached_mode = str(cached_aligned_data.get("alignment_mode", ""))
        if not channel_pdf_needed or cached_mode == "l1":
            data = cached_aligned_data
            print(f"Load cached aligned data from {args.aligned_data}.")
        else:
            print("Cached aligned data mode mismatch; recomputing.")

    if data is None:
        alignment = "l1" if channel_pdf_needed else "peak"
        if alignment == "peak":
            print("Only summary PDF is missing; use direct peak alignment.")
        data = build_channel_histograms_multi(
            fsmp_data,
            sim_sources,
            alignment=alignment,
            use_tts=not args.no_tts,
            rif_sigma_hamamatsu=args.rif_sigma_hamamatsu,
            rif_sigma_nnvt=args.rif_sigma_nnvt,
            dark_subtract=args.dark_subtract,
            dark_window=args.dark_window,
        )
        data["alignment_mode"] = np.asarray(alignment)
        np.savez_compressed(args.aligned_data, **data)
        print(f"Wrote {args.aligned_data}")

    if channel_pdf_needed:
        plot_data(data, args.opt, args.plot_range, args.run)
    if summary_pdf_needed:
        plot_summary_data(data, args.summary_opt, args.plot_range, args.run, args.z)
    if stats_csv_needed:
        write_comparison_stats_csv(args.stats_csv, data, args.run, args.z)


if __name__ == "__main__":
    main()
