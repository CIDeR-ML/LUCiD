"""CI diagnostic: per-event speed test for the PhotonSim → LUCiD pipeline.

This is a *diagnostic* — it always exits 0 (unless the harness itself
crashes). The output is a JSON report + Markdown summary intended to be
uploaded as a CI artifact. Comparing artifacts across commits surfaces
regressions in pipeline wall time without gating merges on a budget.

Approach: one ``run_job.py`` invocation per config at a fixed N; we
stream its stdout line-by-line, timestamping each line as it arrives,
and harvest:

* **Stage boundaries** ("=== Step 0/2/3 ===" headers, "Successfully
  wrote ...") to measure GENIE / PhotonSim / LUCiD wall times and
  isolate each init.
* **PhotonSim total internal elapsed** from the line
  ``PhotonSim exit code: 0 (elapsed Xs)``. Per-event average =
  internal_elapsed / N. PhotonSim doesn't print high-resolution per-
  event timing, so per-event distribution is not available without
  source instrumentation.
* **LUCiD per-event** from the writer's existing prints
  ``Event total time: X.XXs`` and ``Simulation completed in X.XXs``,
  plus per-batch ``Batch N save time: X.XXXs``. These give the full
  per-event distribution at 0.01 s resolution — variance, warm-up on
  event 1, and any tail dominate.

LUCiD ``init_s`` is computed as
``stage_wall - Σ per_event_total - Σ batch_save``, so it captures
everything before the first event (Python imports inside the writer,
ROOT scan, geometry build, lookup-table construction).

Usage:

    python3 ci_tests/speed_test.py \\
        --configs dataprod_01_mu.json dataprod_06_e_low_energy.json \\
        --n-events 20 \\
        --out-json /out/speed.json --out-md /out/speed.md

GENIE configs need ``GENIE_XSEC_FILE`` set in the environment to a
spline matching the tune declared in the config; see
``LUCiD/docs/QUICKSTART_DOCKER.md``. When the variable is unset the
GENIE configs are silently skipped (with a note in the report).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_JOB = REPO / "lucid" / "production" / "run_job.py"
CONFIGS_DIR = REPO / "lucid" / "production" / "configs"

DEFAULT_CONFIGS = [
    "GeV/01_mu.json",                 # single mu — baseline
    "Solar/01_e_low_energy.json",     # low-E electron — fast, EM-dominated
    "GeV/13_genie_numu_nue.json",     # GENIE numu+nue on water (G18_10a_02_11b)
]

# ── Stdout patterns from run_job.py + PhotonSim + LUCiD writer ────────
RE_STEP0 = re.compile(r"^=== Step 0: GENIE")
RE_STEP2 = re.compile(r"^=== Step 2:")
RE_STEP3 = re.compile(r"^=== Step 3:")
RE_STEP4 = re.compile(r"^=== Step 4:")
RE_PHOTONSIM_EXIT = re.compile(
    r"PhotonSim exit code:\s*\d+\s*\(elapsed\s*([\d.]+)\s*s\)"
)
RE_LUCID_EVENT_TOTAL = re.compile(r"^\s+Event total time:\s*([\d.]+)s")
RE_LUCID_EVENT_SIM = re.compile(r"^\s+Simulation completed in\s+([\d.]+)s")
RE_LUCID_STAGE = re.compile(r"^\s+\[timing\]\s+(\w+)\s+([\d.]+)s\s*$")
RE_LUCID_BATCH_SAVE = re.compile(r"^Batch\s+\d+\s+save time:\s*([\d.]+)s")
RE_LUCID_DONE = re.compile(r"^Successfully wrote\s+\d+\s+batches?")


def uses_genie(config_path: Path) -> bool:
    """True iff the config's primary source resolves to GENIE.

    Two schemas exist: single-vertex configs put ``primary_source`` at
    the top level; pile-up configs put it on each entry of ``vertices``.
    """
    cfg = json.loads(config_path.read_text())
    if cfg.get("primary_source") == "genie":
        return True
    return any((v.get("primary_source") == "genie") for v in (cfg.get("vertices") or []))


def run_with_line_timing(cmd: list[str], env: dict) -> tuple[int, list, float]:
    """Run ``cmd`` and stream stdout (merged with stderr).

    Returns (returncode, timeline, total_wall_s) where ``timeline`` is a
    list of ``(t_rel_s, line_str)``: ``t_rel_s`` is the wall time at
    which the line was *received* by Python, relative to subprocess
    launch.
    """
    t0 = time.perf_counter()
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    timeline = []
    assert p.stdout is not None
    for line in p.stdout:
        timeline.append((time.perf_counter() - t0, line.rstrip("\n")))
    rc = p.wait()
    return rc, timeline, time.perf_counter() - t0


def parse_timeline(timeline, n_events: int, has_genie: bool) -> dict:
    """Reduce the timestamped stdout stream to per-stage timing data."""
    t_step0 = t_step2 = t_step3 = t_step4 = None
    photonsim_internal_elapsed = None
    event_total: list[float] = []
    event_sim: list[float] = []
    batch_save: list[float] = []
    stage_times: dict[str, list[float]] = {}
    t_lucid_done = None

    for ts, line in timeline:
        if RE_STEP0.match(line):
            t_step0 = ts
        elif RE_STEP2.match(line):
            t_step2 = ts
        elif RE_STEP3.match(line):
            t_step3 = ts
        elif RE_STEP4.match(line):
            t_step4 = ts
        elif (m := RE_PHOTONSIM_EXIT.search(line)):
            photonsim_internal_elapsed = float(m.group(1))
        elif (m := RE_LUCID_STAGE.match(line)):
            stage_times.setdefault(m.group(1), []).append(float(m.group(2)))
        elif (m := RE_LUCID_EVENT_TOTAL.match(line)):
            event_total.append(float(m.group(1)))
        elif (m := RE_LUCID_EVENT_SIM.match(line)):
            event_sim.append(float(m.group(1)))
        elif (m := RE_LUCID_BATCH_SAVE.match(line)):
            batch_save.append(float(m.group(1)))
        elif RE_LUCID_DONE.match(line):
            t_lucid_done = ts

    out: dict = {"n_events": n_events, "stages": {}}

    # GENIE: Step 0 → Step 2.
    if has_genie and t_step0 is not None and t_step2 is not None:
        out["stages"]["genie"] = {
            "wall_s": t_step2 - t_step0,
            "per_event_avg_s": (t_step2 - t_step0) / n_events,
        }

    # PhotonSim: Step 2 → Step 3 (subprocess overhead included), with
    # internal elapsed parsed from PhotonSim's own timing line.
    if t_step2 is not None and t_step3 is not None:
        ps_wall = t_step3 - t_step2
        ps = {
            "wall_s": ps_wall,
            "internal_elapsed_s": photonsim_internal_elapsed,
        }
        if photonsim_internal_elapsed is not None:
            ps["subprocess_overhead_s"] = ps_wall - photonsim_internal_elapsed
            ps["per_event_avg_s"] = photonsim_internal_elapsed / n_events
        out["stages"]["photonsim"] = ps

    # LUCiD: Step 3 → "Successfully wrote ..." (or Step 4 fallback).
    lucid_end = t_lucid_done if t_lucid_done is not None else t_step4
    if t_step3 is not None and lucid_end is not None:
        lc_wall = lucid_end - t_step3
        per_event_sum = sum(event_total)
        save_sum = sum(batch_save)
        lc = {
            "wall_s": lc_wall,
            "per_event_total_s": event_total,
            "per_event_sim_s": event_sim,
            "batch_save_s": batch_save,
            "per_stage_s": stage_times,
            # init = wall - per-event work - save = preamble (imports,
            # ROOT scan, geometry, lookup tables).
            "init_s": lc_wall - per_event_sum - save_sum,
        }
        if event_total:
            lc["per_event_total_mean_s"] = per_event_sum / len(event_total)
            lc["per_event_total_min_s"] = min(event_total)
            lc["per_event_total_max_s"] = max(event_total)
        if event_sim:
            lc["per_event_sim_mean_s"] = sum(event_sim) / len(event_sim)
        out["stages"]["lucid"] = lc

    return out


def measure_config(
    config_path: Path,
    *,
    n_events: int,
    work_root: Path,
    photonsim_bin: str | None,
    log_path: Path | None = None,
) -> dict:
    """Run the pipeline once at N events and harvest per-event timings.

    If ``log_path`` is given, write the raw timestamped stdout there for
    later inspection (PAD_SIZE, photon counts, geometry banner, anything
    the parser doesn't capture).
    """
    is_genie = uses_genie(config_path)
    if is_genie and not os.environ.get("GENIE_XSEC_FILE"):
        return {"skipped": "GENIE_XSEC_FILE not set"}

    out_dir = work_root / config_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(RUN_JOB),
        "--config", str(config_path),
        "--n-events", str(n_events),
        "--job-id", "1",
        "--master-seed", "42",
        "--output-dir", str(out_dir),
    ]
    env = os.environ.copy()
    if photonsim_bin:
        env["PHOTONSIM_BIN"] = photonsim_bin

    rc, timeline, total_wall = run_with_line_timing(cmd, env)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            for ts, line in timeline:
                f.write(f"{ts:8.3f}  {line}\n")

    parsed = parse_timeline(timeline, n_events, has_genie=is_genie)
    parsed["is_genie"] = is_genie
    parsed["wall_total_s"] = total_wall
    parsed["returncode"] = rc
    parsed["ok"] = rc == 0
    if log_path is not None:
        parsed["raw_log"] = str(log_path)
    if rc != 0:
        # Tail of the stream so the report is self-diagnosing.
        parsed["stderr_tail"] = [ln for _, ln in timeline[-12:]]
    return parsed


def write_markdown(report: dict, out_md: Path) -> None:
    lines: list[str] = []
    m = report["meta"]
    lines.append("# Pipeline speed test\n")
    lines.append(f"- generated:  `{m['timestamp']}`")
    lines.append(f"- platform:   `{m['platform']}`")
    lines.append(f"- python:     `{m['python']}`")
    lines.append(f"- n events:   `{m['n_events']}`")
    lines.append("")
    lines.append("Single-run-per-config harvest of per-event timings from "
                 "the writer's existing per-event prints (LUCiD has 10 ms "
                 "resolution; PhotonSim only reports its total internal "
                 "elapsed time, so per-event distribution is not available "
                 "for that stage). All times in seconds.\n")

    # Per-stage summary table.
    lines.append("| config | stage | wall (s) | init (s) | per-event mean (s) | per-event min/max (s) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cfg, data in report["configs"].items():
        if data.get("skipped"):
            lines.append(f"| `{cfg}` | _skipped: {data['skipped']}_ | — | — | — | — |")
            continue
        if data.get("error"):
            lines.append(f"| `{cfg}` | _error: {data['error']}_ | — | — | — | — |")
            continue
        if not data.get("ok"):
            lines.append(f"| `{cfg}` | _failed (rc={data.get('returncode')})_ | — | — | — | — |")
            continue
        for stage, sd in data["stages"].items():
            wall = sd.get("wall_s")
            if stage == "lucid":
                init = sd.get("init_s")
                mean = sd.get("per_event_total_mean_s")
                lo = sd.get("per_event_total_min_s")
                hi = sd.get("per_event_total_max_s")
                mm = f"{lo:.3f} / {hi:.3f}" if lo is not None else "—"
                lines.append(
                    f"| `{cfg}` | {stage} | "
                    f"{_fmt(wall)} | {_fmt(init)} | "
                    f"{_fmt(mean)} | {mm} |"
                )
            elif stage == "photonsim":
                init_elapsed = sd.get("internal_elapsed_s")
                mean = sd.get("per_event_avg_s")
                lines.append(
                    f"| `{cfg}` | {stage} | "
                    f"{_fmt(wall)} | "
                    f"{_fmt(sd.get('subprocess_overhead_s'))} (subproc) | "
                    f"{_fmt(mean)} | _no per-event distribution_ |"
                )
            else:  # genie
                mean = sd.get("per_event_avg_s")
                lines.append(
                    f"| `{cfg}` | {stage} | {_fmt(wall)} | — | {_fmt(mean)} | _no per-event distribution_ |"
                )
    lines.append("")

    # Per-stage LUCiD breakdown (event 0 vs the rest, per stage, per config).
    # Surfaces JIT compile cost on event 0 and ranks stages by mean cost.
    any_stages = any(
        (data.get("stages", {}).get("lucid", {}) or {}).get("per_stage_s")
        for data in report["configs"].values()
        if not (data.get("skipped") or data.get("error") or not data.get("ok"))
    )
    if any_stages:
        lines.append("## Per-stage LUCiD breakdown\n")
        lines.append("Event 0 vs events 1+ for each in-process LUCiD sub-stage. "
                     "Stages should sum to roughly the per-event total; a large "
                     "event-0 vs events-1+ gap on `simulate` indicates JIT "
                     "compile cost on the first event.\n")
        lines.append("| config | stage | event 0 (s) | mean 1+ (s) | min 1+ (s) | max 1+ (s) | n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for cfg, data in report["configs"].items():
            if data.get("skipped") or data.get("error") or not data.get("ok"):
                continue
            lc = data.get("stages", {}).get("lucid") or {}
            stage_map = lc.get("per_stage_s") or {}
            if not stage_map:
                continue
            # Stable order: emit in the source-code order if present, then any extras alphabetically.
            canonical = ["root_read", "preprocess", "simulate", "post_jax", "meta_contain"]
            ordered = [s for s in canonical if s in stage_map]
            ordered += sorted(s for s in stage_map if s not in canonical)
            for stage in ordered:
                ts = stage_map[stage]
                if not ts:
                    continue
                ev0 = ts[0]
                rest = ts[1:]
                if rest:
                    mean = sum(rest) / len(rest)
                    lo, hi = min(rest), max(rest)
                else:
                    mean = lo = hi = None
                lines.append(
                    f"| `{cfg}` | {stage} | {_fmt(ev0)} | {_fmt(mean)} | "
                    f"{_fmt(lo)} | {_fmt(hi)} | {len(ts)} |"
                )
        lines.append("")

    # Per-event LUCiD detail (where we have it).
    lines.append("## Per-event LUCiD timing\n")
    lines.append("Event total time (s, includes ROOT read + sim + bookkeeping per event):\n")
    for cfg, data in report["configs"].items():
        if data.get("skipped") or data.get("error") or not data.get("ok"):
            continue
        lc = data.get("stages", {}).get("lucid")
        if not lc:
            continue
        ts = lc.get("per_event_total_s") or []
        if not ts:
            continue
        per_line = "  ".join(f"{t:.3f}" for t in ts)
        lines.append(f"- `{cfg}`:  {per_line}")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n")


def _fmt(x):
    if x is None:
        return "—"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--configs", nargs="*", default=None,
                   help=f"Config filenames under {CONFIGS_DIR.relative_to(REPO)}. "
                        f"Default: {DEFAULT_CONFIGS}")
    p.add_argument("--n-events", type=int, default=20,
                   help="Events to simulate per config (default 20).")
    p.add_argument("--photonsim-bin", default=os.environ.get("PHOTONSIM_BIN"),
                   help="Path to PhotonSim binary. Defaults to $PHOTONSIM_BIN.")
    p.add_argument("--work-dir", default=None,
                   help="Where to put per-config output dirs. Default: a tempdir.")
    p.add_argument("--keep-work", action="store_true",
                   help="Don't delete the work dir on exit (useful for debugging).")
    p.add_argument("--out-json", required=True, type=Path,
                   help="Path for the JSON report.")
    p.add_argument("--out-md", type=Path, default=None,
                   help="Optional Markdown summary path.")
    p.add_argument("--logs-dir", type=Path, default=None,
                   help="Directory for per-config raw stdout logs (one "
                        "<config_stem>.log per config, with leading "
                        "wall-time-relative-to-launch column). Defaults to "
                        "the parent of --out-json. Pass --no-logs to skip.")
    p.add_argument("--no-logs", action="store_true",
                   help="Don't save raw stdout logs (default is to save them).")
    args = p.parse_args(argv)

    configs = list(args.configs) if args.configs else list(DEFAULT_CONFIGS)

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="lucid_speed_"))
    work_root.mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "n_events": args.n_events,
            "photonsim_bin": args.photonsim_bin,
            "configs": configs,
        },
        "configs": {},
    }

    if args.no_logs:
        logs_dir = None
    else:
        logs_dir = args.logs_dir or args.out_json.parent
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== speed_test: {len(configs)} configs × N={args.n_events} ===")
    print(f"work dir: {work_root}")
    if logs_dir is not None:
        print(f"logs dir: {logs_dir}")

    for name in configs:
        cfg_path = CONFIGS_DIR / name
        if not cfg_path.is_file():
            print(f"  {name}: NOT FOUND, skipping")
            report["configs"][name] = {"skipped": "config file not found"}
            continue
        print(f"  measuring {name} ...", flush=True)
        try:
            log_path = (logs_dir / f"{cfg_path.stem}.log") if logs_dir else None
            report["configs"][name] = measure_config(
                cfg_path,
                n_events=args.n_events,
                work_root=work_root,
                photonsim_bin=args.photonsim_bin,
                log_path=log_path,
            )
        except Exception as e:
            report["configs"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"    error: {e}", file=sys.stderr)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
        print(f"wrote {args.out_md}")

    print("\n=== summary ===")
    for name, data in report["configs"].items():
        if data.get("skipped") or data.get("error") or not data.get("ok"):
            note = data.get("skipped") or data.get("error") or f"failed (rc={data.get('returncode')})"
            print(f"  {name}: {note}")
            continue
        print(f"  {name}:")
        for stage, sd in data["stages"].items():
            if stage == "lucid":
                mean = sd.get("per_event_total_mean_s") or 0
                init = sd.get("init_s") or 0
                lo = sd.get("per_event_total_min_s") or 0
                hi = sd.get("per_event_total_max_s") or 0
                print(f"    {stage}: init={init:.2f}s  per_event mean={mean*1000:.1f}ms "
                      f"(min {lo*1000:.1f}, max {hi*1000:.1f})")
            elif stage == "photonsim":
                ie = sd.get("internal_elapsed_s") or 0
                avg = sd.get("per_event_avg_s") or 0
                print(f"    {stage}: internal={ie:.2f}s  per_event avg={avg*1000:.1f}ms (no distribution)")
            else:
                wall = sd.get("wall_s") or 0
                avg = sd.get("per_event_avg_s") or 0
                print(f"    {stage}: wall={wall:.2f}s  per_event avg={avg*1000:.1f}ms")

    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
