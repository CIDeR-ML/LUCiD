#!/usr/bin/env python3
"""Per-config timing + output-size report for an S3DF dataprod sweep.

Walks `<base>/<detector>/config_NNNNNN/` directories produced
by `submit_all_configs.sh` and, for each config, harvests:

* From every `job_*-<slurm_id>.out` in the dir — pooled across jobs:
    * Total job wall (from sbatch `Job started`/`Job ended` `date` lines).
    * PhotonSim internal elapsed (from the
      `PhotonSim exit code: 0 (elapsed Xs)` line).
    * LUCiD per-event totals (`Event total time: X.XXs`).
    * LUCiD batch save times (`Batch N save time: X.XXXs`).
    * Step markers (Step 0/2/3/4) — used only to flag missing stages.
* From `sacct -j <slurm_ids>` if available — Elapsed and MaxRSS per job.
* From the on-disk `{sensor,inst,seg,labl}/wc_*.h5` tree — total bytes,
  per-stream bytes, per-event bytes.

The job-wall and PhotonSim numbers are *per job*: we report mean / min /
max across the N jobs of that config. The LUCiD per-event distribution
is *pooled* across all events of all jobs (so 2 jobs × 100 events = 200
samples).

Stage walls that current SLURM .out cannot recover (GENIE step, LUCiD-
section wall, LUCiD init) are reported as `null` and noted in the
markdown. To fill those, the sbatch wrapper would need to prepend
relative timestamps to each line (e.g. `... | ts -s "%.s"`).

Usage:
    # One config dir directly
    python3 dataprod_metrics_report.py --config-dir <out>/SK_like/config_000001

    # All configs under a base
    python3 dataprod_metrics_report.py --base-dir <out>/SK_like

    # Both forms accept --out-json / --out-md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

# ---------- regex catalogue -------------------------------------------------

RE_JOB_OUT_NAME    = re.compile(r"^job_(\d+)-(\d+)\.out$")
RE_JOB_STARTED     = re.compile(r"^Job started:\s*(.+)$")
RE_JOB_ENDED       = re.compile(r"^Job ended:\s*(.+)$")
RE_STEP            = re.compile(r"^=== Step (\d+):")
RE_PHOTONSIM_EXIT  = re.compile(r"PhotonSim exit code:\s*(\d+)\s*\(elapsed\s*([\d.]+)\s*s\)")
# Matches both the single-vertex `GENIE elapsed: Xs` and the pile-up
# per-vertex `GENIE vertex N elapsed: Xs` form. We sum across whatever
# matches in a single job.
RE_GENIE_ELAPSED   = re.compile(r"^GENIE(?:\s+vertex\s+\d+)?\s+elapsed:\s*([\d.]+)\s*s")
RE_LUCID_EVENT     = re.compile(r"^\s+Event total time:\s*([\d.]+)s")
RE_LUCID_BATCH_SAV = re.compile(r"^Batch\s+\d+\s+save time:\s*([\d.]+)s")
RE_LUCID_STAGE     = re.compile(r"^\s+\[timing\]\s+(\w+)\s+([\d.]+)s\s*$")
RE_LUCID_DONE      = re.compile(r"^Successfully wrote\s+\d+\s+batches?")

# `date` default on the SLURM nodes: e.g. "Mon Apr 20 18:57:42 PDT 2026"
DATE_FMT = "%a %b %d %H:%M:%S %Z %Y"


# ---------- per-job parser --------------------------------------------------


def parse_job_out(path: Path) -> dict:
    """Reduce one SLURM stdout file to a per-job timing dict."""
    started_str = ended_str = None
    photonsim_rc = None
    photonsim_elapsed = None
    genie_elapsed_total = 0.0  # sum of GENIE elapsed lines (single or per-vertex)
    n_genie_lines = 0
    event_totals: list[float] = []
    batch_saves: list[float] = []
    sub_stage_times: dict[str, list[float]] = {}
    steps_seen: set[int] = set()
    lucid_done = False

    with path.open("r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_JOB_STARTED.match(line)
            if m:
                started_str = m.group(1).strip()
                continue
            m = RE_JOB_ENDED.match(line)
            if m:
                ended_str = m.group(1).strip()
                continue
            m = RE_STEP.match(line)
            if m:
                steps_seen.add(int(m.group(1)))
                continue
            m = RE_PHOTONSIM_EXIT.search(line)
            if m:
                photonsim_rc = int(m.group(1))
                photonsim_elapsed = float(m.group(2))
                continue
            m = RE_GENIE_ELAPSED.match(line)
            if m:
                genie_elapsed_total += float(m.group(1))
                n_genie_lines += 1
                continue
            m = RE_LUCID_EVENT.match(line)
            if m:
                event_totals.append(float(m.group(1)))
                continue
            m = RE_LUCID_BATCH_SAV.match(line)
            if m:
                batch_saves.append(float(m.group(1)))
                continue
            m = RE_LUCID_STAGE.match(line)
            if m:
                sub_stage_times.setdefault(m.group(1), []).append(float(m.group(2)))
                continue
            if RE_LUCID_DONE.match(line):
                lucid_done = True

    job_wall_s = None
    if started_str and ended_str:
        try:
            t0 = datetime.strptime(started_str, DATE_FMT)
            t1 = datetime.strptime(ended_str, DATE_FMT)
            job_wall_s = (t1 - t0).total_seconds()
        except ValueError:
            pass  # unrecognized tz/format — leave as None

    m = RE_JOB_OUT_NAME.match(path.name)
    job_index = int(m.group(1)) if m else None
    slurm_id  = int(m.group(2)) if m else None

    return {
        "out_file": str(path),
        "job_index": job_index,
        "slurm_id": slurm_id,
        "job_wall_s": job_wall_s,
        "photonsim_rc": photonsim_rc,
        "photonsim_elapsed_s": photonsim_elapsed,
        "genie_elapsed_s": genie_elapsed_total if n_genie_lines else None,
        "lucid_event_total_s": event_totals,
        "lucid_batch_save_s": batch_saves,
        "lucid_substage_s": sub_stage_times,
        "steps_seen": sorted(steps_seen),
        "lucid_done": lucid_done,
    }


# ---------- sacct enrichment ------------------------------------------------


def fetch_sacct(slurm_ids: list[int]) -> dict[int, dict]:
    """Return {slurm_id: {elapsed_s, maxrss_mb}} for the given ids.

    Best-effort: returns {} if sacct is unavailable or the call fails.
    Reads the .batch step (where MaxRSS is recorded) and prefers it, but
    falls back to the parent step's Elapsed when the .batch row is absent.
    """
    if not slurm_ids:
        return {}
    cmd = [
        "sacct", "-j", ",".join(str(i) for i in slurm_ids),
        "--format=JobID,Elapsed,MaxRSS,State,ExitCode",
        "--parsable2", "--noheader",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}

    out: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jobid, elapsed, maxrss, state, _exit = parts[:5]
        # JobID may be "12345" (parent), "12345.batch" (the actual step), etc.
        base = jobid.split(".", 1)[0]
        try:
            sid = int(base)
        except ValueError:
            continue
        rec = out.setdefault(sid, {})
        # State + Elapsed live on the parent row.
        if "." not in jobid:
            rec["state"] = state
            rec["elapsed"] = elapsed
            rec["elapsed_s"] = _hms_to_seconds(elapsed)
        # MaxRSS lives on the .batch row.
        if jobid.endswith(".batch") and maxrss:
            rec["maxrss_raw"] = maxrss
            rec["maxrss_mb"] = _rss_to_mb(maxrss)
    return out


def _hms_to_seconds(s: str) -> float | None:
    if not s:
        return None
    # Optional days: "D-HH:MM:SS" or "HH:MM:SS" or "MM:SS.FFF"
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        try:
            days = int(d)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts_f) == 3:
        h, m, sec = parts_f
    elif len(parts_f) == 2:
        h, m, sec = 0.0, parts_f[0], parts_f[1]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


def _rss_to_mb(s: str) -> float | None:
    """sacct MaxRSS is "1234K", "1234M", "1.23G", or bare digits (KiB)."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    suffix = s[-1].upper()
    if suffix in {"K", "M", "G", "T"}:
        try:
            v = float(s[:-1])
        except ValueError:
            return None
        scale = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024 * 1024}[suffix]
        return v * scale
    try:
        return float(s) / 1024.0  # bare = KiB
    except ValueError:
        return None


# ---------- output-size accounting -----------------------------------------


def measure_output_sizes(config_dir: Path) -> dict:
    """Return per-stream + total bytes for a config dir.

    Counts all `wc_*.h5` under `{sensor,inst,seg,labl}/` plus the README.
    """
    streams = ["sensor", "inst", "seg", "labl"]
    per_stream: dict[str, dict] = {}
    grand = 0
    for s in streams:
        sd = config_dir / s
        if not sd.is_dir():
            per_stream[s] = {"bytes": 0, "files": 0}
            continue
        files = sorted(sd.glob("wc_*.h5"))
        b = sum(p.stat().st_size for p in files)
        per_stream[s] = {
            "bytes": b,
            "files": len(files),
            "file_list": [p.name for p in files],
        }
        grand += b

    # Misc on-disk artifacts (sbatch scripts, READMEs, .out, .err) — count
    # them separately so they don't confuse the per-event size calc.
    misc = 0
    for entry in config_dir.iterdir():
        if entry.is_file():
            misc += entry.stat().st_size

    return {
        "h5_total_bytes": grand,
        "per_stream": per_stream,
        "misc_bytes": misc,  # sbatch/README/.out/.err sit here
    }


# ---------- combine across jobs --------------------------------------------


def stat_summary(xs: list[float]) -> dict | None:
    if not xs:
        return None
    out = {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
    }
    if len(xs) >= 2:
        out["stdev"] = statistics.stdev(xs)
    if len(xs) >= 5:
        # rough percentiles
        s = sorted(xs)
        out["p50"] = s[len(s) // 2]
        out["p95"] = s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]
    return out


def combine_config(config_dir: Path, n_events_per_job_hint: int | None) -> dict:
    """Collect all `job_*.out` files in `config_dir`, aggregate, add sizes."""
    out_files = sorted(config_dir.glob("job_*-*.out"))
    jobs = [parse_job_out(p) for p in out_files]
    slurm_ids = [j["slurm_id"] for j in jobs if j["slurm_id"] is not None]
    sacct_map = fetch_sacct(slurm_ids)
    for j in jobs:
        j["sacct"] = sacct_map.get(j["slurm_id"]) if j["slurm_id"] is not None else None

    pooled_event = [t for j in jobs for t in j["lucid_event_total_s"]]
    pooled_save  = [t for j in jobs for t in j["lucid_batch_save_s"]]
    job_walls    = [j["job_wall_s"] for j in jobs if j["job_wall_s"] is not None]
    ps_elapsed   = [j["photonsim_elapsed_s"] for j in jobs if j["photonsim_elapsed_s"] is not None]
    genie_elapsed = [j["genie_elapsed_s"] for j in jobs if j.get("genie_elapsed_s") is not None]

    # Per-job "leftover" = job_wall − GENIE − PhotonSim − Σevent_total − Σbatch_save.
    # GENIE and PhotonSim are self-reported; events + saves come from the
    # LUCiD writer. What remains is LUCiD init + JAX JIT warmup
    # (`_warmup_buckets` compiles each kernel once at startup) plus a
    # tiny amount of subprocess/Python overhead.
    leftover_per_job = []
    for j in jobs:
        if j["job_wall_s"] is None or j["photonsim_elapsed_s"] is None:
            continue
        ev_sum = sum(j["lucid_event_total_s"])
        sv_sum = sum(j["lucid_batch_save_s"])
        ge = j.get("genie_elapsed_s") or 0.0
        leftover_per_job.append(
            j["job_wall_s"] - ge - j["photonsim_elapsed_s"] - ev_sum - sv_sum
        )
    sacct_walls  = [j["sacct"]["elapsed_s"] for j in jobs
                    if j.get("sacct") and j["sacct"].get("elapsed_s") is not None]
    rss_mb       = [j["sacct"]["maxrss_mb"] for j in jobs
                    if j.get("sacct") and j["sacct"].get("maxrss_mb") is not None]

    # Per-event PhotonSim avg, computed per job then summarized across jobs.
    n_per_job = n_events_per_job_hint
    per_job_ps_per_event = []
    if n_per_job:
        per_job_ps_per_event = [e / n_per_job for e in ps_elapsed]

    # Output size + per-event bytes (pooled across jobs).
    sizes = measure_output_sizes(config_dir)
    n_events_total = len(pooled_event) if pooled_event else (
        len(jobs) * n_per_job if n_per_job else 0
    )
    if n_events_total:
        sizes["bytes_per_event_total"] = sizes["h5_total_bytes"] / n_events_total
        sizes["per_stream_bytes_per_event"] = {
            s: d["bytes"] / n_events_total for s, d in sizes["per_stream"].items()
        }

    # Status / completeness flags. Authoritative source is sacct's State
    # when available — SLURM stdout sometimes gets truncated on COMPLETED
    # jobs (last batch flush after the writer exits), so the absence of
    # the "Successfully wrote" marker is only a real incompleteness
    # signal when sacct disagrees (FAILED / TIMEOUT / OUT_OF_MEMORY).
    incomplete_reasons = []
    for j in jobs:
        sacct_state = (j.get("sacct") or {}).get("state")
        if sacct_state and sacct_state != "COMPLETED":
            incomplete_reasons.append(
                f"job_{j['job_index']:06d}: sacct state={sacct_state}"
            )
        elif not sacct_state and not j["lucid_done"]:
            # No sacct evidence either way — fall back to the stdout marker.
            incomplete_reasons.append(
                f"job_{j['job_index']:06d}: no 'Successfully wrote' marker (and sacct unavailable)"
            )
        if j["photonsim_rc"] not in (0, None):
            incomplete_reasons.append(
                f"job_{j['job_index']:06d}: PhotonSim rc={j['photonsim_rc']}"
            )

    return {
        "config_dir": str(config_dir),
        "n_jobs": len(jobs),
        "n_events_per_job": n_per_job,
        "n_events_total_observed": len(pooled_event),
        "incomplete_reasons": incomplete_reasons,
        "timing": {
            "job_wall_s":            stat_summary(job_walls),
            "sacct_wall_s":          stat_summary(sacct_walls),
            "genie_elapsed_s":       stat_summary(genie_elapsed),
            "photonsim_elapsed_s":   stat_summary(ps_elapsed),
            "photonsim_per_event_s": stat_summary(per_job_ps_per_event),
            "lucid_event_total_s":   stat_summary(pooled_event),
            "lucid_batch_save_s":    stat_summary(pooled_save),
            "leftover_init_warmup_s": stat_summary(leftover_per_job),
            "sacct_maxrss_mb":       stat_summary(rss_mb),
        },
        "sizes": sizes,
        "jobs": jobs,
    }


# ---------- markdown ---------------------------------------------------------


def _fmt_n(x, kind="s"):
    if x is None:
        return "—"
    if kind == "MB":
        return f"{x:.1f}"
    if kind == "MB-bytes":
        # Render bytes as MiB
        return f"{x / (1024 * 1024):.2f}"
    if kind == "B":
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) < 10:
        return f"{x:.3f}"
    return f"{x:.2f}"


def _fmt_stat(s, *, kind="s", show="mean_minmax"):
    if not s:
        return "—"
    if show == "mean_minmax":
        return f"{_fmt_n(s['mean'], kind)} ({_fmt_n(s['min'], kind)} / {_fmt_n(s['max'], kind)})"
    if show == "mean_p":
        p50 = s.get("p50"); p95 = s.get("p95")
        return f"{_fmt_n(s['mean'], kind)} (p50 {_fmt_n(p50, kind)}, p95 {_fmt_n(p95, kind)})"
    return _fmt_n(s["mean"], kind)


def write_markdown(report: dict, out_md: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Dataprod metrics — {report['meta']['root']}")
    lines.append(f"_generated {report['meta']['timestamp']}_")
    lines.append("")
    lines.append("Per-job stage walls are summarized as **mean (min / max)** "
                 "across the N jobs of each config. LUCiD per-event timings "
                 "are **pooled across all events of all jobs**.")
    lines.append("")
    lines.append("**`leftover` column** = `job_wall − GENIE − PhotonSim − "
                 "Σ(event_total) − Σ(batch_save)`. With GENIE and PhotonSim "
                 "now self-reporting their elapsed wall, this column is "
                 "purely **LUCiD init + JAX JIT warmup** (`_warmup_buckets` "
                 "compiles each kernel once at startup, ~80–120 s on CPU) "
                 "plus a tiny amount of subprocess/Python overhead.")
    lines.append("")

    # Timing table
    lines.append("## Timing")
    lines.append("")
    lines.append("| config | jobs | events | job wall (s) | GENIE elapsed (s) | PhotonSim elapsed (s) | PhotonSim/event (s) | LUCiD/event (s) | leftover (s) | sacct MaxRSS (MB) |")
    lines.append("|---|---:|---:|---|---|---|---|---|---|---|")
    for cfg, data in report["configs"].items():
        if "error" in data:
            lines.append(f"| `{cfg}` | _error: {data['error']}_ | | | | | | | | |")
            continue
        t = data["timing"]
        lines.append(
            f"| `{cfg}` | {data['n_jobs']} | "
            f"{data['n_events_total_observed']} | "
            f"{_fmt_stat(t['job_wall_s'])} | "
            f"{_fmt_stat(t.get('genie_elapsed_s'))} | "
            f"{_fmt_stat(t['photonsim_elapsed_s'])} | "
            f"{_fmt_stat(t['photonsim_per_event_s'])} | "
            f"{_fmt_stat(t['lucid_event_total_s'], show='mean_p')} | "
            f"{_fmt_stat(t.get('leftover_init_warmup_s'))} | "
            f"{_fmt_stat(t['sacct_maxrss_mb'], kind='MB')} |"
        )
    lines.append("")

    # Sizes table
    lines.append("## Output size")
    lines.append("")
    lines.append("| config | total H5 (MiB) | sensor (MiB) | inst (MiB) | seg (MiB) | labl (MiB) | bytes / event |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cfg, data in report["configs"].items():
        if "error" in data:
            continue
        s = data["sizes"]
        per = s["per_stream"]
        bpe = s.get("bytes_per_event_total")
        lines.append(
            f"| `{cfg}` | "
            f"{_fmt_n(s['h5_total_bytes'], 'MB-bytes')} | "
            f"{_fmt_n(per['sensor']['bytes'], 'MB-bytes')} | "
            f"{_fmt_n(per['inst']['bytes'], 'MB-bytes')} | "
            f"{_fmt_n(per['seg']['bytes'], 'MB-bytes')} | "
            f"{_fmt_n(per['labl']['bytes'], 'MB-bytes')} | "
            f"{_fmt_n(bpe, 'B') if bpe else '—'} |"
        )
    lines.append("")

    # Incomplete-job warnings
    any_incomplete = any(
        data.get("incomplete_reasons") for data in report["configs"].values()
        if "error" not in data
    )
    if any_incomplete:
        lines.append("## Incomplete jobs")
        lines.append("")
        for cfg, data in report["configs"].items():
            if data.get("incomplete_reasons"):
                lines.append(f"- `{cfg}`:")
                for r in data["incomplete_reasons"]:
                    lines.append(f"  - {r}")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")


# ---------- entry point -----------------------------------------------------


def discover_configs(base_dir: Path) -> list[Path]:
    return sorted(p for p in base_dir.glob("config_*") if p.is_dir())


def hint_n_events_per_job(config_dir: Path) -> int | None:
    """Best-effort recovery of n_events_per_job from the README."""
    readme = config_dir / "README.md"
    if not readme.is_file():
        return None
    m = re.search(r"Events/job\*\*:\s*(\d+)", readme.read_text())
    if m:
        return int(m.group(1))
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config-dir", type=Path,
                   help="Path to a single config_NNNNNN/ directory.")
    g.add_argument("--base-dir", type=Path,
                   help="Path under which config_NNNNNN/ live "
                        "(e.g. <out>/SK_like).")
    p.add_argument("--out-json", type=Path, default=None,
                   help="JSON report path. Default: <base>/metrics_report.json.")
    p.add_argument("--out-md", type=Path, default=None,
                   help="Markdown report path. Default: <base>/metrics_report.md.")
    p.add_argument("--n-events-per-job", type=int, default=None,
                   help="Override n_events_per_job hint (defaults to README).")
    args = p.parse_args(argv)

    if args.config_dir:
        roots = [args.config_dir.resolve()]
        report_root = args.config_dir.resolve()
    else:
        roots = discover_configs(args.base_dir.resolve())
        report_root = args.base_dir.resolve()
        if not roots:
            print(f"No config_* dirs under {args.base_dir}", flush=True)
            return 1

    report = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "root": str(report_root),
            "n_configs": len(roots),
        },
        "configs": {},
    }

    for cfg_dir in roots:
        name = cfg_dir.name
        try:
            n_per = args.n_events_per_job or hint_n_events_per_job(cfg_dir)
            report["configs"][name] = combine_config(cfg_dir, n_per)
        except Exception as e:  # pragma: no cover - report and continue
            report["configs"][name] = {"error": f"{type(e).__name__}: {e}"}

    out_json = args.out_json or (report_root / "metrics_report.json")
    out_md   = args.out_md   or (report_root / "metrics_report.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report, out_md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print()

    # Tiny stdout summary so a CI run is self-diagnosing.
    for cfg, data in report["configs"].items():
        if "error" in data:
            print(f"  {cfg}: {data['error']}")
            continue
        t = data["timing"]
        s = data["sizes"]
        wall = t["job_wall_s"]["mean"] if t["job_wall_s"] else None
        ps   = t["photonsim_elapsed_s"]["mean"] if t["photonsim_elapsed_s"] else None
        ev   = t["lucid_event_total_s"]
        bpe  = s.get("bytes_per_event_total")
        print(f"  {cfg}: jobs={data['n_jobs']} events={data['n_events_total_observed']}  "
              f"job_wall_mean={_fmt_n(wall)}s  ps_mean={_fmt_n(ps)}s  "
              f"lucid/ev={_fmt_n(ev['mean']) if ev else '—'}s  "
              f"out={_fmt_n(s['h5_total_bytes'], 'MB-bytes')} MiB  "
              f"({_fmt_n(bpe, 'B') if bpe else '—'} B/ev)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
