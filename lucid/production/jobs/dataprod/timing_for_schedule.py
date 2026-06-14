#!/usr/bin/env python3
"""Derive per-config `seconds_per_event` for events_schedule blocks.

Walks `<base>/<detector>/config_NNNNNN/` directories produced by the
S3DF dataprod fan-out, parses the latest `job_*-<slurmid>.out` for each
config, and prints two things:

  1. A markdown table the user can drop into DataProduction_README.md
     (columns: config, partition, n_events, LUCiD median, PhotonSim/event,
     seconds_per_event).
  2. A JSON blob with one entry per config_id containing the same fields
     plus a ready-to-paste `events_schedule` block.

The per-event cost reported as `seconds_per_event` is

      median(LUCiD `Event total time:` lines) + PhotonSim_elapsed / n_events

That is intentionally slightly pessimistic — PhotonSim_elapsed includes
the one-off Geant4 init, which inflates the per-event number for small
n_events. The bias decreases with longer timing runs. The result is
suitable as a default for the events_schedule.seconds_per_event field;
revise downward if you've run a high-n timing pass.

Usage:
    python3 timing_for_schedule.py --base-dir <timing_root>/SK_like \\
        [--partition roma] [--target-seconds 3600] \\
        [--events-per-dataset 20000]

The --partition / --target-seconds / --events-per-dataset args are
metadata only — they appear in the printed `events_schedule` blocks so
you can paste them straight into the per-config JSONs.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

RE_JOB_OUT_NAME    = re.compile(r"^job_(\d+)-(\d+)\.out$")
RE_JOB_STARTED     = re.compile(r"^Job started:\s*(.+)$")
RE_JOB_ENDED       = re.compile(r"^Job ended:\s*(.+)$")
RE_PHOTONSIM_EXIT  = re.compile(r"PhotonSim exit code:\s*(\d+)\s*\(elapsed\s*([\d.]+)\s*s\)")
RE_GENIE_ELAPSED   = re.compile(r"^GENIE(?:\s+vertex\s+\d+)?\s+elapsed:\s*([\d.]+)\s*s")
RE_LUCID_EVENT     = re.compile(r"^\s+Event total time:\s*([\d.]+)s")

DATE_FMT = "%a %b %d %H:%M:%S %Z %Y"


def latest_job_out(config_dir: Path) -> Optional[Path]:
    """Return the most recent job_*-<slurmid>.out under config_dir.

    For nested configs (e.g. monoenergetic scans), recurses one level deep.
    """
    best: tuple[int, Optional[Path]] = (-1, None)
    for pat in ("job_*-*.out", "*MeV/job_*-*.out"):
        for p in config_dir.glob(pat):
            m = RE_JOB_OUT_NAME.match(p.name)
            if not m:
                continue
            slurm_id = int(m.group(2))
            if slurm_id > best[0]:
                best = (slurm_id, p)
    return best[1]


def parse_out(path: Path) -> dict:
    photonsim_elapsed = None
    genie_total = 0.0
    n_genie = 0
    event_totals: list[float] = []
    started = ended = None
    with path.open(errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_JOB_STARTED.match(line)
            if m:
                started = m.group(1).strip(); continue
            m = RE_JOB_ENDED.match(line)
            if m:
                ended = m.group(1).strip(); continue
            m = RE_PHOTONSIM_EXIT.search(line)
            if m:
                photonsim_elapsed = float(m.group(2)); continue
            m = RE_GENIE_ELAPSED.match(line)
            if m:
                genie_total += float(m.group(1)); n_genie += 1; continue
            m = RE_LUCID_EVENT.match(line)
            if m:
                event_totals.append(float(m.group(1))); continue
    wall = None
    if started and ended:
        try:
            wall = (datetime.strptime(ended, DATE_FMT)
                    - datetime.strptime(started, DATE_FMT)).total_seconds()
        except ValueError:
            pass
    return {
        "out_file": str(path),
        "n_events": len(event_totals),
        "lucid_event_median_s": statistics.median(event_totals) if event_totals else None,
        "lucid_event_mean_s":   statistics.fmean(event_totals)  if event_totals else None,
        "photonsim_elapsed_s":  photonsim_elapsed,
        "genie_elapsed_s":      genie_total if n_genie else None,
        "job_wall_s":           wall,
    }


def derive_seconds_per_event(stats: dict) -> Optional[float]:
    """median(LUCiD/event) + PhotonSim_elapsed / n_events.

    Returns None when n_events==0 (job didn't reach the event loop) so
    the caller can flag it instead of injecting a bogus number.
    """
    n = stats["n_events"]
    if not n:
        return None
    lucid = stats["lucid_event_median_s"] or 0.0
    ps    = (stats["photonsim_elapsed_s"] or 0.0) / n
    return lucid + ps


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", type=Path, required=True,
                   help="<scan_root>/<detector>/  (e.g. timing_20260515/SK_like/)")
    p.add_argument("--partition", type=str, default="roma",
                   help="Label only — appears in the markdown table and the "
                        "printed events_schedule blocks. Default: %(default)s.")
    p.add_argument("--target-seconds", type=float, default=3600,
                   help="target_seconds_per_job for the printed events_schedule "
                        "stubs. Default: %(default)s.")
    p.add_argument("--events-per-dataset", type=int, default=20000,
                   help="events_per_dataset for the printed events_schedule "
                        "stubs. Default: %(default)s.")
    p.add_argument("--json-only", action="store_true",
                   help="Skip the markdown table; print only JSON.")
    args = p.parse_args()

    if not args.base_dir.is_dir():
        print(f"error: base dir not found: {args.base_dir}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for cdir in sorted(args.base_dir.glob("config_*")):
        m = re.match(r"^config_(\d+)$", cdir.name)
        if not m:
            continue
        out = latest_job_out(cdir)
        if out is None:
            rows.append({"config_id": cdir.name, "status": "NO_OUT"})
            continue
        st = parse_out(out)
        s_per = derive_seconds_per_event(st)
        rows.append({
            "config_id":            cdir.name,
            "config_number":        int(m.group(1)),
            "status":               "OK" if s_per is not None else "EMPTY",
            "out_file":             st["out_file"],
            "n_events":             st["n_events"],
            "lucid_event_median_s": st["lucid_event_median_s"],
            "photonsim_elapsed_s":  st["photonsim_elapsed_s"],
            "genie_elapsed_s":      st["genie_elapsed_s"],
            "job_wall_s":           st["job_wall_s"],
            "seconds_per_event":    s_per,
        })

    # Markdown table
    if not args.json_only:
        print(f"## Per-config seconds_per_event on `{args.partition}` "
              f"(n_events ≈ {rows[0].get('n_events') if rows else '?'})")
        print()
        print("| Config | n_events | LUCiD median (s) | PhotonSim/event (s) | "
              "**seconds_per_event** | Job wall (s) |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["status"] != "OK":
                print(f"| {r['config_id']} | — | — | — | **—** | — "
                      f" *({r['status']})* |")
                continue
            ps_pe = (r["photonsim_elapsed_s"] or 0) / max(r["n_events"], 1)
            print(f"| {r['config_id']} | {r['n_events']} | "
                  f"{r['lucid_event_median_s']:.3f} | {ps_pe:.3f} | "
                  f"**{r['seconds_per_event']:.3f}** | "
                  f"{r['job_wall_s']:.0f} |")
        print()
        print(f"### Suggested `events_schedule` blocks "
              f"(target_seconds_per_job={int(args.target_seconds)}, "
              f"events_per_dataset={args.events_per_dataset})")
        print()
        for r in rows:
            if r["status"] != "OK":
                continue
            print(f"<!-- {r['config_id']} -->")
            print(json.dumps({
                "events_schedule": {
                    "events_per_dataset":    args.events_per_dataset,
                    "target_seconds_per_job": args.target_seconds,
                    "seconds_per_event":     round(r["seconds_per_event"], 3),
                }
            }, indent=2))
            print()

    # Machine-readable JSON dump on stderr (avoids polluting the table on stdout).
    payload = {"partition": args.partition, "configs": rows}
    if args.json_only:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
