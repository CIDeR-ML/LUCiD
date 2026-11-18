"""Read the timing lines a production job prints to its stdout.

The lines come from the sbatch wrapper (``Job started`` / ``Job ended``, rendered in
``platform/adapters.py``), ``dataprod/run_job.py`` (``=== Step N:``, ``GENIE elapsed``,
``PhotonSim exit code``) and the event loops in ``dataprod/drivers.py`` (``Event total time``,
``Simulation completed in``, ``[timing] <stage>``, ``Batch N save time``, ``Successfully wrote``).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# SLURM stdout file name: job_<job index>-<slurm id>.out
JOB_OUT_NAME = re.compile(r"^job_(\d+)-(\d+)\.out$")

# `date` default on the SLURM nodes: e.g. "Mon Apr 20 18:57:42 PDT 2026"
DATE_FMT = "%a %b %d %H:%M:%S %Z %Y"

# kind -> pattern; the first group(s) are the value. GENIE prints one line per vertex in pile-up.
_LINES = {
    "job_started": re.compile(r"^Job started:\s*(.+)$"),
    "job_ended": re.compile(r"^Job ended:\s*(.+)$"),
    "step": re.compile(r"^=== Step (\d+):"),
    "photonsim_exit": re.compile(r"PhotonSim exit code:\s*(\d+)\s*\(elapsed\s*([\d.]+)\s*s\)"),
    "genie_elapsed": re.compile(r"^GENIE(?:\s+vertex\s+\d+)?\s+elapsed:\s*([\d.]+)\s*s"),
    "event_total": re.compile(r"^\s+Event total time:\s*([\d.]+)s"),
    "event_sim": re.compile(r"^\s+Simulation completed in\s+([\d.]+)s"),
    "batch_save": re.compile(r"^Batch\s+\d+\s+save time:\s*([\d.]+)s"),
    "stage": re.compile(r"^\s+\[timing\]\s+(\w+)\s+([\d.]+)s\s*$"),
    "done": re.compile(r"^Successfully wrote\s+\d+\s+batches?"),
}


def classify(line: str) -> Optional[tuple[str, tuple[str, ...]]]:
    """``(kind, groups)`` for a timing line (one of the ``_LINES`` kinds), else ``None``."""
    for kind, pattern in _LINES.items():
        m = pattern.search(line)
        if m:
            return kind, m.groups()
    return None


def parse_job_log(lines: Iterable[str]) -> dict:
    """Reduce one job's stdout to its timings.

    ``job_wall_s`` is ``Job ended - Job started`` (``None`` when either is missing or not in
    :data:`DATE_FMT`); ``photonsim_elapsed_s`` is the last PhotonSim exit line's; GENIE times are
    summed over vertices; the per-event, per-batch and per-stage lists keep the print order.
    ``Simulation completed in`` lines are only classified (the CI speed test reads them).
    """
    started = ended = None
    photonsim_rc = photonsim_elapsed = None
    genie_total, n_genie = 0.0, 0
    event_total: list[float] = []
    batch_save: list[float] = []
    stages: dict[str, list[float]] = {}
    steps: set[int] = set()
    done = False
    for line in lines:
        hit = classify(line.rstrip("\n"))
        if hit is None:
            continue
        kind, g = hit
        if kind == "job_started":
            started = g[0].strip()
        elif kind == "job_ended":
            ended = g[0].strip()
        elif kind == "step":
            steps.add(int(g[0]))
        elif kind == "photonsim_exit":
            photonsim_rc, photonsim_elapsed = int(g[0]), float(g[1])
        elif kind == "genie_elapsed":
            genie_total += float(g[0])
            n_genie += 1
        elif kind == "event_total":
            event_total.append(float(g[0]))
        elif kind == "batch_save":
            batch_save.append(float(g[0]))
        elif kind == "stage":
            stages.setdefault(g[0], []).append(float(g[1]))
        elif kind == "done":
            done = True

    job_wall_s = None
    if started and ended:
        try:
            job_wall_s = (datetime.strptime(ended, DATE_FMT)
                          - datetime.strptime(started, DATE_FMT)).total_seconds()
        except ValueError:
            pass  # a timezone strptime does not know: no wall time
    return {
        "job_wall_s": job_wall_s,
        "photonsim_rc": photonsim_rc,
        "photonsim_elapsed_s": photonsim_elapsed,
        "genie_elapsed_s": genie_total if n_genie else None,
        "lucid_event_total_s": event_total,
        "lucid_batch_save_s": batch_save,
        "lucid_substage_s": stages,
        "steps_seen": sorted(steps),
        "lucid_done": done,
    }


def parse_job_out(path: Path) -> dict:
    """:func:`parse_job_log` of one ``job_<index>-<slurm id>.out``, with its name's two ids."""
    with path.open("r", errors="replace") as f:
        out = parse_job_log(f)
    m = JOB_OUT_NAME.match(path.name)
    return {
        "out_file": str(path),
        "job_index": int(m.group(1)) if m else None,
        "slurm_id": int(m.group(2)) if m else None,
        **out,
    }
