#!/usr/bin/env python3
"""SLURM fan-out for SIREN training hyperparameter scans.

Reads a JSON config that defines a baseline + a list of explicit override
dicts, materializes one sub-folder + sbatch per run under
``<output_root>/<scan_name>/<run_name>/``, and either prepares only or
submits with ``-s``.

Folder names are auto-derived from each run's diff against the baseline,
e.g. ``patience=40 -> p40``, ``zero_threshold=1e-4 -> z1e-04``. The empty
diff yields ``baseline``.

For the full layout / scan philosophy see ``README.md`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Paths ------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
S3DF_JOBS_DIR = SCRIPT_DIR.parent
USER_PATHS_DEFAULT = S3DF_JOBS_DIR / "user_paths.sh"


def load_user_paths(path: Path) -> Dict[str, str]:
    """Source user_paths.sh in a subshell and extract its env."""
    if not path.is_file():
        raise FileNotFoundError(
            f"user_paths.sh not found at {path}. "
            f"Copy {path.parent / 'user_paths.sh.template'} and configure it."
        )
    proc = subprocess.run(
        ["bash", "-c", f"set -e; source {shlex.quote(str(path))} && env"],
        capture_output=True, text=True, check=True,
    )
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


# --- Hyperparameter ↔ CLI mapping ------------------------------------------

# Maps a baseline / run-override key to the matching `lucid-train-siren` flag.
# Unknown keys raise (see resolve_run) so typos don't silently lose params.
FLAG_MAP: Dict[str, str] = {
    "material":            "--material",
    "particle":            "--particle",
    "data_type":           "--data-type",
    "h5_path":             "--h5-path",
    "num_steps":           "--num-steps",
    "batch_size":          "--batch-size",
    "learning_rate":       "--learning-rate",
    "min_lr":              "--min-lr",
    "patience":            "--patience",
    "lr_reduction_factor": "--lr-reduction-factor",
    "zero_threshold":      "--zero-threshold",
    "zero_keep_frac":      "--zero-keep-frac",
    "energy_balance":      "--energy-balance",
    "target_importance":   "--target-importance",
    "val_split":           "--val-split",
    "hidden_features":     "--hidden-features",
    "hidden_layers":       "--hidden-layers",
    "w0":                  "--w0",
    "weight_decay":        "--weight-decay",
    "grad_clip_norm":      "--grad-clip-norm",
    "seed":                "--seed",
    "log_every":           "--log-every",
    "val_every":           "--val-every",
    "checkpoint_every":    "--checkpoint-every",
    "prediction_plot_every": "--prediction-plot-every",
}

# Short labels for folder names. Unmapped keys fall back to the raw key.
NAME_MAP: Dict[str, str] = {
    "patience":       "p",
    "zero_threshold": "z",
    "zero_keep_frac": "zf",
    "energy_balance": "ebal",
    "target_importance": "ti",
    "batch_size":     "b",
    "learning_rate":  "lr",
    "min_lr":         "minlr",
    "num_steps":      "steps",
    "hidden_features": "hf",
    "hidden_layers":   "hl",
    "w0":              "w0",
    "val_split":       "vs",
    "weight_decay":    "wd",
    "grad_clip_norm":  "gc",
}


# SLURM defaults tuned for SIREN training. The PhotonSim CPU-job defaults in
# `user_paths.sh` are wrong for this, so we hardcode our own here. Config can
# override any of these via a top-level `slurm` block.
SLURM_DEFAULTS: Dict[str, str] = {
    "partition": "roma",
    "account":   "mli:cider-ml",
    "time":      "04:00:00",
    "memory":    "32000",
    "cpus":      "4",
    "gpus":      "1",
}

DEFAULT_OUTPUT_ROOT = Path("/sdf/data/neutrino/cjesus/SIREN_files/training_tests")


# --- Formatting helpers ----------------------------------------------------

def _fmt_value(v: Any) -> str:
    """Compact formatter for folder names.

    Floats < 1 → scientific (``1e-02``); other floats → ``%g``; ints stay
    as-is; strings stay verbatim.
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1:
            return f"{v:.0e}".replace("e+", "e").replace("e-0", "e-")
        return f"{v:g}"
    return str(v)


def derive_run_name(diff: Dict[str, Any]) -> str:
    """Compact folder name from a run's diff against baseline."""
    if not diff:
        return "baseline"
    parts = [f"{NAME_MAP.get(k, k)}{_fmt_value(v)}"
             for k, v in sorted(diff.items())]
    return "_".join(parts)


def resolve_run(baseline: Dict[str, Any],
                overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge baseline + overrides, validate keys, return the full config."""
    unknown = [k for k in overrides if k not in FLAG_MAP]
    if unknown:
        raise ValueError(
            f"Unknown override key(s): {unknown}. Add them to FLAG_MAP in "
            f"generate_jobs.py if they're real new CLI flags."
        )
    merged = dict(baseline)
    merged.update(overrides)
    missing = [k for k in ("material", "particle", "data_type", "h5_path")
               if k not in merged]
    if missing:
        raise ValueError(
            f"baseline missing required keys: {missing}"
        )
    return merged


# --- SBATCH ----------------------------------------------------------------

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --job-name=train_{run_name}
#SBATCH --output={run_dir}/job-%j.out
#SBATCH --error={run_dir}/job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --mem={memory}
#SBATCH --time={time}

set -eu -o pipefail
echo "SLURM Job ID: ${{SLURM_JOB_ID}}"
echo "Job started:  $(date)"
echo "Node:         $(hostname)"

# Expose the isolated lucid user-site (jax-cuda12-plugin lives there) to the
# container, otherwise JAX falls back to CPU. Mirrors the jupyter() wrapper
# in ~/.bashrc — see JUPYTER_SETUP.md.
export APPTAINERENV_PYTHONUSERBASE="{LUCID_ENV_BASE}"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs \\
    -B {LUCID_DEV_PATH}:/opt/LUCiD \\
    {LUCID_IMAGE_PATH} \\
    lucid-train-siren {cli_args}

echo "Job ended: $(date)"
"""


def build_cli_args(resolved: Dict[str, Any], run_dir: Path) -> str:
    """Build the `lucid-train-siren` argument string from the resolved config."""
    pieces: List[str] = ["--output-dir", shlex.quote(str(run_dir)),
                         "--no-monitoring"]
    for key, value in resolved.items():
        if key not in FLAG_MAP:
            continue  # already validated; just defensive
        pieces.extend([FLAG_MAP[key], shlex.quote(_fmt_cli(value))])
    return " ".join(pieces)


def _fmt_cli(v: Any) -> str:
    """Format a Python value the way the CLI accepts it."""
    if isinstance(v, bool):
        # Booleans are not currently represented as values in lucid-train-siren;
        # all on/off flags are dest pairs (e.g. --prediction-plots /
        # --no-prediction-plots). If we ever need them here, add the special
        # case at that point.
        raise ValueError("boolean baseline values aren't supported")
    if isinstance(v, float):
        # Avoid scientific-notation surprises for the CLI parser: %.10g is
        # round-trip-safe for the floats we use here.
        return f"{v:.10g}"
    return str(v)


# --- Driver ----------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="Scan config JSON (see configs/water_mu_v1.json).")
    p.add_argument("-s", "--submit", action="store_true",
                   help="Submit jobs to SLURM (default: prepare only).")
    p.add_argument("-t", "--test", action="store_true",
                   help="Test mode: only the first run.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override the config's output_root.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-emit runs whose final_training_progress.png "
                        "already exists (default: skip them).")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT,
                   help="Path to user_paths.sh "
                        "(only LUCID_DEV_PATH / LUCID_IMAGE_PATH are read).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.submit and shutil.which("sbatch") is None:
        print("error: sbatch not on PATH; --submit requires SLURM.",
              file=sys.stderr)
        return 2

    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = json.loads(args.config.read_text())

    scan_name: str = cfg["name"]
    baseline: Dict[str, Any] = cfg["baseline"]
    runs_raw: List[Dict[str, Any]] = cfg["runs"]
    slurm_overrides: Dict[str, str] = {k: str(v) for k, v in
                                        (cfg.get("slurm") or {}).items()}
    slurm = {**SLURM_DEFAULTS, **slurm_overrides}

    output_root = (args.output_root.resolve() if args.output_root
                   else Path(cfg.get("output_root", DEFAULT_OUTPUT_ROOT)).resolve())

    env = load_user_paths(args.user_paths)
    for k in ("LUCID_DEV_PATH", "LUCID_IMAGE_PATH"):
        if not env.get(k):
            print(f"error: {k} not set in {args.user_paths}", file=sys.stderr)
            return 2
    # LUCID_ENV_BASE points at the isolated user-site holding the
    # jax-cuda12-plugin (see JUPYTER_SETUP.md). It's normally exported in
    # the user's ~/.bashrc, not user_paths.sh; we read it from the shell
    # env (which load_user_paths inherits), with a final fall-back path.
    if not env.get("LUCID_ENV_BASE"):
        env["LUCID_ENV_BASE"] = "/sdf/data/neutrino/cjesus/python_envs/lucid"
        print(f"warning: LUCID_ENV_BASE not set; defaulting to "
              f"{env['LUCID_ENV_BASE']}", file=sys.stderr)

    # Resolve every run + check for collisions.
    runs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for raw in runs_raw:
        diff = dict(raw)
        run_name = derive_run_name(diff)
        if run_name in seen:
            raise ValueError(
                f"folder-name collision: {run_name!r} produced by both "
                f"{seen[run_name]} and {diff}. Adjust one of the runs."
            )
        seen[run_name] = diff
        resolved = resolve_run(baseline, diff)
        runs.append((run_name, diff, resolved))

    if args.test and runs:
        runs = runs[:1]

    print(f"=== SIREN training scan: {scan_name} ===")
    print(f"baseline:    {baseline}")
    print(f"runs:        {len(runs)}")
    print(f"output_root: {output_root}")
    print(f"partition:   {slurm['partition']}   "
          f"time={slurm['time']} mem={slurm['memory']} gpus={slurm['gpus']}")
    print("")

    scan_dir = output_root / scan_name
    scan_dir.mkdir(parents=True, exist_ok=True)

    prepared = submitted = skipped = 0
    for run_name, diff, resolved in runs:
        run_dir = scan_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        done_marker = run_dir / "final_training_progress.png"
        if done_marker.is_file() and not args.no_skip_existing:
            print(f"  skip (already finished): {run_dir}")
            skipped += 1
            continue

        # Save the resolved config next to the sbatch for reproducibility.
        (run_dir / "config.json").write_text(
            json.dumps(
                {"scan": scan_name, "run_name": run_name,
                 "diff_from_baseline": diff, "resolved": resolved,
                 "slurm": slurm},
                indent=2,
            ) + "\n"
        )

        cli_args = build_cli_args(resolved, run_dir)
        sbatch = SBATCH_TEMPLATE.format(
            partition=slurm["partition"], account=slurm["account"],
            time=slurm["time"], memory=slurm["memory"],
            cpus=slurm["cpus"], gpus=slurm["gpus"],
            run_name=run_name, run_dir=run_dir,
            LUCID_DEV_PATH=env["LUCID_DEV_PATH"],
            LUCID_IMAGE_PATH=env["LUCID_IMAGE_PATH"],
            LUCID_ENV_BASE=env["LUCID_ENV_BASE"],
            cli_args=cli_args,
        )
        sb_path = run_dir / "submit.sbatch"
        sb_path.write_text(sbatch)
        sb_path.chmod(0o755)
        prepared += 1

        if args.submit:
            r = subprocess.run(["sbatch", str(sb_path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED sbatch for {run_name}: {r.stderr.strip()}",
                      file=sys.stderr)
                continue
            submitted += 1
            print(f"  submitted {run_name}  -> {r.stdout.strip()}")
        else:
            print(f"  prepared  {run_name}  -> {sb_path}")

    print("")
    print("=== Scan complete ===")
    print(f"Prepared:        {prepared}")
    print(f"Submitted:       {submitted}")
    print(f"Skipped (done):  {skipped}")
    print(f"Output:          {scan_dir}")
    if not args.submit and prepared:
        print("\nNext: pass -s to submit. Already-finished runs (with "
              "final_training_progress.png) are skipped by default; "
              "--no-skip-existing forces a rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
