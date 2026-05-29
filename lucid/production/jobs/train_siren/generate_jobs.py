#!/usr/bin/env python3
"""SLURM fan-out for SIREN training hyperparameter scans (Stage 3).

Reads a JSON config that defines a baseline + a list of explicit override
dicts, materializes one sub-folder + sbatch per run under
``<output_root>/<scan_name>/<run_name>/``, and either prepares only or
submits with ``-s``.

The planning logic (FLAG_MAP validation, baseline+override resolution,
folder-name derivation, CLI arg construction) lives in
`lucid.production.cluster_common.train_planning`; this script only owns
the SLURM-specific submit-script template and `sbatch` call.

Folder names are auto-derived from each run's diff against the baseline,
e.g. ``patience=40 -> p40``, ``zero_threshold=1e-4 -> z1e-04``. The empty
diff yields ``baseline``.

For the full layout / scan philosophy see ``README.md`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Host-side entrypoint: ensure the LUCiD checkout root is on sys.path.
SCRIPT_DIR = Path(__file__).parent
LUCID_ROOT = SCRIPT_DIR.resolve().parents[3]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.cluster_common import htcondor  # noqa: E402, F401
from lucid.production.cluster_common import nersc  # noqa: E402, F401  registers adapter
from lucid.production.cluster_common.cluster import get_adapter  # noqa: E402
from lucid.production.cluster_common.train_planning import (  # noqa: E402
    build_cli_args, derive_run_name, resolve_run,
)
from lucid.production.cluster_common.user_paths import load_user_paths  # noqa: E402


JOBS_DIR = SCRIPT_DIR.parent       # the cluster-portable jobs/ dir
USER_PATHS_DEFAULT = JOBS_DIR / "user_paths.sh"


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

DEFAULT_OUTPUT_ROOT = Path("/sdf/data/neutrino/cjesus/CIDER/SIREN_files/training_tests")


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
                   help="Re-emit runs whose final_training_progress.pdf "
                        "already exists (default: skip them).")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT,
                   help="Path to user_paths.sh "
                        "(only LUCID_DEV_PATH / LUCID_IMAGE_PATH are read).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

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
    adapter = get_adapter(env)
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"error: {adapter.submit_cmd} not on PATH; --submit requires "
              f"this cluster.", file=sys.stderr)
        return 2

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

        done_marker = run_dir / "final_training_progress.pdf"
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
        body = adapter.render_train_run(
            run_dir=run_dir, run_name=run_name, cli_args=cli_args, slurm=slurm,
        )
        sb_path = run_dir / f"submit.{adapter.submit_extension}"
        sb_path.write_text(body)
        sb_path.chmod(0o755)
        # Stable marker for the bash shim's host-side submission pass.
        print(f"[PREPARED] {sb_path}")
        prepared += 1

        if args.submit:
            r = subprocess.run([adapter.submit_cmd, str(sb_path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED {adapter.submit_cmd} for {run_name}: "
                      f"{r.stderr.strip()}", file=sys.stderr)
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
              "final_training_progress.pdf) are skipped by default; "
              "--no-skip-existing forces a rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
