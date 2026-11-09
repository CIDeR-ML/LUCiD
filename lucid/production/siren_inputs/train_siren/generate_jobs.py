#!/usr/bin/env python3
"""Cluster fan-out for SIREN training scans (lucid-train-siren, a GPU job per run).

Reads a JSON config that defines a baseline + a list of explicit override
dicts, materializes one sub-folder + submit file per run under
``<output_root>/<scan_name>/<run_name>/``, and either prepares only or
submits with ``-s``.

The planning logic (FLAG_MAP validation, baseline+override resolution,
folder-name derivation, CLI arg construction) lives in
`lucid.production.siren_inputs.train_planning`; the submit description in the
cluster adapter's `render_train_run`.

Folder names are auto-derived from each run's diff against the baseline,
e.g. ``patience=40 -> p40``, ``zero_threshold=1e-4 -> z1e-04``. The empty
diff yields ``baseline``.

The site comes from ``user_paths.sh``, and a missing setting is an error that names it:
``LUCID_IMAGE_PATH``; ``LUCID_ENV_BASE`` (the Python user site holding the GPU jaxlib the image
lacks); on Slurm ``SIREN_TRAIN_PARTITION`` (a GPU partition) and ``SLURM_ACCOUNT``.
``LUCID_DEV_PATH``, when set, runs that checkout's trainer instead of the image's. A config's
``h5_path`` may name user_paths variables as ``${VAR}``; ``${SIREN_INPUTS_BASE_PATH}`` defaults to
``SIREN_OUTPUT_BASE_PATH``.

For the full layout / scan philosophy see ``README.md`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import re
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

from lucid.production.platform.adapters import get_adapter  # noqa: E402
from lucid.production.siren_inputs.train_planning import (  # noqa: E402
    build_cli_args, derive_run_name, resolve_run,
)
from lucid.production.platform.user_paths import load_user_paths  # noqa: E402


USER_PATHS_DEFAULT = SCRIPT_DIR.parent.parent / "platform" / "user_paths.sh"

# The job's resources; a config's `resources` block overrides any of them (`flavour`: the
# HTCondor job flavour, default CONDOR_TRAIN_FLAVOUR). 150k steps of the nominal recipe took
# 30 min on a turing GPU. Partition and account are the site's, from user_paths.sh.
RESOURCES: Dict[str, str] = {
    "time":   "04:00:00",
    "memory": "32000",
    "cpus":   "4",
    "gpus":   "1",
}
RESOURCE_KEYS = (*RESOURCES, "flavour")

_VAR = re.compile(r"\$\{(\w+)\}")


class SiteError(ValueError):
    """A setting the site (user_paths.sh) or the config lacks."""


def expand(value: str, env: Dict[str, str], where: str) -> str:
    """``value`` with each ``${VAR}`` replaced by its user_paths value; an unset one raises."""
    def sub(m):
        v = env.get(m.group(1))
        if not v:
            raise SiteError(f"{where} names ${{{m.group(1)}}}, which is not set in user_paths.sh")
        return v
    return _VAR.sub(sub, value)


def site_resources(cfg: dict, env: Dict[str, str], adapter_name: str,
                   partition: Optional[str]) -> Dict[str, str]:
    """The job resources: RESOURCES, the config's ``resources``, and the site's partition and
    account (Slurm). Raises SiteError naming a missing setting."""
    if "slurm" in cfg:
        raise SiteError("the config's 'slurm' block is now 'resources' (time, memory, cpus, "
                        "gpus, flavour); partition and account come from user_paths.sh")
    res = {k: str(v) for k, v in (cfg.get("resources") or {}).items()}
    unknown = sorted(set(res) - set(RESOURCE_KEYS))
    if unknown:
        raise SiteError(f"unknown resources {unknown}: a config sets {list(RESOURCE_KEYS)}; "
                        f"partition and account come from user_paths.sh "
                        f"(SIREN_TRAIN_PARTITION, SLURM_ACCOUNT)")
    out = {**RESOURCES, **res}
    if adapter_name in ("slurm", "nersc"):
        out["partition"] = partition or env.get("SIREN_TRAIN_PARTITION", "")
        out["account"] = env.get("SLURM_ACCOUNT", "")
        if not out["partition"]:
            raise SiteError("SIREN_TRAIN_PARTITION is not set in user_paths.sh: the GPU "
                            "partition training runs on (S3DF: ampere, ada, hopper or turing)")
        if not out["account"]:
            raise SiteError("SLURM_ACCOUNT is not set in user_paths.sh")
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="Scan config JSON (see configs/example_water_mu.json).")
    p.add_argument("-s", "--submit", action="store_true",
                   help="Submit jobs (default: prepare only).")
    p.add_argument("-t", "--test", action="store_true",
                   help="Test mode: only the first run.")
    p.add_argument("-P", "--partition", type=str, default=None,
                   help="Override SIREN_TRAIN_PARTITION from user_paths.sh (Slurm).")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override the config's output_root (default: "
                        "<SIREN_OUTPUT_BASE_PATH or OUTPUT_BASE_PATH>/training_tests).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-emit runs whose final_training_progress.pdf "
                        "already exists (default: skip them).")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT,
                   help="Path to user_paths.sh (default: %(default)s).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = json.loads(args.config.read_text())

    scan_name: str = cfg["name"]
    baseline: Dict[str, Any] = dict(cfg["baseline"])
    runs_raw: List[Dict[str, Any]] = [dict(r) for r in cfg["runs"]]

    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)
    # A stage reads earlier stages' outputs from SIREN_INPUTS_BASE_PATH, by default your own.
    if not env.get("SIREN_INPUTS_BASE_PATH"):
        env["SIREN_INPUTS_BASE_PATH"] = env.get("SIREN_OUTPUT_BASE_PATH", "")
    try:
        for k, why in (("LUCID_IMAGE_PATH", "the container image"),
                       ("LUCID_ENV_BASE", "the Python user site with the GPU jaxlib; without it "
                                          "JAX trains on the CPU")):
            if not env.get(k):
                raise SiteError(f"{k} is not set in {args.user_paths} ({why})")
        resources = site_resources(cfg, env, adapter.name, args.partition)
        for where, d in [("baseline", baseline)] + [(f"runs[{i}]", r)
                                                     for i, r in enumerate(runs_raw)]:
            if "h5_path" in d:
                d["h5_path"] = expand(str(d["h5_path"]), env, f"{where}.h5_path")
                if not Path(d["h5_path"]).is_file():
                    raise SiteError(f"{where}.h5_path: no table at {d['h5_path']}")
    except SiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_root:
        output_root = args.output_root.resolve()
    elif cfg.get("output_root"):
        output_root = Path(cfg["output_root"]).resolve()
    else:
        base = env.get("SIREN_OUTPUT_BASE_PATH") or env.get("OUTPUT_BASE_PATH")
        if not base:
            print(f"error: no output root: pass --output-root, set output_root in the config, "
                  f"or set SIREN_OUTPUT_BASE_PATH (or OUTPUT_BASE_PATH) in {args.user_paths}",
                  file=sys.stderr)
            return 2
        output_root = (Path(base) / "training_tests").resolve()
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"error: {adapter.submit_cmd} not on PATH; --submit requires "
              f"this cluster.", file=sys.stderr)
        return 2

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

    lucid = env.get("LUCID_DEV_PATH") or "the image's /opt/LUCiD (LUCID_DEV_PATH unset)"
    print(f"=== SIREN training scan: {scan_name} ===")
    print(f"cluster:     {adapter.name}")
    print(f"table:       {baseline.get('h5_path')}")
    print(f"runs:        {len(runs)}")
    print(f"output_root: {output_root}")
    print(f"LUCiD:       {lucid}")
    print("resources:   " + " ".join(f"{k}={v}" for k, v in resources.items()))
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

        # Save the resolved config next to the submit file for reproducibility.
        (run_dir / "config.json").write_text(
            json.dumps(
                {"scan": scan_name, "run_name": run_name,
                 "diff_from_baseline": diff, "resolved": resolved,
                 "resources": resources},
                indent=2,
            ) + "\n"
        )

        cli_args = build_cli_args(resolved, run_dir)
        try:
            body = adapter.render_train_run(
                run_dir=run_dir, run_name=run_name, cli_args=cli_args, resources=resources,
            )
        except NotImplementedError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
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
        print("Then rank_seeds.py ranks a seed scan, and lucid-siren-promote installs the "
              "chosen run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
