#!/usr/bin/env python3
"""Cluster job for the t0 fit: PhotonSim's calculate_t0.py on the training-input scan.

The simulator's photon emission time is a fit of ``PhotonHist_TimeDistanceNorm`` (photon delay
t - d/c against s/s_max), which PhotonSim fills whenever the macro sets ``/output/smax``. Every
training-input cell (``lucid.production.siren_inputs.generate_jobs``, configs/water_mu.json and
water_mu_lowE.json, water_el.json) sets it, so the t0 inputs are those cells' ``photonsim.root``
files, one per energy: no PhotonSim run of its own. This generator links them into one directory
as ``<particle>_<E>MeV.root`` (calculate_t0.py reads every ``*<E>MeV*.root`` there) and writes
one job that runs, in the container,

    python3 $PHOTONSIM_DEV_PATH/tools/t0_correction/calculate_t0.py <out>/inputs --out-dir <out> \\
        --material M --particle P --smax-data-dir $PHOTONSIM_DEV_PATH/data

Its ``timing_parameters.json`` is ``lucid-siren-promote --t0``'s input.

    inputs   <SIREN_INPUTS_BASE_PATH or SIREN_OUTPUT_BASE_PATH>/training_inputs/<m>/<p>/<E>MeV/
             photonsim.root (--inputs), and $PHOTONSIM_DEV_PATH/data/<m>/<p>/smax_fit.csv
    outputs  <SIREN_OUTPUT_BASE_PATH or OUTPUT_BASE_PATH>/t0_parametrization/<m>/<p>/ (--output):
             inputs/<p>_<E>MeV.root (links), timing_parameters.json, pred_vs_data.png,
             pred_vs_data_examples.png, trends.png, the submit file and its logs

Usage (inside the container, as the other generators):
    run_in_container.sh lucid.production.siren_inputs.t0.generate_jobs --material water --particle mu- [-s]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).parent
LUCID_ROOT = SCRIPT_DIR.resolve().parents[3]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.platform.adapters import get_adapter  # noqa: E402
from lucid.production.platform.user_paths import load_user_paths  # noqa: E402

USER_PATHS_DEFAULT = SCRIPT_DIR.parent.parent / "platform" / "user_paths.sh"
CALCULATE_T0 = Path("tools") / "t0_correction" / "calculate_t0.py"
T0_FORM = "stretched_exp_delay"          # lucid.siren.emitter.T0_FORM; the fit the simulator reads
CELL = re.compile(r"^(\d+)MeV$")
HIST = "PhotonHist_TimeDistanceNorm"     # what calculate_t0.py fits; filled when /output/smax is set


class SetupError(ValueError):
    """A setting, input or tool the stage needs and does not have."""


def find_cells(inputs: Path) -> List[tuple]:
    """``[(E_MeV, photonsim.root), ...]`` of the scan in ``inputs``, by energy."""
    cells = []
    for d in inputs.iterdir() if inputs.is_dir() else []:
        m = CELL.match(d.name)
        if m and (d / "photonsim.root").is_file():
            cells.append((int(m.group(1)), d / "photonsim.root"))
    return sorted(cells)


def link_inputs(cells, links: Path, particle: str) -> None:
    """``links/<particle>_<E>MeV.root`` -> each cell's photonsim.root; stale links go, nothing
    else under ``links`` is touched."""
    links.mkdir(parents=True, exist_ok=True)
    for old in links.glob("*.root"):
        if old.is_symlink():
            old.unlink()
    for energy, root in cells:
        (links / f"{particle}_{energy}MeV.root").symlink_to(root)


def plan(env: dict, material: str, particle: str, inputs: Optional[Path],
         output: Optional[Path]) -> tuple:
    """``(calculate_t0.py, photonsim data dir, inputs dir, cells, output dir)``; raises
    SetupError naming what is missing."""
    ps = env.get("PHOTONSIM_DEV_PATH")
    if not ps:
        raise SetupError("PHOTONSIM_DEV_PATH is not set in user_paths.sh: the PhotonSim checkout "
                         f"whose {CALCULATE_T0} runs the fit and whose data/ holds smax_fit.csv")
    calc = Path(ps) / CALCULATE_T0
    if not calc.is_file():
        raise SetupError(f"no {calc}")
    if T0_FORM not in calc.read_text():
        raise SetupError(f"{calc} predates the {T0_FORM!r} fit the simulator reads; update "
                         f"the PhotonSim checkout (main since May 2026)")
    smax = Path(ps) / "data" / material / particle / "smax_fit.csv"
    if not smax.is_file():
        raise SetupError(f"no s_max fit {smax} (the smax stage writes it)")
    if inputs is None:
        base = env.get("SIREN_INPUTS_BASE_PATH") or env.get("SIREN_OUTPUT_BASE_PATH")
        if not base:
            raise SetupError("no input root: pass --inputs, or set SIREN_OUTPUT_BASE_PATH (or "
                             "SIREN_INPUTS_BASE_PATH) in user_paths.sh")
        inputs = Path(base) / "training_inputs" / material / particle
    cells = find_cells(inputs)
    if not cells:
        raise SetupError(f"no <E>MeV/photonsim.root under {inputs}: run the training-input scan "
                         f"(lucid.production.siren_inputs.generate_jobs, then merge.sh) first")
    import uproot
    for _, root in (cells[0], cells[-1]):
        with uproot.open(root) as f:
            if HIST not in f:
                raise SetupError(f"{root} has no {HIST}: its PhotonSim ran without /output/smax "
                                 f"or predates the histogram; rerun that scan")
    if output is None:
        base = env.get("SIREN_OUTPUT_BASE_PATH") or env.get("OUTPUT_BASE_PATH")
        if not base:
            raise SetupError("no output root: pass --output, or set SIREN_OUTPUT_BASE_PATH (or "
                             "OUTPUT_BASE_PATH) in user_paths.sh")
        output = Path(base) / "t0_parametrization" / material / particle
    return calc, Path(ps) / "data", inputs, cells, output


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", required=True, help="e.g. water")
    p.add_argument("--particle", required=True, help="PhotonSim's name, e.g. mu- or e-")
    p.add_argument("--inputs", type=Path, default=None,
                   help="the training-input scan, <E>MeV/photonsim.root (default: "
                        "<SIREN_INPUTS_BASE_PATH or SIREN_OUTPUT_BASE_PATH>/training_inputs/<m>/<p>)")
    p.add_argument("--output", type=Path, default=None,
                   help="output directory (default: <SIREN_OUTPUT_BASE_PATH or "
                        "OUTPUT_BASE_PATH>/t0_parametrization/<m>/<p>)")
    p.add_argument("-s", "--submit", action="store_true", help="Submit the job (default: prepare).")
    p.add_argument("-P", "--partition", default=None,
                   help="Override SLURM_PARTITION (or CONDOR_JOB_FLAVOUR) from user_paths.sh.")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT,
                   help="Path to user_paths.sh (default: %(default)s).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Prepare the job even when timing_parameters.json exists.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)
    try:
        if not env.get("LUCID_IMAGE_PATH"):
            raise SetupError("LUCID_IMAGE_PATH is not set in user_paths.sh")
        partition = (args.partition or env.get("SLURM_PARTITION")
                     or env.get("CONDOR_JOB_FLAVOUR", ""))
        if not partition:
            raise SetupError("SLURM_PARTITION (or CONDOR_JOB_FLAVOUR) is not set in user_paths.sh")
        partition = partition.split(",")[0].split(":")[0]    # a CPU job: the first partition
        calc, data, inputs, cells, out = plan(env, args.material, args.particle,
                                              args.inputs, args.output)
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"error: {adapter.submit_cmd} not on PATH; --submit requires this cluster.",
              file=sys.stderr)
        return 2

    print(f"=== t0 fit: {args.material}/{args.particle} ===")
    print(f"cluster:    {adapter.name}")
    print(f"inputs:     {inputs}  ({len(cells)} energies, {cells[0][0]}..{cells[-1][0]} MeV)")
    print(f"tool:       {calc}")
    print(f"output:     {out}")
    if (out / "timing_parameters.json").is_file() and not args.no_skip_existing:
        print(f"  skip (fit exists): {out / 'timing_parameters.json'}")
        return 0

    links = out / "inputs"
    link_inputs(cells, links, args.particle)
    command = (f"/opt/conda/bin/python3 {calc} {links} --out-dir {out} "
               f"--material {args.material} --particle {args.particle} --smax-data-dir {data}")
    body = adapter.render_command_job(command=command, cell_dir=out,
                                      job_name=f"t0_{args.material}_{args.particle}",
                                      log_stem="job", partition=partition,
                                      gpus="0")   # the fit is CPU-only
    sb = out / f"submit.{adapter.submit_extension}"
    sb.write_text(body)
    sb.chmod(0o755)
    print(f"[PREPARED] {sb}")
    if args.submit:
        r = subprocess.run([adapter.submit_cmd, str(sb)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED {adapter.submit_cmd}: {r.stderr.strip()}", file=sys.stderr)
            return 1
        print(f"  submitted -> {r.stdout.strip()}")
    print("")
    print(f"Next: once it has run, lucid-siren-promote --t0 {out / 'timing_parameters.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
