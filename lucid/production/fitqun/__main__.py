"""Command line for the fiTQun tuning-input generators.

    python -m lucid.production.fitqun <stage> <action> [options]

Stages that fan out over a grid (``cprofile``) split into a per-cell
``accumulate`` that a batch job runs and a ``build`` that merges the cells into
the file fiTQun reads; stages cheap enough to do in one go (``chargepdf``)
have a single ``scan``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import binning, chargepdf, cprofile


def _cmd_cprofile_accumulate(args) -> int:
    cell = cprofile.accumulate(
        args.input, pdg=args.pdg, momentum_mev=args.momentum,
        s_hi_cm=args.s_hi, s_max_cm=args.s_max, quantile=args.quantile)
    out = cell.save(args.output)
    print(f"{out}: {cell.n_events} events, {cell.n_photons:.1f} photons/primary, "
          f"s_max = {cell.s_max_cm:.1f} cm")
    return 0


def _cmd_cprofile_build(args) -> int:
    cells = {}
    for path in args.cells:
        cell = cprofile.ProfileCell.load(path)
        if cell.pdg != args.pdg:
            raise SystemExit(f"{path}: PDG {cell.pdg} does not match --pdg {args.pdg}")
        key = cell.momentum_mev
        cells[key] = cells[key] + cell if key in cells else cell
    if not cells:
        raise SystemExit("no cells given")

    out = Path(args.output or f"CProf_{args.pdg}_WCSim.root")
    print(f"building {out} from {len(cells)} momentum points "
          f"({min(cells):g}-{max(cells):g} MeV/c)", flush=True)
    cprofile.write_cprofile(out, args.pdg, list(cells.values()))
    print(f"wrote {out}")
    return 0


def _cmd_chargepdf_scan(args) -> int:
    mu_values = None if args.mu is None else np.asarray(args.mu, dtype=np.float64)
    paths = chargepdf.run_scan(
        args.output, n_pmt=args.n_pmt, n_events=args.n_events,
        model=args.model, seed=args.seed, mu_values=mu_values)
    print(f"wrote {len(paths)} charge-PDF files to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lucid.production.fitqun",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = p.add_subparsers(dest="stage", required=True)

    cp = stages.add_parser("cprofile", help="Cherenkov emission profile (PhotonSim)")
    cp_actions = cp.add_subparsers(dest="action", required=True)

    acc = cp_actions.add_parser("accumulate", help="reduce one PhotonSim file to a cell")
    acc.add_argument("input", type=Path, help="PhotonSim output ROOT file")
    acc.add_argument("--pdg", type=int, required=True, choices=sorted(binning.PDG_NAMES))
    acc.add_argument("--momentum", type=float, required=True, help="MeV/c")
    acc.add_argument("-o", "--output", type=Path, required=True, help="cell .npz")
    acc.add_argument("--s-hi", type=float, default=100000.0,
                     help="upper bound on track length for the fine grid, cm "
                          "(default: %(default)s)")
    acc.add_argument("--s-max", type=float, default=None,
                     help="pin s_max (cm) instead of taking it from the data; use "
                          "this to keep split jobs of one cell consistent")
    acc.add_argument("--quantile", type=float, default=0.9999,
                     help="emission-distance quantile defining s_max (default: %(default)s)")
    acc.set_defaults(func=_cmd_cprofile_accumulate)

    bld = cp_actions.add_parser("build", help="merge cells into CProf_<pdg>_WCSim.root")
    bld.add_argument("cells", nargs="+", type=Path)
    bld.add_argument("--pdg", type=int, required=True, choices=sorted(binning.PDG_NAMES))
    bld.add_argument("-o", "--output", type=Path, default=None)
    bld.set_defaults(func=_cmd_cprofile_build)

    ch = stages.add_parser("chargepdf", help="photosensor charge response f(q|mu)")
    ch_actions = ch.add_subparsers(dest="action", required=True)
    scan = ch_actions.add_parser("scan", help="write the whole mu scan")
    scan.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    scan.add_argument("--n-pmt", type=int, required=True,
                      help="PMTs per shot; only sets the statistics per mu point")
    scan.add_argument("--n-events", type=int, default=80,
                      help="shots per mu point below mu=30 (default: %(default)s)")
    scan.add_argument("--model", default="ski",
                      help="digitizer model from the physics config (default: %(default)s)")
    scan.add_argument("--seed", type=int, default=0)
    scan.add_argument("--mu", type=float, nargs="+", default=None,
                      help="override the mu grid (default: the reference mutbl.txt)")
    scan.set_defaults(func=_cmd_chargepdf_scan)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
