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

from . import angular, binning, chargepdf, cprofile, scattable


def _cmd_cprofile_accumulate(args) -> int:
    cell = cprofile.accumulate(
        args.input, pdg=args.pdg, momentum_mev=args.momentum,
        s_max_cm=args.s_max, smax_quantile=args.smax_quantile)
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


def _cmd_sample_accumulate(args) -> int:
    from . import sample_reduce

    g = np.load(args.geometry)
    shard = sample_reduce.reduce_shard(
        args.input,
        pmt_positions_m=g["positions_mm"] / 1000.0,
        pmt_dir_z=g["directions"][:, 2],
        det_radius_cm=float(g["radius"]) * 100.0,
        det_halfheight_cm=float(g["height"]) * 100.0 / 2.0,
        pmt_radius_cm=float(g["sensor_radius"]) * 100.0,
        shell_radii_cm=args.shells)
    out = shard.save(args.output)
    occ = max(c.occupancy for c in shard.scattered.values())
    print(f"{out}: {shard.n_detected:,} detected of {shard.n_photons:,} "
          f"({shard.n_indirect:,} indirect), peak occupancy {100 * occ:.2f}%")
    return 0


def _cmd_sample_build(args) -> int:
    from . import sample_reduce

    total = None
    for path in args.shards:
        shard = sample_reduce.SampleShard.load(path)
        total = shard if total is None else total + shard
    if total is None:
        raise SystemExit("no shards given")

    outdir = Path(args.output)
    (outdir / "angular").mkdir(parents=True, exist_ok=True)
    ratios = {name: total.scattered[name].to_dense().ratio_to(
                        total.direct[name].to_dense())
              for name in total.scattered}
    sca = scattable.write_hdf5(outdir / "scattables.h5", ratios,
                               {"n_detected": total.n_detected,
                                "n_indirect": total.n_indirect,
                                "shards": len(args.shards)})
    print(f"wrote {sca} from {len(args.shards)} shards "
          f"({total.n_detected:,} detected, {total.n_indirect:,} indirect)")

    edges = np.linspace(0.0, 1.0, len(next(iter(total.angular_counts.values()))) + 1)
    written = 0
    for r, counts in sorted(total.angular_counts.items()):
        # fit_cos.C normalises to normal incidence, so a shell whose top bin is
        # empty has no table to write. Report it and keep going: at reduced
        # statistics the outer shells fill long before the inner ones, and
        # losing the whole merge over one thin shell helps nobody.
        if counts[-1] <= 0:
            print(f"  angResp_{r:g}: SKIPPED, {counts.sum():.0f} entries but none "
                  f"at normal incidence -- needs more statistics")
            continue
        p = angular.write_angular_response(
            outdir / "angular" / f"angResp_{r:g}.root", edges, counts,
            total.angular_sumw2[r], shell_r_cm=r)
        print(f"  {p}: {counts.sum():.0f} entries")
        written += 1
    if written == 0:
        print("  no angular shell had enough statistics to normalise", file=sys.stderr)
        return 1
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
    acc.add_argument("--s-max", type=float, default=None,
                     help="pin gsthr (cm) instead of deriving it; use this to "
                          "keep the split jobs of one cell consistent")
    acc.add_argument("--smax-quantile", type=float, default=0.90,
                     help="cumulative-emission fraction defining gsthr; 0.90 "
                          "reproduces the reference tables (default: %(default)s)")
    acc.set_defaults(func=_cmd_cprofile_accumulate)

    bld = cp_actions.add_parser("build", help="merge cells into CProf_<pdg>_WCSim.root")
    bld.add_argument("cells", nargs="+", type=Path)
    bld.add_argument("--pdg", type=int, required=True, choices=sorted(binning.PDG_NAMES))
    bld.add_argument("-o", "--output", type=Path, default=None)
    bld.set_defaults(func=_cmd_cprofile_build)

    sa = stages.add_parser(
        "sample", help="isotropic sample -> angular response + indirect-light tables")
    sa_actions = sa.add_subparsers(dest="action", required=True)

    sacc = sa_actions.add_parser("accumulate", help="reduce one propagated shard")
    sacc.add_argument("input", type=Path, help="photon-shotgun per-photon HDF5")
    sacc.add_argument("-o", "--output", type=Path, required=True, help="shard .npz")
    sacc.add_argument("--geometry", type=Path, required=True,
                      help="detector geometry .npz (positions_mm, directions, ...)")
    sacc.add_argument("--shells", type=float, nargs="+",
                      default=[100.0, 200.0, 400.0, 800.0, 1200.0],
                      help="angular-response shell radii in cm (default: %(default)s)")
    sacc.set_defaults(func=_cmd_sample_accumulate)

    sbld = sa_actions.add_parser("build", help="merge shards into the tuning inputs")
    sbld.add_argument("shards", nargs="+", type=Path)
    sbld.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    sbld.set_defaults(func=_cmd_sample_build)

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
