#!/usr/bin/env python3
"""``lucid-siren-promote``: install a trained emitter in ``data/<material>/<particle>/<kind>/``.

A training run (``lucid-train-siren``) holds the net, its domain and decoding, and the table's
s_max(E) and N_photons(E) fits. Promotion adds what the simulator needs and the trainer cannot
know, checks the run against the s_max fit it was trained on, and writes one emitter through
:func:`lucid.siren.emitter.write_emitter`, validated as an installed emitter:

- ``--t0``: the t0 fit, ``timing_parameters.json`` from PhotonSim's
  ``tools/t0_correction/calculate_t0.py`` (the t0 stage, ``lucid.production.siren_inputs.t0``);
  required for, and only for, a Cherenkov emitter;
- ``--emission-band-nm LO HI``: the wavelength band over which PhotonSim counted the photons
  behind N_photons(E) (water: 274.91 673.83, PhotonSim's 1.84-4.51 eV optical table); required
  for, and only for, a Cherenkov emitter;
- ``--smax-fit``: the ``smax_fit.csv`` the run's table was built with; the run's ``s_max_mm``
  must equal it;
- provenance: the run, ``--selection`` (how it was chosen, e.g. a seed scan's ranking), the
  promotion time and commit.

Usage:
    lucid-siren-promote --run RUN --smax-fit smax_fit.csv \\
        [--t0 timing_parameters.json --emission-band-nm LO HI] [--selection TEXT] \\
        [--dest DIR] [--force]

``--dest`` defaults to ``data/<material>/<particle>/<kind>/`` of this checkout, from the run's
table; an emitter already there is replaced only with ``--force``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lucid.siren.emitter import (PHOTONSIM_PARTICLE, SCHEMA, T0_FORM, emitter_dir, read_weights,
                                 validate_record, write_emitter)


def t0_block(fit: dict) -> dict:
    """A ``calculate_t0.py`` fit (``timing_parameters.json``) as the emitter's ``t0`` block."""
    if fit.get('form') != T0_FORM:
        raise ValueError(f"t0 fit of form {fit.get('form')!r}; expected {T0_FORM!r} (the "
                         f"calculate_t0.py of PhotonSim's tools/t0_correction since May 2026)")
    source = dict(fit.get('provenance', {}))
    if 'source' in source:
        source['tool'] = source.pop('source')
    try:
        return {'form': fit['form'], 'c_mm_per_ns': fit['c_mm_per_ns'],
                'log10_A': fit['A']['log10_poly_logE'],
                'log10_lambda_mm': fit['lambda']['log10_poly_logE'],
                'beta': fit['beta']['poly_logE'], 'source': source}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"the t0 fit lacks {exc}: not a calculate_t0.py timing_parameters.json") from None


def smax_block(csv_path: Path) -> dict:
    """``smax_fit.csv`` as the ``s_max_mm`` block a table built from it gives the emitter."""
    from lucid.siren.training.dataset import table_curves
    from lucid.siren.training.photonsim_data.build_tables import read_smax_csv
    return table_curves({f'smax_{k}': v for k, v in read_smax_csv(csv_path).items()}, 'dedx')[0]


def _differences(a: dict, b: dict, where: str = '') -> list:
    """The keys where two ``s_max_mm`` blocks differ (numbers to 1e-9 relative)."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += _differences(x, y, f'{where}{k}.')
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not np.isclose(x, y, rtol=1e-9, atol=0.0):
                out.append(f'{where}{k}: run {x!r}, csv {y!r}')
        elif x != y:
            out.append(f'{where}{k}: run {x!r}, csv {y!r}')
    return out


def promote(run: Path, smax_fit: Path, t0: dict | None = None,
            emission_band_nm: tuple | None = None, selection: str | None = None) -> tuple:
    """``(record, params)``: the run in ``run`` as an installed emitter. Raises ``ValueError``
    when an input is missing, does not belong to the run, or the result is not valid."""
    rec = json.loads((run / 'emitter.json').read_text())
    validate_record(rec)
    kind, prov = rec['kind'], rec['provenance']
    if kind == 'cherenkov':
        missing = [f for f, v in (('--t0', t0), ('--emission-band-nm', emission_band_nm))
                   if v is None]
        if missing:
            raise ValueError(f"{run} is a Cherenkov emitter: the simulator needs its t0 fit and "
                             f"emission band; pass {' and '.join(missing)}")
    elif t0 is not None or emission_band_nm is not None:
        raise ValueError(f"{run} is a dE/dx emitter: it takes no --t0 or --emission-band-nm")

    want = (prov['material'], prov['particle'])
    with Path(smax_fit).open() as fh:
        row = next(csv.DictReader(fh), {})
    if (row.get('material'), row.get('particle')) != want:
        raise ValueError(f"{smax_fit} is the s_max fit of {row.get('material')}/"
                         f"{row.get('particle')}; the run is {want[0]}/{want[1]}")
    diff = _differences(rec['s_max_mm'], smax_block(smax_fit))
    if diff:
        raise ValueError(f"{run} was trained on a table whose s_max fit is not {smax_fit}: "
                         + '; '.join(diff))
    if t0 is not None and (t0.get('material'), t0.get('particle')) != want:
        raise ValueError(f"the t0 fit is of {t0.get('material')}/{t0.get('particle')}; the run is "
                         f"{want[0]}/{want[1]}")

    out = {}
    for k, v in rec.items():                    # the run's order, the additions before provenance
        if k in ('schema', 't0', 'emission_band_nm'):
            continue
        if k == 'provenance' and kind == 'cherenkov':
            out['t0'] = t0_block(t0)
            out['emission_band_nm'] = [float(x) for x in emission_band_nm]
        out[k] = v
    from lucid.utils import git_commit
    out['provenance'] = {**prov, 'run': str(run.resolve()), 'selection': selection,
                         'smax_fit': str(Path(smax_fit).resolve()),
                         'promoted_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                         'promoted_commit': git_commit()}
    validate_record({'schema': SCHEMA, **out}, promoted=True)
    return out, read_weights(run / 'weights.npz')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Install a trained SIREN emitter',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--run', type=Path, required=True,
                   help='the training run (its emitter.json and weights.npz)')
    p.add_argument('--smax-fit', type=Path, required=True,
                   help="the smax_fit.csv the run's table was built with")
    p.add_argument('--t0', type=Path, default=None,
                   help="Cherenkov: calculate_t0.py's timing_parameters.json")
    p.add_argument('--emission-band-nm', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                   help='Cherenkov: the band (nm) PhotonSim counted N_photons(E) over')
    p.add_argument('--selection', default=None,
                   help='how this run was chosen, recorded in the provenance')
    p.add_argument('--dest', type=Path, default=None,
                   help='output directory (default: data/<material>/<particle>/<kind>/)')
    p.add_argument('--force', action='store_true', help='replace an emitter already at --dest')
    args = p.parse_args(argv)

    for f in [args.run / 'emitter.json', args.run / 'weights.npz', args.smax_fit, args.t0]:
        if f is not None and not f.is_file():
            p.error(f"not found: {f}")
    try:
        t0 = json.loads(args.t0.read_text()) if args.t0 else None
        rec, params = promote(args.run, args.smax_fit, t0, args.emission_band_nm, args.selection)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dest = args.dest
    if dest is None:
        particle = {v: k for k, v in PHOTONSIM_PARTICLE.items()}.get(rec['provenance']['particle'])
        if particle is None:
            p.error(f"no LUCiD particle for {rec['provenance']['particle']!r}; pass --dest")
        dest = emitter_dir(particle, rec['provenance']['material'], rec['kind'])
    if (dest / 'emitter.json').exists() and not args.force:
        p.error(f"{dest} already holds an emitter; pass --force to replace it")
    write_emitter(dest, rec, params, promoted=True)
    print(f"promoted {args.run} -> {dest}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
