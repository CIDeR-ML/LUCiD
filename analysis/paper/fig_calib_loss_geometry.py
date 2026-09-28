#!/usr/bin/env python3
"""Figure: 2D calibration loss geometry — raw descent vs Fisher preconditioning.

A 51x51 grid over the two 405 nm optical parameters, scattering length (x) and absorption
length (y), all other parameters pinned at truth, illuminated by ONE collimated laser from
the top of the tank. Both panels show the same log10 L surface and differ only in the vector
field and the overlaid optimizer paths:

    left   raw steepest descent          -grad L
    right  Gauss-Newton                  -F^-1 grad L,  F = the per-point Fisher

The Neyman chi-square surface is a razor valley, so raw descent crawls along it while the
Fisher preconditioner turns the valley into a bowl and the step goes almost straight to the
minimum.

    python fig_calib_loss_geometry.py                  # compute surface + paths, then plot
    python fig_calib_loss_geometry.py --generate-data  # just compute
    python fig_calib_loss_geometry.py --plot-results   # just plot from existing data
    python fig_calib_loss_geometry.py --generate-data --shards 10   # the published production

Data generation is four stages, run in order by ``--generate-data``:

    1. surface       utils/compute_2d_neyman.py, sharded round-robin over the grid
    2. combine       utils/combine_2d.py   (shards are disjoint, so summing = union)
    3. trajectories  utils/traj_2d_neyman.py
    4. render        utils/calib_plots.loss_geometry

The published run used 10 shards on a 10-GPU node. ``--shards 1`` runs the whole grid in one
process, which is correct but slow. Stage 3 MUST carry the same K / N_PH / NK_* as stage 1,
which is why both read them from ``utils/calibration.py:LANDSCAPE_RECIPE`` rather than from
flags.

``XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`` is required: at JAX's 0.75 default a shard's 9.51 GiB
peak does not fit an 11.26 GiB card. ``calibration.LANDSCAPE_ENV`` sets it for every stage.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # LUCiD/

# This process is a LAUNCHER: every GPU stage runs in a subprocess. Importing `calibration`
# initialises JAX, which on a GPU node preallocates device 0 -- the card shard 0 is given -- and
# shard 0 then OOMs. So the parent is pinned to CPU BEFORE the import, and `_env` restores the
# real CUDA_VISIBLE_DEVICES for the children; it must be captured before the pin.
_REAL_CUDA = os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'

from analysis.paper.utils import paths, calibration, calib_plots   # noqa: E402

FIGURE = 'calib_loss_geometry'
UTILS = Path(__file__).resolve().parent / 'utils'


def _env(extra=None):
    e = {**os.environ, **calibration.LANDSCAPE_RECIPE, **calibration.LANDSCAPE_ENV}
    # Undo the parent's CPU pin for the child: it exists only to keep this launcher off the GPUs.
    # Leaving JAX_PLATFORMS=cpu in the child would silently run the whole landscape on CPU --
    # correct, and slow enough to look like a hang.
    e.pop('JAX_PLATFORMS', None)
    if _REAL_CUDA is None:
        e.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        e['CUDA_VISIBLE_DEVICES'] = _REAL_CUDA
    e.update(extra or {})          # per-shard CUDA_VISIBLE_DEVICES wins over the restore
    return e


def generate_data(shards, ddir):
    ddir.mkdir(parents=True, exist_ok=True)

    # Stage 1 -- surface. Clear old partials first, and stamp the launch time: without a
    # freshness check a run in which every shard OOMs leaves the PREVIOUS landscape npz in
    # place, and the figure silently re-renders stale data as if nothing had gone wrong.
    for old in ddir.glob('landscape2d_neyman_part*.npz'):
        old.unlink()
    stamp = time.time()

    procs = []
    for k in range(shards):
        extra = {'SHARD_ID': str(k), 'NSHARDS': str(shards)}
        if shards > 1:
            extra['CUDA_VISIBLE_DEVICES'] = str(k)      # one GPU per shard, as published
        log = ddir / f'shard_{k}.log'
        print(f'[{FIGURE}] shard {k}/{shards} -> {log}', flush=True)
        with open(log, 'w') as fh:
            procs.append(subprocess.Popen([sys.executable, str(UTILS / 'compute_2d_neyman.py')],
                                          env=_env(extra), stdout=fh, stderr=subprocess.STDOUT))
    failed = [k for k, p in enumerate(procs) if p.wait() != 0]
    if failed:
        raise SystemExit(f'[{FIGURE}] shards {failed} failed — see {ddir}/shard_*.log. '
                         f'NOT combining; any existing landscape npz is left untouched.')

    # Every shard must have written a partial since launch (see the stamp above).
    parts = sorted(ddir.glob('landscape2d_neyman_part*.npz'))
    fresh = [p for p in parts if p.stat().st_mtime >= stamp]
    if len(parts) != shards or len(fresh) != shards:
        raise SystemExit(f'[{FIGURE}] expected {shards} fresh partials, found {len(parts)} present / '
                         f'{len(fresh)} newer than launch. NOT combining — stale data would render '
                         f'silently. See {ddir}/shard_*.log.')

    # Stage 2 -- combine.
    if shards > 1:
        r = subprocess.run([sys.executable, str(UTILS / 'combine_2d.py')], env=_env())
        if r.returncode != 0:
            raise SystemExit(f'[{FIGURE}] combine failed (exit {r.returncode})')

    # Stage 3 -- trajectories. Same K / N_PH / NK_* as the surface, single device.
    print(f'[{FIGURE}] trajectories', flush=True)
    r = subprocess.run([sys.executable, str(UTILS / 'traj_2d_neyman.py')],
                       env=_env({'CUDA_VISIBLE_DEVICES': '0'} if shards > 1 else None))
    if r.returncode != 0:
        raise SystemExit(f'[{FIGURE}] trajectories failed (exit {r.returncode})')


def plot_results(ddir, out_fig):
    calib_plots.loss_geometry(ddir, out_fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--shards', type=int, default=calibration.LANDSCAPE_SHARDS,
                    help='parallel grid shards, one GPU each (published: 10)')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', default=None, help='figure output dir')
    a = ap.parse_args()

    ddir = Path(a.data_dir) if a.data_dir else paths.data_dir(FIGURE)
    out_fig = Path(a.out) if a.out else paths.figure_dir()

    both = not (a.generate_data or a.plot_results)      # no flags => do both
    if a.generate_data or both:
        generate_data(a.shards, ddir)
    if a.plot_results or both:
        plot_results(ddir, out_fig)


if __name__ == '__main__':
    main()
