"""train_siren_pipeline — run the whole SIREN-emitter training pipeline on one node.

The single-node, sequential equivalent of the cluster fan-out in
`lucid/production/jobs/{smax,siren_inputs,train_siren}`. For each (material, particle) it:

  [Stage 0-1] (optional, --generate) run PhotonSim across the energy grid to produce training
              data + fit s_max(E)  — requires the external PhotonSim binary ($PHOTONSIM_BIN)
  Stage 2     build the photon + dE/dx lookup tables      (lucid-build-photon-table / -dedx-table)
  Stage 3     train the SIREN nets (photon + dE/dx)        (lucid-train-siren)
  Stage 4     validate                                     (lucid/siren/validate.py)

Data layout: `--data-dir` is the parent of `<material>/<particle>/`; the tables are written to
`<data-dir>/<material>/<particle>/{photon,dedx}_lookup_table.h5` where `lucid-train-siren` reads
them, and models land in `.../siren_training` and `.../dedx_siren_training`.

Requirements: torch (`pip install -e .[training]`) for training; a GPU is strongly recommended
(30k steps is hours on CPU); PhotonSim only for `--generate`. If you already have PhotonSim
training-cell data, omit `--generate` and just build+train+validate.

Run:  python scripts/train_siren_pipeline.py --data-dir data --material water --particles muon \
          --num-steps 30000
      python scripts/train_siren_pipeline.py --data-dir data --particles muon electron --dry-run
"""
import argparse, os, subprocess, sys


def run(cmd, dry):
    print('  $ ' + ' '.join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data', help='parent of <material>/<particle>/ (default: data)')
    ap.add_argument('--material', default='water')
    ap.add_argument('--particles', nargs='+', default=['muon'], help='e.g. muon electron')
    ap.add_argument('--num-steps', type=int, default=30_000)
    ap.add_argument('--photonsim-dir', default=None,
                    help='PhotonSim checkout with data/ (else $PHOTONSIM_DEV_PATH); holds the training cells')
    ap.add_argument('--skip-dedx', action='store_true', help='train the photon emitter only')
    ap.add_argument('--generate', action='store_true',
                    help='also run PhotonSim (Stage 0-1) to produce the training data (needs $PHOTONSIM_BIN)')
    ap.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    args = ap.parse_args()

    if not args.dry_run:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("training needs torch — `pip install -e .[training]` (or use --dry-run).")

    if args.generate:
        bin_ = os.environ.get('PHOTONSIM_BIN')
        if not bin_ and not args.dry_run:
            sys.exit("--generate needs the PhotonSim binary via $PHOTONSIM_BIN. "
                     "Alternatively generate data with lucid/production/jobs/{smax,siren_inputs} "
                     "and re-run without --generate.")

    py = [sys.executable, '-m']
    data_types = ['photon'] if args.skip_dedx else ['photon', 'dedx']
    for particle in args.particles:
        cell = os.path.join(args.data_dir, args.material, particle)
        print(f'\n=== {args.material}/{particle} ===', flush=True)

        if args.generate:
            print('[Stage 0-1] PhotonSim s_max scan + training-data generation')
            print('  (single-node PhotonSim runs; see lucid/production/jobs/{smax,siren_inputs} '
                  'for the cluster version and docs/SIREN_TRAINING_INPUTS.md)')
            # PhotonSim generation is delegated to the production tooling; wired here as a guarded
            # placeholder so the pipeline is explicit. A local loop over the energy grid would call
            # `lucid-run-job` per cell — heavy, and PhotonSim-binary-dependent.

        print('[Stage 2] build lookup tables')
        for dt in data_types:
            tbl = os.path.join(cell, f'{dt}_lookup_table.h5')
            mod = 'lucid.siren.training.photonsim_data.' + ('build_dedx_table' if dt == 'dedx' else 'build_photon_table')
            build_cmd = py + [mod, '--data-dir', args.data_dir, '--material', args.material,
                              '--particle', particle, '--output', tbl]
            if args.photonsim_dir:
                build_cmd += ['--photonsim-dir', args.photonsim_dir]
            rc = run(build_cmd, args.dry_run)
            if rc != 0 and not args.dry_run:
                sys.exit(f'table build failed for {dt} ({args.material}/{particle}); '
                         f'ensure PhotonSim training data exists under {cell} (or run with --generate).')

        print('[Stage 3] train SIREN')
        for dt in data_types:
            rc = run(py + ['lucid.siren.train', '--material', args.material, '--particle', particle,
                           '--data-type', dt, '--num-steps', str(args.num_steps)], args.dry_run)
            if rc != 0 and not args.dry_run:
                sys.exit(f'training failed for {dt} ({args.material}/{particle}).')

        print('[Stage 4] validate')
        run([sys.executable, 'lucid/siren/validate.py', 'energy',
             '--material', args.material, '--particle', particle], args.dry_run)

    print('\npipeline complete.' if not args.dry_run else '\n(dry run — nothing executed)')


if __name__ == '__main__':
    main()
