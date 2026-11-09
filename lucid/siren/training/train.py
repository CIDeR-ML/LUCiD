#!/usr/bin/env python3
"""
PhotonSim SIREN Training Script: ``lucid-train-siren``.

Train a SIREN emitter on a PhotonSim lookup table (``lucid-build-photon-table`` /
``lucid-build-dedx-table``). The kind (Cherenkov or dE/dx), material and particle come from the
table. The working point is ``--recipe`` (default: the committed ``nominal.json``, the recipe of
every shipped emitter), with any field overridden by its flag. Into ``--output-dir`` go:

- ``emitter.json`` + ``weights.npz``: the emitter, with the recipe and losses recorded
  (``lucid-siren-promote`` installs it in ``data/<material>/<particle>/<kind>/``);
- ``training_history.json``, ``final_training_progress.pdf``, ``prediction_plots/``, and
  ``checkpoint_step_<N>.npz`` with ``--checkpoint-every``.

Usage:
    lucid-train-siren --h5-path T --output-dir D [--recipe R] [--<field> V ...]
                      [--init W] [--seed N]

    # a seed of the nominal recipe
    lucid-train-siren --h5-path photon_lookup_table.h5 --output-dir runs/s1 --seed 1
    # fine-tune water's emitter for a new material
    lucid-train-siren --h5-path wbls/photon_lookup_table.h5 --output-dir runs/wbls \\
        --init data/water/muon/cherenkov/weights.npz --num-steps 20000

A run never overwrites an emitter: an ``--output-dir`` that holds ``emitter.json`` is refused.
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lucid.siren.training.dataset import PhotonSimDataset
from lucid.siren.training.trainer import NOMINAL, Recipe, SIRENTrainer

logger = logging.getLogger(__name__)

_TYPES = {'int': int, 'float': float, 'str': str}


def _bool(s: str) -> bool:
    if s not in ('true', 'false'):
        raise argparse.ArgumentTypeError(f"expected true or false, got {s!r}")
    return s == 'true'


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description='Train a SIREN emitter on a PhotonSim table',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument('--h5-path', type=Path, required=True, help='the lookup table (.h5)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='where the emitter, history and plots go (must hold no emitter.json)')
    parser.add_argument('--recipe', type=Path, default=NOMINAL,
                        help='recipe JSON (default: the committed nominal.json)')
    fields = dataclasses.fields(Recipe)
    for f in fields:
        parser.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, default=None,
                            type=_bool if f.type == 'bool' else _TYPES[f.type],
                            help=f'override the recipe field {f.name}')
    parser.add_argument('--init', type=Path, default=None,
                        help='weights.npz to start from (an emitter or a checkpoint)')
    parser.add_argument('--log-every', type=int, default=10, help='history interval (steps)')
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='write checkpoint_step_<N>.npz every N steps (default 0: never)')
    parser.add_argument('--monitoring', action='store_true',
                        help='live loss/LR plots (Jupyter; needs IPython)')
    parser.add_argument('--notebook-mode', action='store_true',
                        help='Force notebook mode for plot display')

    # In-training prediction-vs-truth plots
    parser.add_argument('--prediction-plots', action=argparse.BooleanOptionalAction, default=True,
                        help='predicted-vs-table PDFs every --prediction-plot-every steps')
    parser.add_argument('--prediction-plot-every', type=int, default=10000,
                        help='refresh interval in steps (default: 10000)')
    parser.add_argument('--prediction-plot-energies', type=str,
                        default='200,1000,2000,5000,10000,80000',
                        help='energies in MeV (default: 200,1000,2000,5000,10000,80000)')
    parser.add_argument('--prediction-plot-distance-slices', type=str,
                        default='0.1,0.25,0.5,0.75',
                        help='s/s_max slices in [0, 1] (default: 0.1,0.25,0.5,0.75)')
    parser.add_argument('--prediction-plot-angle-slices-deg', type=str, default='21,41,61,90',
                        help='Cherenkov tables: angle slices in degrees (default: 21,41,61,90)')
    parser.add_argument('--prediction-plot-dedx-slices', type=str, default='100,250,500,800',
                        help='dE/dx tables: slices in keV/mm (default: 100,250,500,800)')
    args = parser.parse_args(argv)

    out = args.output_dir
    if (out / 'emitter.json').exists():
        parser.error(f"{out} already holds an emitter; choose another --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    recipe = Recipe.load(args.recipe, **{f.name: getattr(args, f.name) for f in fields
                                         if getattr(args, f.name) is not None})
    # The resolved recipe, for a run in progress (TrainingMonitor); emitter.json records it too.
    (out / 'recipe.json').write_text(json.dumps(dataclasses.asdict(recipe), indent=1) + '\n')
    logger.info("recipe: %s", dataclasses.asdict(recipe))

    dataset = PhotonSimDataset(args.h5_path, val_split=recipe.val_split,
                               zero_threshold=recipe.zero_threshold,
                               zero_keep_frac=recipe.zero_keep_frac,
                               energy_balance=recipe.energy_balance,
                               target_importance=recipe.target_importance)
    trainer = SIRENTrainer(dataset, recipe, output_dir=out, init=args.init,
                           log_every=args.log_every, checkpoint_every=args.checkpoint_every)

    if args.notebook_mode:
        from lucid.siren.training.monitor import setup_matplotlib_for_notebook
        setup_matplotlib_for_notebook()
    if args.monitoring:
        from lucid.siren.training.monitor import LiveTrainingCallback, TrainingMonitor
        trainer.add_callback(LiveTrainingCallback(TrainingMonitor(out, live_plotting=True),
                                                  update_every=args.log_every * 5,
                                                  plot_every=args.log_every * 20))
    if args.prediction_plots:
        from lucid.siren.training.prediction_plot import (
            PredictionComparisonCallback, parse_float_list, parse_int_list,
        )
        # The x-axis slices in the table's units: radians (the CLI takes degrees) or keV/mm.
        if dataset.kind == 'cherenkov':
            xaxis_slices = np.radians(parse_float_list(args.prediction_plot_angle_slices_deg))
        else:
            xaxis_slices = parse_float_list(args.prediction_plot_dedx_slices)
        trainer.add_callback(PredictionComparisonCallback(
            h5_path=args.h5_path, dataset=dataset, output_dir=out,
            energies_mev=parse_int_list(args.prediction_plot_energies),
            distance_slices=parse_float_list(args.prediction_plot_distance_slices),
            xaxis_slices=xaxis_slices, every=args.prediction_plot_every))

    trainer.train()
    trainer.save_history(out)
    plt.close(trainer.plot_training_history(save_path=out / 'final_training_progress.pdf'))
    trainer.save_emitter(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
