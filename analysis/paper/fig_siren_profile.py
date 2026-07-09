#!/usr/bin/env python3
"""Figure: SIREN Cherenkov emission profiles (angle vs distance) at several energies.

Two figures, one per particle: the trained SIREN Cherenkov model evaluated over the
(opening angle, distance-along-track) plane at a set of energies. Uses the validation
suite's energy study (lucid.siren.validate.PhotonSimValidator.energy_study) and names
the output per particle so muon and electron don't overwrite each other.

    python fig_siren_profile.py                              # muon + electron, default energies
    python fig_siren_profile.py --particles muon --energies 500,1000,1500,2000
    python fig_siren_profile.py --threshold 0.005 --out <dir>

Model-only (no PhotonSim ROOT needed): reads the bundled SIREN models under
data/<material>/<particle>/siren_training/.
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

MATERIAL = 'water'
DEFAULT_ENERGIES = [500, 1000, 1500]
DEFAULT_THRESHOLD = 0.005                               # LogNorm vmin (masks the low tail)


def make_profile(particle, energies, threshold, out):
    """Render one particle's SIREN profile figure -> <out>/<particle>_siren_cherenkov_profile."""
    from lucid.siren.validate import PhotonSimValidator
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tmp = paths.data_dir('siren_profile', 'local') / particle
    tmp.mkdir(parents=True, exist_ok=True)

    v = PhotonSimValidator(material=MATERIAL, particle=particle)
    v.energy_study(energies=list(energies), threshold=threshold,
                   output_dir=str(tmp), fmt='png')
    for ext in ('png', 'pdf'):
        src = tmp / f'energy_study_threshold_{threshold}.{ext}'
        # energy_study writes png; re-save the same figure as pdf too if only png exists
        if not src.exists() and ext == 'pdf':
            continue
        dst = out / f'{particle}_siren_cherenkov_profile.{ext}'
        if src.exists():
            shutil.copyfile(src, dst)
    print(f'wrote {out}/{particle}_siren_cherenkov_profile.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--particles', default='muon,electron',
                    help='comma list (default: muon,electron)')
    ap.add_argument('--energies', default=None, help='comma list in MeV')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    energies = [float(x) for x in a.energies.split(',')] if a.energies else DEFAULT_ENERGIES
    out = Path(a.out) if a.out else paths.figure_dir()
    for particle in a.particles.split(','):
        make_profile(particle.strip(), energies, a.threshold, out)


if __name__ == '__main__':
    main()
