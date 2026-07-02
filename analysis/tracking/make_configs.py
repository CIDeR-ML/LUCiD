#!/usr/bin/env python3
"""Generate the study config JSONs for the tracking campaigns.

Three study axes, each for muon and electron:

  nrays  — sweep the predictor photon budget ``n_rays`` at fixed 1000 MeV / SK_like.
  energy — sweep 300..2100 MeV (step 100) at fixed SK_like; reads <E>MeV_500events.root.
  geom   — sweep SK_like PMT count 2k..20k (step 1k) at fixed 1000 MeV; needs make_geometries.py.

Writes ``configs/<study>/<particle>/config_NN.json`` next to this script. Each JSON is merged
over pipeline.DEFAULT_CONFIG at load time, so only the study-varying keys are written.

    python analysis/tracking/make_geometries.py        # first, for the geom study
    python analysis/tracking/make_configs.py            # all studies, both particles
    python analysis/tracking/make_configs.py --studies nrays --particles muon --n-events 50
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / 'configs'
GEOM_DIR = HERE / 'geometries'                              # from make_geometries.py

# PhotonSim ROOTs (data_generation/submit_gen.sh writes these). muon->mu-, electron->e-.
ROOT_BASE = '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water'
PART_DIR = {'muon': 'mu-', 'electron': 'e-'}

PHYS = 'config/SK_like_physics_config.json'
SK_GEOM = 'config/SK_like_geom_config.json'

NRAYS_SWEEP = [5000, 10000, 25000, 50000, 100000, 150000, 250000]
ENERGIES = list(range(300, 2101, 100))                     # 300..2100 step 100 (19)
SENSOR_COUNTS = list(range(2000, 20001, 1000))             # must match make_geometries.py


def _root(particle, energy):
    return f'{ROOT_BASE}/{PART_DIR[particle]}/{energy}MeV_500events.root'


def _nrays_for_energy(E):
    """Predictor photon budget scaled modestly with energy (more true light at high E)."""
    return 100_000 if E < 800 else 150_000 if E < 1500 else 250_000


def _write(study, particle, idx, cfg):
    out_dir = CONFIG_DIR / study / particle
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg['name'] = f'config_{idx:02d}'
    cfg['particle'] = particle
    cfg['study'] = study
    cfg['phys_config'] = PHYS
    path = out_dir / f'config_{idx:02d}.json'
    path.write_text(json.dumps(cfg, indent=2))
    return path


def gen_nrays(particle, n_events):
    for i, nr in enumerate(NRAYS_SWEEP):
        _write('nrays', particle, i, dict(
            geom_config=SK_GEOM, root_file=_root(particle, 1000),
            energy_nominal_MeV=1000, n_rays=nr, n_events=n_events))
    return len(NRAYS_SWEEP)


def gen_energy(particle, n_events):
    for i, E in enumerate(ENERGIES):
        _write('energy', particle, i, dict(
            geom_config=SK_GEOM, root_file=_root(particle, E),
            energy_nominal_MeV=E, n_rays=_nrays_for_energy(E), n_events=n_events))
    return len(ENERGIES)


def gen_geom(particle, n_events):
    for i, n in enumerate(SENSOR_COUNTS):
        geom = (GEOM_DIR / f'SK_like_{n}_geom_config.json').relative_to(REPO_ROOT)
        _write('geom', particle, i, dict(
            geom_config=str(geom), root_file=_root(particle, 1000),
            energy_nominal_MeV=1000, n_rays=250_000, n_events=n_events))
    return len(SENSOR_COUNTS)


GENERATORS = {'nrays': gen_nrays, 'energy': gen_energy, 'geom': gen_geom}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--studies', nargs='+', default=list(GENERATORS),
                    choices=list(GENERATORS))
    ap.add_argument('--particles', nargs='+', default=['muon', 'electron'],
                    choices=['muon', 'electron'])
    ap.add_argument('--n-events', type=int, default=100, help='events per config (default 100)')
    args = ap.parse_args()

    for study in args.studies:
        for particle in args.particles:
            n = GENERATORS[study](particle, args.n_events)
            print(f"{study:6s} / {particle:8s}: {n} configs -> "
                  f"{(CONFIG_DIR / study / particle).relative_to(REPO_ROOT)}")
    print(f"\nDone. n_events={args.n_events} per config. "
          f"(geom study requires: python analysis/tracking/make_geometries.py)")


if __name__ == '__main__':
    main()
