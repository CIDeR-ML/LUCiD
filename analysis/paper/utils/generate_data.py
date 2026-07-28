#!/usr/bin/env python3
"""Generate the PhotonSim energy-sweep ROOTs for the tracking 'energy' study.

Monoenergetic, individual-photon (data-like) events — the SAME recipe that made the
distributed ``<E>MeV_100events.root`` files, at 500 events/energy for BOTH mu- and e-:

    energies : 300..2100 MeV, step 100  (19 points)
    events   : 500 per file
    output   : <BASE>/<mu-|e->/<E>MeV_500events.root

One CPU SLURM job per (particle, energy) runs the baked PhotonSim binary on a generated
macro (milano, mli:cider-ml). This mirrors the existing
``.../ROOT_files/LARGE_files/water/mu-/submit_gen.sh`` but adds electrons and the full sweep.

    # write macros + sbatch scripts, do NOT queue (inspect first)
    python analysis/paper/utils/generate_data.py
    # write and queue everything (replacing stale files in BASE)
    python analysis/paper/utils/generate_data.py --submit --clean

muon macros disable the muon-decay processes for a clean single track; electrons are stable
so that block is dropped (per project decision).
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE = '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water'
IMAGE = '/sdf/data/neutrino/cjesus/software/images/lucid.sif'
PHOTONSIM_BIN = '/opt/PhotonSim/build/PhotonSim'
BINDS = '/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs'
ACCOUNT = 'mli:cider-ml'
PARTITION = 'milano'
BEAM_ON = 500
ENERGIES = list(range(300, 2101, 100))          # 300..2100 step 100
PART_DIR = {'muon': 'mu-', 'electron': 'e-'}

# muon-specific: inactivate decay (proc 1) + muMinusCaptureAtRest (7) on mu-/mu+.
MU_DECAY_BLOCK = """/particle/select mu-
/particle/process/inactivate 1
/particle/process/inactivate 7
/particle/select mu+
/particle/process/inactivate 1
"""


def macro_text(particle, energy, out_root):
    g4name = PART_DIR[particle]
    seed_off = 1 if particle == 'muon' else 5      # keep mu-/e- seed streams distinct
    s1, s2 = energy * 10 + seed_off, energy * 10 + seed_off + 1
    decay = MU_DECAY_BLOCK if particle == 'muon' else ''
    decay_hdr = ("# Disable muon decay (clean single track, matches production)\n" + decay
                 if decay else "# Electrons are stable -- no decay-process inactivation needed.\n")
    return f"""# PhotonSim: {BEAM_ON} {g4name} events in water at {energy} MeV, photons + segments stored.
/output/filename {out_root}

/run/initialize

# Per-photon storage on (segments + track info are always written)
/photon/storeIndividual true

{decay_hdr}
/gun/clearPrimaries
/gun/addPrimary {g4name} {energy} MeV
/gun/position 0 0 0 m
/gun/direction 0 0 1

/random/setSeeds {s1} {s2}

/run/beamOn {BEAM_ON}
"""


def sbatch_text(particle, energy, macro, log_dir):
    return f"""#!/bin/bash
#SBATCH --partition={PARTITION}
#SBATCH --account={ACCOUNT}
#SBATCH --job-name=psim_{PART_DIR[particle]}_{energy}
#SBATCH --output={log_dir}/run_{energy}MeV-%j.out
#SBATCH --error={log_dir}/run_{energy}MeV-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=06:00:00

apptainer exec -B {BINDS} {IMAGE} \\
    {PHOTONSIM_BIN} {macro}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--particles', nargs='+', default=['muon', 'electron'],
                    choices=['muon', 'electron'])
    ap.add_argument('--energies', nargs='+', type=int, default=ENERGIES)
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--submit', action='store_true', help='queue the jobs (else just write files)')
    ap.add_argument('--clean', action='store_true',
                    help='remove existing *events.root in each particle dir not in this sweep')
    args = ap.parse_args()

    wanted = {f'{E}MeV_{BEAM_ON}events.root' for E in args.energies}

    for particle in args.particles:
        pdir = Path(args.base) / PART_DIR[particle]
        macro_dir = pdir / 'macros'; log_dir = pdir / 'logs'; slurm_dir = pdir / 'slurm'
        for d in (macro_dir, log_dir, slurm_dir):
            d.mkdir(parents=True, exist_ok=True)

        if args.clean:
            for f in pdir.glob('*events.root'):
                if f.name not in wanted:
                    print(f"  rm stale {f}"); f.unlink()

        print(f"\n{particle} ({PART_DIR[particle]}) -> {pdir}")
        for E in args.energies:
            out_root = f'{pdir}/{E}MeV_{BEAM_ON}events.root'
            macro = macro_dir / f'gen_{E}MeV.mac'
            macro.write_text(macro_text(particle, E, out_root))
            script = slurm_dir / f'gen_{E}MeV.sh'
            script.write_text(sbatch_text(particle, E, macro, log_dir)); script.chmod(0o755)
            if args.submit:
                subprocess.run(['sbatch', str(script)], check=True)
                print(f"  submitted {E} MeV")
            else:
                print(f"  wrote {macro.name} + {script.name}")

    if not args.submit:
        print("\nDry run. Re-run with --submit (add --clean to drop stale files first).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
