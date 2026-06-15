"""Scaling + wavelength check across energies and particles (muon, electron).

For each downloaded ROOT (new OpticalPhotonsRaw + OpticalPhotons schema) reports:
  - GEANT4 mean NOpticalPhotons/event vs the net's nphot(E)  -> SCALING ratio
  - PhotonWavelength [min,max] emission band                 -> WAVELENGTH band
so we can confirm the net's yield matches GEANT4 at every energy for each particle,
and the emission band is consistent before running optimization.
"""
import os, sys, glob, json
import numpy as np, uproot
_ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'; sys.path.insert(0, _ROOT)
DL = '/sdf/group/neutrino/omara/LUCiD_dlcheck/data/water'


def net_nphot_fn(particle):
    md = json.load(open(f'{DL}/{particle}/siren_training/trained_model/photonsim_siren_metadata.json'))
    n = md['nphot']; a, b, c = float(n['a']), float(n['b']), float(n['c'])
    return lambda E: a * E ** b + c


for particle in ('muon', 'electron'):
    nphot_fn = net_nphot_fn(particle)
    roots = sorted(glob.glob(f'{DL}/{particle}/*MeV_*events.root'))
    print(f'\n=== {particle.upper()} (net nphot = a*E^b+c) ===', flush=True)
    print(f'{"file":>26} {"E":>6} {"G4 NOpt/ev":>12} {"net nphot":>11} {"net/G4":>7} {"wl[min,max]":>16}', flush=True)
    for rp in roots:
        try:
            f = uproot.open(rp)
            md = f['OpticalPhotons']
            E = float(md['PrimaryEnergy'].array(library='np')[0])
            N = md['NOpticalPhotons'].array(library='np')
            g4 = float(np.mean(N))
            net = float(nphot_fn(E))
            # wavelength band from first ~3 chunks of the raw tree
            raw = f['OpticalPhotonsRaw']
            wl = np.concatenate([np.asarray(x, np.float64)
                                 for x in raw['PhotonWavelength'].array(library='np', entry_stop=3)])
            print(f'{os.path.basename(rp):>26} {E:6.0f} {g4:12.0f} {net:11.0f} {net/g4:7.3f} '
                  f'[{wl.min():6.1f},{wl.max():6.1f}]', flush=True)
        except Exception as e:
            print(f'{os.path.basename(rp):>26}  SKIP ({type(e).__name__}: partial/unreadable)', flush=True)
print('\nDONE', flush=True)
