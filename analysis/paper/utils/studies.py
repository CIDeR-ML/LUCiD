"""Build tracking-study config dicts for the paper figures.

These are ordinary ``analysis/paper`` configs (the same schema ``run_study.py``
and ``submit_job.py`` consume) — just assembled programmatically so a figure can
scale itself between a small local run and the full S3DF production by changing
one argument (``n_events``).

The figure scripts build every config from the constants here, so this module defines
the published working point once.

The reconstruction settings below are the paper defaults: contained, random true
t0 (+-15 ns), uncorrected, TTS = 2.1 ns with a matched likelihood kernel
(``sigma == TTS``), and the ``Abe_2013`` per-PMT charge-resolution model.
"""
from pathlib import Path

# Paper-default reconstruction settings (the TTS=2.1 / Abe_2013 published set).
TTS = 2.1
GN = {'fisher_mode': 'ad', 'lr': 4.0, 'nkeys': 8, 'niters': 150, 'sigma': 2.1}
CHARGE_RESOLUTION = 'Abe_2013'

# A muon at 1 GeV, water, SK-like geometry — the nominal working point.
_PART_DIR = {'muon': 'mu-', 'electron': 'e-'}
_MASS = {'muon': 105.658, 'electron': 0.511}

NRAYS_TAGS = {'5k': 5000, '10k': 10000, '25k': 25000, '50k': 50000,
              '100k': 100000, '150k': 150000, '250k': 250000}


def default_root(particle: str, energy_mev: int, n_events: int, root_base: str) -> str:
    """Path to the PhotonSim ROOT for (particle, energy). Overridable via root_base."""
    return f'{root_base}/{_PART_DIR[particle]}/{energy_mev}MeV_{n_events}events.root'


def base_config(particle, energy_mev, n_rays, n_events, root_file, name,
                study='nrays', geom_config='config/SK_like_geom_config.json',
                time_weight=1.0, tts=TTS, charge_resolution=CHARGE_RESOLUTION, gn=None):
    """One reconstruction config (paper defaults)."""
    g = dict(GN if gn is None else gn)
    if time_weight != 1.0:
        g['time_weight'] = time_weight
    return {
        'name': name,
        'particle': particle,
        'study': study,
        'geom_config': geom_config,
        'phys_config': 'config/SK_like_physics_config.json',
        'root_file': root_file,
        'energy_nominal_MeV': energy_mev,
        'n_events': n_events,
        'event_start': 0,
        'n_rays': n_rays,
        'placement_seed_stride': 1000,
        'true_t0_range': [-15.0, 15.0],
        'tts': tts,
        # data-sim charge-resolution model: None | "Abe_2013" | "Bellamy_94"
        'charge_resolution': charge_resolution,
        'gn': g,
    }


# Paper working points for the energy and geometry scans.
ENERGIES = list(range(400, 1801, 100))            # 400..1800 MeV (paper range)
W075_ENERGIES = [1600, 1700, 1800]                # time-weight 0.75 above the crossover
W_CROSSOVER = 1600                                # w=1 below, w=0.75 at/above
SENSORS = list(range(2000, 18001, 2000))          # 2k..18k sensors


def energy_configs(particle, n_events, energies=None, time_weight=1.0, n_rays=250000,
                   root_base=None, root_events=500, name_prefix='escan'):
    """The energy scan for one particle: one config per energy."""
    energies = energies or ENERGIES
    root_base = root_base or '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water'
    out = []
    for E in energies:
        root = default_root(particle, E, root_events, root_base)
        cfg = base_config(particle, E, n_rays, n_events, root, name=f'{name_prefix}_{E}',
                          study='energy', time_weight=time_weight)
        out.append(cfg)
    return out


def geom_configs(n_events, sensors=None, energy_mev=1000, n_rays=250000, particle='muon',
                 root_base=None, root_events=500):
    """The geometry scan: one config per sensor count (1 GeV muons)."""
    sensors = sensors or SENSORS
    root_base = root_base or '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water'
    root = default_root(particle, energy_mev, root_events, root_base)
    return [base_config(particle, energy_mev, n_rays, n_events, root, name=f'gscan_{N}',
                        study='geom',
                        geom_config=f'analysis/paper/geometries/SK_like_{N}_geom_config.json')
            for N in sensors]


def nrays_configs(particle, n_events, tags=None, energy_mev=1000, root_base=None,
                  root_events=500):
    """The nrays scan for one particle: one config per ray-count tag."""
    tags = tags or list(NRAYS_TAGS)
    root_base = root_base or '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water'
    root = default_root(particle, energy_mev, root_events, root_base)
    pfx = {'muon': 'mu', 'electron': 'el'}[particle]
    return [base_config(particle, energy_mev, NRAYS_TAGS[t], n_events, root,
                        name=f'nrays_{pfx}_{t}') for t in tags]


def mass(particle):
    return _MASS[particle]
