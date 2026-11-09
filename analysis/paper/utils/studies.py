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
import copy
from dataclasses import asdict, replace

from lucid.fitting import FIT_RECIPE, MODEL_RECIPE, SEED_RECIPE
from lucid.pdg import PDG_MASS_MEV

from analysis.paper.utils import paths

# The published reconstruction working point is the library's: FIT_RECIPE (the fit),
# MODEL_RECIPE (predictor and likelihood) and SEED_RECIPE (the seeder), recorded in full in every
# config. The data's TTS is the study's; the likelihood kernel sigma is matched to it.
TTS = 2.1
CHARGE_RESOLUTION = 'Abe_2013'

# The rest of the published working point: data simulation and event placement settings every
# study shares. TrackingPipeline indexes each of them; a config without one raises.
PIPELINE = {
    'nbuf': 'auto',                      # data buffer: 'auto' = max photons/event measured from
                                         # the ROOT file (a fixed int overrides)
    # propagation grid of the data simulator and the model; 8 candidates per ray instead of the
    # library's 4 (the most sensors one cell holds), so the soft model's deposit can reach a sensor
    # of a neighbouring cell. 4 loses vertex and t0 accuracy at low ray counts; 16 fits no better
    # than 8 and takes ~1.6x the time per event (paired scan, 2026-11-06)
    'grid': {'n_cap': 80, 'n_angular': 120, 'n_height': 80, 'max_candidates_per_ray': 8},
    # placement (lucid.sources.placement): the vertex uniform in the cylinder r <= fidr,
    # |z| <= fidz (m), a uniform rotation about it, redrawn (at most placement_max_tries times)
    # until vertex + s_max(E)·dir is inside the detector volume scaled by containment_margin
    'fidr': 12.0,
    'fidz': 12.0,
    'containment_margin': 0.95,
    'placement_max_tries': 200,
    # every per-event stream (rotation, vertex, t0, data simulation, seeder) is
    # derive_event_keys(event_seed, 0, event): an event index is the same placed event in every
    # study
    'event_seed': 100003,
    # true t0 per event (ns), uniform in [lo, hi], added to the hit times
    'true_t0_range': [-15.0, 15.0],
}

# A muon at 1 GeV, water, SK-like geometry — the nominal working point.
_PART_DIR = {'muon': 'mu-', 'electron': 'e-'}
_MASS = {'muon': PDG_MASS_MEV[13], 'electron': PDG_MASS_MEV[11]}

NRAYS_TAGS = {'5k': 5000, '10k': 10000, '25k': 25000, '50k': 50000,
              '100k': 100000, '150k': 150000, '250k': 250000}


def default_root(particle: str, energy_mev: int, n_events: int, root_base: str) -> str:
    """Path to the PhotonSim ROOT for (particle, energy). Overridable via root_base."""
    return f'{root_base}/{_PART_DIR[particle]}/{energy_mev}MeV_{n_events}events.root'


def base_config(particle, energy_mev, n_rays, n_events, root_file, name,
                study='nrays', geom_config='config/SK_like_geom_config.json',
                time_weight=1.0, fit=FIT_RECIPE):
    """One complete reconstruction config: the published working point plus this point's identity."""
    return {
        **copy.deepcopy(PIPELINE),
        'name': name,
        'particle': particle,
        'study': study,
        'geom_config': geom_config,
        'phys_config': 'config/SK_like_physics_config.json',
        'root_file': root_file,
        'energy_nominal_MeV': energy_mev,
        'n_events': n_events,
        'event_start': 0,
        'tts': TTS,
        # data-sim charge-resolution model: None | "Abe_2013" | "Bellamy_94"
        'charge_resolution': CHARGE_RESOLUTION,
        'model': asdict(replace(MODEL_RECIPE, n_rays=n_rays, time_weight=time_weight)),
        'fit': asdict(fit),
        # the seeder: the library's published recipe, unchanged. Its energy window (200..3000 MeV
        # in 50 MeV steps) covers every study energy, and its first-light t0 range (+-20 ns)
        # covers true_t0_range.
        'seed': asdict(SEED_RECIPE),
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
    root_base = root_base or paths.S3DF_ROOT_FILES
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
    root_base = root_base or paths.S3DF_ROOT_FILES
    root = default_root(particle, energy_mev, root_events, root_base)
    return [base_config(particle, energy_mev, n_rays, n_events, root, name=f'gscan_{N}',
                        study='geom',
                        geom_config=f'analysis/paper/geometries/SK_like_{N}_geom_config.json')
            for N in sensors]


def nrays_configs(particle, n_events, tags=None, energy_mev=1000, root_base=None,
                  root_events=500):
    """The nrays scan for one particle: one config per ray-count tag."""
    tags = tags or list(NRAYS_TAGS)
    root_base = root_base or paths.S3DF_ROOT_FILES
    root = default_root(particle, energy_mev, root_events, root_base)
    pfx = {'muon': 'mu', 'electron': 'el'}[particle]
    return [base_config(particle, energy_mev, NRAYS_TAGS[t], n_events, root,
                        name=f'nrays_{pfx}_{t}') for t in tags]


def named_config(name, n_events, root_base=None):
    """The published study config called ``name`` (e.g. ``nrays_mu_250k``, ``escan_1200``,
    ``escanw75_1700``, ``gscan_8000``), with ``n_events`` events. Hand-written diagnostics start
    from one of these, so an event index means the same placed event as in the study."""
    every = (nrays_configs('muon', n_events, root_base=root_base)
             + nrays_configs('electron', n_events, root_base=root_base)
             + energy_configs('muon', n_events, root_base=root_base)
             + energy_configs('muon', n_events, energies=W075_ENERGIES, time_weight=0.75,
                              root_base=root_base, name_prefix='escanw75')
             + geom_configs(n_events, root_base=root_base))
    for cfg in every:
        if cfg['name'] == name:
            return cfg
    raise KeyError(f'no study config named {name!r}; known: {sorted(c["name"] for c in every)}')


def mass(particle):
    return _MASS[particle]
