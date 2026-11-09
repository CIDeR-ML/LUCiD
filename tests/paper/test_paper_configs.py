"""The paper figures' config contract: every study config is complete, and nothing fills a gap.

``studies.base_config`` writes the whole published working point (the ``model``, ``fit`` and
``seed`` records and ``PIPELINE``), and
``TrackingPipeline`` refuses a config that lacks a key, so a hand-written config cannot silently
fall back to a different placement or simulator setting than the published studies.

Static: no PhotonSim ROOT input, no SIREN weights and no GPU, so the gate always runs.
"""
import ast
from dataclasses import asdict

import pytest
from tests.helpers.paths import REPO

PAPER = REPO / 'analysis' / 'paper'
PIPELINE = PAPER / 'utils' / 'pipeline.py'


def _indexed_keys():
    """Keys TrackingPipeline.__init__ indexes with config['x'], read from the source."""
    tree = ast.parse(PIPELINE.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'TrackingPipeline')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    keys = {n.slice.value for n in ast.walk(init)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == 'config' and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)}
    assert keys, 'found no config[...] accesses — the extractor has drifted from the source'
    return keys


def _every_study_config():
    from analysis.paper.utils import studies
    rb = '/nonexistent'
    return (studies.nrays_configs('muon', 2, root_base=rb)
            + studies.nrays_configs('electron', 2, root_base=rb)
            + studies.energy_configs('muon', 2, root_base=rb)
            + studies.energy_configs('muon', 2, energies=studies.W075_ENERGIES, time_weight=0.75,
                                     root_base=rb, name_prefix='escanw75')
            + studies.geom_configs(2, root_base=rb))


def test_every_study_config_supplies_what_the_pipeline_indexes():
    for cfg in _every_study_config():
        missing = sorted(_indexed_keys() - set(cfg))
        assert not missing, f"{cfg['name']}: TrackingPipeline would raise on {missing}"


def test_a_config_missing_a_key_is_refused_by_name():
    from analysis.paper.utils.pipeline import _check_complete
    cfg = _every_study_config()[0]
    del cfg['event_seed']
    with pytest.raises(KeyError, match='event_seed'):
        _check_complete(cfg)


def test_named_config_is_the_study_config():
    from analysis.paper.utils import studies
    for cfg in _every_study_config():
        assert studies.named_config(cfg['name'], 2, root_base='/nonexistent') == cfg
    with pytest.raises(KeyError, match='no study config named'):
        studies.named_config('nrays_mu_7k', 2)


def test_the_time_weight_is_075_exactly_on_the_high_energy_pass():
    """README.md ("Running a campaign on S3DF"): the energy scan runs time_weight=1.0 throughout
    plus a 0.75 pass at/above 1600 MeV (``W_CROSSOVER``). So w == 0.75 iff the config is in the
    ``escanw75`` pass, and that pass covers exactly the scan energies at/above the crossover.
    Every config states its w (the model record carries it), so none relies on ReconModel's
    default."""
    from analysis.paper.utils import studies
    assert studies.W_CROSSOVER == 1600
    assert studies.W075_ENERGIES == [E for E in studies.ENERGIES if E >= 1600]
    for cfg in _every_study_config():
        high_pass = cfg['name'].startswith('escanw75_')
        assert cfg['model']['time_weight'] == (0.75 if high_pass else 1.0), cfg['name']
        if high_pass:
            assert cfg['energy_nominal_MeV'] >= studies.W_CROSSOVER, cfg['name']


# ---------------------------------------------------------------------------------------------
# The published working point. Each value is the one analysis/paper/README.md ("Reconstruction
# defaults", "Tracking studies") states; the source is named per assertion,
# and the library recipes the paper records (FIT_RECIPE, MODEL_RECIPE) are pinned in full. A
# change to any of them changes what the published figures mean, so it must be a deliberate
# edit of this file as well as of studies.py or lucid.fitting.
# ---------------------------------------------------------------------------------------------

def test_every_config_reconstructs_at_the_published_working_point():
    from analysis.paper.utils import studies
    for cfg in _every_study_config():
        n, fit, model = cfg['name'], cfg['fit'], cfg['model']
        assert cfg['tts'] == 2.1, n                                # README: TTS 2.1 ns
        assert cfg['charge_resolution'] == 'Abe_2013', n           # README: Abe_2013
        assert (fit['lr'], fit['nkeys'], fit['steps']) == (4.0, 8, 150), n   # README: Fisher-GN
        assert fit == dict(nkeys=8, steps=150, lr=4.0, lr_final=1.5, lam=0.01, mu=0.1,
                           refresh=8, max_step=3.0, polyak=40), n
        assert {k: v for k, v in model.items() if k not in ('n_rays', 'time_weight')} == dict(
            K=8, temperature=0.1, delta=1.0, energy_scale='nphot'), n
        assert cfg['true_t0_range'] == [-15.0, 15.0], n            # README: random true t0 ±15 ns
        # README: "contained" vertices. The margin's value is studies.py's own; only that the
        # containment requirement is on is documented.
        assert cfg['containment_margin'] is not None, n
        # README (Tracking studies): the seeder is "the library's published recipe, unchanged".
        assert cfg['seed'] == asdict(studies.SEED_RECIPE), n


def test_event_placement_seeds_are_the_published_ones():
    """Event i draws every per-event stream from ``derive_event_keys(event_seed, 0, i)``.

    The README does not state the seed; it is pinned because a published HDF5 records only the
    config it ran with, so a changed seed silently places different vertices than the published
    runs while every figure still draws. One seed for every study makes an event index the same
    placed event in every study (F-032; before it, ``placement_seed_base + i *
    placement_seed_stride`` with base 100003, the same value).
    """
    for cfg in _every_study_config():
        assert cfg['event_seed'] == 100003, cfg['name']
        assert 'placement_seed_base' not in cfg and 'placement_seed_stride' not in cfg, cfg['name']


# derive_event_keys before the seeder's and the fit's streams were added (2026-11-01, tidy/C-seed;
# 2026-11-02, tidy/C3C): the values a published event was placed, timed and simulated with.
_PUBLISHED_STREAMS = {
    (100003, 0, 0): {'vertex_seed': 1807019899, 't0_seed': 893221958, 'scint_seed': 423615696,
                     'rot_seed': 1753058572, 'sim_key': [136710672, 1546091913]},
    (100003, 0, 7): {'vertex_seed': 638017827, 't0_seed': 714190626, 'scint_seed': 1770460378,
                     'rot_seed': 774646005, 'sim_key': [918463829, 2711357876]},
    (5, 2, 3, 1): {'vertex_seed': 620073529, 't0_seed': 1713299730, 'scint_seed': 599785069,
                   'rot_seed': 219752923, 'sim_key': [2809697725, 96264164]},
}


@pytest.mark.parametrize('args', sorted(_PUBLISHED_STREAMS), ids=str)
@pytest.mark.parametrize('name', ['seed_key', 'fit_key'])
def test_the_seeder_and_the_fit_have_their_own_streams_and_the_others_are_unchanged(args, name):
    """The pipeline seeds event i from ``derive_event_keys(event_seed, 0, i)['seed_key']`` and fits
    it from ``['fit_key']``. Each stream is folded off its own tag, so every stream an event
    already drew is bit-identical; each is the same for the same event and differs between events
    and from the event's other streams."""
    import numpy as np
    from lucid.sources.seed_utils import derive_event_keys
    keys = derive_event_keys(*args)
    got = {k: (v if isinstance(v, int) else np.asarray(v).tolist())
           for k, v in keys.items() if k not in ('seed_key', 'fit_key')}
    assert got == _PUBLISHED_STREAMS[args]
    key = np.asarray(keys[name]).tolist()
    assert key == np.asarray(derive_event_keys(*args)[name]).tolist()
    others = {'sim_key': got['sim_key'],
              **{k: np.asarray(keys[k]).tolist() for k in ('seed_key', 'fit_key') if k != name}}
    assert key not in others.values()
    other = (*args[:2], args[2] + 1, *args[3:])
    assert key != np.asarray(derive_event_keys(*other)[name]).tolist()


def test_the_sweeps_are_the_documented_ones():
    """README.md ("Running a campaign on S3DF"): energy 400..1800 MeV step 100 at a flat 250k rays
    (muon); geom SK_like 2k..18k PMTs step 2k at 1000 MeV muon; nrays at 1000 MeV SK_like, muon and
    electron.

    The published nrays sets (nrays_{mu,el}_refl2) stop at 250k, as ``studies.NRAYS_TAGS`` does.
    """
    from analysis.paper.utils import studies
    assert studies.ENERGIES == list(range(400, 1801, 100))
    assert studies.SENSORS == list(range(2000, 18001, 2000))
    assert studies.NRAYS_TAGS == {
        '5k': 5000, '10k': 10000, '25k': 25000, '50k': 50000, '100k': 100000,
        '150k': 150000, '250k': 250000}
    rb = '/nonexistent'
    for cfg in studies.energy_configs('muon', 2, root_base=rb):
        assert (cfg['particle'], cfg['model']['n_rays']) == ('muon', 250000), cfg['name']
    for cfg in studies.geom_configs(2, root_base=rb):
        assert (cfg['particle'], cfg['model']['n_rays'], cfg['energy_nominal_MeV']) == \
            ('muon', 250000, 1000), cfg['name']
        n_pmt = int(cfg['name'].split('_')[1])
        assert cfg['geom_config'].endswith(f'SK_like_{n_pmt}_geom_config.json'), cfg['name']
    for particle in ('muon', 'electron'):
        for cfg in studies.nrays_configs(particle, 2, root_base=rb):
            assert cfg['energy_nominal_MeV'] == 1000, cfg['name']
            assert cfg['geom_config'].endswith('SK_like_geom_config.json'), cfg['name']


def test_the_seeder_covers_every_study_point():
    """studies.py: the seed recipe's energy window (200..3000 MeV) covers every study energy and
    its first-light t0 range (±20 ns) covers true_t0_range. A seed scan that misses the truth
    picks a start on the window edge, and the fit inherits it."""
    from analysis.paper.utils import studies
    r = studies.SEED_RECIPE
    e_lo, e_hi = r.energy_center - r.energy_delta, r.energy_center + r.energy_delta
    for cfg in _every_study_config():
        assert e_lo <= cfg['energy_nominal_MeV'] <= e_hi, cfg['name']
        lo, hi = cfg['true_t0_range']
        assert r.t0_min <= lo and hi <= r.t0_max, cfg['name']


def test_a_config_is_a_copy_not_a_view_of_the_working_point():
    """A hand-written diagnostic edits its config (README "Minimal local validation"
    sets ``c['fit']['steps'] = 15``); that must not leak into PIPELINE or the library recipes
    and so into every later config of the same process."""
    from analysis.paper.utils import studies
    cfg = studies.named_config('nrays_mu_250k', 2, root_base='/nonexistent')
    cfg['fit']['steps'] = 15
    cfg['seed']['t0_min'] = 0.0
    cfg['true_t0_range'][0] = 0.0
    again = studies.named_config('nrays_mu_250k', 2, root_base='/nonexistent')
    assert again['fit']['steps'] == studies.FIT_RECIPE.steps == 150
    assert again['seed']['t0_min'] == studies.SEED_RECIPE.t0_min == -20.0
    assert again['true_t0_range'] == [-15.0, 15.0]
