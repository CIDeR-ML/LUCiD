"""The train_siren job generator: dry runs against a temporary user_paths.sh. The site (image,
GPU user site, GPU partition, account) comes from user_paths.sh, a missing setting is an error
that names it, and the tracked configs run on any site that has their tables."""
import json
from pathlib import Path

import pytest

from lucid.production.siren_inputs.train_siren import generate_jobs as gj

CONFIGS = Path(gj.__file__).parent / 'configs'
SITE = ('CLUSTER', 'LUCID_IMAGE_PATH', 'LUCID_ENV_BASE', 'LUCID_DEV_PATH', 'PHOTONSIM_DEV_PATH',
        'SLURM_ACCOUNT', 'SIREN_TRAIN_PARTITION', 'SIREN_INPUTS_BASE_PATH', 'SIREN_OUTPUT_BASE_PATH',
        'OUTPUT_BASE_PATH', 'APPTAINER_BINDS', 'CONDOR_TRAIN_FLAVOUR', 'LOG_BASE_PATH')


@pytest.fixture
def site(tmp_path, monkeypatch):
    """``write(**settings)`` -> a user_paths.sh path; the shell's own settings do not leak in."""
    for k in SITE:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / 'lucid.sif').touch()
    for p in ('muon', 'electron'):
        for t in ('photon', 'dedx'):
            table = tmp_path / 'in' / 'training_inputs' / 'water' / 'lookup_tables' / p / f'{t}_lookup_table.h5'
            table.parent.mkdir(parents=True, exist_ok=True)
            table.touch()
    base = {'CLUSTER': 'slurm', 'LUCID_IMAGE_PATH': str(tmp_path / 'lucid.sif'),
            'LUCID_ENV_BASE': '/env/gpu', 'SLURM_ACCOUNT': 'fac:repo',
            'SIREN_TRAIN_PARTITION': 'ampere', 'SIREN_OUTPUT_BASE_PATH': str(tmp_path / 'out'),
            'SIREN_INPUTS_BASE_PATH': str(tmp_path / 'in'), 'APPTAINER_BINDS': '/sdf'}

    def write(**settings):
        env = {k: v for k, v in {**base, **settings}.items() if v is not None}
        up = tmp_path / 'user_paths.sh'
        up.write_text(''.join(f'export {k}="{v}"\n' for k, v in env.items()))
        return str(up)
    return write


@pytest.mark.parametrize('config', sorted(CONFIGS.glob('*.json')), ids=lambda p: p.stem)
def test_a_tracked_config_prepares_gpu_jobs_from_the_site(site, tmp_path, config):
    assert gj.main(['-c', str(config), '--user-paths', site()]) == 0
    cfg = json.loads(config.read_text())
    subs = sorted((tmp_path / 'out' / 'training_tests' / cfg['name']).glob('*/submit.sbatch'))
    assert len(subs) == len(cfg['runs'])
    body = subs[0].read_text()
    assert '#SBATCH --partition=ampere\n' in body and '#SBATCH --account=fac:repo\n' in body
    assert '#SBATCH --gpus=1\n' in body and 'APPTAINERENV_PYTHONUSERBASE="/env/gpu"' in body
    assert f"--h5-path {tmp_path / 'in' / 'training_inputs' / 'water' / 'lookup_tables'}" in body
    assert ':/opt/LUCiD' not in body                    # no LUCID_DEV_PATH: the image's LUCiD
    assert 'cjesus' not in body and 'mli:cider-ml' not in body


def test_the_seed_runs_and_the_dev_checkout(site, tmp_path):
    cfg = CONFIGS / 'water_mu_cherenkov_seeds.json'
    assert gj.main(['-c', str(cfg), '--user-paths', site(LUCID_DEV_PATH='/dev/LUCiD'),
                    '-P', 'turing']) == 0
    body = (tmp_path / 'out' / 'training_tests' / 'water_mu_cherenkov_seeds' / 'seed3'
            / 'submit.sbatch').read_text()
    assert '--seed 3' in body and '--checkpoint-every 10000' in body
    assert '-B /dev/LUCiD:/opt/LUCiD' in body and '#SBATCH --partition=turing\n' in body


@pytest.mark.parametrize('unset, named', [
    ('SIREN_TRAIN_PARTITION', 'SIREN_TRAIN_PARTITION is not set'),
    ('SLURM_ACCOUNT', 'SLURM_ACCOUNT is not set'),
    ('LUCID_ENV_BASE', 'LUCID_ENV_BASE is not set'),
    ('LUCID_IMAGE_PATH', 'LUCID_IMAGE_PATH is not set'),
    ('SIREN_INPUTS_BASE_PATH', 'no table at'),          # then it is SIREN_OUTPUT_BASE_PATH's
])
def test_a_missing_site_setting_is_an_error_that_names_it(site, tmp_path, capsys, unset, named):
    cfg = CONFIGS / 'water_mu_cherenkov_seeds.json'
    assert gj.main(['-c', str(cfg), '--user-paths', site(**{unset: None})]) == 2
    assert named in capsys.readouterr().err
    assert not (tmp_path / 'out' / 'training_tests').exists()


@pytest.mark.parametrize('change, named', [
    ({'slurm': {'partition': 'roma'}}, "'slurm' block is now 'resources'"),
    ({'resources': {'account': 'x:y'}}, "unknown resources ['account']"),
    ({'baseline': {'h5_path': '${NOWHERE}/t.h5'}}, '${NOWHERE}, which is not set'),
])
def test_a_config_cannot_carry_site_settings(site, tmp_path, capsys, change, named):
    cfg = {**json.loads((CONFIGS / 'example_water_mu.json').read_text()), **change}
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert gj.main(['-c', str(tmp_path / 'c.json'), '--user-paths', site()]) == 2
    assert named in capsys.readouterr().err


def test_htcondor_needs_no_partition(site, tmp_path):
    up = site(CLUSTER='htcondor', SIREN_TRAIN_PARTITION=None, SLURM_ACCOUNT=None,
              LOG_BASE_PATH=str(tmp_path / 'logs'))
    assert gj.main(['-c', str(CONFIGS / 'water_el_dedx_seeds.json'), '--user-paths', up]) == 0
    sub = (tmp_path / 'out' / 'training_tests' / 'water_el_dedx_seeds' / 'seed0'
           / 'submit.sub').read_text()
    assert 'PYTHONUSERBASE=/env/gpu' in sub and 'request_gpus            = 1' in sub
