"""The PhotonSim scan generators (stage 0, s_max; stage 1, training inputs): dry runs of the
tracked smoke-test configs against a temporary user_paths.sh. Each cell is one lucid-run-job
--skip-lucid in the container, firing the particle along +z; stage 1 sets /output/smax from the
s_max fit; a missing setting is an error that names it."""
import json
from pathlib import Path

import pytest

from lucid.production.siren_inputs import generate_jobs as stage1
from lucid.production.siren_inputs.smax import generate_jobs as stage0

HERE = Path(stage1.__file__).parent
SITE = ('CLUSTER', 'LUCID_IMAGE_PATH', 'PHOTONSIM_DEV_PATH', 'LUCID_DEV_PATH', 'SLURM_ACCOUNT',
        'SLURM_PARTITION', 'SIREN_OUTPUT_BASE_PATH', 'OUTPUT_BASE_PATH', 'APPTAINER_BINDS',
        'DEFAULT_GPUS', 'GENIE_XSEC_FILE')


@pytest.fixture
def site(tmp_path, monkeypatch):
    for k in SITE:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / 'lucid.sif').touch()
    fit = tmp_path / 'PhotonSim' / 'data' / 'water' / 'mu-' / 'smax_fit.csv'
    fit.parent.mkdir(parents=True)
    fit.write_text('material,particle,form,A,B,fit_min_mev,fit_max_mev,quantile,'
                   'quantile_multiplier,generated_at_utc\n'
                   'water,mu-,A*E^B,6.122104,0.978952,500,50000,0.9999,1.1,synthetic\n')
    base = {'CLUSTER': 'slurm', 'LUCID_IMAGE_PATH': str(tmp_path / 'lucid.sif'),
            'PHOTONSIM_DEV_PATH': str(tmp_path / 'PhotonSim'), 'SLURM_ACCOUNT': 'fac:repo',
            'SLURM_PARTITION': 'milano', 'SIREN_OUTPUT_BASE_PATH': str(tmp_path / 'out'),
            'APPTAINER_BINDS': '/sdf', 'DEFAULT_GPUS': '0'}

    def write(**settings):
        env = {k: v for k, v in {**base, **settings}.items() if v is not None}
        up = tmp_path / 'user_paths.sh'
        up.write_text(''.join(f'export {k}="{v}"\n' for k, v in env.items()))
        return str(up)
    return write


def _check_job(sub: Path, energy: int):
    body = sub.read_text()
    assert '#SBATCH --partition=milano\n' in body and '#SBATCH --account=fac:repo\n' in body
    assert '--skip-lucid' in body and f'--override-energy-MeV {energy}\n' in body
    assert f'-B {sub.parents[5] / "PhotonSim"}:/opt/PhotonSim' in body


def test_stage1_sets_smax_from_the_fit_per_cell(site, tmp_path):
    cfg = HERE / 'configs' / 'water_mu_test.json'
    assert stage1.main(['-c', str(cfg), '--user-paths', site()]) == 0
    energies = json.loads(cfg.read_text())['energy_list_MeV']
    for E in energies:
        cell = tmp_path / 'out' / 'training_inputs' / 'water' / 'mu-' / f'{E}MeV'
        c = json.loads((cell / 'photonsim_config.json').read_text())
        assert c['smax_mm'] == pytest.approx(6.122104 * E ** 0.978952)
        assert c['fixed_direction_z'] and c['disable_decays'] and not c['store_individual_photons']
        _check_job(cell / 'submit.sbatch', E)


def test_stage0_computes_smax_so_sets_none(site, tmp_path):
    cfg = HERE / 'smax' / 'configs' / 'water_mu_test.json'
    assert stage0.main(['-c', str(cfg), '--user-paths', site()]) == 0
    E = json.loads(cfg.read_text())['energy_list_MeV'][0]
    cell = tmp_path / 'out' / 'smax_parametrization' / 'water' / 'mu-' / f'{E}MeV'
    assert 'smax_mm' not in json.loads((cell / 'photonsim_config.json').read_text())
    _check_job(cell / 'submit.sbatch', E)


@pytest.mark.parametrize('stage, config', [(stage0, 'smax/configs/water_mu_test.json'),
                                           (stage1, 'configs/water_mu_test.json')])
@pytest.mark.parametrize('unset, named', [
    ({'SLURM_ACCOUNT': None}, 'SLURM_ACCOUNT'),
    ({'LUCID_IMAGE_PATH': None}, 'LUCID_IMAGE_PATH'),
    ({'SIREN_OUTPUT_BASE_PATH': None}, 'SIREN_OUTPUT_BASE_PATH (or OUTPUT_BASE_PATH, or -o)'),
])
def test_a_missing_setting_is_named(site, capsys, stage, config, unset, named):
    assert stage.main(['-c', str(HERE / config), '--user-paths', site(**unset)]) == 2
    assert named in capsys.readouterr().err
