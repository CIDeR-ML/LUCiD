"""The t0 stage: the job generator (a dry run against a temporary user_paths.sh), and PhotonSim's
calculate_t0.py on a tiny synthetic scan, promoted into an emitter whose emission time is the
delay the scan was made with.

The fit needs a PhotonSim checkout (``PHOTONSIM_DEV_PATH``, as the stage itself); without one
that test skips.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lucid.production.siren_inputs.t0 import generate_jobs as gj

SITE = ('CLUSTER', 'LUCID_IMAGE_PATH', 'PHOTONSIM_DEV_PATH', 'SLURM_ACCOUNT', 'SLURM_PARTITION',
        'SIREN_INPUTS_BASE_PATH', 'SIREN_OUTPUT_BASE_PATH', 'OUTPUT_BASE_PATH', 'APPTAINER_BINDS',
        'LUCID_DEV_PATH', 'CONDOR_JOB_FLAVOUR')
SMAX_CSV = ("material,particle,form,A,B,fit_min_mev,fit_max_mev,quantile,quantile_multiplier,"
            "generated_at_utc\nwater,mu-,A*E^B,6.122104,0.978952,500,50000,0.9999,1.1,synthetic\n")


def test_the_stage_and_the_simulator_read_one_t0_form():
    from lucid.siren.emitter import T0_FORM
    assert gj.T0_FORM == T0_FORM


@pytest.fixture
def site(tmp_path, monkeypatch):
    for k in SITE:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / 'lucid.sif').touch()
    ps = tmp_path / 'PhotonSim'
    (ps / 'tools' / 't0_correction').mkdir(parents=True)
    (ps / 'tools' / 't0_correction' / 'calculate_t0.py').write_text('"form": "stretched_exp_delay"\n')
    (ps / 'data' / 'water' / 'mu-').mkdir(parents=True)
    (ps / 'data' / 'water' / 'mu-' / 'smax_fit.csv').write_text(SMAX_CSV)
    uproot = pytest.importorskip('uproot')
    for e in (150, 1000, 20000):
        cell = tmp_path / 'in' / 'training_inputs' / 'water' / 'mu-' / f'{e}MeV'
        cell.mkdir(parents=True)
        with uproot.recreate(cell / 'photonsim.root') as f:
            f['PhotonHist_TimeDistanceNorm'] = (np.ones((2, 2)), np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 2.0]))
    base = {'CLUSTER': 'slurm', 'LUCID_IMAGE_PATH': str(tmp_path / 'lucid.sif'),
            'PHOTONSIM_DEV_PATH': str(ps), 'SLURM_ACCOUNT': 'fac:repo',
            'SLURM_PARTITION': 'roma:130,milano:272', 'SIREN_INPUTS_BASE_PATH': str(tmp_path / 'in'),
            'SIREN_OUTPUT_BASE_PATH': str(tmp_path / 'out'), 'APPTAINER_BINDS': '/sdf'}

    def write(**settings):
        env = {k: v for k, v in {**base, **settings}.items() if v is not None}
        up = tmp_path / 'user_paths.sh'
        up.write_text(''.join(f'export {k}="{v}"\n' for k, v in env.items()))
        return ['--material', 'water', '--particle', 'mu-', '--user-paths', str(up)]
    return write


def test_the_job_fits_the_training_inputs_with_photonsim(site, tmp_path):
    assert gj.main(site()) == 0
    out = tmp_path / 'out' / 't0_parametrization' / 'water' / 'mu-'
    links = sorted((out / 'inputs').iterdir())
    assert [p.name for p in links] == ['mu-_1000MeV.root', 'mu-_150MeV.root', 'mu-_20000MeV.root']
    assert all(p.resolve() == tmp_path / 'in' / 'training_inputs' / 'water' / 'mu-'
               / p.name[4:-5] / 'photonsim.root' for p in links)
    body = (out / 'submit.sbatch').read_text()
    calc = tmp_path / 'PhotonSim' / 'tools' / 't0_correction' / 'calculate_t0.py'
    assert (f"/opt/conda/bin/python3 {calc} {out / 'inputs'} --out-dir {out} --material water "
            f"--particle mu- --smax-data-dir {tmp_path / 'PhotonSim' / 'data'}") in body
    assert '#SBATCH --partition=roma\n' in body and '#SBATCH --account=fac:repo\n' in body
    # rerun: the links are replaced, not duplicated; a finished fit is not redone
    assert gj.main(site()) == 0 and len(list((out / 'inputs').iterdir())) == 3
    (out / 'timing_parameters.json').write_text('{}')
    (out / 'submit.sbatch').unlink()
    assert gj.main(site()) == 0 and not (out / 'submit.sbatch').exists()


@pytest.mark.parametrize('change, named', [
    ({'PHOTONSIM_DEV_PATH': None}, 'PHOTONSIM_DEV_PATH is not set'),
    ({'SLURM_PARTITION': None}, 'SLURM_PARTITION (or CONDOR_JOB_FLAVOUR) is not set'),
    ({'SIREN_INPUTS_BASE_PATH': None}, 'no <E>MeV/photonsim.root under'),
    ({'SIREN_INPUTS_BASE_PATH': None, 'SIREN_OUTPUT_BASE_PATH': None}, 'no input root'),
    ('old', 'predates the'),
    ('no smax', 'no s_max fit'),
    ('no hist', 'has no PhotonHist_TimeDistanceNorm'),
])
def test_a_missing_input_is_an_error_that_names_it(site, tmp_path, capsys, change, named):
    calc = tmp_path / 'PhotonSim' / 'tools' / 't0_correction' / 'calculate_t0.py'
    if change == 'old':
        calc.write_text('json_file = scan_dir / "timing_parameters.json"\n')
    if change == 'no smax':
        (tmp_path / 'PhotonSim' / 'data' / 'water' / 'mu-' / 'smax_fit.csv').unlink()
    if change == 'no hist':
        import uproot
        with uproot.recreate(tmp_path / 'in' / 'training_inputs' / 'water' / 'mu-' / '20000MeV'
                             / 'photonsim.root') as f:
            f['PhotonHist_Distance'] = (np.ones(2), np.array([0.0, 1.0, 2.0]))
    assert gj.main(site(**(change if isinstance(change, dict) else {}))) == 2
    assert named in capsys.readouterr().err


# ---- the fit itself: PhotonSim's calculate_t0.py on a synthetic scan --------------------------

PHOTONSIM = os.environ.get('PHOTONSIM_DEV_PATH')
CALC = Path(PHOTONSIM or '/nonexistent') / gj.CALCULATE_T0
TRUE = {'A': [-1.14, 0.0, -0.07, 0.003], 'lambda': [-0.51, 1.0, 0.16, -0.027],
        'beta': [11.6, -11.5, 3.8, -0.39]}      # the shipped muon fit, rounded


def _cubic(c, E):
    x = np.log10(E)
    return c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3


def _delay(d_mm, E):
    A, lam, beta = 10 ** _cubic(TRUE['A'], E), 10 ** _cubic(TRUE['lambda'], E), _cubic(TRUE['beta'], E)
    return A * (np.exp((d_mm / lam) ** beta) - 1.0)


def _scan(directory: Path, energies, smax_mm):
    """One ROOT file per energy whose PhotonHist_TimeDistanceNorm has, in each s/s_max bin, a
    mean delay of exactly _delay(d): the counts split between the two delay bins around it."""
    uproot = pytest.importorskip('uproot')
    u_edges, t_edges = np.linspace(0.0, 1.0, 101), np.linspace(-0.5, 2.5, 501)
    u, t = 0.5 * (u_edges[1:] + u_edges[:-1]), 0.5 * (t_edges[1:] + t_edges[:-1])
    for E in energies:
        target = _delay(u * smax_mm(E), E)
        counts = np.zeros((u.size, t.size))
        i = np.clip(np.searchsorted(t, target) - 1, 0, t.size - 2)
        w = (target - t[i]) / (t[i + 1] - t[i])
        counts[np.arange(u.size), i] = 1000.0 * (1 - w)
        counts[np.arange(u.size), i + 1] = 1000.0 * w
        with uproot.recreate(directory / f'mu-_{E}MeV.root') as f:
            f['PhotonHist_TimeDistanceNorm'] = (counts, u_edges, t_edges)


@pytest.mark.skipif(not CALC.is_file(), reason='needs a PhotonSim checkout (PHOTONSIM_DEV_PATH)')
def test_calculate_t0_recovers_the_delay_and_promotes_into_the_emitter(tmp_path):
    pytest.importorskip('scipy')
    from lucid.siren.training.promote import t0_block
    from lucid.sources.siren_rays import predict_t0
    (tmp_path / 'data' / 'water' / 'mu-').mkdir(parents=True)
    (tmp_path / 'data' / 'water' / 'mu-' / 'smax_fit.csv').write_text(SMAX_CSV)
    scan = tmp_path / 'inputs'
    scan.mkdir()
    energies = [200, 400, 800, 1600, 3200, 6400, 12800]
    _scan(scan, energies, lambda E: 6.122104 * E ** 0.978952)
    r = subprocess.run([sys.executable, str(CALC), str(scan), '--out-dir', str(tmp_path / 'fit'),
                        '--material', 'water', '--particle', 'mu-',
                        '--smax-data-dir', str(tmp_path / 'data')],
                       capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': 'Agg'})
    assert r.returncode == 0, r.stdout + r.stderr
    fit = json.loads((tmp_path / 'fit' / 'timing_parameters.json').read_text())
    assert fit['provenance']['n_energies'] == len(energies)
    block = t0_block(fit)
    t0 = tuple(np.asarray(block[k]) for k in ('log10_A', 'log10_lambda_mm', 'beta'))
    for E in (300.0, 1000.0, 5000.0):
        d = np.linspace(100.0, 0.9 * 6.122104 * E ** 0.978952, 7)
        got = np.asarray(predict_t0(d / 1000.0, E, t0, block['c_mm_per_ns']))
        np.testing.assert_allclose(got - d / block['c_mm_per_ns'], _delay(d, E),
                                   rtol=0.02, atol=2e-3)       # ns
