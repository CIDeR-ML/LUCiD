"""``lucid-siren-promote``: a training run becomes an installed emitter only with what the
simulator needs (a Cherenkov emitter's t0 fit and emission band) and only against the s_max fit it
was trained on."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('uproot')                   # build_tables, which reads smax_fit.csv
from lucid.siren.emitter import read_emitter, write_emitter                     # noqa: E402
from lucid.siren.net import SIREN                                               # noqa: E402
from lucid.siren.training.promote import main                                   # noqa: E402

SMAX_CSV = ("material,particle,form,A,B,a,b1,b2,E0,fit_min_mev,fit_max_mev,n_used,quantile,"
            "quantile_multiplier,generated_at_utc\n"
            "water,mu-,A*E^B,6.122104,0.978952,,,,,500,50000,21,0.9999,1.1,2026-05-13T11:14:34+00:00\n")
# calculate_t0.py's timing_parameters.json, as it writes it
T0 = {'form': 'stretched_exp_delay', 'particle': 'mu-', 'material': 'water', 'c_mm_per_ns': 299.792,
      'A': {'log10_poly_logE': [-1.14, 0.0, -0.07, 0.003]},
      'lambda': {'log10_poly_logE': [-0.51, 1.0, 0.16, -0.027]},
      'beta': {'poly_logE': [11.6, -11.5, 3.8, -0.39]},
      'provenance': {'source': 'PhotonSim/tools/t0_correction/calculate_t0.py', 'scan_dir': '/t0/in',
                     'n_energies': 3, 'energy_range_mev': [150, 100000]}}
BAND = ['--emission-band-nm', '274.91', '673.83']


def _run(tmp_path, kind='cherenkov', **smax):
    """A training run as lucid-train-siren writes it (no t0, no band)."""
    axis2 = 'angle_rad' if kind == 'cherenkov' else 'dedx_kev_per_mm'
    rec = {'kind': kind, 'net': {'hidden_features': 8, 'hidden_layers': 1, 'w0': 30.0},
           'domain': {'energy_mev': [150.0, 100000.0], axis2: [0.01, 3.1], 's_over_smax': [0.001, 0.999]},
           'target': {'log10_min': -3.0, 'log10_max': 2.5, 'offset': 1e-3}, 'grid_bins': 16,
           's_max_mm': {'form': 'A*E^B', 'params': {'A': 6.122104, 'B': 0.978952, **smax},
                        'fit_min_mev': 500, 'fit_max_mev': 50000, 'quantile': 0.9999,
                        'quantile_multiplier': 1.1, 'generated_at_utc': '2026-05-13T11:14:34+00:00'},
           'provenance': {'material': 'water', 'particle': 'mu-', 'table': 't.h5', 'steps': 10,
                          'losses': {'final_train': 1.0, 'final_val': 1.0, 'best_val': 1.0},
                          'init': None, 'commit': 'abc', 'trained_at_utc': 'then'},
           'training': {'recipe': {'seed': 1}}}
    if kind == 'cherenkov':
        rec['n_photons'] = {'form': 'logpoly', 'coeffs': [12.0, 1.0], 'u0': 6.0, 'du': 2.0}
    run = tmp_path / f'run_{kind}'
    params = SIREN(hidden_features=8, hidden_layers=1, out_features=1).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    write_emitter(run, rec, params)
    (tmp_path / 'smax_fit.csv').write_text(SMAX_CSV)
    (tmp_path / 't0.json').write_text(json.dumps(T0))
    return run


def _args(tmp_path, run, *extra):
    return ['--run', str(run), '--smax-fit', str(tmp_path / 'smax_fit.csv'),
            '--dest', str(tmp_path / 'installed'), *extra]


def test_a_cherenkov_run_is_installed_with_its_t0_fit_and_band(tmp_path):
    run = _run(tmp_path)
    assert main(_args(tmp_path, run, '--t0', str(tmp_path / 't0.json'), *BAND,
                      '--selection', 'seed scan: lowest final val_loss')) == 0
    e = read_emitter(tmp_path / 'installed', promoted=True)
    np.testing.assert_allclose(np.stack(e.t0), [T0['A']['log10_poly_logE'],
                                                T0['lambda']['log10_poly_logE'],
                                                T0['beta']['poly_logE']], rtol=1e-6)
    assert e.emission_band == (274.91, 673.83) and e.c_mm_per_ns == 299.792
    for a, b in zip(jax.tree.leaves(read_emitter(run).params), jax.tree.leaves(e.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    rec = json.loads((tmp_path / 'installed' / 'emitter.json').read_text())
    prov = rec['provenance']
    assert prov['run'] == str(run.resolve()) and prov['selection'].startswith('seed scan')
    assert prov['smax_fit'] == str((tmp_path / 'smax_fit.csv').resolve())
    assert prov['commit'] == 'abc' and prov['promoted_at_utc'] and 'promoted_commit' in prov
    assert rec['t0']['source'] == {'tool': 'PhotonSim/tools/t0_correction/calculate_t0.py',
                                   'scan_dir': '/t0/in', 'n_energies': 3,
                                   'energy_range_mev': [150, 100000]}
    # what is there is replaced only on request
    with pytest.raises(SystemExit):
        main(_args(tmp_path, run, '--t0', str(tmp_path / 't0.json'), *BAND))
    assert main(_args(tmp_path, run, '--t0', str(tmp_path / 't0.json'), *BAND, '--force')) == 0


@pytest.mark.parametrize('extra, what', [
    ([], '--t0 and --emission-band-nm'),
    (['--t0', 'T0'], 'pass --emission-band-nm'),
    (BAND, 'pass --t0'),
])
def test_a_cherenkov_run_without_t0_or_band_is_refused(tmp_path, capsys, extra, what):
    run = _run(tmp_path)
    extra = [str(tmp_path / 't0.json') if a == 'T0' else a for a in extra]
    assert main(_args(tmp_path, run, *extra)) == 2
    assert what in capsys.readouterr().err
    assert not (tmp_path / 'installed').exists()


def test_a_run_is_checked_against_its_inputs(tmp_path, capsys):
    """The s_max fit must be the run's; the t0 fit must be the run's material and particle; a
    dE/dx emitter takes neither a t0 fit nor a band."""
    t0 = ['--t0', str(tmp_path / 't0.json'), *BAND]
    assert main(_args(tmp_path, _run(tmp_path, B=0.9), *t0)) == 2
    assert 'params.B: run 0.9' in capsys.readouterr().err
    run = _run(tmp_path)
    (tmp_path / 't0.json').write_text(json.dumps({**T0, 'particle': 'e-'}))
    assert main(_args(tmp_path, run, *t0)) == 2
    assert 'the t0 fit is of water/e-' in capsys.readouterr().err
    (tmp_path / 't0.json').write_text(json.dumps({k: v for k, v in T0.items() if k != 'beta'}))
    assert main(_args(tmp_path, run, *t0)) == 2
    assert "the t0 fit lacks 'beta'" in capsys.readouterr().err
    dedx = _run(tmp_path, 'dedx')
    assert main(_args(tmp_path, dedx, '--t0', str(tmp_path / 't0.json'))) == 2
    assert 'takes no --t0' in capsys.readouterr().err
    assert main(_args(tmp_path, dedx)) == 0
    assert read_emitter(tmp_path / 'installed', promoted=True).kind == 'dedx'
