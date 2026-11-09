"""``lucid-train-siren``: a tiny run on a synthetic table writes an emitter the loader reads, whose
density is the trainer's own prediction, and whose rays run."""
import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('uproot')
from lucid.siren.emitter import _decode, density, read_emitter, read_weights     # noqa: E402
from lucid.siren.training.photonsim_data.build_tables import _fit_n_photons      # noqa: E402
from lucid.siren.training.train import main                                      # noqa: E402
from lucid.siren.training.trainer import NOMINAL, Recipe                         # noqa: E402

ENERGIES = np.geomspace(200, 3200, 10).astype(int)
T0 = {'form': 'stretched_exp_delay', 'c_mm_per_ns': 299.792,
      'A': {'log10_poly_logE': [-1.14, 0.0, -0.07, 0.003]},
      'lambda': {'log10_poly_logE': [-0.51, 1.0, 0.16, -0.027]},
      'beta': {'poly_logE': [11.6, -11.5, 3.8, -0.39]},
      'provenance': {'source': 'synthetic'}}
TINY = ['--hidden-features', '16', '--hidden-layers', '1', '--batch-size', '256',
        '--val-every', '20', '--zero-keep-frac', '1.0', '--val-split', '0.1',
        '--no-prediction-plots']


@pytest.fixture
def table(tmp_path):
    """A 10 x 20 x 20 photon table: a Gaussian ring at 0.72 rad, fading along s/s_max."""
    angle, dist = np.linspace(0.0, np.pi, 21), np.linspace(0.0, 1.0, 21)
    ac, dc = 0.5 * (angle[1:] + angle[:-1]), 0.5 * (dist[1:] + dist[:-1])
    ring = np.exp(-0.5 * ((ac[:, None] - 0.72) / 0.15) ** 2) * (1.0 - dc[None, :])
    avg = ENERGIES[:, None, None] * 0.5 * ring[None]
    path = tmp_path / 'photon_lookup_table.h5'
    with h5py.File(path, 'w') as f:
        f['data/photon_table_average'] = avg
        f['coordinates/energy_centers'] = ENERGIES
        f['coordinates/angle_centers'] = ac
        f['coordinates/distance_centers'] = dc
        m = f.create_group('metadata').attrs
        m.update({'material': 'water', 'particle': 'mu-', 'distance_axis': 's_over_smax',
                  'smax_form': 'A*E^B', 'smax_A': 6.1, 'smax_B': 0.98, 'smax_fit_min_mev': 500,
                  'smax_fit_max_mev': 50000, 'smax_quantile': 0.9999,
                  'smax_quantile_multiplier': 1.1, 'smax_generated_at_utc': 'synthetic'})
        for k, v in _fit_n_photons(ENERGIES, avg.sum(axis=(1, 2))).items():
            m[f'nphot_{k}'] = v
    return path


def test_a_tiny_run_writes_the_emitter_the_loader_reads(table, tmp_path):
    out = tmp_path / 'run'
    (tmp_path / 't0.json').write_text(json.dumps(T0))
    main(['--h5-path', str(table), '--output-dir', str(out), '--num-steps', '300',
          '--t0', str(tmp_path / 't0.json'), *TINY])
    hist = json.loads((out / 'training_history.json').read_text())
    assert np.mean(hist['train_loss'][-5:]) < 0.5 * hist['train_loss'][0]       # it learns
    assert (out / 'final_training_progress.pdf').is_file()

    e = read_emitter(out)
    recipe = json.loads((out / 'emitter.json').read_text())['training']['recipe']
    assert recipe == {**json.loads(NOMINAL.read_text()), 'num_steps': 300, 'hidden_features': 16,
                      'hidden_layers': 1, 'batch_size': 256, 'val_every': 20,
                      'zero_keep_frac': 1.0, 'val_split': 0.1}
    assert e.kind == 'cherenkov' and e.t0 is not None

    # The density is the trainer's own prediction, bit for bit, on the table grid: the dataset
    # normalises with the emitter's normalisation (zero_keep_frac=1 keeps every bin, in order).
    from lucid.siren.training.dataset import PhotonSimDataset
    ds = PhotonSimDataset(table, zero_threshold=1e-3, zero_keep_frac=1.0)
    with h5py.File(table) as f:
        ac, dc = f['coordinates/angle_centers'][:], f['coordinates/distance_centers'][:]
    X, S = np.meshgrid(ac, dc, indexing='ij')
    n = X.size
    for i, E in enumerate(ENERGIES):
        x = ds.data['inputs_normalized'][i * n:(i + 1) * n]
        own = _decode(jnp.squeeze(e.net.apply(e.params, jnp.asarray(x))),
                      e.log10_range[1], e.log10_range[0], e.target_offset)
        got = density(e, e.params, float(E), jnp.asarray(X.ravel(), jnp.float32),
                      jnp.asarray(S.ravel(), jnp.float32))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(own))

    from lucid.sources.siren_rays import make_cherenkov_surrogate_fn
    _, origins, w = make_cherenkov_surrogate_fn(e)(
        jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), 800.0, 1000, e.params, jax.random.PRNGKey(0))
    assert np.all(np.isfinite(np.asarray(origins)))
    np.testing.assert_allclose(float(w.sum()), float(e.n_photons(800.0)), rtol=1e-5)


def test_init_starts_from_the_given_weights_and_an_emitter_is_never_overwritten(table, tmp_path):
    first = tmp_path / 'first'
    main(['--h5-path', str(table), '--output-dir', str(first), '--num-steps', '2',
          '--checkpoint-every', '1', *TINY])
    with pytest.raises(SystemExit):
        main(['--h5-path', str(table), '--output-dir', str(first), '--num-steps', '2', *TINY])
    # zero learning rate: a warm start ends where it began
    second = tmp_path / 'second'
    main(['--h5-path', str(table), '--output-dir', str(second), '--num-steps', '2',
          '--learning-rate', '0', '--init', str(first / 'checkpoint_step_1.npz'), *TINY])
    for a, b in zip(jax.tree.leaves(read_weights(first / 'checkpoint_step_1.npz')),
                    jax.tree.leaves(read_emitter(second).params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_an_unknown_recipe_field_is_refused(tmp_path):
    bad = tmp_path / 'r.json'
    bad.write_text(json.dumps({**json.loads(NOMINAL.read_text()), 'momentum': 0.9}))
    with pytest.raises(ValueError, match='momentum'):
        Recipe.load(bad)


@pytest.mark.parametrize('emitter', sorted(NOMINAL.parents[3].glob('data/*/*/*/emitter.json')),
                         ids=lambda p: '/'.join(p.parts[-4:-1]))
def test_every_shipped_emitter_records_the_nominal_recipe_with_its_seed(emitter):
    recipe = json.loads(emitter.read_text())['training']['recipe']
    assert recipe == {**json.loads(NOMINAL.read_text()), 'seed': recipe['seed']}
    Recipe.load(NOMINAL, seed=recipe['seed'])
