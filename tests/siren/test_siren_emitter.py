"""``lucid.siren.emitter``: one trained emitter per directory, written and read by one schema.

The round trip and the curve checks need no model; the shipped-model cases read the downloaded
water weights (marker ``requires_emitter``).
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.siren.net import SIREN
from lucid.siren.emitter import (_decode, density, emitter_dir, load_emitter, make_n_photons_fn,
                                 read_emitter, read_weights, write_emitter)
from lucid.utils import base_dir_path

ROOT = Path(base_dir_path())
# Photon yield at 1 GeV from the shipped polynomials.
YIELD_1GEV = {'muon': 201927.7, 'electron': 214863.8}


def _record(kind='cherenkov'):
    axis2 = {'cherenkov': ('angle_rad', [0.01, 3.1]), 'dedx': ('dedx_kev_per_mm', [1.0, 999.0])}
    r = {'kind': kind, 'net': {'hidden_features': 8, 'hidden_layers': 1, 'w0': 30.0},
         'domain': {'energy_mev': [100.0, 5000.0], axis2[kind][0]: axis2[kind][1],
                    's_over_smax': [0.001, 0.999]},
         'target': {'log10_min': -3.0, 'log10_max': 2.5, 'offset': 1e-3},
         'grid_bins': 16,
         's_max_mm': {'form': 'A*E^B', 'params': {'A': 6.1, 'B': 0.98}}}
    if kind == 'cherenkov':
        r['n_photons'] = {'form': 'logpoly', 'coeffs': [12.0, 1.0], 'u0': 6.0, 'du': 2.0}
        r['t0'] = {'form': 'stretched_exp_delay', 'c_mm_per_ns': 299.792,
                   'log10_A': [-1.1, 0.0, -0.07, 0.003], 'log10_lambda_mm': [-0.5, 1.0, 0.16, -0.03],
                   'beta': [11.6, -11.5, 3.8, -0.39]}
    return r


@pytest.mark.parametrize('kind', ['cherenkov', 'dedx'])
def test_write_then_read_is_the_same_emitter(tmp_path, kind):
    """What the trainer writes is what the loader reads: the params bit for bit, the domain, and
    the density, which is the decoded net output on the normalised inputs."""
    rec = _record(kind)
    net = SIREN(hidden_features=8, hidden_layers=1, out_features=1)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    write_emitter(tmp_path, rec, params)
    e = read_emitter(tmp_path)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(e.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert e.kind == kind and e.grid_bins == 16 and (e.n_photons is None) == (kind == 'dedx')
    ax2 = jnp.linspace(e.lo[1], e.hi[1], 7)
    r = jnp.linspace(e.lo[2], e.hi[2], 7)
    x = jnp.stack([jnp.full(7, -1.0 + 2 * (1000.0 - 100.0) / 4900.0),
                   jnp.linspace(-1.0, 1.0, 7), jnp.linspace(-1.0, 1.0, 7)], 1)
    raw = net.apply(params, x)[:, 0]
    expect = np.maximum(10.0 ** (np.asarray(raw, np.float64) * 5.5 - 3.0) - 1e-3, 0.0)
    # atol: float32 resolution of the decoded value at the offset (1e-3) it subtracts
    np.testing.assert_allclose(np.asarray(density(e, e.params, 1000.0, ax2, r)), expect,
                               rtol=1e-4, atol=1e-8)
    assert e.s_max_m(1000.0) == 6.1 * 1000.0 ** 0.98 / 1000.0
    if kind == 'cherenkov':     # in predict_t0's order: log10 A, log10 lambda, beta
        t0 = rec['t0']
        np.testing.assert_allclose(np.stack(e.t0), [t0['log10_A'], t0['log10_lambda_mm'],
                                                    t0['beta']], rtol=1e-6)


def test_decoding_inverts_the_training_encoding():
    """The net learns ``(log10(t + offset) - log10_min) / (log10_max - log10_min)``; decoding
    returns ``t``, an empty bin (t = 0, the encoding's minimum) included: no offset floor."""
    offset, t_max = 1e-3, 300.0
    lo, hi = np.log10(offset), np.log10(t_max + offset)
    t = np.array([0.0, 1e-3, 0.05, 1.0, 30.0, t_max])
    raw = (np.log10(t + offset) - lo) / (hi - lo)
    got = np.asarray(_decode(jnp.asarray(raw, jnp.float32), hi, lo, offset), np.float64)
    assert got[0] < 1e-9 and np.all(got >= 0.0)
    np.testing.assert_allclose(got[1:], t[1:], rtol=2e-5, atol=1e-8)


def test_an_emitter_of_another_schema_or_t0_form_is_refused(tmp_path):
    net = SIREN(hidden_features=8, hidden_layers=1, out_features=1)
    write_emitter(tmp_path, _record(), net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3))))
    meta = json.loads((tmp_path / 'emitter.json').read_text())
    (tmp_path / 'emitter.json').write_text(json.dumps({**meta, 't0': {**meta['t0'], 'form': 'linear'}}))
    with pytest.raises(ValueError, match='unsupported t0 form'):
        read_emitter(tmp_path)
    (tmp_path / 'emitter.json').write_text(json.dumps({**meta, 'schema': 0}))
    with pytest.raises(ValueError, match='schema'):
        read_emitter(tmp_path)


@pytest.mark.parametrize('particle, kind', [('pion', 'cherenkov'), ('mu-', 'cherenkov'),
                                            ('muon', 'scintillation')])
def test_a_particle_or_kind_with_no_emitter_raises(particle, kind):
    with pytest.raises(ValueError, match='no SIREN emitter'):
        emitter_dir(particle, 'water', kind)


@pytest.mark.parametrize('material', sorted(p.stem for p in (ROOT / 'config' / 'materials').glob('*.json')))
def test_every_material_names_an_emitter_set_with_both_particles(material):
    """A material JSON's "siren" key names the emitters it runs (wbls and ice: water's)."""
    siren = json.loads((ROOT / 'config' / 'materials' / f'{material}.json').read_text())['siren']
    for particle in ('muon', 'electron'):
        for kind in ('cherenkov', 'dedx'):
            assert (emitter_dir(particle, siren, kind) / 'emitter.json').is_file()


# ---- the shipped curves (tracked; no model needed) --------------------------------------------

@pytest.mark.parametrize('particle', ['muon', 'electron'])
def test_shipped_photon_yield_is_positive_increasing_and_spans_its_fit(particle):
    block = json.loads((emitter_dir(particle, 'water', 'cherenkov') / 'emitter.json')
                       .read_text())['n_photons']
    fn = make_n_photons_fn(block)
    np.testing.assert_allclose(float(fn(1000.0)), YIELD_1GEV[particle], rtol=1e-4)
    # the centred basis spans exactly the fitted range
    assert np.isclose(np.exp(block['u0'] - block['du']), block['fit_min_mev'], rtol=1e-6)
    assert np.isclose(np.exp(block['u0'] + block['du']), block['fit_max_mev'], rtol=1e-6)
    E = np.geomspace(block['fit_min_mev'], block['fit_max_mev'], 400)
    N = np.asarray(fn(E))
    assert np.all(N > 0) and np.all(np.diff(N) > 0)


# ---- the shipped water models ----------------------------------------------------------------

@pytest.mark.parametrize('particle, kind', [
    pytest.param(p, k, marks=pytest.mark.requires_emitter(p, 'water', dedx=k == 'dedx'))
    for p in ('muon', 'electron') for k in ('cherenkov', 'dedx')])
def test_the_shipped_emitter_holds_every_weight_of_its_file(particle, kind):
    e = load_emitter(particle, 'water', kind)
    path = emitter_dir(particle, 'water', kind) / 'weights.npz'
    with np.load(path) as f:
        stored = np.sort(np.concatenate([np.ravel(f[k]) for k in f.files]))
    loaded = np.sort(np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree.leaves(e.params)]))
    np.testing.assert_array_equal(loaded, stored)
    assert jax.tree.structure(read_weights(path)) == jax.tree.structure(e.params)
