"""The photon-yield curve nphot(E) and how it reaches the emitter.

nphot(E) is the absolute scale of the SIREN emission model, and the reconstruction differentiates
through it. The legacy power law in the downloaded metadata misfits its own training table, so a
log-log polynomial is shipped in the repository as data/water/<particle>/nphot.json and overlaid
onto the downloaded metadata at load. These tests pin each step of that: which form is selected, what the shipped coefficients
evaluate to, that the file is FOUND on every supported directory layout, and that a file fitted to
a different model is refused rather than applied.
"""
import json
import os
import types
import warnings

import numpy as np
import pytest

import lucid.siren.core as core
from lucid.siren.core import make_power_law_fn
from lucid.siren.training.inference import SIRENPredictor, repo_nphot_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = {p: os.path.join(ROOT, 'data', 'water', p, 'nphot.json') for p in ('muon', 'electron')}
# Photon yield at 1 GeV from the shipped polynomials, used to pin the coefficients below.
YIELD_1GEV = {'muon': 201927.7, 'electron': 214863.8}


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    core._WARNED_LEGACY_NPHOT.clear()
    yield
    core._WARNED_LEGACY_NPHOT.clear()


# ---- form selection and evaluation ---------------------------------------------------------------

def test_coefficients_select_the_polynomial_whatever_form_says():
    """Selection is by the PRESENCE of coeffs; the metadata's own form field keeps describing the
    legacy a/b/c it sits beside, so it must not decide."""
    block = {'form': 'A*E^B+C', 'a': 1.0, 'b': 1.0, 'c': 0.0,
             'coeffs': [1.0, 0.5, -0.1], 'u0': 7.0, 'du': 2.0}
    E = np.array([200.0, 1000.0, 5000.0])
    t = (np.log(E) - 7.0) / 2.0
    expect = np.exp(1.0 + 0.5 * t - 0.1 * t ** 2)
    np.testing.assert_allclose(np.asarray(make_power_law_fn(block)(E)), expect, rtol=1e-5)


def test_no_coefficients_falls_back_loudly_and_names_each_model_once():
    a = {'form': 'A*E^B+C', 'a': 2.0, 'b': 1.0, 'c': 0.0, '_origin': 'model-A'}
    b = dict(a, _origin='model-B')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = make_power_law_fn(a)
        make_power_law_fn(a)                      # same model again: no second warning
        make_power_law_fn(b)                      # a DIFFERENT model must still be named
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(msgs) == 2, msgs
    assert 'model-A' in msgs[0] and 'model-B' in msgs[1]
    np.testing.assert_allclose(float(f(10.0)), 20.0, rtol=1e-6)


def test_unknown_form_is_refused():
    with pytest.raises(ValueError, match='unknown nphot form'):
        make_power_law_fn({'form': 'spline', 'a': 1, 'b': 1, 'c': 0})


# ---- the shipped coefficients --------------------------------------------------------------------

@pytest.mark.parametrize('particle', ['muon', 'electron'])
def test_shipped_polynomial_reproduces_its_recorded_yield(particle):
    with open(SHIPPED[particle]) as fh:
        block = json.load(fh)
    fn = make_power_law_fn(block)
    np.testing.assert_allclose(float(fn(1000.0)), YIELD_1GEV[particle], rtol=1e-4)
    # the centred basis spans exactly the fitted range
    assert np.isclose(np.exp(block['u0'] - block['du']), block['fit_min_mev'], rtol=1e-6)
    assert np.isclose(np.exp(block['u0'] + block['du']), block['fit_max_mev'], rtol=1e-6)
    # strictly positive and increasing across the fitted range -- the legacy form crossed zero
    E = np.geomspace(block['fit_min_mev'], block['fit_max_mev'], 400)
    N = np.asarray(fn(E))
    assert np.all(N > 0) and np.all(np.diff(N) > 0)


# ---- finding the file ----------------------------------------------------------------------------

def _model_tree(root, material='water', particle='muon', link_to=None, ship=True):
    """data/<material>/<particle>/{nphot.json, siren_training -> ...}/trained_model/<stem>."""
    pdir = root / 'data' / material / particle
    pdir.mkdir(parents=True)
    if ship:
        (pdir / 'nphot.json').write_text('{}')
    if link_to is None:
        (pdir / 'siren_training' / 'trained_model').mkdir(parents=True)
    else:
        os.symlink(link_to, pdir / 'siren_training')
    return pdir / 'siren_training' / 'trained_model' / 'photonsim_siren'


def test_found_when_siren_training_is_a_real_directory(tmp_path):
    stem = _model_tree(tmp_path)
    assert repo_nphot_path(stem) == tmp_path / 'data' / 'water' / 'muon' / 'nphot.json'


def _store(root, particle='muon'):
    """An external store as `download_data.sh --store-dir` builds it, with a DECOY nphot.json
    beside the model, so a lookup that follows the absolute link would pick the wrong file."""
    st = root / 'store' / 'water' / particle / 'siren_training'
    (st / 'trained_model').mkdir(parents=True)
    (st.parent / 'nphot.json').write_text('{"decoy": true}')
    return st


def test_found_when_siren_training_links_outside_the_repo(tmp_path):
    """--store-dir install: water's siren_training is an ABSOLUTE link into the store. The file
    that must be found is the repository's, not whatever sits beside the model in the store."""
    stem = _model_tree(tmp_path / 'repo', link_to=_store(tmp_path))
    assert repo_nphot_path(stem) == tmp_path / 'repo' / 'data' / 'water' / 'muon' / 'nphot.json'


def test_found_through_the_relative_link_other_materials_use(tmp_path):
    """wbls and ice link siren_training to ../../water/<particle>/siren_training on every install;
    the coefficients belong to that model, so they reach water's file."""
    _model_tree(tmp_path)                                               # water, real dir, with file
    stem = _model_tree(tmp_path, material='wbls', link_to='../../water/muon/siren_training',
                       ship=False)
    assert repo_nphot_path(stem) == tmp_path / 'data' / 'water' / 'muon' / 'nphot.json'


def test_same_answer_for_other_materials_on_a_store_install(tmp_path):
    """The store install: water links OUT (absolute), wbls links to water (relative). wbls must
    still reach the repository's water file -- the same answer as the default install -- and not
    the store's decoy, which is what following every link would find."""
    repo = tmp_path / 'repo'
    _model_tree(repo, link_to=_store(tmp_path))
    stem = _model_tree(repo, material='wbls', link_to='../../water/muon/siren_training', ship=False)
    assert repo_nphot_path(stem) == repo / 'data' / 'water' / 'muon' / 'nphot.json'


def test_the_dedx_model_beside_it_is_not_given_the_cherenkov_curve(tmp_path):
    """nphot.json belongs to the Cherenkov model (siren_training). The dE/dx model in the same
    particle directory has its own, different nphot block; keying on the directory alone would hand it
    the Cherenkov coefficients, which the mismatch check refuses, failing every dE/dx load."""
    _model_tree(tmp_path)
    (tmp_path / 'data' / 'water' / 'muon' / 'nphot.json').write_text(json.dumps(
        {'coeffs': [1.0], 'for_model_legacy': {'a': 247.4, 'b': 0.99, 'c': -3.2e4}}))
    dedx = tmp_path / 'data' / 'water' / 'muon' / 'dedx_siren_training' / 'trained_model'
    dedx.mkdir(parents=True)
    stem = dedx / 'dedx_siren'
    assert repo_nphot_path(stem) is None
    out = _apply(stem, {'form': 'A*E^B+C', 'a': 0.91, 'b': 1.0, 'c': -9.7})   # would raise if applied
    assert 'coeffs' not in out


def test_absent_everywhere_gives_none(tmp_path):
    assert repo_nphot_path(_model_tree(tmp_path, ship=False)) is None


# ---- applying it ---------------------------------------------------------------------------------

def _apply(stem, nphot):
    fake = types.SimpleNamespace(model_path=stem, metadata={'nphot': dict(nphot)})
    SIRENPredictor._apply_repo_nphot(fake)
    return fake.metadata['nphot']


def test_matching_file_is_overlaid_without_its_bookkeeping(tmp_path):
    stem = _model_tree(tmp_path)
    shipped = tmp_path / 'data' / 'water' / 'muon' / 'nphot.json'
    shipped.write_text(json.dumps({'coeffs': [1.0], 'u0': 0.0, 'du': 1.0, 'form': 'logpoly',
                                   'for_model_legacy': {'a': 1.0, 'b': 2.0, 'c': 3.0},
                                   'source': 'provenance only'}))
    out = _apply(stem, {'form': 'A*E^B+C', 'a': 1.0, 'b': 2.0, 'c': 3.0})
    assert out['coeffs'] == [1.0] and out['_origin'] == str(shipped)
    assert 'for_model_legacy' not in out and 'source' not in out


def test_file_fitted_to_a_different_model_is_refused(tmp_path):
    stem = _model_tree(tmp_path)
    (tmp_path / 'data' / 'water' / 'muon' / 'nphot.json').write_text(json.dumps(
        {'coeffs': [1.0], 'for_model_legacy': {'a': 1.0, 'b': 2.0, 'c': 3.0}}))
    with pytest.raises(ValueError, match='different model'):
        _apply(stem, {'a': 9.0, 'b': 2.0, 'c': 3.0})


def test_no_file_leaves_the_block_alone_but_records_the_model(tmp_path):
    stem = _model_tree(tmp_path, ship=False)
    out = _apply(stem, {'form': 'A*E^B+C', 'a': 1.0, 'b': 2.0, 'c': 3.0})
    assert 'coeffs' not in out and out['_origin'] == str(stem)


# ---- the real models, when present ----------------------------------------------------------------

@pytest.mark.parametrize('particle', ['muon', 'electron'])
def test_real_model_uses_this_trees_file(particle):
    stem = os.path.join(ROOT, 'data', 'water', particle, 'siren_training', 'trained_model',
                        'photonsim_siren')
    if not os.path.exists(stem + '_weights.npz'):
        pytest.skip('trained SIREN model not present (fetched separately, not in the image)')
    pred = SIRENPredictor(stem)
    assert pred.metadata['nphot']['_origin'] == SHIPPED[particle]
    fn = make_power_law_fn(pred.metadata['nphot'])
    np.testing.assert_allclose(float(fn(1000.0)), YIELD_1GEV[particle], rtol=1e-4)
