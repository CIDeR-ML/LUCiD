"""The lookup-table builder's N_photons(E) fit: the log-log polynomial the emitter evaluates."""
import numpy as np
import pytest

pytest.importorskip('uproot')
from lucid.siren.emitter import make_n_photons_fn                                     # noqa: E402
from lucid.siren.training.photonsim_data.build_tables import _fit_n_photons          # noqa: E402

E = np.unique(np.geomspace(150, 100000, 300).astype(int))
TRUE = np.array([13.7, 3.35, -0.24, 0.32, -0.19, 0.07, -0.16, 0.12])   # the shipped muon fit, rounded


def _known(E):
    t = (np.log(E) - 0.5 * np.log(150 * 100000)) / (0.5 * np.log(100000 / 150))
    return np.exp(np.polynomial.polynomial.polyval(t, TRUE))


def test_a_known_logpoly_is_recovered():
    fit = _fit_n_photons(E, _known(E))
    np.testing.assert_allclose(fit['coeffs'], TRUE, atol=1e-10)
    assert np.isclose(np.exp(fit['u0'] - fit['du']), 150) and np.isclose(np.exp(fit['u0'] + fit['du']), 100000)
    assert fit['max_frac'] < 1e-10
    # the block is what the emitter evaluates
    np.testing.assert_allclose(np.asarray(make_n_photons_fn(fit)(E.astype(float))), _known(E), rtol=1e-5)


def test_an_energy_without_photons_is_refused():
    y = _known(E)
    y[0] = 0.0
    with pytest.raises(ValueError, match='no photons'):
        _fit_n_photons(E, y)
