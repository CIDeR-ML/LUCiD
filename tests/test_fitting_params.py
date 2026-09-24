"""lucid.fitting.params.CalibrationParams — the theta <-> physical map.

Extracted from the reference calibration engine, where it lived as `build_dp`/`to_real` inside a
figure script. The engine pin (tests/reconciliation/test_calib_engine_pin.py) gates that the
extraction is bit-identical in the fit; these tests gate the map's own properties, cheaply and
without a simulator.
"""
import numpy as np
import pytest

from lucid.fitting.params import CalibrationParams

W = 5
WALL_R, WALL_F = 0.05, 0.55          # literature-anchored values the campaign uses
SENS_R, SENS_F = 0.25, 0.90
PHYS = [{'scatter_length': 100.0 + 10 * i,
         'absorption_length': 300.0 + 20 * i,
         'qe': 0.20 + 0.01 * i} for i in range(W)]


def _params(basis):
    return CalibrationParams(n_wavelengths=W, basis=basis)


@pytest.mark.parametrize('basis', ['rlogit', 'specdiff'])
def test_round_trip_to_real(basis):
    """theta_from_physical -> to_real returns the physical values it was built from."""
    p = _params(basis)
    th = p.theta_from_physical(PHYS, WALL_R, WALL_F, SENS_R, SENS_F)
    got = p.to_real(th)
    assert th.shape == (p.P,) and got.shape == (p.P,)
    for i, t in enumerate(PHYS):
        np.testing.assert_allclose(got[3 * i:3 * i + 3],
                                   [t['scatter_length'], t['absorption_length'], t['qe']],
                                   rtol=1e-12)
    np.testing.assert_allclose(
        got[p.NG:], [WALL_R * WALL_F, WALL_R * (1 - WALL_F),
                     SENS_R * SENS_F, SENS_R * (1 - SENS_F)], rtol=1e-12)


def test_bases_agree_on_physical_content():
    """rlogit and specdiff are different COORDINATES for the same physics, not different models.

    Their theta vectors differ; what they report must not.
    """
    a, b = _params('rlogit'), _params('specdiff')
    ta = a.theta_from_physical(PHYS, WALL_R, WALL_F, SENS_R, SENS_F)
    tb = b.theta_from_physical(PHYS, WALL_R, WALL_F, SENS_R, SENS_F)
    assert not np.allclose(ta[a.NG:], tb[b.NG:]), 'the two bases should differ in theta'
    np.testing.assert_allclose(a.to_real(ta), b.to_real(tb), rtol=1e-12)


def test_reporting_basis_amplifies_the_diffuse_component():
    """The ×9 lever, asserted rather than left as folklore.

    We fit (log R, logit f) and report R·f and R·(1−f). Since
    d log(1−f)/d log f = −f/(1−f) = −9 at f = 0.90, a small error in the FITTED parameter shows up
    ~9× larger in the REPORTED diffuse component. This is why quoting (R, f) — or stating the
    lever — matters when reporting recovery.
    """
    p = _params('rlogit')
    th = p.theta_from_physical(PHYS, WALL_R, WALL_F, SENS_R, SENS_F)
    base = p.to_real(th)
    eps = 1e-4
    per = th.copy(); per[p.NG + 3] += eps                      # perturb logit f_s only
    got = p.to_real(per)
    d_spec = abs(got[p.NG + 2] / base[p.NG + 2] - 1)           # R_s·f
    d_diff = abs(got[p.NG + 3] / base[p.NG + 3] - 1)           # R_s·(1−f)
    ratio = d_diff / d_spec
    assert 8.0 < ratio < 10.0, f'expected ~f/(1-f) = 9 at f=0.90, got {ratio:.2f}'


def test_matches_the_reference_engines_inline_construction():
    """theta_from_physical reproduces what calib_fit.py built by hand.

    Guards the one part of the extraction the engine pin does NOT cover: the pin exercises to_dp
    and to_real through the fit, but the engine still builds theta0 inline, so an inconsistency
    here would go unnoticed until someone used the library helper and got a different start.
    """
    p = _params('rlogit')
    # the engine's construction, transcribed from calib_fit.py:98-108
    tvec_opt = []
    for t in PHYS:
        tvec_opt += [t['scatter_length'], t['absorption_length'], t['qe']]
    _logit = lambda f: np.log(f / (1 - f))
    expected = np.array(list(np.log(tvec_opt)) +
                        [np.log(WALL_R), _logit(WALL_F), np.log(SENS_R), _logit(SENS_F)])
    np.testing.assert_allclose(
        p.theta_from_physical(PHYS, WALL_R, WALL_F, SENS_R, SENS_F), expected, rtol=0, atol=0)


def test_mie_is_frozen_and_visible():
    """Mie is dropped, not folded in — and that must be a visible argument, not a buried literal."""
    p = _params('rlogit')
    assert p.mie_scatter_length == 1e6 and p.g == 0.9
    assert CalibrationParams(n_wavelengths=W, basis='rlogit', mie_scatter_length=123.0).mie_scatter_length == 123.0


def test_rejects_an_unknown_basis():
    with pytest.raises(ValueError, match='basis'):
        CalibrationParams(n_wavelengths=W, basis='nope')
