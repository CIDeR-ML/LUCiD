"""lucid.fitting.params.CalibrationParams — the theta <-> physical map.

These tests check the map's own properties, cheaply and without a simulator.
"""
import numpy as np
import pytest

from lucid.fitting.params import CalibrationParams

W = 5
WALL_R, WALL_F = 0.05, 0.55          # literature-anchored values
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
    """The fit-to-report basis change amplifies errors in the diffuse component ~9x at f = 0.90.

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


@pytest.mark.parametrize('basis', ['rlogit', 'specdiff'])
def test_what_the_simulator_sees_is_what_is_reported(basis):
    """``to_dp`` (the forward's input) and ``to_real`` (the report) describe the same physics.

    The round trip above only exercises ``theta_from_physical`` and ``to_real``; a layout or sign
    error in ``to_dp`` alone would fit one model and report another. For every wavelength the
    DetectorParams leaves must equal the reported optics, and the reflection rates and specular
    fractions must recombine into the reported spec/diff components, all against the physical
    values the vector was built from.
    """
    from lucid.detector_params import _flatten_detector_params
    p = _params(basis)
    th = p.theta_from_physical(PHYS, WALL_R, WALL_F, SENS_R, SENS_F)
    real = p.to_real(th)
    gains = np.linspace(0.9, 1.1, 7)
    for wl, t in enumerate(PHYS):
        f = {k: np.asarray(v) for k, v in _flatten_detector_params(p.to_dp(th, wl, gains)).items()}
        got = [f['scatter_length'], f['absorption_length'], f['qe']]
        np.testing.assert_allclose(got, [t['scatter_length'], t['absorption_length'], t['qe']],
                                   rtol=1e-5)                       # float32 exp
        np.testing.assert_allclose(got, real[3 * wl:3 * wl + 3], rtol=1e-5)
        Rw, fw = f['wall_reflection_rate'], f['wall_fspec']
        Rs, fs = f['sensor_reflection_rate'], f['sensor_fspec']
        np.testing.assert_allclose([Rw, fw, Rs, fs], [WALL_R, WALL_F, SENS_R, SENS_F], rtol=1e-5)
        np.testing.assert_allclose([Rw * fw, Rw * (1 - fw), Rs * fs, Rs * (1 - fs)], real[p.NG:],
                                   rtol=1e-5)
        np.testing.assert_allclose(f['qe_corrections'], gains, rtol=1e-6)
        # Mie is frozen at the constructor's values in what the simulator sees, not just stored
        assert float(f['mie_scatter_length']) == 1e6 and float(f['g']) == pytest.approx(0.9)
    other = CalibrationParams(n_wavelengths=W, basis=basis, mie_scatter_length=123.0, g=0.5)
    f = _flatten_detector_params(other.to_dp(th, 0, gains))
    assert float(f['mie_scatter_length']) == 123.0 and float(f['g']) == 0.5


def test_rejects_an_unknown_basis():
    with pytest.raises(ValueError, match='basis'):
        CalibrationParams(n_wavelengths=W, basis='nope')
