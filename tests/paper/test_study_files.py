"""A paper figure is drawn only from finished, complete study files.

``ovv_plots.open_study`` is the one reader. Before it, each plot mode skipped a missing point, a
file still locked by its job, and a run that had lost events, and drew the figure anyway.
"""
import h5py
import pytest

from analysis.paper.utils.ovv_plots import find_study, open_study


def _study(path, requested=2, done=2, groups=2, finished=True, failed=None):
    with h5py.File(path, 'w') as f:
        f.attrs['n_events_requested'] = requested
        f.attrs['n_events_done'] = done
        if finished:
            f.attrs['finished'] = '2026-10-28T00:00:00'
        if failed is not None:
            f.attrs['n_events_failed'] = failed
        ev = f.create_group('events')
        for i in range(groups):
            ev.create_group(f'ev{i:04d}')
    return path


def test_a_complete_run_opens(tmp_path):
    with open_study(_study(tmp_path / 'a.h5')) as f:
        assert len(f['events']) == 2


@pytest.mark.parametrize('kw', [dict(finished=False), dict(done=1, groups=1),
                                dict(groups=1), dict(failed=1)])
def test_an_incomplete_run_is_refused(tmp_path, kw):
    with pytest.raises(ValueError, match='not a finished, complete study run'):
        open_study(_study(tmp_path / 'a.h5', **kw))


def test_a_missing_point_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_study(tmp_path / 'nrays_mu_5k')


def test_the_run_dir_names_its_file(tmp_path):
    d = tmp_path / 'nrays_mu_5k'; d.mkdir()
    _study(d / 'other.h5')
    assert find_study(d) == d / 'other.h5'
    _study(d / 'nrays_mu_5k.h5')
    assert find_study(d) == d / 'nrays_mu_5k.h5'


def test_momentum_error_is_dp_over_p_of_the_kinetic_energy_error():
    """dp/p = E_tot |dE| / p^2 (p^2 = E_tot^2 - m^2), checked against a central difference of p(T)
    for a muon and an electron; the momentum curves read each file's own particle."""
    import numpy as np
    from analysis.paper.utils.ovv_plots import mom_err_pct
    from analysis.paper.utils.studies import mass
    for particle, T in (('muon', 1000.0), ('muon', 300.0), ('electron', 50.0)):
        m = mass(particle)
        p = lambda t: np.sqrt((t + m) ** 2 - m ** 2)
        h, dE = 1e-3, 7.0
        want = (p(T + h) - p(T - h)) / (2 * h) * dE / p(T) * 100
        np.testing.assert_allclose(mom_err_pct(dE, T, m), want, rtol=1e-8)
