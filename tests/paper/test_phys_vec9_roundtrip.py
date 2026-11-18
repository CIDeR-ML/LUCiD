"""The loss-landscape figure's 9-vector <-> physical-parameter maps are exact inverses, and the
physical slots are the ones `PHYS_NAMES` names. Pure numpy (no ROOT input, SIREN weights or GPU), so it always runs.
"""
import numpy as np
import pytest

from lucid.fitting import track_from_vec9, vec9_dir, vec9_from_track
from analysis.paper.fig_loss_landscape import PHYS_NAMES, phys_to_vec9, vec9_to_phys

# (vertex, unit direction, energy MeV, t0 ns). The third case has an azimuth and a t0 of similar
# magnitude and opposite sign, where a swap of the two would still look plausible.
CASES = [
    ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 1000.0, 0.0),
    ((-4.5, 0.25, -8.0), (0.6, -0.8, 0.0), 500.0, 3.75),
    ((6.5875, -3.2625, 8.0187), (0.7878 * np.cos(-3.6049), 0.7878 * np.sin(-3.6049), 0.6160),
     1000.0, -2.7618),
]


@pytest.mark.parametrize('vtx,d,E,t0', CASES)
def test_phys_to_vec9_inverts_vec9_to_phys(vtx, d, E, t0):
    v9 = vec9_from_track(E, vtx, d, t0=t0)
    assert np.allclose(phys_to_vec9(vec9_to_phys(v9)), v9, rtol=0, atol=1e-12)


@pytest.mark.parametrize('vtx,d,E,t0', CASES)
def test_vec9_to_phys_inverts_phys_to_vec9(vtx, d, E, t0):
    p = vec9_to_phys(vec9_from_track(E, vtx, d, t0=t0))
    assert np.allclose(vec9_to_phys(phys_to_vec9(p)), p, rtol=0, atol=1e-12)


def test_the_physical_slots_are_the_named_ones():
    p = dict(zip(PHYS_NAMES, vec9_to_phys(vec9_from_track(750.0, (1.0, 2.0, 3.0), (0.6, -0.8, 0.0),
                                                          t0=4.25))))
    assert (p['x'], p['y'], p['z'], p['E'], p['t0']) == (1.0, 2.0, 3.0, 750.0, 4.25)
    assert np.isclose(p['theta'], np.pi / 2)          # direction is in the z=0 plane
    assert np.isclose(p['phi'], np.arctan2(-0.8, 0.6))


@pytest.mark.parametrize('vtx,d,E,t0', CASES + [((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 300.0, 1.0),
                                                ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 300.0, 0.0)])
def test_the_simulator_sees_the_track_it_was_given(vtx, d, E, t0):
    """``track_from_vec9`` (what the simulator receives) decodes ``vec9_from_track`` exactly.

    Its angle decode (``vec9_angles``) is shared with ``vec9_to_phys``, its direction then comes
    from ``ParticleParams.direction``, and ``vec9_dir`` decodes the pairs on its own; checked side
    by side. The extra cases are the poles of the polar angle and an azimuth of pi, where arctan2
    branch handling matters.
    """
    v9 = vec9_from_track(E, vtx, d, t0=t0)
    tr = track_from_vec9(v9)
    d = np.asarray(d, float) / np.linalg.norm(d)
    np.testing.assert_allclose(np.asarray(tr.direction), d, atol=1e-6)
    np.testing.assert_allclose(vec9_dir(v9), d, atol=1e-12)
    np.testing.assert_allclose(np.asarray(tr.position), vtx, atol=1e-6)
    assert float(tr.energy) == pytest.approx(E) and float(tr.t0) == pytest.approx(t0, abs=1e-6)


def test_vec9_dir_refuses_a_zero_sin_cos_pair():
    import pytest
    from lucid.fitting import vec9_dir
    with pytest.raises(ValueError, match='zero sin/cos pair'):
        vec9_dir([1000., 0., 0., 0., 0., 0., 1., 0., 0.])


def test_vec9_dir_decodes_a_trajectory_at_any_pair_scale():
    """A ``(T, 9)`` trajectory decodes row by row, and scaling a sin/cos pair (an unnormalised
    Gauss-Newton iterate) leaves the decoded direction unit and unchanged: the decode divides by
    the pair's norm and nothing else."""
    traj = np.stack([vec9_from_track(E, vtx, d, t0=t0) for vtx, d, E, t0 in CASES])
    dirs = vec9_dir(traj)
    assert dirs.shape == (len(CASES), 3)
    for row, d in zip(traj, dirs):
        np.testing.assert_array_equal(vec9_dir(row), d)
    for s in (1e-6, 1e-3, 7.0, 1e3):
        scaled = traj.copy()
        scaled[:, 4:6] *= s
        scaled[:, 6:8] *= 1.0 / s
        np.testing.assert_allclose(vec9_dir(scaled), dirs, rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(vec9_dir(scaled), axis=1), 1.0, rtol=0, atol=1e-15)
