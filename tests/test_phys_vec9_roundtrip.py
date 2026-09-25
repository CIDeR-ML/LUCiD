"""The 9-vector <-> physical-parameter maps must be exact inverses.

`vec9_to_phys` returns `[x, y, z, phi, theta, t0, E]`, which is easy to misread as
`[x, y, z, t0, theta, phi, E]` -- the angles bracket a scalar either way, and both readings put
theta in the middle. A hand-rolled inverse with that swap puts t0 into the azimuth slot and the
azimuth into t0, and nothing else catches it: perturbing a coordinate still moves the quantity its
label names, so every derivative is correct, but it is taken at the wrong POINT and a result
reported "at truth" is not.

So the inverse lives once, beside the forward map, and this pins the round trip in both
directions. Pure numpy (no ROOT input, SIREN weights or GPU), so it always runs.
"""
import numpy as np
import pytest

from analysis.paper.utils.pipeline import phys_to_vec9, truth9, vec9_to_phys

# (vertex, unit direction, energy MeV, t0 ns). The third case has an azimuth and a t0 of similar
# magnitude and opposite sign, where a swap stays plausible-looking.
CASES = [
    ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 1000.0, 0.0),
    ((-4.5, 0.25, -8.0), (0.6, -0.8, 0.0), 500.0, 3.75),
    ((6.5875, -3.2625, 8.0187), (0.7878 * np.cos(-3.6049), 0.7878 * np.sin(-3.6049), 0.6160),
     1000.0, -2.7618),
]


def _unit(d):
    d = np.asarray(d, float)
    return d / np.linalg.norm(d)


@pytest.mark.parametrize('vtx,d,E,t0', CASES)
def test_phys_to_vec9_inverts_vec9_to_phys(vtx, d, E, t0):
    v9, _ = truth9(np.asarray(vtx, float), _unit(d), E, t0=t0)
    assert np.allclose(phys_to_vec9(vec9_to_phys(v9)), v9, rtol=0, atol=1e-12)


@pytest.mark.parametrize('vtx,d,E,t0', CASES)
def test_vec9_to_phys_inverts_phys_to_vec9(vtx, d, E, t0):
    v9, _ = truth9(np.asarray(vtx, float), _unit(d), E, t0=t0)
    p = vec9_to_phys(v9)
    assert np.allclose(vec9_to_phys(phys_to_vec9(p)), p, rtol=0, atol=1e-12)


def test_the_physical_slots_are_where_the_docstring_says():
    """Name the indices explicitly, so a reordering fails here rather than in a gradient run."""
    v9, _ = truth9(np.array([1.0, 2.0, 3.0]), _unit((0.6, -0.8, 0.0)), 750.0, t0=4.25)
    x, y, z, phi, theta, t0, E = vec9_to_phys(v9)
    assert (x, y, z) == (1.0, 2.0, 3.0)
    assert E == 750.0
    assert t0 == 4.25
    assert np.isclose(theta, np.pi / 2)              # direction is in the z=0 plane
    assert np.isclose(phi, np.arctan2(-0.8, 0.6))


def test_swapping_phi_and_t0_is_detected():
    """A phi<->t0-swapped inverse must fail the round trip, not slip through.

    The round-trip tests alone only prove the two functions agree with each other. t0 is kept well
    away from the azimuth, since a t0 close to it would round-trip under the swap.
    """
    v9, _ = truth9(np.array([1.0, 2.0, 3.0]), _unit((0.6, -0.8, 0.0)), 750.0, t0=4.25)
    p = vec9_to_phys(v9)

    def swapped(q):                                   # inverse with phi and t0 swapped
        return np.stack([q[6], q[0], q[1], q[2],
                         np.sin(q[4]), np.cos(q[4]), np.sin(q[5]), np.cos(q[5]), q[3]])

    assert not np.allclose(swapped(p), v9, rtol=0, atol=1e-8)


# --------------------------------------------------------------------------------------------
# The SECOND convention. `fig_loss_landscape.py` keeps the notebook order
# [X, Y, Z, t0, theta, phi, E] because its scan pairs, half-ranges and labels are indexed in it.
# It is a different ordering, not a wrong one — the hazard is only ever mixing the two, so these
# pin each pair against ITS OWN forward map and then pin that the pairs are genuinely different.


@pytest.mark.parametrize('vtx,d,E,t0', CASES)
def test_notebook_pair_round_trips(vtx, d, E, t0):
    from analysis.paper.fig_loss_landscape import phys_from_vec9, vec9_from_phys
    v9, _ = truth9(np.asarray(vtx, float), _unit(d), E, t0=t0)
    assert np.allclose(vec9_from_phys(phys_from_vec9(v9)), v9, rtol=0, atol=1e-12)


def test_the_two_conventions_are_not_interchangeable():
    """Crossing the pair boundary must produce a DIFFERENT 9-vector, or none of this matters."""
    from analysis.paper.fig_loss_landscape import phys_from_vec9, vec9_from_phys
    v9, _ = truth9(np.array([1.0, 2.0, 3.0]), _unit((0.6, -0.8, 0.0)), 750.0, t0=4.25)
    # notebook inverse fed the pipeline's physical vector
    assert not np.allclose(vec9_from_phys(vec9_to_phys(v9)), v9, rtol=0, atol=1e-8)
    # and the mirror image
    assert not np.allclose(phys_to_vec9(phys_from_vec9(v9)), v9, rtol=0, atol=1e-8)
    # the two physical vectors differ in exactly the phi and t0 slots
    a, b = vec9_to_phys(v9), phys_from_vec9(v9)
    assert np.allclose(a[[0, 1, 2, 4, 6]], b[[0, 1, 2, 4, 6]])
    assert np.allclose([a[3], a[5]], [b[5], b[3]])
