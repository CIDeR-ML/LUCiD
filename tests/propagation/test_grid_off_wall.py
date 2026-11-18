"""
Systematic probe: how does the cylinder grid lookup behave when
sensors are not exactly on the wall?

The sensor → grid-cell assignment in
``lucid.propagation.cylinder.assign_sensors_to_grid`` classifies a
sensor as "on the wall" only when

    | sqrt(x**2 + y**2) - R |  <=  sensor_radius

A sensor further off the nominal wall than ``sensor_radius`` fails
all three branch tests (on_wall / on_top / on_bottom) and falls
through to the no-assignment return: it is silently invisible to
the grid lookup, and rays will never find it.

These tests pin that behaviour down so the snap-to-wall default in
:meth:`Cylinder.from_pmt_file` cannot regress without us noticing. That the snapped real
detectors (SK, HK, WCTE) leave no sensor out of the grid at the default grid, and overflow no
cell, is checked in tests/propagation/test_sensor_map_validation.py.
"""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from lucid.geometry import Cylinder
from lucid.propagation.cylinder import assign_sensors_to_grid
from tests.helpers.paths import REPO


@pytest.fixture(scope="module")
def cyl_dims():
    """Detector envelope used by these probes — chosen to roughly
    match an SK-scale cylinder so sensor_radius = 0.254 m."""
    return dict(r=10.0, h=20.0, sensor_radius=0.254,
                n_cap=10, n_angular=20, n_height=10)


def _has_assignment(assignments_row):
    """A sensor was assigned if any of its 4 cell slots is not -1."""
    return bool(jnp.any(assignments_row != -1))


# ── Direct probes against assign_sensors_to_grid ──────────────────────


def test_sweep_off_wall_distances(cyl_dims):
    """A barrel sensor is assigned exactly when it sits within sensor_radius of the wall
    (inclusive), and silently dropped beyond it."""
    R = cyl_dims['r']; sr = cyl_dims['sensor_radius']
    distances = np.array([0.0, 0.001, 0.014, 0.030, sr * 0.5, sr * 0.9, sr,
                          sr * 1.001, sr * 1.05, sr * 1.5, 2 * sr, 0.5, 1.0])
    sensors = jnp.stack([
        jnp.array([(R - d), 0.0, 0.0]) for d in distances
    ])
    result = assign_sensors_to_grid(sensors, sr, R, cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assigned = [_has_assignment(result[i]) for i in range(len(distances))]
    expected = [d <= sr for d in distances]
    assert assigned == expected, (
        f"Off-wall assignment sweep: expected {expected}, got {assigned}\n"
        f"distances: {distances.tolist()}"
    )


def test_off_wall_boundary_is_inclusive(cyl_dims):
    """|r - R| == sensor_radius is still on the wall. 0.254 is not exact in binary, so the sweep
    above cannot hit the boundary itself; with R = 10 and sensor_radius = 0.25 every value below
    is exact in float32, and R - d == 9.75 sits exactly on the threshold."""
    R, sr = 10.0, 0.25
    sensors = jnp.array([[R - sr, 0.0, 0.0], [R - sr - 2.0 ** -18, 0.0, 0.0]])
    result = assign_sensors_to_grid(sensors, sr, R, cyl_dims['h'], cyl_dims['n_cap'],
                                    cyl_dims['n_angular'], cyl_dims['n_height'])
    assert [_has_assignment(result[i]) for i in range(2)] == [True, False]


def test_sweep_off_cap_distances(cyl_dims):
    """The caps use the same tolerance along z: a cap sensor (r_xy well inside the wall) is
    assigned while it is within sensor_radius of z = +-h/2 and dropped beyond it."""
    h = cyl_dims['h']; sr = cyl_dims['sensor_radius']
    offs = np.array([0.0, 0.5 * sr, 0.9 * sr, 1.1 * sr, 2 * sr])
    sensors = jnp.array([[1.0, 2.0, s * (h / 2 - d)] for s in (1, -1) for d in offs])
    result = assign_sensors_to_grid(sensors, sr, cyl_dims['r'], h, cyl_dims['n_cap'],
                                    cyl_dims['n_angular'], cyl_dims['n_height'])
    assigned = [_has_assignment(result[i]) for i in range(len(sensors))]
    expected = [d < sr for _ in (1, -1) for d in offs]
    assert assigned == expected, (assigned, expected)
    # and to the right cap: part code 1 on top, 2 on the bottom
    assert int(result[0, 0, 2]) == 1 and int(result[len(offs), 0, 2]) == 2


# ── End-to-end: a real-detector npz with snap off ──────────────────────


@pytest.fixture(scope="module")
def repo_root():
    return str(REPO)


@pytest.mark.slow                 # loads the SK and WCTE npz geometries
def test_snap_off_drops_off_wall_sensors_for_wcte_only(repo_root):
    """Without snapping, SK keeps every PMT and WCTE drops some.

    SK's barrel rings alternate by ~14 mm, well inside its 254 mm sensor_radius, so every PMT is
    still within the on-wall tolerance. WCTE's mPMT domes hold their central PMTs ~67 mm in from
    the wall, beyond its 40 mm sensor_radius, so those fall out of the grid unless snapped.
    """
    sk_path   = os.path.join(repo_root, 'config', 'sk_geometry.npz')
    wcte_path = os.path.join(repo_root, 'config', 'wcte_geometry.npz')

    def n_unassigned(path):
        det = Cylinder.from_pmt_file(path, snap_to_wall=False)
        a = det.assign_sensor_to_cells(jnp.asarray(det.all_points), float(det.S_radius),
                                       det.make_grid())
        return int(jnp.sum(~jnp.any(a != -1, axis=(1, 2))))

    assert n_unassigned(sk_path) == 0
    assert n_unassigned(wcte_path) > 0
