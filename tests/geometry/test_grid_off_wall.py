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
:meth:`Cylinder.from_pmt_file` cannot regress without us noticing.
"""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from lucid.geometry import Cylinder
from lucid.propagation.cylinder import assign_sensors_to_grid


@pytest.fixture(scope="module")
def cyl_dims():
    """Detector envelope used by these probes — chosen to roughly
    match an SK-scale cylinder so sensor_radius = 0.254 m."""
    return dict(r=10.0, h=20.0, sensor_radius=0.254,
                n_cap=10, n_angular=20, n_height=10)


def _make_barrel_sensor(theta, z, r_off, R):
    """A single barrel sensor at azimuth ``theta`` and height ``z``,
    pushed inward by ``r_off`` (positive = inside the wall)."""
    return jnp.array([[(R - r_off) * np.cos(theta),
                       (R - r_off) * np.sin(theta),
                       z]])


def _has_assignment(assignments_row):
    """A sensor was assigned if any of its 4 cell slots is not -1."""
    return bool(jnp.any(assignments_row != -1))


# ── Direct probes against assign_sensors_to_grid ──────────────────────


def test_sensor_exactly_on_wall_is_assigned(cyl_dims):
    sensors = _make_barrel_sensor(theta=0.0, z=0.0, r_off=0.0, R=cyl_dims['r'])
    result = assign_sensors_to_grid(sensors, cyl_dims['sensor_radius'],
                                     cyl_dims['r'], cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assert _has_assignment(result[0]), "Sensor on the wall must be assigned"


def test_sensor_inside_wall_within_sensor_radius_is_assigned(cyl_dims):
    """A sensor pushed inward by less than sensor_radius should still
    be classified as on_wall (the |r-R|<=sensor_radius test passes)."""
    half_radius = cyl_dims['sensor_radius'] * 0.5
    sensors = _make_barrel_sensor(0.0, 0.0, r_off=half_radius, R=cyl_dims['r'])
    result = assign_sensors_to_grid(sensors, cyl_dims['sensor_radius'],
                                     cyl_dims['r'], cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assert _has_assignment(result[0]), \
        "Sensor inside the wall by < sensor_radius must still be assigned"


def test_sensor_off_wall_at_threshold_is_assigned(cyl_dims):
    """At exactly r_off == sensor_radius the boundary test is inclusive
    (`<=`) — the sensor should still be assigned."""
    sensors = _make_barrel_sensor(0.0, 0.0, r_off=cyl_dims['sensor_radius'],
                                  R=cyl_dims['r'])
    result = assign_sensors_to_grid(sensors, cyl_dims['sensor_radius'],
                                     cyl_dims['r'], cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assert _has_assignment(result[0]), \
        "Sensor at the |r-R|<=sensor_radius boundary must be assigned"


def test_sensor_off_wall_just_beyond_threshold_is_invisible(cyl_dims):
    """As soon as r_off > sensor_radius the on_wall test fails. With
    z==0 the on_top/on_bottom tests also fail, so the sensor falls
    through to the no-assignment branch — silently invisible."""
    just_beyond = cyl_dims['sensor_radius'] * 1.05
    sensors = _make_barrel_sensor(0.0, 0.0, r_off=just_beyond,
                                  R=cyl_dims['r'])
    result = assign_sensors_to_grid(sensors, cyl_dims['sensor_radius'],
                                     cyl_dims['r'], cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assert not _has_assignment(result[0]), (
        "REGRESSION GUARD: a sensor more than sensor_radius off the "
        "wall must drop out of the grid (invisible to ray-tracing). "
        "If this assertion ever fails the off-wall behaviour has "
        "changed and the snap_to_wall default can be reconsidered."
    )


def test_sensor_far_off_wall_is_invisible(cyl_dims):
    sensors = _make_barrel_sensor(0.0, 0.0, r_off=1.0, R=cyl_dims['r'])
    result = assign_sensors_to_grid(sensors, cyl_dims['sensor_radius'],
                                     cyl_dims['r'], cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assert not _has_assignment(result[0])


def test_sweep_off_wall_distances(cyl_dims):
    """Print-friendly sweep that documents the breakdown distance."""
    R = cyl_dims['r']; sr = cyl_dims['sensor_radius']
    distances = np.array([0.0, 0.001, 0.014, 0.030, sr * 0.9, sr,
                          sr * 1.001, sr * 1.5, 2 * sr, 0.5, 1.0])
    sensors = jnp.stack([
        jnp.array([(R - d), 0.0, 0.0]) for d in distances
    ])
    result = assign_sensors_to_grid(sensors, sr, R, cyl_dims['h'],
                                     cyl_dims['n_cap'], cyl_dims['n_angular'],
                                     cyl_dims['n_height'])
    assigned = [_has_assignment(result[i]) for i in range(len(distances))]
    # The transition happens exactly at sensor_radius
    expected = [d <= sr for d in distances]
    assert assigned == expected, (
        f"Off-wall assignment sweep: expected {expected}, got {assigned}\n"
        f"distances: {distances.tolist()}"
    )


# ── End-to-end: a real-detector npz with snap on vs off ────────────────


@pytest.fixture(scope="module")
def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_snap_default_keeps_all_sensors_on_grid(repo_root):
    """For each of the four real detectors, with snap_to_wall=True
    every sensor must end up assigned to at least one grid cell.
    This is a smoke test that the loaded geometry is actually usable
    by the propagator."""
    # Only the publicly-distributable geofile-derived npz files are
    # exercised here — *_official files are not in git for many users.
    for npz in ('sk_geometry.npz', 'hk_geometry.npz', 'wcte_geometry.npz'):
        path = os.path.join(repo_root, 'config', npz)
        det = Cylinder.from_pmt_file(path, snap_to_wall=True)
        det.configure_grid()
        assignments = det.assign_sensor_to_cells(
            jnp.asarray(det.all_points), float(det.S_radius)
        )
        # Each sensor row is shape (4, 3); -1 means no assignment.
        # Row is "missing" if all 4 slots are -1.
        any_slot = jnp.any(assignments != -1, axis=(1, 2))
        n_missing = int(jnp.sum(~any_slot))
        assert n_missing == 0, (
            f"{npz}: {n_missing}/{len(det.all_points)} sensors fell "
            f"out of the grid even with snap_to_wall=True — snap is "
            f"not aggressive enough or radius/height is wrong."
        )


def test_snap_off_drops_off_wall_sensors_for_sk(repo_root):
    """SK's barrel has alternating-ring offsets of ~14 mm. With
    sensor_radius = 0.254 m those are still within the |r-R|<=sr
    tolerance, so we don't expect any to drop. WCTE's mPMT domes
    push central PMTs ~67 mm in — also within 0.040 m of the wall?
    No, 67 mm > 40 mm sensor_radius, so WCTE off-wall PMTs WILL
    drop without snap. This pins both behaviours."""
    sk_path   = os.path.join(repo_root, 'config', 'sk_geometry.npz')
    wcte_path = os.path.join(repo_root, 'config', 'wcte_geometry.npz')

    # SK without snap — 14 mm < 254 mm sensor_radius → all assigned
    det = Cylinder.from_pmt_file(sk_path, snap_to_wall=False)
    det.configure_grid()
    assignments = det.assign_sensor_to_cells(
        jnp.asarray(det.all_points), float(det.S_radius)
    )
    any_slot = jnp.any(assignments != -1, axis=(1, 2))
    sk_missing = int(jnp.sum(~any_slot))
    assert sk_missing == 0, (
        f"Even SK's 14 mm barrel offsets should fit inside its "
        f"254 mm sensor_radius window; got {sk_missing} missing."
    )

    # WCTE without snap — central mPMT PMTs are ~67 mm in vs
    # sensor_radius=40 mm → must drop a sizable number
    det = Cylinder.from_pmt_file(wcte_path, snap_to_wall=False)
    det.configure_grid()
    assignments = det.assign_sensor_to_cells(
        jnp.asarray(det.all_points), float(det.S_radius)
    )
    any_slot = jnp.any(assignments != -1, axis=(1, 2))
    wcte_missing = int(jnp.sum(~any_slot))
    assert wcte_missing > 0, (
        "Without snap, WCTE's mPMT-dome PMTs that sit > 40 mm in "
        "from the wall should be invisible to the grid; got 0 missing."
    )


# ── Propagator sensor-map regression ──────────────────────────────────


def _forward_assignment_stats(det):
    """Count how many forward-assigned sensors each grid cell has,
    and how many sensors end up fully orphaned."""
    import numpy as np
    fwd = np.asarray(det.assign_sensor_to_cells(
        jnp.asarray(det.all_points), float(det.S_radius)))
    total_cells = det.total_grid_cells()
    cell_counts = np.zeros(total_cells, dtype=int)
    for s in range(fwd.shape[0]):
        for slot in range(fwd.shape[1]):
            c = fwd[s, slot]
            if (c == -1).all():
                continue
            idx = int(det.point_to_grid_cell_from_coords(c))
            if 0 <= idx < total_cells:
                cell_counts[idx] += 1
    row_has_any = np.any(fwd.reshape(len(fwd), -1, 3).max(axis=2) != -1, axis=1)
    orphans = int((~row_has_any).sum())
    return cell_counts, orphans


@pytest.mark.parametrize("npz,label", [
    ('sk_geometry.npz',    'SK'),
    ('hk_geometry.npz',    'HK'),
    ('wcte_geometry.npz',  'WCTE'),
])
def test_default_grid_respects_max_candidates_per_ray(repo_root, npz, label):
    """With the default configure_grid (nearest-neighbour sizing) plus
    the default max_candidates_per_ray=4, no grid cell must receive more
    than 4 geometric forward assignments and no sensor may be orphaned.

    Previously the grid was sized from cylinder surface area with a
    safety factor, which silently undercounted for mPMT-clustered
    layouts (WCTE: ~500 cells with >4 sensors, ~100 sensors dropped
    from the inverse map). The nearest-neighbour rule adapts to the
    actual sensor layout."""
    det = Cylinder.from_pmt_file(
        os.path.join(repo_root, 'config', npz), snap_to_wall=True)
    det.configure_grid()            # default: max_candidates_per_ray=4
    cell_counts, orphans = _forward_assignment_stats(det)
    assert orphans == 0, f"{label}: {orphans} sensors orphaned from the grid"
    assert int(cell_counts.max()) <= 4, (
        f"{label}: max forward assignments per cell is "
        f"{cell_counts.max()} (> default max_candidates_per_ray=4); "
        f"configure_grid should be producing a finer grid. "
        f"{(cell_counts > 4).sum()} cells overflow."
    )


@pytest.mark.parametrize("npz,label", [
    ('sk_geometry.npz',    'SK'),
    ('hk_geometry.npz',    'HK'),
    ('wcte_geometry.npz',  'WCTE'),
])
def test_create_propagator_emits_no_sensor_map_warnings(repo_root, npz, label):
    """End-to-end: running create_propagator on each real detector
    must not trigger ANY of the ``Sensor map:`` validator warnings
    (missing sensors, overcrowding, or forward/inverse inconsistency).
    """
    import warnings as pywarnings
    from lucid.propagation.shared import create_propagator

    det = Cylinder.from_pmt_file(
        os.path.join(repo_root, 'config', npz), snap_to_wall=True)
    with pywarnings.catch_warnings(record=True) as w_list:
        pywarnings.simplefilter("always")
        create_propagator(det, jnp.asarray(det.all_points),
                          float(det.S_radius), temperature=0.2)
    sensor_map_warnings = [w for w in w_list
                           if "Sensor map" in str(w.message)]
    assert not sensor_map_warnings, (
        f"{label}: create_propagator emitted Sensor-map warnings: "
        + "; ".join(str(w.message) for w in sensor_map_warnings)
    )
