"""The polygonal-prism geometry, and the SK inner detector it exists to describe.

Two independent things are pinned here:

1. **The declared polygon is the real one.** SK's barrel is a 38-gon
   (``WCBarrelNumPMTHorizontal/WCPMTperCellHorizontal = 152/4``) of apothem
   ``WCIDDiameter/2``, and every PMT in ``config/sk_geometry.npz`` sits exactly
   ``pmt_offset = sphere_radius - expose_height`` outside it. No fitting, no snapping —
   if the declaration were wrong the residual would be centimetres, not picometres.
2. **The ray tracer is exact.** The convex-polytope exit is checked against a
   brute-force reduction over all 40 half-spaces, including which face is hit.
"""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from lucid.geometry import PolygonalCylinder, generate_detector
from lucid.propagation.polygon import (
    batch_intersect_polygon_with_grid,
    calculate_polygon_normals,
    polygon_bounds_check,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# WCSim SetSuperKGeometry + PMT20inch
SK_N_SIDES = 38
SK_APOTHEM = 16.84075          # WCIDDiameter / 2
SK_HEIGHT = 36.2               # WCIDHeight
SK_EXPOSE = 0.18
SK_APERTURE = 0.254
SK_SPHERE_R = 0.269211111111   # (expose^2 + aperture^2) / (2 expose)
SK_OFFSET = SK_SPHERE_R - SK_EXPOSE


@pytest.fixture(scope="module")
def sk():
    return generate_detector(os.path.join(REPO, 'config', 'SK_geom_config.json'))


@pytest.fixture(scope="module")
def toy():
    """A small prism with algorithmic placement — fast, and exercises the non-npz path."""
    det = PolygonalCylinder(n_sides=11, apothem=3.0, height=7.0,
                            n_sensors=600, sensor_radius=0.1, expose_height=0.06)
    det.configure_grid()
    return det


# ── 1. the SK polygon ─────────────────────────────────────────────────


def test_sk_loads_as_polygon(sk):
    assert isinstance(sk, PolygonalCylinder)
    assert sk.n_sides == SK_N_SIDES
    assert sk.apothem == pytest.approx(SK_APOTHEM)
    assert sk.H == pytest.approx(SK_HEIGHT)
    assert len(sk.all_points) == 11096


def test_sk_pmt_centres_sit_exactly_pmt_offset_outside_the_wall(sk):
    """The cross-check. No snapping is applied, so this is the geometry speaking."""
    worst = float(np.abs(sk.surface_offsets() - sk.pmt_offset).max())
    assert worst < 1e-9, f"worst residual {worst:.3e} m"


def test_sk_pmt_sphere_invariants(sk):
    """The photocathode is a spherical cap; one radius plus one offset fixes it."""
    assert sk.sphere_radius == pytest.approx(SK_SPHERE_R, abs=1e-9)
    assert sk.pmt_offset == pytest.approx(SK_OFFSET, abs=1e-9)
    # protrusion into the water == expose height
    assert sk.sphere_radius - sk.pmt_offset == pytest.approx(SK_EXPOSE, abs=1e-9)
    # the circle where the sphere crosses the wall == the 20-inch aperture
    rim = np.sqrt(sk.sphere_radius ** 2 - sk.pmt_offset ** 2)
    assert rim == pytest.approx(SK_APERTURE, abs=1e-9)
    # the propagator's sensor radius must be the radius of curvature, not the aperture
    assert sk.S_radius == pytest.approx(SK_SPHERE_R, abs=1e-9)
    assert sk.aperture_radius == pytest.approx(SK_APERTURE)


@pytest.mark.parametrize("n_sides", [36, 37, 39, 40, 76])
def test_wrong_side_count_is_rejected(n_sides):
    """A neighbouring N leaves centimetres of residual — the check must catch it."""
    with pytest.raises(ValueError, match="not on the declared"):
        PolygonalCylinder.from_pmt_file(
            os.path.join(REPO, 'config', 'sk_geometry.npz'),
            n_sides=n_sides, apothem=SK_APOTHEM, height=SK_HEIGHT,
            expose_height=SK_EXPOSE)


def test_sk_coverage_matches_the_published_40_percent(sk):
    """Independent sanity check on the aperture radius via photocathode coverage."""
    perimeter = sk.n_sides * sk.panel_width
    area = perimeter * sk.H + perimeter * sk.apothem       # barrel + two caps
    cov = len(sk.all_points) * np.pi * sk.aperture_radius ** 2 / area
    assert 0.39 < cov < 0.41, f"coverage {cov:.4f}"


# ── 2. the ray tracer ─────────────────────────────────────────────────


def _reference_exit(o, d, n_sides, apothem, h):
    """Brute force: exit = min over every half-space the ray is approaching."""
    dphi = 2 * np.pi / n_sides
    ang = (np.arange(n_sides) + 0.5) * dphi
    nrm = np.zeros((n_sides + 2, 3))
    nrm[:n_sides, 0], nrm[:n_sides, 1] = np.cos(ang), np.sin(ang)
    nrm[n_sides], nrm[n_sides + 1] = [0, 0, 1], [0, 0, -1]
    off = np.concatenate([np.full(n_sides, apothem), [h / 2, h / 2]])
    den = d @ nrm.T
    with np.errstate(divide='ignore', invalid='ignore'):
        tk = np.where(den > 1e-12, (off[None, :] - o @ nrm.T) / den, np.inf)
    return tk.min(1), tk.argmin(1)


def _sample_inside(n, n_sides, apothem, h, seed=3):
    rng = np.random.default_rng(seed)
    circum = apothem / np.cos(np.pi / n_sides)
    out = np.empty((0, 3))
    while len(out) < n:
        c = rng.uniform(-1, 1, (n, 3)) * np.array([circum, circum, h / 2])
        out = np.concatenate([out, c[np.asarray(polygon_bounds_check(
            jnp.asarray(c), apothem, h, n_sides))]])
    return out[:n]


def test_intersection_picks_the_same_exit_face_as_brute_force(toy):
    n, a, h = toy.n_sides, toy.apothem, toy.H
    o = _sample_inside(20000, n, a, h)
    rng = np.random.default_rng(4)
    d = rng.normal(size=o.shape); d /= np.linalg.norm(d, axis=1, keepdims=True)

    intersects, t, is_wall, is_top, _, _, _, panel = batch_intersect_polygon_with_grid(
        jnp.asarray(o), jnp.asarray(d), a, h, n, toy._n_u, toy._n_height, toy._n_cap)
    t_ref, face_ref = _reference_exit(o, d, n, a, h)

    assert bool(np.all(np.asarray(intersects)))
    face = np.where(np.asarray(is_wall), np.asarray(panel),
                    np.where(np.asarray(is_top), n, n + 1))
    assert np.array_equal(face, face_ref)
    # float32 arithmetic on ~10 m paths; the median is at the eps floor
    assert np.median(np.abs(np.asarray(t) - t_ref)) < 1e-5


def test_exit_point_straddles_the_boundary(toy):
    n, a, h = toy.n_sides, toy.apothem, toy.H
    o = _sample_inside(20000, n, a, h)
    rng = np.random.default_rng(5)
    d = rng.normal(size=o.shape); d /= np.linalg.norm(d, axis=1, keepdims=True)
    t_ref, _ = _reference_exit(o, d, n, a, h)
    eps = 1e-3
    inside_before = np.asarray(polygon_bounds_check(
        jnp.asarray(o + (t_ref - eps)[:, None] * d), a, h, n))
    inside_after = np.asarray(polygon_bounds_check(
        jnp.asarray(o + (t_ref + eps)[:, None] * d), a, h, n))
    assert inside_before.all()
    assert not inside_after.any()


def test_normals_are_unit_outward_and_face_aligned(toy):
    n, a, h = toy.n_sides, toy.apothem, toy.H
    o = _sample_inside(5000, n, a, h)
    rng = np.random.default_rng(6)
    d = rng.normal(size=o.shape); d /= np.linalg.norm(d, axis=1, keepdims=True)
    _, t, is_wall, is_top, _, _, p, panel = batch_intersect_polygon_with_grid(
        jnp.asarray(o), jnp.asarray(d), a, h, n, toy._n_u, toy._n_height, toy._n_cap)
    nn = np.asarray(calculate_polygon_normals(is_wall, is_top, panel, n))

    assert np.allclose(np.linalg.norm(nn, axis=1), 1.0, atol=1e-5)
    # every exit normal has a positive component along the ray (the ray is leaving)
    assert (np.einsum('ij,ij->i', nn, d) > 0).all()
    # wall normals are the panel normals, exactly
    wall = np.asarray(is_wall)
    ang = (np.asarray(panel)[wall] + 0.5) * (2 * np.pi / n)
    assert np.allclose(nn[wall, 0], np.cos(ang), atol=1e-5)
    assert np.allclose(nn[wall, 1], np.sin(ang), atol=1e-5)
    assert np.allclose(nn[wall, 2], 0.0, atol=1e-6)


def test_bounds_check_matches_all_half_spaces(toy):
    n, a, h = toy.n_sides, toy.apothem, toy.H
    rng = np.random.default_rng(8)
    circum = a / np.cos(np.pi / n)
    q = rng.uniform(-1.2, 1.2, (50000, 3)) * np.array([circum, circum, h / 2])
    ang = (np.arange(n) + 0.5) * (2 * np.pi / n)
    ref = ((q[:, :1] * np.cos(ang)[None, :] + q[:, 1:2] * np.sin(ang)[None, :]).max(1) <= a) \
        & (np.abs(q[:, 2]) <= h / 2)
    got = np.asarray(polygon_bounds_check(jnp.asarray(q), a, h, n))
    assert np.array_equal(got, ref)


# ── 3. the grid ───────────────────────────────────────────────────────


def test_grid_columns_never_straddle_a_panel_edge(sk):
    sk.configure_grid()
    assert sk._n_angular % sk.n_sides == 0
    assert sk._n_angular == sk.n_sides * sk._n_u


def test_grid_cell_centres_lie_on_the_surface(toy):
    centers = np.asarray(toy.grid_cell_centers())
    n_wall = toy._n_angular * toy._n_height
    wall, caps = centers[:n_wall], centers[n_wall:]
    dphi = 2 * np.pi / toy.n_sides
    phi = np.arctan2(wall[:, 1], wall[:, 0]) % (2 * np.pi)
    ang = (np.floor(phi / dphi) + 0.5) * dphi
    perp = wall[:, 0] * np.cos(ang) + wall[:, 1] * np.sin(ang)
    assert np.allclose(perp, toy.apothem, atol=1e-5)
    assert np.allclose(np.abs(caps[:, 2]), toy.H / 2, atol=1e-6)


def test_every_sensor_reaches_the_grid(sk):
    """Sensor centres sit outside the wall by design; the assignment must still find
    them, or those PMTs become invisible to every ray."""
    sk.configure_grid()
    assignments = sk.assign_sensor_to_cells(jnp.asarray(sk.all_points),
                                            float(sk.S_radius))
    missing = int(np.sum(~np.any(np.asarray(assignments) != -1, axis=(1, 2))))
    assert missing == 0, f"{missing} sensors have no grid cell"
