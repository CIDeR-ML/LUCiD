"""Placing a simulated event in a detector: the one convention every caller uses.

PhotonSim records an event in its own frame: the primary starts at a recorded vertex ``v0``
along a recorded direction ``d0`` (:func:`lucid.sources.root_reader.read_primary`). Placing it
is a rigid motion about that vertex (:func:`rigid_motion`),

    x -> R (x - v0) + v,        d -> R d,

with ``R`` uniform over SO(3) and ``v`` uniform in a region the caller chooses (a
:mod:`lucid.geometry.volume` value). The placed primary starts at ``v`` along ``R d0``, so the
vertex is uniform in the region and the direction isotropic, whatever the event's extent.
Distances from the vertex and all relative angles are kept, so emission times stay valid.

The caller owns the random streams (one for the rotation, one for the vertex).

The production datasets draw their vertices from a fiducial fraction of the detector's bounds
instead (:func:`detector_bounds`, :func:`sample_vertex`).
"""
import numpy as np


def random_rotation(rng):
    """Uniform random proper rotation (Shoemake's quaternion method).

    Uniform over SO(3) -- unlike independent Euler angles, which bunch near the
    poles. Returns a (3, 3) float64 matrix with det = +1.
    """
    u1, u2, u3 = rng.random(3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    t1, t2 = 2.0 * np.pi * u2, 2.0 * np.pi * u3
    w, x, y, z = s2 * np.cos(t2), s1 * np.sin(t1), s1 * np.cos(t1), s2 * np.sin(t2)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def draw_placement(direction, *, region, rotation_rng, vertex_rng, accept, max_tries):
    """Draw a placement ``(rotation, vertex, placed_direction)`` for a primary along ``direction``.

    ``vertex`` (m) is uniform in ``region``; ``accept(vertex, placed_direction) -> bool`` is the
    caller's selection (e.g. a contained track end point). Rejected draws are redrawn from the
    same streams; after ``max_tries`` rejections this raises.
    """
    direction = np.asarray(direction, dtype=np.float64)
    for _ in range(max_tries):
        rotation = random_rotation(rotation_rng)
        vertex = region.from_unit(vertex_rng.uniform(0.0, 1.0, 3))
        placed = rotation @ direction
        if accept(vertex, placed):
            return rotation, vertex, placed
    raise RuntimeError(f'no accepted placement in {max_tries} draws in {region}')


def rigid_motion(points, directions, *, pivot, rotation, vertex):
    """``points -> R (points - pivot) + vertex`` and ``directions -> R directions``: ``pivot`` is
    turned by ``R`` and lands on ``vertex``.

    ``points``, ``pivot`` and ``vertex`` share one unit; ``(N, 3)`` arrays in, computed in the
    inputs' common dtype (the caller picks the precision).
    """
    R = np.asarray(rotation)
    return ((np.asarray(points) - np.asarray(pivot)) @ R.T + np.asarray(vertex),
            np.asarray(directions) @ R.T)


def detector_bounds(det_geom):
    """The ``detector_bounds`` dict of a ``DetectorGeometry``: its shape and dimensions (m).

    Reads ``det_geom.detector_type`` and the per-shape attributes of ``det_geom.detector``
    (``Cylinder.r/H``, ``Sphere.r``, ``Box.L/W/H``). Returns ``None`` for shapes that do not
    enclose a canonical volume (e.g. 'string' arrays).
    """
    if det_geom is None or det_geom.detector is None:
        return None
    dt = str(det_geom.detector_type).lower()
    d = det_geom.detector
    if dt == 'cylinder':
        return {'type': 'cylinder', 'radius': float(d.r), 'height': float(d.H)}
    if dt == 'sphere':
        return {'type': 'sphere', 'radius': float(d.r)}
    if dt == 'box':
        return {'type': 'box',
                'length': float(d.L),
                'width':  float(d.W),
                'height': float(d.H)}
    return None


def sample_vertex(detector_bounds, rng, *, r_frac=0.9, z_frac=0.9,
                              sensor_positions=None, pmt_radius=0.0, max_tries=100):
    """Draw a random vertex inside the fiducial volume of the detector.

    Cylinder: uniform in (r, theta, z) with r <= r_frac*R, |z| <= z_frac*H/2.
    Sphere:   uniform in the r_frac*R ball.
    Box:      uniform in the {r,z}_frac-scaled box.

    Defaults (r_frac = z_frac = 0.9, no PMT rejection) reproduce the standard
    fiducial used by all single-vertex / pile-up configs. The supernova path
    passes r_frac = z_frac = 0.995 and ``sensor_positions`` + ``pmt_radius`` to
    fill nearly the whole inner volume while redrawing any vertex that lands
    inside a PMT (SN-only).

    Returns a length-3 float32 array in meters.
    """
    if detector_bounds is None:
        return np.zeros(3, dtype=np.float32)
    kind = detector_bounds['type']
    tree = None
    if sensor_positions is not None and pmt_radius > 0.0:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(sensor_positions, dtype=np.float64))

    def _draw():
        u = rng.uniform(0, 1, size=3)
        if kind == 'cylinder':
            r = (detector_bounds['radius'] * r_frac) * np.sqrt(u[0])
            theta = 2.0 * np.pi * u[1]
            z = (2.0 * u[2] - 1.0) * detector_bounds['height'] * z_frac / 2.0
            return np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float32)
        if kind == 'sphere':
            r = (detector_bounds['radius'] * r_frac) * (u[0] ** (1.0 / 3.0))
            cos_t = 2.0 * u[1] - 1.0
            phi = 2.0 * np.pi * u[2]
            sin_t = np.sqrt(1.0 - cos_t * cos_t)
            return (r * np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
                    ).astype(np.float32)
        if kind == 'box':
            return np.array([
                (2.0 * u[0] - 1.0) * detector_bounds['length'] * r_frac / 2.0,
                (2.0 * u[1] - 1.0) * detector_bounds['width']  * r_frac / 2.0,
                (2.0 * u[2] - 1.0) * detector_bounds['height'] * z_frac / 2.0,
            ], dtype=np.float32)
        raise ValueError(f"Unknown detector_bounds type: {kind!r}")

    v = _draw()
    if tree is None:
        return v
    for _ in range(max_tries):                       # redraw vertices inside a PMT
        if tree.query(v.astype(np.float64))[0] > pmt_radius:
            return v
        v = _draw()
    return v                                         # give up (vanishingly unlikely)
