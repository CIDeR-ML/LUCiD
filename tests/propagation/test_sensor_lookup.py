"""Surface-propagator defects that the rest of the suite let through, each pinned by an oracle.

Every case here builds its answer in float64 from first principles, not from the library:

* **Ray-sphere geometry.** A ray o + t d (|d| = 1) passes a sphere of radius r_s centred at c
  at distance rho = |(c - o) - t_c d|, t_c = (c - o).d. If rho < r_s it enters at
  t_e = t_c - sqrt(r_s^2 - rho^2), at E = o + t_e d, where the outward normal the propagator
  reports is (c - E) / r_s.
* **The grid's definition.** Cylinder barrel cells are ``n_angular`` equal azimuth bins from
  phi = 0 by ``n_height`` equal z bins from -H/2; sphere cells are ``n`` equal polar bins from
  theta = 0 by ``2n`` equal azimuth bins from phi = 0. A ray from the axis (cylinder) or the
  centre (sphere) lands at a point whose cell follows from its direction alone.

1. **A sensor whose disk crosses a cell boundary is found from the far cell** (cylinder and
   sphere). The grid lists, per cell, the sensors geometry assigns to it, then tops up with the
   sensors nearest the cell centre. The top-up hides a forward assignment that is dropped
   wherever a cell has a free slot (on WCTE_like at its default and the paper grid, zeroing the
   left-neighbour assignment loses none of 19552 rim-aimed rays), so these
   detectors crowd the neighbouring cell with exactly ``max_candidates_per_ray`` = 4 sensors of
   its own: then no slot is left over and only the forward assignment can put the boundary
   sensor there. Sixteen rays aimed at 0.9 r_s around the sensor, some landing across the
   boundary, must all stop on its sphere.
2. **The stop flag is the leg's own segment-sphere test** (``compute_sensor_intersections_base``):
   flagged exactly when the segment [0, t_geometry] enters the sphere.
3. **A ray through two sensor spheres stops on the first** (``process_intersection_normals``).
4. **The validator reports an inverse map that is too small**, with the count brute force gives.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from lucid.geometry import Cylinder, Sphere
from lucid.overlap import create_overlap_prob
from lucid.propagation.base import compute_sensor_intersections_base, find_closest_sensors
from lucid.propagation.shared import create_propagator, validate_sensor_map

N_AIM = 16                  # rays per boundary sensor, evenly spaced around it
AIM_FRAC = 0.9              # aimed at 0.9 r_s from the centre: entered, and near the rim
# float32 stop points on a ~1 m detector are good to ~1e-7 m; 1e-5 m is 100x that and still
# 1/8000 of a sensor radius, so a stop on any other sphere or on the wall fails by >> tol.
POS_TOL = 1e-5


def _stop_check(prop, S, r_s, target, o, d, inside_margin):
    """Run ``prop`` on rays aimed at sensor ``target``; return the float64 entry geometry and the
    propagator's answer. Asserts the rays are well posed: they enter the target, inside the
    volume, and no other sphere."""
    c = S[target]
    t_c = np.sum((c - o) * d, axis=1)
    rho = np.linalg.norm((c - o) - t_c[:, None] * d, axis=1)
    assert np.all(rho < r_s) and np.all(t_c > 0)
    t_e = t_c - np.sqrt(r_s ** 2 - rho ** 2)
    E = o + t_e[:, None] * d
    assert np.all(inside_margin(E) > 1e-3), "entry point not inside the volume"
    # No other sphere is entered (the step kernel gives it no charge either); 10% clear of r_s,
    # far beyond float32's ~1e-7 m.
    for j in range(len(S)):
        if j != target:
            tj = np.sum((S[j] - o) * d, axis=1)
            dj = np.linalg.norm((S[j] - o) - tj[:, None] * d, axis=1)
            assert np.all(dj > 1.1 * r_s), (j, dj.min() / r_s)
    out = prop(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32))
    idx = np.asarray(out['sensor_indices'])
    is_t = idx == target
    return dict(
        E=E, normal=(c - E) / r_s,
        listed=is_t.any(axis=0),
        flagged=np.any(is_t & np.asarray(out['inside_sensor']), axis=0),
        weight=np.sum(np.where(is_t, np.asarray(out['sensor_weights']), 0.0), axis=0),
        pos=np.asarray(out['positions']), nrm=np.asarray(out['normals']),
    )


def _assert_stops_on_target(res, label):
    miss = ~res['listed']
    assert not miss.any(), (f"{label}: {miss.sum()}/{miss.size} rays aimed at 0.9 r_s around the "
                            f"sensor do not list it (the cell they land in lost it)")
    assert res['flagged'].all(), label
    # step kernel: full weight, less first_hit_survival's 1e-6 clip
    np.testing.assert_allclose(res['weight'], 1.0, atol=1e-5, err_msg=label)
    err = np.linalg.norm(res['pos'] - res['E'], axis=1)
    assert err.max() < POS_TOL, f"{label}: stop point off the sensor entry by {err.max():.2e} m"
    nerr = np.linalg.norm(res['nrm'] - res['normal'], axis=1)
    assert nerr.max() < 1e-4, f"{label}: normal off by {nerr.max():.2e}"


# ── 1a. Cylinder barrel: every neighbour direction, and the azimuth wrap ──────────────────────

CYL_R, CYL_H, CYL_RS = 1.0, 2.0, 0.08
CYL_GRID = dict(n_cap=4, n_angular=8, n_height=4)
DA = 2 * np.pi / CYL_GRID['n_angular']        # 45 degrees, 0.785 m of arc
DZ = CYL_H / CYL_GRID['n_height']             # 0.5 m
HALF = 0.5 * CYL_RS / CYL_R                   # half a sensor radius, as an azimuth


def _cyl(phi, z):
    return np.array([CYL_R * np.cos(phi), CYL_R * np.sin(phi), z])


def _cyl_layout(p_phi, p_z, n_ia, n_iz):
    """The boundary sensor P, then 4 crowders well inside cell (n_ia, n_iz): 0.3 cells from its
    centre in azimuth and 0.12 m in z, so each is >= 1.6 r_s from every cell edge and is
    forward-assigned to that cell alone."""
    pc, zc = (n_ia + 0.5) * DA, -CYL_H / 2 + (n_iz + 0.5) * DZ
    crowd = [_cyl(pc + sa * 0.3 * DA, zc + sz * 0.12) for sa in (-1, 1) for sz in (-1, 1)]
    return np.array([_cyl(p_phi, p_z)] + crowd)


# name: (P's azimuth, P's z, the neighbour cell its disk crosses into). P's own cell is
# (2, 2) = 90..135 deg, z 0..0.5 m, except in the two wrap cases.
_CYL_CASES = {
    'left': (2 * DA + HALF, 0.25, (1, 2)),
    'right': (3 * DA - HALF, 0.25, (3, 2)),
    'down': (2.5 * DA, 0.0 + 0.5 * CYL_RS, (2, 1)),
    'up': (2.5 * DA, 0.5 - 0.5 * CYL_RS, (2, 3)),
    'wrap_left': (HALF, 0.25, (7, 2)),                   # cell 0 reaching back into cell 7
    'wrap_right': (2 * np.pi - HALF, 0.25, (0, 2)),      # cell 7 reaching on into cell 0
}


def _cyl_cell(p):
    phi = np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)
    return np.floor(phi / DA).astype(int), np.floor((p[:, 2] + CYL_H / 2) / DZ).astype(int)


@pytest.mark.filterwarnings("ignore:Sensor map")
@pytest.mark.parametrize("case", list(_CYL_CASES))
def test_cylinder_sensor_is_found_from_the_cell_its_disk_crosses_into(case):
    """Kills propagation-17 (``include_left`` zeroed): in the 'left' and 'wrap_left' cases the
    rays that land in the neighbour cell find only its 4 crowders and stop on the wall."""
    p_phi, p_z, ncell = _CYL_CASES[case]
    S = _cyl_layout(p_phi, p_z, *ncell)
    det = Cylinder(CYL_R, CYL_H, 40, CYL_RS)       # supplies the grid; sensors come from S
    prop, _ = create_propagator(det, jnp.asarray(S, jnp.float32), CYL_RS, temperature=None,
                                **CYL_GRID)

    # Rays from the axis at P's height, aimed at 16 points 0.9 r_s from P in the wall's tangent
    # plane. From the axis the landing azimuth is the aim point's azimuth.
    a = np.arange(N_AIM) * 2 * np.pi / N_AIM
    e_phi = np.array([-np.sin(p_phi), np.cos(p_phi), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    tgt = S[0] + AIM_FRAC * CYL_RS * (np.cos(a)[:, None] * e_phi + np.sin(a)[:, None] * e_z)
    o = np.tile([0.0, 0.0, p_z], (N_AIM, 1))
    d = (tgt - o) / np.linalg.norm(tgt - o, axis=1, keepdims=True)
    t_w = CYL_R / np.linalg.norm(d[:, :2], axis=1)
    wall = o + t_w[:, None] * d
    ia, iz = _cyl_cell(wall)
    across = (ia == ncell[0]) & (iz == ncell[1])
    assert across.sum() >= 3, f"{case}: only {across.sum()} rays land across the boundary"

    res = _stop_check(prop, S, CYL_RS, 0, o, d,
                      lambda p: np.minimum(CYL_R - np.hypot(p[:, 0], p[:, 1]),
                                           CYL_H / 2 - np.abs(p[:, 2])))
    _assert_stops_on_target(res, f"cylinder {case}")


# ── 1b. Sphere ─────────────────────────────────────────────────────────────────────────────────

SPH_R, SPH_RS, SPH_NDIV = 1.0, 0.06, 8
DT = np.pi / SPH_NDIV                       # polar cell, 22.5 degrees
DP = 2 * np.pi / (2 * SPH_NDIV)             # azimuth cell, 22.5 degrees


def _sph(theta, phi):
    return SPH_R * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                             np.cos(theta)])


def _sph_layout(p_theta, p_phi, n_it, n_ip):
    """P, then 4 crowders 0.25 cells from the centre of cell (n_it, n_ip) in each angle."""
    tc, pc = (n_it + 0.5) * DT, (n_ip + 0.5) * DP
    crowd = [_sph(tc + st * 0.25 * DT, pc + sp * 0.25 * DP) for st in (-1, 1) for sp in (-1, 1)]
    return np.array([_sph(p_theta, p_phi)] + crowd)


T_EQ = 4.5 * DT                                   # 101.25 deg, the centre of polar cell 4
DPHI_EQ = 0.5 * SPH_RS / (SPH_R * np.sin(T_EQ))   # half a sensor radius of arc, as an azimuth
T_HI = 1.5 * DT                                   # 33.75 deg, the centre of polar cell 1


# name: (P's theta, P's phi, the neighbour cell). Equatorial cases, where arc and angle agree.
_SPH_CASES = {
    'phi_left': (T_EQ, 3 * DP + DPHI_EQ, (4, 2)),
    'phi_right': (T_EQ, 4 * DP - DPHI_EQ, (4, 4)),
    'theta_up': (4 * DT - 0.5 * SPH_RS / SPH_R, 3.5 * DP, (4, 3)),   # cell 3 into cell 4
    'theta_down': (4 * DT + 0.5 * SPH_RS / SPH_R, 3.5 * DP, (3, 3)),
    'phi_wrap': (T_EQ, DPHI_EQ, (4, 15)),
    # 1.3 r_s / R of azimuth from the edge: 0.72 r_s of arc at sin(theta) = 0.556, inside the
    # disk's phi reach r_s / (R sin theta) only (LOOK_INTO 2.24)
    'phi_left_high_latitude': (T_HI, 3 * DP + 1.3 * SPH_RS / SPH_R, (1, 2)),
}


@pytest.mark.filterwarnings("ignore:Sensor map")
@pytest.mark.parametrize("case", list(_SPH_CASES))
def test_sphere_sensor_is_found_from_the_cell_its_disk_crosses_into(case):
    """Kills propagation-22 (angular size quartered): a sensor half a radius from a cell edge is
    no longer assigned across it, and rays landing there find only the crowders."""
    p_theta, p_phi, ncell = _SPH_CASES[case]
    S = _sph_layout(p_theta, p_phi, *ncell)
    det = Sphere(SPH_R, 40, SPH_RS)
    prop, _ = create_propagator(det, jnp.asarray(S, jnp.float32), SPH_RS, temperature=None,
                                n_divisions=SPH_NDIV)

    a = np.arange(N_AIM) * 2 * np.pi / N_AIM
    e_t = np.array([np.cos(p_theta) * np.cos(p_phi), np.cos(p_theta) * np.sin(p_phi),
                    -np.sin(p_theta)])
    e_p = np.array([-np.sin(p_phi), np.cos(p_phi), 0.0])
    tgt = S[0] + AIM_FRAC * SPH_RS * (np.cos(a)[:, None] * e_t + np.sin(a)[:, None] * e_p)
    o = np.zeros((N_AIM, 3))
    d = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    across = (np.floor(theta / DT) == ncell[0]) & (np.floor(phi / DP) == ncell[1])
    assert across.sum() >= 3, f"{case}: only {across.sum()} rays land across the boundary"

    res = _stop_check(prop, S, SPH_RS, 0, o, d, lambda p: SPH_R - np.linalg.norm(p, axis=1))
    _assert_stops_on_target(res, f"sphere {case}")


# ── 2. The stop flag is the leg's segment-sphere test ─────────────────────────────────────────


def test_leg_bound_flags_exactly_the_legs_that_enter_the_sphere():
    """``compute_sensor_intersections_base`` flags a stop on a sensor
    exactly when the travelled segment [0, t_geometry] enters its sphere, and its step-kernel
    charge agrees (the docstring's "charge and geometry agree").

    Float64 oracle: a ray at closest approach rho < r_s enters at t_e = t_c - sqrt(r_s^2 - rho^2);
    the leg enters iff t_e < t_geometry. The volume test is made permissive so that the leg end
    alone decides: in the full propagator a convex volume's own bounds check already rejects an
    entry beyond the leg end, so there this flag cannot be told apart from a looser one (checked
    on 262k WCTE_like and JUNO rays: propagation-05 leaves the output bit-identical). This pins the
    function's contract for any caller whose ``t_geometry`` is shorter than the volume exit.

    Kills propagation-05 (``d_eff < 2 r_s``): a leg ending 0.3 r_s short of a sphere it would
    enter at rho = 0.5 r_s has d_eff = 1.2 r_s and is flagged.
    """
    r_s = 0.1
    rng = np.random.default_rng(5)
    n = 4000
    c = np.zeros(3)
    # rays along +x from x = -1 with impact parameter b in [0, 2 r_s), random azimuth about x
    b = rng.uniform(0.0, 2 * r_s, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    o = np.stack([-np.ones(n), b * np.cos(psi), b * np.sin(psi)], axis=1)
    d = np.tile([1.0, 0.0, 0.0], (n, 1))
    t_geo = rng.uniform(0.7, 1.3, n)            # the leg ends anywhere from before to past it
    t_c = 1.0
    h = np.sqrt(np.maximum(r_s ** 2 - b ** 2, 0.0))
    t_e = t_c - h
    enters = (b < r_s) & (t_e < t_geo)
    # float32 cannot decide within ~1e-6 of either threshold
    decidable = (np.abs(b - r_s) > 1e-5) & ((b > r_s) | (np.abs(t_e - t_geo) > 1e-5))
    # the deterministic cases named above, and the 0.5 r_s / 1.5 r_s pair
    o = np.concatenate([o, [[-1, 0.5 * r_s, 0], [-1, 0.5 * r_s, 0], [-1, 1.5 * r_s, 0],
                            [-1, 0.5 * r_s, 0]]])
    d = np.concatenate([d, np.tile([1.0, 0, 0], (4, 1))])
    h5 = np.sqrt(0.75) * r_s
    t_geo = np.concatenate([t_geo, [t_c - h5 - 0.3 * r_s, t_c - h5 + 0.2 * r_s, 10.0, 10.0]])
    enters = np.concatenate([enters, [False, True, False, True]])
    decidable = np.concatenate([decidable, [True] * 4])

    w, _, _, _, flag, pts = compute_sensor_intersections_base(
        jnp.zeros(len(o), jnp.int32), jnp.asarray(c[None, :], jnp.float32), r_s,
        jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32),
        lambda p: jnp.ones(p.shape[0], bool), create_overlap_prob(None, r_s),
        t_geometry=jnp.asarray(t_geo, jnp.float32))
    flag, w = np.asarray(flag), np.asarray(w)
    assert decidable.sum() > 0.99 * len(o)
    bad = decidable & (flag != enters)
    b_all = np.hypot(o[:, 1], o[:, 2])
    assert not bad.any(), (f"{bad.sum()} legs disagree with the segment-sphere test; first: "
                           f"b = {b_all[bad][0] / r_s:.3f} r_s, t_geometry = {t_geo[bad][0]:.4f}, "
                           f"flagged {flag[bad][0]}")
    np.testing.assert_array_equal(w[decidable] > 0.5, enters[decidable])
    assert enters[decidable].sum() > 300 and (~enters[decidable]).sum() > 300


# ── 3. Two spheres on one ray: the first one entered wins ─────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Sensor map")
def test_ray_through_two_sensors_stops_on_the_first():
    """A chord just inside the barrel wall threads two PMT spheres, A at phi = 0 and B at
    phi = 0.3 rad. Run forwards it must stop at A's entry point, backwards at B's, with that
    sphere's normal. Kills propagation-06 (the LAST sensor entered wins): both directions then
    stop on the far sphere. Running both directions also rules out "lowest slot wins"."""
    S = np.array([_cyl(0.0, 0.0), _cyl(0.3, 0.0), _cyl(np.pi, 0.0), _cyl(np.pi + 0.5, 0.0)])
    det = Cylinder(CYL_R, CYL_H, 40, CYL_RS)
    prop, _ = create_propagator(det, jnp.asarray(S, jnp.float32), CYL_RS, temperature=None,
                                **CYL_GRID)
    z = np.array([-0.02, 0.0, 0.02])
    p1 = np.stack([0.99 * np.cos(-0.2) * np.ones(3), 0.99 * np.sin(-0.2) * np.ones(3), z], 1)
    p2 = np.stack([0.99 * np.cos(0.5) * np.ones(3), 0.99 * np.sin(0.5) * np.ones(3), z], 1)
    o = np.concatenate([p1, p2])
    d = np.concatenate([p2 - p1, p1 - p2])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    first = np.array([0, 0, 0, 1, 1, 1])

    # float64: both spheres are entered, inside the volume, and in the expected order
    t_e = np.empty((2, len(o)))
    for j in (0, 1):
        t_c = np.sum((S[j] - o) * d, axis=1)
        rho = np.linalg.norm((S[j] - o) - t_c[:, None] * d, axis=1)
        assert np.all(rho < 0.9 * CYL_RS)
        t_e[j] = t_c - np.sqrt(CYL_RS ** 2 - rho ** 2)
        E = o + t_e[j][:, None] * d
        assert np.all(np.hypot(E[:, 0], E[:, 1]) < CYL_R - 1e-3)
    assert np.all(np.argmin(t_e, axis=0) == first)
    assert np.all(np.abs(t_e[0] - t_e[1]) > 0.2)
    E = o + t_e[first, np.arange(len(o))][:, None] * d

    out = prop(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32))
    idx, ins = np.asarray(out['sensor_indices']), np.asarray(out['inside_sensor'])
    for j in (0, 1):                               # both are candidates, both spheres entered
        assert np.all(np.any((idx == j) & ins, axis=0)), j
    err = np.linalg.norm(np.asarray(out['positions']) - E, axis=1)
    assert err.max() < POS_TOL, f"stop point {err.max():.3f} m from the first sphere's entry"
    nrm = (S[first] - E) / CYL_RS
    np.testing.assert_allclose(np.asarray(out['normals']), nrm, atol=1e-4)
    # first-hit charge: the first sphere takes it all
    w = np.asarray(out['sensor_weights'])
    w_first = np.sum(np.where(idx == first[None, :], w, 0.0), axis=0)
    np.testing.assert_allclose(w_first, 1.0, atol=1e-5)


# ── 4. The validator reports a too-small inverse map ──────────────────────────────────────────


def test_validator_reports_assignments_missing_from_a_too_small_inverse_map():
    """An inverse map built with 2 slots per cell for a layout whose cells hold up to 5 sensors.
    The map takes forward assignments first, so a cell holding n keeps 2 and loses n - 2; brute
    force over the forward map gives the totals the warning must state. (``create_propagator``
    now sizes the slots to fit, so this path is reachable only for a map built elsewhere, which
    is why it is driven directly.) Kills propagation-09 (the missing-assignment warning
    silenced)."""
    k_small = 2
    S = _cyl_layout(*_CYL_CASES['left'][:2], *_CYL_CASES['left'][2])   # cell (1, 2) holds 5
    det = Cylinder(CYL_R, CYL_H, 40, CYL_RS)
    grid = det.make_grid(**CYL_GRID)
    S32 = jnp.asarray(S, jnp.float32)
    fwd = det.assign_sensor_to_cells(S32, CYL_RS, grid)
    inv = det.build_inverted_sensor_map(
        fwd, find_closest_sensors(det.grid_cell_centers(grid), S32, k_small), k_small, grid)

    # brute force: (sensor, cell) pairs from the forward map, and the per-cell excess
    pairs = [(s, int(det.point_to_grid_cell_from_coords(c, grid)))
             for s, row in enumerate(np.asarray(fwd)) for c in row if not np.all(c == -1)]
    assert len(set(pairs)) == len(pairs)           # no repeated cell, so pairs == slots checked
    counts = np.bincount([cell for _, cell in pairs], minlength=det.total_grid_cells(grid))
    n_missing = int(np.maximum(counts - k_small, 0).sum())
    assert counts.max() == 5 and n_missing == 3

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_sensor_map(fwd, inv, len(S), det, grid, k_small)
    msgs = [str(x.message) for x in w]
    missing = [m for m in msgs if "missing from the inverse map" in m]
    assert len(missing) == 1, msgs
    assert (f"{n_missing}/{len(pairs)} geometric assignments are missing from the inverse map"
            in missing[0]), missing[0]
    assert f"max_candidates_per_ray={k_small} overflow" in missing[0], missing[0]
    crowd = [m for m in msgs if "exceeds max_candidates_per_ray" in m]
    assert len(crowd) == 1 and f"per cell ({counts.max()})" in crowd[0], msgs
