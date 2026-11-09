"""Rays aimed at known sensors, through the full surface propagator and its grid.

Every ray here is built so that its answer is known before the propagator runs. The truths are
independent of the code under test:

* **Geometry, in float64.** For a ray o + t*d (|d| = 1) and a sensor sphere of radius r_s at c,
  the closest approach is t_c = (c - o).d at distance rho = |(c - o) - t_c d|. When rho < r_s
  the ray enters the sphere at t_e = t_c - sqrt(r_s^2 - rho^2), at the point E = o + t_e d.
* **The overlap kernel in closed form.** The propagator weights a candidate by the fraction of
  a 2-D Gaussian of width sigma = temperature * r_s, centred a distance rho from the sensor
  centre, that falls on the sensor's disk of radius r_s. That is P(|X|^2 <= r_s^2) for
  X ~ N((rho, 0), sigma^2 I), and |X|^2 / sigma^2 is noncentral chi-square with 2 degrees of
  freedom and noncentrality (rho / sigma)^2:

      f(rho) = ncx2.cdf((r_s / sigma)^2, df=2, nc=(rho / sigma)^2)  =  1 - Q_1(rho/sigma, r_s/sigma)

  The distance is to the leg the photon travels, which ends at the wall (t_geometry): where the
  closest approach t_c lies beyond it, rho is replaced by sqrt(rho^2 + (t_c - t_geometry)^2).
  About half the rays aimed at a wall sensor from inside are such rays.

  (Q_1 the Marcum Q-function; M. K. Simon, *Probability Distributions Involving Gaussian
  Random Variables*, Springer 2002, ch. 2). The library integrates the kernel numerically into a
  lookup table, so this is a different route to the same number.

What is checked, ray by ray:

1. **The target is a candidate.** The grid lookup must list the sensor a ray is aimed at. It is
   the lookup's job, and a wrong cell index on either side of it (ray side in
   ``intersect_*_with_grid``, sensor side in ``assign_sensors_to_grid``) would drop it. The
   distance fallback in the inverse map can rescue such a bug, so the second test below aims at
   every *cell* a sensor's disk reaches, where the fallback has less room to hide it.
2. **Its weight is f(rho)**, for rays that pass no other sensor closely enough to share the
   deposit (``first_hit_survival`` would then split it).
3. **It is entered exactly when the float64 geometry says so**: rho < r_s, t_e > 0 and E inside
   the detector volume. Where it is entered the stop time is t_e, the stop point is E and the
   normal is (c - E) / r_s.

The default lane runs one small detector (WCTE_like, 2444 PMTs). The sweep over larger
cylinders, the paper grid, a sphere with polar PMTs and two boxes is ``slow``.
"""
import os

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.spatial import cKDTree
from scipy.stats import ncx2

from lucid.geometry import generate_detector, Cylinder, Sphere, Box
from lucid.propagation.shared import create_propagator
from tests.helpers.paths import REPO


# The explicit grid the published pipeline passes to every cylinder
# (analysis/paper/utils/studies.py PIPELINE['grid']; also scripts/benchmark_forward.py,
# examples/seed_reconstruct.py), without its max_candidates_per_ray=8.
PAPER_GRID = dict(n_cap=80, n_angular=120, n_height=80)

BATCH = 4096          # one compiled shape per propagator: the last batch is padded


def _aim_id(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return 'paper' if v else 'default'
    return None


def _config(name):
    return os.path.join(REPO, 'config', f'{name}_geom_config.json')


def overlap_exact(rho, r_s, sigma):
    """Disk-under-offset-Gaussian overlap, closed form (see module docstring)."""
    if sigma is None:
        return (np.asarray(rho) < r_s).astype(float)
    return ncx2.cdf((r_s / sigma) ** 2, 2, (np.asarray(rho) / sigma) ** 2)


def run_batched(prop, origins, directions):
    """Run the propagator over fixed-size batches; return per-slot numpy arrays, (C, N)."""
    n = len(origins)
    outs = {k: [] for k in ('sensor_indices', 'sensor_weights', 'inside_sensor', 'times',
                            'positions', 'normals')}
    for s in range(0, n, BATCH):
        o = origins[s:s + BATCH]
        d = directions[s:s + BATCH]
        m = len(o)
        if m < BATCH:                      # pad with copies of the first ray, dropped below
            o = np.concatenate([o, np.repeat(o[:1], BATCH - m, axis=0)])
            d = np.concatenate([d, np.repeat(d[:1], BATCH - m, axis=0)])
        r = prop(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32))
        for k in outs:
            a = np.asarray(r[k])
            if k == 'times' and a.ndim == 3:
                a = a[..., 0]
            outs[k].append(a[:m] if k in ('positions', 'normals') else a[:, :m])
    return {k: np.concatenate(v, axis=0 if k in ('positions', 'normals') else 1)
            for k, v in outs.items()}


def inside_margin(det, p):
    """Signed distance of points p into the detector volume (positive inside), float64.

    Written from the shape's defining inequalities, not from the library's ``contains``.
    """
    if isinstance(det, Cylinder):
        return np.minimum(det.r - np.hypot(p[:, 0], p[:, 1]), det.H / 2 - np.abs(p[:, 2]))
    if isinstance(det, Sphere):
        return det.r - np.linalg.norm(p, axis=1)
    if isinstance(det, Box):
        half = np.array([det.L / 2, det.W / 2, det.H / 2])
        return np.min(half[None, :] - np.abs(p), axis=1)
    raise TypeError(type(det))


def aim_rays(S, r_s, per_sensor, rng, rho_frac=0.8, rho_min_frac=0.0):
    """`per_sensor` rays at every sensor, from 0.3*c (inside every shipped volume), each aimed at
    a point a uniform-in-disk distance <= rho_frac*r_s from the centre, in the plane normal to the
    origin-to-centre axis."""
    k = np.repeat(np.arange(len(S)), per_sensor)
    c = S[k]
    o = 0.3 * c
    ax = c - o
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    u = np.cross(ax, rng.normal(size=ax.shape))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rho = r_s * np.sqrt(rng.uniform(rho_min_frac ** 2, rho_frac ** 2, size=len(k)))
    tgt = c + rho[:, None] * u
    d = tgt - o
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d, k


def check_aimed_rays(det, prop, grid, sigma, per_sensor=4, seed=0, rho_frac=0.8, rho_min_frac=0.0):
    """Run every check in the module docstring; return the counts for the caller to assert on."""
    S = np.asarray(det.all_points, np.float64)
    r_s = float(det.S_radius)
    o, d, k = aim_rays(S, r_s, per_sensor, np.random.default_rng(seed), rho_frac, rho_min_frac)
    out = run_batched(prop, o, d)
    idx = out['sensor_indices']
    is_tgt = idx == k[None, :]

    # float64 geometry for the target
    oc = S[k] - o
    t_c = np.sum(oc * d, axis=1)
    rho = np.linalg.norm(oc - t_c[:, None] * d, axis=1)
    t_e = t_c - np.sqrt(np.maximum(r_s ** 2 - rho ** 2, 0.0))
    E = o + t_e[:, None] * d
    margin = inside_margin(det, E)
    enters = (rho < r_s) & (t_e > 0) & (margin >= 0)
    # Float32 cannot decide a point within ~1e-5 m of the volume boundary; leave those out.
    decidable = np.abs(margin) > 1e-5

    # The nearest approach of the ray line to any OTHER sensor ahead of the origin. Neighbours
    # are searched within 12 r_s of the target, past which no shipped layout brings a sensor
    # back near a ray that is leaving the volume.
    tree = cKDTree(S)
    d_other = np.full(len(k), np.inf)
    for i, nb in enumerate(tree.query_ball_point(S[k], 12 * r_s)):
        nb = [j for j in nb if j != k[i]]
        if not nb:
            continue
        v = S[nb] - o[i]
        tj = v @ d[i]
        dj = np.linalg.norm(v - tj[:, None] * d[i], axis=1)
        if np.any(tj > 0):
            d_other[i] = dj[tj > 0].min()
    # Weight-isolated: no other sensor gets f > 1e-6 (beyond r_s + 5 sigma), so the first-hit
    # product cannot share the target's deposit. Stop-isolated: no other sphere is entered, so
    # the stop point must be on the target.
    iso_w = d_other > r_s + (5 * sigma if sigma is not None else 0.0)
    iso_stop = d_other > r_s

    listed = is_tgt.any(axis=0)
    w_tgt = np.sum(np.where(is_tgt, out['sensor_weights'], 0.0), axis=0)
    in_tgt = np.any(is_tgt & out['inside_sensor'], axis=0)
    slot = np.argmax(is_tgt, axis=0)
    t_tgt = out['times'][slot, np.arange(len(k))]
    # the leg ends where the ray leaves the detector (the propagator's own intersect_ray)
    t_geo = np.asarray(det.intersect_ray(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32),
                                         grid)[1], np.float64).reshape(-1)
    f = overlap_exact(np.hypot(rho, np.maximum(t_c - t_geo, 0.0)), r_s, sigma)

    # The entry flag and the per-candidate time do not depend on other sensors.
    geo = decidable & listed
    ent = geo & enters
    stop = ent & iso_stop

    def _max(x, m):
        return float(np.max(x[m])) if m.any() else 0.0

    return dict(
        n=len(k), n_iso_w=int(iso_w.sum()), n_stop=int(stop.sum()), n_geo=int(geo.sum()),
        not_listed=int((~listed).sum()),
        weight_err=_max(np.abs(w_tgt - f), iso_w & listed),
        # the first-hit product only takes weight away, so even a shared deposit is <= f
        weight_over=float(np.max(w_tgt - f)),
        enter_mismatch=int(np.sum(geo & (in_tgt != enters))),
        n_entered=int(ent.sum()),
        # t_e where the sphere is entered, t_c (closest approach) where it is not
        time_err=_max(np.abs(t_tgt - np.where(enters, t_e, t_c)), geo),
        pos_err=_max(np.linalg.norm(out['positions'] - E, axis=1), stop),
        normal_err=_max(np.linalg.norm(out['normals'] - (S[k] - E) / r_s, axis=1), stop),
    )


def assert_aimed(res, label, weight_tol, min_iso_w=0.9):
    """The assertions shared by the default-lane and slow cases.

    ``min_iso_w`` is the fraction of rays that must be weight-isolated for the exact-weight check
    to mean something; a layout packed at ~2 r_s (TAO) has none, and keeps only the bound.
    """
    assert res['not_listed'] == 0, (
        f"{label}: {res['not_listed']}/{res['n']} rays aimed within 0.8 r_s of a sensor do not "
        f"list it as a candidate")
    assert res['n_iso_w'] >= min_iso_w * res['n'], res
    assert res['weight_err'] < weight_tol, (
        f"{label}: target weight differs from the closed-form overlap by {res['weight_err']:.2e}")
    assert res['weight_over'] < weight_tol, res
    assert res['n_geo'] > 0.95 * res['n'], res
    assert res['enter_mismatch'] == 0, (
        f"{label}: {res['enter_mismatch']} rays disagree with float64 geometry on whether the "
        f"target sphere is entered inside the volume")
    assert res['n_entered'] > 0.5 * res['n'], res
    # t ~ metres in float32: ~1e-6 relative; 1e-4 m is the photon_step nudge.
    assert res['time_err'] < 1e-4, res
    assert res['n_stop'] > 0.5 * res['n'], res
    assert res['pos_err'] < 1e-4, res
    assert res['normal_err'] < 1e-3, res


# LUT interpolation error against the closed form is <= 2e-5 at temperature 0.2 and 0.1
# (measured: tests/string/test_overlap_fix.py); float32 rho adds < 1e-5 at these scales.
# 5e-4 leaves a 25x margin and still fails a misplaced sigma by 10%.
WEIGHT_TOL = 5e-4


@pytest.fixture(scope="module")
def wcte_like():
    det = generate_detector(_config('WCTE_like'))
    prop, grid = create_propagator(det, jnp.asarray(det.all_points, jnp.float32),
                                   float(det.S_radius), temperature=0.2)
    return det, prop, grid


def test_aimed_rays_hit_their_sensor_wcte_like(wcte_like):
    det, prop, grid = wcte_like
    res = check_aimed_rays(det, prop, grid, 0.2 * float(det.S_radius), per_sensor=4)
    assert_aimed(res, 'WCTE_like', WEIGHT_TOL)


def test_near_misses_are_not_entered_wcte_like(wcte_like):
    """Rays passing 1.1-1.8 r_s from a sensor centre do not enter it. Wherever the sensor is still
    a candidate it carries the overlap tail f(rho) and no entry flag, and its time is the
    closest-approach parameter."""
    det, prop, grid = wcte_like
    res = check_aimed_rays(det, prop, grid, 0.2 * float(det.S_radius), per_sensor=2,
                           rho_frac=1.8, rho_min_frac=1.1)
    assert res['n_entered'] == 0
    assert res['n_geo'] > 0.5 * res['n'], res          # most near misses are still candidates
    assert res['enter_mismatch'] == 0, res
    assert res['weight_err'] < WEIGHT_TOL, res
    assert res['time_err'] < 1e-4, res


# ── Every cell a sensor's disk reaches (cylinder) ─────────────────────────────────────────────


def cell_probe_rays(det, grid, frac=0.9, eps=1e-4):
    """Normal-incidence rays landing inside every grid cell that comes within frac*r_s of a sensor.

    For each sensor and each of the 5x5 cells around its own cell, the landing point is the point
    of that cell nearest the sensor (pulled eps inside the cell). A barrel point is reached by a
    radial ray from the axis at its height, a cap point by an axial ray from z = 0. A sensor
    whose line distance to that ray is under frac*r_s must be a candidate there.

    The cell boundaries come from the propagator's grid record (``n_angular`` equal azimuth bins from
    phi = 0, ``n_height`` equal z bins from -H/2, ``n_cap`` x ``n_cap`` equal squares over
    [-r, r]^2), not from ``intersect_cylinder_with_grid``.
    """
    S = np.asarray(det.all_points, np.float64)
    r_s = float(det.S_radius)
    r, H = det.r, det.H
    na, nh, nc = grid.n_angular, grid.n_height, grid.n_cap
    da, dz, dc = 2 * np.pi / na, H / nh, 2 * r / nc
    O, D, K = [], [], []
    for kk, s in enumerate(S):
        if det.ID_to_case[kk] == 0:
            phi = np.arctan2(s[1], s[0]) % (2 * np.pi)
            ia, iz = int(phi // da), int((s[2] + H / 2) // dz)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    j = iz + dj
                    if not 0 <= j < nh:
                        continue
                    pp = np.clip(phi, (ia + di) * da + eps / r, (ia + di + 1) * da - eps / r)
                    pz = np.clip(s[2], -H / 2 + j * dz + eps, -H / 2 + (j + 1) * dz - eps)
                    o = np.array([0.0, 0.0, pz])
                    dvec = np.array([np.cos(pp), np.sin(pp), 0.0])
                    v = s - o
                    if np.linalg.norm(v - (v @ dvec) * dvec) < frac * r_s and v @ dvec > 0:
                        O.append(o); D.append(dvec); K.append(kk)
        else:
            up = 1.0 if det.ID_to_case[kk] == 1 else -1.0
            ix, iy = int((s[0] + r) // dc), int((s[1] + r) // dc)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    a, b = ix + di, iy + dj
                    if not (0 <= a < nc and 0 <= b < nc):
                        continue
                    px = np.clip(s[0], -r + a * dc + eps, -r + (a + 1) * dc - eps)
                    py = np.clip(s[1], -r + b * dc + eps, -r + (b + 1) * dc - eps)
                    if px * px + py * py > (r - eps) ** 2:
                        continue
                    if np.hypot(s[0] - px, s[1] - py) < frac * r_s:
                        O.append([px, py, 0.0]); D.append([0.0, 0.0, up]); K.append(kk)
    return np.array(O), np.array(D), np.array(K)


def count_unlisted_in_reached_cells(det, prop, grid):
    o, d, k = cell_probe_rays(det, grid)
    out = run_batched(prop, o, d)
    listed = np.any(out['sensor_indices'] == k[None, :], axis=0)
    return len(k), int((~listed).sum()), len(np.unique(k[~listed]))


def test_every_cell_a_sensor_reaches_lists_it_wcte_like(wcte_like):
    """``assign_sensors_to_grid`` records at most 4 of the up to 9 cells a sensor overlaps
    (WCTE_like at the default grid: 304 sensors want more). Whatever it drops must come back
    through the distance fallback, or rays landing there miss the sensor. This pins that rescue,
    and with it the agreement of the ray-side and sensor-side cell indices."""
    det, prop, grid = wcte_like
    n, n_bad, n_sens = count_unlisted_in_reached_cells(det, prop, grid)
    assert n > 3 * len(det.all_points), n           # several cells per sensor were probed
    assert n_bad == 0, (f"WCTE_like: {n_bad}/{n} rays landing within 0.9 r_s of a sensor "
                        f"({n_sens} sensors) do not list it")


# ── Slow sweep: larger cylinders, the paper grid, sphere, box ─────────────────────────────────

_SWEEP_CYL = [
    ('WCTE', {}), ('IWCD', {}), ('IWCD', PAPER_GRID), ('SK_like', {}), ('SK_like', PAPER_GRID),
    ('EOS', {}), ('BigHK', {}),
    ('BigHK', PAPER_GRID),             # the paper grid needs 6 slots here; the propagator raises K
]


@pytest.mark.slow
@pytest.mark.parametrize("name,grid", _SWEEP_CYL, ids=_aim_id)
def test_every_cell_a_sensor_reaches_lists_it(name, grid):
    det = generate_detector(_config(name))
    prop, grid_record = create_propagator(det, jnp.asarray(det.all_points, jnp.float32),
                                          float(det.S_radius), temperature=None, **grid)
    n, n_bad, n_sens = count_unlisted_in_reached_cells(det, prop, grid_record)
    assert n_bad == 0, (f"{name} {grid or 'default grid'}: {n_bad}/{n} rays landing within "
                        f"0.9 r_s of a sensor ({n_sens} sensors) do not list it")


_BOX_GRAZING = pytest.mark.xfail(strict=True, reason=(
    "Grazing rays on long boxes: a ray aimed within 0.8 r_s of a PMT centre at grazing incidence "
    "passes through the PMT sphere ~3 r_s before it meets the wall, but candidates are looked up "
    "only in the cells around the wall hit (lucid/propagation/shared.py create_propagator), so "
    "the PMT is not a candidate (nuSCOPE 90/9856 rays, MidBox 3/17864 after F-036/F-037; the "
    "+u/+v-only box assignment that caused ~4% is fixed). A modelling limit, see LOOK_INTO 2.21."))

_SWEEP_AIM = [
    # name, grid, temperature, min weight-isolated fraction
    ('SK_like', {}, 0.1, 0.9), ('SK_like', PAPER_GRID, 0.1, 0.9),
    ('IWCD', {}, 0.2, 0.25),   # spacing ~2.6 r_s: a third of the rays are isolated
    ('JUNO', {}, 0.2, 0.4),    # sphere, PMT spacing ~2.5 r_s: about half the rays are isolated
    ('TAO', {}, 0.2, 0.0),     # sphere packed at ~2 r_s: no ray is isolated; the bound still holds
    pytest.param('MidBox', {}, 0.2, 0.0, marks=_BOX_GRAZING),
    pytest.param('nuSCOPE', {}, 0.2, 0.0, marks=_BOX_GRAZING),
]


@pytest.mark.slow
@pytest.mark.parametrize("name,grid,temperature,min_iso_w", _SWEEP_AIM, ids=_aim_id)
def test_aimed_rays_hit_their_sensor(name, grid, temperature, min_iso_w):
    det = generate_detector(_config(name))
    prop, grid_record = create_propagator(det, jnp.asarray(det.all_points, jnp.float32),
                                          float(det.S_radius), temperature=temperature, **grid)
    res = check_aimed_rays(det, prop, grid_record, temperature * float(det.S_radius), per_sensor=2)
    assert_aimed(res, f"{name} {grid or 'default grid'}", WEIGHT_TOL, min_iso_w=min_iso_w)
