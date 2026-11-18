"""Gradients through the shared propagator, against closed forms.

1. **Sensor weight vs the ray.** A ray whose line passes a sensor centre at distance rho takes
   weight f(rho), the disk-under-Gaussian overlap, which is a noncentral chi-square CDF
   (see tests/propagation/test_propagation_aimed_rays.py):

       f(rho) = F_2(x; lam),   x = (r_s/sigma)^2,  lam = (rho/sigma)^2

   With dF_k(x; lam)/dlam = -(F_k(x; lam) - F_{k+2}(x; lam)) / 2 (differentiate the Poisson
   mixture of central chi-squares term by term), f'(rho) = -(rho/sigma^2) (F_2 - F_4).
   For the ray o + t d, with v = c - o and p = v - (v.d) d (|p| = rho),

       d rho / d o = -p / rho,      d rho / d d = -(v.d) p / rho.

2. **Wall and cap hit points vs the origin.** A ray leaving the axis radially along u at
   height z hits the barrel at P = (r u, z); moving the origin by do moves P by
   (I - u u^T) do in xy and by do_z in z, and turns the wall normal by (I - u u^T) do_xy / r.
   An axial ray from (x0, y0, 0) hits the top cap at (x0, y0, H/2).

3. **Charge vs QE.** Charge is linear in qe, so d(sum q)/d qe = sum(q) / qe exactly; and the
   scatter-length derivative through the whole chain equals its central difference.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree
from scipy.stats import ncx2

from lucid.propagation.shared import create_propagator
from lucid.geometry import generate_detector
from lucid.simulation.photon_step import make_photon_iteration_update_factors_safe
from lucid.simulation.reflection import ScalarReflection
from lucid.simulation.hits import first_arrival_expected
from tests.helpers.params import on_deposits
from tests.helpers.paths import REPO

pytestmark = pytest.mark.slow

TEMPERATURE = 0.2

_CACHE = {}


def _built(name):
    if name not in _CACHE:
        det = generate_detector(os.path.join(REPO, 'config', f'{name}_geom_config.json'))
        prop, grid = create_propagator(det, jnp.asarray(det.all_points, jnp.float32),
                                       float(det.S_radius), temperature=TEMPERATURE)
        _CACHE[name] = (det, prop, grid)
    return _CACHE[name]


def _unit(v):
    return v / np.linalg.norm(v)


def _overlap_slope(rho, r_s, sigma):
    x, lam = (r_s / sigma) ** 2, (rho / sigma) ** 2
    return -(rho / sigma ** 2) * (ncx2.cdf(x, 2, lam) - ncx2.cdf(x, 4, lam))


def _aim_isolated(det, prop, grid, rho_frac=0.95, n_try=200):
    """A ray from 0.3 c passing sensor c at rho = rho_frac * r_s, offset in the direction that
    keeps every other sensor beyond r_s + 5 sigma of the line (so no other candidate shares the
    deposit), and listing c as a candidate. Returns (k, o, d, slot).

    Its closest approach to c lies on the leg it travels (before the wall), so the leg bound is
    inactive and the weight is f(rho) of the line distance; tests/propagation/test_deposit_leg_bound.py
    covers the bound."""
    S = np.asarray(det.all_points, np.float64)
    r_s = float(det.S_radius)
    reach = r_s * (1 + 5 * TEMPERATURE)
    tree = cKDTree(S)
    for k in np.random.default_rng(1).permutation(len(S))[:n_try]:
        c = S[k]
        o = 0.3 * c
        ax = _unit(c - o)
        e1 = _unit(np.cross(ax, [0.3, 0.5, 0.8]))
        e2 = np.cross(ax, e1)
        nb = [j for j in tree.query_ball_point(c, 12 * r_s) if j != k]
        best = None
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            d = _unit(c + rho_frac * r_s * (np.cos(a) * e1 + np.sin(a) * e2) - o)
            v = S[nb] - o
            t = v @ d
            dist = np.linalg.norm(v - t[:, None] * d, axis=1)[t > 0]
            m = dist.min() if dist.size else np.inf
            if best is None or m > best[0]:
                best = (m, d)
        if best[0] <= reach:
            continue
        d = best[1]
        t_geo = float(np.asarray(det.intersect_ray(jnp.asarray(o[None], jnp.float32),
                                                   jnp.asarray(d[None], jnp.float32), grid)[1]).reshape(-1)[0])
        if (c - o) @ d >= t_geo:
            continue
        out = prop(jnp.asarray(o[None], jnp.float32), jnp.asarray(d[None], jnp.float32))
        hit = np.asarray(out['sensor_indices'])[:, 0] == k
        if hit.any():
            return k, o, d, int(np.argmax(hit))
    pytest.fail(f"no isolated, listed target found in {n_try} tries")


@pytest.mark.parametrize("name", ['WCTE_like', 'JUNO', 'nuSCOPE'])
def test_weight_gradient_matches_the_closed_form(name):
    """d w / d origin and d w / d direction for an edge ray (rho = 0.95 r_s, where the slope
    is steepest), against f'(rho) times the geometric derivatives of rho.

    The lookup table is piecewise linear in d, so autodiff reads a segment's slope; its error
    against the true slope is < 1e-2 across the sensor edge at temperature 0.2
    (lucid/overlap.py _NUM_DENSE comment). rtol 2e-2 fails a factor 2, a sign, or a sigma off by
    5%; a stop_gradient anywhere on the weight path gives 0 and fails.
    """
    det, prop, grid = _built(name)
    r_s = float(det.S_radius)
    sigma = TEMPERATURE * r_s
    k, o, d, slot = _aim_isolated(det, prop, grid)
    c = np.asarray(det.all_points, np.float64)[k]

    def w(o_, d_):
        out = prop(o_[None], d_[None])
        return out['sensor_weights'][slot, 0]

    o32, d32 = jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32)
    g_o, g_d = jax.grad(w, argnums=(0, 1))(o32, d32)

    v = c - o
    p = v - (v @ d) * d
    rho = np.linalg.norm(p)
    fp = _overlap_slope(rho, r_s, sigma)
    npt.assert_allclose(float(w(o32, d32)), ncx2.cdf((r_s / sigma) ** 2, 2, (rho / sigma) ** 2),
                        atol=5e-4)
    want_o = fp * (-p / rho)
    want_d = fp * (-(v @ d) * p / rho)
    assert np.linalg.norm(np.asarray(g_o) - want_o) < 2e-2 * np.linalg.norm(want_o), (g_o, want_o)
    assert np.linalg.norm(np.asarray(g_d) - want_d) < 2e-2 * np.linalg.norm(want_d), (g_d, want_d)


def _clear_wall_point(det, min_sep):
    """A barrel point (phi, z) at least min_sep from every sensor centre."""
    S = np.asarray(det.all_points, np.float64)
    tree = cKDTree(S)
    for z in np.linspace(-0.3, 0.3, 7) * det.H:
        for phi in np.linspace(0.05, 2 * np.pi, 97, endpoint=False):
            P = np.array([det.r * np.cos(phi), det.r * np.sin(phi), z])
            if tree.query(P)[0] > min_sep:
                return phi, z
    pytest.fail("no clear wall point")


def test_wall_hit_position_and_normal_jacobian():
    det, prop, _ = _built('WCTE_like')
    r_s = float(det.S_radius)
    phi, z = _clear_wall_point(det, 1.5 * r_s)       # the radial ray enters no sensor
    u = np.array([np.cos(phi), np.sin(phi), 0.0])
    o0 = jnp.asarray([0.0, 0.0, z], jnp.float32)
    d0 = jnp.asarray(u, jnp.float32)

    def pos(o):
        return prop(o[None], d0[None])['positions'][0]

    def nrm(o):
        return prop(o[None], d0[None])['normals'][0]

    npt.assert_allclose(np.asarray(pos(o0)), [det.r * u[0], det.r * u[1], z], atol=1e-5)
    npt.assert_allclose(np.asarray(nrm(o0)), u, atol=1e-6)
    proj = np.eye(3)
    proj[:2, :2] -= np.outer(u[:2], u[:2])
    J_pos = np.asarray(jax.jacfwd(pos)(o0))
    J_nrm = np.asarray(jax.jacfwd(nrm)(o0))
    npt.assert_allclose(J_pos, proj, atol=1e-4)
    want_n = np.zeros((3, 3))
    want_n[:2, :2] = proj[:2, :2] / det.r
    npt.assert_allclose(J_nrm, want_n, atol=1e-4)


def test_cap_hit_position_jacobian():
    det, prop, _ = _built('WCTE_like')
    S = np.asarray(det.all_points, np.float64)
    tree = cKDTree(S)
    for x0 in np.linspace(-0.5, 0.5, 21):
        P = np.array([x0, 0.37, det.H / 2])
        if tree.query(P)[0] > 1.5 * float(det.S_radius):
            break
    else:
        pytest.fail("no clear cap point")
    o0 = jnp.asarray([P[0], P[1], 0.0], jnp.float32)
    d0 = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)

    def pos(o):
        return prop(o[None], d0[None])['positions'][0]

    npt.assert_allclose(np.asarray(pos(o0)), P, atol=1e-5)
    npt.assert_allclose(np.asarray(jax.jacfwd(pos)(o0)), np.diag([1.0, 1.0, 0.0]), atol=1e-5)


# ── Through photon_step and the hits ──────────────────────────────────────────────────────────


def _charge(det, prop, scatter_length, qe, key=jax.random.PRNGKey(42), n=64):
    """Total charge from n rays through one expected-value photon step (the
    _common_propagation chain, one iteration)."""
    k1, k2 = jax.random.split(key)
    origins = 0.5 * jax.random.uniform(k1, (n, 3), minval=-1.0, maxval=1.0)
    dirs = jax.random.normal(k2, (n, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    result = prop(origins, dirs)
    normals = result['normals']
    surface_distances = jnp.linalg.norm(result['positions'] - origins, axis=1) - 1e-6
    hit_sensor = jnp.max(result['inside_sensor'], axis=0)
    keys = jax.random.split(key, n)
    refl = ScalarReflection(wall_rate=jnp.asarray(0.5), sensor_rate=jnp.asarray(0.3))
    _, _, _, detect, atten, _, _, _ = jax.vmap(
        make_photon_iteration_update_factors_safe(),
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, 0, None, 0, None)
    )(origins, dirs, jnp.zeros(n), surface_distances, normals,
      scatter_length, 1.0e9, 0.0, refl, 100.0, hit_sensor, jnp.asarray(0.0), keys, 0.2253)
    w = (result['sensor_weights'] * detect[None, :] * atten[None, :]).reshape(-1)
    q, _ = on_deposits(first_arrival_expected, w, result['sensor_indices'].reshape(-1),
                       (result['times'] / 0.2253).reshape(-1), len(det.all_points),
                       qe=qe, qe_corrections=jnp.ones(len(det.all_points)))
    return jnp.sum(q)


def test_grad_charge_wrt_qe_is_charge_over_qe():
    det, prop, _ = _built('WCTE_like')
    q = float(_charge(det, prop, 50.0, 0.2))
    assert q > 0.5                                  # several rays deposit
    g = float(jax.grad(lambda qe: _charge(det, prop, 50.0, qe))(0.2))
    npt.assert_allclose(g, q / 0.2, rtol=1e-6)


def test_grad_charge_wrt_scatter_length_matches_central_difference():
    """Charge is smooth in L (each deposit carries exp(-Dd/L)). Central difference with
    h = 1 m at L = 50 m: truncation ~(h/L)^2 ~ 4e-4, float32 roundoff ~1e-4 of the difference;
    rtol 3e-3."""
    det, prop, _ = _built('WCTE_like')
    f = lambda L: _charge(det, prop, L, 0.2)
    g = float(jax.grad(f)(50.0))
    fd = (float(f(51.0)) - float(f(49.0))) / 2.0
    assert g > 0                                    # longer L, more charge
    npt.assert_allclose(g, fd, rtol=3e-3)
