"""Data-driven starts for the track fit, and the reconstruction that uses them.

The published method, from the observed per-PMT charge and first-arrival time alone:

1. **energy** — scan a fixed probe track over energy and keep the total-charge match;
2. **vertex A, first-light grid** — a hierarchical vertex grid at each of several t0 values,
   scored on first-arrival residuals (:func:`lucid.losses.grid_origin_time_loss`, no simulation);
3. **vertex B, time multilateration** — :func:`seed_vertex_time`;
4. **direction** — a hierarchical cone search at each vertex, scored on charge;
5. **seed F** — :func:`fuse_seeds`, B's transverse vertex with A's longitudinal one;
6. **pick** — the data loss of A, B and F before fitting; A is kept unless another start beats
   it by ``pick_margin`` (:func:`pick_by_margin`);
7. **fit** — one :func:`lucid.fitting.fit_track` from the picked start.

Every number is in the recipe (:data:`SEED_RECIPE` is the published one) or comes from the
detector and medium; nothing here has a default. The random stream is the caller's: one key per
event, split here into the energy scan's and each cone search's.
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import ParticleParams
from lucid.losses import counts_loss, energy_loss, grid_origin_time_loss
from lucid.fitting.recon import fit_track, vec9_dir, vec9_from_track
from lucid.utils import cartesian_to_spherical, spherical_to_cartesian

@dataclass(frozen=True)
class SeedRecipe:
    """The settings of the seeding method; every field is required."""
    # energy scan: linspace(center - delta, center + delta, steps) MeV at a probe track through the
    # detector centre along probe_direction. Only the total charge is scored, so the probe's
    # t0 does not enter.
    energy_center: float
    energy_delta: float
    energy_steps: int
    energy_probe_direction: tuple[float, float, float]
    # first-light grid: t0 in linspace(t0_min, t0_max, t0_n_div); at each, grid_levels levels of
    # a grid_n_div^3 grid, each level refining around the previous best. The t0 range must
    # cover the true t0 of the sample. early_penalty weights arrivals earlier than the hypothesis.
    t0_min: float
    t0_max: float
    t0_n_div: int
    grid_n_div: int
    grid_levels: int
    early_penalty: float
    # time multilateration (see seed_vertex_time). time_speed is the group velocity of light in
    # water; the simulation propagates at the phase velocity c/n, which the grid uses.
    time_speed: float
    time_vgrid: int
    time_n_refine: int
    time_gate: float
    time_bright_frac: float
    time_q_edge: float
    # cone direction search: initial_div polar angles pole to pole (one direction per pole,
    # 2*initial_div azimuths between), then cone_levels - 1 cones of cone_rings rings around the
    # best so far, opening cone_max_deg and shrinking by cone_reduction per level.
    cone_levels: int
    cone_initial_div: int
    cone_max_deg: float
    cone_reduction: float
    cone_rings: int
    # pre-fit pick: a start replaces A only if its data loss is lower by this fraction.
    pick_margin: float


# The recipe the paper validated (analysis/paper/utils/studies.py runs it unchanged).
SEED_RECIPE = SeedRecipe(
    energy_center=1600.0, energy_delta=1400.0, energy_steps=57,
    energy_probe_direction=(1.0, 1.0, 1.0),
    t0_min=-20.0, t0_max=20.0, t0_n_div=9, grid_n_div=5, grid_levels=5, early_penalty=100.0,
    time_speed=0.2167, time_vgrid=11, time_n_refine=20, time_gate=18.0, time_bright_frac=0.6,
    time_q_edge=0.10,
    cone_levels=3, cone_initial_div=8, cone_max_deg=90.0, cone_reduction=0.5, cone_rings=4,
    pick_margin=0.01,
)


def _track(energy, position, theta, phi, t0):
    return ParticleParams(energy=jnp.asarray(energy), position=jnp.asarray(position),
                          theta=jnp.asarray(theta), phi=jnp.asarray(phi), t0=jnp.asarray(t0))


def energy_scan(pred, obs_counts, *, energies, vertex, direction, key):
    """Energy whose total predicted charge best matches the observed total, at a fixed track.

    ``pred`` is the per-photon track simulator. One forward per energy, each on its own key split
    from ``key``; the first of equal losses wins.
    """
    d = np.asarray(direction, float)
    theta, phi = cartesian_to_spherical(d / np.linalg.norm(d), np)
    best_energy, best_loss = None, float('inf')
    for energy in energies:
        key, _ = jax.random.split(key)
        total_charge = pred(_track(energy, vertex, theta, phi, 0.0), key)[3]
        loss = energy_loss(total_charge, obs_counts)
        if loss < best_loss:
            best_loss, best_energy = loss, energy
    return float(best_energy)


def _grid_points(center, half_size, spacing, region):
    """The points of ``region`` on the grid of ``spacing`` spanning ``center ± half_size``."""
    n = int(round(2 * half_size / spacing)) + 1             # the count the spacing implies
    axes = [np.linspace(c - half_size, c + half_size, n) for c in center]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return pts[region.contains(pts)]


def first_light_grid(sensor_positions, obs_counts, obs_times, *, region, t0_values, n_div,
                     levels, sensor_radius, light_speed, early_penalty):
    """Vertex and t0 minimising the first-arrival residual loss over a hierarchical grid.

    For each t0 the search starts on a ``n_div``-spaced grid over ``region``'s bounding box and
    refines ``levels`` times, each level a grid of half a cell around the previous best. Points
    outside ``region`` are not evaluated. Returns ``(vertex (3,), t0)``.
    """
    lo, hi = region.bounding_box()
    best = (float('inf'), None, None)
    for t0 in t0_values:
        loss_at = jax.vmap(lambda v, t0=t0: grid_origin_time_loss(
            v, sensor_positions, obs_times, obs_counts, t0, sensor_radius=sensor_radius,
            light_speed=light_speed, early_penalty=early_penalty))
        center, half = tuple(0.5 * (lo + hi)), float(np.max(hi - lo)) / 2
        for _ in range(levels):
            spacing = 2 * half / n_div
            pts = _grid_points(center, half, spacing, region)
            if pts.size == 0:
                raise ValueError(f'first-light grid: no point of the {spacing:.3g} m grid around '
                                 f'{center} lies in the region')
            losses = loss_at(jnp.asarray(pts))
            if not bool(jnp.all(jnp.isfinite(losses))):
                raise FloatingPointError('first-light grid: non-finite loss')
            i = jnp.argmin(losses)
            vertex = jnp.array(pts[i])
            if float(losses[i]) < best[0]:
                best = (float(losses[i]), vertex, t0)
            center, half = tuple(float(c) for c in vertex), spacing / 2
    return np.array(best[1]), float(best[2])


def seed_vertex_time(sensor_positions, obs_counts, obs_times, *, region, speed, vgrid, n_refine,
                     gate, bright_frac, q_edge):
    """Robust multilateration vertex and t0 from first-arrival times; no simulation.

    A direct photon reaching PMT ``P`` at ``T`` left the vertex ``v`` at ``t0`` and travelled at
    ``speed`` (m/ns): ``T ≈ t0 + |P - v| / speed``.

    Many hits are scattered or reflected photons that arrive tens of ns late, and dim PMTs are
    dominated by them. So:
    1. keep the brightest ``bright_frac`` of hits;
    2. scan a ``vgrid``³ grid over ``region``'s bounding box, the points inside ``region`` only,
       scoring each by the sqrt-charge-weighted count of inliers. A hit is an inlier when its
       residual is within ``±gate`` ns of the early edge (quantile ``q_edge`` of the
       back-projected emission time);
    3. polish with ``n_refine`` Gauss-Newton steps on the inliers alone, setting ``t0`` to their
       weighted-mean residual each step and bringing the vertex back into ``region`` (its
       nearest point) after each.

    Returns ``(vertex (3,), t0)``.
    """
    pos = np.asarray(sensor_positions, float)
    oc = np.asarray(obs_counts, float); ot = np.asarray(obs_times, float)
    hit = oc > 0
    P = pos[hit]; T = ot[hit]; q = np.maximum(oc[hit], 0.)
    bm = q >= np.quantile(q, 1. - bright_frac); P, T, q = P[bm], T[bm], q[bm]
    w = np.sqrt(q)

    lo, hi = region.bounding_box()
    gx, gy, gz = np.meshgrid(*(np.linspace(a, b, vgrid) for a, b in zip(lo, hi)), indexing='ij')
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    grid = grid[region.contains(grid)]
    best = (-1.0, grid[0])
    for v in grid:                                             # coarse scan: maximise weighted inliers
        d = T - np.linalg.norm(P - v, axis=1) / speed          # back-projected emission time
        res = d - np.quantile(d, q_edge)                       # reference to the early edge
        s = w[(res > -gate) & (res < gate)].sum()
        if s > best[0]: best = (s, v)
    v = best[1].astype(float); t0 = float(np.quantile(T - np.linalg.norm(P - v, axis=1) / speed, q_edge))
    for _ in range(n_refine):                                  # GN polish on inliers only
        r = np.linalg.norm(P - v, axis=1) + 1e-6; pt = r / speed; d = T - pt
        res = d - np.quantile(d, q_edge); inl = (res > -gate) & (res < gate)
        Pi, ri, pti, wi, Ti = P[inl], r[inl], pt[inl], w[inl], T[inl]
        sw = wi.sum() + 1e-12; t0 = float(np.sum(wi * (Ti - pti)) / sw); resid = Ti - t0 - pti
        g = (Pi - v) / ri[:, None]; J = g / speed
        Hh = (J * wi[:, None]).T @ J + 1e-3 * np.eye(3); gg = (J * wi[:, None]).T @ resid
        v = region.clip(v - np.clip(np.linalg.solve(Hh, gg), -2., 2.))   # step clipped to 2 m
    return v, float(t0)


def _cone_points(axis, max_angle_rad, num_rings):
    """Directions on ``num_rings`` rings around the unit ``axis``, out to ``max_angle_rad``."""
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.99 else np.cross(axis, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    points = []
    for i in range(num_rings):
        theta = (i + 1) * max_angle_rad / num_rings
        ring_radius, z = np.sin(theta), np.cos(theta)
        n = max(1, int(2 * np.pi * ring_radius * num_rings))       # spacing matched along theta
        for j in range(n):
            phi = 2 * np.pi * j / n
            points.append(z * axis + ring_radius * (np.cos(phi) * u + np.sin(phi) * v))
    return np.array(points)


def cone_direction(pred, obs_counts, vertex, t0, energy, *, levels, initial_div, max_angle_deg,
                   reduction, rings, key):
    """Direction from a hierarchical cone search on the per-PMT charge loss.

    Level 0 scores distinct directions over the sphere: ``initial_div`` polar angles from pole
    to pole, one direction at each pole and ``2*initial_div`` azimuths on each ring between.
    Each later level scores a cone of ``rings`` rings around the best direction so far, the
    opening starting at ``max_angle_deg`` and shrinking by ``reduction``. The best is carried
    across levels with its loss, so the incumbent competes with its neighbourhood without being
    scored again. Returns the best of all levels as a unit vector. Each forward takes its own key
    split from ``key``.
    """
    thetas = np.linspace(0, np.pi, initial_div)[1:-1]
    phis = np.linspace(0, 2 * np.pi, 2 * initial_div, endpoint=False)
    directions = np.array([[0.0, 0.0, 1.0]]
                          + [spherical_to_cartesian(t, p, np) for t in thetas for p in phis]
                          + [[0.0, 0.0, -1.0]])
    max_angle = np.radians(max_angle_deg)
    best_loss, best = float('inf'), None
    for level in range(levels):
        for direction in directions:
            theta, phi = cartesian_to_spherical(direction, np)
            key, _ = jax.random.split(key)
            total_charge = pred(_track(energy, vertex, theta, phi, t0), key)[3]
            loss = counts_loss(obs_counts, total_charge)
            if loss < best_loss:
                best_loss, best = loss, direction.copy()
        if level < levels - 1:
            directions = _cone_points(best, max_angle, rings)
            max_angle *= reduction
    theta, phi = cartesian_to_spherical(best, np)
    return spherical_to_cartesian(theta, phi, np)


def fuse_seeds(seed_a, seed_b):
    """Combine the two seeds, taking each one's strong component. No truth is used.

    Time multilateration (``seed_b``) is good transverse to the track and poor along it; the
    first-light grid (``seed_a``) is longitudinally unbiased and has the better direction. The
    vertex is ``vtx_b`` plus the part of ``vtx_a - vtx_b`` along seed_a's direction; direction and
    energy come from ``seed_a``. Their t0 biases have opposite sign (a early, b late), so t0 is
    the mean.
    """
    a = np.asarray(seed_a, float)
    b = np.asarray(seed_b, float)
    d = vec9_dir(a)
    out = a.copy()
    out[1:4] = b[1:4] + float(np.dot(a[1:4] - b[1:4], d)) * d
    out[8] = 0.5 * (a[8] + b[8])
    return out


def pick_by_margin(losses, prefer, margin):
    """Index of the winning start: keep ``prefer`` unless another beats it by ``margin`` x |loss|.

    A data loss is a noisy and slightly biased ranking of starts (the model is not exactly at
    truth), so a marginally lower loss does not mean a better vertex; the margin keeps the
    preferred start unless another genuinely wins.
    """
    base = losses[prefer]
    thr = base - margin * abs(base)
    cand = [i for i in range(len(losses)) if i != prefer and losses[i] < thr]
    return min(cand, key=lambda i: losses[i]) if cand else prefer


def seed_track(pred, obs_counts, obs_times, *, recipe, key):
    """Seeds A, B and F (9-vectors) from the observed charge and times alone.

    The detector is the predictor's own (``pred.det_geom``): its sensor positions, sensor radius
    and the volume both vertex seeders search, and the light speed (m/ns) the simulation
    propagates at. ``key`` is the event's own: the energy scan and the cone searches at A and at
    B each take an independent key split from it. Returns ``{'A', 'B', 'F'}``.
    """
    r = recipe
    detector = pred.det_geom.detector
    energy_key, cone_key_a, cone_key_b = jax.random.split(key, 3)
    pos = np.asarray(detector.all_points)                  # float64; not det_geom.sensor_points
    region = detector.volume
    oc, ot = jnp.asarray(obs_counts), jnp.asarray(obs_times)
    lo, hi = region.bounding_box()
    e0 = energy_scan(pred, oc, energies=np.linspace(r.energy_center - r.energy_delta,
                                                    r.energy_center + r.energy_delta,
                                                    r.energy_steps),
                     vertex=0.5 * (lo + hi), direction=r.energy_probe_direction,
                     key=energy_key)
    vtx_a, t0_a = first_light_grid(
        jnp.asarray(pos), oc, ot, region=region,
        t0_values=np.linspace(r.t0_min, r.t0_max, r.t0_n_div),
        n_div=r.grid_n_div, levels=r.grid_levels, sensor_radius=detector.S_radius,
        light_speed=pred.det_geom.speed_of_light, early_penalty=r.early_penalty)
    vtx_b, t0_b = seed_vertex_time(pos, obs_counts, obs_times, region=region,
                                   speed=r.time_speed, vgrid=r.time_vgrid,
                                   n_refine=r.time_n_refine, gate=r.time_gate,
                                   bright_frac=r.time_bright_frac, q_edge=r.time_q_edge)

    def with_direction(vtx, t0, cone_key):
        d = cone_direction(pred, oc, jnp.asarray(vtx), t0, e0, levels=r.cone_levels,
                           initial_div=r.cone_initial_div, max_angle_deg=r.cone_max_deg,
                           reduction=r.cone_reduction, rings=r.cone_rings, key=cone_key)
        return vec9_from_track(e0, np.asarray(vtx), d, t0=t0)

    seed_a = with_direction(vtx_a, t0_a, cone_key_a)
    seed_b = with_direction(vtx_b, t0_b, cone_key_b)
    return {'A': seed_a, 'B': seed_b, 'F': fuse_seeds(seed_a, seed_b)}


def reconstruct(model, obs_counts, obs_times, *, seed, fit, seed_key, fit_key):
    """Reconstruct one event: seeds A, B, F, the pre-fit pick, then one fit.

    ``seed`` is the :class:`SeedRecipe` and ``seed_key`` the event's seeding key
    (:func:`seed_track`); ``fit`` the :class:`~lucid.fitting.recon.FitRecipe` and ``fit_key``
    the event's fit key (:func:`~lucid.fitting.recon.fit_track`), whose ``fit.keys(fit_key)``
    the pick also scores each start with. Returns
    ``(theta, info)``, ``info`` holding the ``seeds``, their pre-fit ``losses``, the index
    ``which`` that was fitted (0 A, 1 B, 2 F), and the fit ``history``.
    """
    seeds = seed_track(model.pred, obs_counts, obs_times, recipe=seed, key=seed_key)
    starts = [seeds['A'], seeds['B'], seeds['F']]
    keys = fit.keys(fit_key)

    def data_loss(start):
        return float(np.mean([float(model.loss(np.asarray(start), obs_counts, obs_times, k))
                              for k in keys]))

    losses = [data_loss(s) for s in starts]
    which = pick_by_margin(losses, prefer=0, margin=seed.pick_margin)
    theta, history = fit_track(model, obs_counts, obs_times, starts[which], fit, key=fit_key)
    return theta, dict(seeds=seeds, losses=losses, which=which, history=history)
