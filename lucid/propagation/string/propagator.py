"""
String-telescope photon propagator factory.

Parallel to ``lucid.propagation.shared.create_propagator`` but replaces
the surface-grid → inverted-sensor-map path with the DDA → string-match
volumetric path.

Produces a JIT-compiled ``propagate_photons(origins, directions)`` that
returns the same output dict shape as the shared propagator, so the
existing photon_step / simulator code works unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lucid.propagation.base import compute_sensor_intersections_base
from lucid.overlap import create_overlap_prob
from lucid.propagation.string.hash import build_string_hash
from lucid.propagation.string.match import match_segment_batch
from lucid.geometry.string_sizing import size_string_propagator, auto_n_dom_snap


def create_string_propagator(
    detector,
    sensor_positions,
    sensor_radius,
    temperature=0.2,
    lambda_abs=100.0,
    lambda_scat=30.0,
    eps_seg=0.01,
    max_str_per_cell=6,
    n_dom_snap=None,
):
    """Build a JIT-compiled string-telescope propagator.

    Parameters
    ----------
    detector : StringTelescope
    sensor_positions : jnp.ndarray, (N_sensors, 3)
    sensor_radius : float
    temperature : float
        Soft-assignment kernel sigma = temperature * sensor_radius.
    lambda_abs : float
        Absorption length (m) — worst case across wavelengths.
    lambda_scat : float
        Scattering length (m) — worst case across wavelengths.
    eps_seg : float
        Per-segment truncation tolerance (default 1%).
    max_str_per_cell : int
        Spatial hash density bound.
    n_dom_snap : int or None
        DOMs per candidate string. If None, auto-derived from curvature.

    Returns
    -------
    propagate_photons : callable
        JIT-compiled, same output signature as shared.create_propagator.
    """
    sensor_positions = jnp.array(sensor_positions)
    tables = detector.get_jax_tables()

    # ── Auto-derive sizing ──
    mean_dz = float(np.median(np.diff(
        np.sort(detector.dom_s_offsets[0, :detector.n_dom_per_str_np[0]]))))
    dxy = _estimate_inter_string_spacing(detector.string_anchors)

    if n_dom_snap is None:
        n_dom_snap = auto_n_dom_snap(float(detector.string_curv.max()), mean_dz)

    sizing = size_string_propagator(
        lambda_abs=lambda_abs,
        lambda_scat=lambda_scat,
        dxy=dxy,
        eps_seg=eps_seg,
        max_str_per_cell=max_str_per_cell,
        n_dom_snap=n_dom_snap,
    )

    # ── Build spatial hash ──
    r_filter = sensor_radius + (temperature or 0.0) * sensor_radius * 3.0 + \
               float(detector.string_curv.max())

    cell_map_np, grid_origin_np, grid_shape_np, cell_size, hash_stats = \
        build_string_hash(
            detector.string_anchors,
            detector.string_tops,
            cell_size=sizing.cell_size,
            r_filter=r_filter,
            max_strings_per_cell=max_str_per_cell,
        )

    print(f"String propagator: {detector.n_str} strings, {detector.n_sensors} DOMs, "
          f"n_dom_snap={n_dom_snap}, max_cells/seg={sizing.max_cells_per_segment}, "
          f"K_min={sizing.K_min}")
    print(f"  Hash: {hash_stats['total_cells']} cells, "
          f"max_occ={hash_stats['max_occupancy']}, "
          f"coverage={hash_stats['string_coverage']:.0%}")

    # ── Convert to JAX ──
    cell_map = jnp.array(cell_map_np)
    grid_origin = jnp.array(grid_origin_np)
    grid_shape = jnp.array(grid_shape_np)

    max_cells = sizing.max_cells_per_segment
    max_dom_per_segment = sizing.max_dom_per_segment

    # ── Overlap probability ──
    if temperature is None:
        overlap_prob = create_overlap_prob(None, sensor_radius)
    else:
        overlap_prob = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    # ── Bounds check closure ──
    env_r_sq = detector.envelope_radius ** 2
    env_z_min = detector.envelope_z_min
    env_z_max = detector.envelope_z_max

    def bounds_check(positions):
        xy_sq = positions[..., 0]**2 + positions[..., 1]**2
        return (xy_sq <= env_r_sq) & \
               (positions[..., 2] >= env_z_min) & \
               (positions[..., 2] <= env_z_max)

    # ── Envelope exit intersection (for surface_distance when no DOM hit) ──
    @jax.jit
    def envelope_exit_t(origins, directions):
        """Distance along ray to envelope cylinder exit (barrel or caps)."""
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        # Barrel: ox²+oy² + 2t(ox·dx+oy·dy) + t²(dx²+dy²) = R²
        a = dx**2 + dy**2 + 1e-30
        b = 2.0 * (ox * dx + oy * dy)
        c = ox**2 + oy**2 - env_r_sq
        disc = b**2 - 4 * a * c
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        t_barrel = jnp.where(disc > 0, (-b + sqrt_disc) / (2 * a), 1e10)

        # Caps
        t_top = jnp.where(jnp.abs(dz) > 1e-20,
                          (env_z_max - oz) / dz, 1e10)
        t_bot = jnp.where(jnp.abs(dz) > 1e-20,
                          (env_z_min - oz) / dz, 1e10)

        # Only forward intersections
        t_barrel = jnp.where(t_barrel > 0, t_barrel, 1e10)
        t_top = jnp.where(t_top > 0, t_top, 1e10)
        t_bot = jnp.where(t_bot > 0, t_bot, 1e10)

        return jnp.minimum(jnp.minimum(t_barrel, t_top), t_bot)

    # ── Precompute static args ──
    string_anchors_j = tables['string_anchors']
    string_axes_j = tables['string_axes']
    dom_s_offsets_j = tables['dom_s_offsets']
    dom_global_ids_j = tables['dom_global_ids']
    string_s_min_j = tables['string_s_min']
    string_s_max_j = tables['string_s_max']

    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        """Trace photon rays through string-telescope geometry.

        Parameters
        ----------
        photon_origins : (n_rays, 3)
        photon_directions : (n_rays, 3)

        Returns
        -------
        dict matching shared propagator output contract.
        """
        single_ray = photon_origins.ndim == 1
        if single_ray:
            photon_origins = photon_origins[None, :]
            photon_directions = photon_directions[None, :]

        n_rays = photon_origins.shape[0]

        # 1. Find candidate DOMs via DDA + string match
        candidate_ids = jax.lax.stop_gradient(
            match_segment_batch(
                photon_origins, photon_directions,
                grid_origin, grid_shape, cell_size, cell_map,
                string_anchors_j, string_axes_j,
                dom_s_offsets_j, dom_global_ids_j,
                string_s_min_j, string_s_max_j,
                max_cells, max_str_per_cell, n_dom_snap,
                r_filter,
            )
        )
        # candidate_ids: (n_rays, max_dom_per_segment)

        # 2. Compute ray-sphere intersections via existing base function
        def compute_for_slot(slot_sensors):
            return compute_sensor_intersections_base(
                slot_sensors, sensor_positions, sensor_radius,
                photon_origins, photon_directions,
                bounds_check, overlap_prob)

        (weights, sensor_distances, sensor_indices,
         sensor_normals_all, inside_sensor,
         sensor_hit_positions) = jax.vmap(
            compute_for_slot, in_axes=1, out_axes=0)(candidate_ids)

        # 3. Compute "geometry" hit for non-sensor rays (envelope exit)
        t_envelope = envelope_exit_t(photon_origins, photon_directions)
        envelope_points = photon_origins + t_envelope[:, None] * photon_directions

        # Envelope outward normal (barrel-dominant approximation)
        env_normals_xy = envelope_points[:, :2]
        env_normal_len = jnp.linalg.norm(env_normals_xy, axis=1, keepdims=True) + 1e-10
        geometry_normals = jnp.concatenate([
            env_normals_xy / env_normal_len,
            jnp.zeros((n_rays, 1))
        ], axis=1)

        # 4. Determine final positions and normals
        #
        # Volume-mode override: use overlap WEIGHT instead of the hard
        # inside_sensor (distance < r) flag. This is critical for large-σ
        # variance reduction — near-miss photons have nonzero overlap
        # weight and must be treated as "reaching" the DOM so that
        # surface_distance (and thus detect_prob) is computed relative
        # to the DOM, not the distant envelope exit.
        #
        # The hard inside_sensor flag would route near-misses to the
        # envelope exit (surface_distance ~ 500m → detect_prob ≈ 0),
        # zeroing out the overlap weight and destroying mean preservation.
        has_weight = weights > 1e-10
        hit_any = jnp.any(has_weight, axis=0)  # (n_rays,)

        valid_dists = jnp.where(has_weight, sensor_distances.squeeze(-1), 1e10)
        best_slot = jnp.argmin(valid_dists, axis=0)  # (n_rays,)
        best_positions = sensor_hit_positions[best_slot, jnp.arange(n_rays)]
        best_normals = sensor_normals_all[best_slot, jnp.arange(n_rays)]

        hit_positions = jnp.where(hit_any[:, None], best_positions, envelope_points)
        final_normals = jnp.where(hit_any[:, None], best_normals, geometry_normals)

        result = {
            'sensor_distances': sensor_distances,
            'sensor_weights': weights,
            'sensor_indices': sensor_indices,
            'per_sensor_positions': sensor_hit_positions,
            'positions': hit_positions,
            'normals': final_normals,
            'sensor_normals': sensor_normals_all,
            'inside_sensor': has_weight,
        }

        if single_ray:
            result = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, result)

        return result

    propagate_photons.sizing = sizing
    propagate_photons.hash_stats = hash_stats

    return propagate_photons


def _estimate_inter_string_spacing(anchors):
    """Estimate typical inter-string spacing from anchor positions."""
    from scipy.spatial import KDTree
    xy = anchors[:, :2]
    tree = KDTree(xy)
    dists, _ = tree.query(xy, k=2)
    return float(np.median(dists[:, 1]))
