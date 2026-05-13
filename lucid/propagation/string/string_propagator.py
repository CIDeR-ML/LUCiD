"""
String-telescope propagator — standard LUCiD propagator interface.

Returns the same output dict as the cylinder/sphere/box propagators so it
plugs directly into the simulator's K-loop and photon_step.

Geometry: brute-force distance to all strings, top-K selection, DOM snap,
ray-DOM closest approach. Vertical-string specialization when applicable.

Usage:
    propagate_fn = create_string_propagator(detector, sensor_radius, temperature=0.2)
    result = propagate_fn(origins, directions)
    # result has: sensor_weights, sensor_indices, sensor_distances,
    #             positions, normals, inside_sensor, ...
"""

import jax
import jax.numpy as jnp
import numpy as np

from lucid.overlap import create_overlap_prob
from lucid.geometry.string_sizing import auto_n_dom_snap


def create_string_propagator(
    detector,
    sensor_radius,
    temperature=0.2,
    n_closest=None,
    n_dom_snap=None,
):
    """Build a JIT-compiled string propagator (standard interface).

    Parameters
    ----------
    detector : StringTelescope
    sensor_radius : float
    temperature : float
    n_closest : int or None
        Strings to check per ray. None → auto.
    n_dom_snap : int or None
        DOMs per selected string. None → auto from curvature.

    Returns
    -------
    propagate_fn : callable
        propagate_fn(origins, directions) → dict
        Same output contract as shared.create_propagator.
    """
    tables = detector.get_jax_tables()
    string_anchors = tables['string_anchors']
    string_axes = tables['string_axes']
    dom_s_offsets = tables['dom_s_offsets']
    dom_global_ids = tables['dom_global_ids']

    sensor_positions = jnp.array(detector.all_points)
    n_str = detector.n_str
    max_dom = dom_s_offsets.shape[1]

    env_r_sq = detector.envelope_radius ** 2
    env_z_min = detector.envelope_z_min
    env_z_max = detector.envelope_z_max

    curv_max = float(detector.string_curv.max())
    mean_dz = float(np.median(np.diff(
        np.sort(detector.dom_s_offsets[0, :detector.n_dom_per_str_np[0]]))))

    if n_dom_snap is None:
        n_dom_snap = auto_n_dom_snap(curv_max, mean_dz)
    if n_closest is None:
        n_closest = 2 if curv_max < mean_dz * 0.5 else 3

    n_cand = n_closest * n_dom_snap

    if temperature is None:
        overlap_fn = create_overlap_prob(None, sensor_radius)
    else:
        overlap_fn = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    # Vertical specialization
    axes_np = np.array(string_axes)
    vertical = bool(np.allclose(axes_np, np.array([0., 0., 1.]), atol=1e-3))

    ax0, ax1, ax2 = string_anchors[:, 0], string_anchors[:, 1], string_anchors[:, 2]
    if not vertical:
        sax0, sax1, sax2 = string_axes[:, 0], string_axes[:, 1], string_axes[:, 2]

    _snap_offsets = jnp.arange(n_dom_snap)

    print(f"String propagator: {n_str} strings, {detector.n_sensors} DOMs, "
          f"n_closest={n_closest}, n_dom_snap={n_dom_snap}, n_cand={n_cand}"
          f"{', vertical=ON' if vertical else ''}")

    @jax.jit
    def propagate_fn(photon_origins, photon_directions):
        """Trace photon rays through string geometry.

        Parameters
        ----------
        photon_origins : (n_rays, 3)
        photon_directions : (n_rays, 3)

        Returns
        -------
        dict with keys matching the shared propagator contract:
            sensor_weights, sensor_indices, sensor_distances,
            positions, normals, inside_sensor,
            per_sensor_positions, sensor_normals
        """
        single = photon_origins.ndim == 1
        if single:
            photon_origins = photon_origins[None, :]
            photon_directions = photon_directions[None, :]

        n = photon_origins.shape[0]
        d0, d1, d2 = photon_directions[:, 0], photon_directions[:, 1], photon_directions[:, 2]
        o0, o1, o2 = photon_origins[:, 0], photon_origins[:, 1], photon_origins[:, 2]

        # ── 1. Ray-to-string distances ──
        if vertical:
            dxy_sq = d0**2 + d1**2
            w0 = o0[:, None] - ax0[None, :]
            w1 = o1[:, None] - ax1[None, :]
            origin_xy = jnp.sqrt(w0**2 + w1**2)
            scalar_triple = d1[:, None] * w0 - d0[:, None] * w1
            line_dist = jnp.abs(scalar_triple) / jnp.sqrt(dxy_sq[:, None] + 1e-18)
            wd_cross = d0[:, None] * w0 + d1[:, None] * w1
            t_ray_all = -wd_cross / (dxy_sq[:, None] + 1e-9)
            use_skew = (t_ray_all > 0) & (dxy_sq[:, None] > 1e-6)
            dist_all = jnp.where(use_skew, line_dist, origin_xy)
        else:
            cx = d1[:, None] * sax2[None, :] - d2[:, None] * sax1[None, :]
            cy = d2[:, None] * sax0[None, :] - d0[:, None] * sax2[None, :]
            cz = d0[:, None] * sax1[None, :] - d1[:, None] * sax0[None, :]
            cross_sq = cx**2 + cy**2 + cz**2
            w0 = o0[:, None] - ax0[None, :]
            w1 = o1[:, None] - ax1[None, :]
            w2 = o2[:, None] - ax2[None, :]
            scalar_triple = w0 * cx + w1 * cy + w2 * cz
            line_dist = jnp.abs(scalar_triple) / jnp.sqrt(cross_sq + 1e-18)
            dd_g = jnp.sum(photon_directions**2, axis=1)
            da_g = jnp.sum(photon_directions[:, None, :] * string_axes[None, :, :], axis=-1)
            wa_g = jnp.sum(jnp.stack([w0, w1, w2], axis=-1) * string_axes[None, :, :], axis=-1)
            wd_g = jnp.sum(jnp.stack([w0, w1, w2], axis=-1) * photon_directions[:, None, :], axis=-1)
            denom_g = dd_g[:, None] - da_g**2 + 1e-9
            t_ray_all = (da_g * wa_g - wd_g) / denom_g
            origin_perp = jnp.sqrt(jnp.maximum(w0**2 + w1**2 + w2**2 - wa_g**2, 0.0))
            use_skew = (t_ray_all > 0) & (cross_sq > 1e-6)
            dist_all = jnp.where(use_skew, line_dist, origin_perp)

        # ── 2. Top-K closest strings ──
        _, top_idx = jax.lax.top_k(-dist_all, n_closest)

        # ── 3. s_string for selected strings ──
        if vertical:
            dd = dxy_sq + d2**2
            sel_wa = o2[:, None] - ax2[top_idx]
            sel_wd = ((o0[:, None] - ax0[top_idx]) * d0[:, None] +
                      (o1[:, None] - ax1[top_idx]) * d1[:, None] +
                      sel_wa * d2[:, None])
            sel_denom = dxy_sq[:, None] + 1e-9
            sel_s = (sel_wa * dd[:, None] - sel_wd * d2[:, None]) / sel_denom
        else:
            sel_anch = string_anchors[top_idx]
            sel_axes = string_axes[top_idx]
            sel_w = photon_origins[:, None, :] - sel_anch
            dd = jnp.sum(photon_directions**2, axis=1)
            sel_da = jnp.sum(photon_directions[:, None, :] * sel_axes, axis=-1)
            sel_wa = jnp.sum(sel_w * sel_axes, axis=-1)
            sel_wd = jnp.sum(sel_w * photon_directions[:, None, :], axis=-1)
            sel_denom = dd[:, None] - sel_da**2 + 1e-9
            sel_s = (sel_wa * dd[:, None] - sel_wd * sel_da) / sel_denom

        # ── 4. DOM snap ──
        sel_offsets = dom_s_offsets[top_idx]
        below = sel_offsets <= sel_s[:, :, None]
        k_right = below.sum(axis=-1)
        k_start = jnp.clip(k_right - n_dom_snap // 2, 0, max_dom - n_dom_snap)
        dom_local = k_start[:, :, None] + _snap_offsets[None, None, :]
        dom_local = jnp.clip(dom_local, 0, max_dom - 1)
        cand_ids = dom_global_ids[top_idx[:, :, None], dom_local].reshape(n, n_cand)

        # ── 5. Ray-DOM closest approach ──
        cand_pos = sensor_positions[cand_ids]
        oc = photon_origins[:, None, :] - cand_pos
        d_norm = photon_directions / (jnp.linalg.norm(photon_directions, axis=1, keepdims=True) + 1e-10)
        t_closest = -jnp.sum(oc * d_norm[:, None, :], axis=-1)
        t_clamped = jnp.maximum(t_closest, 0.0)
        closest_pts = photon_origins[:, None, :] + t_clamped[:, :, None] * d_norm[:, None, :]
        to_sensor = closest_pts - cand_pos
        perp_dist = jnp.sqrt(jnp.sum(to_sensor**2, axis=-1) + 1e-18)

        # ── 6. Overlap weights ──
        ov = overlap_fn(perp_dist)
        valid = cand_ids >= 0
        ov = jnp.where(valid, ov, 0.0)

        # ── 7. Sensor normals (outward from DOM center) ──
        sensor_normals = -to_sensor / (jnp.linalg.norm(to_sensor, axis=-1, keepdims=True) + 1e-10)

        # ── 8. inside_sensor: overlap-based (volume model) ──
        has_weight = ov > 1e-10

        # ── 9. Best hit for positions/normals (closest DOM with weight) ──
        weighted_t = jnp.where(has_weight, t_clamped, 1e10)
        best_slot = jnp.argmin(weighted_t, axis=1)
        best_pos = closest_pts[jnp.arange(n), best_slot]
        best_normal = sensor_normals[jnp.arange(n), best_slot]

        # Envelope exit fallback for rays that miss all DOMs
        hit_any = jnp.any(has_weight, axis=1)
        ox, oy, oz = o0, o1, o2
        dx, dy, dz = d0, d1, d2
        a_env = dx**2 + dy**2 + 1e-30
        b_env = 2.0 * (ox * dx + oy * dy)
        c_env = ox**2 + oy**2 - env_r_sq
        disc_env = b_env**2 - 4 * a_env * c_env
        sqrt_disc = jnp.sqrt(jnp.maximum(disc_env, 0.0))
        t_barrel = jnp.where(disc_env > 0, (-b_env + sqrt_disc) / (2 * a_env), 1e10)
        t_top = jnp.where(jnp.abs(dz) > 1e-20, (env_z_max - oz) / dz, 1e10)
        t_bot = jnp.where(jnp.abs(dz) > 1e-20, (env_z_min - oz) / dz, 1e10)
        t_env = jnp.minimum(jnp.minimum(
            jnp.where(t_barrel > 0, t_barrel, 1e10),
            jnp.where(t_top > 0, t_top, 1e10)),
            jnp.where(t_bot > 0, t_bot, 1e10))
        env_pts = photon_origins + t_env[:, None] * photon_directions
        env_normals_xy = env_pts[:, :2]
        env_norm_len = jnp.linalg.norm(env_normals_xy, axis=1, keepdims=True) + 1e-10
        env_normals = jnp.concatenate([env_normals_xy / env_norm_len, jnp.zeros((n, 1))], axis=1)

        positions = jnp.where(hit_any[:, None], best_pos, env_pts)
        normals = jnp.where(hit_any[:, None], best_normal, env_normals)

        # ── Pack output (transpose to match shared propagator: slot-first) ──
        result = {
            'sensor_weights': ov.T,                               # (n_cand, n_rays)
            'sensor_indices': cand_ids.T,                         # (n_cand, n_rays)
            'sensor_distances': t_clamped.T[:, :, None],          # (n_cand, n_rays, 1)
            'positions': positions,                                # (n_rays, 3)
            'normals': normals,                                    # (n_rays, 3)
            'inside_sensor': has_weight.T,                         # (n_cand, n_rays)
            'per_sensor_positions': closest_pts.transpose(1, 0, 2),  # (n_cand, n_rays, 3)
            'sensor_normals': sensor_normals.transpose(1, 0, 2),     # (n_cand, n_rays, 3)
        }

        if single:
            result = jax.tree.map(lambda x: x.squeeze(1) if x.ndim > 1 and x.shape[1] == 1
                                  else (x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x),
                                  result)
            # For single ray, shapes collapse: (n_cand,1) → (n_cand,), (1,3) → (3,)

        return result

    propagate_fn.n_cand = n_cand
    propagate_fn.n_closest = n_closest
    propagate_fn.n_dom_snap = n_dom_snap

    return propagate_fn
