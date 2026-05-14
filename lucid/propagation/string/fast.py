"""
Fast string-telescope simulator: brute-force string matching + lax.scan.

Replaces DDA+hash+match pipeline with direct distance computation to all
strings, picks the N closest, snaps to nearest DOMs. The full K-loop is
compiled into a single JIT'd function via lax.scan.

Handles straight and curved strings via fitted-axis approach:
  - String selection: skew-line distance to fitted axis (exact for straight,
    approximate for curved — error bounded by curv_max)
  - DOM snap: comparison-based bracket finding on dom_s_offsets, works for
    both uniform and non-uniform DOM spacing
  - n_dom_snap auto-derived from curvature via auto_n_dom_snap
  - Ray-DOM distance: exact (uses actual DOM positions)
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from lucid.overlap import create_overlap_prob
from lucid.simulation.optics import solve_rayleigh_inverse_cdf
from lucid.geometry.string_sizing import auto_n_dom_snap


def create_fast_string_simulator(
    detector,
    sensor_radius,
    temperature=0.2,
    lambda_abs=100.0,
    lambda_scat=30.0,
    speed_of_light=0.2254,
    n_closest=None,
    n_dom_snap=None,
):
    """Build a JIT-compiled string-telescope simulator.

    Returns a function that runs the full K-loop:
        simulate(origins, dirs, weights, K, key) -> (dom_charges, dom_time_weighted)

    Parameters
    ----------
    detector : StringTelescope
    sensor_radius : float
    temperature : float
    lambda_abs, lambda_scat : float
    speed_of_light : float
    n_closest : int or None
        Strings to check per ray. None → auto (2 for straight, 3 for curved).
    n_dom_snap : int or None
        DOMs per selected string. None → auto from curvature.
    """
    tables = detector.get_jax_tables()
    string_anchors = tables['string_anchors']
    string_axes = tables['string_axes']
    dom_s_offsets = tables['dom_s_offsets']       # (n_str, max_dom), sorted, +inf padded
    dom_global_ids = tables['dom_global_ids']     # (n_str, max_dom), -1 padded
    string_s_min = tables['string_s_min']
    string_s_max = tables['string_s_max']

    sensor_positions = jnp.array(detector.all_points)
    n_str = detector.n_str
    n_sensors = detector.n_sensors
    max_dom = dom_s_offsets.shape[1]

    n_dom_per_str = jnp.array(detector.n_dom_per_str_np)

    env_r_sq = detector.envelope_radius ** 2
    env_z_min = detector.envelope_z_min
    env_z_max = detector.envelope_z_max

    curv_max = float(detector.string_curv.max())
    mean_dz = float(np.median(np.diff(
        np.sort(detector.dom_s_offsets[0, :detector.n_dom_per_str_np[0]]))))

    # Auto-derive n_dom_snap from curvature
    if n_dom_snap is None:
        n_dom_snap = auto_n_dom_snap(curv_max, mean_dz)

    # Auto-derive n_closest: 2 for straight strings, +1 if significant curvature
    if n_closest is None:
        n_closest = 2 if curv_max < mean_dz * 0.5 else 3

    n_cand = n_closest * n_dom_snap

    # Overlap function
    if temperature is None:
        overlap_fn = create_overlap_prob(None, sensor_radius)
    else:
        overlap_fn = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    # Detect vertical strings (all axes ≈ [0,0,1])
    axes_np = np.array(string_axes)
    vertical = bool(np.allclose(axes_np, np.array([0., 0., 1.]), atol=1e-3))

    ax0, ax1, ax2 = string_axes[:, 0], string_axes[:, 1], string_axes[:, 2]
    anch0, anch1, anch2 = string_anchors[:, 0], string_anchors[:, 1], string_anchors[:, 2]

    # Static arange for DOM snap window
    _snap_offsets = jnp.arange(n_dom_snap)

    print(f"Fast string simulator: {n_str} strings, {n_sensors} DOMs, "
          f"n_closest={n_closest}, n_dom_snap={n_dom_snap}, n_cand={n_cand}")
    if vertical:
        print(f"  Vertical specialization: ON")
    if curv_max > 0:
        print(f"  Max curvature: {curv_max:.2f}m (mean_dz={mean_dz:.1f}m)")

    @partial(jax.jit, static_argnames=('K',))
    def simulate(origins, directions, weights, K, key):
        """Run full string simulation for one event.

        Parameters
        ----------
        origins : (n, 3)     photon starting positions
        directions : (n, 3)  photon starting directions
        weights : (n,)       photon intensities
        K : int              scatter iterations (static)
        key : PRNGKey

        Returns
        -------
        dom_charges : (n_sensors,)
        dom_time_weighted : (n_sensors,)
        """
        n = origins.shape[0]

        def bounds_check(pos):
            xy_sq = pos[..., 0]**2 + pos[..., 1]**2
            return ((xy_sq <= env_r_sq) &
                    (pos[..., 2] >= env_z_min) &
                    (pos[..., 2] <= env_z_max))

        def envelope_exit(pos, dirs):
            ox, oy, oz = pos[:, 0], pos[:, 1], pos[:, 2]
            dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
            a = dx**2 + dy**2 + 1e-30
            b = 2.0 * (ox * dx + oy * dy)
            c = ox**2 + oy**2 - env_r_sq
            disc = b**2 - 4 * a * c
            sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
            t_barrel = jnp.where(disc > 0, (-b + sqrt_disc) / (2 * a), 1e10)
            t_top = jnp.where(jnp.abs(dz) > 1e-20, (env_z_max - oz) / dz, 1e10)
            t_bot = jnp.where(jnp.abs(dz) > 1e-20, (env_z_min - oz) / dz, 1e10)
            t_barrel = jnp.where(t_barrel > 0, t_barrel, 1e10)
            t_top = jnp.where(t_top > 0, t_top, 1e10)
            t_bot = jnp.where(t_bot > 0, t_bot, 1e10)
            return jnp.minimum(jnp.minimum(t_barrel, t_top), t_bot)

        def scatter_directions_batch(dirs, key):
            n_loc = dirs.shape[0]
            k1, k2 = jax.random.split(key)
            u1 = jax.random.uniform(k1, (n_loc,))
            u2 = jax.random.uniform(k2, (n_loc,))

            cos_theta = solve_rayleigh_inverse_cdf(u1)
            sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta**2, 0.0))
            phi = 2.0 * jnp.pi * u2

            lx = sin_theta * jnp.cos(phi)
            ly = sin_theta * jnp.sin(phi)
            lz = cos_theta

            z = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10)
            t_vec = jnp.where(
                jnp.abs(z[:, 0:1]) < 0.9,
                jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), z.shape),
                jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), z.shape))
            x = jnp.cross(t_vec, z)
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
            y = jnp.cross(z, x)

            new_dirs = x * lx[:, None] + y * ly[:, None] + z * lz[:, None]
            return new_dirs / (jnp.linalg.norm(new_dirs, axis=1, keepdims=True) + 1e-10)

        def step_fn(carry, k_idx):
            pos, dirs, times, survival, dom_q, dom_tw, key = carry
            key, k_scatter, k_dir = jax.random.split(key, 3)

            inside = bounds_check(pos)
            safe_pos = jnp.where(inside[:, None], pos, jax.lax.stop_gradient(pos))
            safe_dir = jnp.where(inside[:, None], dirs, jax.lax.stop_gradient(dirs))

            d0, d1, d2 = safe_dir[:, 0], safe_dir[:, 1], safe_dir[:, 2]
            o0, o1, o2 = safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2]

            # ── 1. Ray-to-string distances ──
            # Skew-line distance when the ray passes the string forward,
            # origin-to-string distance otherwise. Handles the near-parallel
            # degeneracy (dxy_sq → 0) where the skew formula gives 0/0.
            if vertical:
                dxy_sq = d0**2 + d1**2
                w0 = o0[:, None] - anch0[None, :]
                w1 = o1[:, None] - anch1[None, :]

                origin_xy = jnp.sqrt(w0**2 + w1**2)

                scalar_triple = d1[:, None] * w0 - d0[:, None] * w1
                line_dist = jnp.abs(scalar_triple) / jnp.sqrt(dxy_sq[:, None] + 1e-18)

                wd_cross = d0[:, None] * w0 + d1[:, None] * w1
                t_ray_all = -wd_cross / (dxy_sq[:, None] + 1e-9)

                use_skew = (t_ray_all > 0) & (dxy_sq[:, None] > 1e-6)
                dist_all = jnp.where(use_skew, line_dist, origin_xy)
            else:
                cx = d1[:, None] * ax2[None, :] - d2[:, None] * ax1[None, :]
                cy = d2[:, None] * ax0[None, :] - d0[:, None] * ax2[None, :]
                cz = d0[:, None] * ax1[None, :] - d1[:, None] * ax0[None, :]
                cross_sq = cx**2 + cy**2 + cz**2
                w0 = o0[:, None] - anch0[None, :]
                w1 = o1[:, None] - anch1[None, :]
                w2 = o2[:, None] - anch2[None, :]

                scalar_triple = w0 * cx + w1 * cy + w2 * cz
                line_dist = jnp.abs(scalar_triple) / jnp.sqrt(cross_sq + 1e-18)

                dd_g = jnp.sum(safe_dir**2, axis=1)
                da_g = jnp.sum(safe_dir[:, None, :] * string_axes[None, :, :], axis=-1)
                wa_g = jnp.sum(jnp.stack([w0, w1, w2], axis=-1) * string_axes[None, :, :], axis=-1)
                wd_g = jnp.sum(jnp.stack([w0, w1, w2], axis=-1) * safe_dir[:, None, :], axis=-1)
                denom_g = dd_g[:, None] - da_g**2 + 1e-9
                t_ray_all = (da_g * wa_g - wd_g) / denom_g

                origin_perp = jnp.sqrt(jnp.maximum(
                    w0**2 + w1**2 + w2**2 - wa_g**2, 0.0))
                use_skew = (t_ray_all > 0) & (cross_sq > 1e-6)
                dist_all = jnp.where(use_skew, line_dist, origin_perp)

            # ── 2. Top-K closest strings ──
            _, top_idx = jax.lax.top_k(-dist_all, n_closest)

            # ── 3. s_string for selected strings only ──
            if vertical:
                dd = dxy_sq + d2**2
                sel_wa = o2[:, None] - anch2[top_idx]
                sel_wd = ((o0[:, None] - anch0[top_idx]) * d0[:, None] +
                          (o1[:, None] - anch1[top_idx]) * d1[:, None] +
                          sel_wa * d2[:, None])
                sel_denom = dxy_sq[:, None] + 1e-9
                sel_s = (sel_wa * dd[:, None] - sel_wd * d2[:, None]) / sel_denom
            else:
                sel_anch = string_anchors[top_idx]
                sel_axes = string_axes[top_idx]
                sel_w = safe_pos[:, None, :] - sel_anch
                dd = jnp.sum(safe_dir**2, axis=1)
                sel_da = jnp.sum(safe_dir[:, None, :] * sel_axes, axis=-1)
                sel_wa = jnp.sum(sel_w * sel_axes, axis=-1)
                sel_wd = jnp.sum(sel_w * safe_dir[:, None, :], axis=-1)
                sel_denom = dd[:, None] - sel_da**2 + 1e-9
                sel_s = (sel_wa * dd[:, None] - sel_wd * sel_da) / sel_denom

            # ── 4. DOM snap: comparison-based bracket ──
            # Works for both uniform and non-uniform spacing.
            # Equivalent to searchsorted but fully vectorized (no branches).
            sel_offsets = dom_s_offsets[top_idx]  # (n, n_closest, max_dom)
            below = sel_offsets <= sel_s[:, :, None]  # (n, n_closest, max_dom)
            k_right = below.sum(axis=-1)  # (n, n_closest)

            k_start = jnp.clip(k_right - n_dom_snap // 2, 0, max_dom - n_dom_snap)
            dom_local = k_start[:, :, None] + _snap_offsets[None, None, :]  # (n, nc, nds)
            dom_local = jnp.clip(dom_local, 0, max_dom - 1)

            cand_ids = dom_global_ids[
                top_idx[:, :, None], dom_local
            ].reshape(n, n_cand)

            # ── 5. Ray-DOM closest approach (RAY distance, clamped t≥0) ──
            cand_pos = sensor_positions[cand_ids]
            oc = safe_pos[:, None, :] - cand_pos
            d_norm = safe_dir / (jnp.linalg.norm(safe_dir, axis=1, keepdims=True) + 1e-10)
            t_closest = -jnp.sum(oc * d_norm[:, None, :], axis=-1)
            t_clamped = jnp.maximum(t_closest, 0.0)
            closest_pts = safe_pos[:, None, :] + t_clamped[:, :, None] * d_norm[:, None, :]
            perp_dist = jnp.sqrt(
                jnp.sum((closest_pts - cand_pos)**2, axis=-1) + 1e-18)

            # ── 6. Overlap + volume step ──
            ov_weights = overlap_fn(perp_dist)
            valid = cand_ids >= 0
            ov_weights = jnp.where(valid, ov_weights, 0.0)

            reach = jnp.exp(-t_clamped / lambda_scat)
            per_dom_charges = ov_weights * reach
            total_detected = jnp.minimum(jnp.sum(per_dom_charges, axis=1), 1.0)

            seg_len = envelope_exit(safe_pos, safe_dir)
            u = jax.random.uniform(k_scatter, (n,))
            prob_term = -jnp.expm1(-seg_len / lambda_scat)
            scatter_dist = -lambda_scat * jnp.log1p(-u * prob_term)

            new_pos = safe_pos + scatter_dist[:, None] * d_norm
            new_dir = scatter_directions_batch(safe_dir, k_dir)
            new_times = times + scatter_dist / speed_of_light

            scatter_atten = jnp.exp(-scatter_dist / lambda_abs)
            cont = (1.0 - total_detected) * scatter_atten
            inside_new = bounds_check(new_pos)
            safe_cont = jnp.where(inside_new, cont, 0.0)

            # ── 7. Charge accumulation ──
            physical = weights * survival
            weighted_q = per_dom_charges * physical[:, None]

            flat_ids = jnp.where(valid, cand_ids, 0).ravel()
            flat_wq = jnp.where(valid, weighted_q, 0.0).ravel()
            dom_q = dom_q.at[flat_ids].add(flat_wq)

            dom_hit_t = times[:, None] + t_clamped / speed_of_light
            flat_twq = jnp.where(valid, dom_hit_t * weighted_q, 0.0).ravel()
            dom_tw = dom_tw.at[flat_ids].add(flat_twq)

            new_survival = survival * safe_cont
            return (new_pos, new_dir, new_times, new_survival, dom_q, dom_tw, key), None

        init = (
            origins, directions,
            jnp.zeros(n), jnp.ones(n),
            jnp.zeros(n_sensors), jnp.zeros(n_sensors),
            key,
        )
        (_, _, _, _, dom_charges, dom_time_weighted, _), _ = jax.lax.scan(
            step_fn, init, jnp.arange(K))

        return dom_charges, dom_time_weighted

    return simulate
