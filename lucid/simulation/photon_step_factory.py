"""Unified photon-step factory (PR2 generic structure).

ONE source per mode, STATICALLY specialized on ``has_interface``:

  * ``has_interface=False`` ⇒ byte-identical to the legacy single-medium steps
    (``photon_iteration_sample`` / ``photon_iteration_update_factors``): the same key-split
    count (6 / 8) and the same arithmetic; the interface code and its extra key are never
    traced.
  * ``has_interface=True``  ⇒ matches the legacy nested steps (7 / 9 split + the sampled
    Snell/Fresnel/TIR interface outcome), returning the updated ``medium_id`` as an 8th value.

This replaces the duplicated ``*_nested`` functions with one factory and lets the simulator
treat single-medium as the degenerate ``I=0`` case rather than a privileged branch.
"""
import jax
import jax.numpy as jnp

from lucid.simulation.optics import (
    normalize, compute_reflection_direction, sample_cosine_hemisphere,
    sample_scatter_distance, compute_scatter_direction,
    create_local_frame, solve_rayleigh_inverse_cdf,
)
from lucid.wavelength.scattering import (
    compute_mie_scatter_direction, hg_sample_cos_theta, hg_logpdf, rayleigh_logpdf,
)
from lucid.simulation.reflection import scalar_reflection
from lucid.simulation.photon_step import _interface_refract_reflect


def make_photon_step(mode, has_interface, reflection_fn=scalar_reflection):
    """Return a per-photon step. ``mode`` ∈ {'sample','update_factors'}.

    The step's positional signature is the legacy 14-arg one; when ``has_interface`` is True it
    also takes ``(hit_interface, medium_id, n_inner, n_outer)`` and returns an 8-tuple (the
    7-tuple + ``new_medium_id``).
    """
    if mode == 'sample':
        def step(position, direction, time, surface_distance,
                 normal, scatter_length, mie_scatter_length, g, refl_params,
                 absorption_length, hit_sensor, lam, rng_key, speed_of_light,
                 hit_interface=None, medium_id=None, n_inner=None, n_outer=None):
            # 6 keys single-medium; +1 (k[6]) for the sampled interface transmit/reflect outcome.
            # JAX split-prefix (split(k,6)==split(k,7)[:6]) keeps k[0..5] — and thus the whole
            # single-medium stream — byte-identical to the legacy 6-split. `lam` is unused (MC path).
            k = jax.random.split(rng_key, 7 if has_interface else 6)
            reflection_rate = jnp.where(hit_sensor, refl_params.sensor_rate, refl_params.wall_rate)
            mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
            inv_total = 1.0 / scatter_length + 1.0 / mie_safe
            scatter_length = 1.0 / inv_total
            p_mie = (1.0 / mie_safe) / inv_total
            scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k[1])
            reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
            u1 = jax.random.uniform(k[0])
            reaches_surface = u1 < reach_surface_prob
            u2 = jax.random.uniform(k[1])
            scatters = ~reaches_surface
            epsilon = 1e-4
            inward_normal = -normal
            specular_dir = compute_reflection_direction(direction, normal)
            diffuse_dir = sample_cosine_hemisphere(inward_normal, k[3])
            wall_refl_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
            rayleigh_dir = compute_scatter_direction(direction, k[2])
            mie_dir = compute_mie_scatter_direction(direction, k[2], g)
            is_mie = jax.random.uniform(k[4]) < p_mie
            chosen_scatter_dir = jnp.where(is_mie, mie_dir, rayleigh_dir)

            if has_interface:
                dir_n = normalize(direction)
                iface_dir, iface_medium, _s, _t = _interface_refract_reflect(
                    dir_n, normal, medium_id, n_inner, n_outer, jax.random.uniform(k[6]))
                detects = reaches_surface & (~hit_interface) & (u2 >= reflection_rate)
                at_interface = reaches_surface & hit_interface
                surf_dir = jnp.where(hit_interface, iface_dir, wall_refl_dir)
                surf_pos = jnp.where(
                    hit_interface,
                    position + surface_distance * dir_n + epsilon * iface_dir,
                    position + surface_distance * dir_n + epsilon * normalize(inward_normal))
                new_dir = jnp.where(scatters, chosen_scatter_dir, surf_dir)
                new_pos = jnp.where(scatters, position + scatter_distance * dir_n, surf_pos)
                new_medium_id = jnp.where(at_interface, iface_medium, medium_id)
            else:
                reflects = reaches_surface & (u2 < reflection_rate)
                detects = reaches_surface & (u2 >= reflection_rate)
                new_pos = jnp.where(
                    scatters,
                    position + scatter_distance * normalize(direction),
                    position + surface_distance * normalize(direction) + epsilon * normalize(inward_normal))
                new_dir = jnp.where(
                    reflects, wall_refl_dir,
                    jnp.where(scatters, chosen_scatter_dir, direction))

            distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
            new_time = time + distance_traveled / speed_of_light
            # Binary absorption on a DEDICATED key k[5] — must stay distinct from the scatter-dir
            # key k[2], else absorption correlates with scatter angle (PORT_PLAN §4.3).
            survival_prob = jnp.exp(-distance_traveled / absorption_length)
            attenuation = (jax.random.uniform(k[5]) < survival_prob).astype(jnp.float32)
            detect_prob = detects.astype(jnp.float32)
            reflection_attenuation = attenuation
            continuing_factor = jnp.where(detects, 0.0, attenuation)
            logp_increment = jnp.zeros_like(new_time)

            base = (new_pos, new_dir, new_time, detect_prob,
                    reflection_attenuation, continuing_factor, logp_increment)
            return base + (new_medium_id,) if has_interface else base
        return step

    if mode == 'update_factors':
        def step(position, direction, time, surface_distance,
                 normal, scatter_length, mie_scatter_length, g, refl_params,
                 absorption_length, hit_sensor, lam, rng_key, speed_of_light,
                 hit_interface=None, medium_id=None, n_inner=None, n_outer=None):
            # 8 keys single-medium (k[6],k[7] unused, kept for split-prefix parity); +1 (k[8])
            # for the sampled interface outcome. Gradient design (PORT_PLAN): TRACK params flow
            # PATHWISE (Dd/positions/time LIVE); OPTICAL scatter params flow via the DiCE score
            # lf/la with d,Dd,cmie,cray STOP-GRADIENT'd inside it (no double count with the
            # pathwise `reach`); the optical time gradient rides the LIVE reparam free path d_live.
            k = jax.random.split(rng_key, 9 if has_interface else 8)
            sg = jax.lax.stop_gradient
            mie_safe = jnp.maximum(mie_scatter_length, 1e-6)
            mu_tot = 1.0 / scatter_length + 1.0 / mie_safe
            p_mie = scatter_length / (scatter_length + mie_safe)
            u0 = jax.random.uniform(k[0])
            d_live = -jnp.log1p(-sg(u0)) / mu_tot
            d = sg(d_live)
            Dd = surface_distance
            is_scat = d < Dd
            dist = jnp.where(is_scat, d, Dd)
            atten = jnp.exp(-dist / absorption_length)
            is_mie = jax.random.uniform(k[2]) < sg(p_mie)
            ua = jax.random.uniform(k[3])
            phi = jax.random.uniform(k[4]) * 2.0 * jnp.pi
            cmie = hg_sample_cos_theta(ua, sg(g))
            cray = jnp.clip(solve_rayleigh_inverse_cdf(ua), -1.0, 1.0)
            cth = jnp.where(is_mie, cmie, cray)
            sth = jnp.sqrt(jnp.clip(1.0 - cth**2, 0.0, 1.0))
            frame = create_local_frame(direction)
            local = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
            scat_dir = normalize(frame @ local)
            epsilon = 1e-4
            inward_normal = -sg(normal)
            refl_prob, reflection_dir, lr = reflection_fn(
                direction, normal, hit_sensor, refl_params, lam, k[5])
            dir_n = normalize(direction)
            scatter_pos = position + d * dir_n
            surface_pos = position + Dd * dir_n + epsilon * normalize(inward_normal)
            lf = jnp.where(is_scat, jnp.log(mu_tot) - mu_tot * sg(d), -mu_tot * sg(Dd))
            la = jnp.where(is_scat,
                           jnp.where(is_mie,
                                     jnp.log(p_mie) + hg_logpdf(sg(cmie), g),
                                     jnp.log1p(-p_mie) + rayleigh_logpdf(sg(cray))),
                           0.0)
            reach = jnp.exp(-mu_tot * Dd)
            atten_surf = jnp.exp(-Dd / absorption_length)
            distance_for_time = jnp.where(is_scat, d_live, Dd)
            new_time = time + distance_for_time / speed_of_light
            reflection_attenuation = jnp.ones_like(reach)

            if has_interface:
                radial = sg(normal)
                iface_dir, iface_medium, iface_score, _t = _interface_refract_reflect(
                    dir_n, radial, medium_id, n_inner, n_outer, jax.random.uniform(k[8]))
                iface_pos = position + Dd * dir_n + epsilon * iface_dir
                at_interface = hit_interface & (~is_scat)
                surf_dir = jnp.where(hit_interface, iface_dir, reflection_dir)
                surf_pos = jnp.where(hit_interface, iface_pos, surface_pos)
                new_dir = jnp.where(is_scat, scat_dir, surf_dir)
                new_pos = jnp.where(is_scat, scatter_pos, surf_pos)
                new_medium_id = jnp.where(at_interface, iface_medium, medium_id)
                surf_score = jnp.where(hit_interface, iface_score, lr)
                logp_increment = lf + la + jnp.where(is_scat, 0.0, surf_score)
                detect_prob = jnp.where(hit_interface, 0.0,
                                        reach * (1.0 - refl_prob) * atten_surf)
                cont_surface = jnp.where(hit_interface, atten, refl_prob * atten)
                continuing_factor = jnp.where(is_scat, atten, cont_surface)
            else:
                new_dir = jnp.where(is_scat, scat_dir, reflection_dir)
                new_pos = jnp.where(is_scat, scatter_pos, surface_pos)
                logp_increment = lf + la + lr
                detect_prob = reach * (1.0 - refl_prob) * atten_surf
                continuing_factor = jnp.where(is_scat, atten, refl_prob * atten)

            base = (new_pos, new_dir, new_time, detect_prob,
                    reflection_attenuation, continuing_factor, logp_increment)
            return base + (new_medium_id,) if has_interface else base
        return step

    raise ValueError(f"mode must be 'sample' or 'update_factors', got {mode!r}")
