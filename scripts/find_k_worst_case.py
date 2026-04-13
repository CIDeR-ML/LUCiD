#!/usr/bin/env python3
"""Find the K needed so no photon charge is lost, across all wavelengths.

Runs one simulation per wavelength at large K_max. The per-bounce charge
is extracted from the scan output (shape K x sensors x rays) to determine
when cumulative charge reaches the threshold.

Usage:
    python scripts/find_k_worst_case.py --config config/SK_like_geom_config.json
    python scripts/find_k_worst_case.py --config config/SK_geom_config.json --detector-type superk
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Find worst-case K across wavelengths")
    parser.add_argument("--config", required=True, help="Detector geometry JSON")
    parser.add_argument("--detector-type", default="Cylinder")
    parser.add_argument("--nphot", type=int, default=50_000)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--wall-refl", type=float, default=0.2)
    parser.add_argument("--sensor-refl", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Max charge loss fraction (default: 0.01%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from lucid.geometry import generate_detector
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.simulation.config import SimConfig
    from lucid.detector_params import DetectorParams
    from lucid.sources import isotropic_source
    from lucid.wavelength.medium import make_medium
    from lucid.simulation.photon_step import photon_iteration_update_factors_safe
    from lucid.simulation.sensor_response import make_hits_simulation
    from lucid.simulation.types import PhotonState
    from functools import partial

    det_geom = DetectorGeometry.from_config(
        args.config, temperature=None, detector_type=args.detector_type,
        n_cap=150, n_angular=250, n_height=150)

    NUM_SENSORS = det_geom.num_sensors
    propagate_photons = det_geom.propagator
    SPEED_OF_LIGHT = det_geom.speed_of_light
    detector = det_geom.detector

    wl_grid = jnp.linspace(280, 700, 421)
    medium = make_medium('water', wavelength_grid=wl_grid)

    wavelengths = [300, 325, 350, 375, 400, 425, 450, 475, 500, 550, 600]
    key = jax.random.PRNGKey(args.seed)
    Nphot = args.nphot
    K = args.k_max

    @partial(jax.jit, static_argnames=('n_rays', 'K', 'num_sensors'))
    def run_and_get_per_bounce(positions, directions, intensities,
                               scatter_lengths, absorption_lengths,
                               wall_refl, sensor_refl, qe, qe_corrections,
                               key, n_rays, K, num_sensors):
        """Run propagation, return per-bounce charge array (K,)."""
        initial_survival = jnp.ones(n_rays)

        def propagation_step(carry, i):
            state = carry
            key, _ = jax.random.split(state.key)
            prop_results = propagate_photons(state.positions, state.directions)
            hit_sensor = jnp.max(prop_results['inside_sensor'], axis=0)
            surface_distances = jnp.linalg.norm(
                prop_results['positions'] - state.positions, axis=1) - 1e-6

            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            (new_pos, new_dir, new_times,
             detect_probs, refl_att, continuing) = jax.vmap(
                photon_iteration_update_factors_safe,
                in_axes=(0,0,0,0,0, 0,None,None,0, 0,0,None)
            )(state.positions, state.directions, state.times,
              surface_distances, prop_results['normals'],
              scatter_lengths, wall_refl, sensor_refl, absorption_lengths,
              hit_sensor, rng_keys, SPEED_OF_LIGHT)

            inside = detector.bounds_check(new_pos)
            safe_cont = jnp.where(inside, continuing, 0.0)
            new_survival = state.survival * safe_cont

            phys_int = intensities * state.survival
            weights = prop_results['sensor_weights'] * phys_int[None, :] * (detect_probs * refl_att)[None, :]

            # Per-bounce detected charge (sum over sensors and rays)
            bounce_charge = jnp.sum(weights * qe * qe_corrections[prop_results['sensor_indices']])

            new_state = PhotonState(
                positions=new_pos, directions=new_dir,
                times=new_times, survival=new_survival, key=key)
            return new_state, bounce_charge

        init_state = PhotonState(
            positions=positions, directions=directions,
            times=jnp.zeros(n_rays), survival=initial_survival, key=key)

        final_state, per_bounce_charge = jax.lax.scan(
            jax.remat(propagation_step), init_state, jnp.arange(K))

        return per_bounce_charge, final_state.survival

    print(f"\nK_max={K}, Nphot={Nphot}, threshold={args.threshold:.0e}")
    print(f"\n{'wl':>5s}  {'L_scat':>8s}  {'L_abs':>8s}  {'survival':>10s}  {'K_needed':>8s}")
    print("-" * 50)

    worst_k = 0
    worst_wl = 0
    all_results = {}

    source = isotropic_source(position=[0.0, 0.0, 0.0], intensity=1e8)
    dirs, origins, intensities = source(Nphot, key)

    for wl in wavelengths:
        sc = float(jnp.interp(float(wl), wl_grid, medium.scatter_coeff))
        ac = float(jnp.interp(float(wl), wl_grid, medium.absorption_coeff))
        L_s, L_a = 1.0 / sc, 1.0 / ac

        scatter_arr = jnp.full(Nphot, L_s)
        absorb_arr = jnp.full(Nphot, L_a)
        qe_corr = jnp.ones(NUM_SENSORS)

        per_bounce, final_surv = run_and_get_per_bounce(
            origins, dirs, intensities,
            scatter_arr, absorb_arr,
            jnp.array(args.wall_refl), jnp.array(args.sensor_refl),
            jnp.array(0.2), qe_corr,
            key, Nphot, K, NUM_SENSORS)

        per_bounce = jnp.array(per_bounce)
        total = float(jnp.sum(per_bounce))

        if total < 1e-10:
            print(f"{wl:5d}  {L_s:8.1f}  {L_a:8.1f}  {'---':>10s}  {'---':>8s}")
            continue

        # Survival ratio from per-bounce charge
        ratios = []
        for k in range(2, K):
            if float(per_bounce[k-1]) > 0:
                ratios.append(float(per_bounce[k]) / float(per_bounce[k-1]))
        survival = sum(ratios) / len(ratios) if ratios else 0.0

        # Find K where remaining < threshold
        cumulative = 0.0
        k_needed = K
        for k in range(K):
            cumulative += float(per_bounce[k])
            remaining = (total - cumulative) / total
            if remaining < args.threshold:
                k_needed = k + 1
                break

        print(f"{wl:5d}  {L_s:8.1f}  {L_a:8.1f}  {survival:10.4f}  {k_needed:8d}")
        all_results[wl] = (L_s, L_a, survival, k_needed, per_bounce, total)

        if k_needed > worst_k:
            worst_k = k_needed
            worst_wl = wl

    # Detailed breakdown at worst case
    if worst_wl in all_results:
        L_s, L_a, surv, _, per_bounce, total = all_results[worst_wl]
        print(f"\n{'='*60}")
        print(f"  Worst case: {worst_wl}nm (L_scat={L_s:.0f}m, L_abs={L_a:.0f}m)")
        print(f"  K needed for {args.threshold:.0e} loss: {worst_k}")
        print(f"{'='*60}")
        print(f"\n  {'K':>3s}  {'bounce_charge':>14s}  {'cumulative%':>12s}  {'remaining':>12s}")
        cumulative = 0.0
        for k in range(K):
            bc = float(per_bounce[k])
            cumulative += bc
            remaining = (total - cumulative) / total
            bar = '█' * int(bc / total * 40)
            print(f"  {k+1:3d}  {bc:14.1f}  {cumulative/total:12.6f}  {remaining:12.2e}  {bar}")
            if remaining < 1e-8:
                break


if __name__ == "__main__":
    main()
