#!/usr/bin/env python3
"""Find the optimal K (max scattering iterations) for a given detector configuration.

Runs the simulation at a large K_max and analyzes per-iteration charge deposition
to determine the minimum K needed to capture a target fraction of total charge
for a specified percentile of rays.

Works in both monochromatic and wavelength-dependent modes. When wavelength
dependence is active, scatter_length and absorption_length vary per photon,
potentially requiring higher K at wavelengths with longer mean free paths.

Usage:
    # Monochromatic (default physics)
    python scripts/find_optimal_k.py --config config/SK_like_geom_config.json

    # With wavelength dependence
    python scripts/find_optimal_k.py --config config/SK_like_geom_config.json --wavelength

    # Custom parameters
    python scripts/find_optimal_k.py --config config/WCTE_geom_config.json \
        --scatter-length 100 --absorption-length 200 --k-max 15 \
        --target 0.995 --percentile 99

    # Different source types
    python scripts/find_optimal_k.py --config config/SK_like_geom_config.json --source laser
    python scripts/find_optimal_k.py --config config/SK_like_geom_config.json --source isotropic
    python scripts/find_optimal_k.py --config config/SK_like_geom_config.json --source track
"""
import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def analyze_k_convergence(log_w, K, NPHOT, target_frac=0.99, target_percentile=95):
    """Analyze per-iteration charge from likelihood output.

    Parameters
    ----------
    log_w : array
        Flat log-weights from likelihood output, shape (K * chunk,).
    K : int
        Number of iterations used.
    NPHOT : int
        Number of photon rays.
    target_frac : float
        Target cumulative charge fraction (e.g. 0.99 = 99%).
    target_percentile : float
        Percentile of rays to check (e.g. 95 = p95 ray must meet target).

    Returns
    -------
    dict with analysis results.
    """
    chunk = len(log_w) // K
    max_sc = chunk // NPHOT
    weights_per_k = np.exp(np.array(log_w).reshape(K, max_sc, NPHOT)).sum(axis=1)  # (K, NPHOT)

    # Aggregate charge per iteration
    charge_per_iter = weights_per_k.sum(axis=1)
    total_charge = charge_per_iter.sum()

    # Photon counts per iteration: number of photons depositing charge at each K
    # (threshold chosen small enough to catch any meaningful deposit)
    deposit_thresh = 1e-10
    photons_per_iter = (weights_per_k > deposit_thresh).sum(axis=1)

    # Per-ray cumulative fraction
    cumulative = np.cumsum(weights_per_k, axis=0)
    total_per_ray = cumulative[-1]
    active = total_per_ray > 1e-20
    n_active = active.sum()

    # Cumulative photon counts: photons that have deposited at least deposit_thresh by K
    cumulative_photons = (cumulative > deposit_thresh).sum(axis=1)

    frac = cumulative[:, active] / (total_per_ray[None, active] + 1e-30)

    # Find optimal K: smallest k where p_target of rays have deposited >= target_frac
    optimal_k = K
    for k in range(K):
        p_val = np.percentile(frac[k], 100 - target_percentile)
        if p_val >= target_frac:
            optimal_k = k + 1
            break

    # Per-bounce survival ratios (aggregate, skip first bounce)
    ratios = []
    for k in range(2, K):
        if charge_per_iter[k - 1] > 0:
            ratios.append(charge_per_iter[k] / charge_per_iter[k - 1])
    mean_survival = np.mean(ratios) if ratios else 0.0

    # Estimate missed charge at optimal_k using geometric series
    if optimal_k < K and charge_per_iter[optimal_k - 1] > 0 and total_charge > 0:
        last_frac = charge_per_iter[optimal_k - 1] / total_charge
        r = mean_survival
        missed_estimate = last_frac * r / (1 - r) if r < 1 else float('inf')
    else:
        missed_estimate = 0.0

    # Percentile table for cumulative fraction
    percentile_table = {}
    for k in range(K):
        pcts = np.percentile(frac[k], [5, 25, 50, 75, 90, 95, 99])
        percentile_table[k + 1] = {
            'p5': pcts[0], 'p25': pcts[1], 'p50': pcts[2],
            'p75': pcts[3], 'p90': pcts[4], 'p95': pcts[5], 'p99': pcts[6],
            'mean': frac[k].mean(),
        }

    return {
        'optimal_k': optimal_k,
        'target_frac': target_frac,
        'target_percentile': target_percentile,
        'total_charge': total_charge,
        'n_active': int(n_active),
        'n_photons': NPHOT,
        'charge_per_iter': charge_per_iter.tolist(),
        'photons_per_iter': photons_per_iter.tolist(),
        'cumulative_photons': cumulative_photons.tolist(),
        'mean_survival_ratio': mean_survival,
        'missed_estimate_at_optimal': missed_estimate,
        'percentile_table': percentile_table,
    }


def print_report(result, params_info=""):
    """Print formatted report."""
    print(f"\n{'='*70}")
    print(f"  K Convergence Analysis{f'  ({params_info})' if params_info else ''}")
    print(f"{'='*70}")
    print(f"  Photons: {result['n_photons']:,}  |  Active: {result['n_active']:,}  |  "
          f"Total charge: {result['total_charge']:.1f}")
    print(f"  Mean per-bounce survival: {result['mean_survival_ratio']:.3f}")

    n_active = result['n_active']
    print(f"\n  Per-iteration charge and photon counts:")
    print(f"    {'K':>3s}  {'charge':>12s}  {'% total':>8s}  "
          f"{'N_photons':>10s}  {'cum N':>10s}  {'cum %':>7s}")
    for k, (charge, n_ph, cum_ph) in enumerate(zip(
            result['charge_per_iter'],
            result['photons_per_iter'],
            result['cumulative_photons'])):
        frac = charge / result['total_charge'] if result['total_charge'] > 0 else 0
        cum_frac = cum_ph / n_active if n_active > 0 else 0
        bar = '█' * int(frac * 30)
        print(f"    K={k+1:2d}  {charge:12.1f}  {frac:7.2%}  "
              f"{n_ph:10,d}  {cum_ph:10,d}  {cum_frac:6.2%}  {bar}")

    print(f"\n  Cumulative charge fraction (percentiles over rays):")
    print(f"  {'K':>4s}  {'p5':>7s}  {'p25':>7s}  {'p50':>7s}  {'p75':>7s}  "
          f"{'p90':>7s}  {'p95':>7s}  {'p99':>7s}")
    for k, pcts in result['percentile_table'].items():
        print(f"  {k:4d}  {pcts['p5']:7.4f}  {pcts['p25']:7.4f}  {pcts['p50']:7.4f}  "
              f"{pcts['p75']:7.4f}  {pcts['p90']:7.4f}  {pcts['p95']:7.4f}  {pcts['p99']:7.4f}")

    p = result['target_percentile']
    t = result['target_frac']
    print(f"\n  Target: p{p} of rays depositing >= {t:.1%} of charge")
    print(f"  >>> Optimal K = {result['optimal_k']} <<<")
    if result['missed_estimate_at_optimal'] > 0:
        print(f"  Estimated missed charge beyond K={result['optimal_k']}: "
              f"{result['missed_estimate_at_optimal']:.4%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Find optimal K for a detector configuration")
    parser.add_argument("--config", required=True, help="Detector geometry JSON config")
    parser.add_argument("--source", default="track", choices=["laser", "isotropic", "track", "data"],
                        help="Source type (default: track)")
    parser.add_argument("--nphot", type=int, default=50_000, help="Number of photons (default: 50000)")
    parser.add_argument("--k-max", type=int, default=12, help="Maximum K to test (default: 12)")
    parser.add_argument("--root-file", default=None,
                        help="PhotonSim ROOT file for --source data")
    parser.add_argument("--entry", type=int, default=0,
                        help="Entry index in ROOT file (default: 0)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Propagation temperature (default: None = step function)")
    parser.add_argument("--scatter-length", type=float, default=50.0, help="Scatter length in m")
    parser.add_argument("--absorption-length", type=float, default=50.0, help="Absorption length in m")
    parser.add_argument("--wall-reflection", type=float, default=0.2, help="Wall reflection rate")
    parser.add_argument("--sensor-reflection", type=float, default=0.2, help="Sensor reflection rate")
    parser.add_argument("--qe", type=float, default=0.065, help="Quantum efficiency")
    parser.add_argument("--target", type=float, default=0.999,
                        help="Target cumulative charge fraction (default: 0.999)")
    parser.add_argument("--percentile", type=float, default=95,
                        help="Percentile of rays to check (default: 95)")
    parser.add_argument("--wavelength", action="store_true",
                        help="Enable wavelength-dependent mode")
    parser.add_argument("--energy", type=float, default=500.0,
                        help="Particle energy in MeV (for track mode, default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams, ParticleParams

    det = generate_detector(args.config)
    NUM_SENSORS = len(det.all_points)

    dp = DetectorParams(
        scatter_length=args.scatter_length,
        wall_reflection_rate=args.wall_reflection,
        sensor_reflection_rate=args.sensor_reflection,
        absorption_length=args.absorption_length,
        qe=args.qe,
        qe_corrections=jnp.ones(NUM_SENSORS),
    )

    grid_kw = dict(n_cap=150, n_angular=250, n_height=150)
    key = jax.random.PRNGKey(args.seed)

    params_info = (f"scatter={args.scatter_length}m, absorption={args.absorption_length}m, "
                   f"wall_refl={args.wall_reflection}, sensor_refl={args.sensor_reflection}")

    if args.wavelength:
        from lucid.wavelength import make_medium, compute_effective_properties
        medium = make_medium("water", wavelength_grid=jnp.linspace(300, 600, 301))

        # Show effective properties at key wavelengths
        print("\nWavelength-dependent effective properties:")
        print(f"  {'wl (nm)':>8s}  {'eff_scatter':>12s}  {'eff_absorption':>14s}  {'ratio L_s/L_a':>14s}")
        for wl in [300, 350, 400, 450, 500, 550, 600]:
            eff_s, eff_a, eff_q = compute_effective_properties(
                dp, medium, wavelengths=jnp.array([float(wl)]))
            s_val = float(eff_s[0]) if hasattr(eff_s, '__len__') else float(eff_s)
            a_val = float(eff_a[0]) if hasattr(eff_a, '__len__') else float(eff_a)
            print(f"  {wl:8d}  {s_val:12.1f}m  {a_val:14.1f}m  {s_val/a_val:14.2f}")
        params_info += ", wavelength=ON"

    t_start = time.perf_counter()

    # All sources use hit_mode='per_photon' to expose per-iteration log_w for analysis.
    if args.source == "track":
        sim = setup_event_simulator(
            args.config, args.nphot, temperature=args.temperature, K=args.k_max,
            is_data=False, is_calibration=False,
            default_detector_params=dp, hit_mode='per_photon', **grid_kw)

        track = ParticleParams(
            energy=jnp.array(args.energy),
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=jnp.array(0.5),
            phi=jnp.array(1.0),
            t0=jnp.array(0.0),
        )
        output = sim(track, key)
        nphot_actual = args.nphot

    elif args.source in ("laser", "isotropic"):
        from lucid.sources import laser_source, isotropic_source
        if args.source == "laser":
            source = laser_source(position=[0.0, 0.0, det.H / 2 - 0.1], intensity=100_000_000)
        else:
            source = isotropic_source(position=[0.0, 0.0, 0.0], intensity=100_000_000)

        sim = setup_event_simulator(
            args.config, args.nphot, temperature=args.temperature, K=args.k_max,
            is_data=False, is_calibration=True,
            default_detector_params=dp, hit_mode='per_photon', **grid_kw)
        output = sim(source, key)
        nphot_actual = args.nphot

    elif args.source == "data":
        if args.root_file is None:
            raise ValueError("--root-file is required for --source data")
        from lucid.sources.event_io import read_photon_data_from_photonsim

        photon_data = read_photon_data_from_photonsim(args.root_file, args.entry)
        nphot_actual = int(photon_data['photon_origins'].shape[0])
        photon_data['N'] = nphot_actual
        photon_data['rotation_axis'] = jnp.array([0.0, 0.0, 1.0])
        photon_data['rotation_angle'] = jnp.array(0.0)
        photon_data['apply_rotation'] = False
        photon_data['apply_translation'] = False
        photon_data['translation_vector'] = jnp.array([0.0, 0.0, 0.0])

        sim = setup_event_simulator(
            args.config, nphot_actual, temperature=args.temperature, K=args.k_max,
            is_data=True, apply_smearing=False,
            default_detector_params=dp, hit_mode='per_photon', **grid_kw)

        part = ParticleParams(
            energy=jnp.array(photon_data.get('energy', 1000.0)),
            position=jnp.zeros(3),
            theta=jnp.array(0.0), phi=jnp.array(0.0), t0=jnp.array(0.0),
        )
        output = sim(part, key, photon_data)
        params_info += f", ROOT={os.path.basename(args.root_file)} entry={args.entry}"

    log_w = output[0]
    elapsed = time.perf_counter() - t_start

    result = analyze_k_convergence(
        log_w, args.k_max, nphot_actual,
        target_frac=args.target,
        target_percentile=args.percentile,
    )
    print(f"\n  Source: {args.source}  |  Nphot: {nphot_actual:,}  |  "
          f"K_max: {args.k_max}  |  Time: {elapsed:.1f}s")
    print_report(result, params_info)


if __name__ == "__main__":
    main()
